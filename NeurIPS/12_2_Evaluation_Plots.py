import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime
import gc
import re

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
print(f"Using device: {device}")

DATA_DIR = r"E:\OfflineEMG\extracted_trials_shifted"

classes = [
    "Trial_1_Rest", "Trial_2_Ball_Short", "Trial_3_VPen_Short", "Trial_4_HPen_Short",
    "Trial_5_Bottle_Short", "Trial_6_Mug_Short", "Trial_7_Card_Short"
]
class_labels_short = ["Rest", "Ball", "VPen", "HPen", "Bottle", "Mug", "Card"]
num_classes = len(classes)

EMG_FS = 1925.0
IMU_FS = 74.0

NOISE_START  = -1.5
NOISE_END    = -0.6
EMD_START    = -0.6
EMD_END      =  0.0
MOVE_END     =  1.5
WINDOW_S = 0.2
STEP_S   = 0.05
N_EMG_CHANNELS = 8
N_IMU_CHANNELS = 24

# ========================================================================
# SPECTROGRAM PREPROCESSING
# ========================================================================
def preprocess_stft(X_window, fs, nperseg, noverlap, max_freq):
    channels = X_window.shape[1]
    num_freqs = nperseg // 2
    f_tmp, t_tmp, Zxx_tmp = stft(X_window[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    num_frames = len(t_tmp)
    mag_out = np.zeros((num_freqs, num_frames, channels), dtype=np.float32)
    
    for c in range(channels):
        f, t, Zxx = stft(X_window[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        mag = mag[:num_freqs, :]
        f = f[:num_freqs]
        if max_freq is not None:
            mag[f > max_freq, :] = 0.0
        mag_out[:, :, c] = mag
        
    return mag_out, num_freqs, num_frames

def load_test_data(base_dir, emg_nperseg, emg_noverlap, emg_max_freq, imu_nperseg, imu_noverlap, imu_max_freq):
    print("Extracting test windows ONCE...")
    X_test_emg, X_test_imu, y_test_raw, y_test_move, t_test = [], [], [], [], []

    class_csv_map = {}
    min_trials = 999999
    for class_name in classes:
        files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        class_csv_map[class_name] = files
        if len(files) < min_trials:
            min_trials = len(files)

    # We also need the train raw sets to compute the normalisation stats!
    X_train_emg_raw, X_train_imu_raw = [], []

    for class_idx, class_name in enumerate(classes):
        csv_files = class_csv_map[class_name][:min_trials]
        n_train = int(min_trials * 0.8)

        for file_idx, csv_path in enumerate(csv_files):
            is_train = (file_idx < n_train)

            with open(csv_path, 'r') as f:
                for _ in range(5): f.readline()
                num_cols = len(f.readline().split(','))

            df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
            if len(df) <= 2: continue
            df = df.iloc[2:].reset_index(drop=True)

            emg_time_cols = [c for c in df.columns if 'Time' in c and 'ACC' not in c]
            if not emg_time_cols: continue
            emg_time_vals = pd.to_numeric(df[emg_time_cols[0]], errors='coerce').values
            emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
            emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values

            imu_time_col = [c for c in df.columns if 'ACC' in c and 'Time' in c][0]
            valid_imu_mask = pd.notnull(pd.to_numeric(df[imu_time_col], errors='coerce'))
            imu_time_vals = pd.to_numeric(df[imu_time_col][valid_imu_mask], errors='coerce').values
            imu_cols = [c for c in df.columns if 'ACC' in c and '(G)' in c]
            imu_data = df[imu_cols][valid_imu_mask].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

            req_len_emg = int(WINDOW_S * EMG_FS)
            target_len_emg = max(req_len_emg, emg_nperseg)
            
            req_len_imu = int(WINDOW_S * IMU_FS)
            target_len_imu = max(req_len_imu, imu_nperseg)

            trial_emg_stfts, trial_imu_stfts, trial_labels, trial_move_labels, trial_times = [], [], [], [], []

            if class_name == "Trial_1_Rest":
                t_start_trial = emg_time_vals[0]
                t_end_trial = emg_time_vals[-1]
                t_end = t_start_trial + WINDOW_S
                
                while t_end <= t_end_trial:
                    t_start = t_end - WINDOW_S
                    
                    mask_emg = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    win_emg = emg_data[mask_emg]
                    if len(win_emg) == 0:
                        win_emg = np.zeros((target_len_emg, N_EMG_CHANNELS))
                    elif len(win_emg) < target_len_emg:
                        pad = target_len_emg - len(win_emg)
                        win_emg = np.pad(win_emg, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_emg) > target_len_emg:
                        win_emg = win_emg[:target_len_emg]

                    mask_imu = (imu_time_vals >= t_start) & (imu_time_vals <= t_end)
                    win_imu = imu_data[mask_imu]
                    if len(win_imu) == 0:
                        win_imu = np.zeros((target_len_imu, N_IMU_CHANNELS))
                    elif len(win_imu) < target_len_imu:
                        pad = target_len_imu - len(win_imu)
                        win_imu = np.pad(win_imu, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_imu) > target_len_imu:
                        win_imu = win_imu[:target_len_imu]

                    emg_stft, _, _ = preprocess_stft(win_emg, EMG_FS, emg_nperseg, emg_noverlap, emg_max_freq)
                    imu_stft, _, _ = preprocess_stft(win_imu, IMU_FS, imu_nperseg, imu_noverlap, imu_max_freq)
                    
                    artificial_t = NOISE_START + (t_end - t_start_trial)
                    if artificial_t > MOVE_END: artificial_t = MOVE_END
                    
                    trial_emg_stfts.append(emg_stft)
                    trial_imu_stfts.append(imu_stft)
                    trial_labels.append(class_idx)
                    trial_move_labels.append(class_idx)
                    trial_times.append(artificial_t)
                    t_end += STEP_S
            else:
                t_end = NOISE_START + WINDOW_S
                while t_end <= MOVE_END:
                    t_start = t_end - WINDOW_S
                    
                    mask_emg = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    win_emg = emg_data[mask_emg]
                    if len(win_emg) == 0:
                        win_emg = np.zeros((target_len_emg, N_EMG_CHANNELS))
                    elif len(win_emg) < target_len_emg:
                        pad = target_len_emg - len(win_emg)
                        win_emg = np.pad(win_emg, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_emg) > target_len_emg:
                        win_emg = win_emg[:target_len_emg]
    
                    mask_imu = (imu_time_vals >= t_start) & (imu_time_vals <= t_end)
                    win_imu = imu_data[mask_imu]
                    if len(win_imu) == 0:
                        win_imu = np.zeros((target_len_imu, N_IMU_CHANNELS))
                    elif len(win_imu) < target_len_imu:
                        pad = target_len_imu - len(win_imu)
                        win_imu = np.pad(win_imu, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_imu) > target_len_imu:
                        win_imu = win_imu[:target_len_imu]
    
                    emg_stft, _, _ = preprocess_stft(win_emg, EMG_FS, emg_nperseg, emg_noverlap, emg_max_freq)
                    imu_stft, _, _ = preprocess_stft(win_imu, IMU_FS, imu_nperseg, imu_noverlap, imu_max_freq)
                    
                    trial_emg_stfts.append(emg_stft)
                    trial_imu_stfts.append(imu_stft)
                    trial_labels.append(class_idx)
                    trial_move_labels.append(class_idx)
                    trial_times.append(t_end)
                    t_end += STEP_S

            trial_emg_stfts = np.array(trial_emg_stfts)
            trial_imu_stfts = np.array(trial_imu_stfts)
            trial_times = np.array(trial_times)
            
            # Baseline Noise Subtraction
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                base_emg_profile = np.mean(trial_emg_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_emg_stfts = np.maximum(trial_emg_stfts - base_emg_profile, 0.0)
                base_imu_profile = np.mean(trial_imu_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_imu_stfts = np.maximum(trial_imu_stfts - base_imu_profile, 0.0)

            if is_train:
                X_train_emg_raw.extend(trial_emg_stfts)
                X_train_imu_raw.extend(trial_imu_stfts)
            else:
                X_test_emg.extend(trial_emg_stfts)
                X_test_imu.extend(trial_imu_stfts)
                y_test_raw.extend(trial_labels)
                y_test_move.extend(trial_move_labels)
                t_test.extend(trial_times)

    print(f"Data Loaded: {len(X_train_emg_raw)} train windows, {len(X_test_emg)} test windows.")
    return (
        np.array(X_train_emg_raw), np.array(X_train_imu_raw),
        np.array(X_test_emg),  np.array(X_test_imu),  np.array(y_test_raw),  np.array(y_test_move),  np.array(t_test)
    )

# ========================================================================
# DYNAMIC MULTI-MODAL MODEL ARCHITECTURE
# ========================================================================
class MultiModalNet(nn.Module):
    def __init__(self, emg_channels=8, emg_frames=26, emg_freqs=128, 
                 imu_channels=24, imu_frames=8, imu_freqs=4, num_classes=7,
                 base_filters=32, dropout_rate=0.0, emg_kernel_size=3, imu_kernel_size=2):
        super(MultiModalNet, self).__init__()
        
        f1 = base_filters
        f2 = base_filters * 4
        f3 = base_filters * 8
        
        self.emg_encoder = nn.Sequential(
            nn.Conv2d(emg_channels, f1, kernel_size=(emg_kernel_size, emg_kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(f1, f2, kernel_size=(emg_kernel_size, emg_kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(f2, f3, kernel_size=(emg_kernel_size, emg_kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Conv2d(imu_channels, f1, kernel_size=(imu_kernel_size, imu_kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(f1, f2, kernel_size=(imu_kernel_size, imu_kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        def get_flattened_size(encoder, c, h, w):
            with torch.no_grad():
                x = torch.zeros(1, c, h, w)
                out = encoder(x)
                return out.view(1, -1).size(1)
        
        self.emg_flat_size = get_flattened_size(self.emg_encoder, emg_channels, emg_frames, emg_freqs)
        self.imu_flat_size = get_flattened_size(self.imu_encoder, imu_channels, imu_frames, imu_freqs)
        
        self.fc = nn.Sequential(
            nn.Linear(self.emg_flat_size + self.imu_flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
                
    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        emg_features = emg_features.view(emg_features.size(0), -1)
        
        imu_features = self.imu_encoder(imu_x)
        imu_features = imu_features.view(imu_features.size(0), -1)
        
        combined = torch.cat((emg_features, imu_features), dim=1)
        return self.fc(combined)

def get_latest_grid_search_dir():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results")
    dirs = [d for d in glob.glob(os.path.join(base_dir, "*_MultiModal_Grid_Search")) if os.path.isdir(d)]
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

def parse_config_from_foldername(folder_name):
    # e.g., filters16_drop0.0_lr0.001_k3
    match = re.search(r"filters(\d+)_drop([\d.]+)_lr([\d.]+)_k(\d+)", folder_name)
    if match:
        return {
            'base_filters': int(match.group(1)),
            'dropout': float(match.group(2)),
            'learning_rate': float(match.group(3)),
            'kernel_size': int(match.group(4))
        }
    return None

def generate_plots_for_model(model_path, config, results_dir, test_loader, t_test, true_labels, num_emg_frames, num_emg_freqs, num_imu_frames, num_imu_freqs):
    print(f"Generating plots for: {os.path.basename(results_dir)}")
    
    model = MultiModalNet(
        emg_channels=N_EMG_CHANNELS, 
        emg_frames=num_emg_frames, 
        emg_freqs=num_emg_freqs,
        imu_channels=N_IMU_CHANNELS, 
        imu_frames=num_imu_frames, 
        imu_freqs=num_imu_freqs, 
        num_classes=num_classes,
        base_filters=config['base_filters'],
        dropout_rate=config['dropout'],
        emg_kernel_size=config['kernel_size'],
        imu_kernel_size=2
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs_emg, inputs_imu, _ in test_loader:
            inputs_emg, inputs_imu = inputs_emg.to(device), inputs_imu.to(device)
            outputs = model(inputs_emg, inputs_imu)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    pred_labels = np.argmax(preds, axis=1)
    
    unique_times = sorted(list(set(np.round(t_test, 2))))
    overall_acc_over_time = []
    class_accuracies_over_time = {cls: [] for cls in classes}
    
    for t_val in unique_times:
        mask = np.isclose(np.round(t_test, 2), t_val)
        y_true_t = true_labels[mask]
        y_pred_t = pred_labels[mask]
        overall_acc_over_time.append(np.mean(y_true_t == y_pred_t) * 100.0 if len(y_true_t) > 0 else np.nan)
        for cls_idx, cls in enumerate(classes):
            cls_mask = (y_true_t == cls_idx)
            class_accuracies_over_time[cls].append(np.mean(y_true_t[cls_mask] == y_pred_t[cls_mask]) * 100.0 if np.sum(cls_mask) > 0 else np.nan)
            
    # Accuracy Over Time Plot
    plt.figure(figsize=(12, 6))
    plt.bar(unique_times, overall_acc_over_time, width=STEP_S*0.8, color='purple', edgecolor='black', alpha=0.8)
    plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
    plt.title(f'Overall Accuracy Over Time\nConfig: {os.path.basename(results_dir)}')
    plt.xlabel('Time Relative to Onset (s)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(NOISE_START, MOVE_END)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Per_Class_Accuracy_Over_Time_BarChart.png"), dpi=150)
    plt.close()
    
    class_plot_dir = os.path.join(results_dir, "Per_Class_BarCharts")
    os.makedirs(class_plot_dir, exist_ok=True)
    
    for cls in classes:
        plt.figure(figsize=(12, 6))
        plt.bar(unique_times, class_accuracies_over_time[cls], width=STEP_S*0.8, color='teal', alpha=0.8, edgecolor='black')
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2)
        plt.title(f'Accuracy Over Time for Class: {cls}\nConfig: {os.path.basename(results_dir)}')
        plt.xlabel('Time Relative to Onset (s)')
        plt.ylabel('Accuracy (%)')
        plt.xlim(NOISE_START, MOVE_END)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(class_plot_dir, f"Accuracy_Over_Time_{cls}.png"), dpi=150)
        plt.close()
        
    class_labels = classes
    emd_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    avg_emd = np.mean(true_labels[emd_mask] == pred_labels[emd_mask]) * 100.0 if np.any(emd_mask) else 0.0

    if np.any(emd_mask):
        cm_emd = confusion_matrix(true_labels[emd_mask], pred_labels[emd_mask], labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_emd, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"EMD Phase Confusion Matrix (-0.6s to 0.0s)\nAccuracy: {avg_emd:.2f}%\nConfig: {os.path.basename(results_dir)}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix_EMD_only.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    grid_search_dir = get_latest_grid_search_dir()
    if not grid_search_dir:
        print("No MultiModal Grid Search directory found!")
        exit(1)
        
    print(f"Targeting Grid Search Directory: {grid_search_dir}")
    
    # Check if there are any models to evaluate
    model_paths = glob.glob(os.path.join(grid_search_dir, "*", "best_emd_model.pt"))
    if not model_paths:
        print("No 'best_emd_model.pt' files found in any subdirectories!")
        exit(1)
    
    # Fixed Spectrogram Parameters from 12_1
    EMG_NPERSEG = 256
    EMG_NOVERLAP = 254
    EMG_MAX_FREQ = 962.0
    IMU_NPERSEG = 8
    IMU_NOVERLAP = 7
    IMU_MAX_FREQ = None
    
    (X_train_emg_raw, X_train_imu_raw,
     X_test_emg_raw, X_test_imu_raw, y_test_raw, y_test_move, t_test) = load_test_data(
         DATA_DIR, EMG_NPERSEG, EMG_NOVERLAP, EMG_MAX_FREQ, IMU_NPERSEG, IMU_NOVERLAP, IMU_MAX_FREQ)
         
    if len(X_test_emg_raw) == 0:
        print("No test data found! Exiting.")
        exit(1)
        
    # Normalize EMG 
    mean_emg = np.mean(X_train_emg_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_emg = np.std(X_train_emg_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_test_emg_norm = (X_test_emg_raw - mean_emg) / (std_emg + 1e-8)

    # Normalize IMU
    mean_imu = np.mean(X_train_imu_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_imu = np.std(X_train_imu_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_test_imu_norm = (X_test_imu_raw - mean_imu) / (std_imu + 1e-8)

    # Convert to Tensors
    X_test_emg_t = torch.from_numpy(X_test_emg_norm).permute(0, 3, 2, 1).float()
    X_test_imu_t = torch.from_numpy(X_test_imu_norm).permute(0, 3, 2, 1).float()
    y_test_t = torch.from_numpy(y_test_raw).long()
    
    test_dataset = TensorDataset(X_test_emg_t, X_test_imu_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_emg_frames = X_test_emg_t.shape[2]
    num_emg_freqs = X_test_emg_t.shape[3]
    num_imu_frames = X_test_imu_t.shape[2]
    num_imu_freqs = X_test_imu_t.shape[3]
    
    for model_path in model_paths:
        config_dir = os.path.dirname(model_path)
        folder_name = os.path.basename(config_dir)
        config = parse_config_from_foldername(folder_name)
        if config:
            generate_plots_for_model(
                model_path, config, config_dir, 
                test_loader, t_test, y_test_move, 
                num_emg_frames, num_emg_freqs, num_imu_frames, num_imu_freqs
            )
            
    print("\nAll evaluation plots successfully generated!")
