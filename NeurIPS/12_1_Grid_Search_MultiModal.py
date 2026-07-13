import os
# Limit CPU parallelism to prevent thermal overload
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import datetime
import gc
import itertools

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
print(f"Using device: {device}")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = r"E:\OfflineEMG\extracted_trials_shifted"
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results", timestamp + "_MultiModal_Grid_Search")

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

def load_and_window_data(base_dir, emg_nperseg, emg_noverlap, emg_max_freq, imu_nperseg, imu_noverlap, imu_max_freq):
    print("Extracting EMG and IMU windows ONCE...")
    X_train_emg, X_train_imu, y_train_raw, y_train_move, t_train = [], [], [], [], []
    X_test_emg, X_test_imu, y_test_raw, y_test_move, t_test = [], [], [], [], []

    class_csv_map = {}
    min_trials = 999999
    for class_name in classes:
        files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        class_csv_map[class_name] = files
        if len(files) < min_trials:
            min_trials = len(files)

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
                X_train_emg.extend(trial_emg_stfts)
                X_train_imu.extend(trial_imu_stfts)
                y_train_raw.extend(trial_labels)
                y_train_move.extend(trial_move_labels)
                t_train.extend(trial_times)
            else:
                X_test_emg.extend(trial_emg_stfts)
                X_test_imu.extend(trial_imu_stfts)
                y_test_raw.extend(trial_labels)
                y_test_move.extend(trial_move_labels)
                t_test.extend(trial_times)

    return (
        np.array(X_train_emg), np.array(X_train_imu), np.array(y_train_raw), np.array(y_train_move), np.array(t_train),
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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                
    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        emg_features = emg_features.view(emg_features.size(0), -1)
        
        imu_features = self.imu_encoder(imu_x)
        imu_features = imu_features.view(imu_features.size(0), -1)
        
        combined = torch.cat((emg_features, imu_features), dim=1)
        return self.fc(combined)

# ========================================================================
# GRID SEARCH EXECUTION
# ========================================================================
def run_model_experiment(model_config, X_train_emg, X_train_imu, y_train_raw, 
                         X_test_emg, X_test_imu, y_test_raw, y_test_move, t_test, 
                         results_dir):
    print(f"\n{'='*60}\nEvaluating Config: {model_config}\n{'='*60}")
    os.makedirs(results_dir, exist_ok=True)
    
    X_train_emg_t = torch.from_numpy(X_train_emg).permute(0, 3, 2, 1).float()
    X_test_emg_t = torch.from_numpy(X_test_emg).permute(0, 3, 2, 1).float()
    X_train_imu_t = torch.from_numpy(X_train_imu).permute(0, 3, 2, 1).float()
    X_test_imu_t = torch.from_numpy(X_test_imu).permute(0, 3, 2, 1).float()
    
    y_train_t = torch.from_numpy(y_train_raw).long()
    y_test_t = torch.from_numpy(y_test_raw).long()
    
    train_dataset = TensorDataset(X_train_emg_t, X_train_imu_t, y_train_t)
    test_dataset = TensorDataset(X_test_emg_t, X_test_imu_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # EMD-Only Validation Loader for Checkpointing
    emd_test_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    test_emd_dataset = TensorDataset(
        X_test_emg_t[emd_test_mask], 
        X_test_imu_t[emd_test_mask], 
        y_test_t[emd_test_mask]
    )
    test_emd_loader = DataLoader(test_emd_dataset, batch_size=32, shuffle=False)

    model = MultiModalNet(
        emg_channels=N_EMG_CHANNELS, 
        emg_frames=X_train_emg_t.shape[2], 
        emg_freqs=X_train_emg_t.shape[3],
        imu_channels=N_IMU_CHANNELS, 
        imu_frames=X_train_imu_t.shape[2], 
        imu_freqs=X_train_imu_t.shape[3], 
        num_classes=num_classes,
        base_filters=model_config['base_filters'],
        dropout_rate=model_config['dropout'],
        emg_kernel_size=model_config['kernel_size'],
        imu_kernel_size=2  # Fixed small kernel for IMU due to low freq resolution
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    
    epochs = 10
    history = {'accuracy': [], 'val_accuracy': [], 'val_emd_accuracy': []}
    
    best_emd_acc = 0.0
    best_model_path = os.path.join(results_dir, "best_emd_model.pt")
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs_emg, inputs_imu, targets in train_loader:
            inputs_emg, inputs_imu, targets = inputs_emg.to(device), inputs_imu.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_emg, inputs_imu)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_acc = correct / total if total > 0 else 0
        
        # Evaluate EMD-Only for Checkpointing
        model.eval()
        emd_val_correct, emd_val_total = 0, 0
        with torch.no_grad():
            for inputs_emg, inputs_imu, targets in test_emd_loader:
                inputs_emg, inputs_imu, targets = inputs_emg.to(device), inputs_imu.to(device), targets.to(device)
                outputs = model(inputs_emg, inputs_imu)
                _, predicted = outputs.max(1)
                emd_val_total += targets.size(0)
                emd_val_correct += predicted.eq(targets).sum().item()
                
        epoch_emd_val_acc = emd_val_correct / emd_val_total if emd_val_total > 0 else 0
        
        history['accuracy'].append(epoch_acc)
        history['val_emd_accuracy'].append(epoch_emd_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Acc: {epoch_acc*100:.2f}% - EMD Val Acc: {epoch_emd_val_acc*100:.2f}%")
        
        if epoch_emd_val_acc >= best_emd_acc:
            best_emd_acc = epoch_emd_val_acc
            torch.save(model.state_dict(), best_model_path)
    
    print(f"Loading best model for final evaluation (Best EMD Acc: {best_emd_acc*100:.2f}%)")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final EVALUATION
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
    true_labels = y_test_move
    
    noise_mask = (t_test >= NOISE_START) & (t_test <= NOISE_END)
    emd_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    move_mask = (t_test > EMD_END) & (t_test <= MOVE_END)
    
    avg_overall = np.mean(true_labels == pred_labels) * 100.0
    avg_noise = np.mean(true_labels[noise_mask] == pred_labels[noise_mask]) * 100.0 if np.any(noise_mask) else 0.0
    avg_emd = np.mean(true_labels[emd_mask] == pred_labels[emd_mask]) * 100.0 if np.any(emd_mask) else 0.0
    avg_move = np.mean(true_labels[move_mask] == pred_labels[move_mask]) * 100.0 if np.any(move_mask) else 0.0

    summary_text = (
        f"{'='*60}\n"
        f"MULTIMODAL HYPERPARAMETER CLASSIFICATION SUMMARY\n"
        f"Model: MultiModalNet\n"
        f"Grid Config: {model_config}\n"
        f"{'='*60}\n"
        f"Overall Average Accuracy:          {avg_overall:.2f}%\n"
        f"  Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
        f"  EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
        f"  Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
        f"{'='*60}\n"
    )
    print(summary_text)
    with open(os.path.join(results_dir, "accuracy_summary.txt"), 'w') as f:
        f.write(summary_text)


if __name__ == "__main__":
    # Fixed Spectrogram Parameters
    EMG_NPERSEG = 256
    EMG_NOVERLAP = 254
    EMG_MAX_FREQ = 962.0
    IMU_NPERSEG = 8
    IMU_NOVERLAP = 7
    IMU_MAX_FREQ = None
    
    (X_train_emg_raw, X_train_imu_raw, y_train_raw, y_train_move, t_train, 
     X_test_emg_raw, X_test_imu_raw, y_test_raw, y_test_move, t_test) = load_and_window_data(
         DATA_DIR, EMG_NPERSEG, EMG_NOVERLAP, EMG_MAX_FREQ, IMU_NPERSEG, IMU_NOVERLAP, IMU_MAX_FREQ)
     
    print(f"Data Loaded: {len(X_train_emg_raw)} train, {len(X_test_emg_raw)} test.")
    
    if len(X_train_emg_raw) == 0:
        print("No data found! Exiting.")
        exit(1)
    
    # Normalize EMG Once
    mean_emg = np.mean(X_train_emg_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_emg = np.std(X_train_emg_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_train_emg_norm = (X_train_emg_raw - mean_emg) / (std_emg + 1e-8)
    X_test_emg_norm = (X_test_emg_raw - mean_emg) / (std_emg + 1e-8)

    # Normalize IMU Once
    mean_imu = np.mean(X_train_imu_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_imu = np.std(X_train_imu_raw, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_train_imu_norm = (X_train_imu_raw - mean_imu) / (std_imu + 1e-8)
    X_test_imu_norm = (X_test_imu_raw - mean_imu) / (std_imu + 1e-8)

    # Apply EMD-focused training mask (t <= 0.0s) for training set only!
    emd_train_mask = (t_train <= EMD_END)
    X_train_emg_norm = X_train_emg_norm[emd_train_mask]
    X_train_imu_norm = X_train_imu_norm[emd_train_mask]
    y_train_raw_filtered = y_train_raw[emd_train_mask]
    
    # Model Hyperparameter Grid (2 x 3 x 2 x 2 = 24 configs)
    base_filters_params = [16, 32]
    dropout_params = [0.0, 0.3, 0.5]
    learning_rate_params = [0.001, 0.0005]
    kernel_size_params = [3, 5]
    
    grid_configs = []
    for bf, drop, lr, ks in itertools.product(base_filters_params, dropout_params, learning_rate_params, kernel_size_params):
        grid_configs.append({
            "base_filters": bf,
            "dropout": drop,
            "learning_rate": lr,
            "kernel_size": ks
        })
        
    print(f"Starting MultiModal Grid Search with {len(grid_configs)} configurations.")
    
    for config in grid_configs:
        config_name = f"filters{config['base_filters']}_drop{config['dropout']}_lr{config['learning_rate']}_k{config['kernel_size']}"
        config_dir = os.path.join(BASE_RESULTS_DIR, config_name)
        
        run_model_experiment(
            model_config=config, 
            X_train_emg=X_train_emg_norm, 
            X_train_imu=X_train_imu_norm, 
            y_train_raw=y_train_raw_filtered, 
            X_test_emg=X_test_emg_norm, 
            X_test_imu=X_test_imu_norm, 
            y_test_raw=y_test_raw, 
            y_test_move=y_test_move, 
            t_test=t_test, 
            results_dir=config_dir
        )
        
    print(f"\nAll model configurations completed. Results saved to: {BASE_RESULTS_DIR}")
