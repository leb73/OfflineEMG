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
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, stft, tf2sos
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import datetime
import gc

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
print(f"Using device: {device}")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = r"E:\OfflineEMG\extracted_trials_shifted"
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results", timestamp + "_Final_Optimised_Model")

classes = [
    "Trial_1_Rest",
    "Trial_2_Ball_Short",
    "Trial_3_VPen_Short",
    "Trial_4_HPen_Short",
    "Trial_5_Bottle_Short",
    "Trial_6_Mug_Short",
    "Trial_7_Card_Short"
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

N_SENSORS = 8
N_SIGNALS = 1
N_CHANNELS = N_SENSORS * N_SIGNALS

# ========================================================================
# SPECTROGRAM PREPROCESSING
# ========================================================================
def preprocess_stft(X_window, fs, nperseg, noverlap, max_freq):
    channels = X_window.shape[1]
    
    # Calculate expected dimensions
    num_freqs = nperseg // 2
    
    f_tmp, t_tmp, Zxx_tmp = stft(X_window[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    num_frames = len(t_tmp)
    
    mag_out = np.zeros((num_freqs, num_frames, channels), dtype=np.float32)
    
    for c in range(channels):
        f, t, Zxx = stft(X_window[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        
        mag = mag[:num_freqs, :]
        f = f[:num_freqs]
        
        mag[f > max_freq, :] = 0.0
            
        mag_out[:, :, c] = mag
        
    return mag_out, num_freqs, num_frames

def load_and_window_data(base_dir, nperseg, noverlap, max_freq):
    print(f"Extracting EMG windows (nperseg={nperseg}, noverlap={noverlap}, max_freq={max_freq})...")
    X_train_raw, y_train_raw, y_train_move_raw, t_train = [], [], [], []
    X_test_raw,  y_test_raw,  y_test_move_raw,  t_test  = [], [], [], []

    raw_stfts_by_class = {idx: [] for idx in range(len(classes))}
    raw_times_by_class = {idx: [] for idx in range(len(classes))}

    class_csv_map = {}
    min_trials = 999999
    for class_name in classes:
        files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        class_csv_map[class_name] = files
        if len(files) < min_trials:
            min_trials = len(files)

    num_freqs_detected = None
    num_frames_detected = None

    for class_idx, class_name in enumerate(classes):
        csv_files = class_csv_map[class_name][:min_trials]
        if len(csv_files) == 0:
            continue
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
            time_vals = pd.to_numeric(df[emg_time_cols[0]], errors='coerce').values

            emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
            emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values
            combined_data = emg_data

            req_len = int(WINDOW_S * EMG_FS)
            target_len = max(req_len, nperseg)
            trial_stfts, trial_labels, trial_move_labels, trial_times = [], [], [], []

            if class_name == "Trial_1_Rest":
                t_start_trial = time_vals[0]
                t_end_trial = time_vals[-1]
                t_end = t_start_trial + WINDOW_S
                
                while t_end <= t_end_trial:
                    t_start = t_end - WINDOW_S
                    mask = (time_vals >= t_start) & (time_vals <= t_end)
                    win = combined_data[mask]
    
                    if len(win) == 0:
                        win = np.zeros((target_len, N_CHANNELS))
                    elif len(win) < target_len:
                        pad = target_len - len(win)
                        win = np.pad(win, ((0, pad), (0, 0)), mode='constant')
                    elif len(win) > target_len:
                        win = win[:target_len]
    
                    stft_mag, freq_count, frame_count = preprocess_stft(win, EMG_FS, nperseg, noverlap, max_freq)
                    
                    if num_freqs_detected is None:
                        num_freqs_detected = freq_count
                        num_frames_detected = frame_count
                    
                    artificial_t = NOISE_START + (t_end - t_start_trial)
                    if artificial_t > MOVE_END: artificial_t = MOVE_END
                    
                    trial_stfts.append(stft_mag)
                    trial_labels.append(class_idx)
                    trial_move_labels.append(class_idx)
                    trial_times.append(artificial_t)
                    t_end += STEP_S
            else:
                t_end = NOISE_START + WINDOW_S
                while t_end <= MOVE_END:
                    t_start = t_end - WINDOW_S
                    mask = (time_vals >= t_start) & (time_vals <= t_end)
                    win = combined_data[mask]
    
                    if len(win) == 0:
                        win = np.zeros((target_len, N_CHANNELS))
                    elif len(win) < target_len:
                        pad = target_len - len(win)
                        win = np.pad(win, ((0, pad), (0, 0)), mode='constant')
                    elif len(win) > target_len:
                        win = win[:target_len]
    
                    stft_mag, freq_count, frame_count = preprocess_stft(win, EMG_FS, nperseg, noverlap, max_freq)
                    
                    if num_freqs_detected is None:
                        num_freqs_detected = freq_count
                        num_frames_detected = frame_count
                    
                    trial_stfts.append(stft_mag)
                    trial_labels.append(class_idx)
                    trial_move_labels.append(class_idx)
                    trial_times.append(t_end)
                    t_end += STEP_S

            trial_stfts = np.array(trial_stfts)
            trial_times = np.array(trial_times)
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                base_stft_profile = np.mean(trial_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_stfts = np.maximum(trial_stfts - base_stft_profile, 0.0)

            raw_stfts_by_class[class_idx].append(trial_stfts)
            raw_times_by_class[class_idx].append(trial_times)

            if is_train:
                X_train_raw.extend(trial_stfts)
                y_train_raw.extend(trial_labels)
                y_train_move_raw.extend(trial_move_labels)
                t_train.extend(trial_times)
            else:
                X_test_raw.extend(trial_stfts)
                y_test_raw.extend(trial_labels)
                y_test_move_raw.extend(trial_move_labels)
                t_test.extend(trial_times)

    return (
        np.array(X_train_raw), np.array(y_train_raw), np.array(y_train_move_raw), np.array(t_train),
        np.array(X_test_raw),  np.array(y_test_raw),  np.array(y_test_move_raw),  np.array(t_test),
        raw_stfts_by_class, raw_times_by_class, num_freqs_detected, num_frames_detected
    )

def plot_per_timestep_stacked_spectrograms(raw_stfts_by_class, raw_times_by_class, out_dir, nperseg, max_freq):
    os.makedirs(out_dir, exist_ok=True)
    emg_ch_idx  = list(range(N_SENSORS))

    for class_idx, class_name in enumerate(classes):
        label_short = class_labels_short[class_idx]
        trials = raw_stfts_by_class.get(class_idx, [])
        times_list = raw_times_by_class.get(class_idx, [])
        if not trials: continue

        all_stfts = np.concatenate(trials, axis=0)
        all_times = np.concatenate(times_list, axis=0)
        unique_times = sorted(set(np.round(all_times, 2)))
        class_out_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        f_bins = np.fft.rfftfreq(nperseg, 1.0/EMG_FS)
        limit_idx = np.sum(f_bins <= max_freq)

        all_mean_specs = []
        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if np.any(mask):
                all_mean_specs.append(np.mean(all_stfts[mask], axis=0))
        if all_mean_specs:
            all_mean_specs = np.array(all_mean_specs)
            # Calculate global min and max PER SENSOR across all timesteps
            global_vmin_per_sensor = all_mean_specs[:, :limit_idx, :].min(axis=(0, 1, 2))
            global_vmax_per_sensor = all_mean_specs[:, :limit_idx, :].max(axis=(0, 1, 2))
        else:
            global_vmin_per_sensor = np.zeros(N_SENSORS)
            global_vmax_per_sensor = np.ones(N_SENSORS)

        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if not np.any(mask): continue
            mean_spec = np.mean(all_stfts[mask], axis=0)

            t_start = t_val - WINDOW_S
            t_end = t_val

            fig, axes = plt.subplots(N_SENSORS, 1, figsize=(10, 2.2 * N_SENSORS), sharex=True)
            fig.suptitle(f"{label_short} — All Sensors Stacked (t = {t_val:+.2f}s, n={np.sum(mask)})", fontsize=15, fontweight='bold', y=1.02)

            for sensor_idx in range(N_SENSORS):
                ax = axes[sensor_idx]
                sensor_spec = mean_spec[:, :, sensor_idx]
                sensor_spec_sliced = sensor_spec[:limit_idx, :]

                vmin = global_vmin_per_sensor[sensor_idx]
                vmax = global_vmax_per_sensor[sensor_idx]
                im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                               cmap='turbo', interpolation='bilinear',
                               extent=[t_start, t_end, 0, max_freq],
                               vmin=vmin, vmax=vmax)
                
                ax.set_ylabel(f"S{sensor_idx + 1}", fontsize=10, fontweight='bold', rotation=0, labelpad=20)
                ax.tick_params(labelsize=8)
                fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

                if sensor_idx < N_SENSORS - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(f"Time (s)", fontsize=11)

            plt.tight_layout()
            plt.savefig(os.path.join(class_out_dir, f"{label_short}_all_sensors_stacked_{t_val:+.2f}s.png"), dpi=100, bbox_inches='tight')
            plt.close(fig)

def plot_continuous_spectrograms(raw_stfts_by_class, raw_times_by_class, out_dir, nperseg, max_freq):
    print("\nGenerating continuous stacked spectrograms...")
    os.makedirs(out_dir, exist_ok=True)
    
    for class_idx, class_name in enumerate(classes):
        label = class_labels_short[class_idx]
        trials = raw_stfts_by_class.get(class_idx, [])
        times_list = raw_times_by_class.get(class_idx, [])
        if not trials: continue

        all_stfts = np.concatenate(trials, axis=0)
        all_times = np.concatenate(times_list, axis=0)
        unique_times = sorted(set(np.round(all_times, 2)))
        
        class_out_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        n_windows = len(unique_times)
        num_freqs = all_stfts.shape[1]
        
        continuous = np.zeros((num_freqs, n_windows, N_SENSORS), dtype=np.float32)

        for w_idx, t_val in enumerate(unique_times):
            mask = np.isclose(np.round(all_times, 2), t_val)
            if not np.any(mask): continue
            mean_spec = np.mean(all_stfts[mask], axis=0)
            continuous[:, w_idx, :] = np.mean(mean_spec, axis=1)
            
        time_array = np.array(unique_times)
        
        f_bins = np.fft.rfftfreq(nperseg, 1.0/EMG_FS)
        limit_idx = np.sum(f_bins <= max_freq)

        cmaps_to_test = ['magma', 'viridis', 'plasma', 'inferno', 'turbo', 'jet', 'cividis']

        for cmap in cmaps_to_test:
            fig, axes = plt.subplots(N_SENSORS, 1, figsize=(14, 2.2 * N_SENSORS), sharex=True)
            fig.suptitle(f"{label} — All Sensors ({cmap})", fontsize=15, fontweight='bold', y=1.005)

            for sensor_idx in range(N_SENSORS):
                ax = axes[sensor_idx]
                sensor_spec = continuous[:, :, sensor_idx]
                sensor_spec_sliced = sensor_spec[:limit_idx, :]

                vmin, vmax = sensor_spec_sliced.min(), sensor_spec_sliced.max()
                im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                               cmap=cmap, interpolation='bilinear',
                               extent=[time_array[0], time_array[-1], 0, max_freq],
                               vmin=vmin, vmax=vmax)
                ax.axvline(x=0.0, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
                ax.set_ylabel(f"S{sensor_idx + 1}", fontsize=10, fontweight='bold', rotation=0, labelpad=20)
                ax.tick_params(labelsize=8)
                fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

                if sensor_idx < N_SENSORS - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("Time (s)", fontsize=11)

            plt.tight_layout()
            stacked_path = os.path.join(class_out_dir, f"{label}_all_sensors_stacked_{cmap}.png")
            fig.savefig(stacked_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        print(f"  {label}: Stacked plots saved for various colormaps")

# ========================================================================
# MODEL ARCHITECTURE (EMGNet) - "Best of Both Worlds" Config
# base_filters=32, dropout=0.0, kernel_size=5
# ========================================================================
class EMGNetClassifier(nn.Module):
    def __init__(self, num_channels=8, num_frames=9, num_frequencies=128, num_classes=7):
        super(EMGNetClassifier, self).__init__()
        self.num_frames = num_frames
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(32, 128, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(max(1, num_frames // 4), 1), mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(max(1, num_frames // 2), 1), mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 32, kernel_size=(5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(num_frames, 1), mode='bilinear', align_corners=False)
        )
        
        self.fc = nn.Linear(32 * num_frames, num_classes)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ========================================================================
# EXECUTION
# ========================================================================
def run_experiment(nperseg, noverlap, max_freq, results_dir):
    print(f"\n{'='*60}\nStarting Final Experiment: nperseg={nperseg}, noverlap={noverlap}, max_freq={max_freq}\n{'='*60}")
    os.makedirs(results_dir, exist_ok=True)
    
    (X_train, y_train_raw, y_train_move, t_train, 
     X_test, y_test_raw, y_test_move, t_test, 
     raw_stfts_by_class, raw_times_by_class, 
     num_freqs, num_frames) = load_and_window_data(DATA_DIR, nperseg, noverlap, max_freq)
     
    print(f"Extracted {len(X_train)} train windows and {len(X_test)} test windows.")
    print(f"Spectrogram dimensions: num_frequencies={num_freqs}, num_frames={num_frames}")
    
    if len(X_train) == 0:
        print("No data found! Skipping...")
        return
        
    mean_val = np.mean(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
    std_val = np.std(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
    
    X_train -= mean_val
    X_train /= (std_val + 1e-8)
    X_test -= mean_val
    X_test /= (std_val + 1e-8)
    
    timestep_stacked_dir = os.path.join(results_dir, "Per_Timestep_Stacked_Spectrograms")
    plot_per_timestep_stacked_spectrograms(raw_stfts_by_class, raw_times_by_class, timestep_stacked_dir, nperseg, max_freq)

    stacked_spec_dir = os.path.join(results_dir, "Continuous_Stacked_Spectrograms")
    plot_continuous_spectrograms(raw_stfts_by_class, raw_times_by_class, stacked_spec_dir, nperseg, max_freq)

    del raw_stfts_by_class, raw_times_by_class
    gc.collect()

    emg_ch_sample  = list(range(N_SENSORS))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Sample Spectrogram — Window 0 (post-normalisation)", fontsize=12)
    spec = np.mean(X_train[0, :, :, emg_ch_sample], axis=-1)
    f_bins = np.fft.rfftfreq(nperseg, 1.0/EMG_FS)
    limit_idx = np.sum(f_bins <= max_freq)
    spec_sliced = spec[:limit_idx, :]
    im = ax.imshow(spec_sliced, aspect='auto', origin='lower', cmap='turbo', extent=[0, spec_sliced.shape[1], 0, max_freq])
    ax.set_title("EMG (mean across 8 sensors)", fontsize=9)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sample_spectrogram.png"), dpi=150)
    plt.close()
    
    X_train_t = torch.from_numpy(X_train).permute(0, 3, 2, 1).float()
    X_test_t = torch.from_numpy(X_test).permute(0, 3, 2, 1).float()
    y_train_t = torch.from_numpy(y_train_raw).long()
    y_test_t = torch.from_numpy(y_test_raw).long()
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = EMGNetClassifier(num_channels=8, num_frames=num_frames, num_frequencies=num_freqs, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        print(f"Epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc*100:.2f}% - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc*100:.2f}%")
    
    plt.figure()
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "training_accuracy.png"))
    plt.close()
    
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pt"))
    
    # EVALUATION
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = y_test_move
    
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
                
    noise_mask = (t_test >= NOISE_START) & (t_test <= NOISE_END)
    emd_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    move_mask = (t_test > EMD_END) & (t_test <= MOVE_END)
    
    avg_overall = np.mean(true_labels == pred_labels) * 100.0
    avg_noise = np.mean(true_labels[noise_mask] == pred_labels[noise_mask]) * 100.0 if np.any(noise_mask) else 0.0
    avg_emd = np.mean(true_labels[emd_mask] == pred_labels[emd_mask]) * 100.0 if np.any(emd_mask) else 0.0
    avg_move = np.mean(true_labels[move_mask] == pred_labels[move_mask]) * 100.0 if np.any(move_mask) else 0.0

    summary_text = (
        f"{'='*60}\n"
        f"CNN CLASSIFICATION ACCURACY SUMMARY\n"
        f"Model: EMGNet Classifier (PyTorch)\n"
        f"Grid Config: nperseg={nperseg}, noverlap={noverlap}, max_freq={max_freq}\n"
        f"{'='*60}\n"
        f"Overall Average Accuracy:          {avg_overall:.2f}%\n"
        f"  Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
        f"  EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
        f"  Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
        f"{'='*60}\n"
    )
    print(f"\n{summary_text}")
    with open(os.path.join(results_dir, "accuracy_summary.txt"), 'w') as f:
        f.write(summary_text)

    plt.figure(figsize=(12, 6))
    plt.bar(unique_times, overall_acc_over_time, width=STEP_S*0.8, color='purple', edgecolor='black', alpha=0.8)
    plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
    plt.title(f'Overall Accuracy Over Time (nperseg={nperseg}, max_freq={max_freq})')
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
        plt.title(f'Accuracy Over Time for Class: {cls}')
        plt.xlabel('Time Relative to Onset (s)')
        plt.ylabel('Accuracy (%)')
        plt.xlim(NOISE_START, MOVE_END)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(class_plot_dir, f"Accuracy_Over_Time_{cls}.png"), dpi=150)
        plt.close()
        
    class_labels = classes + ["Rest"]
    
    if np.any(emd_mask):
        cm_emd = confusion_matrix(true_labels[emd_mask], pred_labels[emd_mask], labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_emd, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"EMD Phase Confusion Matrix (-0.6s to 0.0s)\nAccuracy: {avg_emd:.2f}%")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix_EMD_only.png"), dpi=150)
        plt.close()
        
    metrics_dir = os.path.join(results_dir, "Timestep_Metrics")
    cm_dir = os.path.join(metrics_dir, "Confusion_Matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    metrics_records = []
    for t_val in unique_times:
        mask = np.isclose(np.round(t_test, 2), t_val)
        y_true_t = true_labels[mask]
        y_pred_t = pred_labels[mask]
        if len(y_true_t) > 0:
            acc = np.mean(y_true_t == y_pred_t)
            p, r, f, _ = precision_recall_fscore_support(y_true_t, y_pred_t, average='macro', zero_division=0)
            metrics_records.append({'Time': t_val, 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f})
            cm_t = confusion_matrix(y_true_t, y_pred_t, labels=range(num_classes))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f"Confusion Matrix at {t_val}s\nAccuracy: {acc*100:.1f}%")
            plt.ylabel('True'); plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, f"CM_{t_val:.2f}s.png"), dpi=100)
            plt.close()
            
    pd.DataFrame(metrics_records).to_csv(os.path.join(metrics_dir, "timestep_metrics.csv"), index=False)
    print(f"Done evaluating final model.")


if __name__ == "__main__":
    run_experiment(nperseg=256, noverlap=254, max_freq=962.0, results_dir=BASE_RESULTS_DIR)
