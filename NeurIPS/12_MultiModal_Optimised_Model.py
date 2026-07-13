import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import datetime
import gc

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = r"E:\OfflineEMG\extracted_trials_shifted"
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results", timestamp + "_MultiModal_Model")

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

N_SENSORS = 8
N_EMG_CHANNELS = 8
N_IMU_CHANNELS = 24

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
    print("Extracting EMG and IMU windows...")
    X_train_emg, X_train_imu, y_train_raw, y_train_move, t_train = [], [], [], [], []
    X_test_emg, X_test_imu, y_test_raw, y_test_move, t_test = [], [], [], [], []

    raw_stfts_by_class_emg = {idx: [] for idx in range(len(classes))}
    raw_stfts_by_class_imu = {idx: [] for idx in range(len(classes))}
    raw_times_by_class = {idx: [] for idx in range(len(classes))}

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

            trial_emg_stfts, trial_imu_stfts, trial_times = [], [], []

            if class_name == "Trial_1_Rest":
                t_start_trial = emg_time_vals[0]
                t_end_trial = emg_time_vals[-1]
                t_end = t_start_trial + WINDOW_S
                
                while t_end <= t_end_trial:
                    t_start = t_end - WINDOW_S
                    
                    # EMG Window
                    mask_emg = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    win_emg = emg_data[mask_emg]
                    if len(win_emg) == 0:
                        win_emg = np.zeros((target_len_emg, N_EMG_CHANNELS))
                    elif len(win_emg) < target_len_emg:
                        pad = target_len_emg - len(win_emg)
                        win_emg = np.pad(win_emg, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_emg) > target_len_emg:
                        win_emg = win_emg[:target_len_emg]
    
                    # IMU Window
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
                    trial_times.append(artificial_t)
    
                    if is_train:
                        X_train_emg.append(emg_stft)
                        X_train_imu.append(imu_stft)
                        y_train_raw.append(class_idx)
                        y_train_move.append(class_idx)
                        t_train.append(artificial_t)
                    else:
                        X_test_emg.append(emg_stft)
                        X_test_imu.append(imu_stft)
                        y_test_raw.append(class_idx)
                        y_test_move.append(class_idx)
                        t_test.append(artificial_t)
                    t_end += STEP_S
            else:
                t_end = NOISE_START + WINDOW_S
                while t_end <= MOVE_END:
                    t_start = t_end - WINDOW_S
                    
                    # EMG Window
                    mask_emg = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    win_emg = emg_data[mask_emg]
                    if len(win_emg) == 0:
                        win_emg = np.zeros((target_len_emg, N_EMG_CHANNELS))
                    elif len(win_emg) < target_len_emg:
                        pad = target_len_emg - len(win_emg)
                        win_emg = np.pad(win_emg, ((0, pad), (0, 0)), mode='constant')
                    elif len(win_emg) > target_len_emg:
                        win_emg = win_emg[:target_len_emg]
    
                    # IMU Window
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
                    trial_times.append(t_end)
    
                    if is_train:
                        X_train_emg.append(emg_stft)
                        X_train_imu.append(imu_stft)
                        y_train_raw.append(class_idx)
                        y_train_move.append(class_idx)
                        t_train.append(t_end)
                    else:
                        X_test_emg.append(emg_stft)
                        X_test_imu.append(imu_stft)
                        y_test_raw.append(class_idx)
                        y_test_move.append(class_idx)
                        t_test.append(t_end)
                    t_end += STEP_S

            trial_emg_stfts = np.array(trial_emg_stfts)
            trial_imu_stfts = np.array(trial_imu_stfts)
            trial_times = np.array(trial_times)
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                base_emg_profile = np.mean(trial_emg_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_emg_stfts = np.maximum(trial_emg_stfts - base_emg_profile, 0.0)
                
                base_imu_profile = np.mean(trial_imu_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_imu_stfts = np.maximum(trial_imu_stfts - base_imu_profile, 0.0)

            raw_stfts_by_class_emg[class_idx].append(trial_emg_stfts)
            raw_stfts_by_class_imu[class_idx].append(trial_imu_stfts)
            raw_times_by_class[class_idx].append(trial_times)

    return (
        np.array(X_train_emg), np.array(X_train_imu), np.array(y_train_raw), np.array(y_train_move), np.array(t_train),
        np.array(X_test_emg),  np.array(X_test_imu),  np.array(y_test_raw),  np.array(y_test_move),  np.array(t_test),
        raw_stfts_by_class_emg, raw_stfts_by_class_imu, raw_times_by_class
    )

def plot_per_timestep_stacked_spectrograms(raw_stfts_by_class, raw_times_by_class, out_dir, nperseg, max_freq, fs, n_sensors, prefix):
    os.makedirs(out_dir, exist_ok=True)

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

        f_bins = np.fft.rfftfreq(nperseg, 1.0/fs)
        if max_freq is not None:
            limit_idx = np.sum(f_bins <= max_freq)
        else:
            limit_idx = len(f_bins)

        all_mean_specs = []
        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if np.any(mask):
                all_mean_specs.append(np.mean(all_stfts[mask], axis=0))
        if all_mean_specs:
            all_mean_specs = np.array(all_mean_specs)
            global_vmin_per_sensor = all_mean_specs[:, :limit_idx, :].min(axis=(0, 1, 2))
            global_vmax_per_sensor = all_mean_specs[:, :limit_idx, :].max(axis=(0, 1, 2))
        else:
            global_vmin_per_sensor = np.zeros(n_sensors)
            global_vmax_per_sensor = np.ones(n_sensors)

        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if not np.any(mask): continue
            mean_spec = np.mean(all_stfts[mask], axis=0)

            t_start = t_val - WINDOW_S
            t_end = t_val

            fig, axes = plt.subplots(n_sensors, 1, figsize=(10, min(1.5 * n_sensors, 40)), sharex=True)
            if n_sensors == 1:
                axes = [axes]
            
            fig.suptitle(f"{label_short} ({prefix}) — Stacked (t = {t_val:+.2f}s, n={np.sum(mask)})", fontsize=15, fontweight='bold', y=1.02)

            for sensor_idx in range(n_sensors):
                ax = axes[sensor_idx]
                sensor_spec = mean_spec[:, :, sensor_idx]
                sensor_spec_sliced = sensor_spec[:limit_idx, :]

                vmin = global_vmin_per_sensor[sensor_idx]
                vmax = global_vmax_per_sensor[sensor_idx]
                im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                               cmap='turbo', interpolation='bilinear',
                               extent=[t_start, t_end, 0, max_freq if max_freq else f_bins[-1]],
                               vmin=vmin, vmax=vmax)
                
                ax.set_ylabel(f"S{sensor_idx + 1}", fontsize=10 if n_sensors <= 8 else 8, fontweight='bold', rotation=0, labelpad=20)
                ax.tick_params(labelsize=8)
                fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

                if sensor_idx < n_sensors - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(f"Time (s)", fontsize=11)

            plt.tight_layout()
            plt.savefig(os.path.join(class_out_dir, f"{prefix}_{label_short}_stacked_{t_val:+.2f}s.png"), dpi=100, bbox_inches='tight')
            plt.close(fig)

def plot_continuous_spectrograms(raw_stfts_by_class, raw_times_by_class, out_dir, nperseg, max_freq, fs, n_sensors, prefix):
    print(f"\nGenerating continuous stacked spectrograms for {prefix}...")
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
        
        continuous = np.zeros((num_freqs, n_windows, n_sensors), dtype=np.float32)

        for w_idx, t_val in enumerate(unique_times):
            mask = np.isclose(np.round(all_times, 2), t_val)
            if not np.any(mask): continue
            mean_spec = np.mean(all_stfts[mask], axis=0)
            continuous[:, w_idx, :] = np.mean(mean_spec, axis=1)
            
        time_array = np.array(unique_times)
        
        f_bins = np.fft.rfftfreq(nperseg, 1.0/fs)
        if max_freq is not None:
            limit_idx = np.sum(f_bins <= max_freq)
        else:
            limit_idx = len(f_bins)

        cmaps_to_test = ['turbo']

        for cmap in cmaps_to_test:
            fig, axes = plt.subplots(n_sensors, 1, figsize=(14, min(1.5 * n_sensors, 40)), sharex=True)
            if n_sensors == 1:
                axes = [axes]
            fig.suptitle(f"{prefix} - {label} — All Sensors ({cmap})", fontsize=15, fontweight='bold', y=1.005)

            for sensor_idx in range(n_sensors):
                ax = axes[sensor_idx]
                sensor_spec = continuous[:, :, sensor_idx]
                sensor_spec_sliced = sensor_spec[:limit_idx, :]

                vmin, vmax = sensor_spec_sliced.min(), sensor_spec_sliced.max()
                im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                               cmap=cmap, interpolation='bilinear',
                               extent=[time_array[0], time_array[-1], 0, max_freq if max_freq else f_bins[-1]],
                               vmin=vmin, vmax=vmax)
                ax.axvline(x=0.0, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
                ax.set_ylabel(f"S{sensor_idx + 1}", fontsize=10 if n_sensors <= 8 else 8, fontweight='bold', rotation=0, labelpad=20)
                ax.tick_params(labelsize=8)
                fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

                if sensor_idx < n_sensors - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("Time (s)", fontsize=11)

            plt.tight_layout()
            stacked_path = os.path.join(class_out_dir, f"{prefix}_{label}_all_sensors_stacked_{cmap}.png")
            fig.savefig(stacked_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        print(f"  {label} ({prefix}): Stacked plots saved")

class MultiModalNet(nn.Module):
    def __init__(self, emg_channels=8, emg_frames=26, emg_freqs=128, 
                 imu_channels=24, imu_frames=8, imu_freqs=4, num_classes=7):
        super(MultiModalNet, self).__init__()
        
        self.emg_encoder = nn.Sequential(
            nn.Conv2d(emg_channels, 32, kernel_size=(3, 3), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(32, 128, kernel_size=(3, 3), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Conv2d(imu_channels, 32, kernel_size=(2, 2), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
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
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
                
    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        emg_features = emg_features.view(emg_features.size(0), -1)
        
        imu_features = self.imu_encoder(imu_x)
        imu_features = imu_features.view(imu_features.size(0), -1)
        
        combined = torch.cat((emg_features, imu_features), dim=1)
        return self.fc(combined)

def run_experiment(emg_nperseg=256, emg_noverlap=240, emg_max_freq=962.0, 
                   imu_nperseg=8, imu_noverlap=7, imu_max_freq=None, results_dir=""):
    print(f"Starting Multi-Modal Experiment")
    os.makedirs(results_dir, exist_ok=True)
    
    (X_train_emg, X_train_imu, y_train_raw, y_train_move, t_train, 
     X_test_emg, X_test_imu, y_test_raw, y_test_move, t_test,
     raw_stfts_by_class_emg, raw_stfts_by_class_imu, raw_times_by_class) = load_and_window_data(
         DATA_DIR, emg_nperseg, emg_noverlap, emg_max_freq, imu_nperseg, imu_noverlap, imu_max_freq)
     
    print(f"Extracted {len(X_train_emg)} train windows and {len(X_test_emg)} test windows.")
    print(f"EMG Spectrogram dimensions: {X_train_emg.shape[1:]}")
    print(f"IMU Spectrogram dimensions: {X_train_imu.shape[1:]}")
    
    if len(X_train_emg) == 0:
        return
        
    # Generate Spectrogram Plots
    emg_ts_dir = os.path.join(results_dir, "Per_Timestep_Stacked_Spectrograms_EMG")
    plot_per_timestep_stacked_spectrograms(raw_stfts_by_class_emg, raw_times_by_class, emg_ts_dir, emg_nperseg, emg_max_freq, EMG_FS, N_EMG_CHANNELS, "EMG")

    imu_ts_dir = os.path.join(results_dir, "Per_Timestep_Stacked_Spectrograms_IMU")
    plot_per_timestep_stacked_spectrograms(raw_stfts_by_class_imu, raw_times_by_class, imu_ts_dir, imu_nperseg, imu_max_freq, IMU_FS, N_IMU_CHANNELS, "IMU")

    emg_cont_dir = os.path.join(results_dir, "Continuous_Stacked_Spectrograms_EMG")
    plot_continuous_spectrograms(raw_stfts_by_class_emg, raw_times_by_class, emg_cont_dir, emg_nperseg, emg_max_freq, EMG_FS, N_EMG_CHANNELS, "EMG")

    imu_cont_dir = os.path.join(results_dir, "Continuous_Stacked_Spectrograms_IMU")
    plot_continuous_spectrograms(raw_stfts_by_class_imu, raw_times_by_class, imu_cont_dir, imu_nperseg, imu_max_freq, IMU_FS, N_IMU_CHANNELS, "IMU")

    del raw_stfts_by_class_emg, raw_stfts_by_class_imu, raw_times_by_class
    gc.collect()

    # Normalise EMG
    mean_val_emg = np.mean(X_train_emg, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_val_emg = np.std(X_train_emg, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_train_emg = (X_train_emg - mean_val_emg) / (std_val_emg + 1e-8)
    X_test_emg = (X_test_emg - mean_val_emg) / (std_val_emg + 1e-8)
    
    # Normalise IMU
    mean_val_imu = np.mean(X_train_imu, axis=(0, 2), keepdims=True, dtype=np.float32)
    std_val_imu = np.std(X_train_imu, axis=(0, 2), keepdims=True, dtype=np.float32)
    X_train_imu = (X_train_imu - mean_val_imu) / (std_val_imu + 1e-8)
    X_test_imu = (X_test_imu - mean_val_imu) / (std_val_imu + 1e-8)
    
    # Save a sample spectrogram for EMG after normalisation
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Sample Spectrogram — Window 0 (post-normalisation)", fontsize=12)
    spec = np.mean(X_train_emg[0, :, :, :], axis=-1)
    f_bins = np.fft.rfftfreq(emg_nperseg, 1.0/EMG_FS)
    limit_idx = np.sum(f_bins <= emg_max_freq) if emg_max_freq else len(f_bins)
    spec_sliced = spec[:limit_idx, :]
    im = ax.imshow(spec_sliced, aspect='auto', origin='lower', cmap='turbo', extent=[0, spec_sliced.shape[1], 0, emg_max_freq if emg_max_freq else f_bins[-1]])
    ax.set_title("EMG (mean across 8 sensors)", fontsize=9)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sample_spectrogram.png"), dpi=150)
    plt.close()

    # Apply EMD-focused training mask
    emd_train_mask = (t_train <= EMD_END)
    X_train_emg = X_train_emg[emd_train_mask]
    X_train_imu = X_train_imu[emd_train_mask]
    y_train_raw = y_train_raw[emd_train_mask]
    y_train_move = y_train_move[emd_train_mask]
    t_train = t_train[emd_train_mask]
    
    print(f"Filtered training windows to EMD focus (t <= {EMD_END}s). Train windows: {len(X_train_emg)}")

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
    
    # Create EMD-only test loader for validation
    emd_test_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    test_emd_dataset = TensorDataset(
        X_test_emg_t[emd_test_mask], 
        X_test_imu_t[emd_test_mask], 
        y_test_t[emd_test_mask]
    )
    test_emd_loader = DataLoader(test_emd_dataset, batch_size=32, shuffle=False)
    
    model = MultiModalNet(emg_channels=N_EMG_CHANNELS, emg_frames=X_train_emg_t.shape[2], emg_freqs=X_train_emg_t.shape[3],
                          imu_channels=N_IMU_CHANNELS, imu_frames=X_train_imu_t.shape[2], imu_freqs=X_train_imu_t.shape[3], 
                          num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'val_emd_accuracy': []}
    
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
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs_emg, inputs_imu, targets in test_loader:
                inputs_emg, inputs_imu, targets = inputs_emg.to(device), inputs_imu.to(device), targets.to(device)
                outputs = model(inputs_emg, inputs_imu)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Evaluate on EMD-only test set (for checkpointing)
        emd_val_correct, emd_val_total = 0, 0
        with torch.no_grad():
            for inputs_emg, inputs_imu, targets in test_emd_loader:
                inputs_emg, inputs_imu, targets = inputs_emg.to(device), inputs_imu.to(device), targets.to(device)
                outputs = model(inputs_emg, inputs_imu)
                _, predicted = outputs.max(1)
                emd_val_total += targets.size(0)
                emd_val_correct += predicted.eq(targets).sum().item()
                
        epoch_emd_val_acc = emd_val_correct / emd_val_total if emd_val_total > 0 else 0
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        history['val_emd_accuracy'].append(epoch_emd_val_acc)
        print(f"Epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc*100:.2f}% - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc*100:.2f}% - val_emd_accuracy: {epoch_emd_val_acc*100:.2f}%")
        
        if epoch_emd_val_acc >= best_emd_acc:
            best_emd_acc = epoch_emd_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved new best model (EMD Acc: {best_emd_acc*100:.2f}%)")
    
    print(f"Loading best model for final evaluation (Best EMD Acc: {best_emd_acc*100:.2f}%)")
    model.load_state_dict(torch.load(best_model_path))
    
    plt.figure()
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation (All)')
    plt.plot(history['val_emd_accuracy'], label='Validation (EMD Only)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "training_accuracy.png"))
    plt.close()
    
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pt"))
    
    # Evaluate
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
        f"MULTI-MODAL CLASSIFICATION ACCURACY SUMMARY\n"
        f"Model: MultiModalNet (EMG + IMU Spectrograms)\n"
        f"EMG STFT: nperseg={emg_nperseg}, noverlap={emg_noverlap}\n"
        f"IMU STFT: nperseg={imu_nperseg}, noverlap={imu_noverlap}\n"
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
    plt.title(f'Overall Accuracy Over Time (MultiModal)')
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
    run_experiment(emg_nperseg=256, emg_noverlap=254, emg_max_freq=962.0, 
                   imu_nperseg=8, imu_noverlap=7, imu_max_freq=None, 
                   results_dir=BASE_RESULTS_DIR)
