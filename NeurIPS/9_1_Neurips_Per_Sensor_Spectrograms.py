import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, sosfiltfilt, stft, tf2sos

# ========================================================================
# CONFIG
# ========================================================================
DATA_DIR = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials_shifted"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Offline_Training_Results", timestamp + "_Per_Sensor_Spectrograms")
os.makedirs(RESULTS_DIR, exist_ok=True)

classes = [
    "Trial_2_Ball_Short",
    "Trial_3_VPen_Short",
    "Trial_4_HPen_Short",
    "Trial_5_Bottle_Short",
    "Trial_6_Mug_Short",
    "Trial_7_Card_Short"
]
class_labels_short = ["Ball", "VPen", "HPen", "Bottle", "Mug", "Card"]

EMG_FS       = 1925.0
NOISE_START  = -1.5
NOISE_END    = -0.6
MOVE_END     =  1.5
WINDOW_S     = 0.2
STEP_S       = 0.05

N_SENSORS  = 8
N_CHANNELS = 8

SUBSET_TIMES = [-1.3, -0.8, -0.3, 0.0, 0.3, 0.8, 1.3]

# ========================================================================
# FILTERING (same as script 9)
# ========================================================================
def build_emg_filters(fs, num_channels):
    filters = []
    nyq = fs / 2.0
    # Discard frequencies above 64Hz in bandpass filtering
    bp_high = min(64.0, nyq - 5.0)
    for i in range(num_channels):
        b, a = iirnotch(50.0, 30.0, fs)
        sos_notch = tf2sos(b, a)
        sos_bp = butter(4, [40.0, bp_high], btype='band', fs=fs, output='sos')
        filters.append({'sos_n': sos_notch, 'sos_b': sos_bp})
    return filters

def apply_emg_filters(data, fs):
    num_channels = data.shape[1]
    filters = build_emg_filters(fs, num_channels)
    filt_data = np.zeros_like(data)
    for i in range(num_channels):
        f_cfg = filters[i]
        x = data[:, i]
        x = sosfiltfilt(f_cfg['sos_n'], x)
        x = sosfiltfilt(f_cfg['sos_b'], x)
        filt_data[:, i] = x
    return filt_data

# ========================================================================
# STFT (same as script 9)
# ========================================================================
def preprocess_stft(X_window, fs):
    channels = X_window.shape[1]
    mag_out = np.zeros((128, 9, channels), dtype=np.float32)
    for c in range(channels):
        f, t, Zxx = stft(X_window[:, c], fs=fs, nperseg=256, noverlap=240)
        mag = np.abs(Zxx)
        mag = mag[:128, :]
        # Discard frequencies above 64Hz by zeroing them out
        mag[f[:128] > 64.0, :] = 0.0
        
        if mag.shape[1] > 9:
            mag = mag[:, :9]
        elif mag.shape[1] < 9:
            pad_w = 9 - mag.shape[1]
            mag = np.pad(mag, ((0, 0), (0, pad_w)), mode='constant')
        mag_out[:, :, c] = mag
    return mag_out  # (128, 9, 8)

# ========================================================================
# LOAD DATA → per-class, per-trial STFTs with spectral subtraction
# ========================================================================
def load_all_trials(base_dir):
    """
    Returns:
        data_by_class[class_idx] = list of (trial_stfts, trial_times)
            trial_stfts: (n_windows, 128, 9, 8)
            trial_times: (n_windows,)
    """
    print("Loading and computing per-trial spectrograms...")
    data_by_class = {idx: [] for idx in range(len(classes))}

    for class_idx, class_name in enumerate(classes):
        csv_files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        if not csv_files:
            print(f"  WARNING: No files found for {class_name}")
            continue

        for csv_path in csv_files:
            with open(csv_path, 'r') as f:
                for _ in range(5):
                    f.readline()
                num_cols = len(f.readline().split(','))

            df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
            if len(df) <= 2:
                continue
            df = df.iloc[2:].reset_index(drop=True)

            emg_time_cols = [c for c in df.columns if 'Time' in c and 'ACC' not in c]
            if not emg_time_cols:
                continue
            time_vals = pd.to_numeric(df[emg_time_cols[0]], errors='coerce').values

            emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
            emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values
            # emg_data = apply_emg_filters(emg_data, EMG_FS)

            req_len = int(WINDOW_S * EMG_FS)

            trial_stfts = []
            trial_times = []

            t_end = NOISE_START + WINDOW_S
            while t_end <= MOVE_END:
                t_start = t_end - WINDOW_S
                mask = (time_vals >= t_start) & (time_vals <= t_end)
                win = emg_data[mask]

                if len(win) == 0:
                    win = np.zeros((req_len, N_CHANNELS))
                elif len(win) < req_len:
                    pad = req_len - len(win)
                    win = np.pad(win, ((0, pad), (0, 0)), mode='constant')
                elif len(win) > req_len:
                    win = win[:req_len]

                stft_mag = preprocess_stft(win, EMG_FS)
                trial_stfts.append(stft_mag)
                trial_times.append(t_end)

                t_end += STEP_S

            # Spectral subtraction
            trial_stfts = np.array(trial_stfts)
            trial_times = np.array(trial_times)
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                base_stft_profile = np.mean(trial_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_stfts = np.maximum(trial_stfts - base_stft_profile, 0.0)

            data_by_class[class_idx].append((trial_stfts, trial_times))

        print(f"  {class_labels_short[class_idx]}: {len(data_by_class[class_idx])} trials loaded")

    return data_by_class

# ========================================================================
# COMPUTE MEAN SPECTROGRAM PER SENSOR PER TIME STEP (across trials)
# ========================================================================
def compute_mean_spectrograms(data_by_class):
    """
    Returns:
        mean_specs[class_idx] = dict mapping rounded time → (128, 9, 8) mean spectrogram
        unique_times_by_class[class_idx] = sorted list of unique rounded times
    """
    mean_specs = {}
    unique_times_by_class = {}

    for class_idx in range(len(classes)):
        trials = data_by_class[class_idx]
        if not trials:
            mean_specs[class_idx] = {}
            unique_times_by_class[class_idx] = []
            continue

        all_stfts = np.concatenate([t[0] for t in trials], axis=0)
        all_times = np.concatenate([t[1] for t in trials], axis=0)

        unique_times = sorted(set(np.round(all_times, 2)))
        time_to_mean = {}

        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if np.any(mask):
                time_to_mean[t_val] = np.mean(all_stfts[mask], axis=0)  # (128, 9, 8)

        mean_specs[class_idx] = time_to_mean
        unique_times_by_class[class_idx] = unique_times

    return mean_specs, unique_times_by_class

# ========================================================================
# PLOT 1: Individual per-sensor spectrograms at every time step
# ========================================================================
def plot_individual_sensor_spectrograms(mean_specs, unique_times_by_class, out_dir):
    """
    For each object → 8 sensor folders → one spectrogram per time step.
    """
    print("\nGenerating individual per-sensor spectrograms...")

    for class_idx, class_name in enumerate(classes):
        label = class_labels_short[class_idx]
        time_to_mean = mean_specs[class_idx]
        unique_times = unique_times_by_class[class_idx]
        if not unique_times:
            continue

        class_dir = os.path.join(out_dir, class_name)

        for sensor_idx in range(N_SENSORS):
            sensor_dir = os.path.join(class_dir, f"Sensor_{sensor_idx + 1}")
            os.makedirs(sensor_dir, exist_ok=True)

            f_bins = np.fft.rfftfreq(256, 1.0/EMG_FS)
            limit_idx = np.sum(f_bins <= 64.0)
            for t_val in unique_times:
                spec = time_to_mean[t_val][:, :, sensor_idx]  # (128, 9)
                spec_sliced = spec[:limit_idx, :]

                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(spec_sliced, aspect='auto', origin='lower',
                              cmap='magma', interpolation='nearest',
                              extent=[0, 8, 0, 64.0])
                ax.set_title(f"{label} — Sensor {sensor_idx + 1} — t = {t_val:+.2f}s",
                             fontsize=10, fontweight='bold')
                ax.set_xlabel("Time frame", fontsize=9)
                ax.set_ylabel("Frequency (Hz)", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(sensor_dir, f"spec_t{t_val:+.2f}s.png"), dpi=100)
                plt.close(fig)

        print(f"  {label}: 8 sensor folders × {len(unique_times)} time steps saved")

# ========================================================================
# PLOT 2: Continuous spectrograms (one wide plot per sensor per object)
# ========================================================================
def plot_continuous_spectrograms(mean_specs, unique_times_by_class, out_dir):
    """
    For each object, produce:
      - 8 individual continuous spectrograms (one per sensor)
      - 1 stacked figure with all 8 sensors vertically for easy comparison
    
    Each spectrogram shows frequency (y) vs continuous time (x) from -1.5s to 1.5s.
    """
    print("\nGenerating continuous spectrograms...")

    for class_idx, class_name in enumerate(classes):
        label = class_labels_short[class_idx]
        time_to_mean = mean_specs[class_idx]
        unique_times = unique_times_by_class[class_idx]
        if not unique_times:
            continue

        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        n_windows = len(unique_times)
        continuous = np.zeros((128, n_windows, N_SENSORS), dtype=np.float32)

        for w_idx, t_val in enumerate(unique_times):
            spec = time_to_mean[t_val]  # (128, 9, 8)
            continuous[:, w_idx, :] = np.mean(spec, axis=1)  # (128, 8)

        time_array = np.array(unique_times)

        # ── Individual per-sensor continuous spectrograms ──
        f_bins = np.fft.rfftfreq(256, 1.0/EMG_FS)
        limit_idx = np.sum(f_bins <= 64.0)

        for sensor_idx in range(N_SENSORS):
            sensor_spec = continuous[:, :, sensor_idx]
            sensor_spec_sliced = sensor_spec[:limit_idx, :]

            fig, ax = plt.subplots(figsize=(14, 4))
            im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                           cmap='magma', interpolation='bilinear',
                           extent=[time_array[0], time_array[-1], 0, 64.0])
            ax.axvline(x=0.0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(f"{label} — Sensor {sensor_idx + 1}",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Frequency (Hz)", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Magnitude")
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, f"Sensor_{sensor_idx + 1}_continuous.png"), dpi=150)
            plt.close(fig)

        # ── Stacked 8-panel figure (matching the requested layout) ──
        fig, axes = plt.subplots(N_SENSORS, 1, figsize=(14, 2.2 * N_SENSORS), sharex=True)
        fig.suptitle(f"{label} — All Sensors", fontsize=15, fontweight='bold', y=1.005)

        for sensor_idx in range(N_SENSORS):
            ax = axes[sensor_idx]
            sensor_spec = continuous[:, :, sensor_idx]
            sensor_spec_sliced = sensor_spec[:limit_idx, :]

            # Per-sensor colour scale
            vmin, vmax = sensor_spec_sliced.min(), sensor_spec_sliced.max()
            im = ax.imshow(sensor_spec_sliced, aspect='auto', origin='lower',
                           cmap='magma', interpolation='bilinear',
                           extent=[time_array[0], time_array[-1], 0, 64.0],
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
        stacked_path = os.path.join(class_dir, f"{label}_all_sensors_stacked.png")
        fig.savefig(stacked_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  {label}: 8 individual + 1 stacked plot saved")

# ========================================================================
# MAIN
# ========================================================================
if __name__ == "__main__":
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    data_by_class = load_all_trials(DATA_DIR)
    mean_specs, unique_times_by_class = compute_mean_spectrograms(data_by_class)

    # Free raw trial data after computing means
    del data_by_class
    import gc
    gc.collect()

    plot_individual_sensor_spectrograms(mean_specs, unique_times_by_class, RESULTS_DIR)
    plot_continuous_spectrograms(mean_specs, unique_times_by_class, RESULTS_DIR)

    print(f"\nDone! All outputs saved to: {RESULTS_DIR}")
