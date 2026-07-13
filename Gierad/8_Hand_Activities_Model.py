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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials_shifted"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

classes = [
    "Trial_2_Ball_Short",
    "Trial_3_VPen_Short",
    "Trial_4_HPen_Short",
    "Trial_5_Bottle_Short",
    "Trial_6_Mug_Short",
    "Trial_7_Card_Short"
]

# Human-readable labels for plots
class_labels_short = ["Ball", "VPen", "HPen", "Bottle", "Mug", "Card"]
num_classes = len(classes) + 1  # +1 for Rest class

EMG_FS = 1925.0
IMU_FS = 74.0

NOISE_START  = -1.5
NOISE_END    = -0.6
EMD_START    = -0.6
EMD_END      =  0.0
MOVE_END     =  1.5

WINDOW_S = 0.2
STEP_S   = 0.05

# ========================================================================
# FILTERING
# ========================================================================
def build_emg_filters(fs, num_channels):
    filters = []
    nyq = fs / 2.0
    bp_high = min(400.0, nyq - 5.0)
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
# SPECTROGRAM PREPROCESSING
# ========================================================================
# Channel layout: 8 sensors × 1 signals = 8 channels
N_SENSORS = 8
N_SIGNALS = 1  # EMG only
N_CHANNELS = N_SENSORS * N_SIGNALS  # 8

def preprocess_stft(X_window, fs):
    """
    Computes STFT on a 200ms window.
    nperseg=256, noverlap=240 (step=16).
    Input:  X_window shape (samples, 8)  — 8 sensors × 1 channels (EMG)
    Returns (128, 9, 8) magnitude array, maintaining the 8*1 channel layout.
    """
    channels = X_window.shape[1]  # expected 8
    mag_out = np.zeros((128, 9, channels), dtype=np.float32)
    
    for c in range(channels):
        f, t, Zxx = stft(X_window[:, c], fs=fs, nperseg=256, noverlap=240)
        mag = np.abs(Zxx)
        mag = mag[:128, :]  # Drop highest freq bin, keep 128 freq bins
        # Discard frequencies above 128Hz by zeroing them out
        mag[f[:128] > 128.0, :] = 0.0
        
        if mag.shape[1] > 9:
            mag = mag[:, :9]
        elif mag.shape[1] < 9:
            pad_w = 9 - mag.shape[1]
            mag = np.pad(mag, ((0, 0), (0, pad_w)), mode='constant')
            
        mag_out[:, :, c] = mag
        
    return mag_out  # (128, 9, 32)

def load_and_window_data(base_dir):
    """
    Load all CSV trials, apply filtering, compute per-window STFTs (128, 9, 8),
    and return train/test splits plus per-class raw STFT stacks for averaging.

    Channel layout (8 channels = 8 sensors × 1):
      sensor k → indices [k] = [EMG]
    """
    print("Extracting EMG windows (8×1 = 8-channel spectrograms)...")
    X_train_raw, y_train_raw, y_train_move_raw, t_train = [], [], [], []
    X_test_raw,  y_test_raw,  y_test_move_raw,  t_test  = [], [], [], []

    # Accumulate raw (pre-norm) STFTs per class for average spectrogram plotting
    # raw_stfts_by_class[class_idx] = list of (n_windows, 128, 9, 8) arrays
    raw_stfts_by_class = {idx: [] for idx in range(len(classes))}
    raw_times_by_class = {idx: [] for idx in range(len(classes))}

    # Find the minimum trial count across all classes to enforce balance
    class_csv_map = {}
    min_trials = 999999
    for class_name in classes:
        files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        class_csv_map[class_name] = files
        if len(files) < min_trials:
            min_trials = len(files)
    print(f"Balanced loading: limiting each class to {min_trials} trials.")

    for class_idx, class_name in enumerate(classes):
        csv_files = class_csv_map[class_name][:min_trials]
        if len(csv_files) == 0:
            print(f"  WARNING: No files found for {class_name}")
            continue

        n_train = int(min_trials * 0.8)

        for file_idx, csv_path in enumerate(csv_files):
            is_train = (file_idx < n_train)

            with open(csv_path, 'r') as f:
                for _ in range(5):
                    f.readline()
                num_cols = len(f.readline().split(','))

            df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
            if len(df) <= 2:
                continue
            df = df.iloc[2:].reset_index(drop=True)

            # Time axis (from first EMG time column)
            emg_time_cols = [c for c in df.columns if 'Time' in c and 'ACC' not in c]
            if not emg_time_cols:
                continue
            time_vals = pd.to_numeric(df[emg_time_cols[0]], errors='coerce').values

            # ----------------------------------------------------------------
            # 1. EMG — all 8 sensors, filtered
            # ----------------------------------------------------------------
            emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
            emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values
            # emg_data = apply_emg_filters(emg_data, EMG_FS)  # shape (N, 8)

            # ----------------------------------------------------------------
            # 2. Use only EMG data
            # ----------------------------------------------------------------
            combined_data = emg_data

            req_len = int(WINDOW_S * EMG_FS)

            # ----------------------------------------------------------------
            # 4. Sliding window → STFT
            # ----------------------------------------------------------------
            trial_stfts       = []
            trial_labels      = []
            trial_move_labels = []
            trial_times       = []

            t_end = NOISE_START + WINDOW_S
            while t_end <= MOVE_END:
                t_start = t_end - WINDOW_S
                mask = (time_vals >= t_start) & (time_vals <= t_end)
                win = combined_data[mask]

                if len(win) == 0:
                    win = np.zeros((req_len, N_CHANNELS))
                elif len(win) < req_len:
                    pad = req_len - len(win)
                    win = np.pad(win, ((0, pad), (0, 0)), mode='constant')
                elif len(win) > req_len:
                    win = win[:req_len]

                stft_mag = preprocess_stft(win, EMG_FS)  # (128, 9, 8)
                label = len(classes) if t_end <= NOISE_END else class_idx

                trial_stfts.append(stft_mag)
                trial_labels.append(label)
                trial_move_labels.append(class_idx)  # Always the movement class
                trial_times.append(t_end)

                t_end += STEP_S

            # ----------------------------------------------------------------
            # 5. Spectral subtraction using noise window baseline
            # ----------------------------------------------------------------
            trial_stfts = np.array(trial_stfts)  # (T, 128, 9, 8)
            trial_times = np.array(trial_times)
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                # Average over noise windows AND time frames → (1, 128, 1, 8)
                base_stft_profile = np.mean(trial_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_stfts = np.maximum(trial_stfts - base_stft_profile, 0.0)

            # Accumulate raw STFTs for average spectrogram visualisation
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
        raw_stfts_by_class, raw_times_by_class
    )


# ========================================================================
# AVERAGE SPECTROGRAM VISUALISATION
# ========================================================================
def plot_average_spectrograms(raw_stfts_by_class, raw_times_by_class, out_dir):
    """
    For each class, compute the mean spectrogram (across all trials that fall
    in a given time window) and save a figure:
      - EMG    (mean across 8 EMG channels)

    Images are saved to out_dir/Average_Spectrograms/<class_name>/
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Generating average spectrogram plots...")

    # Indices of each modality across sensors
    emg_ch_idx  = list(range(N_SENSORS))  # [0,1,2,3,4,5,6,7]

    channel_groups = [
        ("EMG",   emg_ch_idx,  'magma'),
    ]

    for class_idx, class_name in enumerate(classes):
        label_short = class_labels_short[class_idx]
        trials = raw_stfts_by_class.get(class_idx, [])
        times_list = raw_times_by_class.get(class_idx, [])
        if not trials:
            continue

        # Stack all trial window arrays: list of (T, 128, 9, 8) → (total_wins, 128, 9, 8)
        all_stfts = np.concatenate(trials, axis=0)    # (total_wins, 128, 9, 8)
        all_times = np.concatenate(times_list, axis=0) # (total_wins,)

        unique_times = sorted(set(np.round(all_times, 2)))
        class_out_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        for t_val in unique_times:
            mask = np.isclose(np.round(all_times, 2), t_val)
            if not np.any(mask):
                continue

            windows_at_t = all_stfts[mask]  # (n_trials, 128, 9, 8)

            # Mean spectrogram across trials → (128, 9, 8)
            mean_spec = np.mean(windows_at_t, axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            fig.suptitle(
                f"{label_short} — Average Spectrogram at t = {t_val:+.2f}s"
                f"  (n={np.sum(mask)} trials)",
                fontsize=13, fontweight='bold'
            )

            # Average across modality channels → (128, 9)
            spec_mod = np.mean(mean_spec[:, :, emg_ch_idx], axis=-1)
            f_max = 128.0 * (EMG_FS / 256.0)
            im = ax.imshow(
                spec_mod, aspect='auto', origin='lower',
                cmap='magma', interpolation='nearest',
                extent=[0, spec_mod.shape[1], 0, f_max]
            )
            ax.set_ylim(0, 128.0)
            ax.set_title("EMG", fontsize=11)
            ax.set_xlabel("Time frame", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            fname = f"avg_spec_{t_val:+.2f}s.png"
            plt.savefig(os.path.join(class_out_dir, fname), dpi=100)
            plt.close(fig)

        print(f"  Saved average spectrograms for {label_short} ({len(unique_times)} time steps)")

    print(f"Average spectrogram plots saved to: {out_dir}")

# ========================================================================
# MODEL ARCHITECTURE (FIGLAB VGG STYLE)
# ========================================================================
def build_model(input_shape, num_classes):
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 1))) # 128->64, Time remains 9
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 1))) # 64->32, Time remains 9
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 1))) # 32->16, Time remains 9
    
    # Block 4
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 1))) # 16->8, Time remains 9
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ========================================================================
# EXECUTION
# ========================================================================
if __name__ == "__main__":
    X_train, y_train_raw, y_train_move, t_train, X_test, y_test_raw, y_test_move, t_test, \
        raw_stfts_by_class, raw_times_by_class = load_and_window_data(DATA_DIR)
    print(f"Extracted {len(X_train)} train windows and {len(X_test)} test windows.")
    
    if len(X_train) == 0:
        print("No data found! Check paths.")
        exit()
        
    print("Normalizing STFT features per-frequency and per-channel...")
    mean_val = np.mean(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
    std_val = np.std(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
    
    X_train -= mean_val
    X_train /= (std_val + 1e-8)
    X_test -= mean_val
    X_test /= (std_val + 1e-8)
    
    # Verify spectrogram shape: must be (N, 128, 9, 8) = 8 sensors × 1 channels
    print(f"Train spectrogram shape: {X_train.shape}  (expected: (N, 128, 9, {N_CHANNELS}))")
    assert X_train.shape[1] == 128 and X_train.shape[2] == 9 and X_train.shape[3] == N_CHANNELS, \
        f"Unexpected shape {X_train.shape[1:]}; expected (128, 9, {N_CHANNELS})"

    # ── Average spectrogram visualisation (before normalisation) ──────────
    avg_spec_dir = os.path.join(RESULTS_DIR, "Average_Spectrograms")
    if os.path.exists(avg_spec_dir) and any(os.path.isdir(os.path.join(avg_spec_dir, d)) for d in os.listdir(avg_spec_dir)):
        print(f"Average spectrograms already exist in {avg_spec_dir}. Skipping generation...")
    else:
        plot_average_spectrograms(raw_stfts_by_class, raw_times_by_class, avg_spec_dir)

    # Free up memory to prevent crashes
    import gc
    del raw_stfts_by_class
    del raw_times_by_class
    gc.collect()

    # Plot a sample normalised spectrogram (one panel, first window)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    emg_ch_sample  = list(range(N_SENSORS))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Sample Spectrogram — Window 0 (post-normalisation)", fontsize=12)
    spec = np.mean(X_train[0, :, :, emg_ch_sample], axis=-1)
    f_max = 128.0 * (EMG_FS / 256.0)
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, spec.shape[1], 0, f_max])
    ax.set_ylim(0, 128.0)
    ax.set_title("EMG (mean across 8 sensors)", fontsize=9)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_spectrogram.png"), dpi=150)
    plt.close()
    
    y_train = to_categorical(y_train_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)
    
    print("Building model...")
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()
    
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "training_accuracy.png"))
    plt.close()
    
    # ========================================================================
    # COMPREHENSIVE TEMPORAL EVALUATION (Aligned with Combined Pipeline)
    # ========================================================================
    print("Evaluating model and generating comprehensive plots...")
    model.save(os.path.join(RESULTS_DIR, "final_model.h5"))
    
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    # Use movement-class labels as ground truth for accuracy (matching Combined pipeline).
    # Rest is still a valid prediction class but predicting Rest is always wrong —
    # accuracy measures whether the model identifies the correct movement class.
    true_labels = y_test_move
    
    unique_times = sorted(list(set(np.round(t_test, 2))))
    overall_acc_over_time = []
    class_accuracies_over_time = {cls: [] for cls in classes}
    
    for t_val in unique_times:
        mask = np.isclose(np.round(t_test, 2), t_val)
        y_true_t = true_labels[mask]
        y_pred_t = pred_labels[mask]
        
        if len(y_true_t) > 0:
            overall_acc_over_time.append(np.mean(y_true_t == y_pred_t) * 100.0)
        else:
            overall_acc_over_time.append(np.nan)
            
        for cls_idx, cls in enumerate(classes):
            cls_mask = (y_true_t == cls_idx)
            if np.sum(cls_mask) > 0:
                acc = np.mean(y_true_t[cls_mask] == y_pred_t[cls_mask]) * 100.0
                class_accuracies_over_time[cls].append(acc)
            else:
                class_accuracies_over_time[cls].append(np.nan)
                
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
        f"Model: Gierad CNN\n"
        f"{'='*60}\n"
        f"Overall Average Accuracy:          {avg_overall:.2f}%\n"
        f"  Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
        f"  EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
        f"  Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
        f"{'='*60}\n"
    )
    print(f"\n{summary_text}")
    with open(os.path.join(RESULTS_DIR, "accuracy_summary.txt"), 'w') as f:
        f.write(summary_text)

    # Plot Accuracy over time for best model
    plt.figure(figsize=(12, 6))
    plt.bar(unique_times, overall_acc_over_time, width=STEP_S*0.8, color='purple', edgecolor='black', alpha=0.8)
    plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
    plt.title('Overall Accuracy Over Time - CNN Model')
    plt.xlabel('Time Relative to Onset (s)')
    plt.ylabel('Accuracy (%)')
    plt.xlim(NOISE_START, MOVE_END)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Per_Class_Accuracy_Over_Time_BarChart.png"), dpi=150)
    plt.close()
    
    # Per-Class Accuracy Over Time
    class_plot_dir = os.path.join(RESULTS_DIR, "Per_Class_BarCharts")
    os.makedirs(class_plot_dir, exist_ok=True)
    
    for cls in classes:
        cls_accs = class_accuracies_over_time[cls]
        plt.figure(figsize=(12, 6))
        plt.bar(unique_times, cls_accs, width=STEP_S*0.8, color='teal', alpha=0.8, edgecolor='black')
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
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
    
    # EMD Phase Confusion Matrix
    if np.any(emd_mask):
        emd_true = true_labels[emd_mask]
        emd_pred = pred_labels[emd_mask]
        cm_emd = confusion_matrix(emd_true, emd_pred, labels=range(num_classes))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_emd, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"EMD Phase Confusion Matrix (-0.6s to 0.0s)\nAccuracy: {avg_emd:.2f}%")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_EMD_only.png"), dpi=150)
        plt.close()
        
    # Per-timestep Confusion Matrices and Metrics
    metrics_dir = os.path.join(RESULTS_DIR, "Timestep_Metrics")
    cm_dir = os.path.join(metrics_dir, "Confusion_Matrices")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    metrics_records = []
    for t_val in unique_times:
        mask = np.isclose(np.round(t_test, 2), t_val)
        y_true_t = true_labels[mask]
        y_pred_t = pred_labels[mask]
        
        if len(y_true_t) > 0:
            acc = np.mean(y_true_t == y_pred_t)
            p, r, f, _ = precision_recall_fscore_support(y_true_t, y_pred_t, average='macro', zero_division=0)
            metrics_records.append({
                'Time': t_val, 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f
            })
            
            cm_t = confusion_matrix(y_true_t, y_pred_t, labels=range(num_classes))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f"Confusion Matrix at {t_val}s\nAccuracy: {acc*100:.1f}%")
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, f"CM_{t_val:.2f}s.png"), dpi=100)
            plt.close()
            
    pd.DataFrame(metrics_records).to_csv(os.path.join(metrics_dir, "timestep_metrics.csv"), index=False)
    print("Done! All evaluation metrics matching the Combined pipeline have been successfully generated.")
