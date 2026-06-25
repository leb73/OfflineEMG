import os
import sys
import glob
import time
import json
import collections
import random
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, welch, iirnotch, tf2sos

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit


# Make runs deterministic
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


print("starting")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
EMG_FS          = 1926.0
WINDOW_SIZE_MS  = 150             
STEP_SIZE_MS    = 50              # 50ms step size as requested
SEQUENCE_LENGTH = 5
L2_REG = 0.002

# Phase boundaries (seconds relative to movement onset at t=0)
NOISE_START  = -1.5
NOISE_END    = -0.6   # Background noise phase ends here
EMD_START    = -0.6    # Electromechanical delay starts here
EMD_END      =  0.0    # Movement onset
MOVE_END     =  1.5    # Movement phase ends here

REST_LABEL   = "rest"  # Pseudo-class for background noise

# ============================================================================
# FILTER SETUP
# ============================================================================
def build_filters(fs, num_channels, channel_types):
    filters = []
    nyq = fs / 2.0
    bp_high = min(400.0, nyq - 5.0)
    for i in range(num_channels):
        ctype = channel_types[i] if channel_types else 0
        if ctype == 0:
            if nyq > 55.0:
                b, a = iirnotch(50.0, 30.0, fs)
                sos_notch = tf2sos(b, a)
            else:
                sos_notch = None
            sos_bp = butter(4, [40.0, bp_high], btype='band', fs=fs, output='sos')
            filters.append({'type': 'emg', 'sos_n': sos_notch, 'sos_b': sos_bp})
        else:
            lp_freq = min(20.0, nyq - 5.0)
            sos_lp = butter(2, max(lp_freq, 5.0), btype='low', fs=fs, output='sos')
            filters.append({'type': 'imu', 'sos': sos_lp})
    return filters

def apply_filters(data, fs, channel_types):
    num_channels = data.shape[1]
    filters = build_filters(fs, num_channels, channel_types)
    filt_data = np.zeros_like(data)
    for i in range(num_channels):
        f_cfg = filters[i]
        x = data[:, i]
        if f_cfg['type'] == 'emg':
            if f_cfg['sos_n'] is not None:
                x = sosfilt(f_cfg['sos_n'], x)
            x = sosfilt(f_cfg['sos_b'], x)
        else:
            x = sosfilt(f_cfg['sos'], x)
        filt_data[:, i] = x
    return filt_data

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
_MFCC_CACHE = {}

def _mfcc_vectorized(x_2d, fs, n_mfcc=4):
    n_fft = min(x_2d.shape[0], 256)
    X = np.abs(np.fft.rfft(x_2d, n=n_fft, axis=0))
    n_filters = 16
    cache_key = (fs, X.shape[0])
    if cache_key in _MFCC_CACHE:
        filterbank = _MFCC_CACHE[cache_key]
    else:
        low_mel    = 0
        high_mel   = 2595 * np.log10(1 + (fs / 2) / 700)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
        bins       = np.floor((n_fft + 1) * hz_points / fs).astype(int)
        bins       = np.clip(bins, 0, X.shape[0] - 1)
        filterbank = np.zeros((n_filters, X.shape[0]))
        for i in range(n_filters):
            left, centre, right = bins[i], bins[i + 1], bins[i + 2]
            if centre > left:
                filterbank[i, left:centre] = np.linspace(0, 1, centre - left)
            if right > centre:
                filterbank[i, centre:right] = np.linspace(1, 0, right - centre)
        _MFCC_CACHE[cache_key] = filterbank

    mel_spec = filterbank @ (X ** 2)
    mel_spec = np.where(mel_spec == 0, 1e-10, mel_spec)
    log_mel  = np.log(mel_spec)

    n_coeff = max(n_mfcc, 4)
    mfccs   = np.zeros((n_coeff, x_2d.shape[1]))
    
    for k in range(n_coeff):
        cos_term = np.cos(np.pi * k * (np.arange(n_filters) + 0.5) / n_filters)
        mfccs[k, :] = np.sum(log_mel * cos_term[:, np.newaxis], axis=0)

    return mfccs[0, :], mfccs[min(2, n_coeff - 1), :]

def extract_window_features(x_window, fs, channel_types=None):
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    rms_val = np.sqrt(np.mean(x ** 2, axis=0))
    mav_val = np.mean(np.abs(x), axis=0)
    p2p_val = np.ptp(x, axis=0)
    rss_val = np.sqrt(np.sum(x ** 2, axis=0))
    
    safe_mav = np.where(mav_val == 0, 1e-10, mav_val)
    sf_val = rms_val / safe_mav
    
    if x.shape[0] > 1:
        zcr_val = ((x[:-1, :] * x[1:, :]) < 0).sum(axis=0) / x.shape[0]
    else:
        zcr_val = np.zeros(num_channels)
        
    mn_f_val = np.zeros(num_channels)
    med_f_val = np.zeros(num_channels)
    mfcc1_val = np.zeros(num_channels)
    mfcc3_val = np.zeros(num_channels)
    
    if channel_types is not None:
        emg_idx = [i for i, ctype in enumerate(channel_types) if ctype == 0 and i < num_channels]
    else:
        emg_idx = list(range(num_channels))
        
    if emg_idx:
        x_emg = x[:, emg_idx]
        m1, m3 = _mfcc_vectorized(x_emg, fs)
        for i, ch_idx in enumerate(emg_idx):
            mfcc1_val[ch_idx] = m1[i]
            mfcc3_val[ch_idx] = m3[i]
            
        f, psd = welch(x_emg, fs=fs, nperseg=min(x_emg.shape[0], 64), axis=0)
        total_p = np.sum(psd, axis=0)
        safe_total_p = np.where(total_p == 0, 1e-10, total_p)
        
        f_col = f[:, np.newaxis]
        mn_f_emg = np.sum(f_col * psd, axis=0) / safe_total_p
        cum_p = np.cumsum(psd, axis=0)
        target_p = total_p / 2.0
        
        for i, ch_idx in enumerate(emg_idx):
            if total_p[i] > 0:
                mn_f_val[ch_idx] = mn_f_emg[i]
                s_idx = np.searchsorted(cum_p[:, i], target_p[i])
                med_f_val[ch_idx] = f[min(s_idx, len(f) - 1)]

    feat_stack = np.column_stack([
        rms_val, p2p_val, mn_f_val, med_f_val, 
        sf_val, rss_val, mav_val, zcr_val, mfcc1_val, mfcc3_val
    ])
    return feat_stack.flatten().astype(np.float32)

def get_sub_windows(df_len, window_samples, step_samples):
    windows, start = [], 0
    while (start + window_samples) <= df_len:
        windows.append((start, start + window_samples))
        start += step_samples
    return windows


def compute_window_time(window_end_sample, fs, trial_offset=-1.5):
    """Convert window end sample index to time relative to onset (t=0).
    
    The data spans -1.5s to +1.5s, so sample 0 corresponds to -1.5s.
    Window time is taken at the END of the window.
    """
    return (window_end_sample / fs) + trial_offset


def extract_raw_features(ds_dict, window_samples, step_samples, channel_types):
    """Extract features for each trial, with per-window time stamps.
    
    Uses the noise phase (-1.5s to -0.6s) for baseline subtraction per trial.
    Returns list of (feat_seq, time_seq, cls_name) tuples.
    """
    raw_trials = []
    for cls_name, trials in ds_dict.items():
        for arr in trials:
            sub_wins = get_sub_windows(len(arr), window_samples, step_samples)
            if len(sub_wins) < SEQUENCE_LENGTH:
                continue
            feat_seq = []
            time_seq = []
            for s, e in sub_wins:
                feat = extract_window_features(arr[s:e], EMG_FS, channel_types)
                feat_seq.append(feat)
                time_seq.append(compute_window_time(e, EMG_FS))
            feat_seq = np.array(feat_seq)
            
            # Use noise phase (-1.5s to -0.6s) for baseline subtraction
            noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
            if len(noise_indices) > 0:
                baseline_mean = np.mean(feat_seq[noise_indices], axis=0, keepdims=True)
                feat_seq = feat_seq - baseline_mean
            
            raw_trials.append((feat_seq, time_seq, cls_name))
    return raw_trials


def assign_phase_label(t, true_class):
    """Assign label based on temporal phase.
    
    -1.5s to -0.6s → 'rest' (background noise)
    -0.6s to  0.0s → true_class (electromechanical delay)
     0.0s to +1.5s → true_class (movement)
    """
    if t <= NOISE_END:
        return REST_LABEL
    else:
        return true_class


def build_sequences(raw_trials, scaler):
    """Build input sequences with phase-aware labelling.
    
    Every window gets a label:
      - 'rest' during noise phase
      - true class during EMD and movement phases
    """
    X_all, Y_raw, groups, times_all = [], [], [], []
    t_idx = 0
    for feat_seq, time_seq, cls_name in raw_trials:
        feat_seq = scaler.transform(feat_seq)
        
        for i in range(len(feat_seq)):
            seq = feat_seq[max(0, i - SEQUENCE_LENGTH + 1) : i + 1]
            if len(seq) < SEQUENCE_LENGTH:
                pad = [np.zeros_like(seq[0])] * (SEQUENCE_LENGTH - len(seq))
                seq = pad + list(seq)
            X_all.append(seq)
            Y_raw.append(assign_phase_label(time_seq[i], cls_name))
            groups.append(t_idx)
            times_all.append(time_seq[i])
                
        t_idx += 1
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    X_arr = np.array(X_all)
    Y_arr = np.array(Y_raw)
    groups_arr = np.array(groups)
    times_arr = np.array(times_all)
    
    return X_arr, Y_arr, groups_arr, times_arr

# ============================================================================
# MODEL DEFINITION
# ============================================================================
def create_model(input_shape, num_classes, cnn_filters=64, kernel_size=3, lstm_units=128, lstm_dropout=0.2, dense_units=256, dense_dropout=0.5, learning_rate=1e-4):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)

    x = layers.Conv1D(cnn_filters, kernel_size, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=lstm_units // 4)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    x = layers.TimeDistributed(layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG)))(x)
    x = layers.TimeDistributed(layers.Dropout(dense_dropout))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "extracted_trials")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from: {base_dir}")
    print(f"Results will be saved to: {out_dir}")
    
    dataset = collections.defaultdict(list)
    channel_types = None
    
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_*_Short")))
    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        parts = folder_name.split("_")
        if len(parts) >= 3:
            cls_label = parts[2].lower()
        else:
            continue
            
        for csv_path in sorted(glob.glob(os.path.join(tf_path, "movement_*.csv"))):
            if not os.path.isfile(csv_path):
                continue
                
            try:
                with open(csv_path, 'r') as f:
                    for _ in range(5): f.readline()
                    num_cols = len(f.readline().split(','))
                
                df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
                if len(df) <= 2: continue
                # Drop units and metadata rows
                df = df.iloc[2:].reset_index(drop=True)
                
                # Filter for -1.5s to 1.5s based on the primary time column
                time_col = [c for c in df.columns if 'Time' in c][0]
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                df = df[(df[time_col] >= NOISE_START) & (df[time_col] <= MOVE_END)].reset_index(drop=True)
                
                emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
                
                if channel_types is None:
                    channel_types = [0] * len(emg_cols)
                    
                df_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
                raw_data = df_data.values
                
                # Apply filters as per the original GUI pipeline
                filt_data = apply_filters(raw_data, EMG_FS, channel_types)
                dataset[cls_label].append(filt_data)
                
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    if not dataset:
        print("No valid trials found. Exiting.")
        return
        
    print("\nDataset Summary:")
    for cls, trials in dataset.items():
        print(f"  {cls}: {len(trials)} trials")
    
    # Build class list: movement classes + rest
    movement_classes = sorted(list(dataset.keys()))
    all_classes = movement_classes + [REST_LABEL]
    le = LabelEncoder()
    le.fit(all_classes)
    
    print(f"\nClasses: {list(le.classes_)}")
    print(f"  Movement classes: {movement_classes}")
    print(f"  Rest class index: {le.transform([REST_LABEL])[0]}")
    
    # Ensure all classes have the same number of trials
    min_trials = min([len(v) for v in dataset.values()])
    for cls in dataset.keys():
        dataset[cls] = random.sample(dataset[cls], min_trials)

    # 70/15/15 Split per class based on trials
    train_ds, val_ds, test_ds = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    rng = np.random.RandomState(42)
    
    for cls_name in sorted(dataset.keys()):
        arr_list = dataset[cls_name]
        n = len(arr_list)
        indices = list(range(n))
        rng.shuffle(indices)
        
        n_train = max(1, int(0.70 * n))
        n_val   = max(1, int(0.15 * n)) if n > 1 else 0
        n_test  = n - n_train - n_val
        
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:n_train+n_val]
        test_indices  = indices[n_train+n_val:]
        
        train_ds[cls_name] = [arr_list[i] for i in train_indices]
        val_ds[cls_name]   = [arr_list[i] for i in val_indices]
        test_ds[cls_name]  = [arr_list[i] for i in test_indices]

    print(f"\nData Splitting (Trials):")
    print(f"  Train: {sum(len(v) for v in train_ds.values())}")
    print(f"  Val:   {sum(len(v) for v in val_ds.values())}")
    print(f"  Test:  {sum(len(v) for v in test_ds.values())}")
    
    window_samples = int((WINDOW_SIZE_MS / 1000) * EMG_FS)
    step_samples   = int((STEP_SIZE_MS / 1000) * EMG_FS)
    
    print(f"\nWindow: {WINDOW_SIZE_MS}ms ({window_samples} samples), Step: {STEP_SIZE_MS}ms ({step_samples} samples)")
    
    print("\nExtracting Features (this may take a moment)...")
    train_raw = extract_raw_features(train_ds, window_samples, step_samples, channel_types)
    val_raw = extract_raw_features(val_ds, window_samples, step_samples, channel_types)
    
    scaler = StandardScaler()
    all_train_feats = np.vstack([t[0] for t in train_raw]) if train_raw else np.array([])
    if len(all_train_feats) > 0:
        scaler.fit(all_train_feats)
        
    X_train, Y_train_raw, _, train_times = build_sequences(train_raw, scaler)
    X_val, Y_val_raw, val_groups, val_times = build_sequences(val_raw, scaler)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("Not enough sequences for training/validation. Exiting.")
        return
    
    # Print label distribution
    unique_train, counts_train = np.unique(Y_train_raw, return_counts=True)
    print("\nTraining label distribution:")
    for lbl, cnt in zip(unique_train, counts_train):
        print(f"  {lbl}: {cnt}")
        
    Y_train_enc = le.transform(Y_train_raw)
    Y_val_enc   = le.transform(Y_val_raw)
    
    Y_train = np.repeat(Y_train_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
    Y_val   = np.repeat(Y_val_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
    
    N_tr, T_, F_ = X_train.shape
    N_v, _, _    = X_val.shape
    
    X_t = X_train
    X_v = X_val
    
    input_shape = (T_, F_)
    num_classes = len(all_classes)
    
    print(f"\nSequence Shapes: Train {X_t.shape}, Val {X_v.shape}")
    print(f"Num classes (including rest): {num_classes}")
    
    # HPO Loop
    param_space = {
        'cnn_filters':   [32, 64],
        'lstm_units':    [64, 128],
        'lstm_dropout':  [0.1, 0.2],
        'dense_units':   [128, 256],
        'dense_dropout': [0.3, 0.5],
        'learning_rate': [1e-4, 5e-4],
    }
    baseline = {'cnn_filters': 64, 'lstm_units': 128, 'lstm_dropout': 0.2, 'dense_units': 256, 'dense_dropout': 0.5, 'learning_rate': 1e-4}
    
    es  = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    best_acc = -1
    best_params = baseline
    num_iter = 4
    
    for search_id in range(num_iter):
        tf.keras.backend.clear_session()
        p = baseline if search_id == 0 else {k: random.choice(v) for k, v in param_space.items()}
        lbl = "Baseline" if search_id == 0 else f"Random {search_id}"
        print(f"\nHPO Search {search_id+1}/{num_iter} ({lbl}): {p}")
        
        model = create_model(input_shape, num_classes, **p)
        model.fit(X_t, Y_train, epochs=15, batch_size=8, validation_data=(X_v, Y_val), callbacks=[es, rlr], verbose=1)
        
        _, acc = model.evaluate(X_v, Y_val, verbose=0)
        print(f"Validation Accuracy: {acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_params = p
            
    print(f"\nBest HPO Params chosen (Val Acc: {best_acc*100:.1f}%): {best_params}")
    
    # Build final model
    tf.keras.backend.clear_session()
    final_model = create_model(input_shape, num_classes, **best_params)
    # Combine Train + Val for final fit
    X_full = np.concatenate([X_t, X_v])
    Y_full = np.concatenate([Y_train, Y_val])
    print("\nTraining final model on combined Train+Val sets...")
    final_model.fit(X_full, Y_full, epochs=20, batch_size=8, verbose=1)
    
    # Save the model so we don't lose good runs!
    model_path = os.path.join(out_dir, "final_model.h5")
    final_model.save(model_path)
    print(f"\nSaved final model to: {model_path}")
    
    # ========================================================================
    # EVALUATION — Per-timestep sliding window metrics
    # ========================================================================
    print("\nEvaluating on Test Set with per-timestep metrics...")
    
    plot_dir = os.path.join(out_dir, "Accuracy_Plots")
    metrics_dir = os.path.join(out_dir, "Metrics_Over_Time")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Collect predictions per trial, aligned by time step
    # global_true_rest: includes rest labels for noise phase (for confusion matrices)
    # global_true_move: always the true movement class (for accuracy-over-time plots)
    global_true_rest = []   # list of arrays with rest labels during noise
    global_true_move = []   # list of arrays with true movement class at all times
    global_preds = []       # list of arrays: model predictions
    global_pred_times = None
    
    for cls_name, trials in test_ds.items():
        cls_idx = le.transform([cls_name])[0]
        rest_idx = le.transform([REST_LABEL])[0]
        
        all_trial_accuracies = []
        pred_times = None
        
        for t_idx, arr in enumerate(trials):
            sub_wins = get_sub_windows(len(arr), window_samples, step_samples)
            if len(sub_wins) < SEQUENCE_LENGTH: continue
            
            feat_seq = []
            time_seq = []
            for s, e in sub_wins:
                feat = extract_window_features(arr[s:e], EMG_FS, channel_types)
                feat_seq.append(feat)
                time_seq.append(compute_window_time(e, EMG_FS))
                
            feat_seq = np.array(feat_seq)
            
            # Baseline subtraction using noise phase
            noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
            if len(noise_indices) > 0:
                baseline_mean = np.mean(feat_seq[noise_indices], axis=0, keepdims=True)
                feat_seq = feat_seq - baseline_mean
            
            feat_seq = scaler.transform(feat_seq)
                
            X_trial = []
            for i in range(len(feat_seq)):
                seq = feat_seq[max(0, i - SEQUENCE_LENGTH + 1) : i + 1]
                if len(seq) < SEQUENCE_LENGTH:
                    pad = [np.zeros_like(seq[0])] * (SEQUENCE_LENGTH - len(seq))
                    seq = pad + list(seq)
                X_trial.append(seq)
            X_trial = np.array(X_trial)
            
            preds = final_model(X_trial, training=False).numpy()
            pred_classes = np.argmax(preds[:, -1, :], axis=-1)
            
            # True labels WITH rest (for confusion matrices)
            true_labels_rest = np.array([
                rest_idx if t <= NOISE_END else cls_idx
                for t in time_seq
            ])
            # True labels as MOVEMENT CLASS at all times (for accuracy plots)
            true_labels_move = np.full(len(time_seq), cls_idx)
            
            # Per-class accuracy: does the model predict the correct MOVEMENT class?
            accuracies = (pred_classes == cls_idx).astype(float) * 100.0
            all_trial_accuracies.append(accuracies)
            
            global_true_rest.append(true_labels_rest)
            global_true_move.append(true_labels_move)
            global_preds.append(pred_classes)
            
            if pred_times is None:
                pred_times = time_seq
            if global_pred_times is None:
                global_pred_times = pred_times
                
        if not all_trial_accuracies:
            continue
            
        # Per-class average accuracy bar plot
        min_len = min(len(acc) for acc in all_trial_accuracies)
        avg_accuracies = np.mean([acc[:min_len] for acc in all_trial_accuracies], axis=0)
        avg_pred_times = pred_times[:min_len]
        
        plt.figure(figsize=(12, 5))
        plt.bar(avg_pred_times, avg_accuracies, width=0.04, edgecolor="black", color='#00d4ff', alpha=0.8)
        plt.axvline(x=NOISE_END, color='orange', linestyle='--', linewidth=1.5, label=f'Noise→EMD ({NOISE_END}s)')
        plt.axvline(x=0.0, color='r', linestyle='--', linewidth=1.5, label='Movement Onset')
        plt.ylim(0, 105)
        plt.xlim(NOISE_START, MOVE_END)
        plt.title(f"Average Accuracy Over Time — Class: {cls_name} ({len(all_trial_accuracies)} trials)")
        plt.xlabel("Time Relative to Onset (s)")
        plt.ylabel(f"Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Average_{cls_name}.png"), dpi=150)
        plt.close()
        
    # ========================================================================
    # Global per-timestep metrics
    # ========================================================================
    if global_preds:
        global_min_len = min(len(p) for p in global_preds)
        global_preds_trunc = np.array([p[:global_min_len] for p in global_preds])
        global_true_rest_trunc = np.array([t[:global_min_len] for t in global_true_rest])
        global_true_move_trunc = np.array([t[:global_min_len] for t in global_true_move])
        global_times_trunc = global_pred_times[:global_min_len]
        
        # Movement-class accuracy over time (what the user wants to see)
        overall_move_accuracies = []
        
        # Per-class accuracy over time for each movement class
        class_accuracies_over_time = {cls: [] for cls in movement_classes}
        
        print(f"\nGenerating metrics for {global_min_len} time steps...")
        
        for t_i in range(global_min_len):
            y_true_rest_t = global_true_rest_trunc[:, t_i]  # with rest labels
            y_true_move_t = global_true_move_trunc[:, t_i]  # movement class only
            y_pred_t = global_preds_trunc[:, t_i]
            
            # Movement classification accuracy: does the model predict the correct
            # movement class? During noise, it predicts 'rest' → wrong → low accuracy.
            move_acc_t = accuracy_score(y_true_move_t, y_pred_t) * 100.0
            overall_move_accuracies.append(move_acc_t)
            
            # Per-movement-class accuracy (recall) at this time step
            for cls_name in movement_classes:
                cls_idx = le.transform([cls_name])[0]
                mask = (y_true_move_t == cls_idx)
                if mask.sum() > 0:
                    cls_acc = (y_pred_t[mask] == cls_idx).mean() * 100.0
                    class_accuracies_over_time[cls_name].append(cls_acc)
                else:
                    class_accuracies_over_time[cls_name].append(np.nan)
            
            # Confusion matrix for this time step (includes rest in labels)
            t_val = global_times_trunc[t_i]
            cm_t = confusion_matrix(y_true_rest_t, y_pred_t, labels=range(len(le.classes_)))
            # Also compute accuracy including rest for the confusion matrix title
            acc_with_rest_t = accuracy_score(y_true_rest_t, y_pred_t) * 100.0
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f"Confusion Matrix (t={t_val:+.3f}s)\nMove Acc: {move_acc_t:.1f}% | With Rest: {acc_with_rest_t:.1f}%")
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f"cm_t{t_i:03d}_{t_val:+.3f}s.png"), dpi=100)
            plt.close()
            
            # Classification report for this time step (uses rest labels)
            report = classification_report(
                y_true_rest_t, y_pred_t, 
                labels=range(len(le.classes_)), 
                target_names=le.classes_, 
                output_dict=True, 
                zero_division=0
            )
            pd.DataFrame(report).transpose().to_csv(
                os.path.join(metrics_dir, f"report_t{t_i:03d}_{t_val:+.3f}s.csv")
            )
        
        # ====================================================================
        # Overall accuracy over time plot
        # ====================================================================
        plt.figure(figsize=(12, 6))
        plt.plot(global_times_trunc, overall_move_accuracies, marker='o', markersize=3, linewidth=2, color='#2196F3', label='Movement Classification Accuracy')
        plt.axvline(x=NOISE_END, color='orange', linestyle='--', linewidth=1.5, label=f'Noise→EMD ({NOISE_END}s)')
        plt.axvline(x=0.0, color='r', linestyle='--', linewidth=1.5, label='Movement Onset')
        chance_level = 100.0 / len(movement_classes)
        plt.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.5, label=f'Chance ({chance_level:.0f}%)')
        plt.title('Movement Classification Accuracy Over Time')
        plt.xlabel('Time Relative to Onset (s)')
        plt.ylabel('Accuracy (%)')
        plt.xlim(NOISE_START, MOVE_END)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Overall_Accuracy_Over_Time.png"), dpi=150)
        plt.close()
        
        # ====================================================================
        # Per-class accuracy over time plot
        # ====================================================================
        plt.figure(figsize=(14, 7))
        
        cmap = plt.colormaps['tab10']
        
        for i, cls_name in enumerate(movement_classes):
            accs = class_accuracies_over_time[cls_name]
            plt.plot(global_times_trunc, accs, marker='.', markersize=2, linewidth=1.5, 
                     color=cmap(i), label=f'{cls_name}', alpha=0.85)
        
        plt.axvline(x=NOISE_END, color='orange', linestyle=':', linewidth=1.5, label=f'Noise→EMD ({NOISE_END}s)')
        plt.axvline(x=0.0, color='r', linestyle=':', linewidth=1.5, label='Movement Onset')
        
        chance_level = 100.0 / len(movement_classes)
        plt.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.5, label=f'Chance ({chance_level:.0f}%)')
        
        plt.title('Per-Class Movement Accuracy (Recall) Over Time')
        plt.xlabel('Time Relative to Onset (s)')
        plt.ylabel('Accuracy (%)')
        plt.xlim(NOISE_START, MOVE_END)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Per_Class_Accuracy_Over_Time.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # ====================================================================
        # Average accuracy summary
        # ====================================================================
        avg_overall = np.mean(overall_move_accuracies)
        
        # Compute average accuracy per phase
        noise_mask = np.array(global_times_trunc) <= NOISE_END
        emd_mask = (np.array(global_times_trunc) > NOISE_END) & (np.array(global_times_trunc) <= EMD_END)
        move_mask = np.array(global_times_trunc) > EMD_END
        
        avg_noise = np.mean([overall_move_accuracies[i] for i in range(len(overall_move_accuracies)) if noise_mask[i]]) if noise_mask.any() else 0
        avg_emd = np.mean([overall_move_accuracies[i] for i in range(len(overall_move_accuracies)) if emd_mask[i]]) if emd_mask.any() else 0
        avg_move = np.mean([overall_move_accuracies[i] for i in range(len(overall_move_accuracies)) if move_mask[i]]) if move_mask.any() else 0
        
        summary_text = (
            f"{'='*60}\n"
            f"MOVEMENT CLASSIFICATION ACCURACY SUMMARY\n"
            f"{'='*60}\n"
            f"Overall Average Accuracy:          {avg_overall:.2f}%\n"
            f"  Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
            f"  EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
            f"  Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
            f"{'='*60}\n"
            f"\nPer-class average movement accuracy over time:\n"
        )
        
        for cls_name in movement_classes:
            accs = class_accuracies_over_time[cls_name]
            valid_accs = [a for a in accs if not np.isnan(a)]
            if valid_accs:
                summary_text += f"  {cls_name:>10s}: {np.mean(valid_accs):.2f}%\n"
            else:
                summary_text += f"  {cls_name:>10s}: N/A\n"
        
        print(f"\n{summary_text}")
        
        # Save summary to file
        with open(os.path.join(out_dir, "accuracy_summary.txt"), 'w') as f:
            f.write(summary_text)
        
        # Also save a final overall confusion matrix (aggregated over all timesteps)
        # Use rest labels for the confusion matrix so rest predictions are visible
        all_true_flat = global_true_rest_trunc.flatten()
        all_pred_flat = global_preds_trunc.flatten()
        
        cm_overall = confusion_matrix(all_true_flat, all_pred_flat, labels=range(len(le.classes_)))
        overall_acc = accuracy_score(all_true_flat, all_pred_flat) * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Overall Test Confusion Matrix (All Time Steps)\nAccuracy: {overall_acc:.2f}%")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
        
        report_overall = classification_report(
            all_true_flat, all_pred_flat, 
            labels=range(len(le.classes_)), 
            target_names=le.classes_, 
            output_dict=True, 
            zero_division=0
        )
        pd.DataFrame(report_overall).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))
        
        print(f"\nSaved confusion matrices per timestep to {metrics_dir}")
        print(f"Saved accuracy plots to {plot_dir}")
        print(f"Saved overall confusion matrix and report to {out_dir}")
    else:
        print("No test set sequences could be extracted.")

if __name__ == "__main__":
    main()
