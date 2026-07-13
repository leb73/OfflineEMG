import os
import sys
import glob
import time
import json
import collections
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, iirnotch, tf2sos

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
from sklearn.metrics import accuracy_score, confusion_matrix

# Make runs deterministic
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("  7_data_optimisation.py - Data Processing Parameters Grid Search")
print("=" * 60)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
EMG_FS          = 1926.0
IMU_FS          = 74.0741

window_sIZE_MS  = 200   
step_sIZE_MS    = 50
SEQUENCE_LENGTH = 5
L2_REG          = 0.002

window_s = window_sIZE_MS / 1000.0
step_s   = step_sIZE_MS / 1000.0

# Phase boundaries (seconds relative to movement onset at t=0)
NOISE_START  = -1.5
NOISE_END    = -0.6
EMD_START    = -0.6
EMD_END      =  0.0
MOVE_END     =  1.5

REST_LABEL   = "rest"

# ============================================================================
# FILTER SETUP
# ============================================================================
def build_emg_filters(fs, num_channels, channel_types):
    filters = []
    nyq = fs / 2.0
    bp_high = min(400.0, nyq - 5.0)
    for i in range(num_channels):
        ctype = channel_types[i] if channel_types else 0
        if nyq > 55.0:
            b, a = iirnotch(50.0, 30.0, fs)
            sos_notch = tf2sos(b, a)
        else:
            sos_notch = None
        sos_bp = butter(4, [40.0, bp_high], btype='band', fs=fs, output='sos')
        filters.append({'sos_n': sos_notch, 'sos_b': sos_bp})
    return filters

def apply_emg_filters(data, fs, channel_types):
    num_channels = data.shape[1]
    filters = build_emg_filters(fs, num_channels, channel_types)
    filt_data = np.zeros_like(data)
    for i in range(num_channels):
        f_cfg = filters[i]
        x = data[:, i]
        if f_cfg['sos_n'] is not None:
            x = sosfilt(f_cfg['sos_n'], x)
        x = sosfilt(f_cfg['sos_b'], x)
        filt_data[:, i] = x
    return filt_data

def lowpass_filter_imu(data: np.ndarray, cutoff: float = 20.0, fs: float = IMU_FS, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(order, min(cutoff, nyq - 1.0), btype='low', fs=fs, output='sos')
    out = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        out[:, ch] = sosfilt(sos, data[:, ch])
    return out

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

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_emg_features(x_window, include_mfcc=True):
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    
    if len(x) == 0:
        return np.zeros(num_channels * (6 if include_mfcc else 4), dtype=np.float32)
        
    rms_val = np.sqrt(np.mean(x ** 2, axis=0))
    var_val = np.var(x, axis=0)
    
    if x.shape[0] > 1:
        wl_val = np.sum(np.abs(np.diff(x, axis=0)), axis=0)
        mac_val = np.mean(np.abs(np.diff(x, axis=0)), axis=0)
    else:
        wl_val = np.zeros(num_channels)
        mac_val = np.zeros(num_channels)

    if include_mfcc:
        if x.shape[0] > 1:
            mfcc1_val, mfcc3_val = _mfcc_vectorized(x, EMG_FS)
        else:
            mfcc1_val = np.zeros(num_channels)
            mfcc3_val = np.zeros(num_channels)
        feat_stack = np.column_stack([rms_val, var_val, wl_val, mac_val, mfcc1_val, mfcc3_val])
    else:
        feat_stack = np.column_stack([rms_val, var_val, wl_val, mac_val])

    return feat_stack.flatten().astype(np.float32)

def extract_imu_features(x_window):
    if len(x_window) == 0:
        return np.zeros(x_window.shape[1] * 8, dtype=np.float32)
        
    feats = []
    for ch in range(x_window.shape[1]):
        x = x_window[:, ch].astype(np.float64)
        mean     = np.mean(x)
        std      = np.std(x)
        rms      = np.sqrt(np.mean(x ** 2))
        p2p      = np.ptp(x) if len(x) > 0 else 0
        auc      = np.trapezoid(np.abs(x)) if len(x) > 1 else 0
        energy   = np.sum(x ** 2)
        zcr      = np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x) if len(x) > 0 else 0
        grad_rms = np.sqrt(np.mean(np.diff(x) ** 2)) if len(x) > 1 else 0.0
        feats.extend([mean, std, rms, p2p, auc, energy, zcr, grad_rms])
    return np.array(feats, dtype=np.float32)

def append_delta_features(feat_seq):
    deltas = np.zeros_like(feat_seq)
    if len(feat_seq) > 1:
        deltas[1:] = feat_seq[1:] - feat_seq[:-1]
    delta2 = np.zeros_like(feat_seq)
    if len(deltas) > 1:
        delta2[1:] = deltas[1:] - deltas[:-1]
    return np.hstack([feat_seq, deltas, delta2])

def extract_emg_custom(x_window):
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    if len(x) == 0:
        return np.zeros(num_channels * 8, dtype=np.float32)
        
    rms = np.sqrt(np.mean(x**2, axis=0))
    p2p = np.ptp(x, axis=0) if len(x) > 0 else np.zeros(num_channels)
    mav = np.mean(np.abs(x), axis=0)
    rss = np.sqrt(np.sum(x**2, axis=0))
    shape_factor = np.divide(rms, mav, out=np.zeros_like(rms), where=mav!=0)
    
    if len(x) > 1:
        zcr = np.sum(np.diff(np.sign(x - np.mean(x, axis=0)), axis=0) != 0, axis=0) / len(x)
    else:
        zcr = np.zeros(num_channels)
        
    mean_freq = np.zeros(num_channels)
    median_freq = np.zeros(num_channels)
    if len(x) > 1:
        from scipy.signal import welch
        for ch in range(num_channels):
            f, Pxx = welch(x[:, ch], fs=EMG_FS, nperseg=min(len(x), 256))
            if np.sum(Pxx) > 0:
                mean_freq[ch] = np.sum(f * Pxx) / np.sum(Pxx)
                cum_Pxx = np.cumsum(Pxx)
                median_freq[ch] = f[np.where(cum_Pxx >= cum_Pxx[-1] / 2)[0][0]]
      
    feat_stack = np.column_stack([rms, p2p, mav, rss, shape_factor, zcr, mean_freq, median_freq])
    return feat_stack.flatten().astype(np.float32)

def extract_imu_custom(x_window):
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    if len(x) == 0:
        return np.zeros(num_channels * 6, dtype=np.float32)
        
    rms = np.sqrt(np.mean(x**2, axis=0))
    p2p = np.ptp(x, axis=0) if len(x) > 0 else np.zeros(num_channels)
    mav = np.mean(np.abs(x), axis=0)
    rss = np.sqrt(np.sum(x**2, axis=0))
    shape_factor = np.divide(rms, mav, out=np.zeros_like(rms), where=mav!=0)
    
    if len(x) > 1:
        zcr = np.sum(np.diff(np.sign(x - np.mean(x, axis=0)), axis=0) != 0, axis=0) / len(x)
    else:
        zcr = np.zeros(num_channels)
        
    feat_stack = np.column_stack([rms, p2p, mav, rss, shape_factor, zcr])
    return feat_stack.flatten().astype(np.float32)

# ============================================================================
# DATA LOADING & WINDOWING
# ============================================================================
def load_and_extract_multimodal_features(base_dir, window_s, step_s):
    dataset = collections.defaultdict(list)
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_*_Short")))
    
    if not trial_folders:
        print(f"  [WARN] No folders found in: {base_dir}")
        return dataset

    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        parts = folder_name.split("_")
        if len(parts) >= 3:
            cls_label = parts[2].lower()
        else:
            continue

        for csv_path in sorted(glob.glob(os.path.join(tf_path, "movement_*.csv"))):
            if not os.path.isfile(csv_path): continue
            try:
                with open(csv_path, 'r') as f:
                    for _ in range(5): f.readline()
                    num_cols = len(f.readline().split(','))
                
                df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
                if len(df) <= 2: continue
                df = df.iloc[2:].reset_index(drop=True)
                
                # Extract EMG Time and Data
                emg_time_cols = [c for c in df.columns if 'Time' in c and 'ACC' not in c]
                if not emg_time_cols: continue
                emg_time_col = emg_time_cols[0]
                emg_time_vals = pd.to_numeric(df[emg_time_col], errors='coerce').values
                
                emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
                emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values
                emg_data = apply_emg_filters(emg_data, EMG_FS, [0]*len(emg_cols))
                
                # Extract IMU Time and Data
                imu_time_cols = [c for c in df.columns if 'ACC' in c and 'Time' in c]
                if not imu_time_cols: continue
                imu_time_col = imu_time_cols[0]
                imu_time_vals = pd.to_numeric(df[imu_time_col], errors='coerce').values
                valid_imu_idx = ~np.isnan(imu_time_vals)
                
                imu_time_vals = imu_time_vals[valid_imu_idx]
                imu_cols = [c for c in df.columns if 'ACC' in c and '(G)' in c]
                imu_data = df[imu_cols].iloc[valid_imu_idx].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
                imu_data = lowpass_filter_imu(imu_data, fs=IMU_FS)
                
                # Baseline correction for IMU (mean over NOISE_START to NOISE_END)
                base_mask = (imu_time_vals >= NOISE_START) & (imu_time_vals <= NOISE_END)
                baseline_imu = imu_data[base_mask].mean(axis=0) if base_mask.sum() > 5 else np.zeros(imu_data.shape[1])
                imu_data = imu_data - baseline_imu
                
                # Restrict to 4 main EMG channels: 1, 2, 6, 7 (indices 1, 2, 6, 7)
                emg_data_opt = emg_data[:, [1, 2, 6, 7]]
                
                # Extract multimodal features over time windows
                emg_feat_seq_all = []
                emg_feat_seq_no_mfcc = []
                emg_feat_seq_custom = []
                imu_feat_seq = []
                imu_feat_seq_custom = []
                time_seq = []
                
                t_end = NOISE_START + window_s
                while t_end <= MOVE_END:
                    t_start = t_end - window_s
                    
                    emg_mask = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    emg_win = emg_data_opt[emg_mask]
                    emg_feat_all = extract_emg_features(emg_win, include_mfcc=True)
                    emg_feat_no_mfcc = extract_emg_features(emg_win, include_mfcc=False)
                    emg_feat_custom = extract_emg_custom(emg_win)
                    
                    imu_mask = (imu_time_vals >= t_start) & (imu_time_vals <= t_end)
                    imu_win = imu_data[imu_mask]
                    imu_feat = extract_imu_features(imu_win)
                    imu_feat_custom = extract_imu_custom(imu_win)
                    
                    emg_feat_seq_all.append(emg_feat_all)
                    emg_feat_seq_no_mfcc.append(emg_feat_no_mfcc)
                    emg_feat_seq_custom.append(emg_feat_custom)
                    imu_feat_seq.append(imu_feat)
                    imu_feat_seq_custom.append(imu_feat_custom)
                    time_seq.append(t_end)
                    
                    t_end += step_s
                
                emg_feat_seq_all = np.array(emg_feat_seq_all)
                emg_feat_seq_no_mfcc = np.array(emg_feat_seq_no_mfcc)
                emg_feat_seq_custom = np.array(emg_feat_seq_custom)
                imu_feat_seq = np.array(imu_feat_seq)
                imu_feat_seq_custom = np.array(imu_feat_seq_custom)
                
                # Baseline correction for EMG features (mean over NOISE_START to NOISE_END)
                noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
                if len(noise_indices) > 0:
                    baseline_emg_mean_all = np.mean(emg_feat_seq_all[noise_indices], axis=0, keepdims=True)
                    emg_feat_seq_all = emg_feat_seq_all - baseline_emg_mean_all
                    
                    baseline_emg_mean_no = np.mean(emg_feat_seq_no_mfcc[noise_indices], axis=0, keepdims=True)
                    emg_feat_seq_no_mfcc = emg_feat_seq_no_mfcc - baseline_emg_mean_no
                    
                    baseline_emg_mean_custom = np.mean(emg_feat_seq_custom[noise_indices], axis=0, keepdims=True)
                    emg_feat_seq_custom = emg_feat_seq_custom - baseline_emg_mean_custom
                
                emg_feat_seq_all = append_delta_features(emg_feat_seq_all)
                emg_feat_seq_no_mfcc = append_delta_features(emg_feat_seq_no_mfcc)
                emg_feat_seq_custom = append_delta_features(emg_feat_seq_custom)
                
                # Concatenate EMG and IMU features!
                multimodal_feat_seq_all = np.hstack([emg_feat_seq_all, imu_feat_seq])
                multimodal_feat_seq_no_mfcc = np.hstack([emg_feat_seq_no_mfcc, imu_feat_seq])
                multimodal_feat_seq_custom = np.hstack([emg_feat_seq_custom, imu_feat_seq_custom])
                
                dataset[cls_label].append((multimodal_feat_seq_all, multimodal_feat_seq_no_mfcc, multimodal_feat_seq_custom, time_seq))
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    return dataset

def assign_phase_label(t, true_class):
    if t <= NOISE_END:
        return REST_LABEL
    else:
        return true_class

def build_sequences(raw_trials, scaler, sequence_length, restrict_to_emd=False):
    X_all, Y_raw, times_all = [], [], []
    for feat_seq, time_seq, cls_name in raw_trials:
        feat_seq = scaler.transform(feat_seq)
        
        for i in range(len(feat_seq)):
            t = time_seq[i]
            if restrict_to_emd and t > EMD_END:
                continue
                
            seq = feat_seq[max(0, i - sequence_length + 1) : i + 1]
            if len(seq) < sequence_length:
                pad = [np.zeros_like(seq[0])] * (sequence_length - len(seq))
                seq = pad + list(seq)
            X_all.append(seq)
            Y_raw.append(assign_phase_label(t, cls_name))
            times_all.append(t)
                
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_all), np.array(Y_raw), np.array(times_all)

# ============================================================================
# ARCHITECTURES TO COMPARE
# ============================================================================
def create_cnn_lstm_attn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_lstm(input_shape, num_classes, filters=64, lstm_units=128, dropout=0.2, lr=1e-4):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.Conv1D(filters, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)
    x = layers.TimeDistributed(layers.Dense(lstm_units, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("Starting data optimisation grid search...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(script_dir), "extracted_trials_shifted")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", timestamp + "_Data_Optimisation")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from: {base_dir}")
    print(f"Results will be saved to: {out_dir}")
    
    grid = {
        'window_size_ms': [150, 200, 250],
        'step_size_ms': [25, 50],
        'sequence_length': [5, 10, 15]
    }
    
    import itertools
    keys = grid.keys()
    combinations = list(itertools.product(*grid.values()))
    
    results = []
    print(f"\nStarting Grid Search. Total configurations: {len(combinations)}")
    
    for idx, values in enumerate(combinations):
        params = dict(zip(keys, values))
        window_s = params['window_size_ms'] / 1000.0
        step_s = params['step_size_ms'] / 1000.0
        seq_len = params['sequence_length']
        
        print(f"\n{'='*50}")
        print(f"--- Trial {idx+1}/{len(combinations)} ---")
        print(f"Params: {params}")
        print(f"{'='*50}")
        
        dataset = load_and_extract_multimodal_features(base_dir, window_s, step_s)
        if not dataset: continue
            
        movement_classes = sorted(list(dataset.keys()))
        all_classes = movement_classes + [REST_LABEL]
        le = LabelEncoder()
        le.fit(all_classes)
        
        min_trials = min([len(v) for v in dataset.values()])
        for cls in dataset.keys():
            dataset[cls] = random.sample(dataset[cls], min_trials)

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
            
            train_ds[cls_name] = [(arr_list[i][0], arr_list[i][1], arr_list[i][2], arr_list[i][3], cls_name) for i in train_indices]
            val_ds[cls_name]   = [(arr_list[i][0], arr_list[i][1], arr_list[i][2], arr_list[i][3], cls_name) for i in val_indices]
            test_ds[cls_name]  = [(arr_list[i][0], arr_list[i][1], arr_list[i][2], arr_list[i][3], cls_name) for i in test_indices]

        # Use All_Features for optimisation
        feat_idx = 0
        train_raw = [(item[feat_idx], item[3], item[4]) for sublist in train_ds.values() for item in sublist]
        val_raw = [(item[feat_idx], item[3], item[4]) for sublist in val_ds.values() for item in sublist]
    
        scaler = StandardScaler()
        all_train_feats = np.vstack([t[0] for t in train_raw]) if train_raw else np.array([])
        if len(all_train_feats) > 0:
            scaler.fit(all_train_feats)
            
        X_train, Y_train_raw, _ = build_sequences(train_raw, scaler, seq_len, restrict_to_emd=False)
        X_val, Y_val_raw, _ = build_sequences(val_raw, scaler, seq_len, restrict_to_emd=False)
        
        Y_train_enc = le.transform(Y_train_raw)
        Y_val_enc   = le.transform(Y_val_raw)
        
        Y_train_seq = np.repeat(Y_train_enc[:, np.newaxis], seq_len, axis=1)
        Y_val_seq   = np.repeat(Y_val_enc[:, np.newaxis], seq_len, axis=1)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(all_classes)
        
        tf.keras.backend.clear_session()
        # Fix architecture to defaults
        model = create_cnn_lstm(input_shape, num_classes, filters=64, lstm_units=128, dropout=0.2, lr=1e-4)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, Y_train_seq,
            validation_data=(X_val, Y_val_seq),
            epochs=100,
            batch_size=8,
            verbose=0,
            callbacks=[early_stop]
        )
        
        val_loss, val_acc = model.evaluate(X_val, Y_val_seq, verbose=0)
        print(f"Result -> val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
        res_dict = params.copy()
        res_dict['val_loss'] = val_loss
        res_dict['val_acc'] = val_acc
        results.append(res_dict)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "data_optimisation_results.csv"), index=False)
    print("\nOptimisation complete! Results saved.")

if __name__ == "__main__":
    main()
