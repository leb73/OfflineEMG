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
from scipy.signal import butter, sosfilt, welch, iirnotch, tf2sos

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Make runs deterministic
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("starting — 4_5_1 Model Comparison (Dedicated EMD)")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
EMG_FS          = 1926.0
WINDOW_SIZE_MS  = 100
STEP_SIZE_MS    = 25
SEQUENCE_LENGTH = 5
L2_REG = 0.002

NOISE_START  = -1.5
NOISE_END    = -0.6
EMD_START    = -0.6
EMD_END      =  0.0
MOVE_END     =  1.5

REST_LABEL   = "rest"

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
            sos_lp = butter(2, max(min(20.0, nyq - 5.0), 5.0), btype='low', fs=fs, output='sos')
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
    return (window_end_sample / fs) + trial_offset

def append_delta_features(feat_seq):
    deltas = np.zeros_like(feat_seq)
    if len(feat_seq) > 1:
        deltas[1:] = feat_seq[1:] - feat_seq[:-1]
    delta2 = np.zeros_like(feat_seq)
    if len(deltas) > 1:
        delta2[1:] = deltas[1:] - deltas[:-1]
    return np.hstack([feat_seq, deltas, delta2])

def extract_raw_features(ds_dict, window_samples, step_samples, channel_types):
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
            
            noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
            if len(noise_indices) > 0:
                baseline_mean = np.mean(feat_seq[noise_indices], axis=0, keepdims=True)
                feat_seq = feat_seq - baseline_mean
            
            feat_seq = append_delta_features(feat_seq)
            raw_trials.append((feat_seq, time_seq, cls_name))
    return raw_trials

def assign_phase_label(t, true_class):
    if t <= NOISE_END:
        return REST_LABEL
    else:
        return true_class

def build_sequences(raw_trials, scaler, restrict_to_emd=False):
    X_all, Y_raw, groups, times_all = [], [], [], []
    t_idx = 0
    for feat_seq, time_seq, cls_name in raw_trials:
        feat_seq = scaler.transform(feat_seq)
        
        for i in range(len(feat_seq)):
            t = time_seq[i]
            if restrict_to_emd and t > EMD_END:
                continue
                
            seq = feat_seq[max(0, i - SEQUENCE_LENGTH + 1) : i + 1]
            if len(seq) < SEQUENCE_LENGTH:
                pad = [np.zeros_like(seq[0])] * (SEQUENCE_LENGTH - len(seq))
                seq = pad + list(seq)
            X_all.append(seq)
            Y_raw.append(assign_phase_label(t, cls_name))
            groups.append(t_idx)
            times_all.append(t)
                
        t_idx += 1
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    return np.array(X_all), np.array(Y_raw), np.array(groups), np.array(times_all)

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

def create_simple_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # No sequence flattening needed, predict for every timestep
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_simple_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_tcn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    for dilation_rate in [1, 2]:
        x = layers.Conv1D(32, 3, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
        x = layers.BatchNormalization()(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transformer(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mlp(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class SklearnWrapper:
    """Wrapper to make sklearn models behave like the Keras models for sequential eval."""
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, Y):
        # Flatten the (batch, seq, features) to (batch, seq * features)
        # We only predict the target for the LAST timestep in the sequence, 
        # so Y should be (batch,) instead of (batch, seq).
        # We will extract the last label.
        X_flat = X.reshape(X.shape[0], -1)
        Y_last = Y[:, -1] if len(Y.shape) > 1 else Y
        self.model.fit(X_flat, Y_last)
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        # Returns shape (batch,)
        return self.model.predict(X_flat)

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "extracted_trials")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", timestamp + "_Model_Comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from: {base_dir}")
    print(f"Results will be saved to: {out_dir}")
    print(f"\n--- MODEL COMPARISON (DEDICATED EMD) ---")
    
    dataset = collections.defaultdict(list)
    channel_types = None
    
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_*_Short")))
    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        parts = folder_name.split("_")
        if len(parts) >= 3:
            cls_label = parts[2].lower()
            if cls_label in ['hpen', 'vpen']:
                continue
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
                
                time_col = [c for c in df.columns if 'Time' in c][0]
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                df = df[(df[time_col] >= NOISE_START) & (df[time_col] <= MOVE_END)].reset_index(drop=True)
                
                emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
                if channel_types is None:
                    channel_types = [0] * len(emg_cols)
                    
                df_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
                raw_data = df_data.values
                
                filt_data = apply_filters(raw_data, EMG_FS, channel_types)
                dataset[cls_label].append(filt_data)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    if not dataset: return
        
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
        
        train_ds[cls_name] = [arr_list[i] for i in train_indices]
        val_ds[cls_name]   = [arr_list[i] for i in val_indices]
        test_ds[cls_name]  = [arr_list[i] for i in test_indices]

    window_samples = int((WINDOW_SIZE_MS / 1000) * EMG_FS)
    step_samples   = int((STEP_SIZE_MS / 1000) * EMG_FS)
    
    print("\nExtracting Features...")
    train_raw = extract_raw_features(train_ds, window_samples, step_samples, channel_types)
    val_raw = extract_raw_features(val_ds, window_samples, step_samples, channel_types)
    
    scaler = StandardScaler()
    all_train_feats = np.vstack([t[0] for t in train_raw]) if train_raw else np.array([])
    if len(all_train_feats) > 0:
        scaler.fit(all_train_feats)
        
    X_train, Y_train_raw, _, _ = build_sequences(train_raw, scaler, restrict_to_emd=True)
    X_val, Y_val_raw, _, _ = build_sequences(val_raw, scaler, restrict_to_emd=True)
    
    Y_train_enc = le.transform(Y_train_raw)
    Y_val_enc   = le.transform(Y_val_raw)
    
    Y_train_seq = np.repeat(Y_train_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
    Y_val_seq   = np.repeat(Y_val_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
    
    X_full = np.concatenate([X_train, X_val])
    Y_full_seq = np.concatenate([Y_train_seq, Y_val_seq])
    Y_full_flat = np.concatenate([Y_train_enc, Y_val_enc])
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(all_classes)
    
    print(f"Training shapes - X: {X_full.shape}, Y_seq: {Y_full_seq.shape}")
    
    models_to_test = {
        "CNN-LSTM-Attn": create_cnn_lstm_attn(input_shape, num_classes),
        "Simple CNN": create_simple_cnn(input_shape, num_classes),
        "Simple LSTM": create_simple_lstm(input_shape, num_classes),
        "TCN": create_tcn(input_shape, num_classes),
        "Transformer": create_transformer(input_shape, num_classes),
        "MLP (FCN)": create_mlp(input_shape, num_classes),
        "Random Forest": SklearnWrapper(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)),
        "XGBoost (Hist)": SklearnWrapper(HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)),
        "Linear SVC": SklearnWrapper(LinearSVC(random_state=42, max_iter=1000)),
        "Logistic Reg": SklearnWrapper(LogisticRegression(random_state=42, max_iter=1000))
    }
    
    trained_models = {}
    
    for m_name, model in models_to_test.items():
        print(f"\nTraining {m_name}...")
        if isinstance(model, SklearnWrapper):
            model.fit(X_full, Y_full_flat)
            trained_models[m_name] = model
        else:
            tf.keras.backend.clear_session()
            model.fit(X_full, Y_full_seq, epochs=20, batch_size=8, verbose=0)
            trained_models[m_name] = model
            print("  Training finished.")
            
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\nEvaluating all models on Test Set...")
    
    # Store results: model_name -> list of overall movement accuracies over time
    model_overall_accs = {m: [] for m in trained_models.keys()}
    global_times = None
    
    for m_name, final_model in trained_models.items():
        global_true_move = []
        global_preds = []
        global_pred_times = None
        
        for cls_name, trials in test_ds.items():
            cls_idx = le.transform([cls_name])[0]
            rest_idx = le.transform([REST_LABEL])[0]
            
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
                
                noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
                if len(noise_indices) > 0:
                    baseline_mean = np.mean(feat_seq[noise_indices], axis=0, keepdims=True)
                    feat_seq = feat_seq - baseline_mean
                
                feat_seq = append_delta_features(feat_seq)
                feat_seq = scaler.transform(feat_seq)
                    
                X_trial = []
                for i in range(len(feat_seq)):
                    seq = feat_seq[max(0, i - SEQUENCE_LENGTH + 1) : i + 1]
                    if len(seq) < SEQUENCE_LENGTH:
                        pad = [np.zeros_like(seq[0])] * (SEQUENCE_LENGTH - len(seq))
                        seq = pad + list(seq)
                    X_trial.append(seq)
                X_trial = np.array(X_trial)
                
                if isinstance(final_model, SklearnWrapper):
                    pred_classes = final_model.predict(X_trial)
                else:
                    preds = final_model.predict(X_trial, verbose=0)
                    pred_classes = np.argmax(preds[:, -1, :], axis=-1)
                
                true_labels_move = np.full(len(time_seq), cls_idx)
                
                global_true_move.append(true_labels_move)
                global_preds.append(pred_classes)
                
                if global_pred_times is None:
                    global_pred_times = time_seq
                    
        if global_preds:
            global_min_len = min(len(p) for p in global_preds)
            global_preds_trunc = np.array([p[:global_min_len] for p in global_preds])
            global_true_move_trunc = np.array([t[:global_min_len] for t in global_true_move])
            
            if global_times is None:
                global_times = global_pred_times[:global_min_len]
            
            overall_move_accuracies = []
            
            for t_i in range(global_min_len):
                y_true_move_t = global_true_move_trunc[:, t_i]
                y_pred_t = global_preds_trunc[:, t_i]
                move_acc_t = accuracy_score(y_true_move_t, y_pred_t) * 100.0
                overall_move_accuracies.append(move_acc_t)
                
            model_overall_accs[m_name] = overall_move_accuracies

    # ========================================================================
    # PLOTTING AND SUMMARIZING
    # ========================================================================
    plt.figure(figsize=(14, 8))
    cmap = plt.colormaps['tab20']
    
    summary_text = (
        f"{'='*60}\n"
        f"MODEL COMPARISON RESULTS (EMD DEDICATED)\n"
        f"Trained only on <= 0.0s, Tested on 4 classes\n"
        f"{'='*60}\n"
    )
    
    for i, (m_name, accs) in enumerate(model_overall_accs.items()):
        if not accs: continue
        plt.plot(global_times, accs, marker='.', markersize=3, linewidth=2, 
                 color=cmap(i), label=f'{m_name}', alpha=0.85)
        
        emd_mask = (np.array(global_times) > NOISE_END) & (np.array(global_times) <= EMD_END)
        avg_emd = np.mean([accs[i] for i in range(len(accs)) if emd_mask[i]]) if emd_mask.any() else 0
        summary_text += f"{m_name:>20s} | EMD Phase Acc: {avg_emd:.2f}%\n"

    plt.axvline(x=NOISE_END, color='orange', linestyle='--', linewidth=1.5, label=f'Noise→EMD ({NOISE_END}s)')
    plt.axvline(x=0.0, color='r', linestyle='--', linewidth=1.5, label='Movement Onset')
    chance_level = 100.0 / 4.0
    plt.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.5, label=f'Chance ({chance_level:.0f}%)')
    
    plt.title('Architecture Comparison: Accuracy Over Time (4 Classes)')
    plt.xlabel('Time Relative to Onset (s)')
    plt.ylabel('Movement Classification Accuracy (%)')
    plt.xlim(NOISE_START, MOVE_END)
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Model_Comparison_Accuracy.png"), dpi=150)
    plt.close()
    
    print(f"\n{summary_text}")
    with open(os.path.join(out_dir, "comparison_summary.txt"), 'w') as f:
        f.write(summary_text)
        
    print(f"\nComparison complete. Check plots in: {out_dir}")

if __name__ == "__main__":
    main()
