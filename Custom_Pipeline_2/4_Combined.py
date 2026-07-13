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
print("  6_1_2. Combined Multimodal (EMG + IMU) Pipeline - Attention Comparison")
print("=" * 60)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
EMG_FS          = 1926.0
IMU_FS          = 74.0741

WINDOW_SIZE_MS  = 200   
STEP_SIZE_MS    = 50
SEQUENCE_LENGTH = 5
L2_REG          = 0.002

WINDOW_S = WINDOW_SIZE_MS / 1000.0
STEP_S   = STEP_SIZE_MS / 1000.0

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


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def append_delta_features(feat_seq):
    import numpy as np
    deltas = np.zeros_like(feat_seq)
    if len(feat_seq) > 1:
        deltas[1:] = feat_seq[1:] - feat_seq[:-1]
    delta2 = np.zeros_like(feat_seq)
    if len(deltas) > 1:
        delta2[1:] = deltas[1:] - deltas[:-1]
    return np.hstack([feat_seq, deltas, delta2])

def aryule(x, order):
    import numpy as np
    import scipy.linalg
    r = np.correlate(x, x, mode='full')
    r = r[len(x)-1 : len(x)+order]
    if r[0] == 0:
        return np.zeros(order)
    R = scipy.linalg.toeplitz(r[:-1])
    try:
        a = scipy.linalg.solve(R, -r[1:])
        return a
    except:
        return np.zeros(order)

def extract_time_domain_features(x_window, fs, is_emg=True):
    import numpy as np
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    
    mav = np.mean(np.abs(x), axis=0)
    rms = np.sqrt(np.mean(x**2, axis=0))
    var = np.var(x, axis=0)
    
    if len(x) > 1:
        wl = np.sum(np.abs(np.diff(x, axis=0)), axis=0)
    else:
        wl = np.zeros(num_channels)
        
    if len(x) > 2:
        ssc = np.sum(np.diff(np.sign(np.diff(x, axis=0)), axis=0) != 0, axis=0)
    else:
        ssc = np.zeros(num_channels)
        
    if len(x) > 1:
        zc = np.sum(np.diff(np.sign(x - np.mean(x, axis=0)), axis=0) != 0, axis=0)
    else:
        zc = np.zeros(num_channels)

    if is_emg:
        if len(x) > 1:
            mac = np.mean(np.abs(np.diff(x, axis=0)), axis=0)
        else:
            mac = np.zeros(num_channels)
        feat_stack = np.column_stack([mav, wl, rms, ssc, zc, var, mac])
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        energy = np.sum(x**2, axis=0)
        if len(x) > 1:
            auc = np.trapezoid(np.abs(x), axis=0)
            grad_rms = np.sqrt(np.mean(np.diff(x, axis=0)**2, axis=0))
        else:
            auc = np.zeros(num_channels)
            grad_rms = np.zeros(num_channels)
        feat_stack = np.column_stack([mav, wl, rms, ssc, zc, var, mean, std, auc, energy, grad_rms])
        
    return feat_stack.flatten().astype(np.float32)

_MFCC_CACHE = {}

def _mfcc_vectorized(x_2d, fs, n_mfcc=4):
    import numpy as np
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

def extract_freq_domain_features(x_window, fs, is_emg=True):
    import numpy as np
    from scipy.signal import welch
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    mnf = np.zeros(num_channels)
    mdf = np.zeros(num_channels)
    pkf = np.zeros(num_channels)
    mnp = np.zeros(num_channels)
    msr = np.zeros(num_channels)
    if len(x) > 1:
        for ch in range(num_channels):
            f, Pxx = welch(x[:, ch], fs=fs, nperseg=min(len(x), 256))
            total_power = np.sum(Pxx)
            if total_power > 0:
                mnf[ch] = np.sum(f * Pxx) / total_power
                cum_Pxx = np.cumsum(Pxx)
                mdf[ch] = f[np.where(cum_Pxx >= total_power / 2)[0][0]]
                pkf[ch] = f[np.argmax(Pxx)]
                mnp[ch] = np.mean(Pxx)
            msr[ch] = np.mean(np.sqrt(np.abs(x[:, ch])))
            
    if is_emg:
        if len(x) > 1:
            mfcc1, mfcc3 = _mfcc_vectorized(x, fs)
        else:
            mfcc1 = np.zeros(num_channels)
            mfcc3 = np.zeros(num_channels)
        feat_stack = np.column_stack([mnf, mdf, pkf, mnp, msr, mfcc1, mfcc3])
    else:
        feat_stack = np.column_stack([mnf, mdf, pkf, mnp, msr])
        
    return feat_stack.flatten().astype(np.float32)

def extract_nonlinear_features(x_window, fs):
    import numpy as np
    import pywt
    from scipy.stats import skew, kurtosis
    x = x_window.astype(np.float64)
    num_channels = x.shape[1]
    if len(x) == 0:
        return np.zeros(num_channels * 11, dtype=np.float32)
    sd = np.std(x, axis=0)
    skew_val = skew(x, axis=0, nan_policy='omit')
    skew_val = np.nan_to_num(skew_val)
    kurt_val = kurtosis(x, axis=0, nan_policy='omit')
    kurt_val = np.nan_to_num(kurt_val)
    hjp_mob = np.zeros(num_channels)
    hjp_comp = np.zeros(num_channels)
    ar_coeffs = np.zeros((num_channels, 4))
    dwt_cA_energy = np.zeros(num_channels)
    dwt_cD_energy = np.zeros(num_channels)
    if len(x) > 4:
        for ch in range(num_channels):
            xc = x[:, ch]
            std_x = np.std(xc)
            dx = np.diff(xc)
            std_dx = np.std(dx)
            ddx = np.diff(dx)
            std_ddx = np.std(ddx)
            if std_x > 0 and std_dx > 0:
                mob_x = std_dx / std_x
                hjp_mob[ch] = mob_x
                mob_dx = std_ddx / std_dx
                if mob_x > 0:
                    hjp_comp[ch] = mob_dx / mob_x
            ar = aryule(xc, 4)
            ar_coeffs[ch, :] = ar
            try:
                cA, cD = pywt.dwt(xc, 'db4')
                dwt_cA_energy[ch] = np.sum(cA**2)
                dwt_cD_energy[ch] = np.sum(cD**2)
            except:
                pass
    feat_stack = np.column_stack([sd, hjp_mob, hjp_comp, ar_coeffs, dwt_cA_energy, dwt_cD_energy, skew_val, kurt_val])
    return feat_stack.flatten().astype(np.float32)

def extract_emg_features(x_window, fs):
    import numpy as np
    t_feat = extract_time_domain_features(x_window, fs, is_emg=True)
    f_feat = extract_freq_domain_features(x_window, fs, is_emg=True)
    nl_feat = extract_nonlinear_features(x_window, fs)
    return np.concatenate([t_feat, f_feat, nl_feat])

def extract_imu_features(x_window, fs):
    import numpy as np
    t_feat = extract_time_domain_features(x_window, fs, is_emg=False)
    f_feat = extract_freq_domain_features(x_window, fs, is_emg=False)
    nl_feat = extract_nonlinear_features(x_window, fs)
    return np.concatenate([t_feat, f_feat, nl_feat])

def load_and_extract_multimodal_features(base_dir):
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
                emg_feat_seq = []
                imu_feat_seq = []
                time_seq = []
                
                t_end = NOISE_START + WINDOW_S
                while t_end <= MOVE_END:
                    t_start = t_end - WINDOW_S
                    
                    emg_mask = (emg_time_vals >= t_start) & (emg_time_vals <= t_end)
                    emg_win = emg_data_opt[emg_mask]
                    emg_feat = extract_emg_features(emg_win, EMG_FS)
                    
                    imu_mask = (imu_time_vals >= t_start) & (imu_time_vals <= t_end)
                    imu_win = imu_data[imu_mask]
                    imu_feat = extract_imu_features(imu_win, IMU_FS)
                    
                    emg_feat_seq.append(emg_feat)
                    imu_feat_seq.append(imu_feat)
                    time_seq.append(t_end)
                    
                    t_end += STEP_S
                
                emg_feat_seq = np.array(emg_feat_seq)
                imu_feat_seq = np.array(imu_feat_seq)
                
                # Baseline correction for EMG features (mean over NOISE_START to NOISE_END)
                noise_indices = [i for i, t in enumerate(time_seq) if t <= NOISE_END]
                if len(noise_indices) > 0:
                    baseline_emg_mean = np.mean(emg_feat_seq[noise_indices], axis=0, keepdims=True)
                    emg_feat_seq = emg_feat_seq - baseline_emg_mean
                
                # Add delta features back for EMG and IMU!
                emg_feat_seq = append_delta_features(emg_feat_seq)
                imu_feat_seq = append_delta_features(imu_feat_seq)
                
                # Concatenate features
                multimodal_feat_seq = np.hstack([emg_feat_seq, imu_feat_seq])
                
                dataset[cls_label].append((multimodal_feat_seq, time_seq))
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    return dataset

def assign_phase_label(t, true_class):
    if t <= NOISE_END:
        return REST_LABEL
    else:
        return true_class

def build_sequences(raw_trials, scaler, pca=None, restrict_to_emd=False):
    X_all, Y_raw, times_all = [], [], []
    for feat_seq, time_seq, cls_name in raw_trials:
        feat_seq = scaler.transform(feat_seq)
        if pca is not None:
            feat_seq = pca.transform(feat_seq)
        
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
            times_all.append(t)
                
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_all), np.array(Y_raw), np.array(times_all)

# ============================================================================
# ARCHITECTURES TO COMPARE
# ============================================================================

def create_cnn_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    # The exact same model but without the attention block (attn_out, Add, LayerNormalization)
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("Starting attention vs no-attention comparison...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(script_dir), "extracted_trials_shifted")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", timestamp + "_4_Combined")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from: {base_dir}")
    print(f"Results will be saved to: {out_dir}")
    
    dataset = load_and_extract_multimodal_features(base_dir)
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
        
        train_ds[cls_name] = [(arr_list[i][0], arr_list[i][1], cls_name) for i in train_indices]
        val_ds[cls_name]   = [(arr_list[i][0], arr_list[i][1], cls_name) for i in val_indices]
        test_ds[cls_name]  = [(arr_list[i][0], arr_list[i][1], cls_name) for i in test_indices]

    for run_mode in ["4_Combined"]:
        print(f"\n" + "="*50)
        print(f"   RUNNING PIPELINE: {run_mode}")
        print("="*50)
        
        current_out_dir = os.path.join(out_dir, run_mode)
        os.makedirs(current_out_dir, exist_ok=True)
        
        train_raw = [(item[0], item[1], item[2]) for sublist in train_ds.values() for item in sublist]
        val_raw = [(item[0], item[1], item[2]) for sublist in val_ds.values() for item in sublist]
        test_raw_dict = {cls_name: [(item[0], item[1], item[2]) for item in sublist] for cls_name, sublist in test_ds.items()}
    
        print("\nPreparing Sequences and Scaling...")
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        pca = PCA(n_components=100) # Reduce to top 100 features
        
        all_train_feats = np.vstack([t[0] for t in train_raw]) if train_raw else np.array([])
        if len(all_train_feats) > 0:
            scaler.fit(all_train_feats)
            pca.fit(scaler.transform(all_train_feats))
            
        X_train, Y_train_raw, _ = build_sequences(train_raw, scaler, pca, restrict_to_emd=False)
        X_val, Y_val_raw, _ = build_sequences(val_raw, scaler, pca, restrict_to_emd=False)
        
        Y_train_enc = le.transform(Y_train_raw)
        Y_val_enc   = le.transform(Y_val_raw)
        
        Y_train_seq = np.repeat(Y_train_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
        Y_val_seq   = np.repeat(Y_val_enc[:, np.newaxis], SEQUENCE_LENGTH, axis=1)
        
        X_full = np.concatenate([X_train, X_val])
        Y_full_seq = np.concatenate([Y_train_seq, Y_val_seq])
        
        input_shape = (X_full.shape[1], X_full.shape[2])
        num_classes = len(all_classes)
        
        print(f"Training shapes - X: {X_full.shape}, Y_seq: {Y_full_seq.shape}")
        
        models_to_test = {
            "CNN-LSTM": create_cnn_lstm(input_shape, num_classes)
        }
        
        trained_models = {}
        for m_name, model in models_to_test.items():
            print(f"\nTraining {m_name}...")
            tf.keras.backend.clear_session()
            model.fit(X_full, Y_full_seq, epochs=20, batch_size=8, verbose=0)
            trained_models[m_name] = model
            print("  Training finished.")
                
        # ========================================================================
        # EVALUATION
        # ========================================================================
        print("\nEvaluating both models on Test Set...")
        
        model_overall_accs = {m: [] for m in trained_models.keys()}
        model_preds_trunc = {}
        model_true_trunc = {}
        global_times = None
        
        for m_name, final_model in trained_models.items():
            global_true_move = []
            global_preds = []
            global_pred_times = None
            
            for cls_name, trials in test_raw_dict.items():
                cls_idx = le.transform([cls_name])[0]
                
                for t_idx, (feat_seq, time_seq, _) in enumerate(trials):
                    feat_seq_scaled = scaler.transform(feat_seq)
                    feat_seq_scaled = pca.transform(feat_seq_scaled)
                    
                    X_trial = []
                    for i in range(len(feat_seq_scaled)):
                        seq = feat_seq_scaled[max(0, i - SEQUENCE_LENGTH + 1) : i + 1]
                        if len(seq) < SEQUENCE_LENGTH:
                            pad = [np.zeros_like(seq[0])] * (SEQUENCE_LENGTH - len(seq))
                            seq = pad + list(seq)
                        X_trial.append(seq)
                    X_trial = np.array(X_trial)
                    
                    preds = final_model.predict(X_trial, verbose=0)
                    pred_classes = np.argmax(preds[:, -1, :], axis=-1)
                    
                    true_labels_move = np.full(len(time_seq), cls_idx)
                    
                    global_true_move.append(true_labels_move)
                    global_preds.append(pred_classes)
                    
                    if global_pred_times is None:
                        global_pred_times = time_seq
                        
            if global_preds:
                global_min_len = min(len(p) for p in global_preds)
                model_preds_trunc[m_name] = np.array([p[:global_min_len] for p in global_preds])
                model_true_trunc[m_name] = np.array([t[:global_min_len] for t in global_true_move])
                
                if global_times is None:
                    global_times = global_pred_times[:global_min_len]
                
                overall_move_accuracies = []
                for t_i in range(global_min_len):
                    y_true_move_t = model_true_trunc[m_name][:, t_i]
                    y_pred_t = model_preds_trunc[m_name][:, t_i]
                    move_acc_t = accuracy_score(y_true_move_t, y_pred_t) * 100.0
                    overall_move_accuracies.append(move_acc_t)
                    
                model_overall_accs[m_name] = overall_move_accuracies
                
        # Generate summary text comparing BOTH models
        summary_text = (
            f"{'='*60}\n"
            f"MULTIMODAL CLASSIFICATION ACCURACY COMPARISON\n"
            f"Run Mode: {run_mode}\n"
            f"Trained on FULL time window\n"
            f"{'='*60}\n"
        )
        
        noise_mask = np.array(global_times) <= NOISE_END
        emd_mask = (np.array(global_times) > NOISE_END) & (np.array(global_times) <= EMD_END)
        move_mask = np.array(global_times) > EMD_END
        
        for m_name in trained_models.keys():
            accs = model_overall_accs[m_name]
            avg_overall = np.mean(accs)
            avg_noise = np.mean([accs[i] for i in range(len(accs)) if noise_mask[i]]) if noise_mask.any() else 0
            avg_emd = np.mean([accs[i] for i in range(len(accs)) if emd_mask[i]]) if emd_mask.any() else 0
            avg_move = np.mean([accs[i] for i in range(len(accs)) if move_mask[i]]) if move_mask.any() else 0
            
            summary_text += (
                f"Model: {m_name}\n"
                f"  Overall Average Accuracy:          {avg_overall:.2f}%\n"
                f"    Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
                f"    EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
                f"    Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
                f"{'-'*40}\n"
            )
        print(f"\n{summary_text}")
        with open(os.path.join(current_out_dir, "accuracy_summary.txt"), 'w') as f:
            f.write(summary_text)

        # Plot combined line chart comparing accuracy over time
        plt.figure(figsize=(12, 6))
        colors = {'CNN-LSTM-Attn': 'purple', 'CNN-LSTM': 'orange'}
        for m_name, accs in model_overall_accs.items():
            plt.plot(global_times, accs, marker='o', linewidth=2, label=m_name, color=colors.get(m_name, 'blue'), alpha=0.8)
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
        plt.title(f'Overall Accuracy Comparison Over Time ({run_mode}) - (EMG+IMU)')
        plt.xlabel('Time Relative to Onset (s)')
        plt.ylabel('Accuracy (%)')
        plt.xlim(NOISE_START, MOVE_END)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(current_out_dir, "Comparison_Accuracy_Over_Time.png"), dpi=150)
        plt.close()

        for m_name, final_model in trained_models.items():
            # EMD Phase Confusion Matrix
            emd_indices = [i for i, t in enumerate(global_times) if EMD_START < t <= EMD_END]
            if emd_indices and model_true_trunc[m_name] is not None:
                emd_true = model_true_trunc[m_name][:, emd_indices].flatten()
                emd_pred = model_preds_trunc[m_name][:, emd_indices].flatten()
                cm_emd = confusion_matrix(emd_true, emd_pred, labels=range(len(le.classes_)))
                emd_acc = accuracy_score(emd_true, emd_pred) * 100
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_emd, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
                plt.title(f"EMD Phase Confusion Matrix (-0.6s to 0.0s)\nModel: {m_name} | Accuracy: {emd_acc:.2f}%")
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(os.path.join(current_out_dir, f"confusion_matrix_EMD_only_{m_name}.png"), dpi=150)
                plt.close()
            
            # Plot individual Accuracy bar chart
            plt.figure(figsize=(12, 6))
            m_accs = model_overall_accs[m_name]
            plt.bar(global_times, m_accs, width=STEP_S*0.8, color='purple' if 'Attn' in m_name else 'orange', edgecolor='black', label=f'{m_name}', alpha=0.8)
            plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
            plt.title(f'Overall Accuracy Over Time - {m_name} ({run_mode})')
            plt.xlabel('Time Relative to Onset (s)')
            plt.ylabel('Accuracy (%)')
            plt.xlim(NOISE_START, MOVE_END)
            plt.ylim(-5, 105)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(current_out_dir, f"Accuracy_Over_Time_{m_name}.png"), dpi=150)
            plt.close()
            
            # Per-Class Accuracies Over Time
            class_accuracies_over_time = {cls: [] for cls in movement_classes}
            for t_i in range(len(global_times)):
                y_true_move_t = model_true_trunc[m_name][:, t_i]
                y_pred_t = model_preds_trunc[m_name][:, t_i]
                
                for cls in movement_classes:
                    cls_idx = le.transform([cls])[0]
                    cls_mask = (y_true_move_t == cls_idx)
                    if np.sum(cls_mask) > 0:
                        acc = accuracy_score(y_true_move_t[cls_mask], y_pred_t[cls_mask]) * 100.0
                        class_accuracies_over_time[cls].append(acc)
                    else:
                        class_accuracies_over_time[cls].append(np.nan)
                        
            # Plot Per-Class Accuracy Over Time (Individual Bar Charts)
            class_plot_dir = os.path.join(current_out_dir, "Per_Class_BarCharts", m_name)
            os.makedirs(class_plot_dir, exist_ok=True)
            
            for cls in movement_classes:
                cls_accs = class_accuracies_over_time[cls]
                plt.figure(figsize=(12, 6))
                plt.bar(global_times, cls_accs, width=STEP_S*0.8, color='teal', alpha=0.8, edgecolor='black', label=f'{cls} Accuracy')
                plt.axvline(x=0.0, color='k', linestyle='--', linewidth=2, label='Movement Onset')
                plt.title(f'Accuracy Over Time for Class: {cls} - {m_name} ({run_mode})')
                plt.xlabel('Time Relative to Onset (s)')
                plt.ylabel('Accuracy (%)')
                plt.xlim(NOISE_START, MOVE_END)
                plt.ylim(-5, 105)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(class_plot_dir, f"Accuracy_Over_Time_{cls}.png"), dpi=150)
                plt.close()

            # Save the model
            final_model.save(os.path.join(current_out_dir, f"final_model_{m_name}.h5"))
            
            # Save timestep metrics to CSV
            metrics_dir = os.path.join(current_out_dir, f"Timestep_Metrics_{m_name}")
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_records = []
            from sklearn.metrics import precision_recall_fscore_support
            
            for t_i, t_val in enumerate(global_times):
                y_true_t = model_true_trunc[m_name][:, t_i]
                y_pred_t = model_preds_trunc[m_name][:, t_i]
                acc_t = accuracy_score(y_true_t, y_pred_t) * 100
                precision, recall, f1, _ = precision_recall_fscore_support(y_true_t, y_pred_t, labels=range(len(le.classes_)), average='macro', zero_division=0)
                
                metrics_records.append({
                    'Time (s)': t_val,
                    'Accuracy (%)': acc_t,
                    'Macro Precision': precision,
                    'Macro Recall': recall,
                    'Macro F1': f1
                })
                
            df_metrics = pd.DataFrame(metrics_records)
            df_metrics.to_csv(os.path.join(metrics_dir, "metrics_over_time.csv"), index=False)

if __name__ == "__main__":
    main()
