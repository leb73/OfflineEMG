"""
4.1 IMU-Only Offline Calibration Pipeline
==========================================
Goal: Train a classifier on IMU (accelerometer) data from 0.0s -> 1.5s
      (post-motion-onset) and then evaluate using a sliding window across the
      full -1.5s to +1.5s timeline.

This is designed to show that:
  - IMU accuracy is at chance BEFORE motion onset (< 0.0s)
  - IMU accuracy rises sharply AFTER motion onset (> 0.0s)

Contrast with the EMG model (script 4) which shows predictive accuracy BEFORE onset.
"""
import os
import glob
import random
import collections
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier

# Make runs deterministic
np.random.seed(42)
random.seed(42)

print("Starting IMU-Only pipeline...")

# ============================================================================
# CONSTANTS
# ============================================================================
EMG_FS         = 1926.0        # Sampling rate (same sensor, IMU channels)
WINDOW_MS      = 200           # Feature extraction window size (ms)
STEP_MS        = 50            # Sliding window step (ms)
TRAIN_START_S  = 0.0           # Train on data from motion onset
TRAIN_END_S    = 1.5           # ... to 1.5s post onset
EVAL_START_S   = -1.5          # Evaluate sliding window from -1.5s
EVAL_END_S     = 1.5           # ... to +1.5s

# ============================================================================
# FILTER: Low-pass for IMU (accelerometer)
# ============================================================================
def lowpass_filter(data, cutoff=20.0, fs=EMG_FS, order=4):
    nyq = fs / 2.0
    sos = butter(order, min(cutoff, nyq - 1.0), btype='low', fs=fs, output='sos')
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = sosfilt(sos, data[:, ch])
    return out

# ============================================================================
# FEATURE EXTRACTION — IMU specific per window
# ============================================================================
def extract_imu_features(window: np.ndarray) -> np.ndarray:
    """
    Extract IMU-relevant statistical and kinematic features from a window.
    window shape: (samples, channels)
    """
    features = []
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        mean  = np.mean(x)
        std   = np.std(x)
        rms   = np.sqrt(np.mean(x**2))
        p2p   = np.ptp(x)
        auc   = np.trapezoid(np.abs(x))
        energy= np.sum(x**2)
        # Zero-crossing rate (rough freq proxy for IMU vibrations)
        zcr   = np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x)
        # Gradient (acceleration of acceleration = jerk proxy)
        grad_rms = np.sqrt(np.mean(np.diff(x)**2)) if len(x) > 1 else 0.0
        features.extend([mean, std, rms, p2p, auc, energy, zcr, grad_rms])
    return np.array(features, dtype=np.float32)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_trial_data(base_dir):
    """
    Load all IMU trial data. Returns dict: class_label -> list of (time_arr, imu_arr).
    """
    dataset = collections.defaultdict(list)
    
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_short_*")))
    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        cls_label = folder_name.split("short_")[-1]
        
        movements_dir = os.path.join(tf_path, "extracted_trials")
        if not os.path.isdir(movements_dir):
            continue
        
        for mov_dir in sorted(glob.glob(os.path.join(movements_dir, "Movement_*"))):
            csv_path = os.path.join(mov_dir, "delsys_data.csv")
            if not os.path.isfile(csv_path):
                continue
            try:
                with open(csv_path, 'r') as f:
                    for _ in range(5): f.readline()
                    num_cols = len(f.readline().split(','))
                
                df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
                if len(df) <= 2:
                    continue
                df = df.iloc[2:].reset_index(drop=True)
                
                time_col = [c for c in df.columns if 'Time' in c][0]
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                emg_time = df[time_col].values.astype(np.float64)
                
                acc_cols = [c for c in df.columns if 'ACC' in c and '(G)' in c]
                if not acc_cols:
                    print(f"  [WARN] No ACC columns in {mov_dir}, skipping.")
                    continue
                
                interpolated_acc = []
                for acc_col in acc_cols:
                    col_idx = df.columns.get_loc(acc_col)
                    acc_time_col = df.columns[col_idx - 1]
                    
                    acc_time = pd.to_numeric(df[acc_time_col], errors='coerce').values
                    acc_data = pd.to_numeric(df[acc_col], errors='coerce').values
                    
                    valid_mask = ~np.isnan(acc_time) & ~np.isnan(acc_data)
                    t_valid = acc_time[valid_mask]
                    d_valid = acc_data[valid_mask]
                    
                    if len(t_valid) < 2:
                        interp_d = np.zeros_like(emg_time)
                    else:
                        interp_d = np.interp(emg_time, t_valid, d_valid)
                    interpolated_acc.append(interp_d)
                
                imu_arr_full = np.column_stack(interpolated_acc)
                
                # Load full range
                mask = (emg_time >= EVAL_START_S) & (emg_time <= EVAL_END_S)
                time_arr = emg_time[mask]
                imu_arr = imu_arr_full[mask]
                
                if len(time_arr) < 10:
                    continue
                
                # Low-pass filter
                imu_arr = lowpass_filter(imu_arr)
                
                dataset[cls_label].append((time_arr, imu_arr))
            except Exception as e:
                print(f"  [WARN] Error loading {csv_path}: {e}")
    
    return dataset

# ============================================================================
# FEATURE EXTRACTION PER TRIAL AT A SPECIFIC END TIME
# ============================================================================
def extract_features_up_to_time(time_arr, imu_arr, end_time_s, window_s):
    """
    Extract features using a window of `window_s` seconds ending at `end_time_s`.
    Returns feature vector or None if not enough data.
    """
    win_start = end_time_s - window_s
    mask = (time_arr >= win_start) & (time_arr <= end_time_s)
    window = imu_arr[mask]
    if len(window) < 10:
        return None
    return extract_imu_features(window)

def extract_features_full_window(time_arr, imu_arr, start_s, end_s):
    """
    Extract features from the full [start_s, end_s] window.
    """
    mask = (time_arr >= start_s) & (time_arr <= end_s)
    window = imu_arr[mask]
    if len(window) < 10:
        return None
    return extract_imu_features(window)

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    base_dir   = r"C:\Users\Lucy\Desktop\Anticipation_Delsys\Synchronised_Trials"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir    = os.path.join(script_dir, "Offline_Training_Results", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    plot_dir = os.path.join(out_dir, "Accuracy_Plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Loading data from: {base_dir}")
    print(f"Results will be saved to: {out_dir}")
    
    dataset = load_trial_data(base_dir)
    
    if not dataset:
        print("ERROR: No data loaded. Check your base_dir and folder structure.")
        return
    
    print(f"\nLoaded classes: {sorted(dataset.keys())}")
    for cls, trials in sorted(dataset.items()):
        print(f"  {cls}: {len(trials)} trials")
    
    # -----------------------------------------------------------------------
    # SPLIT: 70/15/15 per class
    # -----------------------------------------------------------------------
    train_ds = collections.defaultdict(list)
    val_ds   = collections.defaultdict(list)
    test_ds  = collections.defaultdict(list)
    rng      = np.random.RandomState(42)
    
    for cls_name in sorted(dataset.keys()):
        arr_list = dataset[cls_name]
        n        = len(arr_list)
        indices  = list(range(n))
        rng.shuffle(indices)
        
        n_train  = max(1, int(0.70 * n))
        n_val    = max(1, int(0.15 * n)) if n > 1 else 0
        n_test   = n - n_train - n_val
        
        train_ds[cls_name] = [arr_list[i] for i in indices[:n_train]]
        val_ds[cls_name]   = [arr_list[i] for i in indices[n_train:n_train+n_val]]
        test_ds[cls_name]  = [arr_list[i] for i in indices[n_train+n_val:]]
    
    print(f"\nData split:")
    print(f"  Train: {sum(len(v) for v in train_ds.values())} trials")
    print(f"  Val:   {sum(len(v) for v in val_ds.values())} trials")
    print(f"  Test:  {sum(len(v) for v in test_ds.values())} trials")
    
    # -----------------------------------------------------------------------
    # BUILD TRAINING FEATURES: Extract from the full post-onset window [0.0, 1.5]
    # -----------------------------------------------------------------------
    le = LabelEncoder()
    le.fit(sorted(dataset.keys()))
    
    win_s  = WINDOW_MS / 1000.0
    step_s = STEP_MS / 1000.0
    
    print(f"\nExtracting training features from [{TRAIN_START_S}s, {TRAIN_END_S}s] window...")
    
    def build_feature_matrix(ds_dict):
        """Extract overlapping windows from [TRAIN_START_S, TRAIN_END_S] as separate samples."""
        X_list, y_list = [], []
        for cls_name, trials in sorted(ds_dict.items()):
            for time_arr, imu_arr in trials:
                # Get all windows within the post-onset training region
                mask = (time_arr >= TRAIN_START_S) & (time_arr <= TRAIN_END_S)
                t_sub = time_arr[mask]
                d_sub = imu_arr[mask]
                if len(t_sub) < 10:
                    continue
                # Slide window across the post-onset period
                win_samples = int(win_s * EMG_FS)
                step_samples = int(step_s * EMG_FS)
                i = 0
                while i + win_samples <= len(d_sub):
                    window = d_sub[i:i + win_samples]
                    feat = extract_imu_features(window)
                    X_list.append(feat)
                    y_list.append(cls_name)
                    i += step_samples
        if not X_list:
            return np.array([]), np.array([])
        return np.array(X_list), np.array(y_list)
    
    X_train_raw, Y_train_raw = build_feature_matrix(train_ds)
    X_val_raw,   Y_val_raw   = build_feature_matrix(val_ds)
    X_test_raw,  Y_test_raw  = build_feature_matrix(test_ds)
    
    if len(X_train_raw) == 0:
        print("ERROR: No training features extracted. Check time range and data.")
        return
    
    print(f"  Feature dim: {X_train_raw.shape[1]}")
    print(f"  Train samples: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    # Global scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)   if len(X_val_raw) > 0   else np.array([])
    X_test  = scaler.transform(X_test_raw)  if len(X_test_raw) > 0  else np.array([])
    
    Y_train_enc = le.transform(Y_train_raw)
    Y_val_enc   = le.transform(Y_val_raw)   if len(Y_val_raw) > 0   else np.array([])
    Y_test_enc  = le.transform(Y_test_raw)  if len(Y_test_raw) > 0  else np.array([])
    
    # -----------------------------------------------------------------------
    # TRAIN: Random Forest (excellent for tabular IMU features)
    # -----------------------------------------------------------------------
    print("\nTraining Random Forest classifier on post-onset IMU data...")
    
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, Y_train_enc)
    
    if len(X_val) > 0:
        val_acc = accuracy_score(Y_val_enc, clf.predict(X_val)) * 100
        print(f"  Validation Accuracy (on 0.0s-1.5s window): {val_acc:.2f}%")
    
    # -----------------------------------------------------------------------
    # EVALUATE ON TEST SET — at the full post-onset window
    # -----------------------------------------------------------------------
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(Y_test_enc, y_pred) * 100
        
        report = classification_report(Y_test_enc, y_pred, target_names=le.classes_,
                                       output_dict=True, zero_division=0)
        
        class_f1s  = [report[cls]['f1-score'] for cls in le.classes_]
        class_accs = []
        for cls in le.classes_:
            mask = (Y_test_enc == le.transform([cls])[0])
            if np.sum(mask) > 0:
                class_accs.append(accuracy_score(Y_test_enc[mask], y_pred[mask]))
        
        f1_mean,  f1_sd  = np.mean(class_f1s)*100,  np.std(class_f1s)*100
        acc_mean, acc_sd = np.mean(class_accs)*100, np.std(class_accs)*100
        
        print(f"\n{'='*50}")
        print(f"TEST RESULTS (Full post-onset window 0.0s -> 1.5s)")
        print(f"{'='*50}")
        print(f"Overall Accuracy:  {test_acc:.2f}%")
        print(f"Macro Accuracy:    {acc_mean:.2f}% ± {acc_sd:.2f}% SD")
        print(f"Macro F1 Score:    {f1_mean:.2f}% ± {f1_sd:.2f}% SD")
        print(f"{'='*50}")
        
        # Confusion Matrix
        cm = confusion_matrix(Y_test_enc, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"IMU-Only Test Confusion Matrix\n"
                  f"Accuracy: {test_acc:.2f}%  |  F1: {f1_mean:.2f}% ± {f1_sd:.2f}%")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
        print(f"Saved confusion matrix to: {out_dir}")
        
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(out_dir, "classification_report.csv"))
    
    # -----------------------------------------------------------------------
    # SLIDING WINDOW ACCURACY OVER TIME
    # Slide a fixed-size window from EVAL_START_S to EVAL_END_S.
    # At each step, extract features and predict, compare to true label.
    # This reveals WHEN the model can first accurately distinguish objects.
    # -----------------------------------------------------------------------
    print("\nComputing sliding-window accuracy over time (this may take a moment)...")
    
    step_s = STEP_MS / 1000.0
    
    # Build a set of evaluation time points (end of window at each step)
    # Window ends from (EVAL_START_S + win_s) to EVAL_END_S
    eval_times = np.arange(EVAL_START_S + win_s, EVAL_END_S + step_s/2, step_s)
    
    # For each class in test_ds, accumulate trial-level accuracy per time step
    per_class_accs = collections.defaultdict(lambda: [[] for _ in range(len(eval_times))])
    overall_accs   = [[] for _ in range(len(eval_times))]
    
    for cls_name, trials in sorted(test_ds.items()):
        cls_idx = le.transform([cls_name])[0]
        
        for time_arr, imu_arr in trials:
            for t_i, t_end in enumerate(eval_times):
                feat = extract_features_up_to_time(time_arr, imu_arr, t_end, win_s)
                if feat is None:
                    continue
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                pred = clf.predict(feat_scaled)[0]
                correct = int(pred == cls_idx)
                per_class_accs[cls_name][t_i].append(correct)
                overall_accs[t_i].append(correct)
    
    # Compute mean accuracy at each timestep
    avg_per_class = {}
    for cls_name in sorted(test_ds.keys()):
        avg_per_class[cls_name] = [
            np.mean(v) * 100 if v else np.nan
            for v in per_class_accs[cls_name]
        ]
    
    overall_avg = [
        np.mean(v) * 100 if v else np.nan
        for v in overall_accs
    ]
    
    chance_level = 100.0 / len(le.classes_)
    
    # -----------------------------------------------------------------------
    # PLOT: Per-class accuracy over time
    # -----------------------------------------------------------------------
    for cls_name in sorted(test_ds.keys()):
        accs = avg_per_class[cls_name]
        n_trials = len(test_ds[cls_name])
        
        plt.figure(figsize=(12, 5))
        plt.bar(eval_times, accs, width=step_s * 0.9,
                color='#00d4ff', edgecolor='black', alpha=0.8)
        plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Motion Onset (0.0s)')
        plt.axhline(y=chance_level, color='orange', linestyle=':', linewidth=1.5,
                    label=f'Chance ({chance_level:.1f}%)')
        plt.ylim(0, 105)
        plt.xlim(EVAL_START_S, EVAL_END_S)
        plt.title(f"IMU Accuracy Over Time — True Class: {cls_name} ({n_trials} test trials)")
        plt.xlabel("Time relative to motion onset (s)")
        plt.ylabel(f"Accuracy for '{cls_name}' (%)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Accuracy_Over_Time_{cls_name}.png"), dpi=150)
        plt.close()
    
    # -----------------------------------------------------------------------
    # PLOT: Overall average accuracy over time (all classes combined)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.bar(eval_times, overall_avg, width=step_s * 0.9,
            color='#ff6b35', edgecolor='black', alpha=0.85)
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Motion Onset (0.0s)')
    plt.axhline(y=chance_level, color='orange', linestyle=':', linewidth=1.5,
                label=f'Chance ({chance_level:.1f}%)')
    plt.ylim(0, 105)
    plt.xlim(EVAL_START_S, EVAL_END_S)
    plt.title(f"IMU Overall Average Accuracy Over Time\n"
              f"(All {sum(len(v) for v in test_ds.values())} test trials, {len(le.classes_)} classes)")
    plt.xlabel("Time relative to motion onset (s)")
    plt.ylabel("Average Accuracy (%)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Accuracy_Over_Time_Overall.png"), dpi=150)
    plt.close()
    
    # -----------------------------------------------------------------------
    # PLOT: Subset — Mug, Card, Bottle only
    # -----------------------------------------------------------------------
    subset_classes = [c for c in ['mug', 'card', 'bottle'] if c in test_ds]
    if subset_classes:
        subset_accs = []
        for t_i in range(len(eval_times)):
            vals = []
            for cls_name in subset_classes:
                if not np.isnan(avg_per_class[cls_name][t_i]):
                    vals.append(avg_per_class[cls_name][t_i])
            subset_accs.append(np.mean(vals) if vals else np.nan)
        
        subset_trials = sum(len(test_ds[c]) for c in subset_classes)
        
        plt.figure(figsize=(12, 5))
        plt.bar(eval_times, subset_accs, width=step_s * 0.9,
                color='#ffaa00', edgecolor='black', alpha=0.85)
        plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Motion Onset (0.0s)')
        plt.axhline(y=100.0 / len(subset_classes), color='orange', linestyle=':', linewidth=1.5,
                    label=f'Chance ({100.0/len(subset_classes):.1f}%)')
        plt.ylim(0, 105)
        plt.xlim(EVAL_START_S, EVAL_END_S)
        plt.title(f"IMU Accuracy Over Time — Mug, Card, Bottle ({subset_trials} test trials)")
        plt.xlabel("Time relative to motion onset (s)")
        plt.ylabel("Average Accuracy (%)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Accuracy_Over_Time_MugCardBottle.png"), dpi=150)
        plt.close()
    
    print(f"\nSaved accuracy-over-time plots to: {plot_dir}")
    print("\nDone!")

if __name__ == "__main__":
    main()
