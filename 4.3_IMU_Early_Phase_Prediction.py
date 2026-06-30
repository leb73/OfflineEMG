"""
4.3 IMU-Only Offline Pipeline — Full Window Time-Step Analysis
=============================================================
Goal: Demonstrate that IMU (accelerometer) data cannot be used to predict the 
"early" phase of movement (i.e. before motion onset at 0s).

This script:
1. Loads full -1.5s to 1.5s trials of IMU data.
2. Applies a baseline correction by subtracting the pre-onset stationary mean (-1.5 to -0.5s).
3. Balances the classes so each has the exact same number of trials.
4. Splits into Train/Val/Test (70/15/15).
5. Extracts sliding window features (200ms width, 50ms step) across the entire -1.5s to +1.5s span.
6. Trains a Random Forest classifier on ALL training windows.
7. Evaluates the classifier at each time step on the test set.
8. Outputs comprehensive metrics (CSV), a confusion matrix per time step, and an overall Accuracy Over Time plot.
"""

import os
import glob
import random
import collections
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
                             classification_report, precision_recall_fscore_support)
from sklearn.ensemble import RandomForestClassifier

# Reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("  4.3  IMU-Only Early Phase Prediction")
print("=" * 60)

# ============================================================================
# CONSTANTS
# ============================================================================
FS            = 74.0741  # IMU Sample Rate
T_START_S     = -1.5
T_END_S       = 1.5

BASELINE_START_S = -1.5
BASELINE_END_S   = -0.5

WINDOW_MS     = 100
STEP_MS       = 25

WINDOW_S      = WINDOW_MS / 1000.0
STEP_S        = STEP_MS  / 1000.0


# ============================================================================
# FILTER
# ============================================================================
def lowpass_filter(data: np.ndarray, cutoff: float = 20.0,
                   fs: float = FS, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter."""
    nyq = fs / 2.0
    sos = butter(order, min(cutoff, nyq - 1.0), btype='low', fs=fs, output='sos')
    out = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        out[:, ch] = sosfilt(sos, data[:, ch])
    return out


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_imu_features(window: np.ndarray) -> np.ndarray:
    """Extract features from a window of shape (samples, channels)."""
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        mean     = np.mean(x)
        std      = np.std(x)
        rms      = np.sqrt(np.mean(x ** 2))
        p2p      = np.ptp(x)
        auc      = np.trapezoid(np.abs(x))
        energy   = np.sum(x ** 2)
        zcr      = np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x) if len(x) > 0 else 0
        grad_rms = np.sqrt(np.mean(np.diff(x) ** 2)) if len(x) > 1 else 0.0
        feats.extend([mean, std, rms, p2p, auc, energy, zcr, grad_rms])
    return np.array(feats, dtype=np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_preprocess_trials(base_dir: str) -> dict:
    dataset = collections.defaultdict(list)
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_*_Short")))
    
    if not trial_folders:
        print(f"  [WARN] No folders found in: {base_dir}")
        return dataset

    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        parts = folder_name.split("_")
        if len(parts) >= 3:
            cls_label = parts[2]
        else:
            continue

        for csv_path in sorted(glob.glob(os.path.join(tf_path, "movement_*.csv"))):

            try:
                with open(csv_path, 'r') as f:
                    for _ in range(5): f.readline()
                    num_cols = len(f.readline().split(','))

                df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
                if len(df) <= 2: continue
                df = df.iloc[2:].reset_index(drop=True)

                acc_time_cols = [c for c in df.columns if 'ACC' in c and 'Time' in c]
                if not acc_time_cols: continue
                
                time_col = acc_time_cols[0]
                time_vals = pd.to_numeric(df[time_col], errors='coerce').values.astype(np.float64)
                valid_idx = ~np.isnan(time_vals)
                time_vals = time_vals[valid_idx]

                acc_cols = [c for c in df.columns if 'ACC' in c and '(G)' in c]
                if not acc_cols: continue

                imu_full = (df[acc_cols].iloc[valid_idx]
                            .apply(pd.to_numeric, errors='coerce')
                            .fillna(0.0).values.astype(np.float64))

                mask = (time_vals >= T_START_S) & (time_vals <= T_END_S)
                time_arr = time_vals[mask]
                imu_arr  = imu_full[mask]

                if len(time_arr) < 10: continue

                # Lowpass Filter
                imu_arr = lowpass_filter(imu_arr, fs=FS)
                
                # Baseline Correction
                base_mask = (time_arr >= BASELINE_START_S) & (time_arr <= BASELINE_END_S)
                baseline = imu_arr[base_mask].mean(axis=0) if base_mask.sum() > 5 else np.zeros(imu_arr.shape[1])
                imu_arr = imu_arr - baseline
                
                dataset[cls_label].append((time_arr, imu_arr))
            except Exception as exc:
                print(f"  [WARN] Error loading {csv_path}: {exc}")

    return dataset


# ============================================================================
# BUILD DATASET
# ============================================================================
def build_windowed_dataset(ds_dict: dict, t_start: float, t_end: float, win_s: float, step_s: float):
    X, Y = [], []
    eval_times = np.arange(t_start + win_s, t_end + step_s / 2.0, step_s)
    
    for cls_name, trials in ds_dict.items():
        for time_arr, imu_arr in trials:
            for t_e in eval_times:
                t_s = t_e - win_s
                mask = (time_arr >= t_s) & (time_arr <= t_e)
                window = imu_arr[mask]
                if len(window) < 5: continue
                X.append(extract_imu_features(window))
                Y.append(cls_name)
    return np.array(X), np.array(Y)


# ============================================================================
# EVALUATION HELPERS
# ============================================================================
def save_cm(cm, class_names, out_path, acc, t_e):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='lightgray')
                
    ax.tick_params(colors='black', labelsize=10)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f"Confusion Matrix at t={t_e:.2f}s | Acc: {acc:.1f}%", color='black', pad=15, fontsize=14, fontweight='bold')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='black')
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_accuracy_over_time(eval_times, acc_history, chance, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Clean white background for academic papers
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Shading for pre/post onset (subtle)
    ax.axvspan(T_START_S, 0.0, alpha=0.15, color='#B0BEC5', label='Pre-movement Phase')
    ax.axvspan(0.0, T_END_S, alpha=0.15, color='#FFCC80', label='Movement Phase')
    
    # Bar chart
    width = STEP_S * 0.8
    ax.bar(eval_times, acc_history, width=width, color='#546E7A', edgecolor='black', zorder=3)
    
    # Motion onset line
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=2, label='Motion Onset (0s)', zorder=4)
    
    # Remove chance line as requested
    
    ax.tick_params(colors='black', labelsize=12)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    
    ax.grid(True, axis='y', color='#E0E0E0', linestyle='--', zorder=0)
    
    ax.set_xlabel('Time relative to motion onset (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('IMU-Only Predictive Accuracy Over Time', fontsize=16, fontweight='bold')
    ax.legend(facecolor='white', edgecolor='black', fontsize=12, loc='upper left')
    
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300) # high res for paper
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    base_dir = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials_shifted"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", f"{timestamp}_4.3_IMU_Full_Window")
    ts_dir = os.path.join(out_dir, "Time_Step_Metrics")
    os.makedirs(ts_dir, exist_ok=True)

    print("\n1. Loading and preprocessing data...")
    dataset = load_and_preprocess_trials(base_dir)
    if not dataset: return

    # Trial Balancing
    min_trials = min([len(v) for v in dataset.values()])
    print(f"\nBalancing classes to {min_trials} trials each.")
    for cls in dataset.keys():
        dataset[cls] = random.sample(dataset[cls], min_trials)

    # Splitting
    train_ds, val_ds, test_ds = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    n_train = max(1, int(0.70 * min_trials))
    n_val   = max(1, int(0.15 * min_trials))
    for cls, trials in dataset.items():
        train_ds[cls] = trials[:n_train]
        val_ds[cls]   = trials[n_train:n_train+n_val]
        test_ds[cls]  = trials[n_train+n_val:]
        
    print(f"Split sizes per class -> Train: {len(train_ds[cls])}, Val: {len(val_ds[cls])}, Test: {len(test_ds[cls])}")

    le = LabelEncoder()
    le.fit(list(dataset.keys()))
    chance = 100.0 / len(le.classes_)

    print("\n2. Extracting features from POST-ONSET windows only (to prevent noise memorization)...")
    X_train_raw, Y_train_raw = build_windowed_dataset(train_ds, 0.0, T_END_S, WINDOW_S, STEP_S)
    if len(X_train_raw) == 0:
        print("No training data extracted!")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    Y_train_enc = le.transform(Y_train_raw)

    print(f"\n3. Training Random Forest on {len(X_train)} windows...")
    clf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, Y_train_enc)

    print("\n4. Evaluating over time on Test Set...")
    eval_times = np.arange(T_START_S + WINDOW_S, T_END_S + STEP_S / 2.0, STEP_S)
    acc_history = []

    for t_e in eval_times:
        X_t, Y_t = [], []
        for cls_name, trials in test_ds.items():
            for time_arr, imu_arr in trials:
                t_s = t_e - WINDOW_S
                mask = (time_arr >= t_s) & (time_arr <= t_e)
                window = imu_arr[mask]
                if len(window) < 5: continue
                X_t.append(extract_imu_features(window))
                Y_t.append(cls_name)
                
        if len(X_t) == 0:
            acc_history.append(np.nan)
            continue
            
        X_t_scaled = scaler.transform(X_t)
        Y_t_enc = le.transform(Y_t)
        y_pred = clf.predict(X_t_scaled)
        
        # Metrics
        acc = accuracy_score(Y_t_enc, y_pred) * 100
        acc_history.append(acc)
        
        precision, recall, f1, _ = precision_recall_fscore_support(Y_t_enc, y_pred, zero_division=0)
        
        # Save CSV
        df_metrics = pd.DataFrame({
            'Class': le.classes_,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100
        })
        df_metrics.loc['macro'] = ['MACRO', precision.mean()*100, recall.mean()*100, f1.mean()*100]
        df_metrics.loc['overall'] = ['OVERALL_ACC', acc, acc, acc]
        
        df_metrics.to_csv(os.path.join(ts_dir, f"metrics_t_{t_e:.2f}s.csv"), index=False)
        
        # Save CM
        cm = confusion_matrix(Y_t_enc, y_pred, labels=range(len(le.classes_)))
        save_cm(cm, le.classes_, os.path.join(ts_dir, f"cm_t_{t_e:.2f}s.png"), acc, t_e)
        
        print(f"  t={t_e:+.2f}s | Acc: {acc:5.1f}% | Saved metrics & CM.")

        # If it's the final timestep, print out the exact metrics
        if t_e == eval_times[-1]:
            print(f"\n--- Final Timestep (t={t_e:+.2f}s) Metrics ---")
            print(f"Overall Accuracy: {acc:.1f}%")
            print(f"Macro Precision:  {precision.mean()*100:.1f}%")
            print(f"Macro Recall:     {recall.mean()*100:.1f}%")
            print(f"Macro F1-Score:   {f1.mean()*100:.1f}%\n")
            print("Per-Class Details:")
            for i, c_name in enumerate(le.classes_):
                print(f"  {c_name:20s} - Precision: {precision[i]*100:5.1f}%, Recall: {recall[i]*100:5.1f}%, F1: {f1[i]*100:5.1f}%")
            print("---------------------------------------------\n")
            
            # Save final metrics to the top level folder
            df_metrics.to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)

    print("\n5. Generating summary plots...")
    plot_accuracy_over_time(eval_times, acc_history, chance, os.path.join(out_dir, "Accuracy_Over_Time.png"))
    
    print(f"\nDone! Results saved in {out_dir}")

if __name__ == "__main__":
    main()
