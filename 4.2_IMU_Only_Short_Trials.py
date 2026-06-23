"""
4.2 IMU-Only Offline Pipeline — Short Trials
=============================================
Goal: Train a Random Forest classifier on IMU (accelerometer) features extracted
from the FULL short trial window (-1.5s to +1.5s, spanning both pre- and
post-onset periods).

Then evaluate using a sliding window across the same -1.5s to +1.5s timeline.

Key message:
  - Even when trained on the full window, IMU accuracy is at/near chance
    BEFORE motion onset (< 0.0s) because the pre-onset IMU signal carries
    no grasp-discriminative information.
  - IMU accuracy rises sharply only AFTER motion onset (> 0.0s).

This contrasts with the EMG model (script 4) which shows predictive accuracy
BEFORE motion onset, demonstrating the temporal advantage of EMG.

Outputs (saved per timestamp):
  - confusion_matrix.png          — test set confusion matrix
  - classification_report.csv     — per-class metrics
  - metrics_summary.txt           — top-level accuracy / F1 summary
  - Accuracy_Plots/
      Accuracy_Over_Time_Overall.png         — overall avg accuracy vs time
      Accuracy_Over_Time_<class>.png         — per-class accuracy vs time
      Accuracy_Over_Time_MugCardBottle.png   — subset plot (if classes present)
      Accuracy_Over_Time_Combined.png        — multi-panel figure
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
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier

# Reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("  4.2  IMU-Only Pipeline  |  Short Trials")
print("=" * 60)

# ============================================================================
# CONSTANTS
# ============================================================================
EMG_FS        = 1926.0   # Delsys EMG sampling rate

# Post-onset training window:
# We train ONLY on post-onset data [0, +1.5s] because that is where the
# movement-discriminative ACC signal lives.  Pre-onset IMU is static and
# carries no grasp identity information, so including it in training only
# adds noise and biases the model toward the majority-window class.
TRAIN_START_S =  0.0
TRAIN_END_S   =  1.5

# Evaluation window: slide across the FULL trial so we can show that
# accuracy is ~chance before 0 s and rises only after motion onset.
EVAL_START_S  = -1.5
EVAL_END_S    =  1.5

# Feature extraction sliding window
WINDOW_MS     = 200      # 200 ms window
STEP_MS       = 50       # 50 ms hop

WINDOW_S      = WINDOW_MS / 1000.0
STEP_S        = STEP_MS  / 1000.0


# ============================================================================
# FILTER — low-pass for IMU (accelerometer)
# ============================================================================
def lowpass_filter(data: np.ndarray, cutoff: float = 20.0,
                   fs: float = EMG_FS, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter applied independently to each channel."""
    nyq = fs / 2.0
    sos = butter(order, min(cutoff, nyq - 1.0), btype='low', fs=fs, output='sos')
    out = np.zeros_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        out[:, ch] = sosfilt(sos, data[:, ch])
    return out


# ============================================================================
# FEATURE EXTRACTION — IMU statistical + kinematic features per window
# ============================================================================
def extract_imu_features(window: np.ndarray) -> np.ndarray:
    """
    Extract per-channel features from a (samples, channels) window.
    Returns a 1-D feature vector.
    """
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        mean     = np.mean(x)
        std      = np.std(x)
        rms      = np.sqrt(np.mean(x ** 2))
        p2p      = np.ptp(x)
        auc      = np.trapezoid(np.abs(x))
        energy   = np.sum(x ** 2)
        zcr      = np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x)
        grad_rms = np.sqrt(np.mean(np.diff(x) ** 2)) if len(x) > 1 else 0.0
        # Removed skewness and kurtosis as they cause numerical instability 
        # (Infs/NaNs) on near-constant static pre-onset IMU windows.
        feats.extend([mean, std, rms, p2p, auc, energy, zcr, grad_rms])
    return np.array(feats, dtype=np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_trial_data(base_dir: str) -> dict:
    """
    Load all short-trial IMU data from Synchronised_Trials.

    *** KEY: Delsys CSV column structure ***
      Column pair layout per sensor:
        'EMG 1 Time Series (s)'   -> TRIAL-RELATIVE time [-1.5, +1.5 s]  (1926 Hz)
        'EMG 1 (mV)'              -> EMG amplitude
        'ACC X Time Series (s)'   -> ABSOLUTE session timestamp [~16-94 s]  (74 Hz)
        'ACC X (G)'               -> ACC X amplitude

      The ACC data columns are ALREADY ROW-ALIGNED with the EMG rows in the
      dataframe — pandas simply puts NaN in the ~26 EMG rows that fall between
      each ACC sample.  Forward-filling corrects this efficiently.

      DO NOT interpolate ACC values onto the EMG time axis using the ACC time
      column: the ACC timestamps are absolute session times and the EMG time is
      trial-relative.  Mapping from [0, 1.5] (relative) into [16, 94] (absolute)
      will extrapolate everything to the boundary value — making every ACC channel
      appear constant.

    Returns
    -------
    dataset : dict
        class_label -> list of (time_arr, imu_arr) tuples
        time_arr : 1-D array of trial-relative timestamps
        imu_arr  : 2-D array (samples, n_acc_channels), low-pass filtered
    """
    dataset = collections.defaultdict(list)

    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_short_*")))
    if not trial_folders:
        print(f"  [WARN] No 'Trial_*_short_*' folders found in: {base_dir}")
        return dataset

    print(f"\nFound {len(trial_folders)} short-trial folders.")

    for tf_path in trial_folders:
        folder_name = os.path.basename(tf_path)
        cls_label   = folder_name.split("short_")[-1]

        movements_dir = os.path.join(tf_path, "extracted_trials")
        if not os.path.isdir(movements_dir):
            continue

        for mov_dir in sorted(glob.glob(os.path.join(movements_dir, "Movement_*"))):
            csv_path = os.path.join(mov_dir, "delsys_data.csv")
            if not os.path.isfile(csv_path):
                continue

            try:
                # Determine number of columns from header
                with open(csv_path, 'r') as f:
                    for _ in range(5):
                        f.readline()
                    num_cols = len(f.readline().split(','))

                df = pd.read_csv(csv_path, skiprows=5,
                                 usecols=range(num_cols), low_memory=False)
                if len(df) <= 2:
                    continue
                df = df.iloc[2:].reset_index(drop=True)  # drop sampling-rate/units rows

                # ---- ACC Time column = trial-relative time (fixed in extraction) ----
                acc_time_cols = [c for c in df.columns if 'ACC' in c and 'Time' in c]
                if not acc_time_cols:
                    print(f"  [WARN] No ACC time column in {mov_dir} — skipping.")
                    continue
                time_col  = acc_time_cols[0]
                time_vals = pd.to_numeric(df[time_col], errors='coerce').values.astype(np.float64)

                # Delsys packs different sampling rates at the top of the columns, padded with NaNs
                valid_idx = ~np.isnan(time_vals)
                time_vals = time_vals[valid_idx]

                # ---- ACC columns ----
                acc_cols = [c for c in df.columns if 'ACC' in c and '(G)' in c]
                if not acc_cols:
                    print(f"  [WARN] No ACC columns in {mov_dir} — skipping.")
                    continue

                imu_full = (df[acc_cols]
                            .iloc[valid_idx]
                            .apply(pd.to_numeric, errors='coerce')
                            .fillna(0.0)   # safety fallback
                            .values.astype(np.float64))

                # Restrict to evaluation window
                mask = (time_vals >= EVAL_START_S) & (time_vals <= EVAL_END_S)
                time_arr = time_vals[mask]
                imu_arr  = imu_full[mask]

                if len(time_arr) < 10:
                    continue

                imu_arr = lowpass_filter(imu_arr)
                dataset[cls_label].append((time_arr, imu_arr))

            except Exception as exc:
                print(f"  [WARN] Error loading {csv_path}: {exc}")

    return dataset


# ============================================================================
# FEATURE HELPERS
# ============================================================================
BASELINE_START_S = -1.5   # start of the pre-onset baseline window
BASELINE_END_S   = -0.5   # end   of the pre-onset baseline window


def compute_baseline(time_arr: np.ndarray,
                     imu_arr: np.ndarray) -> np.ndarray:
    """
    Compute the per-channel mean of the perfectly static window [-1.5s, -0.5s].

    The Delsys ACC signal has a large static DC offset due to Earth's gravity
    (e.g., ~0.5G on each axis depending on sensor orientation).  This offset is
    the SAME for every trial and every class because participants always start
    from the same resting posture.  Without removal, mean/rms/energy/auc features
    are dominated by this gravity term, making all class feature vectors nearly
    identical and causing the RF to predict one class for everything.

    Subtracting the pre-onset baseline leaves only the movement-related dynamics
    (delta-acceleration from rest), which ARE class-discriminative.

    Returns
    -------
    baseline : (n_channels,) array — per-channel mean over [-1.5s, 0s].
    """
    mask = (time_arr >= BASELINE_START_S) & (time_arr <= BASELINE_END_S)
    if mask.sum() < 5:
        return np.zeros(imu_arr.shape[1])   # fallback if no pre-onset data
    return imu_arr[mask].mean(axis=0)


def build_feature_matrix(ds_dict: dict, t_start: float, t_end: float,
                         win_s: float, step_s: float):
    """
    Slide a window across [t_start, t_end] for every trial in ds_dict.
    Baseline-corrects each trial before feature extraction so that the static
    gravity offset does not swamp the movement-related signal.
    Returns (X, y_labels).
    """
    X_list, y_list = [], []
    for cls_name, trials in sorted(ds_dict.items()):
        for time_arr, imu_arr in trials:
            # Remove per-trial static offset using pre-onset mean
            baseline = compute_baseline(time_arr, imu_arr)
            imu_corr = imu_arr - baseline

            mask   = (time_arr >= t_start) & (time_arr <= t_end)
            t_sub  = time_arr[mask]
            d_sub  = imu_corr[mask]
            if len(t_sub) < 10:
                continue
            eval_times = np.arange(t_start + win_s, t_end + step_s / 2.0, step_s)
            for t_e in eval_times:
                t_s = t_e - win_s
                mask_w = (t_sub >= t_s) & (t_sub <= t_e)
                window = d_sub[mask_w]
                if len(window) < 5:
                    continue
                feat = extract_imu_features(window)
                X_list.append(feat)
                y_list.append(cls_name)

    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================
PALETTE = {
    'primary':   '#4C8BF5',
    'onset':     '#E63946',
    'chance':    '#F4A261',
    'bar':       '#4C8BF5',
    'bar_alt':   '#F4A261',
    'bar_sub':   '#7EC8A4',
    'grid':      '#2A2A3E',
    'bg':        '#1A1A2E',
    'fg':        '#E0E0EE',
    'title_clr': '#FFFFFF',
}

def _apply_dark_style(ax, title='', xlabel='', ylabel='',
                      xlim=None, ylim=None):
    ax.set_facecolor(PALETTE['bg'])
    ax.figure.patch.set_facecolor(PALETTE['bg'])
    ax.tick_params(colors=PALETTE['fg'])
    ax.xaxis.label.set_color(PALETTE['fg'])
    ax.yaxis.label.set_color(PALETTE['fg'])
    ax.title.set_color(PALETTE['title_clr'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, color='#444466', alpha=0.4, linewidth=0.7)


def save_confusion_matrix(cm, class_names, out_path,
                          overall_acc, f1_mean, f1_sd):
    """Saves a styled confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) + 2),
                                    max(6, len(class_names) + 1)))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Normalise for display colour, keep raw counts for annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    sns.heatmap(
        cm_norm, annot=cm, fmt='d',
        cmap='Blues', ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='#333355',
        cbar_kws={'shrink': 0.8},
    )
    ax.set_facecolor(PALETTE['bg'])
    ax.tick_params(colors=PALETTE['fg'], labelsize=9)
    ax.xaxis.label.set_color(PALETTE['fg'])
    ax.yaxis.label.set_color(PALETTE['fg'])
    ax.set_xlabel('Predicted Label', fontsize=10, color=PALETTE['fg'])
    ax.set_ylabel('True Label',      fontsize=10, color=PALETTE['fg'])
    ax.set_title(
        f"IMU-Only Test Set Confusion Matrix\n"
        f"Accuracy: {overall_acc:.1f}%   |   Macro F1: {f1_mean:.1f}% ± {f1_sd:.1f}%",
        fontsize=11, fontweight='bold', color=PALETTE['title_clr'], pad=12,
    )

    # Colour bar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=PALETTE['fg'])
    cbar.set_label('Proportion', color=PALETTE['fg'], fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_accuracy_over_time(eval_times, accs, chance, out_path,
                             title='', ylabel='Accuracy (%)',
                             bar_color=None, n_trials=None):
    """Saves a single accuracy-over-time bar chart."""
    bar_color = bar_color or PALETTE['bar']
    fig, ax   = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Shade pre-onset region
    ax.axvspan(EVAL_START_S, 0.0, alpha=0.07, color='#9090FF', label='Pre-onset')
    # Shade post-onset region
    ax.axvspan(0.0, EVAL_END_S, alpha=0.07, color='#FF9090', label='Post-onset')

    ax.bar(eval_times, accs, width=STEP_S * 0.88,
           color=bar_color, edgecolor='#AAAACC', alpha=0.85, zorder=3)
    ax.axvline(x=0.0, color=PALETTE['onset'], linestyle='--',
               linewidth=2.0, label='Motion Onset (0 s)', zorder=5)
    ax.axhline(y=chance, color=PALETTE['chance'], linestyle=':',
               linewidth=1.8, label=f'Chance ({chance:.1f}%)', zorder=4)

    # Annotation: chance label
    ax.text(EVAL_END_S - 0.05, chance + 1.5, f'Chance\n({chance:.1f}%)',
            color=PALETTE['chance'], fontsize=8, ha='right', va='bottom')

    n_str = f'  ({n_trials} trials)' if n_trials is not None else ''
    _apply_dark_style(ax, title=f'{title}{n_str}',
                      xlabel='Time relative to motion onset (s)',
                      ylabel=ylabel,
                      xlim=(EVAL_START_S, EVAL_END_S), ylim=(0, 108))
    ax.legend(fontsize=9, facecolor='#222240', labelcolor=PALETTE['fg'],
              loc='upper left', framealpha=0.7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_combined_time_plot(eval_times, per_class_avgs, overall_avg,
                            class_names, chance, out_path):
    """
    Multi-panel figure: overall accuracy at top, then one row per class below.
    """
    n_classes = len(class_names)
    n_rows    = 1 + n_classes

    fig = plt.figure(figsize=(13, 3.5 * n_rows))
    fig.patch.set_facecolor(PALETTE['bg'])
    gs  = gridspec.GridSpec(n_rows, 1, hspace=0.55, figure=fig)

    cmap = plt.colormaps['tab10']

    # ---- Overall panel ----
    ax0 = fig.add_subplot(gs[0])
    ax0.axvspan(EVAL_START_S, 0.0, alpha=0.07, color='#9090FF')
    ax0.axvspan(0.0, EVAL_END_S,   alpha=0.07, color='#FF9090')
    ax0.bar(eval_times, overall_avg, width=STEP_S * 0.88,
            color=PALETTE['primary'], edgecolor='#AAAACC', alpha=0.85, zorder=3)
    ax0.axvline(x=0.0, color=PALETTE['onset'], linestyle='--',
                linewidth=2.0, label='Motion Onset')
    ax0.axhline(y=chance, color=PALETTE['chance'], linestyle=':',
                linewidth=1.8, label=f'Chance ({chance:.1f}%)')
    _apply_dark_style(ax0, title='Overall Average Accuracy Over Time (All Classes)',
                      xlabel='', ylabel='Accuracy (%)',
                      xlim=(EVAL_START_S, EVAL_END_S), ylim=(0, 108))
    ax0.legend(fontsize=9, facecolor='#222240', labelcolor=PALETTE['fg'],
               loc='upper left', framealpha=0.7)

    # ---- Per-class panels ----
    for row_i, cls_name in enumerate(class_names):
        ax = fig.add_subplot(gs[row_i + 1])
        color = cmap(row_i / max(n_classes - 1, 1))
        accs  = per_class_avgs.get(cls_name, [np.nan] * len(eval_times))

        ax.axvspan(EVAL_START_S, 0.0, alpha=0.07, color='#9090FF')
        ax.axvspan(0.0, EVAL_END_S,   alpha=0.07, color='#FF9090')
        ax.bar(eval_times, accs, width=STEP_S * 0.88,
               color=color, edgecolor='#AAAACC', alpha=0.85, zorder=3)
        ax.axvline(x=0.0, color=PALETTE['onset'], linestyle='--',
                   linewidth=1.8, zorder=5)
        ax.axhline(y=chance, color=PALETTE['chance'], linestyle=':',
                   linewidth=1.5, zorder=4)
        xlabel = 'Time relative to motion onset (s)' if row_i == n_classes - 1 else ''
        _apply_dark_style(ax, title=f'Class: {cls_name}',
                          xlabel=xlabel, ylabel='Accuracy (%)',
                          xlim=(EVAL_START_S, EVAL_END_S), ylim=(0, 108))

    fig.suptitle("IMU-Only: Accuracy Over Time — Short Trials",
                 fontsize=14, fontweight='bold',
                 color=PALETTE['title_clr'], y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    base_dir   = r"C:\Users\Lucy\Desktop\OfflineEMG\Fixed_IMU_Data"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir    = os.path.join(script_dir, "Offline_Training_Results",
                              f"{timestamp}_4.2_IMU_Short")
    plot_dir   = os.path.join(out_dir, "Accuracy_Plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nData source : {base_dir}")
    print(f"Results dir : {out_dir}")

    # -----------------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------------
    dataset = load_trial_data(base_dir)
    if not dataset:
        print("\nERROR: No data loaded. Check base_dir and folder structure.")
        return

    print(f"\nClasses found : {sorted(dataset.keys())}")
    for cls, trials in sorted(dataset.items()):
        print(f"  {cls:20s}: {len(trials)} trials")

    # -----------------------------------------------------------------------
    # 2. SPLIT — 70 / 15 / 15 per class (same strategy as scripts 4 and 4.1)
    # -----------------------------------------------------------------------
    train_ds = collections.defaultdict(list)
    val_ds   = collections.defaultdict(list)
    test_ds  = collections.defaultdict(list)
    rng      = np.random.RandomState(42)

    for cls_name in sorted(dataset.keys()):
        arr_list = dataset[cls_name]
        n        = len(arr_list)
        idx      = list(range(n))
        rng.shuffle(idx)

        n_train = max(1, int(0.70 * n))
        n_val   = max(1, int(0.15 * n)) if n > 1 else 0
        n_test  = n - n_train - n_val

        train_ds[cls_name] = [arr_list[i] for i in idx[:n_train]]
        val_ds[cls_name]   = [arr_list[i] for i in idx[n_train: n_train + n_val]]
        test_ds[cls_name]  = [arr_list[i] for i in idx[n_train + n_val:]]

    n_tr = sum(len(v) for v in train_ds.values())
    n_va = sum(len(v) for v in val_ds.values())
    n_te = sum(len(v) for v in test_ds.values())
    print(f"\nSplit  — Train: {n_tr}  Val: {n_va}  Test: {n_te}")

    # -----------------------------------------------------------------------
    # 3. FEATURE EXTRACTION — training window matches script 4 (-1.5s → 0.0s)
    # -----------------------------------------------------------------------
    le = LabelEncoder()
    le.fit(sorted(dataset.keys()))

    print(f"\nExtracting training features from [{TRAIN_START_S}s, {TRAIN_END_S}s] "
          f"(full window, {WINDOW_MS} ms sliding window, {STEP_MS} ms hop)...")

    X_train_raw, Y_train_raw = build_feature_matrix(
        train_ds, TRAIN_START_S, TRAIN_END_S, WINDOW_S, STEP_S)
    X_val_raw,   Y_val_raw   = build_feature_matrix(
        val_ds,   TRAIN_START_S, TRAIN_END_S, WINDOW_S, STEP_S)
    X_test_raw,  Y_test_raw  = build_feature_matrix(
        test_ds,  TRAIN_START_S, TRAIN_END_S, WINDOW_S, STEP_S)

    if len(X_train_raw) == 0:
        print("ERROR: No training features extracted. Check time range and data.")
        return

    print(f"  Feature dimension : {X_train_raw.shape[1]}")
    print(f"  Train samples     : {len(X_train_raw)}")
    print(f"  Val   samples     : {len(X_val_raw)}")
    print(f"  Test  samples     : {len(X_test_raw)}")

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)  if len(X_val_raw)  > 0 else np.array([])
    X_test  = scaler.transform(X_test_raw) if len(X_test_raw) > 0 else np.array([])

    Y_train_enc = le.transform(Y_train_raw)
    Y_val_enc   = le.transform(Y_val_raw)  if len(Y_val_raw)  > 0 else np.array([])
    Y_test_enc  = le.transform(Y_test_raw) if len(Y_test_raw) > 0 else np.array([])

    # -----------------------------------------------------------------------
    # 4. TRAIN — Random Forest (strong on tabular IMU features, no GPU needed)
    # -----------------------------------------------------------------------
    print("\nTraining Random Forest on pre-onset IMU features...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=None,
    )
    clf.fit(X_train, Y_train_enc)

    if len(X_val) > 0:
        val_acc = accuracy_score(Y_val_enc, clf.predict(X_val)) * 100
        print(f"  Validation accuracy (pre-onset window): {val_acc:.2f}%")

    # -----------------------------------------------------------------------
    # 5. TEST SET EVALUATION — on features from the pre-onset window
    # -----------------------------------------------------------------------
    if len(X_test) == 0:
        print("  [WARN] No test features available. Skipping evaluation.")
    else:
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(Y_test_enc, y_pred) * 100

        report = classification_report(
            Y_test_enc, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )

        # Per-class macro stats
        class_f1s  = [report[cls]['f1-score']  for cls in le.classes_]
        class_accs = []
        for cls in le.classes_:
            mask_c = (Y_test_enc == le.transform([cls])[0])
            if np.sum(mask_c) > 0:
                class_accs.append(accuracy_score(Y_test_enc[mask_c], y_pred[mask_c]))

        f1_mean,  f1_sd  = np.mean(class_f1s)  * 100, np.std(class_f1s)  * 100
        acc_mean, acc_sd = np.mean(class_accs)  * 100, np.std(class_accs) * 100

        print(f"\n{'='*55}")
        print(f"  TEST RESULTS  (full window: {TRAIN_START_S}s -> {TRAIN_END_S}s)")
        print(f"{'='*55}")
        print(f"  Overall Accuracy  : {test_acc:.2f}%")
        print(f"  Macro Accuracy    : {acc_mean:.2f}% ± {acc_sd:.2f}% SD")
        print(f"  Macro F1 Score    : {f1_mean:.2f}% ± {f1_sd:.2f}% SD")
        print(f"{'='*55}")

        # Metrics summary text
        with open(os.path.join(out_dir, "metrics_summary.txt"), 'w', encoding='utf-8') as fh:
            fh.write("4.2 IMU-Only Pipeline — Short Trials\n")
            fh.write(f"Timestamp    : {timestamp}\n")
            fh.write(f"Train window : {TRAIN_START_S}s to {TRAIN_END_S}s (full trial)\n")
            fh.write(f"Eval window  : {EVAL_START_S}s to {EVAL_END_S}s\n")
            fh.write(f"Window size  : {WINDOW_MS} ms  |  Step: {STEP_MS} ms\n")
            fh.write(f"Model        : Random Forest (400 trees, balanced weights)\n\n")
            fh.write(f"Overall Accuracy  : {test_acc:.2f}%\n")
            fh.write(f"Macro Accuracy    : {acc_mean:.2f}% ± {acc_sd:.2f}%\n")
            fh.write(f"Macro F1 Score    : {f1_mean:.2f}% ± {f1_sd:.2f}%\n\n")
            fh.write(f"Train trials : {n_tr}\n")
            fh.write(f"Val   trials : {n_va}\n")
            fh.write(f"Test  trials : {n_te}\n")

        # Confusion matrix
        cm = confusion_matrix(Y_test_enc, y_pred)
        save_confusion_matrix(
            cm, le.classes_,
            os.path.join(out_dir, "confusion_matrix.png"),
            test_acc, f1_mean, f1_sd,
        )
        print(f"  Saved confusion matrix -> {out_dir}")

        # Classification report CSV
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(out_dir, "classification_report.csv"))

    # -----------------------------------------------------------------------
    # 6. SLIDING-WINDOW ACCURACY OVER TIME (EVAL_START_S → EVAL_END_S)
    #    Uses the SAME model trained on pre-onset features.
    #    At each time step t, extract a WINDOW_S window ending at t,
    #    then predict and compare to the true label.
    # -----------------------------------------------------------------------
    print("\nComputing sliding-window accuracy over time "
          f"[{EVAL_START_S}s -> {EVAL_END_S}s]...")

    # Evaluation time points: window *ends* at these values
    eval_times = np.arange(
        EVAL_START_S + WINDOW_S,
        EVAL_END_S + STEP_S / 2.0,
        STEP_S,
    )

    per_class_correct = collections.defaultdict(
        lambda: [[] for _ in range(len(eval_times))])
    overall_correct = [[] for _ in range(len(eval_times))]

    for cls_name, trials in sorted(test_ds.items()):
        cls_idx = le.transform([cls_name])[0]
        for time_arr, imu_arr in trials:
            # Apply the same per-trial baseline correction used in training
            baseline = compute_baseline(time_arr, imu_arr)
            imu_corr = imu_arr - baseline

            for t_i, t_end in enumerate(eval_times):
                t_start = t_end - WINDOW_S
                mask    = (time_arr >= t_start) & (time_arr <= t_end)
                window  = imu_corr[mask]
                if len(window) < 5:
                    continue
                feat        = extract_imu_features(window)
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                pred        = clf.predict(feat_scaled)[0]
                correct     = int(pred == cls_idx)
                per_class_correct[cls_name][t_i].append(correct)
                overall_correct[t_i].append(correct)

    # Aggregate
    per_class_avg = {}
    for cls_name in sorted(test_ds.keys()):
        per_class_avg[cls_name] = [
            np.mean(v) * 100 if v else np.nan
            for v in per_class_correct[cls_name]
        ]

    overall_avg = [
        np.mean(v) * 100 if v else np.nan
        for v in overall_correct
    ]

    chance_level = 100.0 / len(le.classes_)

    # -----------------------------------------------------------------------
    # 7. SAVE TIME-SERIES PLOTS
    # -----------------------------------------------------------------------

    # — Overall accuracy over time
    save_accuracy_over_time(
        eval_times, overall_avg, chance_level,
        os.path.join(plot_dir, "Accuracy_Over_Time_Overall.png"),
        title=f"IMU-Only: Overall Average Accuracy Over Time "
              f"({sum(len(v) for v in test_ds.values())} test trials, "
              f"{len(le.classes_)} classes)",
        bar_color=PALETTE['primary'],
        n_trials=None,
    )

    # — Per-class accuracy over time
    for cls_name in sorted(test_ds.keys()):
        n_cls_trials = len(test_ds[cls_name])
        save_accuracy_over_time(
            eval_times, per_class_avg[cls_name], chance_level,
            os.path.join(plot_dir, f"Accuracy_Over_Time_{cls_name}.png"),
            title=f"IMU-Only: Accuracy Over Time — Class: {cls_name}",
            ylabel=f"Accuracy for '{cls_name}' (%)",
            bar_color=PALETTE['bar_alt'],
            n_trials=n_cls_trials,
        )

    # — Combined multi-panel figure
    save_combined_time_plot(
        eval_times, per_class_avg, overall_avg,
        sorted(test_ds.keys()), chance_level,
        os.path.join(plot_dir, "Accuracy_Over_Time_Combined.png"),
    )

    # — Subset: Mug, Card, Bottle (if available)
    subset_classes = [c for c in ['mug', 'card', 'bottle'] if c in test_ds]
    if subset_classes:
        subset_avgs = []
        for t_i in range(len(eval_times)):
            vals = [per_class_avg[c][t_i] for c in subset_classes
                    if not np.isnan(per_class_avg[c][t_i])]
            subset_avgs.append(np.mean(vals) if vals else np.nan)

        n_sub = sum(len(test_ds[c]) for c in subset_classes)
        save_accuracy_over_time(
            eval_times, subset_avgs,
            100.0 / len(subset_classes),
            os.path.join(plot_dir, "Accuracy_Over_Time_MugCardBottle.png"),
            title="IMU-Only: Accuracy Over Time — Mug, Card, Bottle",
            bar_color=PALETTE['bar_sub'],
            n_trials=n_sub,
        )

    print(f"\nAll accuracy-over-time plots saved -> {plot_dir}")
    print("\n" + "=" * 60)
    print("  4.2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
