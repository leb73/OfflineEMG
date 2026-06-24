import os
import glob
import collections
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONSTANTS
# ============================================================================
EMG_FS = 1926.0
WINDOW_MS = 150
STEP_MS = 50
T_START_S = -1.5
T_END_S = 1.5
EVAL_TIMES = np.arange(T_START_S + (WINDOW_MS / 1000.0), T_END_S + (STEP_MS / 1000.0) / 2.0, STEP_MS / 1000.0)

# ============================================================================
# FILTER SETUP
# ============================================================================
def preprocess_emg(data, fs=EMG_FS):
    nyq = fs / 2.0
    
    # 1. High pass filter at 10 Hz (3rd order Butterworth)
    b_hp, a_hp = butter(3, 10.0 / nyq, btype='high')
    data_hp = filtfilt(b_hp, a_hp, data, axis=0)
    
    # 2. Rectification
    data_rect = np.abs(data_hp)
    
    # 3. Low pass filter at 5 Hz (3rd order Butterworth)
    b_lp, a_lp = butter(3, 5.0 / nyq, btype='low')
    data_lp = filtfilt(b_lp, a_lp, data_rect, axis=0)
    
    return data_lp

def extract_pattern_vector(time_arr, emg_arr, t_end):
    t_start = t_end - (WINDOW_MS / 1000.0)
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    window = emg_arr[mask]
    
    expected_samples = int((WINDOW_MS / 1000.0) * EMG_FS)
    if len(window) < expected_samples:
        pad_width = expected_samples - len(window)
        window = np.pad(window, ((0, pad_width), (0, 0)), mode='constant')
    elif len(window) > expected_samples:
        window = window[:expected_samples]
        
    return window.flatten()

# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_preprocess_trials(base_dir):
    dataset = collections.defaultdict(list)
    trial_folders = sorted(glob.glob(os.path.join(base_dir, "Trial_*_*_Short")))
    
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
                
                emg_time_cols = [c for c in df.columns if 'EMG' in c and 'Time' in c]
                emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
                
                if not emg_time_cols or not emg_cols: continue
                
                time_col = emg_time_cols[0]
                time_vals = pd.to_numeric(df[time_col], errors='coerce').values.astype(np.float64)
                valid_idx = ~np.isnan(time_vals)
                time_vals = time_vals[valid_idx]
                
                emg_full = (df[emg_cols].iloc[valid_idx]
                            .apply(pd.to_numeric, errors='coerce')
                            .fillna(0.0).values.astype(np.float64))
                            
                mask = (time_vals >= T_START_S) & (time_vals <= T_END_S)
                time_arr = time_vals[mask]
                emg_arr = emg_full[mask]
                
                if len(time_arr) < 100: continue
                
                emg_preproc = preprocess_emg(emg_arr)
                
                # Baseline correction: find the mean noise of the first 1.4s and remove it
                baseline_mask = time_arr < (T_START_S + 1.4)
                if np.sum(baseline_mask) > 0:
                    baseline = np.mean(emg_preproc[baseline_mask], axis=0)
                    emg_preproc = emg_preproc - baseline
                    
                dataset[cls_label].append((time_arr, emg_preproc))
                
            except Exception as exc:
                print(f"Error loading {csv_path}: {exc}")
                
    return dataset

# ============================================================================
# CASCADED ANN
# ============================================================================
def build_ann(input_dim, output_dim, hidden_neurons=25, lr=0.01):
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim, activation='sigmoid'),
        Dense(output_dim, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# MAIN
# ============================================================================
def main():
    base_dir = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(script_dir, "Offline_Training_Results", f"{timestamp}_4.1_Offline_EMG_Inference")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_trials(base_dir)
    if not dataset:
        print("No valid data found.")
        return
        
    print("Balancing classes...")
    min_trials = min([len(v) for v in dataset.values()])
    for cls in dataset.keys():
        dataset[cls] = random.sample(dataset[cls], min_trials)
        
    train_ds, test_ds = collections.defaultdict(list), collections.defaultdict(list)
    n_train = max(1, int(0.70 * min_trials))
    for cls, trials in dataset.items():
        train_ds[cls] = trials[:n_train]
        test_ds[cls] = trials[n_train:]
        
    classes = sorted(list(dataset.keys()))
    le_task = LabelEncoder()
    le_task.fit(classes)
    
    print("Normalizing and Extracting windows...")
    all_train_emg = []
    for cls, trials in train_ds.items():
        for t_arr, e_arr in trials:
            all_train_emg.append(e_arr)
    all_train_emg = np.vstack(all_train_emg)
    scaler = MaxAbsScaler()
    scaler.fit(all_train_emg)
    
    print("Extracting training windows...")
    X_train_list, Y_train_list = [], []
    for cls, trials in train_ds.items():
        for t_arr, e_arr in trials:
            e_arr_norm = scaler.transform(e_arr)
            for t_end in EVAL_TIMES:
                vec = extract_pattern_vector(t_arr, e_arr_norm, t_end)
                X_train_list.append(vec)
                Y_train_list.append(cls)
                
    X_train = np.array(X_train_list)
    Y_train_enc = le_task.transform(Y_train_list)
    
    print("Step 2.1: Clustering...")
    best_k = 2
    best_score = -1
    max_k = min(len(classes), 5)
    best_labels = None
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"Optimal number of clusters: {best_k} (Silhouette: {best_score:.3f})")
    
    cluster_to_tasks = collections.defaultdict(list)
    task_to_cluster = {}
    
    for task_idx in range(len(classes)):
        task_mask = (Y_train_enc == task_idx)
        if np.any(task_mask):
            task_cluster_labels = best_labels[task_mask]
            most_common_cluster = collections.Counter(task_cluster_labels).most_common(1)[0][0]
            cluster_to_tasks[most_common_cluster].append(task_idx)
            task_to_cluster[task_idx] = most_common_cluster
            
    Y_train_cluster = np.array([task_to_cluster[y] for y in Y_train_enc])
    
    print("Step 2.2: Training Cascaded ANNs...")
    input_dim = X_train.shape[1]
    
    ann1 = build_ann(input_dim, best_k)
    ann1.fit(X_train, Y_train_cluster, epochs=50, batch_size=16, verbose=0)
    
    ann2_dict = {}
    for c_id, tasks in cluster_to_tasks.items():
        if len(tasks) > 1:
            mask = (Y_train_cluster == c_id)
            X_c = X_train[mask]
            Y_c = Y_train_enc[mask]
            
            local_le = LabelEncoder()
            Y_c_local = local_le.fit_transform(Y_c)
            
            ann2 = build_ann(input_dim, len(tasks))
            ann2.fit(X_c, Y_c_local, epochs=50, batch_size=16, verbose=0)
            ann2_dict[c_id] = (ann2, local_le)
            
    print("Evaluating Test Set...")
    overall_acc_history = []
    obj_acc_history = collections.defaultdict(list)
    
    metrics_dir = os.path.join(out_dir, "Metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    for t_end in EVAL_TIMES:
        X_test_t, Y_test_t = [], []
        for cls, trials in test_ds.items():
            for t_arr, e_arr in trials:
                e_arr_norm = scaler.transform(e_arr)
                vec = extract_pattern_vector(t_arr, e_arr_norm, t_end)
                X_test_t.append(vec)
                Y_test_t.append(cls)
                
        X_test_t = np.array(X_test_t)
        Y_test_t_enc = le_task.transform(Y_test_t)
        
        c_preds = np.argmax(ann1.predict(X_test_t, verbose=0), axis=1)
        
        final_preds = np.zeros_like(c_preds)
        for i in range(len(X_test_t)):
            c_id = c_preds[i]
            tasks_in_c = cluster_to_tasks.get(c_id, [])
            
            if len(tasks_in_c) == 1:
                final_preds[i] = tasks_in_c[0]
            elif len(tasks_in_c) > 1 and c_id in ann2_dict:
                ann2, local_le = ann2_dict[c_id]
                vec = X_test_t[i:i+1]
                t_pred_local = np.argmax(ann2.predict(vec, verbose=0), axis=1)
                final_preds[i] = local_le.inverse_transform(t_pred_local)[0]
            else:
                final_preds[i] = 0
                
        acc = accuracy_score(Y_test_t_enc, final_preds) * 100
        overall_acc_history.append(acc)
        
        precision, recall, f1, _ = precision_recall_fscore_support(Y_test_t_enc, final_preds, zero_division=0)
        std_acc = np.std(final_preds == Y_test_t_enc) * 100
        
        df_metrics = pd.DataFrame({
            'Class': le_task.classes_,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100
        })
        df_metrics.loc['macro'] = ['MACRO', precision.mean()*100, recall.mean()*100, f1.mean()*100]
        df_metrics.loc['overall'] = ['OVERALL_ACC', acc, acc, acc]
        df_metrics.loc['sd'] = ['SD', std_acc, std_acc, std_acc]
        
        df_metrics.to_csv(os.path.join(metrics_dir, f"metrics_t_{t_end:.2f}s.csv"), index=False)
        
        cm = confusion_matrix(Y_test_t_enc, final_preds, labels=range(len(classes)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix at t={t_end:.2f}s\nAccuracy: {acc:.1f}%")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, f"cm_t_{t_end:.2f}s.png"), dpi=150)
        plt.close()
        
        for c_idx, c_name in enumerate(classes):
            mask = (Y_test_t_enc == c_idx)
            c_acc = accuracy_score(Y_test_t_enc[mask], final_preds[mask]) * 100 if np.sum(mask) > 0 else 0
            obj_acc_history[c_name].append(c_acc)
            
        print(f"t={t_end:+.2f}s -> Acc: {acc:.1f}%")
        
    print("Generating Accuracy Over Time Plots...")
    plots_dir = os.path.join(out_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(EVAL_TIMES, overall_acc_history, marker='o', linewidth=2, color='black')
    plt.ylim(0, 105)
    plt.title("Overall Accuracy Across Time")
    plt.xlabel("Time relative to motion onset (s)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "Overall_Accuracy_Over_Time.png"), dpi=150)
    plt.close()
    
    for c_name, acc_hist in obj_acc_history.items():
        plt.figure(figsize=(10, 5))
        plt.plot(EVAL_TIMES, acc_hist, marker='o', linewidth=2, color='tab:blue')
        plt.ylim(0, 105)
        plt.title(f"Accuracy Across Time: {c_name}")
        plt.xlabel("Time relative to motion onset (s)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Accuracy_Over_Time_{c_name}.png"), dpi=150)
        plt.close()

    print(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
