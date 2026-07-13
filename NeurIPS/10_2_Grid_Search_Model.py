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
from scipy.signal import stft
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import datetime
import gc
import itertools

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
print(f"Using device: {device}")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_DIR = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials_shifted"
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Offline_Training_Results", timestamp + "_Model_Optimisation")

classes = [
    "Trial_2_Ball_Short",
    "Trial_3_VPen_Short",
    "Trial_4_HPen_Short",
    "Trial_5_Bottle_Short",
    "Trial_6_Mug_Short",
    "Trial_7_Card_Short"
]

class_labels_short = ["Ball", "VPen", "HPen", "Bottle", "Mug", "Card"]
num_classes = len(classes) + 1  # +1 for Rest class

EMG_FS = 1925.0
NOISE_START  = -1.5
NOISE_END    = -0.6
EMD_START    = -0.6
EMD_END      =  0.0
MOVE_END     =  1.5
WINDOW_S = 0.2
STEP_S   = 0.05
N_SENSORS = 8
N_CHANNELS = N_SENSORS

# ========================================================================
# SPECTROGRAM PREPROCESSING (Runs Once)
# ========================================================================
def preprocess_stft(X_window, fs, nperseg, noverlap, max_freq):
    channels = X_window.shape[1]
    num_freqs = nperseg // 2
    f_tmp, t_tmp, Zxx_tmp = stft(X_window[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    num_frames = len(t_tmp)
    mag_out = np.zeros((num_freqs, num_frames, channels), dtype=np.float32)
    
    for c in range(channels):
        f, t, Zxx = stft(X_window[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        mag = mag[:num_freqs, :]
        f = f[:num_freqs]
        mag[f > max_freq, :] = 0.0
        mag_out[:, :, c] = mag
        
    return mag_out, num_freqs, num_frames

def load_and_window_data(base_dir, nperseg, noverlap, max_freq):
    print(f"Extracting EMG windows ONCE (nperseg={nperseg}, noverlap={noverlap}, max_freq={max_freq})...")
    X_train_raw, y_train_raw, y_train_move_raw, t_train = [], [], [], []
    X_test_raw,  y_test_raw,  y_test_move_raw,  t_test  = [], [], [], []

    raw_stfts_by_class = {idx: [] for idx in range(len(classes))}
    raw_times_by_class = {idx: [] for idx in range(len(classes))}

    class_csv_map = {}
    min_trials = 999999
    for class_name in classes:
        files = sorted(glob.glob(os.path.join(base_dir, class_name, "*.csv")))
        class_csv_map[class_name] = files
        if len(files) < min_trials:
            min_trials = len(files)

    num_freqs_detected = None
    num_frames_detected = None

    for class_idx, class_name in enumerate(classes):
        csv_files = class_csv_map[class_name][:min_trials]
        if len(csv_files) == 0: continue
        n_train = int(min_trials * 0.8)

        for file_idx, csv_path in enumerate(csv_files):
            is_train = (file_idx < n_train)
            with open(csv_path, 'r') as f:
                for _ in range(5): f.readline()
                num_cols = len(f.readline().split(','))

            df = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
            if len(df) <= 2: continue
            df = df.iloc[2:].reset_index(drop=True)

            emg_time_cols = [c for c in df.columns if 'Time' in c and 'ACC' not in c]
            if not emg_time_cols: continue
            time_vals = pd.to_numeric(df[emg_time_cols[0]], errors='coerce').values

            emg_cols = [c for c in df.columns if 'EMG' in c and '(mV)' in c]
            emg_data = df[emg_cols].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0).values

            req_len = int(WINDOW_S * EMG_FS)
            target_len = max(req_len, nperseg)
            trial_stfts, trial_labels, trial_move_labels, trial_times = [], [], [], []

            t_end = NOISE_START + WINDOW_S
            while t_end <= MOVE_END:
                t_start = t_end - WINDOW_S
                mask = (time_vals >= t_start) & (time_vals <= t_end)
                win = emg_data[mask]

                if len(win) == 0:
                    win = np.zeros((target_len, N_CHANNELS))
                elif len(win) < target_len:
                    pad = target_len - len(win)
                    win = np.pad(win, ((0, pad), (0, 0)), mode='constant')
                elif len(win) > target_len:
                    win = win[:target_len]

                stft_mag, freq_count, frame_count = preprocess_stft(win, EMG_FS, nperseg, noverlap, max_freq)
                
                if num_freqs_detected is None:
                    num_freqs_detected = freq_count
                    num_frames_detected = frame_count
                
                label = len(classes) if t_end <= NOISE_END else class_idx
                trial_stfts.append(stft_mag)
                trial_labels.append(label)
                trial_move_labels.append(class_idx)
                trial_times.append(t_end)
                t_end += STEP_S

            trial_stfts = np.array(trial_stfts)
            trial_times = np.array(trial_times)
            noise_indices = [i for i, t in enumerate(trial_times) if t <= NOISE_END]
            if noise_indices:
                base_stft_profile = np.mean(trial_stfts[noise_indices], axis=(0, 2), keepdims=True)
                trial_stfts = np.maximum(trial_stfts - base_stft_profile, 0.0)

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
        raw_stfts_by_class, raw_times_by_class, num_freqs_detected, num_frames_detected
    )

# ========================================================================
# DYNAMIC MODEL ARCHITECTURE (EMGNet)
# ========================================================================
class EMGNetClassifier(nn.Module):
    def __init__(self, num_channels=8, num_frames=9, num_frequencies=128, num_classes=7,
                 base_filters=32, dropout_rate=0.0, kernel_size=3):
        super(EMGNetClassifier, self).__init__()
        self.num_frames = num_frames
        
        f1 = base_filters
        f2 = base_filters * 4
        f3 = base_filters * 8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, f1, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(f1, f2, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            nn.Conv2d(f2, f3, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(f3, f3, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(max(1, num_frames // 4), 1), mode='bilinear', align_corners=False),
            
            nn.Conv2d(f3, f2, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(max(1, num_frames // 2), 1), mode='bilinear', align_corners=False),
            
            nn.Conv2d(f2, f1, kernel_size=(kernel_size, kernel_size), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=False),
            nn.Upsample(size=(num_frames, 1), mode='bilinear', align_corners=False)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(f1 * num_frames, num_classes)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ========================================================================
# GRID SEARCH EXECUTION
# ========================================================================
def run_model_experiment(model_config, X_train, y_train, X_test, y_test, t_test, num_freqs, num_frames, true_labels, results_dir):
    print(f"\n{'='*60}\nEvaluating Config: {model_config}\n{'='*60}")
    os.makedirs(results_dir, exist_ok=True)
    
    X_train_t = torch.from_numpy(X_train).permute(0, 3, 2, 1).float()
    X_test_t = torch.from_numpy(X_test).permute(0, 3, 2, 1).float()
    y_train_t = torch.from_numpy(y_train).long()
    y_test_t = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = EMGNetClassifier(
        num_channels=8, 
        num_frames=num_frames, 
        num_frequencies=num_freqs, 
        num_classes=num_classes,
        base_filters=model_config['base_filters'],
        dropout_rate=model_config['dropout'],
        kernel_size=model_config['kernel_size']
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    
    epochs = 10
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        print(f"Epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc*100:.2f}% - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc*100:.2f}%")
    
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pt"))
    
    # EVALUATION
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    pred_labels = np.argmax(preds, axis=1)
    
    noise_mask = (t_test >= NOISE_START) & (t_test <= NOISE_END)
    emd_mask = (t_test > NOISE_END) & (t_test <= EMD_END)
    move_mask = (t_test > EMD_END) & (t_test <= MOVE_END)
    
    avg_overall = np.mean(true_labels == pred_labels) * 100.0
    avg_noise = np.mean(true_labels[noise_mask] == pred_labels[noise_mask]) * 100.0 if np.any(noise_mask) else 0.0
    avg_emd = np.mean(true_labels[emd_mask] == pred_labels[emd_mask]) * 100.0 if np.any(emd_mask) else 0.0
    avg_move = np.mean(true_labels[move_mask] == pred_labels[move_mask]) * 100.0 if np.any(move_mask) else 0.0

    summary_text = (
        f"{'='*60}\n"
        f"MODEL HYPERPARAMETER CLASSIFICATION SUMMARY\n"
        f"Model: EMGNet Classifier (PyTorch)\n"
        f"Grid Config: {model_config}\n"
        f"{'='*60}\n"
        f"Overall Average Accuracy:          {avg_overall:.2f}%\n"
        f"  Noise Phase   ({NOISE_START}s to {NOISE_END}s): {avg_noise:.2f}%\n"
        f"  EMD Phase     ({EMD_START}s to {EMD_END}s):  {avg_emd:.2f}%\n"
        f"  Movement Phase ({EMD_END}s to {MOVE_END}s):  {avg_move:.2f}%\n"
        f"{'='*60}\n"
    )
    print(f"\n{summary_text}")
    with open(os.path.join(results_dir, "accuracy_summary.txt"), 'w') as f:
        f.write(summary_text)

if __name__ == "__main__":
    # Fixed Spectrogram Parameters (The Best for EMD Phase)
    NPERSEG = 512
    NOVERLAP = 496
    MAX_FREQ = 962.0
    
    print(f"Loading data once with optimal spectrogram settings (nperseg={NPERSEG}, max_freq={MAX_FREQ})...")
    (X_train_raw, y_train_raw, y_train_move, t_train, 
     X_test_raw, y_test_raw, y_test_move, t_test, 
     _, _, num_freqs, num_frames) = load_and_window_data(DATA_DIR, NPERSEG, NOVERLAP, MAX_FREQ)
     
    print(f"Data Loaded: {len(X_train_raw)} train, {len(X_test_raw)} test.")
    
    if len(X_train_raw) == 0:
        print("No data found! Exiting.")
        exit(1)
    
    # Normalize Data Once
    mean_val = np.mean(X_train_raw, axis=(0, 2), keepdims=True, dtype=np.float64)
    std_val = np.std(X_train_raw, axis=(0, 2), keepdims=True, dtype=np.float64)
    X_train = (X_train_raw - mean_val) / (std_val + 1e-8)
    X_test = (X_test_raw - mean_val) / (std_val + 1e-8)
    
    # Model Hyperparameter Grid (2 x 3 x 2 x 2 = 24 configs)
    base_filters_params = [16, 32]
    dropout_params = [0.0, 0.3, 0.5]
    learning_rate_params = [0.001, 0.0005]
    kernel_size_params = [3, 5]
    
    grid_configs = []
    for bf, drop, lr, ks in itertools.product(base_filters_params, dropout_params, learning_rate_params, kernel_size_params):
        grid_configs.append({
            "base_filters": bf,
            "dropout": drop,
            "learning_rate": lr,
            "kernel_size": ks
        })
        
    print(f"Starting Model Grid Search with {len(grid_configs)} configurations.")
    
    for config in grid_configs:
        config_name = f"filters{config['base_filters']}_drop{config['dropout']}_lr{config['learning_rate']}_k{config['kernel_size']}"
        config_dir = os.path.join(BASE_RESULTS_DIR, config_name)
        
        run_model_experiment(
            model_config=config, 
            X_train=X_train, 
            y_train=y_train_raw, 
            X_test=X_test, 
            y_test=y_test_raw, 
            t_test=t_test, 
            num_freqs=num_freqs, 
            num_frames=num_frames, 
            true_labels=y_test_move, 
            results_dir=config_dir
        )
        
    print(f"\nAll model configurations completed. Results saved to: {BASE_RESULTS_DIR}")
