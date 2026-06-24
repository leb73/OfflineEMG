import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

RAW_DATA_DIR = r"C:\Users\Lucy\Desktop\OfflineEMG\Dan Testing"
OUT_DIR = r"C:\Users\Lucy\Desktop\OfflineEMG\extracted_trials"

os.makedirs(OUT_DIR, exist_ok=True)

# Blocks to process: Trials 2 to 7
trials = [
    ("Trial_2_Ball_Short.csv", 30),
    ("Trial_3_VPen_Short.csv", 28),
    ("Trial_4_HPen_Short.csv", 30),
    ("Trial_5_Bottle_Short.csv", 34),
    ("Trial_6_Mug_Short.csv", 29),
    ("Trial_7_Card_Short.csv", 30)
]

class DraggableLines:
    def __init__(self, ax_top, ax_bot, x_positions):
        self.ax_top = ax_top
        self.ax_bot = ax_bot
        self.figure = ax_top.figure
        self.lines_top = []
        self.lines_bot = []
        for x in x_positions:
            self._add_line(x)
            
        self.selected_idx = None
        
        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def _add_line(self, x):
        l_top = self.ax_top.axvline(x, color='r', picker=5, linewidth=2)
        l_bot = self.ax_bot.axvline(x, color='r', picker=5, linewidth=2)
        self.lines_top.append(l_top)
        self.lines_bot.append(l_bot)
        
    def _remove_line(self, idx):
        self.lines_top[idx].remove()
        self.lines_bot[idx].remove()
        del self.lines_top[idx]
        del self.lines_bot[idx]
        
    def on_press(self, event):
        if event.inaxes not in (self.ax_top, self.ax_bot):
            return
        # Left double click on empty space to add a line
        if event.button == 1 and event.dblclick:
            self._add_line(event.xdata)
            self.figure.canvas.draw_idle()

    def on_pick(self, event):
        # Right click on a line to remove it
        if event.mouseevent.button == 3:
            if event.artist in self.lines_top:
                idx = self.lines_top.index(event.artist)
                self._remove_line(idx)
                self.figure.canvas.draw_idle()
            elif event.artist in self.lines_bot:
                idx = self.lines_bot.index(event.artist)
                self._remove_line(idx)
                self.figure.canvas.draw_idle()
            return
            
        # Left click to drag
        if event.mouseevent.button == 1:
            if event.artist in self.lines_top:
                self.selected_idx = self.lines_top.index(event.artist)
            elif event.artist in self.lines_bot:
                self.selected_idx = self.lines_bot.index(event.artist)
            
    def on_motion(self, event):
        if self.selected_idx is None:
            return
        if event.inaxes not in (self.ax_top, self.ax_bot):
            return
        
        x = event.xdata
        self.lines_top[self.selected_idx].set_xdata([x, x])
        self.lines_bot[self.selected_idx].set_xdata([x, x])
        self.figure.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1:
            self.selected_idx = None
        
    def get_positions(self):
        return sorted([line.get_xdata()[0] for line in self.lines_bot])


def process_block(csv_filename, expected_counts):
    csv_path = os.path.join(RAW_DATA_DIR, csv_filename)
    trial_name = csv_filename.replace('.csv', '')
    print(f"\nProcessing {trial_name}...")
    
    # Read headers
    with open(csv_path, 'r') as f:
        header_lines = [f.readline() for _ in range(7)]
        cols = header_lines[5].strip().split(',')
        
    num_cols = len(cols)
    print("Loading data...")
    df_full = pd.read_csv(csv_path, skiprows=5, usecols=range(num_cols), low_memory=False)
    df_data = df_full.iloc[2:].reset_index(drop=True)
    
    # Pre-calculate sampling rates and group columns
    col_groups = []
    i = 0
    while i < len(df_full.columns):
        col_name = df_full.columns[i]
        # Ignore last empty columns if any
        if 'Unnamed' in col_name and i > 63:
            i += 1
            continue
            
        if 'Time' in str(col_name):
            data_cols = []
            j = i + 1
            while j < len(df_full.columns) and 'Time' not in str(df_full.columns[j]) and ('Unnamed' not in str(df_full.columns[j]) or j <= 63):
                data_cols.append(j)
                j += 1
            col_groups.append({
                'time_col': i,
                'data_cols': data_cols
            })
            i = j
        else:
            i += 1
            
    print("Converting all columns to numeric... This may take a minute.")
    for c in df_full.columns:
        if 'Unnamed' in c and df_full.columns.get_loc(c) > 63:
            continue
        df_data[c] = pd.to_numeric(df_data[c], errors='coerce')

    # Outer Shoulder is 2nd sensor.
    # col 8: EMG Time, col 9: EMG
    # col 10: ACC X Time, 11: ACC X, 12: ACC Y Time, 13: ACC Y, 14: ACC Z Time, 15: ACC Z
    emg_col_name = df_full.columns[9]
    t_emg_col_name = df_full.columns[8]
    
    acc_df = df_data.iloc[:, [10, 11, 13, 15]].dropna().reset_index(drop=True)
    
    time_s = acc_df.iloc[:, 0].values
    acc_x = acc_df.iloc[:, 1].values
    acc_y = acc_df.iloc[:, 2].values
    acc_z = acc_df.iloc[:, 3].values
    
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    dt = np.diff(time_s)
    dt[dt == 0] = np.nan
    jerk_mag = np.abs(np.diff(acc_mag) / dt)
    jerk_mag = np.insert(jerk_mag, 0, 0)
    
    emg_df = df_data.iloc[:, [8, 9]].dropna().reset_index(drop=True)
    t_emg = emg_df.iloc[:, 0].values
    emg_mag = emg_df.iloc[:, 1].values

    print("Finding peaks...")
    # Find all local maxima spaced by ~3 seconds (222 samples)
    peaks, _ = find_peaks(jerk_mag, distance=222)
    
    # Sort by jerk magnitude and strictly take the top expected_counts peaks
    if len(peaks) > expected_counts:
        peak_heights = jerk_mag[peaks]
        top_peaks_indices = np.argsort(peak_heights)[-expected_counts:]
        peaks = sorted(peaks[top_peaks_indices])
    elif len(peaks) < expected_counts:
        print(f"WARNING: Only found {len(peaks)} distinct movements! Expected {expected_counts}.")
        # To guarantee the exact number of lines, we fall back to a smaller distance to grab more peaks
        # Or you can manually add lines. The script will output what it found.
        # But let's try finding peaks with a very low distance just to force the count
        fallback_peaks, _ = find_peaks(jerk_mag, distance=74) # 1 second distance
        if len(fallback_peaks) > expected_counts:
            peak_heights = jerk_mag[fallback_peaks]
            top_peaks_indices = np.argsort(peak_heights)[-expected_counts:]
            peaks = sorted(fallback_peaks[top_peaks_indices])
        else:
            peaks = fallback_peaks
    
    initial_starts = [time_s[p] for p in peaks]
    print(f"Found {len(initial_starts)} initial peaks. Expected: {expected_counts}.")
    
    # ------------------------------------------------------------------------
    # FULL-RECORDING INTERACTIVE UI
    # ------------------------------------------------------------------------
    fig, (ax_emg, ax_acc) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    plt.subplots_adjust(bottom=0.15)
    
    ax_emg.plot(t_emg, emg_mag, label='Outer Shoulder EMG', color='k', alpha=0.7)
    ax_emg.set_ylabel("EMG (mV)")
    ax_emg.set_title(f"Full Recording: {trial_name} - {expected_counts} Expected Movements")
    ax_emg.legend(loc='upper right')
    
    ax_acc.plot(time_s, jerk_mag, label='Jerk Magnitude', color='orange')
    ax_acc.axhline(4, color='green', linestyle='--', label='Initial Threshold=4')
    ax_acc.set_xlabel("Time (s)")
    ax_acc.set_ylabel("Jerk Magnitude")
    
    # Create secondary y-axis for ACC to increase its scale
    ax_acc2 = ax_acc.twinx()
    ax_acc2.plot(time_s, acc_mag, label='ACC Magnitude', color='blue', alpha=0.5)
    ax_acc2.set_ylabel("ACC (G)", color='blue')
    
    # Combine legends
    lines_1, labels_1 = ax_acc.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc2.get_legend_handles_labels()
    ax_acc.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Initialize draggable lines
    draggable_lines = DraggableLines(ax_emg, ax_acc, initial_starts)
    
    # Button
    accept_ax = plt.axes([0.45, 0.02, 0.15, 0.06])
    accept_button = Button(accept_ax, 'Extract & Continue')
    
    final_positions = []
    
    def on_accept(event):
        positions = draggable_lines.get_positions()
        final_positions.extend(positions)
        plt.close(fig)
        
    accept_button.on_clicked(on_accept)
    
    print("\n>>> Interactive Plot opened! Drag the red vertical lines to adjust start times.")
    print(">>> ADD LINE: Double Left-Click")
    print(">>> REMOVE LINE: Right-Click on an existing line")
    print(">>> Use Matplotlib's Zoom/Pan tools to navigate.")
    print(">>> Click 'Extract & Continue' when finished.\n")
    plt.show(block=True)
    
    # ------------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------------
    if not final_positions:
        print("No positions captured. Skipping block.")
        return
        
    trial_out_dir = os.path.join(OUT_DIR, trial_name)
    os.makedirs(trial_out_dir, exist_ok=True)
    
    extracted_data_list = []
    
    print("Extracting trial windows based on final marker positions...")
    for i, t0 in enumerate(final_positions):
        tmin = t0 - 1.5
        tmax = t0 + 1.5
        
        new_cols = {}
        max_len = 0
        
        for group in col_groups:
            t_col_idx = group['time_col']
            t_col_name = df_full.columns[t_col_idx]
            t_series = df_data.iloc[:, t_col_idx].values
            
            mask_ext = (t_series >= tmin) & (t_series <= tmax)
            valid_times = t_series[mask_ext] - t0
            
            new_cols[t_col_name] = valid_times
            max_len = max(max_len, len(valid_times))
            
            for d_col_idx in group['data_cols']:
                d_col_name = df_full.columns[d_col_idx]
                d_series = df_data.iloc[:, d_col_idx].values
                new_cols[d_col_name] = d_series[mask_ext]
                
        padded_cols = {}
        for k, v in new_cols.items():
            if len(v) < max_len:
                padded_cols[k] = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            else:
                padded_cols[k] = v
                
        extracted_df = pd.DataFrame(padded_cols)
        # reorder based on original columns
        final_cols = [c for c in df_full.columns if c in extracted_df.columns]
        extracted_df = extracted_df[final_cols]
        
        mov_csv_path = os.path.join(trial_out_dir, f"movement_{i+1}.csv")
        with open(mov_csv_path, 'w') as f:
            for line in header_lines[:5]:
                f.write(line)
        
        with open(mov_csv_path, 'a') as f:
            f.write(','.join(final_cols) + '\n')
            f.write(','.join([str(df_full.iloc[0][c]) for c in final_cols]) + '\n')
            f.write(','.join([str(df_full.iloc[1][c]) for c in final_cols]) + '\n')
            
        extracted_df.to_csv(mov_csv_path, mode='a', header=False, index=False)
        extracted_data_list.append(extracted_df)
        
    print(f"Extracted {len(extracted_data_list)} movements for {trial_name}.")
    
    # ------------------------------------------------------------------------
    # PLOTTING OVERLAYS
    # ------------------------------------------------------------------------
    print("Generating stacked plots...")
    if len(extracted_data_list) > 0:
        fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
        fig_emg, ax_emg = plt.subplots(figsize=(12, 6))
        
        for mov_df in extracted_data_list:
            try:
                # Outer Shoulder ACC
                t_acc_col = df_full.columns[10]
                acc_x_col = df_full.columns[11]
                acc_y_col = df_full.columns[13]
                acc_z_col = df_full.columns[15]
                
                t_acc = mov_df[t_acc_col].values
                ax_acc.plot(t_acc, mov_df[acc_x_col].values, color='r', alpha=0.3)
                ax_acc.plot(t_acc, mov_df[acc_y_col].values, color='g', alpha=0.3)
                ax_acc.plot(t_acc, mov_df[acc_z_col].values, color='b', alpha=0.3)
            except Exception as e:
                pass
            
            try:
                # Outer Shoulder EMG
                t_emg_col = df_full.columns[8]
                emg_col = df_full.columns[9]
                
                t_emg = mov_df[t_emg_col].values
                ax_emg.plot(t_emg, mov_df[emg_col].values, color='k', alpha=0.3)
            except Exception as e:
                pass
                
        ax_acc.set_title(f"Stacked Outer Shoulder ACC - {trial_name}")
        ax_acc.set_xlabel("Time (s)")
        ax_acc.set_ylabel("ACC (G)")
        fig_acc.savefig(os.path.join(trial_out_dir, f"stacked_ACC_{trial_name}.png"))
        plt.close(fig_acc)
        
        ax_emg.set_title(f"Stacked Outer Shoulder EMG - {trial_name}")
        ax_emg.set_xlabel("Time (s)")
        ax_emg.set_ylabel("EMG (mV)")
        fig_emg.savefig(os.path.join(trial_out_dir, f"stacked_EMG_{trial_name}.png"))
        plt.close(fig_emg)

if __name__ == "__main__":
    for csv_filename, expected in trials:
        process_block(csv_filename, expected)
    print("All trials processed!")
