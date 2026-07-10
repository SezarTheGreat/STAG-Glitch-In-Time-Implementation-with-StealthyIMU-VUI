import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import os

accnpy_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.accnpy"
acc_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.acc"

accnpy_data = np.load(accnpy_path)

def load_sensor(path):
    import csv
    timestamps = []
    z = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                timestamps.append(float(row[0]))
                z.append(float(row[3]))
    return np.array(timestamps), np.array(z)

t_acc, val_acc_z = load_sensor(acc_path)
t_target = np.linspace(0, t_acc[-1] - t_acc[0], accnpy_data.shape[1])
z_res = interp1d(t_acc - t_acc[0], val_acc_z, kind='cubic')(t_target)

# Try high-pass filtering at different cutoffs (e.g., 20Hz, 50Hz, 80Hz)
for cutoff in [5.0, 20.0, 50.0, 80.0]:
    # 500 Hz target sampling rate
    sos = signal.butter(4, cutoff, 'highpass', fs=500.0, output='sos')
    z_filtered = signal.sosfiltfilt(sos, z_res)
    corr = np.corrcoef(z_filtered, accnpy_data[3, :])[0, 1]
    print(f"High-pass cutoff {cutoff} Hz - Correlation: {corr:.5f}")
    
# Try band-pass filtering (e.g. 50-250 Hz)
sos_bp = signal.butter(4, [50.0, 240.0], 'bandpass', fs=500.0, output='sos')
z_bp = signal.sosfiltfilt(sos_bp, z_res)
corr_bp = np.corrcoef(z_bp, accnpy_data[3, :])[0, 1]
print(f"Band-pass 50-240 Hz - Correlation: {corr_bp:.5f}")
