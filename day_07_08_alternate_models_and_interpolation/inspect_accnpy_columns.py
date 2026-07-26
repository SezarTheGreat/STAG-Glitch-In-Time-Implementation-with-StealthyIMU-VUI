import numpy as np
import os
import sys
from scipy.interpolate import interp1d

accnpy_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.accnpy"
acc_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.acc"
gyro_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.gyro"

accnpy_data = np.load(accnpy_path)

def load_sensor(path):
    import csv
    timestamps = []
    x, y, z = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                timestamps.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))
                z.append(float(row[3]))
    return np.array(timestamps), np.vstack([x, y, z])

t_acc, val_acc = load_sensor(acc_path)
t_gyro, val_gyro = load_sensor(gyro_path)

# Resample raw sensors to match accnpy length (3471)
t_target = np.linspace(0, t_acc[-1] - t_acc[0], accnpy_data.shape[1])

acc_x_res = interp1d(t_acc - t_acc[0], val_acc[0, :], kind='cubic')(t_target)
acc_y_res = interp1d(t_acc - t_acc[0], val_acc[1, :], kind='cubic')(t_target)
acc_z_res = interp1d(t_acc - t_acc[0], val_acc[2, :], kind='cubic')(t_target)

# Resample gyro
t_target_gyro = np.linspace(0, t_gyro[-1] - t_gyro[0], accnpy_data.shape[1])
gyro_x_res = interp1d(t_gyro - t_gyro[0], val_gyro[0, :], kind='cubic')(t_target_gyro)
gyro_y_res = interp1d(t_gyro - t_gyro[0], val_gyro[1, :], kind='cubic')(t_target_gyro)
gyro_z_res = interp1d(t_gyro - t_gyro[0], val_gyro[2, :], kind='cubic')(t_target_gyro)

print("Correlation Matrix between accnpy rows and raw sensors:")
print(f"{'accnpy_row':<12} | {'acc_x':<8} | {'acc_y':<8} | {'acc_z':<8} | {'gyro_x':<8} | {'gyro_y':<8} | {'gyro_z':<8}")
print("-" * 75)

raw_signals = [acc_x_res, acc_y_res, acc_z_res, gyro_x_res, gyro_y_res, gyro_z_res]
raw_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

for r in range(4):
    row_data = accnpy_data[r, :]
    corrs = []
    for sig in raw_signals:
        corrs.append(np.corrcoef(row_data, sig)[0, 1])
    print(f"row_{r:<8} | {corrs[0]:.5f} | {corrs[1]:.5f} | {corrs[2]:.5f} | {corrs[3]:.5f} | {corrs[4]:.5f} | {corrs[5]:.5f}")
