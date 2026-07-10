import numpy as np
import os
import sys
from scipy.interpolate import interp1d

accnpy_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.accnpy"
acc_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.acc"

accnpy_data = np.load(accnpy_path)

# Load raw sensor
timestamps = []
x, y, z = [], [], []
import csv
with open(acc_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader, None) # skip header
    for row in reader:
        if len(row) >= 4:
            timestamps.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))

t_acc = np.array(timestamps)
val_acc = np.vstack([x, y, z])

# Let's resample raw z to match accnpy length using a uniform 500 Hz grid
t_shifted = t_acc - t_acc[0]
duration = t_shifted[-1] / 1000.0
# The accnpy length is accnpy_data.shape[1]
t_target = np.linspace(0, t_shifted[-1], accnpy_data.shape[1])
z_resampled = interp1d(t_shifted, val_acc[2, :], kind='cubic')(t_target)

# Compute Z-score normalization of raw z
z_norm = (z_resampled - np.mean(z_resampled)) / np.std(z_resampled)
z_accnpy = accnpy_data[3, :]

print("Resampled raw z shape:", z_resampled.shape)
print("accnpy row 3 shape:", z_accnpy.shape)
correlation = np.corrcoef(z_norm, z_accnpy)[0, 1]
print("Correlation between normalized raw z and accnpy row 3:", correlation)

# Let's see if there is a linear relationship: accnpy_z = a * raw_z + b
a, b = np.polyfit(z_resampled, z_accnpy, 1)
print(f"Linear fit: accnpy_z = {a:.6f} * raw_z + {b:.6f}")
print("Gravity (approx 9.8) * a + b =", 9.8 * a + b)
print("Standard deviation of raw z:", np.std(z_resampled))
print("Standard deviation of accnpy z:", np.std(z_accnpy))
