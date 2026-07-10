import numpy as np
import os
import sys

sys.path.append("projects/stag_original")
from src.pipeline.dataset import load_raw_sensor

accnpy_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.accnpy"
acc_path = "common/data/StealthyIMU_dataset/data/cleanair/0051f94a-fb04-3059-8352-cfa95a79e151/0051f94a-fb04-3059-8352-cfa95a79e151.acc"

accnpy_data = np.load(accnpy_path)
t_acc, val_acc = load_raw_sensor(acc_path)

print("accnpy shape:", accnpy_data.shape)
print("raw val_acc shape:", val_acc.shape)
print("t_acc min/max:", t_acc[0], t_acc[-1], "length:", len(t_acc))
print("First 5 values of row 3 in accnpy:")
print(accnpy_data[3, :5])
print("First 5 values of acc_z in raw val_acc:")
print(val_acc[2, :5])
print("mean of raw acc_z:", np.mean(val_acc[2, :]))
print("std of raw acc_z:", np.std(val_acc[2, :]))
