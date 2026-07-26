import numpy as np
import os
import glob

# Find some .accnpy files
dataset_root = "common/data/StealthyIMU_dataset"
files = glob.glob(os.path.join(dataset_root, "**/*.accnpy"), recursive=True)
if files:
    accnpy_path = files[0]
    data = np.load(accnpy_path)
    print("File path:", accnpy_path)
    print("accnpy shape:", data.shape)
    print("Row 3 statistics: mean =", np.mean(data[3, :]), "std =", np.std(data[3, :]))
    
    # Check if there is a corresponding raw file
    base = os.path.splitext(accnpy_path)[0]
    acc_path = base + ".acc"
    gyro_path = base + ".gyro"
    print("acc exists:", os.path.exists(acc_path))
    print("gyro exists:", os.path.exists(gyro_path))
    if os.path.exists(acc_path):
        with open(acc_path, 'r') as f:
            print("First 3 lines of .acc:")
            for _ in range(3):
                print(f.readline().strip())
else:
    print("No .accnpy files found")
