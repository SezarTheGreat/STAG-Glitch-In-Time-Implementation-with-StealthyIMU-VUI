import pandas as pd
import numpy as np
import os

df = pd.read_csv("projects/stag_original/results/slu_baseline_paper/1235/test-type=direct.csv")
dataset_root = "common/data/StealthyIMU_dataset"

for i in range(5):
    row = df.iloc[i]
    uuid = row['ID']
    wav_path = row['wav']
    base_dir = os.path.dirname(wav_path)
    
    accnpy_path = os.path.join(dataset_root, wav_path.replace('./', '').replace('.wav', '.accnpy'))
    acc_path = os.path.join(dataset_root, base_dir.replace('./', ''), f"{uuid}.acc")
    
    if os.path.exists(accnpy_path) and os.path.exists(acc_path):
        accnpy_data = np.load(accnpy_path)
        print(f"Row {i} - ID: {uuid}")
        print("  CSV Duration:", row['duration'])
        print("  accnpy shape:", accnpy_data.shape)
        # Load raw sensor and print duration
        from scipy.io import wavfile
        rate, wav_data = wavfile.read(os.path.join(dataset_root, wav_path.replace('./', '')))
        print("  WAV duration:", len(wav_data) / rate)
