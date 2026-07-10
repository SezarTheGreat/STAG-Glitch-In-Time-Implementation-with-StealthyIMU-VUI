import pandas as pd
df = pd.read_csv("projects/stag_original/results/slu_baseline_paper/1235/test-type=direct.csv")
print(df.columns)
print("Row 0 ID:", df.ID[0])
print("Row 0 duration:", df.duration[0])
print("Row 0 wav:", df.wav[0])

