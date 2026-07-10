import pandas as pd
df = pd.read_csv("projects/stag_original/results/slu_baseline_paper/1235/test-type=direct.csv")
df_sorted = df.sort_values(by="duration")
print("Durations of first 10 sorted samples:")
print(df_sorted["duration"].head(10).tolist())
print("\nTranscripts of first 10 sorted samples:")
print(df_sorted["transcript"].head(10).tolist())
print("\nDurations of samples 90 to 100:")
print(df_sorted["duration"].iloc[90:100].tolist())
