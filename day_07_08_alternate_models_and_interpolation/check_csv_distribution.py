import pandas as pd
df = pd.read_csv("day_04_05_stag_recreation/results/slu_baseline_paper/1235/test-type=direct.csv")
print("Total rows in test set:", len(df))
print(df['wav'].apply(lambda x: x.split('/')[2]).value_counts())
