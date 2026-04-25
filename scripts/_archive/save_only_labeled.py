# save_only_labeled.py
import pandas as pd

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv"
OUT = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labeled.csv"

df = pd.read_csv(IN, low_memory=False)
print("Total rows in input:", len(df))
df_l = df[df['label_fixed'].notna()].copy()
print("Rows with label_fixed:", len(df_l))
df_l.to_csv(OUT, index=False)
print("Saved:", OUT)
