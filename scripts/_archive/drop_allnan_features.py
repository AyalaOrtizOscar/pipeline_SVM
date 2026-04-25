# drop_allnan_features.py
import pandas as pd
from pathlib import Path
F = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")  # ajusta al path final
OUT = F.with_name(F.stem + ".pruned.csv")
df = pd.read_csv(F, low_memory=False)
n_before = df.shape[1]
allnan = [c for c in df.columns if df[c].isna().all()]
print("Columns all-NaN (will drop):", allnan)
df = df.drop(columns=allnan)
df.to_csv(OUT, index=False)
print("Saved pruned file:", OUT, "cols before:", n_before, "after:", df.shape[1])
