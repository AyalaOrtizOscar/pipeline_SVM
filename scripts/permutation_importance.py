#!/usr/bin/env python3
import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("D:/pipeline_SVM/results/svm_final_fast/best_model.joblib")
FEATURES_CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast"); OUT_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(FEATURES_CSV, low_memory=False)
label_col="label_fixed"
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col].astype(str).copy()
mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
X = X.loc[mask]; y = y.loc[mask]
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

print("Computing permutation importance (this may take some time)...")
res = permutation_importance(model, X_hold, y_hold, n_repeats=30, random_state=42, n_jobs=4, scoring='f1_macro')
df_imp = pd.DataFrame({
    "feature": X_hold.columns,
    "importance_mean": res.importances_mean,
    "importance_std": res.importances_std
}).sort_values("importance_mean", ascending=False)
df_imp.to_csv(OUT_DIR/"permutation_importances.csv", index=False)
print("Saved permutation importances to", OUT_DIR/"permutation_importances.csv")
