#!/usr/bin/env python3
import joblib, pandas as pd, numpy as np, shap
from pathlib import Path
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

# take a manageable sample for SHAP
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
sample = X_hold.sample(n=min(500, len(X_hold)), random_state=42)

print("Model type:", type(model.named_steps['clf'] if 'clf' in model.named_steps else model))
# choose explainer
try:
    clf = model.named_steps['clf']
except Exception:
    clf = model

if hasattr(clf, "predict_proba") and hasattr(clf, "tree_") or "RandomForest" in str(type(clf)):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)
    # shap_values: list (one per class) or array
    # create summary table for absolute mean
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
else:
    print("Falling back to KernelExplainer (may be slow). Using background sample of size 200.")
    background = X_train.sample(n=min(200,len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x), background)
    shap_values = explainer.shap_values(sample, nsamples=200)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

feat_names = sample.columns.tolist()
shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(OUT_DIR/"shap_mean_abs.csv", index=False)
print("Saved SHAP summary to", OUT_DIR/"shap_mean_abs.csv")
