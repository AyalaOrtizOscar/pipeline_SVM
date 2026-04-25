# pca_inspect.py
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv("D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv")
feat_cols = [c for c in df.columns if c not in ['filepath','label','label_clean','mic_type','experiment','duration']]
X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit_transform(Xs)
out = Path("D:/pipeline_SVM/results/pca_baseline.png")
plt.figure(figsize=(7,6))
labels = df['label_clean'].fillna("NA")
for lab in labels.unique():
    idx = labels==lab
    plt.scatter(pca[idx,0], pca[idx,1], label=str(lab), alpha=0.6, s=12)
plt.legend()
plt.title("PCA 2D - baseline features")
plt.savefig(out, dpi=200); plt.close()
print("Guardado:", out)
