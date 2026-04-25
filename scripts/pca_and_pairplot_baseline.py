#!/usr/bin/env python3
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast"); OUT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(FEATURES_CSV, low_memory=False)
label_col="label_fixed"
mask = df[label_col].notna()
df = df.loc[mask]

baseline = ["harmonic_percussive_ratio","centroid_mean","zcr_mean","spectral_flatness_mean",
            "spectral_entropy_mean","onset_rate","duration_s","crest_factor","chroma_std","spectral_contrast_mean"]
available = [f for f in baseline if f in df.columns]
print("Baseline features found:", available)
X = df[available].select_dtypes(include=[np.number]).copy().fillna(df[available].median())
sc = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit_transform(sc)
out = pd.DataFrame({"pc1":pca[:,0], "pc2":pca[:,1], "label": df[label_col].values})
sns.scatterplot(data=out, x="pc1", y="pc2", hue="label", alpha=0.7)
plt.title("PCA sobre baseline")
plt.savefig(OUT_DIR/"pca_baseline.png", dpi=150); plt.close()

# pairplot (muestra 1000 filas máximo)
sample = df.sample(n=min(1000,len(df)), random_state=42)
sns.pairplot(sample[available+ [label_col]], hue=label_col, plot_kws={"alpha":0.5}, diag_kind="kde")
plt.savefig(OUT_DIR/"pairplot_baseline.png", dpi=150)
