# plot_feature_distributions.py
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
ROOT = Path("D:/pipeline_SVM")
F = ROOT/"features"/"features_svm_baseline.csv"
df = pd.read_csv(F, low_memory=False)
# usar label_clean si existe
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
features = ['centroid_mean','zcr','spectral_flatness_mean','spectral_entropy_mean',
            'onset_rate','duration_s','crest_factor','chroma_std','spectral_contrast_mean','harmonic_percussive_ratio']
features = [f for f in features if f in df.columns]
for feat in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=label_col, y=feat, data=df)
    plt.title(feat)
    plt.tight_layout()
    plt.savefig(ROOT/"results"/f"box_{feat}.png", dpi=200)
    plt.close()
print("Guardadas boxplots en:", ROOT/"results")
