# kruskal_cohen_mutual.py
import pandas as pd, numpy as np
from scipy.stats import kruskal
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
ROOT = Path("D:/pipeline_SVM")
F = ROOT/"features"/"features_svm_baseline.csv"
OUT = ROOT/"results"
OUT.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(F, low_memory=False)
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'

# select numeric features only (exclude meta)
meta = {'filepath','label','label_clean','mic_type','experiment','duration','duration_s','wav_path_norm'}
num = df.select_dtypes(include=[np.number]).columns.tolist()
# if num is empty, coerce other feature columns to numeric:
if len(num) == 0:
    potential = [c for c in df.columns if c not in meta]
    df[potential] = df[potential].apply(pd.to_numeric, errors='coerce')
    num = df.select_dtypes(include=[np.number]).columns.tolist()

print("Features numéricas:", len(num))
classes = df[label_col].dropna().unique().tolist()
print("Clases:", classes)

# Kruskal-Wallis
kr = []
for f in num:
    groups = [g.dropna().astype(float).values for _,g in df[[label_col,f]].groupby(label_col)[f]]
    try:
        h,p = kruskal(*groups)
    except Exception:
        h,p = np.nan,np.nan
    kr.append((f, float(h), float(p)))
pd.DataFrame(kr, columns=['feature','H','p']).sort_values('p').to_csv(OUT/"kruskal_features.csv", index=False)

# Cohen's d (pairwise vs majority class)
from math import sqrt
maj = df[label_col].value_counts().idxmax()
cd = []
for f in num:
    a = df[df[label_col]==maj][f].dropna().astype(float)
    for cls in classes:
        if cls==maj: continue
        b = df[df[label_col]==cls][f].dropna().astype(float)
        if len(a)<2 or len(b)<2:
            d = np.nan
        else:
            ma, mb = a.mean(), b.mean()
            sa, sb = a.var(ddof=1), b.var(ddof=1)
            pooled = sqrt(((len(a)-1)*sa + (len(b)-1)*sb) / (len(a)+len(b)-2))
            d = (ma-mb)/pooled if pooled>0 else np.nan
        cd.append((f, cls, d))
pd.DataFrame(cd, columns=['feature','class_vs_major','cohen_d']).to_csv(OUT/"cohen_d_vs_major.csv", index=False)

# Mutual Info (need numeric X and numeric y)
X = df[num].fillna(0)
y = df[label_col].astype('category').cat.codes
mi = mutual_info_classif(X, y, random_state=42)
pd.DataFrame({'feature':X.columns,'mutual_info':mi}).sort_values('mutual_info', ascending=False).to_csv(OUT/"mutual_info_features.csv", index=False)

print("Saved kruskal, cohen_d and mutual_info to:", OUT)
