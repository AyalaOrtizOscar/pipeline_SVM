# separability_stats.py
import pandas as pd, numpy as np
from scipy.stats import kruskal
from pathlib import Path
ROOT=Path("D:/pipeline_SVM")
df=pd.read_csv(ROOT/"features"/"features_svm_baseline.csv", low_memory=False)
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
features = [c for c in df.columns if c not in ['filepath','label','label_clean','mic_type','experiment','duration','duration_s']]
res=[]
for f in features:
    groups = [g.dropna().astype(float).values for _,g in df[[label_col,f]].groupby(label_col)[f]]
    try:
        h,p = kruskal(*groups)
    except Exception as e:
        h,p = np.nan,np.nan
    res.append((f, p))
res = pd.DataFrame(res, columns=['feature','kruskal_p']).sort_values('kruskal_p')
res.to_csv(ROOT/"results"/"kruskal_features.csv", index=False)
print("Guardado kruskal p-values")
