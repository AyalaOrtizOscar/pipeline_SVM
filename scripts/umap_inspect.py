# umap_inspect.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
ROOT="D:/pipeline_SVM"
df=pd.read_csv(ROOT+"/features/features_svm_baseline.csv", low_memory=False)
label='label_clean' if 'label_clean' in df.columns else 'label'
feat = df.select_dtypes(include='number').fillna(0)
X = StandardScaler().fit_transform(feat)
emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(6,6))
for l in df[label].unique():
    idx = df[label]==l
    plt.scatter(emb[idx,0], emb[idx,1], label=str(l), s=8, alpha=0.7)
plt.legend()
plt.title("UMAP features")
plt.savefig("D:/pipeline_SVM/results/umap.png", dpi=200)
