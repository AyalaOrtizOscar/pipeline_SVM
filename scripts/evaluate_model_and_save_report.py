#!/usr/bin/env python3
import joblib, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

MODEL_PATH = Path("D:/pipeline_SVM/results/svm_final_fast/best_model.joblib")
FEATURES_CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Cargando modelo...", MODEL_PATH)
model = joblib.load(MODEL_PATH)
print("Cargando datos...", FEATURES_CSV)
df = pd.read_csv(FEATURES_CSV, low_memory=False)
label_col = "label_fixed"
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col].astype(str).copy()
mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
X = X.loc[mask]; y = y.loc[mask]

X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
y_pred = model.predict(X_hold)

# classification report
report_txt = classification_report(y_hold, y_pred, zero_division=0)
report_dict = classification_report(y_hold, y_pred, output_dict=True, zero_division=0)
print(report_txt)
with open(OUT_DIR/"holdout_classification_report.txt","w",encoding="utf-8") as f: f.write(report_txt)
with open(OUT_DIR/"holdout_classification_report.json","w",encoding="utf-8") as f: json.dump(report_dict,f,indent=2,ensure_ascii=False)

# confusion matrix
labels = list(model.classes_)
cm = confusion_matrix(y_hold, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(OUT_DIR/"holdout_confusion_matrix.csv", index=True)

# PNG
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.yticks(range(len(labels)), labels)
th = cm.max()/2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>th else 'black')
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(OUT_DIR/"holdout_confusion_matrix.png", dpi=150)
plt.close()

# save predictions
preds_df = pd.DataFrame({"index": X_hold.index, "true": y_hold.values, "pred": y_pred})
preds_df.to_csv(OUT_DIR/"holdout_predictions.csv", index=False)
print("Saved outputs in", OUT_DIR.resolve())
