# per_mic_confusion.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
ROOT=Path("D:/pipeline_SVM")
preds = pd.read_csv(ROOT/"results"/"svm_run_20250927_202913"/"holdout_predictions.csv")
# si no están las predicciones, ajusta path
print(preds.groupby('group').size())
for mic, g in preds.groupby('group'):
    cm = confusion_matrix(g['true_label'], g['pred_label'], labels=sorted(preds['true_label'].unique()))
    print("Group:", mic)
    print(pd.DataFrame(cm, index=sorted(preds['true_label'].unique()), columns=sorted(preds['true_label'].unique())))
