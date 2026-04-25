# 1) cargar modelo y ver select__k features
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

model = joblib.load("D:/pipeline_SVM/results/svm_final_fast/best_model.joblib")
# model es un Pipeline: ['scaler','select','clf']
k = model.named_steps['select'].k  # número elegido
scores = model.named_steps['select'].scores_  # si existe
mask = model.named_steps['select'].get_support()
print("k:", k)
# obtener nombres de features (de tu CSV original)
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv", low_memory=False)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected = [col for col, s in zip(num_cols, mask) if s]
print("Selected features:", selected)


