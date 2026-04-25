import pandas as pd
F = "D:/pipeline_SVM/features/features_svm_baseline.csv"
df = pd.read_csv(F, low_memory=False)
# usa label_clean si existe
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
df[label_col] = df[label_col].astype(str).str.strip().replace({'nan':'', 'None':'', 'NoneType':''})
valid = ~df[label_col].isna() & (df[label_col] != '') & (df[label_col].str.lower() != 'nan')
print("Filas totales:", len(df), "filas válidas:", valid.sum(), "filas inválidas:", (~valid).sum())
df_clean = df[valid].copy()
df_clean.to_csv("D:/pipeline_SVM/features/features_svm_baseline.cleaned.csv", index=False)
print("Guardado: features_svm_baseline.cleaned.csv")

