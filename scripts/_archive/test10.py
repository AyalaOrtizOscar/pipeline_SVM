import pandas as pd, numpy as np
df = pd.read_csv("D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv", low_memory=False)  # o el CSV de auditoría que guardaste
print("Columnas totales:", len(df.columns))
print("Columnas ejemplo:", df.columns[:40].tolist())
print("\nValores únicos label (raw):\n", df['label'].value_counts(dropna=False))
# si usas label_clean:
if 'label_clean' in df.columns:
    print("\nlabel_clean counts:\n", df['label_clean'].value_counts(dropna=False))

# Chequeo features numericas
feature_df = df.select_dtypes(include=[np.number])
print("Features numéricas:", len(feature_df.columns))
print("Columnas con todos NaN:", feature_df.columns[feature_df.isna().all()].tolist())
print("Columnas con >50% NaN (posible problema):")
print((feature_df.isna().mean() > 0.5).sum(), "columnas")


