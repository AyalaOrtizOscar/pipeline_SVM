import pandas as pd
from pathlib import Path
f = r"D:\pipeline_SVM\features\features_svm_baseline.csv"
df = pd.read_csv(f, low_memory=False)
print("Filas totales:", len(df))
print("\nValores únicos en label (top 50):")
print(df['label'].value_counts(dropna=False).head(50).to_string())
print("\nColumnas disponibles:", df.columns.tolist())
for col in ['experiment','mic_type','filepath']:
    if col in df.columns:
        nnull = df[col].isna().sum()
        print(f"{col}: {nnull} NaNs")
    else:
        print(f"{col}: (NO EXISTE)")
# Top groups (if exist)
if 'experiment' in df.columns:
    print("\nTop 20 experiments:", df['experiment'].value_counts().head(20).to_string())
elif 'mic_type' in df.columns:
    print("\nTop 20 mic_type:", df['mic_type'].value_counts().head(20).to_string())
else:
    print("No hay columna clara de grupos (experiment/mic_type).")
