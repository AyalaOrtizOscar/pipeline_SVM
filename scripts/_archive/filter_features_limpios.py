# filter_features_limpios.py
import pandas as pd, os
SRC = r"D:\pipeline_SVM\inputs\features_svm_baseline_cleaned.csv"
OUT = r"D:\pipeline_SVM\inputs\features_svm_baseline_limpios.csv"

df = pd.read_csv(SRC, low_memory=False)
# Ajusta si tus filepath usan "limpios" u otra nomenclatura
mask = df['filepath'].astype(str).str.lower().str.contains('limpios')  \
       | df['filepath'].astype(str).str.lower().str.contains('sin falla limpios') \
       | df['filepath'].astype(str).str.lower().str.contains('con falla limpios')
# si tienes columna específica 'source' o 'clean' usa esa
df_limpios = df[mask].copy()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_limpios.to_csv(OUT, index=False)
print("Guardado:", OUT, "filas:", len(df_limpios))
