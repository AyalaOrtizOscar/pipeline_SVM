# integrate_labels_to_harmonized.py
import pandas as pd
from pathlib import Path
H = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.csv")
FIX = Path("D:/pipeline_SVM/results/fix_harmonized_v5/features_svm_harmonized_fixed_v5.csv")
OUT = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labeled.csv")

df_h = pd.read_csv(H, low_memory=False)
df_fix = pd.read_csv(FIX, low_memory=False)[['fp_norm','label_fixed','map_method']]

# ensure fp_norm exists
if 'fp_norm' not in df_h.columns:
    df_h['fp_norm'] = df_h['filepath'].astype(str).str.replace("\\\\","/").str.strip().str.lower()

df = df_h.merge(df_fix, on='fp_norm', how='left', validate='1:1')
# keep only rows with label_fixed (si quieres mantener también sin label, cambia how='left' y guarda)
labeled = df[df['label_fixed'].notna()].copy()
print("Rows labeled:", len(labeled), "of", len(df))
labeled.to_csv(OUT, index=False)
print("Saved:", OUT)
