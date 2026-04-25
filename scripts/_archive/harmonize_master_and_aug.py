#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np

MASTER = Path("D:/pipeline_SVM/features/features_svm_baseline_reextracted.csv")  # ajustar si no reextrajiste
AUG   = Path("D:/pipeline_SVM/features/features_augmented_desgastado.csv")
OUT   = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.csv")

print("Leyendo master:", MASTER)
df_m = pd.read_csv(MASTER, low_memory=False)
print("Leyendo augment:", AUG)
df_a = pd.read_csv(AUG, low_memory=False)

# normalize column names if needed (strip)
df_m.columns = df_m.columns.str.strip()
df_a.columns = df_a.columns.str.strip()

shared = list(set(df_m.columns).intersection(df_a.columns))
print("Columnas compartidas (count):", len(shared))

# Keep metadata cols if present (ensure label, label_clean, filepath, mic_type, experiment, duration)
meta = ["filepath","label","label_clean","mic_type","experiment","duration","basename","wav_path_norm"]
meta_present = [c for c in meta if c in df_m.columns or c in df_a.columns]
print("Meta present:", meta_present)

# Decide final columns: meta_present + shared_features_without_meta
features_shared = [c for c in shared if c not in meta_present]
final_cols = meta_present + features_shared
print("Columnas finales (example 40):", final_cols[:40])

# Subset and concat (filling missing columns)
df_m_sub = df_m.reindex(columns=final_cols)
df_a_sub = df_a.reindex(columns=final_cols)

df_all = pd.concat([df_m_sub, df_a_sub], ignore_index=True, sort=False)
print("Union shape:", df_all.shape)

# For numeric columns, fill NaN with median computed on master (prefer master stats)
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
medians = df_m[num_cols].median()
df_all[num_cols] = df_all[num_cols].fillna(medians)

# For non-numeric, fill with empty string or propagate from master where possible
obj_cols = [c for c in final_cols if c not in num_cols]
df_all[obj_cols] = df_all[obj_cols].fillna("")

df_all.to_csv(OUT, index=False)
print("Guardado harmonized CSV en:", OUT)
