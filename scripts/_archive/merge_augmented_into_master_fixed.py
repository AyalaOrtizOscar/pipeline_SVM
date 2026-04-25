#!/usr/bin/env python3
# merge_augmented_into_master_fixed.py
import pandas as pd
from pathlib import Path

ROOT = Path("D:/pipeline_SVM")
MASTER = ROOT / "features" / "features_svm_baseline.cleaned.csv"
AUG = ROOT / "features" / "features_augmented_desgastado.csv"
OUT = ROOT / "features" / "features_svm_with_augmented.csv"

print("Leyendo master:", MASTER)
df_master = pd.read_csv(MASTER, low_memory=False)
print("Master columnas:", len(df_master.columns))

print("Leyendo augmented:", AUG)
df_aug = pd.read_csv(AUG, low_memory=False)
print("Augmented columnas:", len(df_aug.columns))

# Ensure a basename column exists in both
for df,name in [(df_master,"master"),(df_aug,"aug")]:
    if 'basename' not in df.columns:
        if 'filepath' in df.columns:
            df['basename'] = df['filepath'].astype(str).apply(lambda p: Path(p).name)
            print(f"Creada 'basename' en {name} desde 'filepath'")
        elif 'wav_path' in df.columns:
            df['basename'] = df['wav_path'].astype(str).apply(lambda p: Path(p).name)
            print(f"Creada 'basename' en {name} desde 'wav_path'")
        else:
            print(f"Advertencia: no encuentro 'basename' ni 'filepath' en {name} -- continuaré sin basename")

# Normalize path column to 'fp_norm' in both (if possible)
def add_fp_norm(df):
    if 'filepath' in df.columns:
        df['fp_norm'] = df['filepath'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    elif 'wav_path_norm' in df.columns:
        df['fp_norm'] = df['wav_path_norm'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    elif 'wav_path' in df.columns:
        df['fp_norm'] = df['wav_path'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    else:
        # fallback to basename
        if 'basename' in df.columns:
            df['fp_norm'] = df['basename'].astype(str).str.strip().str.lower()
add_fp_norm(df_master)
add_fp_norm(df_aug)

# make sure label columns exist
if 'label' not in df_master.columns and 'label_clean' in df_master.columns:
    df_master['label'] = df_master['label_clean']
if 'label' not in df_aug.columns and 'label_clean' in df_aug.columns:
    df_aug['label'] = df_aug['label_clean']

# mark aug rows
if 'is_aug' not in df_aug.columns:
    df_aug['is_aug'] = df_aug['basename'].astype(str).str.contains('_aug|augment|auto', case=False, na=False)
if 'is_aug' not in df_master.columns:
    df_master['is_aug'] = False

# Build common columns (union)
common_cols = sorted(set(df_master.columns.tolist()) | set(df_aug.columns.tolist()))
print("Total columnas (union):", len(common_cols))

# Reindex each df to the full set of columns (add NaNs where missing)
df_master_r = df_master.reindex(columns=common_cols)
df_aug_r = df_aug.reindex(columns=common_cols)

# Optional: if you want to avoid exact duplicate filepaths between original and aug,
# keep original's fp_norm as-is and augment's fp_norm can be left (they usually differ).
# Now concat
df_all = pd.concat([df_master_r, df_aug_r], ignore_index=True, sort=False)

# Clean up temporaries if present
for c in ['fp_norm_unique']:
    if c in df_all.columns:
        df_all.drop(columns=[c], inplace=True)

print("Filas master:", len(df_master), "filas aug:", len(df_aug), "-> total:", len(df_all))

df_all.to_csv(OUT, index=False)
print("Guardado en:", OUT)
