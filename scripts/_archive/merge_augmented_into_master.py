#!/usr/bin/env python3
# merge_augmented_into_master.py
import pandas as pd
from pathlib import Path

ROOT = Path("D:/pipeline_SVM")
MASTER = ROOT / "features" / "features_svm_baseline.cleaned.csv"   # tu CSV limpio original
AUG = ROOT / "features" / "features_augmented_desgastado.csv"     # lo que generaste
OUT = ROOT / "features" / "features_svm_with_augmented.csv"

print("Leyendo master:", MASTER)
df_master = pd.read_csv(MASTER, low_memory=False)

print("Leyendo augmented:", AUG)
df_aug = pd.read_csv(AUG, low_memory=False)

# Normaliza nombres de archivo (si existen columnas filepath o basename)
def norm_path_col(df):
    if 'filepath' in df.columns:
        df['fp_norm'] = df['filepath'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    elif 'wav_path' in df.columns:
        df['fp_norm'] = df['wav_path'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    elif 'wav_path_norm' in df.columns:
        df['fp_norm'] = df['wav_path_norm'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
    else:
        # intentar usar basename
        if 'basename' in df.columns:
            df['fp_norm'] = df['basename'].astype(str).str.strip().str.lower()
        else:
            raise SystemExit("No encuentro 'filepath' ni 'basename' en uno de los CSVs. Añade una columna con ruta o basename.")
    return df

df_master = norm_path_col(df_master)
df_aug = norm_path_col(df_aug)

# marca augmentados
# asumimos que los nombres de augment contienen '_aug' o similar; si no, fuerza una columna
if 'is_aug' not in df_aug.columns:
    df_aug['is_aug'] = df_aug['basename'].astype(str).str.contains('_aug|augment|auto', case=False, na=False)

# asegúrate de que la etiqueta exista en df_aug (si no, rellena con 'desgastado')
if 'label' not in df_aug.columns and 'label_clean' in df_aug.columns:
    df_aug['label'] = df_aug['label_clean']
if 'label' not in df_aug.columns:
    df_aug['label'] = 'desgastado'

# normaliza label col name (master usa label_clean)
if 'label_clean' not in df_master.columns:
    if 'label' in df_master.columns:
        df_master['label_clean'] = df_master['label']

if 'label_clean' not in df_aug.columns:
    df_aug['label_clean'] = df_aug['label']

# evita duplicados exactos por fp_norm: si un augment tiene mismo fp_norm que original, agregamos un sufijo
df_aug['fp_norm_unique'] = df_aug['fp_norm']
dups = df_aug['fp_norm'].duplicated(keep=False)
if dups.any():
    # si hay duplicados en augment list, añadir índice
    df_aug.loc[dups, 'fp_norm_unique'] = df_aug.loc[dups].groupby('fp_norm').cumcount().astype(str).radd(df_aug.loc[dups,'fp_norm'] + "_aug_")

# Para merge final, dejamos fp_norm (originals) intactos y mantenemos is_aug flag
# Concatenar
df_master['is_aug'] = False
# usa columns compartidas + extras
common_cols = sorted(list(set(df_master.columns).union(set(df_aug.columns))))
df_all = pd.concat([df_master[common_cols].fillna(""), df_aug[common_cols].fillna("")], ignore_index=True, sort=False)

# limpia columnas temporales
if 'fp_norm_unique' in df_all.columns:
    df_all = df_all.drop(columns=['fp_norm_unique'], errors='ignore')

print("Filas master:", len(df_master), "filas aug:", len(df_aug), "-> total:", len(df_all))

df_all.to_csv(OUT, index=False)
print("Guardado en:", OUT)
