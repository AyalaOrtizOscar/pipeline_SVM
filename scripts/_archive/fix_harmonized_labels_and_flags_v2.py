#!/usr/bin/env python3
"""
fix_harmonized_labels_and_flags_v2.py
Versión robusta para rellenar labels y marcar augment en features_svm_harmonized_for_svm.csv
Busca primero labels en:
 - features/features_svm_baseline.cleaned.csv (preferido)
 - features/features_svm_baseline_reextracted.csv
Si no encuentra, intenta heurística por ruta.
"""
import pandas as pd
from pathlib import Path
import json
import numpy as np
import re

ROOT = Path("D:/pipeline_SVM")
MASTER_CANDIDATES = [
    ROOT/"features"/"features_svm_baseline.cleaned.csv",
    ROOT/"features"/"features_svm_baseline_reextracted.csv",
    ROOT/"features"/"features_svm_baseline.csv",
]
HARM_P = ROOT/"features"/"features_svm_harmonized_for_svm.csv"
AUG_MAP_P = ROOT/"augmented"/"minority_desgastado"/"augmented_mapping.csv"
OUT_DIR = ROOT/"results"/"fix_harmonized_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_path(s):
    if pd.isna(s): return ""
    return str(s).replace("\\","/").strip().lower()

def infer_label_from_path(fp):
    fp = norm_path(fp)
    # Ajusta estas reglas si tus nombres de carpetas son distintos
    if any(x in fp for x in ['sin falla','sin_falla','sin-falla','limpio','limpios','sin_desgaste','sinfalla']):
        return 'sin_desgaste'
    if any(x in fp for x in ['con falla','con_falla','confalla','con-desgaste','desgastado']):
        return 'desgastado'
    if any(x in fp for x in ['medio','median','medianamente','medio_desgaste','medianamente_desgastado']):
        return 'medianamente_desgastado'
    return None

# load harmonized
print("Leyendo harmonized:", HARM_P)
df_h = pd.read_csv(HARM_P, low_memory=False)

# ensure fp_norm exists
if 'fp_norm' not in df_h.columns:
    src = None
    for c in ['filepath','wav_path_norm','wav_path','path']:
        if c in df_h.columns:
            src = c; break
    if src is None:
        raise SystemExit("No encontré columna de ruta en harmonized (esperaba filepath/wav_path_norm/wav_path).")
    df_h['fp_norm'] = df_h[src].astype(str).apply(norm_path)

# initialize label_mapped
df_h['label_mapped'] = np.nan

# try to find a master with labels
master_used = None
for mp in MASTER_CANDIDATES:
    if mp.exists():
        try:
            df_master = pd.read_csv(mp, low_memory=False)
            # normalize master fp
            if 'fp_norm' not in df_master.columns:
                for c in ['filepath','wav_path_norm','wav_path','path']:
                    if c in df_master.columns:
                        df_master['fp_norm'] = df_master[c].astype(str).apply(norm_path)
                        break
            # choose label col
            label_cols = [c for c in ['label_clean','label'] if c in df_master.columns]
            if label_cols:
                labc = label_cols[0]
                print("Usando master", mp, "columna", labc, "para mapear labels.")
                mapping = df_master.dropna(subset=['fp_norm']).set_index('fp_norm')[labc].to_dict()
                df_h['label_mapped'] = df_h['fp_norm'].map(mapping)
                master_used = mp
                break
            else:
                print("Master", mp, "no tiene columna label_clean ni label; buscando siguiente candidate.")
        except Exception as e:
            print("No puedo leer master", mp, e)
# if no mapping found, try alternative: if there is a cleaned master stored elsewhere (user earlier had features_svm_baseline.cleaned.csv)
if master_used is None:
    print("No se halló master con labels. Intentando heurística por ruta para inferir labels...")
    df_h.loc[df_h['label_mapped'].isna(), 'label_mapped'] = df_h.loc[df_h['label_mapped'].isna(), 'fp_norm'].apply(infer_label_from_path)
    print("Labels inferidos por heurística:", df_h['label_mapped'].notna().sum())

# create is_augment flag
df_h['is_augment'] = 0
if AUG_MAP_P.exists():
    try:
        aug_map = pd.read_csv(AUG_MAP_P, low_memory=False)
        # try to find a column with augmented path/basename
        if 'augmented' in aug_map.columns:
            aug_fp = aug_map['augmented'].astype(str).apply(norm_path)
        elif 'aug_path' in aug_map.columns:
            aug_fp = aug_map['aug_path'].astype(str).apply(norm_path)
        elif 'basename' in aug_map.columns:
            aug_fp = aug_map['basename'].astype(str).apply(norm_path)
        else:
            aug_fp = pd.Series([], dtype=str)
        aug_set = set(aug_fp.dropna().unique())
        df_h.loc[df_h['fp_norm'].isin(aug_set), 'is_augment'] = 1
        print("Marcadas is_augment por mapping:", int(df_h['is_augment'].sum()))
    except Exception as e:
        print("Error leyendo augmented mapping:", e)

# fallback detections
df_h.loc[df_h['fp_norm'].str.contains("/augmented/"), 'is_augment'] = 1
df_h.loc[df_h['fp_norm'].str.contains("_aug|aug_auto|aug-auto|augmented", regex=True), 'is_augment'] = 1

print("Total is_augment:", int(df_h['is_augment'].sum()), "de", len(df_h))

# report missing labels before dedupe
n_missing_before = int(df_h['label_mapped'].isna().sum())
print("Labels faltantes antes dedupe:", n_missing_before)

# resolve duplicates by fp_norm (prefer is_augment==0)
dupe_counts = df_h['fp_norm'].duplicated(keep=False).sum()
print("Duplicados exactos por fp_norm (rows involved):", dupe_counts)
def resolve_group(g):
    not_aug = g[g['is_augment']==0]
    if len(not_aug)>0:
        return not_aug.iloc[0]
    else:
        return g.iloc[0]

df_h_no_dupes = df_h.groupby('fp_norm', group_keys=False).apply(resolve_group).reset_index(drop=True)
print("Filas tras resolver duplicados:", len(df_h_no_dupes))

# recompute missing labels
n_missing_after = int(df_h_no_dupes['label_mapped'].isna().sum())
print("Labels faltantes después dedupe:", n_missing_after)

# save examples of missing for manual inspection
missing_examples = df_h_no_dupes[df_h_no_dupes['label_mapped'].isna()].copy()
if not missing_examples.empty:
    missing_examples[['fp_norm','filepath','basename','is_augment']].head(200).to_csv(OUT_DIR/"missing_label_examples.csv", index=False)
    print("Guardadas muestras sin label a:", OUT_DIR/"missing_label_examples.csv")

# finalize: rename label_mapped->label (but keep both)
df_h_no_dupes['label'] = df_h_no_dupes['label_mapped']
df_h_no_dupes.drop(columns=[c for c in ['label_mapped'] if c in df_h_no_dupes.columns], inplace=True)

# save final CSV and report
OUT_P = OUT_DIR/"features_svm_harmonized_fixed_v2.csv"
df_h_no_dupes.to_csv(OUT_P, index=False)
report = {
    "harmonized_rows_before": int(len(df_h)),
    "harmonized_rows_after": int(len(df_h_no_dupes)),
    "n_aug_detected": int(df_h_no_dupes['is_augment'].sum()),
    "n_labels_present": int(df_h_no_dupes['label'].notna().sum()),
    "n_labels_missing": int(df_h_no_dupes['label'].isna().sum()),
    "master_used_for_mapping": str(master_used) if master_used is not None else None
}
with open(OUT_DIR/"fix_report_v2.json","w",encoding="utf8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print("Guardado CSV fijo en:", OUT_P)
print("Guardado reporte en:", OUT_DIR/"fix_report_v2.json")
