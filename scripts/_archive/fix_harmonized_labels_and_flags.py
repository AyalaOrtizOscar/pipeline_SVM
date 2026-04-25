#!/usr/bin/env python3
"""
fix_harmonized_labels_and_flags.py
Repara labels y flags en features_svm_harmonized_for_svm.csv usando
features_svm_baseline_reextracted.csv (master) y, si existe,
augmented/minority_desgastado/augmented_mapping.csv
"""
import pandas as pd
from pathlib import Path
import json
import re

ROOT = Path("D:/pipeline_SVM")
MASTER_P = ROOT/"features"/"features_svm_baseline_reextracted.csv"
HARM_P = ROOT/"features"/"features_svm_harmonized_for_svm.csv"
AUG_MAP_P = ROOT/"augmented"/"minority_desgastado"/"augmented_mapping.csv"
OUT_DIR = ROOT/"results"/"fix_harmonized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_path(s):
    if pd.isna(s): return ""
    return str(s).replace("\\","/").strip().lower()

print("Leyendo master:", MASTER_P)
df_master = pd.read_csv(MASTER_P, low_memory=False)
print("Leyendo harmonized:", HARM_P)
df_h = pd.read_csv(HARM_P, low_memory=False)

# crear columna de fp_norm si no existe
for df in (df_master, df_h):
    if 'fp_norm' not in df.columns:
        # prefer 'filepath' else try 'wav_path_norm'
        src = None
        for c in ['filepath','wav_path_norm','wav_path','fp','path']:
            if c in df.columns:
                src = c; break
        if src is None:
            raise SystemExit("No encontré columna de ruta en uno de los CSVs. Añade 'filepath' o 'wav_path_norm'.")
        df['fp_norm'] = df[src].astype(str).apply(norm_path)

# mapear labels desde master a harmonized por fp_norm
label_cols = [c for c in ['label_clean','label'] if c in df_master.columns]
if not label_cols:
    print("WARNING: master no tiene columna label_clean ni label. No puedo mapear labels automáticamente.")
else:
    labc = label_cols[0]
    print("Usaré", labc, "desde master para mapear labels.")
    # construir mapping fp_norm -> label
    mp = df_master.dropna(subset=['fp_norm']).set_index('fp_norm')[labc].to_dict()
    df_h['label_mapped'] = df_h['fp_norm'].map(mp)
    n_mapped = df_h['label_mapped'].notna().sum()
    print("Labels mapeados por fp_norm:", n_mapped, "de", len(df_h))

# si quedan filas sin label, intentar inferir del filepath (heurística)
def infer_label_from_path(fp):
    fp = norm_path(fp)
    # patrones - ajusta según tu estructura real
    if 'sin falla' in fp or 'sin_falla' in fp or 'sin-falla' in fp or 'limpio' in fp:
        return 'sin_desgaste'
    if 'con falla' in fp or 'con_falla' in fp or 'con-falla' in fp:
        return 'desgastado'
    # medium / medio
    if 'median' in fp or 'medio' in fp or 'medio_desgaste' in fp or 'medianamente' in fp:
        return 'medianamente_desgastado'
    return None

still_missing = df_h['label_mapped'].isna().sum()
if still_missing>0:
    print("Intentando inferir labels por heurística de ruta para", still_missing, "filas...")
    df_h.loc[df_h['label_mapped'].isna(), 'label_mapped'] = df_h.loc[df_h['label_mapped'].isna(), 'fp_norm'].apply(infer_label_from_path)
    print("Ahora labels inferidos total:", df_h['label_mapped'].notna().sum())

# rename final label column
df_h['label'] = df_h['label_mapped']
df_h.drop(columns=[c for c in ['label_mapped'] if c in df_h.columns], inplace=True)

# Crear flag is_augment: 3 niveles - 1) usar map de augmented_mapping.csv si existe
df_h['is_augment'] = 0
if AUG_MAP_P.exists():
    print("Leyendo augmented mapping:", AUG_MAP_P)
    try:
        aug_map = pd.read_csv(AUG_MAP_P, low_memory=False)
        # Normaliza columnas esperadas: 'original' y 'augmented' o 'basename'
        if 'augmented' in aug_map.columns:
            aug_map['aug_fp'] = aug_map['augmented'].astype(str).apply(norm_path)
        elif 'aug_path' in aug_map.columns:
            aug_map['aug_fp'] = aug_map['aug_path'].astype(str).apply(norm_path)
        elif 'augmented_fp' in aug_map.columns:
            aug_map['aug_fp'] = aug_map['augmented_fp'].astype(str).apply(norm_path)
        else:
            # fallback: try basename column
            if 'basename' in aug_map.columns:
                aug_map['aug_fp'] = aug_map['basename'].astype(str).apply(norm_path)
        aug_set = set(aug_map['aug_fp'].dropna().unique())
        df_h.loc[df_h['fp_norm'].isin(aug_set), 'is_augment'] = 1
        print("Marcadas filas is_augment por mapping:", int(df_h['is_augment'].sum()))
    except Exception as e:
        print("No pude usar augmented_mapping:", e)

# complemento de detección: ruta contiene '/augmented/' o folder 'augmented' o basename contiene '_aug'
df_h.loc[df_h['fp_norm'].str.contains("/augmented/"), 'is_augment'] = 1
df_h.loc[df_h['fp_norm'].str.contains("_aug|aug_auto|aug-auto|augmented", regex=True), 'is_augment'] = 1

print("Total is_augment final:", int(df_h['is_augment'].sum()), "de", len(df_h))

# Duplicados por ruta normalizada (fp_norm)
dupes = df_h[df_h.duplicated(subset=['fp_norm'], keep=False)]
print("Duplicados exactos por fp_norm:", len(dupes))
# Resolver: si hay duplicados, priorizar mantener fila con is_augment==0 (original), si hay varios originales mantener first
def resolve_dupes(group):
    # group is dataframe with same fp_norm
    not_aug = group[group['is_augment']==0]
    if len(not_aug)>0:
        return not_aug.iloc[0]
    else:
        return group.iloc[0]

df_h_no_dupes = df_h.groupby('fp_norm', group_keys=False).apply(resolve_dupes).reset_index(drop=True)
print("Filas tras resolver duplicados por fp_norm:", len(df_h_no_dupes))

# Reportes y salvado
report = {
    "original_master_rows": int(len(df_master)),
    "harmonized_rows_before": int(len(df_h)),
    "harmonized_rows_after_dedupe": int(len(df_h_no_dupes)),
    "n_aug_detected": int(df_h_no_dupes['is_augment'].sum()),
    "n_labels_present": int(df_h_no_dupes['label'].notna().sum()),
    "n_labels_missing": int(df_h_no_dupes['label'].isna().sum())
}

with open(OUT_DIR/"fix_report.json","w",encoding="utf8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

OUT_P = OUT_DIR/"features_svm_harmonized_fixed.csv"
df_h_no_dupes.to_csv(OUT_P, index=False)
print("Guardado harmonized fixed en:", OUT_P)
print("Reporte:", OUT_DIR/"fix_report.json")
