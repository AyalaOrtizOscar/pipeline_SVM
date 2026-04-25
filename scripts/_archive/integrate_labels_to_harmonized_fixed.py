#!/usr/bin/env python3
"""
integrate_labels_to_harmonized_fixed.py (versión corregida)
Integra labels desde un CSV "fixed" al CSV "harmonized" usando heurísticas:
  - fp_norm exact
  - basename exact
  - basename_core fallback
Salida:
  - <harmonized>.labelled_full.csv
  - <harmonized>.labeled.csv
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

def norm_fp(s):
    if pd.isna(s):
        return ''
    s = str(s).replace("\\", "/").strip().lower()
    s = re.sub(r'^[a-z]:/', '', s)
    return s

def basename_core_from_name(name):
    if pd.isna(name):
        return ''
    s = str(name)
    s = re.sub(r'(_|-)?(?:aug|auto)[A-Za-z0-9_]*', '', s, flags=re.I)
    s = re.sub(r'(_|-)?(aug|auto).*$', '', s, flags=re.I)
    return s.strip()

def find_label_col(df):
    cand = ['label_fixed','label_mapped','label','label_map','label_clean','label_fixed_final']
    for c in cand:
        if c in df.columns:
            return c
    return None

def mode_or_first(s):
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0] if len(s)>0 else ''

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harmonized", "-H", required=True)
    p.add_argument("--fixed", "-F", required=True)
    p.add_argument("--out-prefix", "-o", default=None)
    args = p.parse_args()

    H = Path(args.harmonized)
    F = Path(args.fixed)
    if not H.exists() or not F.exists():
        print("File not found:", (str(H) if not H.exists() else ""), (str(F) if not F.exists() else ""))
        return

    df_h = pd.read_csv(H, low_memory=False)
    df_f = pd.read_csv(F, low_memory=False)

    print("Harmonized rows:", len(df_h), "Fixed rows:", len(df_f))

    # normalizar/crear fp_norm
    for df in (df_h, df_f):
        if 'fp_norm' not in df.columns:
            if 'filepath' in df.columns:
                df['fp_norm'] = df['filepath'].astype(str)
            elif 'fp' in df.columns:
                df['fp_norm'] = df['fp'].astype(str)
            else:
                df['fp_norm'] = ''
        df['fp_norm'] = df['fp_norm'].fillna('').astype(str).map(norm_fp)

    # asegurar basename
    for df in (df_h, df_f):
        if 'basename' not in df.columns:
            if 'filepath' in df.columns:
                df['basename'] = df['filepath'].astype(str).map(lambda x: Path(x).name if str(x).strip()!='' else '')
            else:
                df['basename'] = df.get('wav_name', '').fillna('').astype(str)

    # basename_core
    df_h['basename_core'] = df_h['basename'].map(basename_core_from_name)
    df_f['basename_core'] = df_f['basename'].map(basename_core_from_name)

    # label col en fixed
    label_col = find_label_col(df_f)
    if label_col is None:
        print("WARNING: no label column detected in fixed CSV. Columns:", df_f.columns.tolist())
    else:
        print("Using label column from fixed CSV:", label_col)
    map_col = 'map_method' if 'map_method' in df_f.columns else None

    # construir df_fix
    cols_for_fix = ['fp_norm', 'basename', 'basename_core']
    df_fix = df_f[cols_for_fix].copy()
    df_fix['label_src'] = df_f[label_col].astype(str).fillna('') if label_col else ''
    df_fix['map_method'] = df_f[map_col].astype(str).fillna('') if map_col else ''

    # index por fp_norm (único)
    df_fix_fp = df_fix[df_fix['fp_norm'].astype(bool)].drop_duplicates(subset='fp_norm', keep='first').set_index('fp_norm')

    # mapas por basename / basename_core
    basename_map = {}
    if df_fix['basename'].astype(bool).any():
        basename_map = (df_fix[df_fix['basename'].astype(bool)].groupby('basename')['label_src']
                        .agg(mode_or_first).to_dict())
    basename_core_map = {}
    if df_fix['basename_core'].astype(bool).any():
        basename_core_map = (df_fix[df_fix['basename_core'].astype(bool)].groupby('basename_core')['label_src']
                             .agg(mode_or_first).to_dict())

    # preparar columnas destino con dtype object
    if 'label_fixed' not in df_h.columns:
        df_h['label_fixed'] = pd.Series([np.nan]*len(df_h), index=df_h.index, dtype=object)
    else:
        df_h['label_fixed'] = df_h['label_fixed'].astype(object)

    if 'label_map_method' not in df_h.columns:
        df_h['label_map_method'] = pd.Series([np.nan]*len(df_h), index=df_h.index, dtype=object)
    else:
        df_h['label_map_method'] = df_h['label_map_method'].astype(object)

    # 1) fp_norm exact
    matched_fp = df_h['fp_norm'].isin(df_fix_fp.index)
    if matched_fp.any():
        df_h.loc[matched_fp, 'label_fixed'] = df_h.loc[matched_fp, 'fp_norm'].map(df_fix_fp['label_src'])
        if 'map_method' in df_fix_fp.columns:
            df_h.loc[matched_fp, 'label_map_method'] = df_h.loc[matched_fp, 'fp_norm'].map(df_fix_fp['map_method']).fillna('fp_norm_exact')
        else:
            df_h.loc[matched_fp, 'label_map_method'] = 'fp_norm_exact'

    # 2) basename exact (solo donde aún falta)
    missing_mask = df_h['label_fixed'].isna() | (df_h['label_fixed'].astype(str).str.strip() == '')
    mask_basename = missing_mask & df_h['basename'].isin(basename_map.keys())
    if mask_basename.any():
        df_h.loc[mask_basename, 'label_fixed'] = df_h.loc[mask_basename, 'basename'].map(basename_map)
        df_h.loc[mask_basename, 'label_map_method'] = df_h.loc[mask_basename, 'label_map_method'].fillna('basename_exact')
        df_h.loc[mask_basename & (df_h['label_map_method'].astype(str).str.strip() == ''), 'label_map_method'] = 'basename_exact'

    # 3) basename_core fallback
    missing_mask = df_h['label_fixed'].isna() | (df_h['label_fixed'].astype(str).str.strip() == '')
    mask_core = missing_mask & df_h['basename_core'].isin(basename_core_map.keys())
    if mask_core.any():
        df_h.loc[mask_core, 'label_fixed'] = df_h.loc[mask_core, 'basename_core'].map(basename_core_map)
        df_h.loc[mask_core, 'label_map_method'] = df_h.loc[mask_core, 'label_map_method'].fillna('basename_core')
        df_h.loc[mask_core & (df_h['label_map_method'].astype(str).str.strip() == ''), 'label_map_method'] = 'basename_core'

    # normalizar vacíos a NaN
    df_h['label_fixed'] = df_h['label_fixed'].replace({'': np.nan, 'nan': np.nan})
    df_h['label_map_method'] = df_h['label_map_method'].replace({'': np.nan, 'nan': np.nan})

    n_total = len(df_h)
    n_labeled = df_h['label_fixed'].notna().sum()
    print(f"Labeled rows after integration: {n_labeled} of {n_total}")

    out_prefix = args.out_prefix if args.out_prefix else H.stem
    out_full = H.with_name(out_prefix + ".labelled_full.csv")
    out_only = H.with_name(out_prefix + ".labeled.csv")

    df_h.to_csv(out_full, index=False)
    df_h[df_h['label_fixed'].notna()].to_csv(out_only, index=False)
    print("Saved full:", out_full)
    print("Saved only-labeled:", out_only)

if __name__ == "__main__":
    main()
