#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np
import json
import re

ROOT = Path("D:/pipeline_SVM")
HARM_P = ROOT/"features"/"features_svm_harmonized_for_svm.csv"
MASTER_P = ROOT/"features"/"features_svm_baseline.cleaned.csv"
OUT_DIR = ROOT/"results"/"fix_harmonized_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(s):
    if pd.isna(s): return ""
    return str(s).replace("\\","/").strip().lower()

def basename_core(name):
    """Strip common augment suffixes to recover original basename.
       Examples: b01001_aug1.wav -> b01001.wav, file-aug.wav -> file.wav,
       broca_6_2_592_aug_auto_3.wav -> broca_6_2_592.wav
    """
    if not isinstance(name, str): return name
    n = name
    # remove common augment markers before extension
    # keep extension
    m = re.match(r"^(?P<base>.+?)(?:[_\-\.](?:aug(?:_[a-z0-9]*)?|augauto|aug-auto|augauto\d+|aug\d+|augauto_\d+|aug_auto|\d+_aug|aug_auto\d+|augauto-\d+|aug$|aug1|aug2|aug3|aug4))(?:\..+)$", n, flags=re.IGNORECASE)
    if m:
        # restore extension
        ext = Path(n).suffix
        return (m.group("base") + ext).lower()
    # as fallback, remove patterns like _aug\d+, _aug, -aug, _aug_auto\d+
    core = re.sub(r"(_|-)(aug|aug\d+|aug_auto|augauto|aug-auto|aug_[0-9]+|_augauto[0-9]+)$", "", Path(n).stem, flags=re.IGNORECASE)
    if core != Path(n).stem:
        return (core + Path(n).suffix).lower()
    return n.lower()

print("Leyendo harmonized:", HARM_P)
df_h = pd.read_csv(HARM_P, low_memory=False)
print("Leyendo master:", MASTER_P)
df_m = pd.read_csv(MASTER_P, low_memory=False)

# ensure fp_norm in both
for df,name in [(df_h,'harm'),(df_m,'master')]:
    if 'fp_norm' not in df.columns:
        cand = None
        for c in ['filepath','wav_path_norm','wav_path','path']:
            if c in df.columns:
                cand = c; break
        if cand:
            df['fp_norm'] = df[cand].astype(str).apply(norm)
            print(f"Creada fp_norm en {name} desde {cand}")
    if 'basename' not in df.columns:
        df['basename'] = df.get('basename', df.get('filepath','')).astype(str).apply(lambda s: Path(s).name.lower())
    if 'experiment' not in df.columns:
        df['experiment'] = df.get('experiment', df.get('fp_norm','')).astype(str).apply(lambda s: next((seg for seg in s.split('/') if seg.startswith('e')), ""))

# label col in master
label_col = None
for c in ['label_clean','label']:
    if c in df_m.columns:
        label_col = c; break
if label_col is None:
    print("ERROR: master no tiene columna de label. Abortar.")
    raise SystemExit(1)

print("Label column in master:", label_col, "non-null count:", int(df_m[label_col].notna().sum()))

# prepare mapping helpers
df_m['basename'] = df_m['basename'].astype(str).str.lower()
df_h['basename'] = df_h['basename'].astype(str).str.lower()
df_m['basename_core'] = df_m['basename'].apply(basename_core)
df_h['basename_core'] = df_h['basename'].apply(basename_core)

df_h['label_mapped'] = np.nan
df_h['map_method'] = ""

# 1) exact fp_norm
m_map = df_m.dropna(subset=['fp_norm']).set_index('fp_norm')[label_col].to_dict()
matched = df_h['fp_norm'].map(m_map)
df_h.loc[matched.notna(), 'label_mapped'] = matched[matched.notna()]
df_h.loc[matched.notna(), 'map_method'] = 'fp_exact'
print("Mapped by exact fp_norm:", int(df_h['map_method'].eq('fp_exact').sum()))

# 2) basename exact
mask = df_h['label_mapped'].isna()
if mask.any():
    basemap = df_m.dropna(subset=['basename']).groupby('basename')[label_col].agg(lambda s:s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
    matched3 = df_h.loc[mask,'basename'].map(basemap)
    df_h.loc[mask & matched3.notna(), 'label_mapped'] = matched3[matched3.notna()]
    df_h.loc[mask & matched3.notna(), 'map_method'] = 'basename_exact'
    print("Mapped by basename exact:", int((df_h['map_method']=='basename_exact').sum()))

# 3) basename_core (strip augment suffixes)
mask = df_h['label_mapped'].isna()
if mask.any():
    coremap = df_m.dropna(subset=['basename_core']).groupby('basename_core')[label_col].agg(lambda s:s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
    matched_core = df_h.loc[mask,'basename_core'].map(coremap)
    df_h.loc[mask & matched_core.notna(), 'label_mapped'] = matched_core[matched_core.notna()]
    df_h.loc[mask & matched_core.notna(), 'map_method'] = 'basename_core'
    print("Mapped by basename_core (stripping augment):", int((df_h['map_method']=='basename_core').sum()))

# 4) experiment + basename_core
mask = df_h['label_mapped'].isna()
if mask.any():
    df_m['exp_core'] = df_m.get('experiment','').astype(str).str.lower().fillna('') + "||" + df_m['basename_core'].astype(str)
    key_series = df_h.get('experiment','').astype(str).str.lower().fillna('') + "||" + df_h['basename_core'].astype(str)
    expmap = df_m.dropna(subset=['exp_core']).set_index('exp_core')[label_col].to_dict()
    matched_exp = key_series[mask].map(expmap)
    df_h.loc[mask & matched_exp.notna(), 'label_mapped'] = matched_exp[matched_exp.notna()]
    df_h.loc[mask & matched_exp.notna(), 'map_method'] = 'exp+basename_core'
    print("Mapped by experiment+basename_core:", int((df_h['map_method']=='exp+basename_core').sum()))

# 5) path tail matching (tail3)
mask = df_h['label_mapped'].isna()
if mask.any():
    def tail3(fp):
        parts = fp.split('/') if isinstance(fp, str) else []
        return '/'.join(parts[-3:]) if len(parts)>=3 else fp
    df_m['tail3'] = df_m['fp_norm'].astype(str).apply(tail3)
    tail_map = df_m.dropna(subset=['tail3']).groupby('tail3')[label_col].agg(lambda s:s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
    key_tail = df_h.loc[mask,'fp_norm'].astype(str).apply(tail3)
    matched_tail = key_tail.map(tail_map)
    df_h.loc[mask & matched_tail.notna(), 'label_mapped'] = matched_tail[matched_tail.notna()]
    df_h.loc[mask & matched_tail.notna(), 'map_method'] = 'tail3'
    print("Mapped by tail3 (path-end):", int((df_h['map_method']=='tail3').sum()))

# stats and save
total_mapped = int(df_h['label_mapped'].notna().sum())
method_counts = df_h['map_method'].value_counts().to_dict()
missing = df_h[df_h['label_mapped'].isna()]
missing[['fp_norm','basename','basename_core','is_augment'] if 'is_augment' in df_h.columns else ['fp_norm','basename','basename_core']].head(500).to_csv(OUT_DIR/"missing_label_examples_v4.csv", index=False)

df_h['label'] = df_h['label_mapped']
OUT_P = OUT_DIR/"features_svm_harmonized_fixed_v4.csv"
df_h.to_csv(OUT_P, index=False)
report = {
    "harmonized_rows": int(len(df_h)),
    "total_mapped": total_mapped,
    "method_counts": method_counts,
    "missing_count": int(len(missing)),
}
with open(OUT_DIR/"fix_report_v4.json","w",encoding="utf8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print("Total mapped after strategies:", total_mapped, "of", len(df_h))
print("Map method counts:", method_counts)
print("Missing examples saved:", OUT_DIR/"missing_label_examples_v4.csv")
print("Saved fixed CSV:", OUT_P)
print("Report:", OUT_DIR/"fix_report_v4.json")
