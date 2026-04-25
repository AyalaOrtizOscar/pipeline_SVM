#!/usr/bin/env python3
"""
Heuristic label mapper v5 - FIXED
- intentos de mapping conservadores
- usa mapping CSVs de augment si existen
- evita warnings de dtype y uso de iteritems
- guarda reporte y ejemplos no mapeados
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re, json, glob

ROOT = Path("D:/pipeline_SVM")
HARM_P = ROOT/"features"/"features_svm_harmonized_for_svm.csv"
MASTER_P = ROOT/"features"/"features_svm_baseline.cleaned.csv"
OUT_DIR = ROOT/"results"/"fix_harmonized_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(s):
    if pd.isna(s): return ""
    return str(s).replace("\\","/").strip().lower()

def basename_core(name):
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    stem = Path(name).stem
    # Remove common augment suffix patterns from the stem
    stem = re.sub(r'(_|-)?aug(_?auto)?[0-9]*$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'(_|-)?aug(auto)?[0-9]*$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'(_|-)?aug[0-9]+', '', stem, flags=re.IGNORECASE)
    # also strip trailing auto tags
    stem = re.sub(r'(_|-)?auto[0-9]*$', '', stem, flags=re.IGNORECASE)
    return stem + Path(name).suffix

def extract_digits(s):
    if not isinstance(s, str): return ""
    m = re.search(r'(\d{3,7})', s)
    if m:
        return m.group(1).lstrip('0') or m.group(1)
    return ""

print("Leyendo harmonized:", HARM_P)
df_h = pd.read_csv(HARM_P, low_memory=False)
print("Leyendo master:", MASTER_P)
df_m = pd.read_csv(MASTER_P, low_memory=False)

# prepare fp_norm & basename
for df,name in [(df_h,'harm'),(df_m,'master')]:
    if 'fp_norm' not in df.columns:
        cand = next((c for c in ['filepath','wav_path_norm','wav_path','path'] if c in df.columns), None)
        if cand:
            df['fp_norm'] = df[cand].astype(str).apply(norm)
            print(f"Creada fp_norm en {name} desde {cand}")
    if 'basename' not in df.columns:
        df['basename'] = df.get('basename', df.get('filepath','')).astype(str).apply(lambda s: Path(s).name.lower())

# label col in master
label_col = next((c for c in ['label_clean','label'] if c in df_m.columns), None)
if label_col is None:
    raise SystemExit("ERROR: master no tiene columna de label.")

# normalize basenames
df_m['basename'] = df_m['basename'].astype(str).str.lower()
df_h['basename'] = df_h['basename'].astype(str).str.lower()
df_m['basename_core'] = df_m['basename'].apply(basename_core)
df_h['basename_core'] = df_h['basename'].apply(basename_core)

# initialize mapping columns with object dtype to avoid dtype warnings
df_h['label_mapped'] = pd.Series([None]*len(df_h), index=df_h.index, dtype=object)
df_h['map_method'] = pd.Series([""]*len(df_h), index=df_h.index, dtype=object)

# load augmented mapping CSVs if exist (augmented_mapping.csv)
aug_map = {}
for csvp in glob.glob(str(ROOT/"augmented"/"**"/"augmented_mapping.csv"), recursive=True):
    try:
        dfa = pd.read_csv(csvp, low_memory=False)
        cols = [c.lower() for c in dfa.columns]
        orig_col = next((c for c in dfa.columns if 'orig' in c.lower() or 'source' in c.lower() or 'original' in c.lower()), None)
        aug_col = next((c for c in dfa.columns if 'aug' in c.lower() or 'augmented' in c.lower() or 'new' in c.lower()), None)
        if orig_col and aug_col:
            for _, r in dfa[[orig_col, aug_col]].dropna().iterrows():
                a = Path(str(r[aug_col])).name.lower()
                o = str(r[orig_col]).replace("\\","/").strip().lower()
                aug_map[a] = o
    except Exception:
        pass
print("Augmented mapping entries found:", len(aug_map))

# helper: create master lookups
m_by_fp = df_m.dropna(subset=['fp_norm']).set_index('fp_norm')[label_col].to_dict()
m_by_basename = df_m.dropna(subset=['basename']).groupby('basename')[label_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
m_by_core = df_m.dropna(subset=['basename_core']).groupby('basename_core')[label_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()

# 1) exact fp_norm
mask_fp = df_h['fp_norm'].astype(str).map(lambda x: x in m_by_fp)
if mask_fp.any():
    df_h.loc[mask_fp, 'label_mapped'] = df_h.loc[mask_fp, 'fp_norm'].map(m_by_fp)
    df_h.loc[mask_fp, 'map_method'] = 'fp_exact'
print("Mapped by exact fp_norm:", int((df_h['map_method']=='fp_exact').sum()))

# 2) basename exact
mask = df_h['label_mapped'].isna()
if mask.any():
    matched = df_h.loc[mask,'basename'].map(m_by_basename)
    take = matched.notna()
    df_h.loc[mask & take, 'label_mapped'] = matched[take]
    df_h.loc[mask & take, 'map_method'] = 'basename_exact'
print("Mapped by basename_exact:", int((df_h['map_method']=='basename_exact').sum()))

# 3) basename_core
mask = df_h['label_mapped'].isna()
if mask.any():
    matched_core = df_h.loc[mask,'basename_core'].map(m_by_core)
    take = matched_core.notna()
    df_h.loc[mask & take, 'label_mapped'] = matched_core[take]
    df_h.loc[mask & take, 'map_method'] = 'basename_core'
print("Mapped by basename_core:", int((df_h['map_method']=='basename_core').sum()))

# 4) augmented_mapping CSV (if we have augmented->original)
if aug_map:
    mask = df_h['label_mapped'].isna()
    if mask.any():
        # create series of original filepaths (if known) aligned with harmonized index
        aug_names = df_h.loc[mask,'basename']
        # map augmented basename -> original filepath (if exists)
        mapped_orig_fp = aug_names.map(lambda n: aug_map.get(n, None))
        # try resolve to master label by fp_norm or basename
        resolved = []
        for idx, orig in mapped_orig_fp.items():
            lab = None
            if isinstance(orig, str) and orig:
                o_norm = orig.replace("\\","/").strip().lower()
                if o_norm in m_by_fp:
                    lab = m_by_fp[o_norm]
                else:
                    bn = Path(o_norm).name
                    lab = m_by_basename.get(bn)
            resolved.append(lab)
        resolved_series = pd.Series(resolved, index=mapped_orig_fp.index)
        take = resolved_series.notna()
        df_h.loc[mask & take, 'label_mapped'] = resolved_series[take]
        df_h.loc[mask & take, 'map_method'] = 'augmented_mapping_csv'
print("Mapped by augmented_mapping_csv:", int((df_h['map_method']=='augmented_mapping_csv').sum()))

# 5) digits-match
mask = df_h['label_mapped'].isna()
if mask.any():
    master_bns = df_m['basename'].astype(str).tolist()
    lab_by_bn = df_m.set_index('basename')[label_col].to_dict()
    def try_digit_map(bn):
        d = extract_digits(bn)
        if not d:
            return None
        for mb in master_bns:
            if d in mb:
                return lab_by_bn.get(mb)
        return None
    candidate = df_h.loc[mask,'basename_core'].map(try_digit_map)
    take = candidate.notna()
    df_h.loc[mask & take, 'label_mapped'] = candidate[take]
    df_h.loc[mask & take, 'map_method'] = 'digits_in_name'
print("Mapped by digits_in_name:", int((df_h['map_method']=='digits_in_name').sum()))

# 6) prefix-match (conservador)
mask = df_h['label_mapped'].isna()
if mask.any():
    master_set = set(df_m['basename'].astype(str).tolist())
    # to speed up, create a small index by first 6 chars of master basenames
    prefix_index = {}
    for mb in master_set:
        key = mb[:6]
        prefix_index.setdefault(key, []).append(mb)
    def try_prefix(bn):
        s = Path(bn).stem
        key = s[:6]
        candidates = prefix_index.get(key, [])
        for mb in candidates:
            if mb.startswith(s[:max(3, len(s)-2)]):
                return df_m.set_index('basename')[label_col].get(mb)
        return None
    cand_pref = df_h.loc[mask,'basename'].map(try_prefix)
    take = cand_pref.notna()
    df_h.loc[mask & take, 'label_mapped'] = cand_pref[take]
    df_h.loc[mask & take, 'map_method'] = 'prefix_match'
print("Mapped by prefix_match:", int((df_h['map_method']=='prefix_match').sum()))

# 7) tail3
mask = df_h['label_mapped'].isna()
if mask.any():
    def tail3(fp):
        parts = fp.split('/') if isinstance(fp, str) else []
        return '/'.join(parts[-3:]) if len(parts)>=3 else fp
    df_m['tail3'] = df_m['fp_norm'].astype(str).apply(tail3)
    tailmap = df_m.dropna(subset=['tail3']).groupby('tail3')[label_col].agg(lambda s:s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()
    cand_tail = df_h.loc[mask,'fp_norm'].astype(str).apply(tail3).map(tailmap)
    take = cand_tail.notna()
    df_h.loc[mask & take, 'label_mapped'] = cand_tail[take]
    df_h.loc[mask & take, 'map_method'] = 'tail3'
print("Mapped by tail3:", int((df_h['map_method']=='tail3').sum()))

# final stats and save
total_mapped = int(df_h['label_mapped'].notna().sum())
method_counts = df_h['map_method'].value_counts().to_dict()
missing = df_h[df_h['label_mapped'].isna()]
missing[['fp_norm','basename','basename_core']].head(1000).to_csv(OUT_DIR/"missing_label_examples_v5.csv", index=False)

# create final conservative label (only where mapped)
df_h['label_fixed'] = df_h['label_mapped']

OUT_P = OUT_DIR/"features_svm_harmonized_fixed_v5.csv"
df_h.to_csv(OUT_P, index=False)

report = {
    "harmonized_rows": int(len(df_h)),
    "total_mapped": total_mapped,
    "method_counts": method_counts,
    "missing_count": int(len(missing)),
    "note": "Review missing_label_examples_v5.csv; prefer mapping from augmented_mapping CSV if available."
}
with open(OUT_DIR/"fix_report_v5.json","w",encoding="utf8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print("Total mapped after heuristics:", total_mapped, "of", len(df_h))
print("Method counts:", method_counts)
print("Missing examples saved:", OUT_DIR/"missing_label_examples_v5.csv")
print("Saved fixed CSV:", OUT_P)
print("Report:", OUT_DIR/"fix_report_v5.json")
