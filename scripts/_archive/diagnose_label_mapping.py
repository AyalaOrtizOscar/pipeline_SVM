#!/usr/bin/env python3
# diagnose_label_mapping.py
import pandas as pd
from pathlib import Path
def norm(s):
    if pd.isna(s): return ""
    return str(s).replace("\\","/").strip().lower()

ROOT = Path("D:/pipeline_SVM")
HARM = ROOT/"features"/"features_svm_harmonized_for_svm.csv"
MASTER = ROOT/"features"/"features_svm_baseline.cleaned.csv"

print("Leyendo harmonized:", HARM)
df_h = pd.read_csv(HARM, low_memory=False)
print("Leyendo master:", MASTER)
df_m = pd.read_csv(MASTER, low_memory=False)

# ensure fp_norm in both
for df,name in [(df_h,"harm"),(df_m,"master")]:
    if 'fp_norm' not in df.columns:
        cand = None
        for c in ['filepath','wav_path_norm','wav_path','path']:
            if c in df.columns:
                cand = c; break
        if cand:
            df['fp_norm'] = df[cand].astype(str).apply(norm)
            print(f"Creada fp_norm en {name} desde {cand}")
        else:
            print(f"Warning: no encontré columna de ruta en {name}")

# label col detection
label_col = None
for c in ['label_clean','label']:
    if c in df_m.columns:
        label_col = c; break
print("Master label column:", label_col)
if label_col:
    n_label_vals = df_m[label_col].notna().sum()
    print("Non-null labels in master:", n_label_vals, "de", len(df_m))
    print("Unique label values (top 20):", df_m[label_col].astype(str).value_counts().head(20).to_dict())

# exact fp_norm overlap
set_h = set(df_h['fp_norm'].dropna().unique())
set_m = set(df_m['fp_norm'].dropna().unique())
inter = set_h & set_m
print("fp_norm unique: harmonized", len(set_h), "master", len(set_m), "intersection exact:", len(inter))
if len(inter)>0:
    print("Ejemplos de matches exactos (máx 10):")
    for v in list(inter)[:10]:
        print(" ", v)

# basename overlap
df_h['basename'] = df_h.get('basename', df_h.get('filepath','')).astype(str).apply(lambda s: Path(s).name.lower())
df_m['basename'] = df_m.get('basename', df_m.get('filepath','')).astype(str).apply(lambda s: Path(s).name.lower())
set_bh = set(df_h['basename'].dropna().unique())
set_bm = set(df_m['basename'].dropna().unique())
b_inter = set_bh & set_bm
print("Basename unique: harmonized", len(set_bh), "master", len(set_bm), "intersection (basename):", len(b_inter))
print("Ejemplos basename match (máx 10):", list(b_inter)[:10])

# show samples where harmonized fp_norm endswith a master fp_norm (suffix match)
suffix_matches = []
for h in list(set_h)[:2000]: # limit for speed
    for m in list(set_m)[:2000]:
        if h.endswith(m) or m.endswith(h):
            suffix_matches.append((h,m))
            break
print("Suffix-like sample matches (ejemplos hasta 10):", suffix_matches[:10])

# print small sample of harmonized rows (first 20) to inspect paths
print("\n---- harmonized sample (first 20 fp_norm) ----")
print(df_h[['fp_norm','basename']].head(20).to_string(index=False))

print("\n---- master sample (first 20 fp_norm,label) ----")
if label_col:
    print(df_m[['fp_norm','basename',label_col]].head(20).to_string(index=False))
else:
    print(df_m[['fp_norm','basename']].head(20).to_string(index=False))

print("\nHecho. Si la intersección exacta es 0 pero basename intersect >0 -> mapeo por basename puede ayudar.")
