# filter_keep_only_originals.py (versión robusta)
import pandas as pd
from pathlib import Path
import re
import os

IN_MATCHED = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios.matched.csv")
ORIG_FEATURES = Path("D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv")
OUT = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.csv")

print("Leyendo matched:", IN_MATCHED)
df = pd.read_csv(IN_MATCHED, low_memory=False)

# detect column that holds wav path
wavcol = None
for c in ("wav_path", "wav", "filepath", "mel_path", "mel"):
    if c in df.columns:
        wavcol = c
        break
if wavcol is None:
    raise SystemExit("No encuentro columna con ruta de wav en el CSV matched. Columnas: " + ", ".join(df.columns))

# normalize paths (use backslashes for matching)
def norm_path(x):
    if pd.isna(x):
        return ""
    p = str(x).replace("/", "\\")
    # collapse repeated separators
    p = re.sub(r"\\+", "\\\\", p)
    return p.strip().lower()

df['wav_path_norm'] = df[wavcol].astype(str).apply(norm_path)

# ensure basename column exists
if 'basename' not in df.columns:
    df['basename'] = df['wav_path_norm'].apply(lambda p: Path(p).name.lower() if p else "")

# 1) excluir augmentaciones por patrón (nombres y carpetas)
aug_patterns = ["_aug", "_aug1", "_aug2", "_aug3", "_aug4", "\\augmented_wavs\\", "\\augmented\\", "\\aug\\", "augmented_wavs", "augmented"]
aug_regex = re.compile("|".join(re.escape(p) for p in aug_patterns), flags=re.IGNORECASE)
mask_no_aug = ~df['wav_path_norm'].apply(lambda s: bool(aug_regex.search(s)))

# 2) preferir rutas bajo D:/dataset y/o que contengan 'limp' o 'clean'
mask_d_dataset = df['wav_path_norm'].str.contains(r"d:\\\\dataset|d:.*dataset", na=False)
mask_limpias = df['wav_path_norm'].str.contains(r"\\con falla limpios\\|\\sin falla limpios\\|\\limp|\\clean\\", na=False)

# combine rule: not augmented AND (in 'limp' path OR in D:/dataset)
keep = mask_no_aug & (mask_limpias | mask_d_dataset)
filtered = df[keep].copy()

# if empty, relax (only exclude augment and keep D:/dataset)
if len(filtered) == 0:
    print("Filtro estricto devolvió 0 filas. Aplico filtro relajado: solo excluir aug y mantener D:/dataset")
    filtered = df[mask_no_aug & mask_d_dataset].copy()

# 3) asegurarse de metadata label/mic_type/duration: si faltan, intentar merge con features_svm_baseline_cleaned.csv
meta_cols_needed = ['filepath', 'label', 'label_clean', 'mic_type', 'experiment', 'duration']
missing_meta = [c for c in meta_cols_needed if c not in filtered.columns or filtered[c].isna().all()]
if missing_meta:
    print("Faltan columnas meta en filtered (o están todas NaN):", missing_meta)
    if ORIG_FEATURES.exists():
        print("Intento merge con", ORIG_FEATURES)
        orig = pd.read_csv(ORIG_FEATURES, low_memory=False)
        if 'filepath' in orig.columns:
            orig['basename'] = orig['filepath'].astype(str).apply(lambda p: Path(p).name.lower() if pd.notna(p) else "")
        use_cols = ['basename'] + [c for c in meta_cols_needed if c in orig.columns]
        orig_small = orig[use_cols].drop_duplicates('basename')
        filtered = filtered.merge(orig_small, on='basename', how='left', suffixes=('','_fromorig'))
        # fill missing from *_fromorig
        for col in meta_cols_needed:
            if col not in filtered.columns and (col + "_fromorig") in filtered.columns:
                filtered[col] = filtered[col + "_fromorig"]
        # also fill NaNs in existing columns from fromorig
        for col in meta_cols_needed:
            if col in filtered.columns and (col + "_fromorig") in filtered.columns:
                filtered[col] = filtered[col].fillna(filtered[col + "_fromorig"])
        still_missing = [c for c in meta_cols_needed if c not in filtered.columns or filtered[c].isna().all()]
        if still_missing:
            print("Después de merge, siguen faltando o vacías:", still_missing)
    else:
        print("No existe archivo original para merge:", ORIG_FEATURES)

# 4) elegir columnas finales: metadata + features numéricas (coerce)
final_meta_candidates = ['filepath','label','label_clean','mic_type','experiment','duration','wav_path']
final_meta = [c for c in final_meta_candidates if c in filtered.columns]
helpers = set(['wav_path_norm','basename'])
# feature candidates: exclude meta, helpers and any columns *_fromorig
feat_cols = [c for c in filtered.columns if c not in final_meta and c not in helpers and not c.endswith('_fromorig')]
# coerce numeric for feature cols
for c in feat_cols:
    filtered[c] = pd.to_numeric(filtered[c], errors='coerce')

# drop columns that are all NaN
feat_cols_clean = [c for c in feat_cols if not filtered[c].isna().all()]
print("Features retenidas:", len(feat_cols_clean))

# decide subset for drop_duplicates
if 'basename' in filtered.columns and filtered['basename'].astype(bool).any():
    subset_col = 'basename'
elif 'filepath' in filtered.columns and filtered['filepath'].astype(bool).any():
    subset_col = 'filepath'
else:
    subset_col = None

final_cols = final_meta + feat_cols_clean
final_cols = [c for c in final_cols if c in filtered.columns]

if subset_col:
    out_df = filtered[final_cols].drop_duplicates(subset=subset_col)
else:
    out_df = filtered[final_cols].drop_duplicates()

out_df.to_csv(OUT, index=False)
print("Guardado:", OUT, "filas:", len(out_df))

for col in ('label','label_clean','mic_type'):
    if col in out_df.columns:
        print(f"Distribución {col}:\n", out_df[col].value_counts(dropna=False).to_string())
