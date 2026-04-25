# filter_keep_only_originals_fixed.py
import pandas as pd
from pathlib import Path
import re
import os
import sys

IN_MATCHED = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios.matched.csv")
ORIG_FEATURES = Path("D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv")
OUT = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.csv")

print("Leyendo matched:", IN_MATCHED)
if not IN_MATCHED.exists():
    print("ERROR: no existe", IN_MATCHED); sys.exit(1)

df = pd.read_csv(IN_MATCHED, low_memory=False)

# ------------- detectar columna que contiene rutas a wav/mel -------------
wavcol = None
candidates = ["wav_path","wav","filepath","mel_path","mel","melpath","path","file"]
for c in candidates:
    if c in df.columns:
        wavcol = c
        break
if wavcol is None:
    # fallback: buscar cualquier columna con 'path' en el nombre
    for c in df.columns:
        if "path" in c.lower() or "file" in c.lower():
            wavcol = c
            break

if wavcol is None:
    print("ERROR: no encuentro columna con ruta de audio en el CSV matched. Columnas disponibles:")
    print(df.columns.tolist())
    raise SystemExit(1)

print("Usando columna para rutas:", wavcol)

def norm_path(x):
    if pd.isna(x):
        return ""
    p = str(x).replace("/", "\\")
    p = re.sub(r"\\+", "\\\\", p)
    return p.strip().lower()

df['wav_path_norm'] = df[wavcol].astype(str).apply(norm_path)

# asegurar columna basename
if 'basename' not in df.columns:
    df['basename'] = df['wav_path_norm'].apply(lambda p: Path(p).name.lower() if p else "")

# ------------- excluir augmentaciones por patrones -------------
aug_patterns = [
    r"_aug\b", r"_aug\d+\b", r"\\augmented_wavs\\", r"\\augmented\\", r"\\aug\\",
    r"\\augmented_", r"augmented_wavs", r"\\aug\\"
]
aug_regex = re.compile("|".join(aug_patterns), flags=re.IGNORECASE)
mask_no_aug = ~df['wav_path_norm'].apply(lambda s: bool(aug_regex.search(s)))

# ------------- preferir audios "limpios" o en D:/dataset -------------
mask_d_dataset = df['wav_path_norm'].str.contains(r"d:\\\\dataset|d:.*dataset", na=False)
mask_limpias = df['wav_path_norm'].str.contains(r"\\con falla limpios\\|\\sin falla limpios\\|\\limp|\\clean\\", na=False)
keep = mask_no_aug & (mask_limpias | mask_d_dataset)
filtered = df[keep].copy()

if len(filtered) == 0:
    print("Filtro estricto devolvió 0 filas. Aplico filtro relajado: solo excluir aug y mantener D:/dataset")
    filtered = df[mask_no_aug & mask_d_dataset].copy()

print("Filas tras filtro inicial:", len(filtered))

# ------------- intentar merge con ORIG_FEATURES si faltan meta columnas -------------
meta_cols_needed = ['filepath','label','label_clean','mic_type','experiment','duration']
missing_meta = [c for c in meta_cols_needed if c not in filtered.columns or filtered[c].isna().all()]

if missing_meta:
    print("Faltan columnas meta en filtered (o están todas NaN):", missing_meta)
    if ORIG_FEATURES.exists():
        print("Intento merge con", ORIG_FEATURES)
        orig = pd.read_csv(ORIG_FEATURES, low_memory=False)
        # asegurar basename en orig
        if 'basename' not in orig.columns and 'filepath' in orig.columns:
            orig['basename'] = orig['filepath'].astype(str).apply(lambda p: Path(p).name.lower() if pd.notna(p) else "")
        # columnas útiles a tomar
        take = ['basename'] + [c for c in meta_cols_needed if c in orig.columns]
        orig_small = orig[take].drop_duplicates('basename')
        filtered = filtered.merge(orig_small, on='basename', how='left', suffixes=('','_fromorig'))
        # rellenar metadatos faltantes desde *_fromorig
        for col in meta_cols_needed:
            fromcol = col + '_fromorig'
            if col not in filtered.columns and fromcol in filtered.columns:
                filtered[col] = filtered[fromcol]
            elif col in filtered.columns and fromcol in filtered.columns:
                filtered[col] = filtered[col].fillna(filtered[fromcol])
        still_missing = [c for c in meta_cols_needed if c not in filtered.columns or filtered[c].isna().all()]
        if still_missing:
            print("Después de merge, siguen faltando o vacías:", still_missing)
    else:
        print("No existe archivo original para merge:", ORIG_FEATURES)

# ------------- preparar lista de features numéricas -------------
# meta finales que queremos mantener (si existen)
final_meta_candidates = ['filepath','label','label_clean','mic_type','experiment','duration','wav_path_norm']
final_meta = [c for c in final_meta_candidates if c in filtered.columns]

# identificar columnas extra que sean features (no meta ni helper)
helpers = set(['wav_path_norm','basename'])
feat_cols = [c for c in filtered.columns if c not in final_meta and c not in helpers and not c.endswith('_fromorig')]
# forzar numeric
for c in feat_cols:
    filtered[c] = pd.to_numeric(filtered[c], errors='coerce')

# quitar columnas todas NaN
feat_cols_clean = [c for c in feat_cols if not filtered[c].isna().all()]
print("Features retenidas:", len(feat_cols_clean))

# ------------- decidir columna para drop_duplicates -------------
if 'basename' in filtered.columns and filtered['basename'].astype(bool).any():
    subset_col = 'basename'
elif 'filepath' in filtered.columns and filtered['filepath'].astype(bool).any():
    subset_col = 'filepath'
elif 'wav_path_norm' in filtered.columns and filtered['wav_path_norm'].astype(bool).any():
    subset_col = 'wav_path_norm'
else:
    subset_col = None

final_cols = final_meta + feat_cols_clean
final_cols = [c for c in final_cols if c in filtered.columns]
if subset_col:
    try:
        out_df = filtered[final_cols].drop_duplicates(subset=subset_col)
    except KeyError:
        print("Warning: subset_col", subset_col, "no encontrada en columnas finales. Haciendo drop_duplicates sin subset.")
        out_df = filtered[final_cols].drop_duplicates()
else:
    out_df = filtered[final_cols].drop_duplicates()

OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)
print("Guardado:", OUT, "filas:", len(out_df))

# mostrar distribuciones relevantes
for col in ('label','label_clean','mic_type'):
    if col in out_df.columns:
        print(f"\nDistribución {col}:\n", out_df[col].value_counts(dropna=False).to_string())
