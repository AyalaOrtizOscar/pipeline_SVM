# harmonize_master_augmented.py
import pandas as pd
from pathlib import Path
ROOT = Path("D:/pipeline_SVM")
MASTER = ROOT / "features" / "features_svm_baseline.cleaned.csv"   # o features_svm_baseline.csv
AUG   = ROOT / "features" / "features_augmented_desgastado.csv"
OUT   = ROOT / "features" / "features_svm_with_augmented.harmonized.csv"
REPORT= ROOT / "results" / "harmonize_report.csv"

print("Leyendo master:", MASTER)
df_m = pd.read_csv(MASTER, low_memory=False)
print("Leyendo augment:", AUG)
df_a = pd.read_csv(AUG, low_memory=False)

# 1) Normalizaciones de nombre comunes (ajusta según tu caso)
rename_map_master = {
    "zcr": "zcr_mean",    # si master tiene zcr en vez de zcr_mean
    "rms": "rms_mean",
    "duration_s": "duration_s",  # keep
}
df_m = df_m.rename(columns=rename_map_master)

# 2) Asegurar columnas meta mínimas
for c in ["label","label_clean","filepath","basename","experiment","mic_type","orig"]:
    if c not in df_m.columns:
        df_m[c] = pd.NA
    if c not in df_a.columns:
        df_a[c] = pd.NA

# 3) union de columnas
all_cols = sorted(set(df_m.columns).union(set(df_a.columns)))
print("Total columnas (union):", len(all_cols))

# 4) Añadir columnas faltantes en cada DF con NaN
for c in all_cols:
    if c not in df_m.columns:
        df_m[c] = pd.NA
    if c not in df_a.columns:
        df_a[c] = pd.NA

# 5) Opcional: Propagar metadata desde master a augment si tienes mapping (por basename)
# if 'basename' in df_m.columns and 'basename' in df_a.columns:
#     mapping = df_m.set_index('basename')[['label','label_clean','experiment','mic_type']].to_dict('index')
#     def fill_meta(row):
#         b = row.get('basename')
#         if pd.isna(row.get('label')) and b in mapping:
#             row['label'] = mapping[b]['label']
#             row['label_clean'] = mapping[b]['label_clean']
#             row['experiment'] = mapping[b]['experiment']
#             row['mic_type'] = mapping[b]['mic_type']
#         return row
#     df_a = df_a.apply(fill_meta, axis=1)

# 6) Concat
df_all = pd.concat([df_m[all_cols], df_a[all_cols]], ignore_index=True, sort=False)

# 7) Report NaN fractions
nan_frac = df_all.isna().mean().sort_values(ascending=False)
report = pd.DataFrame({
    "col": nan_frac.index,
    "nan_frac": nan_frac.values,
    "dtype": [str(df_all[c].dtype) for c in nan_frac.index]
})
report.to_csv(REPORT, index=False)
print("Report saved to:", REPORT)

# 8) Drop all-NaN cols and cols with > threshold (e.g. 50%)
all_nan = report[report['nan_frac'] >= 1.0]['col'].tolist()
high_nan = report[(report['nan_frac']>0.5) & (report['nan_frac']<1.0)]['col'].tolist()

print("Columns all-NaN (removed):", all_nan)
print("Columns >50% NaN (removed):", high_nan)

to_drop = set(all_nan + high_nan)
df_all_clean = df_all.drop(columns=list(to_drop), errors='ignore')

# 9) Optional: keep only numeric features + essential meta
meta_keep = ['filepath','basename','label','label_clean','experiment','mic_type','duration_s','orig','wav_path_norm']
num = df_all_clean.select_dtypes(include=['number']).columns.tolist()
final_cols = meta_keep + [c for c in num if c not in meta_keep]
final_cols = [c for c in final_cols if c in df_all_clean.columns]

print("Final numeric features:", len(final_cols))
df_all_clean[final_cols].to_csv(OUT, index=False)
print("Saved harmonized CSV:", OUT)
