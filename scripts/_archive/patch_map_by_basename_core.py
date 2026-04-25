# patch_map_by_basename_core.py
import pandas as pd
from pathlib import Path

H = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.mapped_by_folder.csv")
F = Path("D:/pipeline_SVM/results/fix_harmonized_v5/features_svm_harmonized_fixed_v5.csv")
OUT = H.with_name(H.stem + ".patched_basename_core.csv")

df_h = pd.read_csv(H, low_memory=False)
df_f = pd.read_csv(F, low_memory=False)

# normalize basename_core to lower for both
def norm(x):
    return str(x).strip().lower() if pd.notna(x) else ""

df_h['basename_core_l'] = df_h.get('basename_core','').map(norm)
df_f['basename_core_l'] = df_f.get('basename_core','').map(norm)

# build basename_core -> mode(label) map from fixed
label_col = None
for c in ['label_fixed','label_mapped','label_clean','label']:
    if c in df_f.columns:
        label_col = c
        break
if label_col is None:
    print("No label column found in fixed. Exiting.")
    raise SystemExit(1)

map_df = df_f[df_f[label_col].notna()].copy()
basename_map = map_df.groupby('basename_core_l')[label_col].agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else s.iloc[0]).to_dict()

# apply for still-missing rows
missing_mask = df_h['label_fixed'].isna() | (df_h['label_fixed'].astype(str).str.lower()=='nan')
to_map_mask = missing_mask & df_h['basename_core_l'].isin(basename_map.keys())
df_h.loc[to_map_mask, 'label_fixed'] = df_h.loc[to_map_mask, 'basename_core_l'].map(basename_map)
df_h.loc[to_map_mask, 'label_map_method'] = df_h.loc[to_map_mask, 'label_map_method'].fillna('') + (';basename_core' if df_h.loc[to_map_mask,'label_map_method'].notna().any() else 'basename_core')

print("Mapped by basename_core count:", to_map_mask.sum())
print("Previously unmapped:", missing_mask.sum(), "Now unmapped:", (df_h['label_fixed'].isna()).sum())
df_h.to_csv(OUT, index=False)
print("Saved patched file:", OUT)
