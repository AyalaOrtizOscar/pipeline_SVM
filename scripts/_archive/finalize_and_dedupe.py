# finalize_and_dedupe.py
import pandas as pd
from pathlib import Path
H = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.csv")
OUT_L = H.with_name(H.stem + ".final_labeled.csv")
OUT_DED = H.with_name(H.stem + ".final_dedup_by_basename.csv")

df = pd.read_csv(H, low_memory=False)

# ensure is_augment column
if 'is_augment' not in df.columns:
    df['is_augment'] = df['basename'].astype(str).str.contains(r'(_|-)?aug|auto', case=False, regex=True).fillna(False)

# keep only labeled
df_l = df[df['label_fixed'].notna()].copy()
df_l.to_csv(OUT_L, index=False)
print("Saved labeled-only:", OUT_L, "Count:", len(df_l))

# dedupe: prefer not-augment, prefer rows with label (already filtered), keep first such sorted by is_augment then any tie-breaker
df_sorted = df.sort_values(by=['basename','is_augment', df['label_fixed'].notna()], ascending=[True, True, False])
df_nd = df_sorted.groupby('basename', group_keys=False).first().reset_index()
df_nd.to_csv(OUT_DED, index=False)
print("Saved deduped by basename:", OUT_DED, "Count:", len(df_nd))
