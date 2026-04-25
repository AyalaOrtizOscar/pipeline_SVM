#!/usr/bin/env python3
# finalize_and_dedupe_fixed.py
import pandas as pd
from pathlib import Path

H = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.csv")
OUT_L = H.with_name(H.stem + ".final_labeled.csv")
OUT_DED = H.with_name(H.stem + ".final_dedup_by_basename.csv")

print("Leyendo:", H)
df = pd.read_csv(H, low_memory=False)

# ensure is_augment column (avoid capturing groups warning)
if 'is_augment' not in df.columns:
    # use non-capturing groups and simpler pattern
    df['is_augment'] = df['basename'].astype(str).str.contains(r'(?:_|-)?(?:aug|auto)', case=False, regex=True).fillna(False)

# create boolean helper for having a label
df['has_label'] = df['label_fixed'].notna()

# Save labeled-only
df_l = df[df['has_label']].copy()
df_l.to_csv(OUT_L, index=False)
print("Saved labeled-only:", OUT_L, "Count:", len(df_l))

# sort so that for each basename we prefer original (is_augment False) and rows that have label (has_label True)
# ascending: basename asc, is_augment asc (False first), has_label desc (True first -> so ascending=False)
df_sorted = df.sort_values(by=['basename', 'is_augment', 'has_label'], ascending=[True, True, False])

# keep first per basename
df_nd = df_sorted.groupby('basename', group_keys=False).first().reset_index()

df_nd.to_csv(OUT_DED, index=False)
print("Saved deduped by basename:", OUT_DED, "Count:", len(df_nd))

# cleanup temp column if you want (not necessary when saved separately)
# df.drop(columns=['has_label'], inplace=True)
