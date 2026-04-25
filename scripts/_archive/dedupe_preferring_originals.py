# dedupe_preferring_originals.py
import pandas as pd
from pathlib import Path

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv"
OUT = "D:/pipeline_SVM/features/features_svm_harmonized_dedup_by_basename.csv"

df = pd.read_csv(IN, low_memory=False)
print("Input rows:", len(df))

# create basename if missing
if 'basename' not in df.columns:
    df['basename'] = df['filepath'].astype(str).map(lambda p: Path(p).name if p and p==p else "")

# detect augment: prefer any explicit column 'is_augment' if present
if 'is_augment' in df.columns:
    df['is_augment'] = df['is_augment'].astype(bool)
else:
    # heuristic: filename contains _aug, _auto, -aug, aug_auto, etc.
    df['is_augment'] = df['basename'].str.contains(r'(_|-)?aug|auto', case=False, regex=True).fillna(False)

# create helper column: labeled present (1) or not (0)
df['label_present'] = df['label_fixed'].notna().astype(int)

# sort so that for each basename the preferred row comes first:
# - prefer rows that have label (label_present desc)
# - prefer originals (is_augment False before True)
df_sorted = df.sort_values(by=['basename', 'label_present', 'is_augment'], ascending=[True, False, True])

# keep first per basename
df_nd = df_sorted.groupby('basename', group_keys=False).first().reset_index(drop=True)

print("Rows after dedupe (unique basename):", len(df_nd))
df_nd.to_csv(OUT, index=False)
print("Saved deduped:", OUT)
