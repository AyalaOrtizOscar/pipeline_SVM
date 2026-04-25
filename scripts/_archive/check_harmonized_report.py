#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

MASTER = Path("D:/pipeline_SVM/features/features_svm_baseline_reextracted.csv")
HARMON = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.csv")
OUTDIR = Path("D:/pipeline_SVM/results/qa_harmonized")
OUTDIR.mkdir(parents=True, exist_ok=True)

def load(p):
    print("Leyendo:", p)
    return pd.read_csv(p, low_memory=False)

df_master = load(MASTER)
df_h = load(HARMON)

print("Master shape:", df_master.shape)
print("Harmonized shape:", df_h.shape)

# label col guess
label_col = 'label_clean' if 'label_clean' in df_h.columns else ('label' if 'label' in df_h.columns else None)
print("Label column:", label_col)

# counts
if label_col:
    vc = df_h[label_col].fillna("").astype(str).str.strip().value_counts(dropna=False)
    print("\nLabel counts:\n", vc.to_string())
    vc.to_csv(OUTDIR/"label_counts.csv")

# Augment detection by basename pattern
if 'basename' in df_h.columns:
    aug_mask = df_h['basename'].astype(str).str.contains('_aug|aug_', case=False, na=False) | df_h['basename'].astype(str).str.contains('augauto|aug-auto', case=False, na=False)
    print("Detected augment rows by basename pattern:", aug_mask.sum(), "of", len(df_h))
    df_h.loc[aug_mask, 'is_augment_detected'] = 1
else:
    aug_mask = pd.Series([False]*len(df_h))
    print("No 'basename' column to detect augment rows.")

# nan fraction per col
nan_frac = df_h.isna().mean().sort_values(ascending=False)
nan_frac.head(40).to_csv(OUTDIR/"nan_fraction_top40.csv")
print("\nTop NaN fractions saved:", OUTDIR/"nan_fraction_top40.csv")

# numeric summary
num_cols = df_h.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric cols count:", len(num_cols))
if len(num_cols)>0:
    df_h[num_cols].describe().T.to_csv(OUTDIR/"numeric_describe.csv")
    print("Numeric describe saved:", OUTDIR/"numeric_describe.csv")

# experiments / mic types
for c in ['experiment','mic_type','group_tmp']:
    if c in df_h.columns:
        s = df_h[c].fillna('').astype(str).str.strip().value_counts()
        s.to_csv(OUTDIR/f"{c}_counts.csv")
        print(f"Saved {c} counts to {OUTDIR/f'{c}_counts.csv'}")

# duplicates check by basename or filepath
dupe_by = 'basename' if 'basename' in df_h.columns else 'filepath'
dupes = df_h[df_h.duplicated(subset=[dupe_by], keep=False)]
print("Duplicates count by", dupe_by, ":", len(dupes))
dupes.head(200).to_csv(OUTDIR/"duplicates_sample.csv", index=False)

# Save small integrity report
report = {
    "master_shape": df_master.shape,
    "harmon_shape": df_h.shape,
    "n_aug_detected_by_basename": int(aug_mask.sum()),
    "label_col": label_col,
    "n_numeric": len(num_cols),
    "top_nan_cols": nan_frac.head(20).to_dict()
}
import json
with open(OUTDIR/"integrity_report.json","w",encoding="utf8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print("Integrity report saved to", OUTDIR/"integrity_report.json")
