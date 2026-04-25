# check_feature_overlap_and_missing.py
import pandas as pd
from pathlib import Path
ROOT = Path("D:/pipeline_SVM")
MASTER = ROOT/"features"/"features_svm_baseline.cleaned.csv"
AUG = ROOT/"features"/"features_augmented_desgastado.csv"

m = pd.read_csv(MASTER, low_memory=False)
a = pd.read_csv(AUG, low_memory=False)

cols_m = set(m.columns)
cols_a = set(a.columns)
only_in_master = sorted(list(cols_m - cols_a))
only_in_aug = sorted(list(cols_a - cols_m))
shared = sorted(list(cols_m & cols_a))

print("Master cols:", len(cols_m))
print("Aug cols:", len(cols_a))
print("Shared cols:", len(shared))
print("\nOnly in master (count):", len(only_in_master))
print(only_in_master[:50])
print("\nOnly in augment (count):", len(only_in_aug))
print(only_in_aug[:80])

# Missingness in the merged union
union = sorted(list(cols_m | cols_a))
dfm = m.reindex(columns=union)
dfa = a.reindex(columns=union)
comb = pd.concat([dfm, dfa], ignore_index=True)

nan_frac = comb.isna().mean().sort_values(ascending=False)
print("\nTop 20 cols by NaN fraction:")
print(nan_frac.head(20).to_string())

# Save report
nan_frac.to_csv(ROOT/"results"/"nan_fraction_per_column_union.csv")
print("\nReport saved to results/nan_fraction_per_column_union.csv")
