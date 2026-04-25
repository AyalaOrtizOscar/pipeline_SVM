# map_unmapped_by_folder.py
import pandas as pd
import re

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv"
OUT_FULL = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.mapped_by_folder.csv"
OUT_ONLY = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labeled.after_folder_map.csv"

df = pd.read_csv(IN, low_memory=False)
print("Total rows:", len(df))
print("Mapped before:", df['label_fixed'].notna().sum())
print("Unmapped before:", df['label_fixed'].isna().sum())

# normalize filepath
df['fp_low'] = df['filepath'].astype(str).str.lower()

# define mapping rules: key substring -> label
rules = {
    'sin_desgaste': 'sin_desgaste',
    'sin falla': 'sin_desgaste',
    'sin_falla': 'sin_desgaste',
    # add others only if you are sure:
    # 'con_falla': 'medianamente_desgastado',   # <--- only enable if you are sure
    # 'desgastado': 'desgastado',
}

applied = 0
for key, label in rules.items():
    mask = df['label_fixed'].isna() & df['fp_low'].str.contains(key, na=False)
    count = mask.sum()
    if count>0:
        df.loc[mask, 'label_fixed'] = label
        df.loc[mask, 'label_map_method'] = df.loc[mask, 'label_map_method'].fillna('') + (';folder_map' if df.loc[mask,'label_map_method'].notna().any() else 'folder_map')
        applied += count
        print(f"Applied rule '{key}' -> {label}, matched: {count}")

print("Total newly mapped by folder rules:", applied)
print("Mapped after:", df['label_fixed'].notna().sum())
print("Unmapped after:", df['label_fixed'].isna().sum())

df.to_csv(OUT_FULL, index=False)
df[df['label_fixed'].notna()].to_csv(OUT_ONLY, index=False)
print("Saved:", OUT_FULL)
print("Saved labeled-only:", OUT_ONLY)
