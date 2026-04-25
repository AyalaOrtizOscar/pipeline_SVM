import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv", low_memory=False)

print("Total filas:", len(df))
print("Filas con label_fixed:", df['label_fixed'].notna().sum())
print("Unmapped (sin label):", df['label_fixed'].isna().sum())

# conteo por label
print(df['label_fixed'].value_counts(dropna=False))

# conteo por método de mapeo
print(df['label_map_method'].value_counts(dropna=False))

# revisa ejemplos sin label
print(df[df['label_fixed'].isna()][['filepath','basename','basename_core','label_map_method']].head(20).to_string(index=False))


