import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.csv", low_memory=False)
print("Total rows:", len(df))
print("Mapped count:", df['label_fixed'].notna().sum())
print(df['label_map_method'].value_counts(dropna=False).head(20))


