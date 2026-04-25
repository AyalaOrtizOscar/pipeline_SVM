# final_checks.py
import pandas as pd

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.csv"
df = pd.read_csv(IN, low_memory=False)

print("Total rows:", len(df))
print("Mapped (label_fixed notna):", df['label_fixed'].notna().sum())
print("\nClass counts:")
print(df['label_fixed'].value_counts(dropna=False).to_string())

if 'mic_type' in df.columns:
    print("\nmic_type counts (top 20):")
    print(df['mic_type'].value_counts(dropna=False).head(20).to_string())

if 'experiment' in df.columns:
    print("\nexperiment counts (top 20):")
    print(df['experiment'].value_counts(dropna=False).head(20).to_string())

# duplicates check by basename
if 'basename' in df.columns:
    dups = df['basename'].duplicated().sum()
    print("\nDuplicate basenames (count):", dups)

# Save a small QA sample of current unmapped and a random sample of mapped
df[df['label_fixed'].isna()].to_csv("D:/pipeline_SVM/results/qa_harmonized/unmapped_final_sample.csv", index=False)
df[df['label_fixed'].notna()].sample(min(200, len(df[df['label_fixed'].notna()]))).to_csv("D:/pipeline_SVM/results/qa_harmonized/mapped_random_sample.csv", index=False)
print("\nSaved QA samples in results/qa_harmonized/")
