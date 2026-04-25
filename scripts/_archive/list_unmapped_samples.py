# list_unmapped_samples.py
import pandas as pd

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.mapped_by_folder.csv"
OUT = "D:/pipeline_SVM/results/fix_harmonized_v5/unmapped_examples.csv"

df = pd.read_csv(IN, low_memory=False)
unmapped = df[df['label_fixed'].isna()].copy()
print("Total rows:", len(df))
print("Unmapped rows:", len(unmapped))
# keep helpful columns
cols = ['filepath','basename','basename_core','label_map_method','fp_norm','mic_type','experiment']
to_save = [c for c in cols if c in unmapped.columns]
unmapped[to_save].to_csv(OUT, index=False)
print("Saved unmapped examples to:", OUT)
print(unmapped[to_save].head(40).to_string(index=False))
