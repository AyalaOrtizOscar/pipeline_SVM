# check_nan_frac.py
import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.csv", low_memory=False)
nan_frac = df.isna().mean().sort_values(ascending=False)
print(nan_frac.head(60).to_string())
nan_frac.to_csv("D:/pipeline_SVM/results/qa_harmonized/nan_fraction_top_all.csv")
print("Saved nan fraction CSV to results/qa_harmonized/nan_fraction_top_all.csv")
