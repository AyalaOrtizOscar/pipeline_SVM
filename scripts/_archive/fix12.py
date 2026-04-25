df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.csv", low_memory=False)
nan_frac = df.isna().mean().sort_values(ascending=False)
print(nan_frac.head(40).to_string())


