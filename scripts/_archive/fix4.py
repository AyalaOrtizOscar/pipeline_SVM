import pandas as pd, pathlib
orig = pd.read_csv("D:/pipeline_SVM/features/features_svm_baseline.csv", low_memory=False)
aug = pd.read_csv("D:/pipeline_SVM/features/features_svm_baseline_augmented.csv", low_memory=False)
combined = pd.concat([orig, aug], ignore_index=True)
combined.to_csv("D:/pipeline_SVM/features/features_svm_baseline_with_aug.csv", index=False)
print("guardado combined:", len(combined))

