import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labeled.csv", low_memory=False)
print("Total labeled:", len(df))
print(df['label_fixed'].value_counts())
print("By mic_type:\n", df['mic_type'].value_counts().head(20))
print("By experiment:\n", df['experiment'].value_counts().head(20))

