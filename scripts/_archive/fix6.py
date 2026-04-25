import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_baseline.cleaned.csv", low_memory=False)
minor = df[df['label_clean']=='desgastado']
print(minor[['filepath','wav_path_norm','experiment']].head(30).to_string(index=False))
print("Total desgastado:", len(minor))

