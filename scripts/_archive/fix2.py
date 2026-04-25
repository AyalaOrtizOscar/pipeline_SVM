import pandas as pd
df=pd.read_csv("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.with_duration.csv", low_memory=False)
print("filas:", len(df))
print(df.columns.tolist())
print(df[['filepath','basename','label','duration']].head(20).to_string(index=False))


