import pandas as pd
import pandas as pd
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_baseline_reextracted.csv", low_memory=False)
print("filas:", len(df))
print("cols:", len(df.columns))
print(df.columns.tolist()[:40])
print(df.head(3).to_string(index=False))
