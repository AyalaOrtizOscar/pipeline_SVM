# match_features_to_clean_wavs.py
import pandas as pd
from pathlib import Path

FEATURES = Path("D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv")
CLEAN_LIST = Path("D:/pipeline_SVM/inputs/clean_wavs_list.csv")
OUT = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios.matched.csv")

df = pd.read_csv(FEATURES, low_memory=False)
clean = pd.read_csv(CLEAN_LIST, low_memory=False)

# normalize basenames
df['basename'] = df['filepath'].astype(str).apply(lambda p: Path(p).name.lower())
clean['basename'] = clean['basename'].astype(str).str.lower()

merged = df.merge(clean[['basename','wav_path','parent']].drop_duplicates('basename'), on='basename', how='inner', suffixes=('','_clean'))
print("Matched rows:", len(merged))
merged.to_csv(OUT, index=False)
print("Guardado:", OUT)
