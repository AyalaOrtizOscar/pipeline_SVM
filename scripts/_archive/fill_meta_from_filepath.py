# fill_meta_from_filepath.py
import pandas as pd
import re
from pathlib import Path

IN = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.mapped_by_folder.csv"
OUT = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_filled.csv"

df = pd.read_csv(IN, low_memory=False)

def infer_experiment(fp):
    if not isinstance(fp, str): return None
    s = fp.lower()
    m = re.search(r'(?:/|\\|^)(e\d(?:[_\-\s]|$)[^/\\]*)', s)
    if m:
        return m.group(1).replace("\\","/").strip()
    m2 = re.search(r'/([be]\d{1,4}[_\-]?.*?)(?:[_\-]aug|$)', s)
    if m2:
        return m2.group(1)
    return None

def infer_mic(fp):
    if not isinstance(fp, str): return None
    s = fp.lower()
    if 'micrófono' in s or 'micro' in s or 'mic_' in s or 'mic' in s:
        # try to capture trailing token like "micrófono1" or "micrófono2 (condensador)"
        m = re.search(r'(micr[oó]fono[^/\\]*)', s)
        if m:
            return m.group(1).strip()
        m2 = re.search(r'(mic[^/\\]*)', s)
        if m2:
            return m2.group(1).strip()
    # fallback: attempt to find "(condensador)" or "(dinámico)"
    if 'condensador' in s:
        return 'condensador'
    if 'dinámico' in s or 'dinamico' in s:
        return 'dinámico'
    return None

# ensure cols exist
if 'experiment' not in df.columns: df['experiment'] = pd.NA
if 'mic_type' not in df.columns: df['mic_type'] = pd.NA

filled_exp = 0
filled_mic = 0
for idx, row in df.iterrows():
    fp = row.get('filepath','')
    if pd.isna(row['experiment']) or row['experiment']=='':
        val = infer_experiment(fp)
        if val:
            df.at[idx,'experiment'] = val
            filled_exp += 1
    if pd.isna(row['mic_type']) or row['mic_type']=='':
        val2 = infer_mic(fp)
        if val2:
            df.at[idx,'mic_type'] = val2
            filled_mic += 1

print(f"Filled experiment for {filled_exp} rows, mic_type for {filled_mic} rows.")
df.to_csv(OUT, index=False)
print("Saved:", OUT)
