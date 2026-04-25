# merge_meta_filled.py
import pandas as pd
from pathlib import Path
H = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.labelled_full.mapped_by_folder.csv")
META = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_filled.csv")
OUT = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.csv")

df_h = pd.read_csv(H, low_memory=False)
df_m = pd.read_csv(META, low_memory=False)

# ensure fp_norm is normalized (lower)
def norm_fp(s):
    return str(s).replace("\\","/").strip().lower()
df_h['fp_norm_l'] = df_h.get('fp_norm', df_h.get('filepath','')).astype(str).map(norm_fp)
df_m['fp_norm_l'] = df_m.get('fp_norm', df_m.get('filepath','')).astype(str).map(norm_fp)

# index meta by fp_norm_l
meta_map = df_m.set_index('fp_norm_l')

updated_exp = 0
updated_mic = 0
for idx, row in df_h.iterrows():
    key = row['fp_norm_l']
    if key in meta_map.index:
        meta_row = meta_map.loc[key]
        # if multiple matches, take first
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]
        # experiment
        if (pd.isna(row.get('experiment')) or str(row.get('experiment')).strip()=='') and str(meta_row.get('experiment', '')).strip()!='':
            df_h.at[idx, 'experiment'] = meta_row.get('experiment')
            updated_exp += 1
        # mic_type
        if (pd.isna(row.get('mic_type')) or str(row.get('mic_type')).strip()=='') and str(meta_row.get('mic_type','')).strip()!='':
            df_h.at[idx, 'mic_type'] = meta_row.get('mic_type')
            updated_mic += 1

print(f"Updated experiment for {updated_exp} rows, mic_type for {updated_mic} rows.")
df_h.to_csv(OUT, index=False)
print("Saved merged output:", OUT)
