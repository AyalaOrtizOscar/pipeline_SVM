#!/usr/bin/env python3
import numpy as np, pandas as pd

meta = pd.read_csv('/storage/mel_dataset_meta.csv')
df   = pd.read_csv('/storage/features_multimodal_merged.csv')

ESP32_COLS = ['esp_rms','esp_rms_db','esp_centroid_mean','esp_zcr','esp_crest_factor','esp_spectral_contrast_mean']
FLOW_COLS  = ['flow_mean_lmin','flow_std_lmin','flow_cv','flow_duty_pulses']

print('=== Label distribution per test ===')
for exp in ['art2_test39','art2_test50','art2_test53']:
    sub = meta[meta['experiment']==exp]
    print(f'  {exp}: n={len(sub)} | {dict(sub["label_str"].value_counts())}')

print('\n=== Columns in df_full ===')
print('  experiment col:', 'experiment' in df.columns)
print('  Cols:', list(df.columns))

print('\n=== ESP32/Flow availability ===')
for exp in ['art2_test39','art2_test50','art2_test53']:
    if 'experiment' in df.columns:
        sub = df[df['experiment']==exp]
    else:
        tid = exp.replace('art2_','')
        sub = df[df['test_id'].astype(str)==tid] if 'test_id' in df.columns else pd.DataFrame()

    if len(sub) > 0:
        e_nn = sub[ESP32_COLS[0]].notna().sum()  if ESP32_COLS[0]  in sub.columns else 0
        f_nn = sub[FLOW_COLS[0]].notna().sum()   if FLOW_COLS[0]   in sub.columns else 0
        print(f'  {exp}: {len(sub)} rows | esp non-null={e_nn} | flow non-null={f_nn}')
        if e_nn > 0 and ESP32_COLS[0] in sub.columns:
            print(f'    esp_rms:     mean={sub[ESP32_COLS[0]].mean():.4f}')
        if f_nn > 0 and FLOW_COLS[0] in sub.columns:
            print(f'    flow_mean:   mean={sub[FLOW_COLS[0]].mean():.4f}  std={sub[FLOW_COLS[0]].std():.4f}')
    else:
        print(f'  {exp}: NOT FOUND in df_full')

print('\n=== Alignment check: meta vs df_full ===')
print(f'  meta rows: {len(meta)}  |  df_full rows: {len(df)}')
if 'experiment' in df.columns:
    print('  df_full experiments:', df['experiment'].value_counts().to_dict())

# Key check: are ESP32 features correlated with wear label?
print('\n=== ESP32/Flow correlation with label ===')
if 'experiment' in df.columns:
    multi_df = df[df['experiment'].isin(['art2_test39','art2_test50','art2_test53'])].copy()
else:
    multi_df = df[df['test_id'].astype(str).str.contains('test39|test50|test53')].copy() if 'test_id' in df.columns else df

if 'label' in multi_df.columns or 'label_str' in multi_df.columns:
    lbl_col = 'label' if 'label' in multi_df.columns else 'label_str'
    for col in FLOW_COLS + ESP32_COLS:
        if col in multi_df.columns:
            by_label = multi_df.groupby(lbl_col)[col].mean()
            print(f'  {col}: {by_label.to_dict()}')
