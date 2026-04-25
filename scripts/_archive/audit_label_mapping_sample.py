# audit_label_mapping_sample.py
import pandas as pd
from pathlib import Path
OUT = Path("D:/pipeline_SVM/results/fix_harmonized_v5")
df = pd.read_csv(OUT/"features_svm_harmonized_fixed_v5.csv", low_memory=False)
samples = {}
for m in ['basename_exact','augmented_mapping_csv','digits_in_name','prefix_match','tail3','fp_exact','']:
    sub = df[df['map_method']==m]
    samples[m] = sub[['fp_norm','basename','basename_core','label_mapped','map_method']].sample(n=min(10, len(sub)), random_state=0)
all_samples = pd.concat(samples.values(), keys=samples.keys(), names=['method','i'])
all_samples.to_csv(OUT/"mapping_audit_samples.csv")
print("Audit saved:", OUT/"mapping_audit_samples.csv")
