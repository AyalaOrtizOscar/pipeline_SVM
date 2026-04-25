#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import itertools
import math

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast"); OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(FEATURES_CSV, low_memory=False)
label_col = "label_fixed"
mask = df[label_col].notna()
df = df.loc[mask]

features = df.select_dtypes(include=[np.number]).columns.tolist()
results = []
for feat in features:
    series = df[[feat, label_col]].dropna()
    groups = [group[feat].values for name, group in series.groupby(label_col)]
    # if there are at least 2 groups with >1 sample
    if len(groups) >= 2 and all(len(g)>1 for g in groups):
        try:
            stat, p_anova = stats.f_oneway(*groups)
        except Exception:
            p_anova = np.nan
        # kruskal
        try:
            statk, p_kruskal = stats.kruskal(*groups)
        except Exception:
            p_kruskal = np.nan
        # eta squared for ANOVA (approx)
        try:
            # compute eta2 as SS_between / SS_total
            grand_mean = series[feat].mean()
            ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for name,g in series.groupby(label_col)[feat])
            ss_total = sum((series[feat]-grand_mean)**2)
            eta2 = ss_between/ss_total if ss_total>0 else np.nan
        except Exception:
            eta2 = np.nan
        results.append({"feature":feat, "p_anova":p_anova, "p_kruskal":p_kruskal, "eta2":eta2})
pd.DataFrame(results).to_csv(OUT_DIR/"anova_kruskal_eta2.csv", index=False)
print("Saved ANOVA/Kruskal results in", OUT_DIR/"anova_kruskal_eta2.csv")
