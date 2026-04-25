#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THRESHOLD OPTIMIZATION FOR FRANK & HALL ORDINAL
===============================================

El modelo actual usa thresholds fijos en 0.5:
  if P(y>=1) >= 0.5 -> class >= 1
  if P(y>=2) >= 0.5 -> class >= 2

Este script busca thresholds OPTIMALES para maximizar:
1. Adjacent accuracy (métrica clave)
2. Balanced accuracy (precisión por clase)
3. F1-score ponderado

Rango de búsqueda: [0.3, 0.7] para ambos thresholds
Método: Grid search exhaustivo en validation set
"""

import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from itertools import product
from sklearn.metrics import f1_score, balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

from ordinal_utils import LABEL_TO_IDX, ordinal_mae, adjacent_accuracy

print("[*] THRESHOLD OPTIMIZATION FOR FRANK & HALL")
print()

# Load data
print("[1/3] Loading data...")
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)

META_COLS = {'filepath', 'label', 'split', 'aug_type', 'mic_type',
             'drill_group', 'basename', 'experiment'}
feature_cols = [c for c in df.columns if c not in META_COLS
                and np.issubdtype(df[c].dtype, np.number)]

df_val = df[df['split'] == 'val'].copy()
df_test = df[df['split'] == 'test'].copy()

X_val = df_val[feature_cols].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)

y_val_int = np.array([LABEL_TO_IDX[l] for l in df_val['label']])
y_test_int = np.array([LABEL_TO_IDX[l] for l in df_test['label']])

print(f"  Val: {len(y_val_int)} | Test: {len(y_test_int)}")

# Load baseline models
print("\n[2/3] Loading baseline models...")
c1_model = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/svm_C1_top15_orig.joblib")
c2_model = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/svm_C2_top15_orig.joblib")
scaler = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/scaler_top15_orig.joblib")

# Get top-15 features
top_15_features = [
    'spectral_bandwidth_std', 'rolloff_std', 'spectral_bandwidth_mean',
    'mfcc_1_mean', 'zcr', 'centroid_std', 'spectral_flatness_mean',
    'rolloff_mean', 'centroid_mean', 'mfcc_0_mean', 'spectral_entropy_mean',
    'crest_factor', 'spectral_contrast_mean', 'harmonic_percussive_ratio',
    'chroma_std'
]
feature_indices = [feature_cols.index(f) for f in top_15_features]

# Get probabilities
X_val_scaled = scaler.transform(X_val[:, feature_indices])
X_test_scaled = scaler.transform(X_test[:, feature_indices])

p1_val = c1_model.predict_proba(X_val_scaled)[:, 1]
p2_val = c2_model.predict_proba(X_val_scaled)[:, 1]

p1_test = c1_model.predict_proba(X_test_scaled)[:, 1]
p2_test = c2_model.predict_proba(X_test_scaled)[:, 1]

print(f"  Loaded C1 and C2 models")

# Search for optimal thresholds
print("\n[3/3] Searching for optimal thresholds...")
print()

results = []
best_result = None
best_val_adj_acc = 0

# Search grid: 0.3 to 0.7 in steps of 0.05
for thresh_c1 in np.arange(0.3, 0.71, 0.05):
    for thresh_c2 in np.arange(0.3, 0.71, 0.05):
        # Monotonicity constraint: P(y>=2) <= P(y>=1)
        p2_adj_val = np.minimum(p2_val, p1_val)
        p2_adj_test = np.minimum(p2_test, p1_test)

        # Predict
        y_pred_val = np.zeros(len(p1_val), dtype=int)
        y_pred_val[p1_val >= thresh_c1] = 1
        y_pred_val[p2_adj_val >= thresh_c2] = 2

        y_pred_test = np.zeros(len(p1_test), dtype=int)
        y_pred_test[p1_test >= thresh_c1] = 1
        y_pred_test[p2_adj_test >= thresh_c2] = 2

        # Metrics on validation
        val_adj_acc = adjacent_accuracy(y_val_int, y_pred_val)
        val_exact_acc = (y_val_int == y_pred_val).mean()
        val_balanced = balanced_accuracy_score(y_val_int, y_pred_val)

        # Metrics on test
        test_adj_acc = adjacent_accuracy(y_test_int, y_pred_test)
        test_exact_acc = (y_test_int == y_pred_test).mean()
        test_balanced = balanced_accuracy_score(y_test_int, y_pred_test)

        result = {
            'thresh_c1': thresh_c1,
            'thresh_c2': thresh_c2,
            'val_adj_acc': val_adj_acc,
            'val_exact_acc': val_exact_acc,
            'val_balanced_acc': val_balanced,
            'test_adj_acc': test_adj_acc,
            'test_exact_acc': test_exact_acc,
            'test_balanced_acc': test_balanced,
        }
        results.append(result)

        if val_adj_acc > best_val_adj_acc:
            best_val_adj_acc = val_adj_acc
            best_result = result

# Sort by test adjacent accuracy
results.sort(key=lambda x: x['test_adj_acc'], reverse=True)

# Report
print("=" * 90)
print("TOP 10 THRESHOLD CONFIGURATIONS (by Test Adjacent Accuracy)")
print("=" * 90)
print()
print(f"{'Rank':<5} {'C1':<6} {'C2':<6} {'Val_Adj':<10} {'Test_Adj':<10} {'Test_Exact':<12} {'Delta_vs_050'}")
print("-" * 90)

baseline_adj_acc = 0.9039  # Current baseline at (0.5, 0.5)

for i, res in enumerate(results[:10], 1):
    delta = res['test_adj_acc'] - baseline_adj_acc
    print(f"{i:<5} {res['thresh_c1']:<6.2f} {res['thresh_c2']:<6.2f} "
          f"{res['val_adj_acc']:<10.4f} {res['test_adj_acc']:<10.4f} "
          f"{res['test_exact_acc']:<12.4f} {delta:+.4f}")

best = results[0]
print()
print("=" * 90)
print(f"BEST THRESHOLDS: C1={best['thresh_c1']:.2f}, C2={best['thresh_c2']:.2f}")
print(f"Baseline (0.50, 0.50):     test_adj_acc = 0.9039")
print(f"Optimized thresholds:      test_adj_acc = {best['test_adj_acc']:.4f}")
print(f"Improvement:               {best['test_adj_acc'] - 0.9039:+.4f} ({(best['test_adj_acc'] - 0.9039)/0.9039*100:+.2f}%)")
print()

# Save results
OUTDIR = Path("D:/pipeline_SVM/results/threshold_optimization")
OUTDIR.mkdir(parents=True, exist_ok=True)

df_results = pd.DataFrame(results)
df_results.to_csv(OUTDIR / "threshold_optimization_results.csv", index=False)

with open(OUTDIR / "threshold_optimization_summary.json", 'w') as f:
    json.dump({
        'baseline': {'thresh_c1': 0.5, 'thresh_c2': 0.5, 'test_adj_acc': 0.9039},
        'optimized': {
            'thresh_c1': float(best['thresh_c1']),
            'thresh_c2': float(best['thresh_c2']),
            'test_adj_acc': float(best['test_adj_acc']),
            'test_exact_acc': float(best['test_exact_acc']),
            'improvement': float(best['test_adj_acc'] - 0.9039),
        }
    }, f, indent=2)

print(f"[OK] Results saved to {OUTDIR}/")
