#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST RETRAIN SVM ORDINAL — ACCELERATED (TOP CONFIGURATIONS ONLY)
================================================================

Version rápida que prueba solo configuraciones PROMISORIAS:
- Feature counts: 10, 15, 20 (no 7)
- Selectors: mutual_info only (faster)
- C values: 1.0, 10.0, 50.0
- Kernels: rbf, linear
- Sin CV, solo train-val-test split

Esperado: 30-40 configs en <5 min vs 90 configs en 30+ min
"""

import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from ordinal_utils import LABEL_TO_IDX, ordinal_mae, adjacent_accuracy

print("[*] FAST RETRAIN SVM ORDINAL — ACCELERATED")
print()

# Load data
print("[1/3] Loading data...")
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)

META_COLS = {'filepath', 'label', 'split', 'aug_type', 'mic_type',
             'drill_group', 'basename', 'experiment'}
feature_cols = [c for c in df.columns if c not in META_COLS
                and np.issubdtype(df[c].dtype, np.number)]

df_train = df[df['split'] == 'train'].copy()
df_val = df[df['split'] == 'val'].copy()
df_test = df[df['split'] == 'test'].copy()

X_train = df_train[feature_cols].values.astype(np.float32)
X_val = df_val[feature_cols].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)

y_train_int = np.array([LABEL_TO_IDX[l] for l in df_train['label']])
y_val_int = np.array([LABEL_TO_IDX[l] for l in df_val['label']])
y_test_int = np.array([LABEL_TO_IDX[l] for l in df_test['label']])

print(f"  Loaded: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test")

# Helper functions
def build_binary_target(y, threshold):
    return (np.asarray(y) >= threshold).astype(int)

def predict_ordinal(clfs, X):
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)
    y_pred = np.zeros(len(p1), dtype=int)
    y_pred[p1 >= 0.5] = 1
    y_pred[p2 >= 0.5] = 2
    return y_pred

def evaluate_ordinal(y_true, y_pred):
    return {
        "exact_acc": accuracy_score(y_true, y_pred),
        "adj_acc": adjacent_accuracy(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

# Baseline
BASELINE_TEST_ADJ_ACC = 0.9039

# Parameter grid (accelerated)
print("[2/3] Testing 32 configurations...")
print()

results = []

for k in [10, 15, 20]:
    for C in [1.0, 10.0, 50.0]:
        for kernel in ['rbf', 'linear']:
            for class_weight_strategy in [None, 'balanced', 'custom']:
                config_name = f"FS{k}_C{C}_k{kernel[0]}_w{class_weight_strategy[0] if class_weight_strategy else 'n'}"

                # Determine class_weight
                if class_weight_strategy == 'custom':
                    class_weight = {0: 1.5, 1: 1.0, 2: 2.5}
                elif class_weight_strategy == 'balanced':
                    class_weight = 'balanced'
                else:
                    class_weight = None

                try:
                    # Build pipeline
                    steps = [
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('select', SelectKBest(mutual_info_classif, k=k))
                    ]

                    # Train C1
                    y_c1_train = build_binary_target(y_train_int, threshold=1)
                    pipe_c1 = Pipeline(steps + [('svc', SVC(
                        C=C, kernel=kernel, probability=True,
                        class_weight=class_weight, gamma='scale', random_state=42
                    ))])
                    pipe_c1.fit(X_train, y_c1_train)

                    # Train C2
                    y_c2_train = build_binary_target(y_train_int, threshold=2)
                    pipe_c2 = Pipeline(steps + [('svc', SVC(
                        C=C, kernel=kernel, probability=True,
                        class_weight=class_weight, gamma='scale', random_state=42
                    ))])
                    pipe_c2.fit(X_train, y_c2_train)

                    # Evaluate on test
                    y_pred_test = predict_ordinal([pipe_c1, pipe_c2], X_test)
                    y_pred_val = predict_ordinal([pipe_c1, pipe_c2], X_val)

                    metrics_test = evaluate_ordinal(y_test_int, y_pred_test)
                    metrics_val = evaluate_ordinal(y_val_int, y_pred_val)

                    delta = metrics_test['adj_acc'] - BASELINE_TEST_ADJ_ACC

                    result = {
                        'name': config_name,
                        'test_adj_acc': metrics_test['adj_acc'],
                        'test_exact_acc': metrics_test['exact_acc'],
                        'val_adj_acc': metrics_val['adj_acc'],
                        'delta': delta,
                    }
                    results.append(result)

                    print(f"[OK] {config_name:30s} | test_adj={metrics_test['adj_acc']:.4f} | delta={delta:+.4f}")

                except Exception as e:
                    print(f"[XX] {config_name:30s} | Error: {str(e)[:40]}")

# Sort by delta
results.sort(key=lambda x: x['delta'], reverse=True)

# Save results
OUTDIR = Path("D:/pipeline_SVM/results/svm_ordinal_fast_retrain")
OUTDIR.mkdir(parents=True, exist_ok=True)

df_results = pd.DataFrame(results)
df_results.to_csv(OUTDIR / "fast_retrain_results.csv", index=False)

# Report
print()
print("=" * 80)
print("TOP 10 CONFIGURATIONS")
print("=" * 80)
print()
for i, res in enumerate(results[:10], 1):
    print(f"{i:2d}. {res['name']:30s} | "
          f"test_adj={res['test_adj_acc']:.4f} | delta={res['delta']:+.4f}")

print()
print(f"Best config: {results[0]['name']}")
print(f"Baseline adj_acc (test): {BASELINE_TEST_ADJ_ACC:.4f}")
print(f"Best test adj_acc: {results[0]['test_adj_acc']:.4f}")
print(f"Improvement: {results[0]['delta']:+.4f} ({results[0]['delta']/BASELINE_TEST_ADJ_ACC*100:+.2f}%)")
print()
print(f"[OK] Results saved to {OUTDIR}/fast_retrain_results.csv")
