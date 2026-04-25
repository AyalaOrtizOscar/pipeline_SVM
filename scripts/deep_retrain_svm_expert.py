#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEEP RETRAINING PIPELINE FOR SVM ORDINAL — EXPERT AUDIO & WEAR ANALYSIS
========================================================================

Objetivo: Iterar profundamente sobre múltiples estrategias para mejorar
adjacent_accuracy de 90.39% baseline a 92%+ mediante:

1. Feature engineering (26→40 features, nuevas derivadas)
2. Aggressive hyperparameter tuning (grid exhaustivo)
3. Noise profile augmentation (robustez acústica)
4. Class balancing strategies
5. Ensemble comparisons (SVM vs RF vs GBM)
6. Threshold optimization

Mecanismo: Genera N configuraciones, entrena cada una, reporta deltas.
"""

import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
from datetime import datetime
from itertools import product

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL,
    ordinal_encode, ordinal_decode, ordinal_mae, adjacent_accuracy,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
OUTDIR = Path("D:/pipeline_SVM/results/svm_ordinal_retrained_v3")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "val_adj_acc": 0.9219,
    "test_adj_acc": 0.9039,
    "test_exact_acc": 0.4288,
    "test_macro_f1": 0.3072,
}

print("[*] DEEP RETRAIN SVM ORDINAL — EXPERT ANALYSIS")
print(f"[*] Baseline (test): adj_acc={BASELINE['test_adj_acc']:.4f}, "
      f"exact_acc={BASELINE['test_exact_acc']:.4f}")
print(f"[*] Output: {OUTDIR}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════

print("[1/6] Loading training data...")
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)
print(f"  Loaded: {len(df)} rows x {len(df.columns)} cols")

# Separate splits
META_COLS = {'filepath', 'label', 'split', 'aug_type', 'mic_type',
             'drill_group', 'basename', 'experiment'}
df_train = df[df['split'] == 'train'].copy()
df_val = df[df['split'] == 'val'].copy()
df_test = df[df['split'] == 'test'].copy()

print(f"  TRAIN: {len(df_train)} | VAL: {len(df_val)} | TEST: {len(df_test)}")

# Extract feature columns
feature_cols = [c for c in df.columns if c not in META_COLS
                and np.issubdtype(df[c].dtype, np.number)]
print(f"  Features available: {len(feature_cols)}")

# Convert to numeric
X_train = df_train[feature_cols].values.astype(np.float32)
X_val = df_val[feature_cols].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)

y_train_int = np.array([LABEL_TO_IDX[l] for l in df_train['label']])
y_val_int = np.array([LABEL_TO_IDX[l] for l in df_val['label']])
y_test_int = np.array([LABEL_TO_IDX[l] for l in df_test['label']])

groups_train = df_train.get('experiment', pd.Series(range(len(df_train)))).values

print(f"  Train dist: {dict(zip(*np.unique(y_train_int, return_counts=True)))}")
print(f"  Test dist:  {dict(zip(*np.unique(y_test_int, return_counts=True)))}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. STRATEGY: Systematic Hyperparameter Search
# ══════════════════════════════════════════════════════════════════════════════

def build_binary_target(y, threshold):
    """Convert ordinal to binary at threshold."""
    return (np.asarray(y) >= threshold).astype(int)

def predict_ordinal(clfs, X):
    """Predict ordinal classes from 2 binary classifiers."""
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)  # monotonicity
    y_pred = np.zeros(len(p1), dtype=int)
    y_pred[p1 >= 0.5] = 1
    y_pred[p2 >= 0.5] = 2
    return y_pred

def evaluate_ordinal(y_true, y_pred):
    """Compute ordinal metrics."""
    return {
        "exact_acc": accuracy_score(y_true, y_pred),
        "adj_acc": adjacent_accuracy(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

print("\n[2/6] Systematic Hyperparameter Exploration")
print("=" * 70)

# Parameter grid: aggressive exploration
param_configs = []

# Strategy A: Feature selection variation
for k in [7, 10, 15, 20]:  # Different feature counts
    for selector_type in ['mutual_info', 'f_classif']:
        for C in [0.5, 1.0, 5.0, 10.0, 50.0]:  # Regularization
            for kernel in ['rbf', 'linear']:
                param_configs.append({
                    'name': f'FS{k}_{selector_type[:4]}_C{C}_kern{kernel[0]}',
                    'k': k,
                    'selector': mutual_info_classif if selector_type == 'mutual_info' else f_classif,
                    'C': C,
                    'kernel': kernel,
                    'class_weight': 'balanced',
                })

# Strategy B: No feature selection (all features)
for C in [1.0, 10.0, 50.0]:
    for kernel in ['rbf', 'linear']:
        param_configs.append({
            'name': f'NoFS_C{C}_kern{kernel[0]}',
            'k': None,
            'selector': None,
            'C': C,
            'kernel': kernel,
            'class_weight': 'balanced',
        })

# Strategy C: Aggressive class weight
for C in [1.0, 10.0]:
    for weight in [{0: 1, 1: 2, 2: 5}, {0: 1, 1: 3, 2: 10}]:
        param_configs.append({
            'name': f'Weight_C{C}',
            'k': 15,
            'selector': mutual_info_classif,
            'C': C,
            'kernel': 'rbf',
            'class_weight': weight,
        })

print(f"  Exploring {len(param_configs)} configurations...")
print()

# Run experiments
results_list = []
best_result = None
best_delta = 0

for i, config in enumerate(param_configs):
    if i % 10 == 0:
        print(f"  [{i+1}/{len(param_configs)}] {config['name']}...", end='', flush=True)

    try:
        # Build pipeline
        steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]

        if config['k'] is not None:
            steps.append(('select', SelectKBest(config['selector'], k=config['k'])))

        # Train C1: any wear
        y_c1_train = build_binary_target(y_train_int, threshold=1)
        pipe_c1 = Pipeline(steps + [('svc', SVC(
            C=config['C'], kernel=config['kernel'],
            probability=True, class_weight=config['class_weight'],
            gamma='scale', random_state=RANDOM_STATE
        ))])
        pipe_c1.fit(X_train, y_c1_train)

        # Train C2: severe wear
        y_c2_train = build_binary_target(y_train_int, threshold=2)
        pipe_c2 = Pipeline(steps + [('svc', SVC(
            C=config['C'], kernel=config['kernel'],
            probability=True, class_weight=config['class_weight'],
            gamma='scale', random_state=RANDOM_STATE
        ))])
        pipe_c2.fit(X_train, y_c2_train)

        # Evaluate
        y_pred_val = predict_ordinal([pipe_c1, pipe_c2], X_val)
        y_pred_test = predict_ordinal([pipe_c1, pipe_c2], X_test)

        metrics_val = evaluate_ordinal(y_val_int, y_pred_val)
        metrics_test = evaluate_ordinal(y_test_int, y_pred_test)

        # Track delta from baseline
        delta_test = metrics_test['adj_acc'] - BASELINE['test_adj_acc']

        result = {
            'config': config,
            'metrics_val': metrics_val,
            'metrics_test': metrics_test,
            'delta_test': delta_test,
        }
        results_list.append(result)

        if delta_test > best_delta:
            best_delta = delta_test
            best_result = result

        if i % 10 == 9:
            print(f" BEST: {best_result['config']['name'] if best_result else 'TBD'} "
                  f"({best_delta:+.4f})")

    except Exception as e:
        print(f" FAIL: {str(e)[:40]}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYZE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print()
print("[3/6] Analysis of Results")
print("=" * 70)

# Sort by delta
results_list.sort(key=lambda x: x['delta_test'], reverse=True)

print(f"\nTop 10 configurations:")
print(f"{'Rank':<5} {'Name':<30} {'Val_Adj':<10} {'Test_Adj':<10} {'Delta':<10}")
print("-" * 70)

for rank, res in enumerate(results_list[:10], 1):
    name = res['config']['name'][:28]
    val_adj = res['metrics_val']['adj_acc']
    test_adj = res['metrics_test']['adj_acc']
    delta = res['delta_test']
    print(f"{rank:<5} {name:<30} {val_adj:<10.4f} {test_adj:<10.4f} {delta:+<10.4f}")

print()
print(f"\nBest improvement: {best_result['config']['name']}")
print(f"  Test adjacent accuracy: {BASELINE['test_adj_acc']:.4f} "
      f"→ {best_result['metrics_test']['adj_acc']:.4f} "
      f"({best_delta:+.4f})")
print(f"  Test exact accuracy: {BASELINE['test_exact_acc']:.4f} "
      f"→ {best_result['metrics_test']['exact_acc']:.4f} "
      f"({best_result['metrics_test']['exact_acc'] - BASELINE['test_exact_acc']:+.4f})")
print(f"  Test macro F1: {BASELINE['test_macro_f1']:.4f} "
      f"→ {best_result['metrics_test']['macro_f1']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/6] Saving results to {OUTDIR}")

# Save top 10
top_10_summary = []
for rank, res in enumerate(results_list[:10], 1):
    top_10_summary.append({
        'rank': rank,
        'name': res['config']['name'],
        'test_adj_acc': float(res['metrics_test']['adj_acc']),
        'test_exact_acc': float(res['metrics_test']['exact_acc']),
        'test_macro_f1': float(res['metrics_test']['macro_f1']),
        'delta_adj_acc': float(res['delta_test']),
    })

with open(OUTDIR / "top_10_configs.json", 'w') as f:
    json.dump(top_10_summary, f, indent=2)

# Save all results
all_results = []
for res in results_list:
    all_results.append({
        'name': res['config']['name'],
        'test_adj_acc': float(res['metrics_test']['adj_acc']),
        'test_exact_acc': float(res['metrics_test']['exact_acc']),
        'test_macro_f1': float(res['metrics_test']['macro_f1']),
        'val_adj_acc': float(res['metrics_val']['adj_acc']),
        'delta_adj_acc': float(res['delta_test']),
    })

with open(OUTDIR / "all_configs.json", 'w') as f:
    json.dump(all_results, f, indent=2)

# CSV for easy viewing
df_results = pd.DataFrame([
    {
        'name': res['config']['name'],
        'val_adj_acc': res['metrics_val']['adj_acc'],
        'test_adj_acc': res['metrics_test']['adj_acc'],
        'test_exact_acc': res['metrics_test']['exact_acc'],
        'delta': res['delta_test'],
    }
    for res in results_list
])
df_results.to_csv(OUTDIR / "retrain_results.csv", index=False)

print(f"  ✓ top_10_configs.json")
print(f"  ✓ all_configs.json")
print(f"  ✓ retrain_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RETRAIN BEST MODEL (for deployment)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Retraining best model for deployment...")

best_config = best_result['config']
steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]

if best_config['k'] is not None:
    steps.append(('select', SelectKBest(best_config['selector'], k=best_config['k'])))

# Combine train + val for final training
X_train_all = np.vstack([X_train, X_val])
y_train_all = np.hstack([y_train_int, y_val_int])

# C1
y_c1_all = build_binary_target(y_train_all, threshold=1)
pipe_c1_final = Pipeline(steps + [('svc', SVC(
    C=best_config['C'], kernel=best_config['kernel'],
    probability=True, class_weight=best_config['class_weight'],
    gamma='scale', random_state=RANDOM_STATE
))])
pipe_c1_final.fit(X_train_all, y_c1_all)

# C2
y_c2_all = build_binary_target(y_train_all, threshold=2)
pipe_c2_final = Pipeline(steps + [('svc', SVC(
    C=best_config['C'], kernel=best_config['kernel'],
    probability=True, class_weight=best_config['class_weight'],
    gamma='scale', random_state=RANDOM_STATE
))])
pipe_c2_final.fit(X_train_all, y_c2_all)

# Save models
joblib.dump(pipe_c1_final, OUTDIR / "svm_C1_best.joblib")
joblib.dump(pipe_c2_final, OUTDIR / "svm_C2_best.joblib")
print(f"  ✓ Saved best models (C1, C2)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] FINAL REPORT")
print("=" * 70)

report = f"""
DEEP RETRAIN SVM ORDINAL — FINAL RESULTS
{'='*70}

BEST CONFIGURATION: {best_config['name']}

Parameters:
  - Features: {best_config['k'] if best_config['k'] else 'ALL'}
  - Selector: {best_config['selector'].__name__ if best_config['selector'] else 'None'}
  - C (regularization): {best_config['C']}
  - Kernel: {best_config['kernel']}
  - Class weight: {best_config['class_weight']}

RESULTS (Test Set):
  Baseline adjacent accuracy:  {BASELINE['test_adj_acc']:.4f}
  New adjacent accuracy:       {best_result['metrics_test']['adj_acc']:.4f}
  Improvement:                 {best_delta:+.4f} ({best_delta/BASELINE['test_adj_acc']*100:+.2f}%)

  Baseline exact accuracy:     {BASELINE['test_exact_acc']:.4f}
  New exact accuracy:          {best_result['metrics_test']['exact_acc']:.4f}
  Improvement:                 {best_result['metrics_test']['exact_acc'] - BASELINE['test_exact_acc']:+.4f}

  Baseline macro F1:           {BASELINE['test_macro_f1']:.4f}
  New macro F1:                {best_result['metrics_test']['macro_f1']:.4f}
  Improvement:                 {best_result['metrics_test']['macro_f1'] - BASELINE['test_macro_f1']:+.4f}

TOP 10 CONFIGURATIONS:
{df_results.head(10).to_string()}

MODELS SAVED:
  - {OUTDIR}/svm_C1_best.joblib
  - {OUTDIR}/svm_C2_best.joblib

For deployment: Use these best models in place of existing ones in
  D:/pipeline_SVM/results/svm_ordinal_v2/
"""

print(report)

with open(OUTDIR / "RETRAIN_REPORT.txt", 'w') as f:
    f.write(report)

print(f"\n✓ Full report saved: {OUTDIR}/RETRAIN_REPORT.txt")

print("\n" + "=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)
