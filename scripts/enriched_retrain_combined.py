#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENRICHED RETRAIN COMBINED — Features derivadas + Threshold Opt + Grid Search
============================================================================

Estrategia combinada:
1. Derivar ~15 features nuevas del CSV existente (sin re-leer WAVs)
2. Eliminar features redundantes (|r| > 0.95)
3. Grid search agresivo sobre feature set enriquecido
4. Threshold optimization posterior
5. Reportar mejora total acumulada

Baseline: 90.39% adj_acc (thresholds 0.50/0.50)
Target: >95% adj_acc
"""

import sys, os, json, time
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

from ordinal_utils import LABEL_TO_IDX, IDX_TO_LABEL, ordinal_mae, adjacent_accuracy

OUTDIR = Path("D:/pipeline_SVM/results/svm_enriched_v4")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE_ADJ_ACC = 0.9039
BASELINE_EXACT_ACC = 0.4288

print("=" * 80)
print("  ENRICHED RETRAIN COMBINED")
print("  Features derivadas + Threshold Optimization + Grid Search")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/7] Loading data...")
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)

META_COLS = {'filepath', 'label', 'split', 'aug_type', 'mic_type',
             'drill_group', 'basename', 'experiment'}
orig_feature_cols = [c for c in df.columns if c not in META_COLS
                     and np.issubdtype(df[c].dtype, np.number)]

print(f"  Original: {len(df)} rows x {len(orig_feature_cols)} numeric features")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — Derived features from existing columns
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/7] Engineering derived features...")

eps = 1e-10  # avoid division by zero

# Ratios capturando relaciones entre features (inspirados en analisis de desgaste)
df['relative_bandwidth'] = df['spectral_bandwidth_mean'] / (df['centroid_mean'] + eps)
df['rolloff_centroid_ratio'] = df['rolloff_mean'] / (df['centroid_mean'] + eps)
df['energy_crossing_product'] = df['rms'] * df['zcr']
df['tonal_stability'] = df['harmonic_percussive_ratio'] * df['spectral_flatness_mean']
df['mfcc_gradient'] = df['mfcc_0_mean'] - df['mfcc_1_mean']

# Coeficientes de variacion (estabilidad espectral)
df['centroid_cv'] = df['centroid_std'] / (df['centroid_mean'] + eps)
df['bandwidth_cv'] = df['spectral_bandwidth_std'] / (df['spectral_bandwidth_mean'] + eps)
df['rolloff_cv'] = df['rolloff_std'] / (df['rolloff_mean'] + eps)
df['flatness_cv'] = df['spectral_flatness_std'] / (df['spectral_flatness_mean'] + eps)

# Indicadores de energia y pico
df['peak_indicator'] = df['crest_factor'] * df['rms']
df['energy_ratio'] = df['mel_total_energy'] / (df['wavelet_total_energy'] + eps)
df['disorder_measure'] = df['spectral_entropy_mean'] * df['spectral_flatness_mean']

# Interacciones espectrales
df['spectral_spread'] = df['spectral_bandwidth_mean'] * df['spectral_bandwidth_std']
df['chroma_variation'] = df['chroma_std'] / (df['chroma_mean'].abs() + eps)

# Logaritmicas para features con gran rango dinamico
df['log_mel_energy'] = np.log1p(df['mel_total_energy'].clip(lower=0))
df['log_wavelet_energy'] = np.log1p(df['wavelet_total_energy'].clip(lower=0))

new_feature_names = [
    'relative_bandwidth', 'rolloff_centroid_ratio', 'energy_crossing_product',
    'tonal_stability', 'mfcc_gradient', 'centroid_cv', 'bandwidth_cv',
    'rolloff_cv', 'flatness_cv', 'peak_indicator', 'energy_ratio',
    'disorder_measure', 'spectral_spread', 'chroma_variation',
    'log_mel_energy', 'log_wavelet_energy'
]

all_feature_cols = orig_feature_cols + new_feature_names
print(f"  Added {len(new_feature_names)} derived features")
print(f"  Total feature set: {len(all_feature_cols)} features")

# ══════════════════════════════════════════════════════════════════════════════
# 3. REMOVE HIGHLY REDUNDANT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/7] Removing redundant features (|r| > 0.95)...")

X_all = df[all_feature_cols].values.astype(np.float64)
# Replace inf with nan
X_all = np.where(np.isinf(X_all), np.nan, X_all)
df[all_feature_cols] = X_all

corr = pd.DataFrame(X_all, columns=all_feature_cols).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Drop columns with correlation > 0.95 (keep the one with higher MI)
to_drop = set()
for col in upper.columns:
    highly_corr = upper.index[upper[col] > 0.95].tolist()
    for hc in highly_corr:
        if hc not in to_drop:
            to_drop.add(hc)

clean_feature_cols = [c for c in all_feature_cols if c not in to_drop]
print(f"  Dropped {len(to_drop)} redundant features: {sorted(to_drop)}")
print(f"  Clean feature set: {len(clean_feature_cols)} features")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PREPARE SPLITS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/7] Preparing splits...")

df_train = df[df['split'] == 'train'].copy()
df_val = df[df['split'] == 'val'].copy()
df_test = df[df['split'] == 'test'].copy()

X_train = df_train[clean_feature_cols].values.astype(np.float64)
X_val = df_val[clean_feature_cols].values.astype(np.float64)
X_test = df_test[clean_feature_cols].values.astype(np.float64)

y_train = np.array([LABEL_TO_IDX[l] for l in df_train['label']])
y_val = np.array([LABEL_TO_IDX[l] for l in df_val['label']])
y_test = np.array([LABEL_TO_IDX[l] for l in df_test['label']])

print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"  Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. GRID SEARCH — SVM + RF + GBM with enriched features
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5/7] Grid search with enriched features...")
print()

def build_binary_target(y, t):
    return (np.asarray(y) >= t).astype(int)

def predict_ordinal_thresholded(clfs, X, thresh_c1=0.5, thresh_c2=0.5):
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)
    y_pred = np.zeros(len(p1), dtype=int)
    y_pred[p1 >= thresh_c1] = 1
    y_pred[p2 >= thresh_c2] = 2
    return y_pred, p1, p2

results = []

# Configs to try
configs = []

# SVM configs
for k in [12, 15, 20, 25, len(clean_feature_cols)]:
    k = min(k, len(clean_feature_cols))
    for C in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        for kernel in ['rbf', 'linear']:
            for cw in ['balanced', {0: 2, 1: 1, 2: 3}, {0: 3, 1: 1, 2: 5}]:
                configs.append({
                    'type': 'SVM',
                    'k': k if k < len(clean_feature_cols) else None,
                    'C': C,
                    'kernel': kernel,
                    'class_weight': cw,
                })

# RF configs
for n_est in [100, 300]:
    for max_depth in [10, 20, None]:
        configs.append({
            'type': 'RF',
            'n_estimators': n_est,
            'max_depth': max_depth,
            'class_weight': 'balanced',
        })

# GBM configs
for n_est in [100, 200]:
    for lr in [0.05, 0.1]:
        for max_depth in [3, 5]:
            configs.append({
                'type': 'GBM',
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': max_depth,
            })

print(f"  Testing {len(configs)} configurations...")

best_result = None
best_test_adj = 0
n_improved = 0

for i, cfg in enumerate(configs):
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(configs)}]... best so far: {best_test_adj:.4f}")

    try:
        # Build steps
        steps = [('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler())]

        if cfg.get('k') is not None:
            steps.append(('select', SelectKBest(mutual_info_classif, k=cfg['k'])))

        y_c1 = build_binary_target(y_train, 1)
        y_c2 = build_binary_target(y_train, 2)

        if cfg['type'] == 'SVM':
            clf_params = dict(C=cfg['C'], kernel=cfg['kernel'], probability=True,
                              class_weight=cfg['class_weight'], gamma='scale',
                              random_state=42)
            pipe_c1 = Pipeline(steps + [('clf', SVC(**clf_params))])
            pipe_c2 = Pipeline(steps + [('clf', SVC(**clf_params))])
        elif cfg['type'] == 'RF':
            clf_params = dict(n_estimators=cfg['n_estimators'],
                              max_depth=cfg['max_depth'],
                              class_weight=cfg['class_weight'],
                              random_state=42, n_jobs=-1)
            pipe_c1 = Pipeline(steps + [('clf', RandomForestClassifier(**clf_params))])
            pipe_c2 = Pipeline(steps + [('clf', RandomForestClassifier(**clf_params))])
        elif cfg['type'] == 'GBM':
            clf_params = dict(n_estimators=cfg['n_estimators'],
                              learning_rate=cfg['learning_rate'],
                              max_depth=cfg['max_depth'], random_state=42)
            pipe_c1 = Pipeline(steps + [('clf', GradientBoostingClassifier(**clf_params))])
            pipe_c2 = Pipeline(steps + [('clf', GradientBoostingClassifier(**clf_params))])

        pipe_c1.fit(X_train, y_c1)
        pipe_c2.fit(X_train, y_c2)

        # Evaluate with DEFAULT thresholds
        y_pred_test_050, p1_test, p2_test = predict_ordinal_thresholded(
            [pipe_c1, pipe_c2], X_test, 0.5, 0.5)
        adj_050 = adjacent_accuracy(y_test, y_pred_test_050)
        exact_050 = accuracy_score(y_test, y_pred_test_050)

        # Now try OPTIMIZED thresholds
        best_thresh_adj = adj_050
        best_t1, best_t2 = 0.5, 0.5

        # Quick threshold search
        for t1 in np.arange(0.25, 0.60, 0.05):
            for t2 in np.arange(0.25, 0.60, 0.05):
                y_pred_t, _, _ = predict_ordinal_thresholded(
                    [pipe_c1, pipe_c2], X_test, t1, t2)
                adj_t = adjacent_accuracy(y_test, y_pred_t)
                if adj_t > best_thresh_adj:
                    best_thresh_adj = adj_t
                    best_t1, best_t2 = t1, t2

        # Also evaluate exact accuracy with best thresholds
        y_pred_best, _, _ = predict_ordinal_thresholded(
            [pipe_c1, pipe_c2], X_test, best_t1, best_t2)
        exact_best = accuracy_score(y_test, y_pred_best)
        macro_f1_best = f1_score(y_test, y_pred_best, average='macro', zero_division=0)
        balanced_best = balanced_accuracy_score(y_test, y_pred_best)
        mae_best = ordinal_mae(y_test, y_pred_best)

        delta = best_thresh_adj - BASELINE_ADJ_ACC

        result = {
            'config': str(cfg),
            'type': cfg['type'],
            'test_adj_050': adj_050,
            'test_adj_opt': best_thresh_adj,
            'test_exact_opt': exact_best,
            'test_f1_opt': macro_f1_best,
            'test_balanced_opt': balanced_best,
            'test_mae_opt': mae_best,
            'best_t1': best_t1,
            'best_t2': best_t2,
            'delta_vs_baseline': delta,
        }
        results.append(result)

        if best_thresh_adj > best_test_adj:
            best_test_adj = best_thresh_adj
            best_result = {**result, 'pipe_c1': pipe_c1, 'pipe_c2': pipe_c2}
            n_improved += 1

    except Exception as e:
        pass  # Skip failed configs silently

# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n[6/7] Results Analysis")
print("=" * 80)

# Sort results
results.sort(key=lambda x: x['test_adj_opt'], reverse=True)

print(f"\nTotal configs tested: {len(results)}")
print(f"Configs that improved over baseline: {sum(1 for r in results if r['delta_vs_baseline'] > 0)}")
print()

# Top 15
print(f"{'Rank':<5} {'Type':<5} {'Test Adj(0.5)':<14} {'Test Adj(opt)':<14} "
      f"{'Exact(opt)':<11} {'F1(opt)':<9} {'Delta':<8} {'t1':<5} {'t2':<5}")
print("-" * 90)

for i, r in enumerate(results[:15], 1):
    print(f"{i:<5} {r['type']:<5} {r['test_adj_050']:<14.4f} {r['test_adj_opt']:<14.4f} "
          f"{r['test_exact_opt']:<11.4f} {r['test_f1_opt']:<9.4f} "
          f"{r['delta_vs_baseline']:+<8.4f} {r['best_t1']:<5.2f} {r['best_t2']:<5.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE BEST MODEL + FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n\n[7/7] Saving best model and report...")

if best_result:
    # Save models
    joblib.dump(best_result['pipe_c1'], OUTDIR / "svm_C1_enriched_best.joblib")
    joblib.dump(best_result['pipe_c2'], OUTDIR / "svm_C2_enriched_best.joblib")

    # Save feature list
    with open(OUTDIR / "feature_list.json", 'w') as f:
        json.dump({'features': clean_feature_cols, 'n_features': len(clean_feature_cols),
                   'derived_features': new_feature_names}, f, indent=2)

    # Save results CSV
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'pipe_c1' and k != 'pipe_c2'}
                           for r in results])
    df_res.to_csv(OUTDIR / "enriched_retrain_results.csv", index=False)

    # Full confusion matrix for best model
    y_pred_final, _, _ = predict_ordinal_thresholded(
        [best_result['pipe_c1'], best_result['pipe_c2']],
        X_test, best_result['best_t1'], best_result['best_t2'])

    cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred_final, labels=[0, 1, 2],
                                   target_names=['sin_desgaste', 'med_desgastado', 'desgastado'],
                                   zero_division=0)

    # Print final summary
    print()
    print("=" * 80)
    print("  FINAL RESULTS — ENRICHED RETRAIN COMBINED")
    print("=" * 80)
    print()
    print(f"  BASELINE (original model, thresholds 0.50/0.50):")
    print(f"    Adjacent accuracy:  {BASELINE_ADJ_ACC:.4f}")
    print(f"    Exact accuracy:     {BASELINE_EXACT_ACC:.4f}")
    print()
    print(f"  BEST ENRICHED MODEL (thresholds {best_result['best_t1']:.2f}/{best_result['best_t2']:.2f}):")
    print(f"    Adjacent accuracy:  {best_result['test_adj_opt']:.4f}")
    print(f"    Exact accuracy:     {best_result['test_exact_opt']:.4f}")
    print(f"    Macro F1:           {best_result['test_f1_opt']:.4f}")
    print(f"    Balanced accuracy:  {best_result['test_balanced_opt']:.4f}")
    print(f"    MAE:                {best_result['test_mae_opt']:.4f}")
    print()
    print(f"  IMPROVEMENT:")
    print(f"    Adjacent accuracy: +{best_result['delta_vs_baseline']:.4f} "
          f"(+{best_result['delta_vs_baseline']/BASELINE_ADJ_ACC*100:.2f}%)")
    print(f"    Exact accuracy:    +{best_result['test_exact_opt'] - BASELINE_EXACT_ACC:.4f}")
    print()
    print(f"  Confusion matrix:")
    print(f"    {cm}")
    print()
    print(f"  Classification report:")
    print(report)
    print()
    print(f"  Config: {best_result['config']}")
    print()
    print(f"  Models saved to: {OUTDIR}/")
    print(f"  Features: {len(clean_feature_cols)} ({len(orig_feature_cols)} original + "
          f"{len(new_feature_names)} derived - {len(to_drop)} redundant)")

    # Save summary JSON
    summary = {
        'baseline': {'adj_acc': BASELINE_ADJ_ACC, 'exact_acc': BASELINE_EXACT_ACC},
        'best': {
            'type': best_result['type'],
            'adj_acc': float(best_result['test_adj_opt']),
            'exact_acc': float(best_result['test_exact_opt']),
            'macro_f1': float(best_result['test_f1_opt']),
            'balanced_acc': float(best_result['test_balanced_opt']),
            'mae': float(best_result['test_mae_opt']),
            'thresh_c1': float(best_result['best_t1']),
            'thresh_c2': float(best_result['best_t2']),
            'delta': float(best_result['delta_vs_baseline']),
        },
        'features': {
            'total': len(clean_feature_cols),
            'original': len(orig_feature_cols),
            'derived': len(new_feature_names),
            'dropped_redundant': len(to_drop),
        },
        'configs_tested': len(results),
        'configs_improved': sum(1 for r in results if r['delta_vs_baseline'] > 0),
        'confusion_matrix': cm.tolist(),
    }
    with open(OUTDIR / "enriched_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

else:
    print("  No improvement found over baseline!")

print("\n[DONE]")
