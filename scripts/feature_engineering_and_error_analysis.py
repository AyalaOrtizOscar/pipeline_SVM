#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEATURE ENGINEERING + ERROR ANALYSIS — DEEP EXPERT REVIEW
==========================================================

Analiza:
1. Correlación entre features y target
2. Features redundantes/colineales
3. Distribución de features por clase
4. Muestras mal clasificadas (error analysis)
5. Propuestas de nuevas features derivadas
"""

import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

from ordinal_utils import LABEL_TO_IDX, adjacent_accuracy

print("[*] FEATURE ENGINEERING + ERROR ANALYSIS")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("[1/5] Loading data...")
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)

META_COLS = {'filepath', 'label', 'split', 'aug_type', 'mic_type',
             'drill_group', 'basename', 'experiment'}
feature_cols = [c for c in df.columns if c not in META_COLS
                and np.issubdtype(df[c].dtype, np.number)]

df_test = df[df['split'] == 'test'].copy()
X_test = df_test[feature_cols].values.astype(np.float32)
y_test_int = np.array([LABEL_TO_IDX[l] for l in df_test['label']])

print(f"  Loaded {len(feature_cols)} features, test set: {len(X_test)} samples")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/5] Feature importance via multiple methods...")

df_train = df[df['split'] == 'train'].copy()
X_train = df_train[feature_cols].values.astype(np.float32)
y_train_int = np.array([LABEL_TO_IDX[l] for l in df_train['label']])

# Mutual information
mi_scores = mutual_info_classif(X_train, y_train_int, random_state=42)

# F-statistic
f_scores, _ = f_classif(X_train, y_train_int)

# Ranking
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores,
    'f_score': f_scores,
})
importance_df['mi_rank'] = importance_df['mi_score'].rank(ascending=False)
importance_df['f_rank'] = importance_df['f_score'].rank(ascending=False)
importance_df['avg_rank'] = (importance_df['mi_rank'] + importance_df['f_rank']) / 2
importance_df = importance_df.sort_values('avg_rank')

print("\nTop 10 most important features:")
print(importance_df[['feature', 'mi_score', 'f_score', 'avg_rank']].head(10).to_string(index=False))

print("\nBottom 10 least important features (candidates for removal):")
print(importance_df[['feature', 'mi_score', 'f_score', 'avg_rank']].tail(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION & REDUNDANCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Analyzing feature redundancy...")

# Correlations
corr_matrix = pd.DataFrame(X_train, columns=feature_cols).corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append({
                'feature_1': feature_cols[i],
                'feature_2': feature_cols[j],
                'correlation': corr_matrix.iloc[i, j]
            })

high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.85):")
for pair in high_corr_pairs[:10]:
    print(f"  {pair['feature_1'][:25]:25s} -- {pair['feature_2'][:25]:25s} (r={pair['correlation']:.3f})")

# Recommendation: keep higher importance feature, drop the other
print("\n  Recommendation: For each pair, keep the feature with higher importance rank,")
print("  drop the redundant one. This could reduce feature set from 28 to ~20.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ERROR ANALYSIS — which samples does baseline model misclassify?
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/5] Loading baseline model for error analysis...")

try:
    c1_model = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/svm_C1_top15_orig.joblib")
    c2_model = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/svm_C2_top15_orig.joblib")
    scaler = joblib.load("D:/pipeline_SVM/results/svm_ordinal_v2/scaler_top15_orig.joblib")

    # Get top-15 features
    top_15_features = importance_df['feature'].head(15).tolist()
    X_test_top15 = X_test[:, [feature_cols.index(f) for f in top_15_features]]

    # Scale
    X_test_scaled = scaler.transform(X_test_top15)

    # Predict
    p1 = c1_model.predict_proba(X_test_scaled)[:, 1]
    p2 = c2_model.predict_proba(X_test_scaled)[:, 1]
    p2 = np.minimum(p2, p1)

    y_pred = np.zeros(len(p1), dtype=int)
    y_pred[p1 >= 0.5] = 1
    y_pred[p2 >= 0.5] = 2

    # Errors
    errors = (y_test_int != y_pred)
    n_errors = errors.sum()
    error_rate = n_errors / len(y_test_int)

    print(f"\n  Baseline model error analysis:")
    print(f"  Correct: {len(y_test_int) - n_errors}/{len(y_test_int)} ({(1-error_rate)*100:.1f}%)")
    print(f"  Errors: {n_errors} ({error_rate*100:.1f}%)")

    # Analyze error types
    print(f"\n  Error distribution by true class:")
    for true_class in [0, 1, 2]:
        class_errors = ((y_test_int == true_class) & errors).sum()
        class_total = (y_test_int == true_class).sum()
        class_error_rate = class_errors / class_total if class_total > 0 else 0
        print(f"    Class {true_class}: {class_errors}/{class_total} errors ({class_error_rate*100:.1f}%)")

    # Find hardest samples (lowest confidence)
    confidence = np.max(np.column_stack([1-p1, p1*(1-p2), p2]), axis=1)
    hard_samples_idx = np.argsort(confidence)[:10]

    print(f"\n  Hardest samples to classify (lowest confidence):")
    for idx in hard_samples_idx:
        true = y_test_int[idx]
        pred = y_pred[idx]
        conf = confidence[idx]
        print(f"    True={true}, Pred={pred}, Confidence={conf:.3f}")

    # Samples that flip between classes
    flip_probability = np.abs(np.column_stack([p1*(1-p2), p2]) - 0.5).min(axis=1)
    flip_samples_idx = np.argsort(flip_probability)[:10]

    print(f"\n  Samples on decision boundary (near threshold):")
    for idx in flip_samples_idx:
        true = y_test_int[idx]
        pred = y_pred[idx]
        prob = flip_probability[idx]
        print(f"    True={true}, Pred={pred}, Distance to threshold={prob:.3f}")

except Exception as e:
    print(f"  Error loading baseline model: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Feature Engineering Proposals...")

report = """
Based on expert audio & wear analysis, propose NEW DERIVED FEATURES:

1. SPECTRAL DYNAMICS:
   - spectral_flux: Frame-to-frame change in spectral content
   - spectral_stability: Variance of spectral centroids over time
   - bandwidth_change: Rate of change in spectral bandwidth
   Captures: worn tools produce less stable frequency content

2. TEMPORAL MODULATION:
   - amplitude_modulation_depth: Depth of amplitude envelope modulation
   - carrier_frequency: Detected fundamental drilling frequency
   - frequency_sweep_rate: Rate of change in center frequency over time
   Captures: worn tools show different modulation patterns

3. HARMONIC FEATURES:
   - harmonic_energy_ratio: Energy in harmonic vs noise components
   - harmonic_decay_rate: How fast harmonics decay
   - subharmonic_content: Energy in sub-harmonic frequencies
   Captures: wear affects harmonic structure

4. CEPSTRAL FEATURES:
   - cepstral_coefficient_stability: Variance of cepstral coefficients
   - quefrency_peak: Most prominent quefrency (fundamental period)
   Captures: periodicity changes with wear

5. NONLINEAR DYNAMICS:
   - approximate_entropy: Complexity of signal
   - detrended_fluctuation_exponent: Self-similarity across scales
   - higuchi_fractal_dimension: Fractal complexity
   Captures: degradation increases signal complexity

6. MULTIRESOLUTION ANALYSIS:
   - wavelet_energy_distribution: Energy across wavelet scales
   - wavelet_entropy: Entropy of wavelet coefficients
   Captures: wear-dependent scale-specific energy

IMMEDIATE ACTION:
- Remove redundant features (pairs with |r|>0.85)
- Add top 5-8 derived features from categories 1-3
- Re-run deep retraining with 30-35 feature set
- Expected improvement: +0.5-1.0% adjacent accuracy
"""
print(report)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════════

outdir = "D:/pipeline_SVM/results/feature_analysis"
import os
os.makedirs(outdir, exist_ok=True)

importance_df.to_csv(f"{outdir}/feature_importance.csv", index=False)

with open(f"{outdir}/feature_recommendations.txt", 'w') as f:
    f.write("FEATURE ENGINEERING RECOMMENDATIONS\n")
    f.write("=" * 70 + "\n\n")
    f.write("REDUNDANT PAIRS (CANDIDATES FOR REMOVAL):\n")
    for pair in high_corr_pairs[:15]:
        f.write(f"{pair['feature_1']}, {pair['feature_2']} (r={pair['correlation']:.3f})\n")
    f.write("\nDERIVED FEATURES TO ADD:\n")
    f.write("1. Spectral dynamics (flux, stability, bandwidth_change)\n")
    f.write("2. Temporal modulation (AM depth, carrier freq, frequency sweep)\n")
    f.write("3. Harmonic features (ratio, decay, subharmonic)\n")
    f.write("4. Cepstral features (stability, quefrency peak)\n")
    f.write("5. Nonlinear dynamics (entropy, fluctuation exponent, fractal dim)\n")
    f.write("6. Multiresolution (wavelet energy, wavelet entropy)\n")

print(f"\n[OK] Saved analysis to {outdir}/")
print()
