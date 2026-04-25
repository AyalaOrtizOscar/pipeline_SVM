#!/usr/bin/env python3
"""
analyze_and_train_v2.py

Pipeline completo post-reetiquetado [15/75]:
  1. Analisis exploratorio (PCA, t-SNE)
  2. Mutual Information para seleccion de features
  3. Entrenamiento SVM ordinal (Frank & Hall)
  4. Evaluacion en val (E2) y test (E3)
  5. SHAP + permutation importance
  6. Generacion de figuras para el articulo

Autor: Claude Code + Oscar Ayala
Fecha: 2026-03-22
"""

import os, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuracion ─────────────────────────────────────────────────────────────

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/svm_ordinal_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 10 features seleccionadas (linea base del articulo)
BASELINE_10 = [
    "harmonic_percussive_ratio",
    "centroid_mean",
    "zcr",
    "spectral_flatness_mean",
    "spectral_entropy_mean",
    "onset_rate",
    "duration_s",
    "crest_factor",
    "chroma_std",
    "spectral_contrast_mean",
]

# Todas las features numericas disponibles
ALL_FEATURES = BASELINE_10 + [
    "rms", "rms_db", "peak", "mel_total_energy",
    "centroid_std", "rolloff_mean", "rolloff_std",
    "spectral_flatness_std", "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "mfcc_0_mean", "mfcc_1_mean", "chroma_mean_first", "chroma_mean",
    "tonnetz_0_mean", "tempo", "wavelet_total_energy",
]

# Orden ordinal
LABEL_ORDER = ["sin_desgaste", "medianamente_desgastado", "desgastado"]
LABEL_MAP = {l: i for i, l in enumerate(LABEL_ORDER)}


# ── Metricas ordinales ───────────────────────────────────────────────────────

def ordinal_metrics(y_true_str, y_pred_str):
    """Calcula metricas ordinales."""
    y_true = np.array([LABEL_MAP[l] for l in y_true_str])
    y_pred = np.array([LABEL_MAP[l] for l in y_pred_str])

    exact_acc = np.mean(y_true == y_pred)
    adj_acc = np.mean(np.abs(y_true - y_pred) <= 1)
    mae = np.mean(np.abs(y_true - y_pred))
    macro_f1 = f1_score(y_true_str, y_pred_str, average="macro", labels=LABEL_ORDER)

    per_class_f1 = {}
    for label in LABEL_ORDER:
        mask = np.array(y_true_str) == label
        if mask.sum() > 0:
            pred_for_class = np.array(y_pred_str)[mask]
            per_class_f1[label] = np.mean(pred_for_class == label)
        else:
            per_class_f1[label] = float("nan")

    return {
        "exact_accuracy": float(exact_acc),
        "adjacent_accuracy": float(adj_acc),
        "ordinal_mae": float(mae),
        "macro_f1": float(macro_f1),
        "per_class_accuracy": per_class_f1,
    }


# ── Frank & Hall (2 clasificadores binarios) ─────────────────────────────────

def encode_frank_hall(y_str):
    """Codifica labels para Frank & Hall: C1=P(y>=1), C2=P(y>=2)."""
    y = np.array([LABEL_MAP[l] for l in y_str])
    c1 = (y >= 1).astype(int)  # algún desgaste?
    c2 = (y >= 2).astype(int)  # desgaste severo?
    return c1, c2


def decode_frank_hall(p1, p2):
    """Decodifica predicciones binarias a labels ordinales."""
    # Forzar monotonicidad: P(y>=2) <= P(y>=1)
    p2 = np.minimum(p1, p2)
    labels = []
    for pi1, pi2 in zip(p1, p2):
        probs = [1 - pi1, pi1 - pi2, pi2]
        idx = int(np.argmax(probs))
        labels.append(LABEL_ORDER[idx])
    return labels


# ── Carga de datos ───────────────────────────────────────────────────────────

def load_data():
    """Carga features y separa splits."""
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded: {len(df)} rows")

    # Solo usar features que existen en el CSV
    available = [f for f in ALL_FEATURES if f in df.columns]
    print(f"Available features: {len(available)}/{len(ALL_FEATURES)}")

    # Split
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # Solo originales para entrenamiento SVM
    df_train_orig = df_train[df_train["aug_type"] == "original"].copy()

    print(f"Train: {len(df_train)} ({len(df_train_orig)} orig), Val: {len(df_val)}, Test: {len(df_test)}")
    print(f"Train labels: {df_train_orig['label'].value_counts().to_dict()}")
    print(f"Val labels:   {df_val['label'].value_counts().to_dict()}")
    print(f"Test labels:  {df_test['label'].value_counts().to_dict()}")

    return df, df_train, df_train_orig, df_val, df_test, available


# ── 1. Mutual Information ────────────────────────────────────────────────────

def analyze_mutual_information(df_train_orig, features, outdir):
    """Calcula MI de cada feature con la clase."""
    print("\n[1/5] Mutual Information...")
    X = df_train_orig[features].values
    y = df_train_orig["label"].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": features, "MI": mi}).sort_values("MI", ascending=False)
    mi_df.to_csv(outdir / "mutual_information.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top = mi_df.head(15)
    colors = ["#e74c3c" if f in BASELINE_10 else "#3498db" for f in top["feature"]]
    ax.barh(range(len(top)), top["MI"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("Feature Importance: Mutual Information con clase de desgaste\n(rojo = baseline 10, azul = otras)")
    plt.tight_layout()
    fig.savefig(outdir / "mutual_information.png", dpi=150)
    plt.close(fig)

    print(f"  Top 5 MI: {mi_df.head(5)[['feature','MI']].to_string(index=False)}")
    return mi_df


# ── 2. PCA ───────────────────────────────────────────────────────────────────

def analyze_pca(df_orig, features, outdir):
    """PCA 2D del dataset completo (solo originales)."""
    print("\n[2/5] PCA...")
    df_plot = df_orig[df_orig["aug_type"] == "original"].copy()
    X = np.nan_to_num(df_plot[features].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By label
    colors_label = {"sin_desgaste": "#2ecc71", "medianamente_desgastado": "#f1c40f", "desgastado": "#e74c3c"}
    for label, color in colors_label.items():
        mask = df_plot["label"] == label
        axes[0].scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"],
                       c=color, label=label, alpha=0.5, s=15)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("PCA - Colored by Wear Label")
    axes[0].legend()

    # By experiment
    exp_colors = {"E1": "#e74c3c", "E2": "#3498db", "E3": "#2ecc71",
                  "E4": "#9b59b6", "E5": "#f39c12", "E6": "#1abc9c", "E7": "#e67e22"}
    for exp, color in exp_colors.items():
        mask = df_plot["experiment"] == exp
        if mask.sum() > 0:
            axes[1].scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"],
                           c=color, label=exp, alpha=0.5, s=15)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("PCA - Colored by Experiment")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "pca_2d.png", dpi=150)
    plt.close(fig)

    print(f"  Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    return pca


# ── 3. t-SNE ─────────────────────────────────────────────────────────────────

def analyze_tsne(df_orig, features, outdir):
    """t-SNE 2D del dataset (solo originales, subsampled si necesario)."""
    print("\n[3/5] t-SNE...")
    df_plot = df_orig[df_orig["aug_type"] == "original"].copy()

    # Subsample if too large
    MAX_SAMPLES = 2000
    if len(df_plot) > MAX_SAMPLES:
        df_plot = df_plot.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_SAMPLES // 3), random_state=RANDOM_STATE)
        )
        print(f"  Subsampled to {len(df_plot)} for t-SNE")

    X = np.nan_to_num(df_plot[features].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    df_plot = df_plot.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors_label = {"sin_desgaste": "#2ecc71", "medianamente_desgastado": "#f1c40f", "desgastado": "#e74c3c"}
    for label, color in colors_label.items():
        mask = df_plot["label"] == label
        axes[0].scatter(df_plot.loc[mask, "tSNE1"], df_plot.loc[mask, "tSNE2"],
                       c=color, label=label, alpha=0.5, s=15)
    axes[0].set_title("t-SNE - Colored by Wear Label")
    axes[0].legend()

    exp_colors = {"E1": "#e74c3c", "E2": "#3498db", "E3": "#2ecc71",
                  "E4": "#9b59b6", "E5": "#f39c12", "E6": "#1abc9c", "E7": "#e67e22"}
    for exp, color in exp_colors.items():
        mask = df_plot["experiment"] == exp
        if mask.sum() > 0:
            axes[1].scatter(df_plot.loc[mask, "tSNE1"], df_plot.loc[mask, "tSNE2"],
                           c=color, label=exp, alpha=0.5, s=15)
    axes[1].set_title("t-SNE - Colored by Experiment")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(outdir / "tsne_2d.png", dpi=150)
    plt.close(fig)
    print(f"  t-SNE done (KL divergence: {tsne.kl_divergence_:.2f})")


# ── 4. SVM Ordinal (Frank & Hall) ────────────────────────────────────────────

def train_svm_ordinal(df_train_orig, df_val, df_test, features, outdir):
    """Entrena 2 SVMs (Frank & Hall) con GridSearchCV."""
    print("\n[4/5] Training SVM Ordinal (Frank & Hall)...")

    # Prepare data
    X_train = np.nan_to_num(df_train_orig[features].values)
    y_train = df_train_orig["label"].values
    groups = df_train_orig["experiment"].values

    X_val = np.nan_to_num(df_val[features].values)
    y_val = df_val["label"].values

    X_test = np.nan_to_num(df_test[features].values)
    y_test = df_test["label"].values

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Frank & Hall encoding
    c1_train, c2_train = encode_frank_hall(y_train)
    c1_val, c2_val = encode_frank_hall(y_val)
    c1_test, c2_test = encode_frank_hall(y_test)

    # Grid search para cada clasificador binario
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1],
        "kernel": ["rbf"],
    }

    # CV strategy: StratifiedGroupKFold por experimento
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for name, c_train, c_val, c_test in [
        ("C1_any_wear", c1_train, c1_val, c1_test),
        ("C2_severe_wear", c2_train, c2_val, c2_test),
    ]:
        print(f"\n  --- {name} ---")
        print(f"  Train: {np.bincount(c_train)} (0s vs 1s)")

        # Class weight to handle imbalance
        grid = GridSearchCV(
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train_s, c_train, groups=groups)

        best = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1: {grid.best_score_:.3f}")

        # Predictions
        p_val = best.predict_proba(X_val_s)[:, 1]
        p_test = best.predict_proba(X_test_s)[:, 1]

        results[name] = {
            "model": best,
            "p_val": p_val,
            "p_test": p_test,
            "best_params": grid.best_params_,
            "cv_f1": grid.best_score_,
        }

        # Save model
        joblib.dump(best, outdir / f"svm_{name}.joblib")

    # Save scaler
    joblib.dump(scaler, outdir / "scaler.joblib")

    # Decode ordinal predictions
    y_pred_val = decode_frank_hall(
        results["C1_any_wear"]["p_val"],
        results["C2_severe_wear"]["p_val"]
    )
    y_pred_test = decode_frank_hall(
        results["C1_any_wear"]["p_test"],
        results["C2_severe_wear"]["p_test"]
    )

    # Metrics
    print("\n  === VAL Results (E2) ===")
    metrics_val = ordinal_metrics(y_val, y_pred_val)
    print(f"  Macro F1: {metrics_val['macro_f1']:.3f}")
    print(f"  Exact Acc: {metrics_val['exact_accuracy']:.3f}")
    print(f"  Adjacent Acc: {metrics_val['adjacent_accuracy']:.3f}")
    print(f"  Ordinal MAE: {metrics_val['ordinal_mae']:.3f}")
    print(f"  {classification_report(y_val, y_pred_val, labels=LABEL_ORDER)}")

    print("\n  === TEST Results (E3) ===")
    metrics_test = ordinal_metrics(y_test, y_pred_test)
    print(f"  Macro F1: {metrics_test['macro_f1']:.3f}")
    print(f"  Exact Acc: {metrics_test['exact_accuracy']:.3f}")
    print(f"  Adjacent Acc: {metrics_test['adjacent_accuracy']:.3f}")
    print(f"  Ordinal MAE: {metrics_test['ordinal_mae']:.3f}")
    print(f"  {classification_report(y_test, y_pred_test, labels=LABEL_ORDER)}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_true, y_pred, title in [
        (axes[0], y_val, y_pred_val, "VAL (E2 - 8mm, dynamic)"),
        (axes[1], y_test, y_pred_test, "TEST (E3 - 6mm, condenser)"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
        disp = ConfusionMatrixDisplay(cm, display_labels=["sin", "med", "des"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outdir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Save predictions
    df_val_pred = df_val[["filepath", "label", "experiment"]].copy()
    df_val_pred["prediction"] = y_pred_val
    df_val_pred["P_any_wear"] = results["C1_any_wear"]["p_val"]
    df_val_pred["P_severe_wear"] = results["C2_severe_wear"]["p_val"]
    df_val_pred.to_csv(outdir / "predictions_val.csv", index=False)

    df_test_pred = df_test[["filepath", "label", "experiment"]].copy()
    df_test_pred["prediction"] = y_pred_test
    df_test_pred["P_any_wear"] = results["C1_any_wear"]["p_test"]
    df_test_pred["P_severe_wear"] = results["C2_severe_wear"]["p_test"]
    df_test_pred.to_csv(outdir / "predictions_test.csv", index=False)

    # Save metrics
    all_metrics = {
        "val": metrics_val,
        "test": metrics_test,
        "C1_best_params": results["C1_any_wear"]["best_params"],
        "C2_best_params": results["C2_severe_wear"]["best_params"],
        "C1_cv_f1": results["C1_any_wear"]["cv_f1"],
        "C2_cv_f1": results["C2_severe_wear"]["cv_f1"],
        "features_used": features,
        "n_features": len(features),
        "thresholds": {"sin_max_pct": 0.15, "des_min_pct": 0.75},
        "train_n": len(df_train_orig),
        "val_n": len(df_val),
        "test_n": len(df_test),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    return results, scaler, X_train_s, y_train, X_test_s, y_test, y_pred_test, metrics_val, metrics_test


# ── 5. Permutation Importance + SHAP ─────────────────────────────────────────

def analyze_importance(results, scaler, X_test, y_test, features, outdir):
    """Permutation importance en test set."""
    print("\n[5/5] Feature Importance (permutation)...")

    # Use C1 model (main discriminator)
    model_c1 = results["C1_any_wear"]["model"]

    X_test_s = scaler.transform(np.nan_to_num(X_test))
    c1_test = (np.array([LABEL_MAP[l] for l in y_test]) >= 1).astype(int)

    perm = permutation_importance(
        model_c1, X_test_s, c1_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(outdir / "permutation_importance.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top = perm_df.head(15)
    colors = ["#e74c3c" if f in BASELINE_10 else "#3498db" for f in top["feature"]]
    ax.barh(range(len(top)), top["importance_mean"].values, xerr=top["importance_std"].values,
            color=colors, capsize=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean F1 decrease")
    ax.set_title("Permutation Importance (C1: any wear) on TEST set\n(rojo = baseline 10, azul = otras)")
    plt.tight_layout()
    fig.savefig(outdir / "permutation_importance.png", dpi=150)
    plt.close(fig)

    print(f"  Top 5:")
    for _, row in perm_df.head(5).iterrows():
        print(f"    {row['feature']:35s} {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")

    return perm_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*65}")
    print(f"  ANALYSIS & TRAINING PIPELINE v2 (relabeled [15/75])")
    print(f"  {ts}")
    print(f"{'='*65}")

    df, df_train, df_train_orig, df_val, df_test, available = load_data()

    # Use available features from baseline 10 (primary), plus extras
    features_10 = [f for f in BASELINE_10 if f in available]
    features_all = [f for f in available if f in ALL_FEATURES]
    print(f"\nUsing baseline 10: {len(features_10)} features")
    print(f"All available: {len(features_all)} features")

    # 1. MI analysis (on all features)
    mi_df = analyze_mutual_information(df_train_orig, features_all, OUTDIR)

    # 2. PCA (on baseline 10)
    pca = analyze_pca(df, features_10, OUTDIR)

    # 3. t-SNE (on baseline 10)
    analyze_tsne(df, features_10, OUTDIR)

    # 4. Train SVM (on baseline 10)
    (results, scaler, X_train_s, y_train,
     X_test_raw, y_test, y_pred_test,
     metrics_val, metrics_test) = train_svm_ordinal(
        df_train_orig, df_val, df_test, features_10, OUTDIR
    )

    # 5. Permutation Importance
    perm_df = analyze_importance(
        results, scaler,
        df_test[features_10].values, df_test["label"].values,
        features_10, OUTDIR
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"\n  Relabeling: [15/75] ({LABEL_ORDER})")
    print(f"  Features: baseline 10")
    print(f"  Train: {len(df_train_orig)} orig (E1+E4+E5+E6+E7)")
    print(f"  Val:   {len(df_val)} (E2, 8mm, dynamic mic)")
    print(f"  Test:  {len(df_test)} (E3, 6mm, condenser mic)")
    print(f"\n  VAL  -> Macro F1={metrics_val['macro_f1']:.3f}, Acc={metrics_val['exact_accuracy']:.3f}, MAE={metrics_val['ordinal_mae']:.3f}")
    print(f"  TEST -> Macro F1={metrics_test['macro_f1']:.3f}, Acc={metrics_test['exact_accuracy']:.3f}, MAE={metrics_test['ordinal_mae']:.3f}")
    print(f"\n  Output: {OUTDIR}")
    print(f"  Files: {list(OUTDIR.iterdir())}")


if __name__ == "__main__":
    main()
