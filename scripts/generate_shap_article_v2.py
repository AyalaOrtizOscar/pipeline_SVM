#!/usr/bin/env python3
"""
generate_shap_article_v2.py — SHAP beeswarm usando modelos top-15 funcionales.

Los modelos svm_ordinal_clean/ tienen predict_proba constante (Platt scaling colapsado).
Usamos svm_ordinal_v2/svm_C*_top15_orig.joblib + scaler_top15_orig.joblib, que SI varian.

Se usa decision_function (margen SVM) en vez de predict_proba para evitar collapse.
"""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("D:/pipeline_SVM/results/svm_ordinal_v2")

FEATS15 = [
    "spectral_bandwidth_std", "rolloff_std", "spectral_bandwidth_mean",
    "mfcc_1_mean", "zcr", "centroid_std", "spectral_flatness_mean",
    "rolloff_mean", "centroid_mean", "mfcc_0_mean", "spectral_entropy_mean",
    "crest_factor", "spectral_contrast_mean", "harmonic_percussive_ratio",
    "chroma_std",
]

NICE = {
    "spectral_bandwidth_std": "Ancho banda (std)",
    "rolloff_std": "Rolloff (std)",
    "spectral_bandwidth_mean": "Ancho banda",
    "mfcc_1_mean": "MFCC-1",
    "zcr": "Cruces por cero",
    "centroid_std": "Centroide (std)",
    "spectral_flatness_mean": "Planitud espectral",
    "rolloff_mean": "Rolloff",
    "centroid_mean": "Centroide espectral",
    "mfcc_0_mean": "MFCC-0 (energía)",
    "spectral_entropy_mean": "Entropía espectral",
    "crest_factor": "Factor de cresta",
    "spectral_contrast_mean": "Contraste espectral",
    "harmonic_percussive_ratio": "Ratio armónico/percusivo",
    "chroma_std": "Chroma (std)",
}

N_BG = 150
N_EVAL = 250
RANDOM_STATE = 42

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def make_predict_fn(svc, scaler, use_decision=True):
    """decision_function(scaled) o predict_proba[:,1]."""
    def fn(X_raw):
        Xs = scaler.transform(X_raw)
        if use_decision:
            return svc.decision_function(Xs)
        return svc.predict_proba(Xs)[:, 1]
    return fn


def run_shap(svc_path, scaler_path, title, out_basename, use_decision=True):
    svc = joblib.load(svc_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(FEATURES_CSV)
    df_test = df[(df["split"] == "test") & (df["aug_type"] == "original")]
    df_train = df[(df["split"] == "train") & (df["aug_type"] == "original")]

    X_test = df_test[FEATS15].values
    X_train = df_train[FEATS15].values

    rng = np.random.RandomState(RANDOM_STATE)
    bg = X_train[rng.choice(len(X_train), min(N_BG, len(X_train)), replace=False)]
    ev_idx = rng.choice(len(X_test), min(N_EVAL, len(X_test)), replace=False)
    X_eval = X_test[ev_idx]

    predict_fn = make_predict_fn(svc, scaler, use_decision=use_decision)

    # Sanity check
    out = predict_fn(X_eval)
    print(f"  predict_fn output: min={out.min():.3f}, max={out.max():.3f}, std={out.std():.3f}")
    if out.std() < 1e-4:
        print("  WARN: output is constant — SHAP will be zero")

    print(f"  Computing SHAP ({N_BG} bg, {N_EVAL} eval, 100 nsamples)...")
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X_eval, nsamples=100)

    feat_labels = [NICE.get(f, f) for f in FEATS15]

    # ── Beeswarm ──
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(
        shap_values, X_eval,
        feature_names=feat_labels,
        show=False, plot_size=None, max_display=15,
    )
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel("Valor SHAP (impacto en margen del clasificador)" if use_decision
               else "Valor SHAP (impacto en P(desgaste))", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{out_basename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTDIR / f"{out_basename}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  -> {out_basename}.png / .pdf")

    # ── Bar ranking (|SHAP| medio) ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_labels = [feat_labels[i] for i in order]
    top_vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.barh(range(len(top_labels)), top_vals[::-1], color="#2a6f97")
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels[::-1])
    ax.set_xlabel("|SHAP| medio (magnitud de impacto)", fontsize=11)
    ax.set_title(title + " — ranking de importancia", fontsize=12, pad=10)
    for i, v in enumerate(top_vals[::-1]):
        ax.text(v + max(top_vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{out_basename}_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTDIR / f"{out_basename}_bar.pdf", bbox_inches="tight")
    plt.close()
    print(f"  -> {out_basename}_bar.png / .pdf")

    # CSV
    csv_df = pd.DataFrame({
        "feature": [FEATS15[i] for i in order],
        "feature_es": top_labels,
        "mean_abs_shap": top_vals,
    })
    csv_df.to_csv(OUTDIR / f"{out_basename}_ranking.csv", index=False)


def main():
    print("=" * 60)
    print("  SHAP BEESWARM v2 — Articulo 1 (modelos top-15)")
    print("=" * 60)

    print("\n[C1] P(desgaste >= 1): ¿Hay algún desgaste?")
    run_shap(
        MODEL_DIR / "svm_C1_top15_orig.joblib",
        MODEL_DIR / "scaler_top15_orig.joblib",
        "SHAP — C1: ¿Hay algún desgaste? (margen SVM)",
        "shap_beeswarm_C1_v2",
        use_decision=True,
    )

    print("\n[C2] P(desgaste >= 2): ¿Desgaste severo?")
    run_shap(
        MODEL_DIR / "svm_C2_top15_orig.joblib",
        MODEL_DIR / "scaler_top15_orig.joblib",
        "SHAP — C2: ¿Desgaste severo? (margen SVM)",
        "shap_beeswarm_C2_v2",
        use_decision=True,
    )

    print("\nListo.")


if __name__ == "__main__":
    main()
