#!/usr/bin/env python3
"""
run_comparison.py — Comparativa SVM ordinal: features originales vs filtradas (noisereduce)

Usa features pre-extraidas (features_curated_splits.csv y features_cleaned.csv)
para entrenar y evaluar SVM ordinal Frank & Hall con los mismos splits.
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL,
    ordinal_decode, ordinal_proba,
    ordinal_mae, adjacent_accuracy
)

warnings.filterwarnings("ignore")

FEATURES_ORIG = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
FEATURES_CLEAN = Path("D:/pipeline_SVM/results/comparison_filtered/features_cleaned.csv")
OUTDIR = Path("D:/pipeline_SVM/results/comparison_filtered")

TOP7 = ["spectral_contrast_mean", "crest_factor", "chroma_std", "zcr",
        "spectral_entropy_mean", "centroid_mean", "harmonic_percussive_ratio"]


def build_binary_target(y_int, threshold):
    return (y_int >= threshold).astype(int)


def train_ordinal_svm(X_train, y_train_int, groups_train, n_jobs=4, rs=42):
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=rs)
    param_grid = {"svc__C": [1.0, 10.0], "svc__gamma": ["scale", 0.1]}
    clfs = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train_int, threshold)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, class_weight="balanced",
                        kernel="rbf", random_state=rs)),
        ])
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1",
                          n_jobs=n_jobs, verbose=0, refit=True)
        gs.fit(X_train, y_bin, groups=groups_train)
        clfs.append(gs.best_estimator_)
        print(f"    C{threshold}: best={gs.best_params_}, cv_f1={gs.best_score_:.3f}")
    return clfs


def predict_ordinal(clfs, X):
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)
    probs = np.stack([p1, p2], axis=1)
    return ordinal_decode(probs), probs


def evaluate(clfs, X, y_true_int, label=""):
    y_pred, probs = predict_ordinal(clfs, X)
    f1 = f1_score(y_true_int, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true_int, y_pred)
    mae = ordinal_mae(y_true_int, y_pred)
    adj = adjacent_accuracy(y_true_int, y_pred)

    # Per-class F1
    report = classification_report(
        y_true_int, y_pred, labels=[0, 1, 2],
        target_names=["sin_desgaste", "med_desgastado", "desgastado"],
        output_dict=True, zero_division=0
    )

    if label:
        print(f"  [{label}] F1={f1:.3f}  Acc={acc:.3f}  AdjAcc={adj:.3f}  MAE={mae:.3f}")
        for cls in ["sin_desgaste", "med_desgastado", "desgastado"]:
            cf1 = report[cls]["f1-score"]
            print(f"    {cls}: F1={cf1:.3f}")

    return {
        "macro_f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "adjacent_accuracy": round(adj, 4),
        "ordinal_mae": round(mae, 4),
        "f1_sin_desgaste": round(report["sin_desgaste"]["f1-score"], 4),
        "f1_med_desgastado": round(report["med_desgastado"]["f1-score"], 4),
        "f1_desgastado": round(report["desgastado"]["f1-score"], 4),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  COMPARATIVA: SVM Original vs SVM con Noisereduce")
    print(f"  Features: top-7")
    print("=" * 70)

    # Load both feature sets
    print("\nLoading features...")
    df_orig = pd.read_csv(FEATURES_ORIG, low_memory=False)
    df_orig = df_orig[df_orig["aug_type"] == "original"].copy().reset_index(drop=True)

    df_clean = pd.read_csv(FEATURES_CLEAN, low_memory=False)

    # Align by filepath
    common = set(df_orig["filepath"]) & set(df_clean["filepath"])
    df_orig = df_orig[df_orig["filepath"].isin(common)].sort_values("filepath").reset_index(drop=True)
    df_clean = df_clean[df_clean["filepath"].isin(common)].sort_values("filepath").reset_index(drop=True)
    print(f"  Aligned: {len(df_orig)} samples")

    # Labels and splits
    y_str = df_orig["label"].astype(str).str.strip()
    y_int = np.array([LABEL_TO_IDX[l] for l in y_str])
    split = df_orig["split"].values
    groups = df_orig["experiment"].fillna("unknown").values

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    for cls_name, cls_idx in LABEL_TO_IDX.items():
        n = (y_int == cls_idx).sum()
        n_test = (y_int[test_mask] == cls_idx).sum()
        print(f"    {cls_name}: {n} total, {n_test} in test")

    results = []
    all_preds = {}

    for source_name, df_src in [("original", df_orig), ("filtered", df_clean)]:
        print(f"\n{'='*60}")
        print(f"  Training: {source_name.upper()}")
        print(f"{'='*60}")

        X = df_src[TOP7].values
        X_train, y_train = X[train_mask], y_int[train_mask]
        g_train = groups[train_mask]

        clfs = train_ordinal_svm(X_train, y_train, g_train, n_jobs=4)

        # Save models
        model_dir = OUTDIR / f"models_{source_name}"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(clfs[0], model_dir / "svm_C1.joblib")
        joblib.dump(clfs[1], model_dir / "svm_C2.joblib")

        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            if mask.sum() == 0:
                continue
            metrics = evaluate(clfs, X[mask], y_int[mask], f"{source_name}/{split_name}")
            metrics["source"] = source_name
            metrics["split"] = split_name
            results.append(metrics)

            # Save predictions
            y_pred, probs = predict_ordinal(clfs, X[mask])
            all_preds[f"{source_name}_{split_name}"] = {
                "y_true": y_int[mask],
                "y_pred": y_pred,
                "probs": probs,
            }

    # Results table
    df_results = pd.DataFrame(results)
    results_path = OUTDIR / f"comparison_{ts}.csv"
    df_results.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("  RESULTADOS COMPARATIVOS — SVM Ordinal (Top-7 features)")
    print("=" * 70)
    header = (f"{'Source':<12} {'Split':<6} {'MacroF1':>8} {'Acc':>7} "
              f"{'AdjAcc':>7} {'MAE':>6} {'F1_sin':>7} {'F1_med':>7} {'F1_des':>7}")
    print(header)
    print("-" * len(header))
    for _, row in df_results.iterrows():
        print(f"{row['source']:<12} {row['split']:<6} "
              f"{row['macro_f1']:>8.3f} {row['accuracy']:>7.3f} "
              f"{row['adjacent_accuracy']:>7.3f} {row['ordinal_mae']:>6.3f} "
              f"{row['f1_sin_desgaste']:>7.3f} {row['f1_med_desgastado']:>7.3f} "
              f"{row['f1_desgastado']:>7.3f}")

    # Differences
    print("\n  DIFERENCIA (filtered - original):")
    for split_name in ["val", "test"]:
        orig_row = df_results[(df_results["source"] == "original") & (df_results["split"] == split_name)]
        filt_row = df_results[(df_results["source"] == "filtered") & (df_results["split"] == split_name)]
        if len(orig_row) and len(filt_row):
            print(f"\n  {split_name}:")
            for metric in ["macro_f1", "accuracy", "adjacent_accuracy",
                           "f1_sin_desgaste", "f1_med_desgastado", "f1_desgastado"]:
                diff = filt_row[metric].values[0] - orig_row[metric].values[0]
                sign = "+" if diff >= 0 else ""
                print(f"    {metric}: {sign}{diff:.4f}")

    # Save comprehensive report
    report = {
        "timestamp": ts,
        "features_used": TOP7,
        "n_features": len(TOP7),
        "noise_profile": "combinado",
        "noisereduce_params": {"stationary": True, "prop_decrease": 0.85, "n_fft": 2048},
        "n_samples": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
        "results": results,
    }
    report_path = OUTDIR / f"comparison_report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos guardados en: {OUTDIR}")
    print(f"  {results_path.name}")
    print(f"  {report_path.name}")
    print(f"  models_original/  models_filtered/")


if __name__ == "__main__":
    main()
