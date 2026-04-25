#!/usr/bin/env python3
"""
train_svm_ordinal_v2.py
========================
SVM ordinal (Frank & Hall) con dataset expandido (7 experimentos).

DIFERENCIAS vs train_svm_ordinal.py (sesion 4):
  - Usa splits pre-definidos del CSV (NO GroupShuffleSplit)
  - Agrupa por experiment para StratifiedGroupKFold
  - Evalua en val Y test por separado
  - Compara con baseline de sesion 4
  - Reporta metricas per-class + confusion matrix

Uso:
  python train_svm_ordinal_v2.py
  python train_svm_ordinal_v2.py --input D:/pipeline_SVM/features/features_curated_splits.csv
"""

import sys, os
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")

import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL, N_ORDINAL,
    ordinal_encode, ordinal_decode, ordinal_proba,
    print_ordinal_report, ordinal_mae, adjacent_accuracy,
    ordinal_confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Configuracion ─────────────────────────────────────────────────────

RANDOM_STATE = 42
CV_SPLITS = 5

# Baseline de sesion 4 (para comparacion)
BASELINE_METRICS = {
    "val":  {"macro_f1": 0.654, "exact_acc": 0.930, "adj_acc": 1.0, "ordinal_mae": 0.070},
    "test": {"macro_f1": 0.558, "exact_acc": 0.940, "adj_acc": 1.0, "ordinal_mae": 0.060},
}
BASELINE_PER_CLASS_TEST = {
    "sin_desgaste": {"f1": 0.304, "precision": 1.0, "recall": 0.179},
    "medianamente_desgastado": {"f1": 0.969, "precision": 0.939, "recall": 1.0},
    "desgastado": {"f1": 0.400, "precision": 1.0, "recall": 0.250},
}

# Columnas metadata que NO son features
META_COLS = {
    'filepath', 'label', 'split', 'aug_type', 'mic_type',
    'drill_group', 'basename', 'experiment'
}


# ── Helpers ───────────────────────────────────────────────────────────

def build_binary_target(y_int, threshold):
    """Crea target binario: 1 si y >= threshold, 0 si no."""
    return (np.asarray(y_int) >= threshold).astype(int)


def predict_ordinal(clfs, X):
    """Predice clases ordinales con restriccion de monotonicidad."""
    p1 = clfs[0].predict_proba(X)[:, 1]  # P(y >= 1)
    p2 = clfs[1].predict_proba(X)[:, 1]  # P(y >= 2)
    p2 = np.minimum(p2, p1)  # monotonicidad
    probs = np.stack([p1, p2], axis=1)
    y_pred = ordinal_decode(probs)
    return y_pred, probs, p1, p2


def compute_metrics(y_true_int, y_pred_int, y_true_str, y_pred_str):
    """Calcula todas las metricas ordinales + per-class."""
    metrics = {
        "exact_accuracy": float(accuracy_score(y_true_int, y_pred_int)),
        "macro_f1": float(f1_score(y_true_int, y_pred_int, average='macro', zero_division=0)),
        "ordinal_mae": float(ordinal_mae(y_true_int, y_pred_int)),
        "adjacent_accuracy": float(adjacent_accuracy(y_true_int, y_pred_int)),
    }

    # Per-class
    report = classification_report(
        y_true_int, y_pred_int,
        labels=[0, 1, 2],
        target_names=["sin_desgaste", "med_desgastado", "desgastado"],
        output_dict=True, zero_division=0
    )
    metrics["per_class"] = report

    # Errores de 2 pasos
    cm = ordinal_confusion_matrix(y_true_int, y_pred_int)
    metrics["two_step_errors"] = int(cm[0, 2] + cm[2, 0])
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compare_with_baseline(metrics, split_name):
    """Compara con baseline sesion 4 y retorna delta string."""
    if split_name not in BASELINE_METRICS:
        return ""

    base = BASELINE_METRICS[split_name]
    lines = [f"\n  --- Comparacion vs baseline sesion 4 ({split_name}) ---"]
    for key in ["macro_f1", "exact_acc", "adj_acc", "ordinal_mae"]:
        # Map key names
        metric_key = {
            "macro_f1": "macro_f1",
            "exact_acc": "exact_accuracy",
            "adj_acc": "adjacent_accuracy",
            "ordinal_mae": "ordinal_mae"
        }[key]

        old = base[key]
        new = metrics.get(metric_key, 0)
        delta = new - old
        direction = "+" if delta >= 0 else ""
        better = "MEJOR" if (delta > 0 and key != "ordinal_mae") or \
                            (delta < 0 and key == "ordinal_mae") else \
                 ("PEOR" if delta != 0 else "IGUAL")
        lines.append(f"  {key:15s}: {old:.4f} -> {new:.4f} ({direction}{delta:.4f}) [{better}]")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SVM ordinal v2 (dataset expandido)")
    parser.add_argument("--input", default="D:/pipeline_SVM/features/features_curated_splits.csv")
    parser.add_argument("--outdir", default="D:/pipeline_SVM/results/svm_ordinal_expanded")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SVM ORDINAL v2 — DATASET EXPANDIDO (7 experimentos)")
    print("  Frank & Hall decomposition + StratifiedGroupKFold")
    print("=" * 70)

    # ── 1. Cargar features ─────────────────────────────────────────────
    print(f"\n[1/6] Cargando features desde {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Total: {len(df)} filas x {len(df.columns)} cols")

    # Verificar columnas obligatorias
    for col in ['split', 'label', 'experiment']:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada. Columnas: {list(df.columns)}")

    # Separar splits
    df_train = df[df['split'] == 'train'].copy().reset_index(drop=True)
    df_val   = df[df['split'] == 'val'].copy().reset_index(drop=True)
    df_test  = df[df['split'] == 'test'].copy().reset_index(drop=True)

    print(f"  TRAIN: {len(df_train)} | VAL: {len(df_val)} | TEST: {len(df_test)}")

    # Verificar: E3 solo en test
    train_exps = set(df_train['experiment'].unique())
    test_exps  = set(df_test['experiment'].unique())
    assert 'E3' not in train_exps, "ERROR: E3 encontrado en train!"
    assert 'E3' in test_exps, "ERROR: E3 no encontrado en test!"
    print(f"  Train experiments: {sorted(train_exps)}")
    print(f"  Test experiments:  {sorted(test_exps)}")

    # Verificar: no augmentados en val/test
    if 'aug_type' in df_val.columns:
        val_aug = df_val[df_val['aug_type'] != 'original']
        test_aug = df_test[df_test['aug_type'] != 'original']
        assert len(val_aug) == 0, f"ERROR: {len(val_aug)} augmentados en val!"
        assert len(test_aug) == 0, f"ERROR: {len(test_aug)} augmentados en test!"
        print("  OK: No augmentados en val/test")

    # ── 2. Preparar features ───────────────────────────────────────────
    print("\n[2/6] Preparando features...")
    feature_cols = [c for c in df.columns if c not in META_COLS
                    and np.issubdtype(df[c].dtype, np.number)]
    print(f"  {len(feature_cols)} features numericas")

    X_train = df_train[feature_cols].values
    X_val   = df_val[feature_cols].values
    X_test  = df_test[feature_cols].values

    y_train_str = df_train['label'].values
    y_val_str   = df_val['label'].values
    y_test_str  = df_test['label'].values

    y_train_int = np.array([LABEL_TO_IDX[l] for l in y_train_str])
    y_val_int   = np.array([LABEL_TO_IDX[l] for l in y_val_str])
    y_test_int  = np.array([LABEL_TO_IDX[l] for l in y_test_str])

    # Grupos para CV: experiment
    groups_train = df_train['experiment'].values if 'experiment' in df_train.columns \
        else df_train.get('drill_group', pd.Series(range(len(df_train)))).values

    print(f"  Distribucion train: {dict(zip(*np.unique(y_train_int, return_counts=True)))}")
    print(f"  Distribucion val:   {dict(zip(*np.unique(y_val_int, return_counts=True)))}")
    print(f"  Distribucion test:  {dict(zip(*np.unique(y_test_int, return_counts=True)))}")
    print(f"  Grupos CV: {np.unique(groups_train).tolist()}")

    # ── 3. GridSearchCV para C1 ────────────────────────────────────────
    print("\n[3/6] GridSearchCV para C1: P(y >= 1) — hay desgaste?")

    y_c1_train = build_binary_target(y_train_int, threshold=1)
    n_pos, n_neg = y_c1_train.sum(), len(y_c1_train) - y_c1_train.sum()
    print(f"  Positivos (hay desgaste): {n_pos} | Negativos (sin desgaste): {n_neg}")

    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    param_grid = [
        {
            'select': [SelectKBest(mutual_info_classif)],
            'select__k': [10, 20],
            'svc__C': [1.0, 10.0, 50.0],
            'svc__kernel': ['rbf'],
        },
        {
            'select': ['passthrough'],
            'svc__C': [1.0, 10.0, 50.0],
            'svc__kernel': ['rbf', 'linear'],
        }
    ]

    pipe_c1 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(mutual_info_classif)),
        ('svc', SVC(probability=True, class_weight='balanced',
                     gamma='scale', random_state=RANDOM_STATE)),
    ])

    t0 = time.time()
    gs_c1 = GridSearchCV(
        pipe_c1, param_grid, cv=cv,
        scoring='f1', n_jobs=args.n_jobs,
        verbose=1, refit=True
    )
    gs_c1.fit(X_train, y_c1_train, groups=groups_train)
    t_c1 = time.time() - t0
    print(f"  Mejor params C1: {gs_c1.best_params_}")
    print(f"  Best CV F1: {gs_c1.best_score_:.4f} ({t_c1:.0f}s)")

    # ── 4. GridSearchCV para C2 ────────────────────────────────────────
    print("\n[4/6] GridSearchCV para C2: P(y >= 2) — desgaste severo?")

    y_c2_train = build_binary_target(y_train_int, threshold=2)
    n_pos, n_neg = y_c2_train.sum(), len(y_c2_train) - y_c2_train.sum()
    print(f"  Positivos (desgastado): {n_pos} | Negativos: {n_neg}")

    pipe_c2 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(mutual_info_classif)),
        ('svc', SVC(probability=True, class_weight='balanced',
                     gamma='scale', random_state=RANDOM_STATE)),
    ])

    t0 = time.time()
    gs_c2 = GridSearchCV(
        pipe_c2, param_grid, cv=cv,
        scoring='f1', n_jobs=args.n_jobs,
        verbose=1, refit=True
    )
    gs_c2.fit(X_train, y_c2_train, groups=groups_train)
    t_c2 = time.time() - t0
    print(f"  Mejor params C2: {gs_c2.best_params_}")
    print(f"  Best CV F1: {gs_c2.best_score_:.4f} ({t_c2:.0f}s)")

    clfs = [gs_c1.best_estimator_, gs_c2.best_estimator_]

    # ── 5. Evaluacion ──────────────────────────────────────────────────
    print("\n[5/6] Evaluacion en VAL y TEST...")

    results = {}
    for split_name, X_eval, y_eval_int, y_eval_str, df_eval in [
        ("val",  X_val,  y_val_int,  y_val_str,  df_val),
        ("test", X_test, y_test_int, y_test_str, df_test),
    ]:
        y_pred_int, probs, p1, p2 = predict_ordinal(clfs, X_eval)
        y_pred_str = np.array([IDX_TO_LABEL[i] for i in y_pred_int])

        metrics = compute_metrics(y_eval_int, y_pred_int, y_eval_str, y_pred_str)
        results[split_name] = metrics

        print(f"\n{'=' * 60}")
        print(f"  {split_name.upper()} SET ({len(X_eval)} muestras)")
        print(f"{'=' * 60}")
        print_ordinal_report(y_eval_int, y_pred_int)

        # Comparar con baseline
        comparison = compare_with_baseline(metrics, split_name)
        if comparison:
            print(comparison)

        # Per-class comparison for test
        if split_name == "test":
            print("\n  --- Per-class comparison vs baseline (TEST) ---")
            report = metrics["per_class"]
            for cls_name, base_vals in BASELINE_PER_CLASS_TEST.items():
                short = cls_name[:15]
                new_f1 = report.get(cls_name, report.get("med_desgastado", {})).get("f1-score", 0)
                delta = new_f1 - base_vals["f1"]
                direction = "+" if delta >= 0 else ""
                print(f"  {short:20s}: F1 {base_vals['f1']:.3f} -> {new_f1:.3f} ({direction}{delta:.3f})")

        # Guardar predicciones
        p_class = ordinal_proba(probs)
        pred_df = df_eval[['filepath', 'label', 'experiment']].copy() if 'experiment' in df_eval.columns \
            else df_eval[['filepath', 'label']].copy()
        pred_df['y_true'] = y_eval_str
        pred_df['y_pred'] = y_pred_str
        pred_df['p_c1'] = p1
        pred_df['p_c2'] = p2
        pred_df['prob_sin'] = p_class[:, 0]
        pred_df['prob_med'] = p_class[:, 1]
        pred_df['prob_des'] = p_class[:, 2]
        pred_df['ordinal_error'] = np.abs(y_eval_int - y_pred_int)
        pred_df.to_csv(OUTDIR / f"predictions_{split_name}.csv", index=False)

    # ── 6. Guardar artefactos ──────────────────────────────────────────
    print(f"\n[6/6] Guardando artefactos en {OUTDIR}")

    joblib.dump(clfs[0], OUTDIR / "svm_C1.joblib")
    joblib.dump(clfs[1], OUTDIR / "svm_C2.joblib")

    for split_name, metrics in results.items():
        # Clean metrics for JSON serialization
        clean = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        clean['confusion_matrix'] = metrics['confusion_matrix']
        (OUTDIR / f"metrics_{split_name}.json").write_text(
            json.dumps(clean, indent=2, ensure_ascii=False, default=str)
        )

    # Resumen de hiperparametros
    summary = {
        "timestamp": ts,
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "train_experiments": sorted(train_exps),
        "test_experiments": sorted(test_exps),
        "best_params_C1": {str(k): str(v) for k, v in gs_c1.best_params_.items()},
        "best_params_C2": {str(k): str(v) for k, v in gs_c2.best_params_.items()},
        "best_cv_f1_C1": float(gs_c1.best_score_),
        "best_cv_f1_C2": float(gs_c2.best_score_),
        "training_time_C1_s": t_c1,
        "training_time_C2_s": t_c2,
    }
    (OUTDIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── Resumen final ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL")
    print(f"{'=' * 70}")
    print(f"  Dataset: {len(df_train)} train + {len(df_val)} val + {len(df_test)} test")
    print(f"  Features: {len(feature_cols)}")
    print(f"  C1 best: {gs_c1.best_params_}")
    print(f"  C2 best: {gs_c2.best_params_}")
    print(f"\n  {'Metrica':<20s} {'VAL':>8s} {'TEST':>8s} {'Base_TEST':>10s} {'Delta':>8s}")
    print(f"  {'-'*56}")
    for key in ["exact_accuracy", "macro_f1", "adjacent_accuracy", "ordinal_mae"]:
        v = results["val"].get(key, 0)
        t = results["test"].get(key, 0)
        base_key = {"exact_accuracy": "exact_acc", "macro_f1": "macro_f1",
                     "adjacent_accuracy": "adj_acc", "ordinal_mae": "ordinal_mae"}[key]
        b = BASELINE_METRICS["test"].get(base_key, 0)
        d = t - b
        print(f"  {key:<20s} {v:>8.4f} {t:>8.4f} {b:>10.4f} {d:>+8.4f}")

    print(f"\n  Artefactos:")
    for f in sorted(OUTDIR.glob("*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
