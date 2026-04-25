#!/usr/bin/env python3
"""
generate_calibration_curve.py

Genera curvas de calibración (reliability diagrams) para clasificación multi-clase.
Soporta:
 - leer probabilidades desde holdout_predictions.csv (columnas prob_<class>)
 - o calcular probabilidades con model.predict_proba() o softmax(model.decision_function(X_hold))
 - trazar una curva por clase (one-vs-rest), calcular Brier score por clase.

Uso ejemplo:
python d:/pipeline_SVM/scripts/generate_calibration_curve.py --run-dir "D:/pipeline_SVM/results/svm_final_fast" --model "D:/pipeline_SVM/results/svm_final_fast/best_pipeline.joblib" --data "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv" --label-col "label_fixed"

Argumentos principales:
 --run-dir: directorio del run (se busca holdout_predictions.csv, y_hold.csv, X_hold.csv aquí)
 --outdir: (opcional) donde guardar PNG/CSV, por defecto run-dir/plots
 --model, --data: (opcionales) para recomputar probabilidades si no se encuentran en holdout_predictions.csv
 --label-col: nombre de columna de label en --data (por defecto 'label_fixed')
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils.multiclass import unique_labels
from math import isnan

sns.set(style="whitegrid")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_holdout_predictions(run_dir: Path):
    fp = run_dir / "holdout_predictions.csv"
    if fp.exists():
        df = pd.read_csv(fp, low_memory=False)
        return df, fp
    return None, None

def detect_prob_cols(df):
    prob_cols = [c for c in df.columns if c.startswith("prob_") or c.startswith("proba_")]
    return prob_cols

def softmax(a):
    # a: (n_samples, n_classes)
    a = np.array(a, dtype=float)
    if a.ndim == 1:
        # binary decision function -> map through sigmoid
        from scipy.special import expit
        return np.vstack([1 - expit(a), expit(a)]).T
    # subtract max for numerical stability
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=1, keepdims=True)

def compute_probs_from_model(model_fp: Path, data_fp: Path, label_col: str):
    if not model_fp.exists():
        print("Modelo no encontrado:", model_fp); return None
    if not data_fp.exists():
        print("CSV de datos no encontrado:", data_fp); return None
    model = joblib.load(model_fp)
    df = pd.read_csv(data_fp, low_memory=False)
    # try to select holdout rows if present (y_hold/index), else use full CSV
    # We'll use any rows that appear in X_hold if available in run dir; caller may provide X_hold instead
    X = df.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        print("No se detectaron columnas numéricas en el CSV para predecir.")
        return None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            classes = list(model.classes_) if hasattr(model, "classes_") else None
            return probs, classes, df
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(X)
            probs = softmax(dec)
            classes = list(model.classes_) if hasattr(model, "classes_") else None
            return probs, classes, df
        else:
            print("El modelo no tiene predict_proba ni decision_function.")
            return None
    except Exception as e:
        print("Error al predecir con el modelo:", e)
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directorio del run (ej. D:/pipeline_SVM/results/svm_final_fast)")
    p.add_argument("--outdir", default=None, help="Directorio salida (por defecto run-dir/plots)")
    p.add_argument("--model", default=None, help="(Opcional) modelo joblib para recomputar probabilidades")
    p.add_argument("--data", default=None, help="(Opcional) CSV con features para recomputar probabilidades")
    p.add_argument("--label-col", default="label_fixed", help="Nombre de columna etiqueta en --data")
    p.add_argument("--n-bins", type=int, default=10, help="Número de bins para calibración")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("Run dir no existe:", run_dir); sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else run_dir / "plots"
    safe_mkdir(outdir)

    # 1) intentar leer holdout_predictions.csv con prob_* cols
    df_pred, pred_fp = load_holdout_predictions(run_dir)
    y_true = None
    probs = None
    classes = None
    df_source = None

    if df_pred is not None:
        prob_cols = detect_prob_cols(df_pred)
        # detect true label col
        possible_true = ['label_fixed','label','true','TRUE','true_label','y_true','target']
        true_col = None
        for c in possible_true:
            if c in df_pred.columns:
                true_col = c; break
        if true_col is None:
            # fallback: find column name containing 'true' or 'label'
            for c in df_pred.columns:
                if 'true' in c.lower() or 'label' in c.lower():
                    true_col = c; break
        if true_col is not None:
            y_true = df_pred[true_col].astype(str).reset_index(drop=True)
        if prob_cols:
            probs = df_pred[prob_cols].values
            # derive class names from column names: prob_<class>
            classes = [c.replace("prob_","").replace("proba_","") for c in prob_cols]
            df_source = df_pred
            print(f"Usando probabilidades desde {pred_fp} columnas: {prob_cols}")
        else:
            print("holdout_predictions.csv encontrado pero no contiene columnas prob_*. Intentaré recomputar con --model/--data si se proveen.")

    # 2) si no tenemos probs, intentar recomputar desde model+data
    if probs is None and args.model and args.data:
        recom = compute_probs_from_model(Path(args.model), Path(args.data), args.label_col)
        if recom:
            probs, classes, df_source = recom
            # attempt to get labels if present
            if args.label_col in df_source.columns:
                y_true = df_source[args.label_col].astype(str).reset_index(drop=True)
            print("Probabilidades recomputadas desde modelo+data.")

    # 3) intentar leer y_hold.csv y X_hold.csv en run_dir (y recalcular probs con model si se dio)
    if probs is None:
        y_hold_fp = run_dir / "y_hold.csv"
        X_hold_fp = run_dir / "X_hold.csv"
        if y_hold_fp.exists() and X_hold_fp.exists() and args.model:
            df_y = pd.read_csv(y_hold_fp, index_col=0)
            df_X = pd.read_csv(X_hold_fp, index_col=0)
            y_true = df_y.iloc[:,0].astype(str).reset_index(drop=True)
            model = joblib.load(args.model)
            X = df_X.select_dtypes(include=[np.number]).copy()
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    classes = list(model.classes_) if hasattr(model, "classes_") else None
                    df_source = pd.concat([df_X.reset_index(drop=True), df_y.reset_index(drop=True)], axis=1)
                    print("Usando X_hold/y_hold y predict_proba del modelo.")
                elif hasattr(model, "decision_function"):
                    dec = model.decision_function(X)
                    probs = softmax(dec)
                    classes = list(model.classes_) if hasattr(model, "classes_") else None
                    df_source = pd.concat([df_X.reset_index(drop=True), df_y.reset_index(drop=True)], axis=1)
                    print("Usando X_hold/y_hold y decision_function->softmax.")
            except Exception as e:
                print("No se pudieron obtener probabilidades desde el modelo con X_hold:", e)

    if probs is None:
        print("No se pudieron obtener probabilidades. Opciones:")
        print("- Proveer holdout_predictions.csv con columnas prob_<class>.")
        print("- Proveer --model y --data (o --model y archivos y_hold/X_hold) para recomputar probabilidades.")
        print("- Reentrenar el SVC con probability=True o calibrar con CalibratedClassifierCV usando un conjunto de calibración separado.")
        sys.exit(2)

    # asegurarnos de que probs tenga shape (n_samples, n_classes)
    probs = np.array(probs)
    n_samples = probs.shape[0]
    if probs.ndim == 1:
        probs = probs.reshape(-1,1)

    # determine classes
    if classes is None:
        # inferir classes si y_true existe
        if y_true is not None:
            classes = list(pd.Series(y_true).unique())
        else:
            # fallback index-based classes
            n_classes = probs.shape[1]
            classes = [f"class_{i}" for i in range(n_classes)]
    classes = [str(c) for c in classes]

    # ensure ordering: if y_true exists, set classes sorted by appearance in y_true
    if y_true is not None:
        labels_order = list(pd.Series(y_true).unique())
        # reorder columns of probs to match classes if possible
        # if classes match labels_order (set equality), reorder
        if set(classes) == set(labels_order):
            # compute index mapping
            idx_map = [classes.index(l) for l in labels_order]
            probs = probs[:, idx_map]
            classes = labels_order
        else:
            # keep classes as-is; but ensure length match
            if probs.shape[1] != len(classes):
                print("Número de columnas de prob* no coincide con número de clases inferidas. Verifique archivos.")
                # continue with columns as is

    # binarize y_true for per-class calibration
    if y_true is None:
        print("No se encontró etiqueta verdadera (y_true). No puedo trazar curva de calibración sin ground truth.")
        sys.exit(3)

    y_true = pd.Series(y_true).astype(str).reset_index(drop=True)
    # create binary array per class
    y_bin = label_binarize(y_true, classes=classes)
    # if label_binarize returns shape (n_samples, 1) for binary, handle
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1,1)

    # for each class compute calibration_curve
    n_bins = args.n_bins
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    calib_rows = []
    brier_rows = []
    for i, cls in enumerate(classes):
        prob_i = probs[:, i]
        true_i = y_bin[:, i]
        # compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(true_i, prob_i, n_bins=n_bins, strategy='uniform')
        # brier
        try:
            brier = brier_score_loss(true_i, prob_i)
        except Exception:
            brier = float('nan')
        brier_rows.append({"class": cls, "brier_score": brier})
        # save per-class calibration points
        for mv, fv in zip(mean_predicted_value, fraction_of_positives):
            calib_rows.append({"class": cls, "mean_predicted_value": float(mv), "fraction_of_positives": float(fv)})
        # plot
        ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f"{cls} (Brier={brier:.3f})")

    # plot baseline
    ax.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fracción de positivos")
    ax.set_title("Curva de calibración (por clase)")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    png_fp = outdir / "calibration_curve.png"
    plt.savefig(png_fp, dpi=200)
    plt.close()
    print("Guardado:", png_fp)

    # save CSVs
    calib_df = pd.DataFrame(calib_rows)
    calib_csv = outdir / "calibration_per_class.csv"
    calib_df.to_csv(calib_csv, index=False)
    print("Saved calibration points CSV ->", calib_csv)

    brier_df = pd.DataFrame(brier_rows)
    brier_csv = outdir / "brier_scores.csv"
    brier_df.to_csv(brier_csv, index=False)
    print("Saved Brier scores CSV ->", brier_csv)

    print("Hecho. Revisa:", png_fp, calib_csv, brier_csv)

if __name__ == "__main__":
    main()
