#!/usr/bin/env python3
"""
regenerate_confusion_fixed.py

Versión robusta para regenerar matriz de confusión y reporte a partir de
holdout_predictions.csv u otros artefactos. Detecta automáticamente
nombres de columna comunes (incluye 'TRUE' en mayúsculas).

Salida (por defecto en run_dir/plots):
 - confusion_matrix_holdout.csv
 - confusion_matrix_holdout.png
 - confusion_matrix_holdout_normalized.png
 - classification_report_holdout.csv

Uso:
 python d:/pipeline_SVM/scripts/regenerate_confusion_fixed.py --run-dir "D:/pipeline_SVM/results/svm_final_fast"

Opcional:
 --outdir "D:/pipeline_SVM/results/svm_final_fast/plots"
 --model ... --data ...  (para recomputar predicciones si no hay holdout_predictions.csv)
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

sns.set(style="whitegrid")

# posibles nombres de columnas que indican etiqueta verdadera / predicha
POSSIBLE_TRUE = ['label_fixed','label','true_label','y_true','target','true','TRUE','actual','REAL','real']
POSSIBLE_PRED = ['pred','y_pred','prediction','prediction_label','predicted','yhat','y_hat','predicción','prediccion']

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_column_ignorecase(cols, candidates):
    # devuelve primer candidate en cols (ignorando mayúsc/minúsc y stripping) o None
    cols_norm = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    # además, pruebo si alguna columna contiene "true" o "pred" como substring
    for cand in candidates:
        for col in cols:
            if cand.strip().lower() in col.strip().lower():
                return col
    return None

def load_holdout_predictions(path: Path):
    df = pd.read_csv(path, low_memory=False)
    # si el CSV contiene una columna "index" que replica el índice, no es problema
    return df

def compute_and_save(y_true, y_pred, outdir: Path):
    # limpiar strings
    y_true = pd.Series(y_true).astype(str).str.strip()
    y_pred = pd.Series(y_pred).astype(str).str.strip()

    # etiquetas en orden de aparición en y_true
    labels = list(pd.Series(y_true).unique())

    # métricas
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Accuracy holdout: {acc:.4f}   f1_macro: {f1m:.4f}")

    # classification report
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    rep_csv = outdir / "classification_report_holdout.csv"
    rep_df.to_csv(rep_csv)
    print(f"Saved classification report -> {rep_csv}")

    # confusion matrix counts
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv = outdir / "confusion_matrix_holdout.csv"
    cm_df.to_csv(cm_csv)
    print(f"Saved confusion matrix CSV -> {cm_csv}")

    # plot absolute counts
    png = outdir / "confusion_matrix_holdout.png"
    plt.figure(figsize=(7,6))
    ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    plt.title('Matriz de confusión (conteos)')
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.close()
    print(f"Saved confusion matrix PNG -> {png}")

    # normalized per true class (row)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # si hay filas con sum 0 (no ejemplos), evita división por 0
        cm_norm = np.nan_to_num(cm_norm)
    cm_norm_df = pd.DataFrame(np.round(cm_norm, 3), index=labels, columns=labels)
    png_norm = outdir / "confusion_matrix_holdout_normalized.png"
    plt.figure(figsize=(7,6))
    ax = sns.heatmap(cm_norm_df, annot=True, fmt='.3f', cmap='Blues', cbar=True)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    plt.title('Matriz de confusión (normalizada por fila)')
    plt.tight_layout()
    plt.savefig(png_norm, dpi=200)
    plt.close()
    print(f"Saved normalized confusion matrix PNG -> {png_norm}")

def try_recompute_from_model(model_fp: Path, data_fp: Path, label_col: str):
    if not model_fp.exists() or not data_fp.exists():
        return None
    model = joblib.load(model_fp)
    df = pd.read_csv(data_fp, low_memory=False)
    # detect numeric features automatically (same as used en pipeline)
    X = df.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        print("No se encontraron columnas numéricas en el CSV indicado para recomputar.")
        return None
    try:
        y_pred = model.predict(X)
    except Exception as e:
        print("No se pudo predecir con el modelo:", e)
        return None
    y_true = df[label_col].astype(str) if label_col in df.columns else None
    return y_true, y_pred

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directorio del run (p.ej. D:/pipeline_SVM/results/svm_final_fast)")
    p.add_argument("--outdir", default=None, help="Directorio donde guardar figuras (por defecto run-dir/plots)")
    p.add_argument("--model", default=None, help="(Opcional) modelo joblib para recomputar predicciones")
    p.add_argument("--data", default=None, help="(Opcional) CSV con features/labels para recomputar predicciones (si se usa --model)")
    p.add_argument("--label-col", default="label_fixed", help="Nombre de la columna de etiqueta si se recomputa desde --data")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("Run dir no existe:", run_dir); sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else run_dir / "plots"
    safe_mkdir(outdir)

    # 1) intentar abrir holdout_predictions.csv
    hold_fp = run_dir / "holdout_predictions.csv"
    if hold_fp.exists():
        df = load_holdout_predictions(hold_fp)
        cols = list(df.columns)
        true_col = find_column_ignorecase(cols, POSSIBLE_TRUE)
        pred_col = find_column_ignorecase(cols, POSSIBLE_PRED)

        # también considerar columnas llamadas exactamente 'TRUE'/'Pred'
        if true_col is None:
            # buscar columna que contenga exactamente 'true' en cualquier mayúsc/min
            for c in cols:
                if c.strip().lower() == 'true' or c.strip() == 'TRUE':
                    true_col = c; break
        if pred_col is None:
            for c in cols:
                if c.strip().lower() == 'pred' or c.strip().lower() == 'prediction':
                    pred_col = c; break

        # si hallamos columna de probas (prob_*) calcular argmax
        prob_cols = [c for c in cols if c.startswith('prob_') or c.startswith('proba_')]
        if pred_col is None and prob_cols:
            probs = df[prob_cols].values
            classes = [c.replace('prob_','').replace('proba_','') for c in prob_cols]
            idx = np.nanargmax(probs, axis=1)
            y_pred = pd.Series([classes[i] for i in idx]).astype(str)
        elif pred_col is not None:
            y_pred = df[pred_col].astype(str)
        else:
            y_pred = None

        y_true = df[true_col].astype(str) if true_col is not None else None

        if y_true is not None and y_pred is not None:
            print(f"Usando holdout_predictions.csv; columnas detectadas -> true: {true_col}, pred: {pred_col if pred_col else 'from prob_*'}")
            compute_and_save(y_true.values, y_pred.values, outdir)
            return
        else:
            print("holdout_predictions.csv detectado pero no se encontró columna real y/o pred. Buscando otras fuentes...")

    # 2) intentar usar confusion_matrix_holdout.csv (regenerar imágenes desde el CSV)
    cm_csv_candidates = [run_dir / "confusion_matrix_holdout.csv", run_dir / "confusion_matrix.csv"]
    for cfp in cm_csv_candidates:
        if cfp.exists():
            cm_df = pd.read_csv(cfp, index_col=0)
            print(f"Found confusion CSV: {cfp}, regenerating PNGs...")
            # guardar copia en outdir
            (outdir / "confusion_matrix_holdout.csv").write_text(cm_df.to_csv())
            # plot absolute and normalized
            plt.figure(figsize=(7,6))
            ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
            plt.title('Matriz de confusión (conteos)')
            plt.tight_layout()
            png = outdir / "confusion_matrix_holdout.png"
            plt.savefig(png, dpi=200); plt.close()
            counts = cm_df.values.astype(float)
            with np.errstate(all='ignore'):
                counts_norm = counts / counts.sum(axis=1)[:, np.newaxis]
                counts_norm = np.nan_to_num(counts_norm)
            cm_norm_df = pd.DataFrame(np.round(counts_norm,3), index=cm_df.index, columns=cm_df.columns)
            png_norm = outdir / "confusion_matrix_holdout_normalized.png"
            plt.figure(figsize=(7,6))
            ax = sns.heatmap(cm_norm_df, annot=True, fmt='.3f', cmap='Blues')
            ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
            plt.title('Matriz de confusión (normalizada por fila)')
            plt.tight_layout()
            plt.savefig(png_norm, dpi=200); plt.close()
            print("Saved PNGs to", outdir)
            return

    # 3) intentar recomputar desde model + data
    if args.model and args.data:
        recom = try_recompute_from_model(Path(args.model), Path(args.data), args.label_col)
        if recom is not None:
            y_true, y_pred = recom
            compute_and_save(y_true.values, y_pred, outdir)
            return

    print("No se pudieron obtener las etiquetas verdaderas y/o predichas. Comprueba que 'holdout_predictions.csv' contiene una columna de etiqueta real (ej. 'TRUE') y una de predicción ('pred').")

if __name__ == "__main__":
    main()
