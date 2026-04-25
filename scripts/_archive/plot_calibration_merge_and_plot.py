#!/usr/bin/env python3
"""
plot_calibration_merge_and_plot.py

Carga probabilidades desde calibration_probs.csv (o holdout_predictions.csv),
encuentra/recupera la etiqueta verdadera (buscando en varios ficheros del run),
fusiona si hace falta y genera:

 - calibration_curve.png
 - calibration_histograms.png
 - calibration_per_class.csv
 - brier_scores.csv

Uso:
 python d:/pipeline_SVM/scripts/plot_calibration_merge_and_plot.py --run-dir "D:/pipeline_SVM/results/svm_final_fast"

Opciones:
 --run-dir   (obligatorio) carpeta del run
 --probs     (opcional) archivo CSV con probabilidades (default: calibration_probs.csv)
 --preds     (opcional) archivo CSV con predicciones/labels (default: holdout_predictions.csv)
 --true-col  (opcional) nombre exacto de la columna de etiqueta si está en el CSV de probs
 --merge-on  (opcional) columnas para hacer merge (coma-sep), ej: "index,filepath,basename"
 --n-bins    (opcional) número de bins para calibración (defecto 10)
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

sns.set(style="whitegrid")

CANDIDATE_TRUE_COLS = ['label_fixed','label','true','TRUE','true_label','y_true','target','y_hold','TRUE_LABEL']
POSSIBLE_PROB_PREFIXES = ('prob_','proba_')

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_true_col_in_df(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in CANDIDATE_TRUE_COLS:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fallback: any column containing 'true' or 'label'
    for c in df.columns:
        lc = c.lower()
        if 'true' in lc or 'label' in lc or 'target' in lc:
            return c
    return None

def find_prob_cols(df):
    return [c for c in df.columns if c.startswith(POSSIBLE_PROB_PREFIXES)]

def try_load_csv(path):
    if not path.exists(): 
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print("Error leyendo", path, e)
        return None

def align_and_find_true(probs_df, run_dir: Path, preds_fp: Path=None, merge_on=None, true_col_arg=None):
    """
    Intenta encontrar y_true en probs_df, si no, busca en preds_fp y en otros ficheros en run_dir.
    Si se encuentra en otro CSV, intenta hacer merge por merge_on columnas si se pasan.
    Si no hay merge_on y el número de filas coincide, puede alinear por orden (con advertencia).
    Devuelve (df_merged, true_col_name) o (None, None) si falla.
    """
    # 1) buscar true en probs_df
    if true_col_arg:
        if true_col_arg in probs_df.columns:
            return probs_df, true_col_arg
        else:
            print(f"true_col especificado '{true_col_arg}' no existe en {probs_df.shape[1]} cols.")
    tc = find_true_col_in_df(probs_df)
    if tc:
        print("Etiqueta verdadera encontrada en archivo de probabilidades:", tc)
        return probs_df, tc

    # 2) intentar cargar preds_fp
    if preds_fp is None:
        preds_candidates = [
            run_dir / "holdout_predictions.csv",
            run_dir / "holdout_predictions.csv",
            run_dir / "calibration_probs.csv",  # already loaded but keep fallback
            run_dir / "holdout_classification_report.csv",
            run_dir / "holdout_classification_report.json",
            run_dir / "y_hold.csv"
        ]
    else:
        preds_candidates = [Path(preds_fp)]
    preds_df = None
    for c in preds_candidates:
        df = try_load_csv(c)
        if df is not None:
            preds_df = df
            preds_path = c
            print("Se cargó posibles etiquetas desde:", c)
            break

    if preds_df is None:
        print("No se encontró archivo de predicciones/labels en el run dir.")
        return None, None

    # buscar true col en preds_df
    tc2 = find_true_col_in_df(preds_df)
    if tc2:
        print("Etiqueta verdadera encontrada en archivo de predicciones:", tc2)
    else:
        # try find common label-like col names
        # if none, we cannot proceed
        print("No se encontró columna de etiqueta verdadera en el candidato de predicciones. Columnas disponibles:", preds_df.columns.tolist())
        return None, None

    # 3) intentar merge: buscar columnas en común
    common_cols = list(set(probs_df.columns).intersection(set(preds_df.columns)))
    if merge_on:
        wanted = [c.strip() for c in merge_on.split(",") if c.strip()!='']
        available = [c for c in wanted if c in probs_df.columns and c in preds_df.columns]
        if available:
            print("Haciendo merge por columnas indicadas disponibles:", available)
            merged = probs_df.merge(preds_df[[*available, tc2]], on=available, how='left', suffixes=('', '_from_preds'))
            # si col true quedó NaN, warn
            if merged[tc2].isna().all():
                print("Merge realizado pero la columna de etiqueta está vacía después del merge.")
            return merged, tc2
        else:
            print("Merge-on especificado pero no hay columnas comunes solicitadas en ambos CSVs.")
    # si no hay merge_on, intentar merge por index si ambos tienen columna 'index'
    if 'index' in probs_df.columns and 'index' in preds_df.columns:
        print("Haciendo merge por 'index' columna.")
        merged = probs_df.merge(preds_df[['index', tc2]], on='index', how='left')
        return merged, tc2
    # si no, intentar merge por 'filepath' o 'basename'
    for key in ('filepath','filepath_from_csv','basename','file','path'):
        if key in probs_df.columns and key in preds_df.columns:
            print(f"Haciendo merge por '{key}'.")
            merged = probs_df.merge(preds_df[[key, tc2]], on=key, how='left')
            return merged, tc2
    # por último, si tienen el mismo número de filas, alinear por orden (con advertencia)
    if len(probs_df) == len(preds_df):
        print("Los CSV tienen el mismo número de filas; alineando por orden de fila (WARNING: asegúrate de que el orden es el mismo).")
        merged = probs_df.copy()
        merged[tc2] = preds_df[tc2].values
        return merged, tc2

    print("No fue posible alinear probabilidades y etiquetas: ni merge por columnas comunes ni igualdad de filas.")
    return None, None

def compute_and_plot_calibration(df, true_col, outdir: Path, n_bins=10):
    prob_cols = [c for c in df.columns if c.startswith(POSSIBLE_PROB_PREFIXES)]
    if not prob_cols:
        print("No se detectaron columnas prob_* en el dataframe.")
        return False
    classes = [c.replace("prob_","").replace("proba_","") for c in prob_cols]
    print("Clases detectadas:", classes)
    y_true = df[true_col].astype(str).reset_index(drop=True)
    probs = df[prob_cols].values
    if len(y_true) != probs.shape[0]:
        mn = min(len(y_true), probs.shape[0])
        print("Longitudes distintas; recortando al tamaño mínimo:", mn)
        y_true = y_true.iloc[:mn]
        probs = probs[:mn,:]
    # binarize
    try:
        y_bin = label_binarize(y_true, classes=classes)
    except Exception as e:
        print("label_binarize falló, revisa que las clases en prob_... coincidan con las etiquetas reales. Error:", e)
        return False

    # plot and save
    outdir.mkdir(parents=True, exist_ok=True)
    calib_rows = []
    brier_rows = []
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    for i, cls in enumerate(classes):
        prob_i = probs[:, i].astype(float)
        true_i = y_bin[:, i].astype(int)
        # clip and sanity
        prob_i = np.clip(prob_i, 0.0, 1.0)
        # compute calibration_curve
        frac_pos, mean_pred = calibration_curve(true_i, prob_i, n_bins=n_bins, strategy='uniform')
        brier = brier_score_loss(true_i, prob_i)
        brier_rows.append({'class': cls, 'brier_score': float(brier)})
        for mv, fv in zip(mean_pred, frac_pos):
            calib_rows.append({'class': cls, 'mean_predicted_value': float(mv), 'fraction_of_positives': float(fv)})
        ax.plot(mean_pred, frac_pos, marker='o', label=f"{cls} (Brier={brier:.3f})")
    ax.plot([0,1],[0,1],"k:", label="perfectly calibrated")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fracción de positivos")
    ax.set_title("Curva de calibración (por clase)")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    png = outdir / "calibration_curve.png"
    plt.savefig(png, dpi=200)
    plt.close()
    print("Guardado:", png)

    # histogramas
    plt.figure(figsize=(max(6,2*len(classes)), 3))
    for i, cls in enumerate(classes):
        plt.subplot(1, len(classes), i+1)
        plt.hist(probs[:,i], bins=10, range=(0,1))
        plt.title(cls); plt.xlabel('pred prob'); plt.ylabel('count')
    plt.tight_layout()
    png_hist = outdir / "calibration_histograms.png"
    plt.savefig(png_hist, dpi=200)
    plt.close()
    print("Guardado:", png_hist)

    # guardar CSVs
    pd.DataFrame(calib_rows).to_csv(outdir / "calibration_per_class.csv", index=False)
    pd.DataFrame(brier_rows).to_csv(outdir / "brier_scores.csv", index=False)
    print("CSV guardados en:", outdir)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--probs", default=None, help="CSV con columnas prob_* (default: calibration_probs.csv)")
    p.add_argument("--preds", default=None, help="CSV con preds/labels si están en otro fichero (default: holdout_predictions.csv)")
    p.add_argument("--true-col", default=None, help="Nombre exacto de columna de etiqueta verdadera si se conoce")
    p.add_argument("--merge-on", default=None, help="Columnas para merge separadas por coma, ej: index,filepath")
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    outdir = run_dir / "plots"
    safe_mkdir(outdir)

    probs_fp = Path(args.probs) if args.probs else run_dir / "calibration_probs.csv"
    if not probs_fp.exists():
        # fallback to holdout_predictions.csv
        fallback = run_dir / "holdout_predictions.csv"
        if fallback.exists():
            probs_fp = fallback
        else:
            print("No se encontró calibration_probs.csv ni holdout_predictions.csv en", run_dir)
            sys.exit(1)

    probs_df = try_load_csv(probs_fp)
    if probs_df is None:
        print("No se pudo leer", probs_fp); sys.exit(1)

    merged_df, true_col = align_and_find_true(probs_df, run_dir, preds_fp=args.preds, merge_on=args.merge_on, true_col_arg=args.true_col)
    if merged_df is None or true_col is None:
        print("No fue posible obtener etiquetas verdaderas. Revisa los archivos y usa --true-col o --preds para indicar el CSV con etiquetas.")
        sys.exit(2)

    ok = compute_and_plot_calibration(merged_df, true_col, outdir, n_bins=args.n_bins)
    if not ok:
        print("La generación de la curva falló; revisa concordancia entre nombres de clases y columnas prob_.")
        sys.exit(3)
    print("Hecho. Revisa", outdir)

if __name__ == "__main__":
    main()
