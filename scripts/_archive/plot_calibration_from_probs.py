#!/usr/bin/env python3
"""
plot_calibration_from_probs.py

Genera curva(s) de calibración (reliability diagrams) y histogramas de probabilidad
a partir de un archivo que contenga probabilidades de clase (por ejemplo calibration_probs.csv
o holdout_predictions.csv con columnas prob_<class>).

Salida (por defecto run_dir/plots):
 - calibration_curve.png
 - calibration_histograms.png
 - calibration_per_class.csv
 - brier_scores.csv

Uso:
 python d:/pipeline_SVM/scripts/plot_calibration_from_probs.py --run-dir "D:/pipeline_SVM/results/svm_final_fast"

El script busca en orden:
 1) run_dir/calibration_probs.csv
 2) run_dir/holdout_predictions.csv
 3) run_dir/plots/holdout_predictions.csv (alternativa)

Si no encuentra probabilidades, sugiere opciones (reentrenar con probability=True o producir archivo calibration_probs.csv).
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

sns.set(style="whitegrid")

POSSIBLE_TRUE = ['label_fixed','label','true','TRUE','true_label','y_true','target','y_hold']

def find_true_column(cols):
    cols_l = [c.lower() for c in cols]
    for cand in POSSIBLE_TRUE:
        if cand.lower() in cols_l:
            # return original-case column name
            return cols[cols_l.index(cand.lower())]
    # fallback: look for a column name containing 'true' or 'label'
    for i,c in enumerate(cols):
        lc = c.lower()
        if 'true' in lc or 'label' in lc or 'target' in lc:
            return c
    return None

def find_prob_columns(cols):
    prob_cols = [c for c in cols if c.startswith('prob_') or c.startswith('proba_')]
    if prob_cols:
        return prob_cols
    # fallback: any column that looks like probability numbers between 0 and 1? (risky)
    # we avoid guessing too much; prefer explicit prob_.* columns
    return []

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_prob_table(run_dir: Path):
    candidates = [
        run_dir / "calibration_probs.csv",
        run_dir / "holdout_predictions.csv",
        run_dir / "plots/holdout_predictions.csv",
        run_dir / "plots/holdout_predictions.csv"
    ]
    for fp in candidates:
        if fp.exists():
            try:
                df = pd.read_csv(fp, low_memory=False)
                print(f"Usando: {fp}")
                return df, fp
            except Exception as e:
                print(f"Error leyendo {fp}: {e}")
    return None, None

def compute_and_plot(df, outdir: Path, n_bins=10):
    cols = list(df.columns)
    true_col = find_true_column(cols)
    prob_cols = find_prob_columns(cols)
    if true_col is None:
        print("No se encontró columna de etiqueta verdadera en el CSV. Búsquedas intentadas:", POSSIBLE_TRUE)
        # but if 'TRUE' exists in uppercase as in your sample, find_true_column should find it
    if not prob_cols:
        print("No se encontraron columnas con prefijo prob_ o proba_. Asegúrate de crear calibration_probs.csv con columnas prob_<class>.")
        return False

    # parse classes from prob column names
    classes = [c.replace("prob_","").replace("proba_","") for c in prob_cols]
    print("Clases detectadas (columnas prob_...):", classes)

    if true_col is None:
        print("No hay etiqueta verdadera; no se puede trazar la curva de calibración con ground truth.")
        return False

    y_true = df[true_col].astype(str).reset_index(drop=True)
    probs = df[prob_cols].values
    n_samples, n_classes = probs.shape
    # ensure shape matches
    if len(y_true) != n_samples:
        print("Advertencia: número de filas entre etiquetas y probabilidades no coincide; intentando alinear por índice.")
        min_n = min(len(y_true), n_samples)
        y_true = y_true.iloc[:min_n]
        probs = probs[:min_n,:]

    # if classes set differs from labels unique set, we will keep order prob_cols->classes
    # binarize
    classes = [str(c) for c in classes]
    try:
        y_bin = label_binarize(y_true, classes=classes)
    except Exception:
        # fallback: produce classes from unique labels in y_true
        labels_order = list(pd.Series(y_true).unique())
        print("Warning: las clases detectadas a partir de columnas prob_ no coinciden con las etiquetas reales. Usando orden de etiquetas en y_true:", labels_order)
        classes = labels_order
        y_bin = label_binarize(y_true, classes=classes)
        # reorder probs columns? assume prob columns are in same class order; otherwise user must ensure names

    # compute calibration curve per class
    calib_rows = []
    brier_rows = []
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    for i, cls in enumerate(classes):
        prob_i = probs[:, i]
        true_i = y_bin[:, i]
        if np.all(np.isnan(prob_i)):
            print(f"All NaNs for class {cls}, saltando.")
            continue
        # clip probabilities to [0,1]
        prob_i = np.clip(prob_i.astype(float), 0.0, 1.0)
        # calibration curve
        frac_pos, mean_pred = calibration_curve(true_i, prob_i, n_bins=n_bins, strategy='uniform')
        # brier
        try:
            brier = brier_score_loss(true_i, prob_i)
        except Exception:
            brier = float('nan')
        brier_rows.append({"class": cls, "brier_score": float(brier)})
        for mv, fv in zip(mean_pred, frac_pos):
            calib_rows.append({"class": cls, "mean_predicted_value": float(mv), "fraction_of_positives": float(fv)})
        ax.plot(mean_pred, frac_pos, marker='o', label=f"{cls} (Brier={brier:.3f})")

    ax.plot([0,1],[0,1],"k:", label="Perfectly calibrated")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fracción de positivos")
    ax.set_title("Curva de calibración (por clase)")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    png = outdir / "calibration_curve.png"
    plt.savefig(png, dpi=200)
    plt.close()
    print("Guardado:", png)

    # histogram of predicted probabilities per class
    plt.figure(figsize=(10, 4 + n_classes*0.2))
    for i, cls in enumerate(classes):
        plt.subplot(1, n_classes, i+1)
        prob_i = probs[:, i]
        plt.hist(prob_i, bins=10, range=(0,1))
        plt.title(cls)
        plt.xlabel("pred prob"); plt.ylabel("count")
    plt.tight_layout()
    png_hist = outdir / "calibration_histograms.png"
    plt.savefig(png_hist, dpi=200)
    plt.close()
    print("Guardado:", png_hist)

    # save CSVs
    pd.DataFrame(calib_rows).to_csv(outdir / "calibration_per_class.csv", index=False)
    pd.DataFrame(brier_rows).to_csv(outdir / "brier_scores.csv", index=False)
    print("CSV guardados: calibration_per_class.csv y brier_scores.csv")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("Run dir no existe:", run_dir); sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    df, fp = load_prob_table(run_dir)
    if df is None:
        print("No se encontraron archivos calibration_probs.csv o holdout_predictions.csv en", run_dir)
        print("Archivos presentes (ejemplo):", [p.name for p in run_dir.iterdir()])
        print("Opciones: generar calibration_probs.csv con columnas prob_<clase> o recalcular probabilidades desde el modelo.")
        sys.exit(2)

    ok = compute_and_plot(df, outdir, n_bins=args.n_bins)
    if not ok:
        print("No fue posible generar las curvas con la información disponible. Revisar columnas del CSV.")
        sys.exit(3)
    print("Terminado. Revisa:", outdir)

if __name__ == "__main__":
    main()
