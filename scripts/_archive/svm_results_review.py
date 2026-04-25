# svm_results_review.py
import os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

ROOT = Path("D:/pipeline_SVM")
RESULTS_DIR = ROOT / "results"

# 1) detectar último run svm_run_*
runs = sorted([p for p in RESULTS_DIR.glob("svm_run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    raise SystemExit("No encontré carpetas svm_run_* en: " + str(RESULTS_DIR))
run = runs[0]
print("Usando run:", run)

# archivos esperados
preds_f = run / "holdout_predictions.csv"
rep_f = run / "classification_report_holdout.csv"
cm_f = run / "confusion_matrix_holdout.csv"
perm_f = run / "permutation_importance_holdout.csv"
pr_png = run / "plots" / "pr_curves_holdout.png"

out_summary = run / "review_summary.txt"
out_top_errors = run / "top_errors_for_relabel.csv"
plots_dir = run / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)

# cargo preds
if not preds_f.exists():
    raise SystemExit("No encuentro holdout_predictions.csv en: " + str(run))

pred_df = pd.read_csv(preds_f)
print("Holdout preds filas:", len(pred_df))

# resumen basico
counts_true = pred_df['true_label'].value_counts()
counts_pred = pred_df['pred_label'].value_counts()

with open(out_summary, "w", encoding="utf8") as f:
    f.write(f"RUN: {run}\n")
    f.write("=== True label counts ===\n")
    f.write(counts_true.to_string())
    f.write("\n\n=== Pred label counts ===\n")
    f.write(counts_pred.to_string())
    f.write("\n\n")

# 2) classification report (si no hay csv, computar)
if rep_f.exists():
    rep_df = pd.read_csv(rep_f, index_col=0)
    with open(out_summary, "a", encoding="utf8") as f:
        f.write("=== Classification report (loaded) ===\n")
        f.write(rep_df.to_string())
        f.write("\n\n")
else:
    cr = classification_report(pred_df['true_label'], pred_df['pred_label'], digits=4, output_dict=True)
    rep_df = pd.DataFrame(cr).transpose()
    rep_df.to_csv(run / "classification_report_holdout_generated.csv")
    with open(out_summary, "a", encoding="utf8") as f:
        f.write("=== Classification report (generated) ===\n")
        f.write(rep_df.to_string())
        f.write("\n\n")

# 3) confusion matrix
if cm_f.exists():
    cm_df = pd.read_csv(cm_f, index_col=0)
else:
    labels = sorted(list(set(pred_df['true_label']) | set(pred_df['pred_label'])))
    cm = confusion_matrix(pred_df['true_label'], pred_df['pred_label'], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(run / "confusion_matrix_holdout_generated.csv")

# plot confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm_df.astype(int), annot=True, fmt="d", cmap="Blues")
plt.ylabel("true")
plt.xlabel("pred")
plt.title("Confusion matrix (holdout)")
plt.tight_layout()
plt.savefig(plots_dir / "confusion_matrix_holdout.png", dpi=200)
plt.close()

# 4) permutation importance
if perm_f.exists():
    perm = pd.read_csv(perm_f)
    topn = perm.sort_values("importance_mean", ascending=False).head(30)
    plt.figure(figsize=(8,6))
    plt.barh(topn['feature'][::-1], topn['importance_mean'][::-1])
    plt.title("Permutation importance (mean) - top 30")
    plt.tight_layout()
    plt.savefig(plots_dir / "perm_importance_top30.png", dpi=200)
    plt.close()
    with open(out_summary, "a", encoding="utf8") as f:
        f.write("Permutation importance top features:\n")
        f.write(topn[['feature','importance_mean']].to_string(index=False))
        f.write("\n\n")
else:
    with open(out_summary, "a", encoding="utf8") as f:
        f.write("No permutation_importance_holdout.csv encontrado.\n\n")

# 5) PR curves (si ya no hay png, intentar calcular)
if not pr_png.exists():
    print("Calculando PR curves...")
    # detectar prob_ columns
    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    if prob_cols:
        plt.figure(figsize=(8,6))
        classes = [c.replace("prob_","") for c in prob_cols]
        for i, col in enumerate(prob_cols):
            y_true_bin = (pred_df['true_label'] == classes[i]).astype(int)
            y_score = pred_df[col].values
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
            ap = average_precision_score(y_true_bin, y_score)
            plt.plot(recall, precision, label=f"{classes[i]} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall por clase (holdout)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "pr_curves_holdout_generated.png", dpi=200)
        plt.close()
        with open(out_summary, "a", encoding="utf8") as f:
            f.write("PR curves generadas: plots/pr_curves_holdout_generated.png\n\n")
    else:
        with open(out_summary, "a", encoding="utf8") as f:
            f.write("No se encuentran columnas prob_*. No se pudieron generar PR curves.\n\n")
else:
    with open(out_summary, "a", encoding="utf8") as f:
        f.write("PR curve preexistente: " + str(pr_png) + "\n\n")

# 6) Top errors / prioridad para relabel
# Priorizar:
#  - muestras donde true!=pred
#  - modelos con baja prob de su prediccion (inseguro)
prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
if prob_cols:
    probs = pred_df[prob_cols].values
    top_prob = probs.max(axis=1)
    pred_df['model_top_prob'] = top_prob
else:
    pred_df['model_top_prob'] = np.nan

pred_df['disagree'] = (pred_df['true_label'] != pred_df['pred_label'])
pred_df['priority_score'] = pred_df['disagree'].astype(int)*100 + (1 - pred_df['model_top_prob'].fillna(0))

top_errors = pred_df.sort_values(['disagree','model_top_prob','priority_score'], ascending=[False, True, False]).head(500)
top_errors.to_csv(out_top_errors, index=False)

print("Resumen escrito en:", out_summary)
print("Top errors/priority written:", out_top_errors)
print("Plots en:", plots_dir)
