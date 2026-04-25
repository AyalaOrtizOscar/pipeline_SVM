#!/usr/bin/env python3
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix

MODEL_PATH = Path("D:/pipeline_SVM/results/svm_final_fast/best_model.joblib")
FEATURES_CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast"); OUT_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(FEATURES_CSV, low_memory=False)
label_col="label_fixed"
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col].astype(str).copy()
mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
X=X.loc[mask]; y=y.loc[mask]

from sklearn.model_selection import train_test_split
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

classes = list(model.classes_)
# heurística para elegir clase objetivo (prioriza 'desgast', 'falla', 'fault', 'con falla')
def detect_target(classes):
    keywords = ['desgast', 'falla', 'fault', 'con falla', 'faulty', 'failure', 'fault']
    for kw in keywords:
        for c in classes:
            if kw in c.lower():
                return c
    # fallback: prefer last class
    return classes[-1]

TARGET_LABEL = detect_target(classes)
idx_target = classes.index(TARGET_LABEL)
print("Target label chosen:", TARGET_LABEL)

# obtener probabilidades o estimar desde decision_function
probas = None
if hasattr(model, "predict_proba"):
    try:
        probas = model.predict_proba(X_hold)
    except Exception:
        probas = None

if probas is None:
    # try decision_function then softmax
    try:
        dec = model.decision_function(X_hold)
        if dec.ndim == 1:
            # binary decision function
            import scipy.special as sc
            probs_target = 1/(1+np.exp(-dec))
            # build full-proba-like array: target prob and 1-target
            probas = np.vstack([1-probs_target, probs_target]).T
            # ensure classes length 2, otherwise mapping may be wrong
            if len(classes) != 2:
                # try to softmax multi-dim decision
                dec2 = model.decision_function(X_hold)
                ex = np.exp(dec2 - np.max(dec2, axis=1, keepdims=True))
                probas = ex / ex.sum(axis=1, keepdims=True)
    except Exception:
        raise SystemExit("No se pudieron obtener probabilidades ni decision_function del modelo.")

# ahora probas debería existir como (n_samples, n_classes)
if probas is None:
    raise SystemExit("No hay probabilidades disponibles para calibrar umbral.")

probs_target = probas[:, idx_target]

# función para predecir etiqueta dada una probabilidad y threshold
def pred_from_prob_row(probas_row, thr, idx_target):
    # si target supera umbral -> target
    if probas_row[idx_target] >= thr:
        return classes[idx_target]
    # si no -> elegir la clase con mayor probabilidad entre las otras
    other_idx = np.arange(len(classes)) != idx_target
    # pick argmax among other classes
    idx_best_other = np.argmax(probas_row[other_idx])
    # map to actual index
    actual_inds = np.arange(len(classes))[other_idx]
    return classes[actual_inds[idx_best_other]]

best = {"threshold": None, "f1": -1.0}
# buscar en 0..1 en pasos de 0.01
for thr in np.linspace(0.0,1.0,101):
    y_pred_thr = [pred_from_prob_row(row, thr, idx_target) for row in probas]
    f1 = f1_score(y_hold, y_pred_thr, average='macro', zero_division=0)
    if f1 > best["f1"]:
        best = {"threshold": float(thr), "f1": float(f1)}

print("Best threshold:", best)
# calcular y guardar confusion al mejor threshold
y_pred_best = [pred_from_prob_row(row, best["threshold"], idx_target) for row in probas]
cm = confusion_matrix(y_hold, y_pred_best, labels=classes)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_DIR/"confusion_at_best_threshold.csv")
print("Saved confusion_at_best_threshold.csv in", OUT_DIR)
