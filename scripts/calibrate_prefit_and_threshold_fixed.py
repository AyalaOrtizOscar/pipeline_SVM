# calibrate_prefit_and_threshold_fixed.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("D:/pipeline_SVM/results/svm_final_fast/best_model.joblib")
CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Cargando pipeline entrenado...", MODEL_PATH)
model = joblib.load(MODEL_PATH)  # pipeline (fitted)

print("Cargando datos...", CSV)
df = pd.read_csv(CSV, low_memory=False)
label_col = "label_fixed"
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col].astype(str).copy()
mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
X = X.loc[mask]; y = y.loc[mask]

# keep a small hold-out for calibration (same split logic used antes)
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# instantiate CalibratedClassifierCV robustly across sklearn versions
cal = None
args_try = [
    {"estimator": model, "method": "sigmoid", "cv": "prefit"},
    {"base_estimator": model, "method": "sigmoid", "cv": "prefit"},
]

for kwargs in args_try:
    try:
        cal = CalibratedClassifierCV(**kwargs)
        break
    except TypeError:
        cal = None

# fallback: positional (estimator as first positional arg)
if cal is None:
    try:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    except Exception as e:
        raise RuntimeError("No se pudo instanciar CalibratedClassifierCV en esta versión de sklearn") from e

print("Calibrando con cv='prefit' (ajusta solo calibrador)...")
cal.fit(X_calib, y_calib)  # rápido: solo ajusta calibrador
joblib.dump(cal, OUT_DIR/"best_model_calibrated.joblib")
print("Guardado modelo calibrado en:", OUT_DIR/"best_model_calibrated.joblib")

# probabilidades sobre set de calibración
probas = cal.predict_proba(X_calib)
classes = list(cal.classes_)
print("Clases:", classes)

# selecciona target (heurística)
def detect_target(classes):
    for kw in ['desgast','falla','fault','con falla','faulty','failure']:
        for c in classes:
            if kw in c.lower():
                return c
    return classes[-1]

TARGET = detect_target(classes)
idx_target = classes.index(TARGET)
print("Target elegido:", TARGET)

probs_target = probas[:, idx_target]

# búsqueda de umbral que maximiza f1_macro (puedes cambiar)
best = {"thr":None, "f1":-1}
for thr in np.linspace(0.0, 1.0, 201):
    preds = []
    for i, p in enumerate(probs_target):
        if p >= thr:
            preds.append(TARGET)
        else:
            # elegimos argmax entre las demás clases
            row = probas[i].copy()
            row[idx_target] = -1
            preds.append(classes[row.argmax()])
    f1 = f1_score(y_calib, preds, average='macro', zero_division=0)
    if f1 > best["f1"]:
        best = {"thr":float(thr), "f1":float(f1)}

print("Mejor umbral (por f1_macro) en set de calibración:", best)
thr = best["thr"]

# reporte y matriz de confusión con ese umbral
y_pred_thr = []
for i,p in enumerate(probs_target):
    if p >= thr:
        y_pred_thr.append(TARGET)
    else:
        row = probas[i].copy(); row[idx_target] = -1
        y_pred_thr.append(classes[row.argmax()])

print("Classification report (umbral):")
print(classification_report(y_calib, y_pred_thr, zero_division=0))
cm = confusion_matrix(y_calib, y_pred_thr, labels=classes)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_DIR/"confusion_calibrated_threshold.csv")
pd.DataFrame(probas, columns=[f"prob_{c}" for c in classes], index=y_calib.index).to_csv(OUT_DIR/"calibration_probs.csv")

print("Matriz de confusión guardada en:", OUT_DIR/"confusion_calibrated_threshold.csv")
print("Probabilidades guardadas en:", OUT_DIR/"calibration_probs.csv")
