# calibrate_prefit_and_threshold.py
import joblib, numpy as np, pandas as pd
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

# reproducible split igual que en entrenamiento (test_size 10%)
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Calibrate (cv='prefit' exige que model ya esté fit)
print("Calibrando con cv='prefit' (ajusta solo calibrador)...")
cal = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
cal.fit(X_calib, y_calib)  # fast: sólo ajusta calibración
joblib.dump(cal, OUT_DIR/"best_model_calibrated.joblib")
print("Guardado modelo calibrado en:", OUT_DIR/"best_model_calibrated.joblib")

# ahora evaluamos sobre X_calib (o X_hold si preferías otro conjunto)
probas = cal.predict_proba(X_calib)
classes = list(cal.classes_)
print("Clases:", classes)

# detecta target heurísticamente (usa 'desgast' si aparece)
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

# búsqueda de umbral que maximiza f1_macro (puedes cambiar a f1 para target si prefieres)
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
# generar reporte con ese umbral
thr = best["thr"]
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
print("Matriz de confusión guardada en:", OUT_DIR/"confusion_calibrated_threshold.csv")

# también guarda probabilidades para inspección
pd.DataFrame(probas, columns=[f"prob_{c}" for c in classes], index=y_calib.index).to_csv(OUT_DIR/"calibration_probs.csv")
print("Probabilidades guardadas en:", OUT_DIR/"calibration_probs.csv")
