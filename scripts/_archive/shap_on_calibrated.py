import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

MODEL_PATH = Path("D:/pipeline_SVM/results/svm_final_fast/best_model_calibrated.joblib")
CSV = Path("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv")
OUT_DIR = Path("D:/pipeline_SVM/results/svm_final_fast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar modelo
print("Cargando modelo calibrado:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# Cargar datos
df = pd.read_csv(CSV, low_memory=False)
label_col = "label_fixed"
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col].astype(str).copy()
mask = y.notna() & (y.str.strip() != '') & (y.str.lower() != 'nan')
X = X.loc[mask]
y = y.loc[mask]

# Muestreo para SHAP
n_bg = 200
n_eval = 500
if X.empty:
    raise ValueError("Dataset vacío después de filtrar etiquetas.")
X_bg = X.sample(n=min(n_bg, len(X)), random_state=0)
X_eval = X.sample(n=min(n_eval, len(X)), random_state=1)

# Selección de explainer
try:
    final = getattr(model, 'named_steps', {}).get('clf', None)
    if final is not None and isinstance(final, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    elif final is not None and isinstance(final, SVC) and final.kernel == 'linear':
        explainer = shap.LinearExplainer(model, X_bg)
    else:
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(pd.DataFrame(z, columns=X.columns)), X_bg, keep_index=False)
except (ValueError, TypeError) as e:
    print(f"Error al inicializar explainer: {e}. Usando KernelExplainer.")
    explainer = shap.KernelExplainer(lambda z: model.predict_proba(pd.DataFrame(z, columns=X.columns)), X_bg, keep_index=False)

# Computar SHAP
print("Computando valores SHAP (esto puede tardar)...")
shap_values = explainer.shap_values(X_eval)

# Determinar clase objetivo
classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps['clf'].classes_)
def detect_target(classes, keywords=['desgast', 'falla', 'fault', 'con falla', 'faulty', 'failure']):
    for kw in keywords:
        for c in classes:
            if kw in c.lower():
                return c
    return None
target = detect_target(classes) or classes[-1]
idx_target = classes.index(target) if target else None

# Calcular mean_abs_shap
def mean_abs_shap_from_shap_values(shap_values):
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(a) if a.ndim == 2 else a.reshape(a.shape[0], -1) for a in shap_values], axis=0)
    else:
        arr = np.asarray(shap_values)
    
    if arr.ndim == 2:
        return np.mean(np.abs(arr), axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] < arr.shape[2]:
            return np.mean(np.abs(arr), axis=(0, 1))
        else:
            arr2 = np.transpose(arr, (2, 0, 1))
            return np.mean(np.abs(arr2), axis=(0, 1))
    else:
        raise RuntimeError(f"shap_values con ndim inesperado: {arr.ndim}")

mean_abs_shap = mean_abs_shap_from_shap_values(shap_values)
feat_names = list(X_eval.columns)

# Verificaciones
if mean_abs_shap.ndim != 1:
    raise RuntimeError(f"mean_abs_shap tiene ndim={mean_abs_shap.ndim}, se esperaba 1")
if len(mean_abs_shap) != len(feat_names):
    raise RuntimeError(f"Nº features mismatch: mean_abs_shap={len(mean_abs_shap)} vs feat_names={len(feat_names)}")

# Guardar resultados
df_imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_shap})
df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
df_imp.to_csv(OUT_DIR / "shap_mean_abs_importance.csv", index=False)
print("Saved SHAP importances:", OUT_DIR / "shap_mean_abs_importance.csv")