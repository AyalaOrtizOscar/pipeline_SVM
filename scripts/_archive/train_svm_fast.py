# train_svm_fast.py (corregido)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# Ajusta rutas si hace falta
INPUT = "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv"
OUTDIR = "D:/pipeline_SVM/results/svm_final_fast"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("Cargando...")
df = pd.read_csv(INPUT, low_memory=False)

label_col = "label_fixed"
if label_col not in df.columns:
    raise SystemExit(f"No se encontró columna {label_col} en {INPUT}")

# X: solo numéricos
X = df.select_dtypes(include=[np.number]).copy()
y = df[label_col]

# Filtrar filas con label válidos:
# - no nulos
# - no cadenas vacías
# - no literal 'nan' (si viene como texto)
y_str = y.astype(str).str.strip()
mask = y.notna() & (y_str != '') & (y_str.str.lower() != 'nan')
X = X.loc[mask]
y = y.loc[mask].astype(str)

print(f"Rows: {len(X)}, features: {X.shape[1]}")

# split
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=mutual_info_classif)),
    ("clf", LinearSVC(max_iter=20000, dual=False))
])

param_dist = {
    "select__k": [10, 20],
    "clf__C": [1.0, 10.0, 50.0]
}

# Ajusta n_jobs según tu RAM/CPU. Con tu i7-8700K: n_jobs=4 es razonable.
n_jobs = 4

search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=8, cv=3,
    n_jobs=n_jobs, verbose=3, random_state=42
)

print("Fitting RandomizedSearchCV ...")
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
best = search.best_estimator_

# evaluar
p = best.predict(X_hold)
print("Holdout report:")
print(classification_report(y_hold, p, zero_division=0))

# guardar modelo
joblib.dump(best, Path(OUTDIR)/"best_model.joblib")
print("Modelo guardado en", Path(OUTDIR)/"best_model.joblib")
