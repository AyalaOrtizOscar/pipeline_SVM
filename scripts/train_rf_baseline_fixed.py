# train_rf_baseline_fixed.py
import pandas as pd, numpy as np, joblib, json, os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path("D:/pipeline_SVM")
F = ROOT/"features"/"features_svm_baseline.csv"
OUT = ROOT/"results"/("rf_run_fixed_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
OUT.mkdir(parents=True, exist_ok=True)

print("Leyendo:", F)
df = pd.read_csv(F, low_memory=False)

# label: prioriza label_clean si existe
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
if label_col not in df.columns:
    raise SystemExit(f"No veo columna de label ({label_col}) en {F}")

# Construir groups de forma robusta
if 'experiment' in df.columns:
    groups = df['experiment'].astype(str)
elif 'mic_type' in df.columns:
    groups = df['mic_type'].astype(str)
elif 'filepath' in df.columns:
    groups = df['filepath'].astype(str).apply(lambda p: Path(p).parent.name if pd.notna(p) else "unknown")
else:
    groups = pd.Series([f"g{i}" for i in range(len(df))])

# Rellenar NaN y limpiar cadenas
groups = groups.fillna("unknown_group").astype(str).str.strip()
df['group_tmp'] = groups

# Informar
print("Número de grupos (unique):", df['group_tmp'].nunique())
print(df['group_tmp'].value_counts().head(20).to_string())

# Features: sólo numéricas (excluye metadatos)
meta = {'filepath','label','label_clean','mic_type','experiment','duration','duration_s','wav_path_norm'}
X = df[[c for c in df.columns if c not in meta]].apply(pd.to_numeric, errors='coerce')
# drop cols all NaN
allnan = X.columns[X.isna().all()].tolist()
if allnan:
    print("Eliminando columnas all-NaN:", allnan)
    X = X.drop(columns=allnan)
if X.shape[1] == 0:
    raise SystemExit("No quedan features numéricas tras limpieza. Revisa el CSV de features.")

y = df[label_col].astype(str).fillna("UNKNOWN").copy()
# --- FILTRO DE FILAS CON LABELS INVÁLIDAS ----
# normalizar label column
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
df[label_col] = df[label_col].astype(str).str.strip().replace({'nan':'', 'None':'', 'NoneType':''})
# considera valores vacíos como inválidos
invalid_mask = df[label_col].isna() | (df[label_col].astype(str).str.lower().isin(['', 'none', 'nan']))
n_invalid = invalid_mask.sum()
if n_invalid > 0:
    print(f"Eliminando filas con label inválido: {n_invalid} filas")
    df = df[~invalid_mask].copy()

le = LabelEncoder(); y_enc = le.fit_transform(y)
print("Clases (label encoder):", dict(zip(le.classes_, range(len(le.classes_)))))

# Opcional: convertir a binario desgastado vs rest (descomenta si quieres)
#y_bin = (y == 'desgastado').astype(int)
#le_bin = None

# Chequeo tamaños por clase
print("Conteo por clase:")
print(pd.Series(y_enc).value_counts().to_string())

# Holdout group-wise
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
try:
    train_idx, test_idx = next(gss.split(X, y_enc, df['group_tmp']))
except Exception as e:
    print("Error en GroupShuffleSplit:", e)
    # fallback simple shuffle split
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y_enc)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_enc[train_idx], y_enc[test_idx]

print("Tamanos: train:", len(X_train), "holdout:", len(X_test))

# Pipeline RF
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=40, class_weight='balanced', random_state=42, n_jobs=-1))
])

print("Entrenando RF...")
pipe.fit(X_train, y_train)

# Guardar
joblib.dump(pipe, OUT/"rf_baseline_fixed.joblib")
with open(OUT/"label_encoder.json","w") as f:
    json.dump({"classes": le.classes_.tolist()}, f)

# Evaluacion holdout
pred = pipe.predict(X_test)
pred_labels = le.inverse_transform(pred)
true_labels = le.inverse_transform(y_test)
pd.DataFrame(classification_report(true_labels, pred_labels, output_dict=True)).transpose().to_csv(OUT/"classification_report_holdout.csv")
pd.DataFrame(confusion_matrix(true_labels,pred_labels), index=le.classes_, columns=le.classes_).to_csv(OUT/"confusion_matrix_holdout.csv")

# Feature importance
fi = pipe.named_steps['rf'].feature_importances_
pd.DataFrame({"feature":X.columns,"importance":fi}).sort_values("importance",ascending=False).to_csv(OUT/"feature_importances_rf.csv", index=False)

print("RF terminado. Resultados en:", OUT)
