# train_svm_no_augment.py (colocar en D:/pipeline_SVM/scripts)
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
import joblib

ROOT = Path("D:/pipeline_SVM")
FEATURES_CSV = ROOT / "inputs" / "features_svm_baseline_limpios_originals.with_duration.csv"
OUT_DIR = ROOT / "results" / ("svm_noaugment_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES_CSV, low_memory=False)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
RANDOM_STATE = 42
N_SPLITS = 5
GROUP_COL = "experiment"   # ajusta: el nombre de columna que contiene el group (ensayo/mic). Si no existe, usa 'mic_type' o folder padre
LABEL_COL = "label"        # nombre de la columna target en tu CSV
# ----------------------------

print("Cargando features desde:", FEATURES_CSV)
df = pd.read_csv(FEATURES_CSV, low_memory=False)

# Chequeos iniciales
if LABEL_COL not in df.columns:
    raise SystemExit(f"No veo columna '{LABEL_COL}' en {FEATURES_CSV}. Columnas disponibles: {df.columns.tolist()}")

# --- 1) LIMPIEZA DE LABELS ---
# Mapa de normalización (ajusta si tienes variantes)
label_map = {
    "medio_desgaste": "medianamente_desgastado",
    "medianamente_desgastado": "medianamente_desgastado",
    "con_desgaste": "desgastado",
    "desgastado": "desgastado",
    "sin_desgaste": "sin_desgaste",
    "sin desgaste": "sin_desgaste",
    "sin_desgaste ": "sin_desgaste",
    "Sin_desgaste": "sin_desgaste",
    "None": None,
    "nan": None,
    "": None,
    " ": None
}

def clean_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s_low = s.lower()
    # mapa directo por keys
    for k, v in label_map.items():
        if k.lower() == s_low:
            return v
    # Si no se mapeó, devolver versión sin espacios y en minúsculas
    if s == "":
        return None
    return s.replace(" ", "_").lower()

df['label_clean'] = df[LABEL_COL].apply(clean_label)
print("Valores únicos (label_clean):")
print(df['label_clean'].value_counts(dropna=False).to_string())

# Eliminar filas sin label válido
n_before = len(df)
df = df[~df['label_clean'].isna()].copy()
print(f"Filas eliminadas por label inválido: {n_before - len(df)} (de {n_before})")

# --- 2) CONSTRUIR GROUPS robustamente ---
if GROUP_COL in df.columns:
    df['group_tmp'] = df[GROUP_COL]
elif 'mic_type' in df.columns:
    df['group_tmp'] = df['mic_type']
    print(f"Usando 'mic_type' como group (no existe {GROUP_COL})")
elif 'filepath' in df.columns:
    df['group_tmp'] = df['filepath'].astype(str).apply(lambda p: Path(p).parent.name if p and isinstance(p, str) else None)
    print("Usando parent folder como group (desde filepath)")
else:
    df['group_tmp'] = np.arange(len(df))
    print("No hay group claro; usando indices como group (poco ideal)")

# rellenar remaining NaNs groups con parent folder si existe filepath
if 'filepath' in df.columns:
    df['group_tmp'] = df['group_tmp'].fillna(df['filepath'].astype(str).apply(lambda p: Path(p).parent.name if p and isinstance(p, str) else "unknown"))

# final fallback
df['group_tmp'] = df['group_tmp'].fillna('unknown_group')

print("Número de grupos (unique):", df['group_tmp'].nunique())
print("Top grupos:\n", df['group_tmp'].value_counts().head(20).to_string())

# --- 3) Seleccionar columnas de features (coerce a numérico) ---
meta_cols = ['filepath', LABEL_COL, 'label_clean', 'mic_type', 'experiment', 'duration']
meta_cols_present = [c for c in meta_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in meta_cols_present]

# Coerce features to numeric (non-numeric -> NaN). Esto evita errores en SelectKBest.
X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# 1) Eliminar columnas que no deberían ser features (meta / group)
for drop_col in ['group_tmp', 'group', 'group_id', 'index']:
    if drop_col in X_raw.columns:
        X_raw = X_raw.drop(columns=[drop_col])

# 2) Mantener sólo columnas numéricas (por si quedó alguna 'object' por error)
X_raw = X_raw.select_dtypes(include=[np.number])

# 3) Eliminar columnas que son todas NaN (no aportan)
cols_all_nan = X_raw.columns[X_raw.isna().all()].tolist()
if cols_all_nan:
    print("Eliminando columnas sin valores observados (all NaN):", cols_all_nan)
    X_raw = X_raw.drop(columns=cols_all_nan)
# Guardar CSV de auditoría
audit_path = ROOT / "inputs" / "features_svm_baseline_cleaned.csv"
audit_path.parent.mkdir(parents=True, exist_ok=True)
df_to_audit = pd.concat([df[meta_cols_present].reset_index(drop=True), X_raw.reset_index(drop=True)], axis=1)
df_to_audit.to_csv(audit_path, index=False)
print("Audit CSV guardado en:", audit_path)
# --- limpieza rápida antes de X ---
# elimina columnas meta si quedaron en feature_cols
for drop_col in ['group_tmp','group','group_id','index']:
    if drop_col in feature_cols:
        feature_cols.remove(drop_col)

# mantén solo columnas numéricas en X
X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# elimina columnas que son todas NaN
cols_all_nan = X_raw.columns[X_raw.isna().all()].tolist()
if cols_all_nan:
    print("Eliminando columnas sin valores observados (all NaN):", cols_all_nan)
    X_raw = X_raw.drop(columns=cols_all_nan)

# (opcional) eliminar columnas con >50% NaN
th = 0.5
cols_high_nan = X_raw.columns[X_raw.isna().mean() > th].tolist()
if cols_high_nan:
    print("Eliminando columnas con >50% NaN:", cols_high_nan)
    X_raw = X_raw.drop(columns=cols_high_nan)

X = X_raw.copy()


y = df['label_clean'].astype(str).copy()
groups = df['group_tmp'].copy()

# Label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Clases (label encoder):", dict(zip(le.classes_, range(len(le.classes_)))))

# --- 4) chequeos: suficientes grupos y clases ---
n_groups = df['group_tmp'].nunique()
if n_groups < 2:
    raise SystemExit("Demasiados pocos grupos; revisa 'group_tmp'. Necesitas al menos 2.")
if n_groups < N_SPLITS:
    print(f"Advertencia: N_SPLITS ({N_SPLITS}) mayor que número de grupos ({n_groups}). Ajustando N_SPLITS -> {max(2, n_groups)}")
    N_SPLITS = max(2, n_groups)

# check class counts
class_counts = pd.Series(y_enc).value_counts()
print("Conteo por clase (encoded):")
print(class_counts.to_string())

# --- 5) Holdout group-wise (GroupShuffleSplit) ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, hold_idx = next(gss.split(X, y_enc, groups))
X_train, X_hold = X.iloc[train_idx], X.iloc[hold_idx]
y_train, y_hold = y_enc[train_idx], y_enc[hold_idx]
groups_train = groups.iloc[train_idx]

print("Tamaños: train:", len(X_train), "holdout:", len(X_hold))

# --- 6) Pipeline & GridSearch ---
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif, k="all")),
    ("svc", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
])

param_grid = {
    "select__k": [10, 20, "all"],
    "svc__kernel": ["rbf"],
    "svc__C": [1.0, 10.0, 50.0],
    "svc__gamma": ["scale", "auto"]
}

cv = GroupKFold(n_splits=N_SPLITS)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=2, return_train_score=True)

print("Iniciando GridSearch...")
gs.fit(X_train, y_train, groups=groups_train)
print("Mejores params:", gs.best_params_, "best_score:", gs.best_score_)

# --- 7) Calibración (opcional) ---
best_pipe = gs.best_estimator_
# best_pipe viene reflitado por GridSearchCV (refit=True), así que podemos usar cv='prefit' en CalibratedClassifierCV
try:
    cal = CalibratedClassifierCV(best_pipe, cv='prefit', method='isotonic')
    cal.fit(X_train, y_train)  # calibra usando el pipeline ya entrenado
    final_clf = cal
    print("CalibratedClassifierCV aplicado (isotonic).")
except Exception as e:
    print("No se pudo aplicar CalibratedClassifierCV con 'prefit' (fallback al best_pipe):", str(e))
    final_clf = best_pipe

# Save model and artifacts
joblib.dump(final_clf, OUT_DIR / "svm_model_best.joblib")
with open(OUT_DIR / "label_encoder.json", "w") as f:
    json.dump({"classes": le.classes_.tolist()}, f)
pd.DataFrame(gs.cv_results_).to_csv(OUT_DIR / "gridcv_results.csv", index=False)
with open(OUT_DIR / "best_params.json", "w") as f:
    json.dump(gs.best_params_, f)
# save feature list
pd.DataFrame({"feature": X.columns.tolist()}).to_csv(OUT_DIR / "feature_list.csv", index=False)

# --- 8) Predicciones holdout ---
probs = final_clf.predict_proba(X_hold)
preds = final_clf.predict(X_hold)
pred_labels = le.inverse_transform(preds)
true_labels = le.inverse_transform(y_hold)

pred_df = pd.DataFrame({
    "filepath": df.iloc[hold_idx].get("filepath", pd.Series([""]*len(hold_idx))).values,
    "group": groups.iloc[hold_idx].values,
    "true_label": true_labels,
    "pred_label": pred_labels,
})
# add probabilities per class
for i, cls in enumerate(le.classes_):
    pred_df[f"prob_{cls}"] = probs[:, i]
pred_df.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)

# --- 9) Reportes ---
report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report_holdout.csv")
cm = confusion_matrix(true_labels, pred_labels, labels=le.classes_)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(OUT_DIR / "confusion_matrix_holdout.csv")

# --- 10) Permutation importance (holdout) ---
print("Computando permutation importance (esto puede tardar)...")
r = permutation_importance(final_clf, X_hold, y_hold, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
feat_imp = pd.DataFrame({
    "feature": X_hold.columns,
    "importance_mean": r.importances_mean,
    "importance_std": r.importances_std
}).sort_values("importance_mean", ascending=False)
feat_imp.to_csv(OUT_DIR / "permutation_importance_holdout.csv", index=False)

# # --- 11) PR curves ---
# plt.figure(figsize=(8,6))
# for i, cls in enumerate(le.classes_):
#     y_true_bin = (y_hold == i).astype(int)
#     y_score = probs[:, i]
#     precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
#     ap = average_precision_score(y_true_bin, y_score)
#     plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend()
# plt.title("Precision-Recall por clase (holdout)")
# plt.grid(True)
# plt.savefig(PLOTS_DIR / "pr_curves_holdout.png", dpi=200)
# plt.close()

# print("Resultados guardados en:", OUT_DIR)
