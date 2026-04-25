# D:\pipeline_SVM\scripts\train_svm_with_smote.py
import json, os, joblib
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------- CONFIG ----------
ROOT = Path("D:/pipeline_SVM")
FEATURES_CSV = ROOT / "features" / "features_svm_baseline.csv"
OUT_DIR = ROOT / "results" / ("svm_run_smote_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5
GROUP_COL = "experiment"   # ajusta si necesario
LABEL_COL = "label"        # columna target
# ----------------------------

print("Cargando features desde:", FEATURES_CSV)
df = pd.read_csv(FEATURES_CSV, low_memory=False)

# --- limpieza / normalización simple de etiquetas (igual que tu pipeline) ---
def clean_label(x):
    if pd.isna(x): return None
    s = str(x).strip().lower().replace(" ", "_")
    # mapeos simples
    if s in ("con_desgaste","desgastado"): return "desgastado"
    if "medio" in s or "medianamente" in s: return "medianamente_desgastado"
    if "sin" in s: return "sin_desgaste"
    return s

df['label_clean'] = df[LABEL_COL].apply(clean_label)
df = df[~df['label_clean'].isna()].copy()

# groups
if GROUP_COL in df.columns:
    df['group_tmp'] = df[GROUP_COL]
elif 'mic_type' in df.columns:
    df['group_tmp'] = df['mic_type']
elif 'filepath' in df.columns:
    df['group_tmp'] = df['filepath'].astype(str).apply(lambda p: Path(p).parent.name)
else:
    df['group_tmp'] = np.arange(len(df))

# features selection (keep numeric features)
meta_cols = ['filepath', LABEL_COL, 'label_clean', 'mic_type', 'experiment', 'duration']
meta_present = [c for c in meta_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in meta_present]

# Coerce numeric
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
# drop all-nan features
X = X.drop(columns=X.columns[X.isna().all()].tolist())
y = df['label_clean'].astype(str).copy()
groups = df['group_tmp'].copy()

# label encoder
le = LabelEncoder(); y_enc = le.fit_transform(y)
print("Clases (label encoder):", dict(zip(le.classes_, range(len(le.classes_)))))

# check class distribution
class_counts = pd.Series(y_enc).value_counts().sort_index()
print("Conteo por clase (encoded):\n", class_counts.to_string())

# Holdout group-wise: GroupShuffleSplit (20% holdout)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, hold_idx = next(gss.split(X, y_enc, groups))
X_train, X_hold = X.iloc[train_idx], X.iloc[hold_idx]
y_train, y_hold = y_enc[train_idx], y_enc[hold_idx]
groups_train = groups.iloc[train_idx]

print("Tamaños: train:", len(X_train), "holdout:", len(X_hold))

# --- SMOTE config: decide sampling_strategy y k_neighbors de forma conservadora ---
# identificar label minoritario (encoded)
unique, counts = np.unique(y_train, return_counts=True)
enc_counts = dict(zip(unique, counts))
min_label_enc = min(enc_counts, key=lambda k: enc_counts[k])
min_count = enc_counts[min_label_enc]
total_train = len(y_train)

# objetivo conservador: llevar minority a al menos median of class counts o al 10% del total (lo que sea mayor), pero no igualar todo
from math import floor
median_count = int(np.median(list(enc_counts.values())))
target_minority = max(median_count, int(total_train * 0.10))
target_minority = min(target_minority, max(enc_counts.values()))  # no exagerar
print(f"Minority encoded label: {min_label_enc} (count {min_count}). Target after SMOTE: {target_minority}")

# sampling dict
sampling_strategy = {min_label_enc: target_minority}

# k_neighbors: must be < n_minority
k_neighbors = min(3, max(1, min_count - 1))
print("SMOTE k_neighbors:", k_neighbors)

smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=RANDOM_STATE)

# --- pipeline imblearn: imputer -> scaler -> SMOTE -> select -> svc ---
pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", smote),
    ("select", SelectKBest(mutual_info_classif, k="all")),
    ("svc", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
])

param_grid = {
    "select__k": [8, 10, "all"],           # ajusta según tus features
    "svc__kernel": ["rbf"],
    "svc__C": [10.0, 50.0],
    "svc__gamma": ["scale", "auto"]
}

cv = GroupKFold(n_splits=min(N_SPLITS, groups_train.nunique()))
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=2, return_train_score=True)

print("Iniciando GridSearch con SMOTE...")
gs.fit(X_train, y_train, groups=groups_train)
print("Mejores params:", gs.best_params_, "best_score:", gs.best_score_)

best_pipe = gs.best_estimator_

# Guardar modelo (sin calibrar) y resultados
joblib.dump(best_pipe, OUT_DIR / "svm_smote_best.joblib")
pd.DataFrame(gs.cv_results_).to_csv(OUT_DIR / "gridcv_results.csv", index=False)
with open(OUT_DIR / "best_params.json","w") as f:
    json.dump(gs.best_params_, f)
with open(OUT_DIR / "label_encoder.json","w") as f:
    json.dump({"classes": le.classes_.tolist()}, f)

# --- predict holdout ---
probs = best_pipe.predict_proba(X_hold)
preds = best_pipe.predict(X_hold)
pred_labels = le.inverse_transform(preds)
true_labels = le.inverse_transform(y_hold)

pred_df = pd.DataFrame({
    "filepath": df.iloc[hold_idx].get("filepath", pd.Series([""]*len(hold_idx))).values,
    "group": groups.iloc[hold_idx].values,
    "true_label": true_labels,
    "pred_label": pred_labels,
})
for i, cls in enumerate(le.classes_):
    pred_df[f"prob_{cls}"] = probs[:, i]
pred_df.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)

# --- reports ---
report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
pd.DataFrame(report).transpose().to_csv(OUT_DIR / "classification_report_holdout.csv")
cm = confusion_matrix(true_labels, pred_labels, labels=le.classes_)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(OUT_DIR / "confusion_matrix_holdout.csv")

# permutation importance on holdout
print("Computando permutation importance...")
r = permutation_importance(best_pipe, X_hold, y_hold, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
feat_imp = pd.DataFrame({
    "feature": X_hold.columns,
    "importance_mean": r.importances_mean,
    "importance_std": r.importances_std
}).sort_values("importance_mean", ascending=False)
feat_imp.to_csv(OUT_DIR / "permutation_importance_holdout.csv", index=False)

# PR curves
plt.figure(figsize=(8,6))
for i, cls in enumerate(le.classes_):
    y_true_bin = (y_hold == i).astype(int)
    y_score = probs[:, i]
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall por clase (holdout) con SMOTE")
plt.grid(True)
plt.savefig(PLOTS_DIR / "pr_curves_holdout_smote.png", dpi=200)
plt.close()

print("Resultados guardados en:", OUT_DIR)
