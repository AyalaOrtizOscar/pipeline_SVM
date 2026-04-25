#!/usr/bin/env python3
import json, os, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path("D:/pipeline_SVM")
FEATURES = ROOT / "features" / "features_svm_with_augmented.csv"
OUT = ROOT / "results" / ("svm_aug_run_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
OUT.mkdir(parents=True, exist_ok=True)

print("Cargando features:", FEATURES)
df = pd.read_csv(FEATURES, low_memory=False)

# Normaliza label
if 'label_clean' in df.columns:
    df['label_clean'] = df['label_clean'].astype(str)
elif 'label' in df.columns:
    df['label_clean'] = df['label'].astype(str)
else:
    raise SystemExit("No encuentro columna label/label_clean")

# Asegurarse columna is_aug
if 'is_aug' not in df.columns:
    df['is_aug'] = df['basename'].astype(str).str.contains('_aug|augment|auto', case=False, na=False)

# Solo filas con label válidos
df = df[df['label_clean'].notna() & (df['label_clean'] != '')].copy()

# Features: quitar meta
meta = ['filepath','basename','label','label_clean','mic_type','experiment','is_aug','duration','wav_path_norm']
feature_cols = [c for c in df.columns if c not in meta]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.loc[:, ~X.isna().all()]  # drop all-NaN columns
y = df['label_clean'].copy()

# Groups: prefer experiment / mic_type / parent folder
if 'experiment' in df.columns and df['experiment'].notna().any():
    groups = df['experiment'].fillna('unknown')
elif 'mic_type' in df.columns:
    groups = df['mic_type'].fillna('unknown')
else:
    groups = df['filepath'].astype(str).apply(lambda p: Path(p).parent.name).fillna('unknown')

# Create holdout indices using only originals (is_aug==False) so holdout contains no augmented samples
orig_idx = df.index[ ~df['is_aug'].astype(bool) ].tolist()
print("Originals count:", len(orig_idx))
if len(orig_idx) < 50:
    print("Pocos originales; ajustar test_size o revisar dataset.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# split only on subset of rows (originals)
train_idx_sub, hold_idx_sub = next(gss.split(df.loc[orig_idx], df.loc[orig_idx,'label_clean'], groups.loc[orig_idx]))
# map back to global indices
hold_idx = [ orig_idx[i] for i in hold_idx_sub ]
train_idx_originals = [ orig_idx[i] for i in train_idx_sub ]

# training indices: todos los indices EXCEPTO holdout (incluye augmented + remaining originals)
train_idx = [i for i in df.index if i not in hold_idx]

X_train, X_hold = X.iloc[train_idx], X.iloc[hold_idx]
y_train, y_hold = y.iloc[train_idx], y.iloc[hold_idx]
groups_train = groups.iloc[train_idx]

print("Tamaños: train:", len(X_train), "holdout:", len(X_hold))

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif, k="all")),
    ("svc", SVC(probability=True, class_weight="balanced", random_state=42))
])

param_grid = {
    "select__k": [5, 10, "all"],
    "svc__kernel": ["rbf"],
    "svc__C": [10.0, 50.0],
    "svc__gamma": ["scale"]
}

cv = GroupKFold(n_splits=5)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=2, return_train_score=True)
gs.fit(X_train, y_train, groups=groups_train)
print("Mejores params:", gs.best_params_, "score:", gs.best_score_)

best = gs.best_estimator_
try:
    cal = CalibratedClassifierCV(best, cv='prefit', method='isotonic')
    cal.fit(X_train, y_train)
    final = cal
except Exception:
    final = best

joblib.dump(final, OUT / "svm_aug_model.joblib")
pd.DataFrame(gs.cv_results_).to_csv(OUT / "gridcv.csv", index=False)

probs = final.predict_proba(X_hold)
preds = final.predict(X_hold)
report = classification_report(y_hold, preds, output_dict=True)
pd.DataFrame(report).transpose().to_csv(OUT / "classification_report_holdout.csv")
pd.DataFrame(confusion_matrix(y_hold, preds), index=np.unique(y_hold), columns=np.unique(y_hold)).to_csv(OUT / "confusion_matrix_holdout.csv")

print("Saved results in:", OUT)
