#!/usr/bin/env python3
"""
train_svm_final.py (fixed)

Usage example:
python d:/pipeline_SVM/scripts/train_svm_final.py --input "D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv" --outdir "D:/pipeline_SVM/results/svm_final" --group-col basename --holdout-frac 0.10 --n-jobs 6
"""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")

def detect_label_column(df):
    for c in ['label_fixed','label_clean','label']:
        if c in df.columns:
            return c
    return None

def numeric_feature_columns(df, exclude=None):
    if exclude is None:
        exclude = ['filepath','basename','fp_norm','label_fixed','label_clean','label',
                   'mic_type','experiment','label_map_method','is_augment','map_method']
    # pick numeric dtype columns not in exclude
    cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="CSV harmonized final (con label_fixed).")
    p.add_argument("--outdir", "-o", default="results/svm_final", help="Output folder.")
    p.add_argument("--group-col", default="basename", help="Column to group by for splits.")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--k-options", nargs="+", default=["10","20","50","all"], help="k options (use 'all' to skip selection).")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()

    t0 = time.strftime("%Y%m%d_%H%M%S")
    OUT = Path(args.outdir) / f"svm_run_{t0}"
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    label_col = detect_label_column(df)
    if label_col is None:
        raise RuntimeError("No label column found. Añade label_fixed o label_clean al CSV.")
    print("Label column:", label_col)

    # drop all-NaN cols
    allnan = df.columns[df.isna().all()].tolist()
    if allnan:
        print("Dropping all-NaN columns:", allnan)
        df = df.drop(columns=allnan)

    # prepare group column
    if args.group_col in df.columns:
        groups = df[args.group_col].astype(str).fillna('')
    elif 'basename' in df.columns:
        groups = df['basename'].astype(str).fillna('')
        print(f"Warning: group-col '{args.group_col}' not found. Using 'basename' instead.")
    else:
        groups = pd.Series([str(i) for i in range(len(df))])

    # labels (force strings)
    y = df[label_col].astype(str).str.strip()
    # keep only valid labels (not blank / 'nan')
    sel = y.notna() & (y != '') & (y.str.lower() != 'nan')
    df = df[sel].reset_index(drop=True)
    y = y[sel].reset_index(drop=True)
    groups = groups[sel].reset_index(drop=True)

    print("Rows available for training:", len(df))

    feat_cols = numeric_feature_columns(df)
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found. Check CSV.")
    print("Numeric features:", len(feat_cols))

    X = df[feat_cols].values

    # create holdout by group
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.random_state)
    train_idx, hold_idx = next(gss.split(X, y, groups))
    X_train, X_hold = X[train_idx], X[hold_idx]
    y_train, y_hold = y.iloc[train_idx].values, y.iloc[hold_idx].values
    g_train, g_hold = groups.iloc[train_idx].values, groups.iloc[hold_idx].values

    print("Train / Holdout sizes:", X_train.shape[0], X_hold.shape[0])

    # parse k options
    ks = []
    for k in args.k_options:
        if isinstance(k, str) and k.lower() == 'all':
            ks.append('all')
        else:
            try:
                ks.append(int(k))
            except Exception:
                pass
    ks = sorted(set(ks), key=lambda x: (str(x)!='all', x))

    # build pipeline with a placeholder 'select' step (we will override in param_grid)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    select_init = SelectKBest(score_func=mutual_info_classif)  # default k=10
    svc = SVC(probability=True, class_weight='balanced', random_state=args.random_state)

    pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('select', select_init),
        ('svc', svc)
    ])

    # cross-validator
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # build param grid as list of dicts to allow 'passthrough'
    svc_C = [1.0, 10.0, 50.0]
    svc_kernels = ['rbf', 'linear']
    svc_gamma = ['scale']

    param_grid = []
    # case: use SelectKBest with numeric ks (exclude 'all')
    numeric_ks = [k for k in ks if k != 'all']
    if numeric_ks:
        param_grid.append({
            'select': [SelectKBest(score_func=mutual_info_classif)],
            'select__k': numeric_ks,
            'svc__C': svc_C,
            'svc__kernel': svc_kernels,
            'svc__gamma': svc_gamma
        })
    # case: no selection (passthrough)
    if 'all' in ks:
        param_grid.append({
            'select': ['passthrough'],
            'svc__C': svc_C,
            'svc__kernel': svc_kernels,
            'svc__gamma': svc_gamma
        })

    print("Parameter grid:", param_grid)

    gs = GridSearchCV(pipe,
                      param_grid,
                      cv=cv,
                      scoring='f1_macro',
                      n_jobs=args.n_jobs,
                      verbose=2,
                      refit=True)

    print("Fitting GridSearchCV ...")
    # pass groups to fit so StratifiedGroupKFold can use them
    gs.fit(X_train, y_train, groups=g_train)
    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    # Evaluate on holdout
    y_pred = best.predict(X_hold)
    y_proba = best.predict_proba(X_hold) if hasattr(best, "predict_proba") else None

    acc = accuracy_score(y_hold, y_pred)
    f1m = f1_score(y_hold, y_pred, average='macro')
    print(f"Holdout accuracy: {acc:.4f}  f1_macro: {f1m:.4f}")

    # save classification report & confusion matrix
    cl_rep = classification_report(y_hold, y_pred, output_dict=True)
    labels_union = sorted(list(set(list(y_train) + list(y_hold))))
    cm = confusion_matrix(y_hold, y_pred, labels=labels_union)
    cm_df = pd.DataFrame(cm, index=labels_union, columns=labels_union)

    (OUT / "holdout_classification_report.json").write_text(json.dumps(cl_rep, indent=2, ensure_ascii=False))
    cm_df.to_csv(OUT / "confusion_matrix_holdout.csv", index=True)

    # save holdout predictions and probabilities
    df_hold = df.iloc[hold_idx].copy()
    df_hold['pred'] = y_pred
    if y_proba is not None:
        proba_cols = [f"prob_{c}" for c in gs.best_estimator_.named_steps['svc'].classes_]
        prob_df = pd.DataFrame(y_proba, columns=proba_cols)
        df_hold = pd.concat([df_hold.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
    df_hold.to_csv(OUT / "holdout_predictions.csv", index=False)

    # save model & metadata
    joblib.dump(gs, OUT / "gridsearch_pipeline.joblib")
    joblib.dump(best, OUT / "best_pipeline.joblib")
    meta = {
        "best_params": gs.best_params_,
        "holdout_accuracy": float(acc),
        "holdout_f1_macro": float(f1m),
        "n_train": int(X_train.shape[0]),
        "n_holdout": int(X_hold.shape[0]),
        "features_count": len(feat_cols),
        "feature_names": feat_cols
    }
    (OUT / "training_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("Saved artifacts to:", OUT)
    print("Done.")

if __name__ == "__main__":
    main()
