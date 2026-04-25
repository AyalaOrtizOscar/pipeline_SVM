# scripts/train_svm_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance


p = argparse.ArgumentParser()
p.add_argument('--features', default=r'D:/pipeline_SVM/features/features_svm_baseline.csv')
# column that contains grouping key (experiment, filepath pattern, etc.)
p.add_argument('--group_col', default='experiment')
p.add_argument('--label_col', default='label')
p.add_argument('--out_model', default=r'D:/pipeline_SVM/models/svm_baseline.joblib')
args = p.parse_args()


F = Path(args.features)
if not F.exists():
    raise SystemExit('No features file: '+str(F))


df = pd.read_csv(F, low_memory=False)
# Filter rows with missing label
df = df[df[args.label_col].notna()]


# encode labels
labels = df[args.label_col].astype(str)
unique = sorted(labels.unique())
label2idx = {v:i for i,v in enumerate(unique)}
y = labels.map(label2idx).values


# groups: fall back to filepath parent folder if experiment missing
if args.group_col in df.columns and df[args.group_col].notna().any():
    groups = df[args.group_col].astype(str).values
else:
    groups = df['filepath'].astype(str).apply(lambda p: str(Path(p).parent)).values


# X: numeric columns only (drop meta)
meta = ['filepath','label','experiment','mic_type']
X = df.drop(columns=[c for c in df.columns if c in meta])
X = X.select_dtypes(include=[np.number])


# pipeline: impute, scale, feature selection, svm
pipe = make_pipeline(
SimpleImputer(strategy='median'),
StandardScaler(),
SelectKBest(mutual_info_classif, k=min(10, X.shape[1])),
SVC(probability=True)
)


param_grid = {
'svc__C': [0.1, 1, 10],
'svc__kernel': ['rbf'],
'svc__gamma': ['scale', 'auto']
}


cv = GroupKFold(n_splits=5)


gs = GridSearchCV(pipe, param_grid, cv=cv.split(X,y,groups), scoring='f1_macro', n_jobs=-1, verbose=2)
print('Starting GridSearch...')
gs.fit(X,y)
print('Best params:', gs.best_params_)
print('Done. Reports saved in D:/pipeline_SVM/results and previews.')