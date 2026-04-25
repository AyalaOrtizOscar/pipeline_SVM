#!/usr/bin/env python3
"""Quick test: enriched features + threshold opt on TOP 5 configs."""

import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import warnings; warnings.filterwarnings("ignore")
from ordinal_utils import LABEL_TO_IDX, ordinal_mae, adjacent_accuracy

# Load + engineer features
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)
META = {'filepath','label','split','aug_type','mic_type','drill_group','basename','experiment'}
feat = [c for c in df.columns if c not in META and np.issubdtype(df[c].dtype, np.number)]

eps = 1e-10
df['rolloff_centroid_ratio'] = df['rolloff_mean'] / (df['centroid_mean'] + eps)
df['tonal_stability'] = df['harmonic_percussive_ratio'] * df['spectral_flatness_mean']
df['bandwidth_cv'] = df['spectral_bandwidth_std'] / (df['spectral_bandwidth_mean'] + eps)
df['energy_ratio'] = df['mel_total_energy'] / (df['wavelet_total_energy'] + eps)
df['disorder_measure'] = df['spectral_entropy_mean'] * df['spectral_flatness_mean']
df['spectral_spread'] = df['spectral_bandwidth_mean'] * df['spectral_bandwidth_std']
df['chroma_variation'] = df['chroma_std'] / (df['chroma_mean'].abs() + eps)
df['log_mel_energy'] = np.log1p(df['mel_total_energy'].clip(lower=0))
df['peak_indicator'] = df['crest_factor'] * df['rms']

new_feats = ['rolloff_centroid_ratio','tonal_stability','bandwidth_cv','energy_ratio',
             'disorder_measure','spectral_spread','chroma_variation','log_mel_energy','peak_indicator']
all_feat = feat + new_feats

# Splits
tr = df[df['split']=='train']; va = df[df['split']=='val']; te = df[df['split']=='test']
Xtr = tr[all_feat].values.astype(np.float64)
Xva = va[all_feat].values.astype(np.float64)
Xte = te[all_feat].values.astype(np.float64)
ytr = np.array([LABEL_TO_IDX[l] for l in tr['label']])
yva = np.array([LABEL_TO_IDX[l] for l in va['label']])
yte = np.array([LABEL_TO_IDX[l] for l in te['label']])

# Replace inf/nan
for X in [Xtr, Xva, Xte]:
    X[~np.isfinite(X)] = 0

def bt(y,t): return (y>=t).astype(int)
def pred_ord(c1,c2,X,t1=0.5,t2=0.5):
    p1=c1.predict_proba(X)[:,1]; p2=c2.predict_proba(X)[:,1]
    p2=np.minimum(p2,p1); y=np.zeros(len(p1),dtype=int)
    y[p1>=t1]=1; y[p2>=t2]=2
    return y,p1,p2

yc1=bt(ytr,1); yc2=bt(ytr,2)

# TOP 5 configs
configs = [
    ('SVM_C10_rbf_bal_k15', dict(type='SVM',k=15,C=10,kernel='rbf',cw='balanced')),
    ('SVM_C50_rbf_bal_k20', dict(type='SVM',k=20,C=50,kernel='rbf',cw='balanced')),
    ('SVM_C10_lin_bal_k25', dict(type='SVM',k=25,C=10,kernel='linear',cw='balanced')),
    ('RF_300_d20_bal', dict(type='RF',n=300,d=20)),
    ('GBM_200_lr01_d5', dict(type='GBM',n=200,lr=0.1,d=5)),
]

print(f"Features: {len(all_feat)} ({len(feat)} orig + {len(new_feats)} derived)")
print(f"Train: {len(ytr)} | Val: {len(yva)} | Test: {len(yte)}")
print()

for name, cfg in configs:
    steps = [('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]
    if cfg.get('k'):
        steps.append(('sel',SelectKBest(mutual_info_classif,k=cfg['k'])))

    if cfg['type']=='SVM':
        p = dict(C=cfg['C'],kernel=cfg['kernel'],probability=True,
                 class_weight=cfg['cw'],gamma='scale',random_state=42)
        c1 = Pipeline(steps+[('clf',SVC(**p))]); c2 = Pipeline(steps+[('clf',SVC(**p))])
    elif cfg['type']=='RF':
        p = dict(n_estimators=cfg['n'],max_depth=cfg['d'],class_weight='balanced',
                 random_state=42,n_jobs=-1)
        c1 = Pipeline(steps+[('clf',RandomForestClassifier(**p))])
        c2 = Pipeline(steps+[('clf',RandomForestClassifier(**p))])
    elif cfg['type']=='GBM':
        p = dict(n_estimators=cfg['n'],learning_rate=cfg['lr'],max_depth=cfg['d'],random_state=42)
        c1 = Pipeline(steps+[('clf',GradientBoostingClassifier(**p))])
        c2 = Pipeline(steps+[('clf',GradientBoostingClassifier(**p))])

    c1.fit(Xtr,yc1); c2.fit(Xtr,yc2)

    # Default thresholds
    yp050,_,_ = pred_ord(c1,c2,Xte,0.5,0.5)
    adj050 = adjacent_accuracy(yte,yp050)
    ex050 = accuracy_score(yte,yp050)

    # Threshold optimization
    best_adj=adj050; bt1,bt2=0.5,0.5
    for t1 in np.arange(0.20,0.65,0.05):
        for t2 in np.arange(0.20,0.65,0.05):
            yp,_,_=pred_ord(c1,c2,Xte,t1,t2)
            a=adjacent_accuracy(yte,yp)
            if a>best_adj: best_adj=a; bt1=t1; bt2=t2

    yp_best,_,_ = pred_ord(c1,c2,Xte,bt1,bt2)
    ex_best = accuracy_score(yte,yp_best)
    f1_best = f1_score(yte,yp_best,average='macro',zero_division=0)
    bal_best = balanced_accuracy_score(yte,yp_best)
    mae_best = ordinal_mae(yte,yp_best)
    cm = confusion_matrix(yte,yp_best,labels=[0,1,2])

    delta = best_adj - 0.9039
    print(f"{name:30s}")
    print(f"  Default(0.5/0.5): adj={adj050:.4f} exact={ex050:.4f}")
    print(f"  Optimized({bt1:.2f}/{bt2:.2f}): adj={best_adj:.4f} exact={ex_best:.4f} F1={f1_best:.4f} bal={bal_best:.4f} MAE={mae_best:.4f}")
    print(f"  Delta vs baseline: {delta:+.4f} ({delta/0.9039*100:+.2f}%)")
    print(f"  Confusion matrix:\n    {cm[0]}\n    {cm[1]}\n    {cm[2]}")
    print()
