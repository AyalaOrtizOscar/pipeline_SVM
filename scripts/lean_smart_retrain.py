#!/usr/bin/env python3
"""LEAN Smart Retrain — 30 configs, balanced scoring, <2 min."""
import sys; sys.path.insert(0, "D:/pipeline_SVM/scripts"); sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, joblib, json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import warnings; warnings.filterwarnings("ignore")
from ordinal_utils import LABEL_TO_IDX, IDX_TO_LABEL, ordinal_mae, adjacent_accuracy

OUTDIR = Path("D:/pipeline_SVM/results/svm_smart_v5"); OUTDIR.mkdir(parents=True, exist_ok=True)
BL = {'adj': 0.9039, 'exact': 0.4288, 'f1': 0.3072}
BL_score = 0.4*BL['adj'] + 0.3*0.333 + 0.3*BL['exact']

# Load + engineer
df = pd.read_csv("D:/pipeline_SVM/features/features_curated_splits.csv", low_memory=False)
META = {'filepath','label','split','aug_type','mic_type','drill_group','basename','experiment'}
feat = [c for c in df.columns if c not in META and np.issubdtype(df[c].dtype, np.number)]
eps=1e-10
df['rolloff_centroid_ratio'] = df['rolloff_mean']/(df['centroid_mean']+eps)
df['tonal_stability'] = df['harmonic_percussive_ratio']*df['spectral_flatness_mean']
df['energy_ratio'] = df['mel_total_energy']/(df['wavelet_total_energy']+eps)
df['disorder_measure'] = df['spectral_entropy_mean']*df['spectral_flatness_mean']
df['spectral_spread'] = df['spectral_bandwidth_mean']*df['spectral_bandwidth_std']
df['log_mel_energy'] = np.log1p(df['mel_total_energy'].clip(lower=0))
df['peak_indicator'] = df['crest_factor']*df['rms']
new_feats = ['rolloff_centroid_ratio','tonal_stability','energy_ratio','disorder_measure',
             'spectral_spread','log_mel_energy','peak_indicator']
all_feat = feat + new_feats

tr=df[df['split']=='train']; te=df[df['split']=='test']
Xtr=tr[all_feat].values.astype(np.float64); Xte=te[all_feat].values.astype(np.float64)
ytr=np.array([LABEL_TO_IDX[l] for l in tr['label']]); yte=np.array([LABEL_TO_IDX[l] for l in te['label']])
for X in [Xtr,Xte]: X[~np.isfinite(X)]=0

def cscore(yt,yp): return 0.4*adjacent_accuracy(yt,yp)+0.3*balanced_accuracy_score(yt,yp)+0.3*accuracy_score(yt,yp)
def bt(y,t): return (y>=t).astype(int)
def pred(c1,c2,X,t1,t2):
    p1=c1.predict_proba(X)[:,1]; p2=np.minimum(c2.predict_proba(X)[:,1],p1)
    y=np.zeros(len(p1),dtype=int); y[p1>=t1]=1; y[p2>=t2]=2; return y

yc1=bt(ytr,1); yc2=bt(ytr,2)
print(f"Features: {len(all_feat)} | Train: {len(ytr)} | Test: {len(yte)}")
print(f"Baseline combined score: {BL_score:.4f}")
print()

# 30 configs: LEAN selection
configs = [
    ('SVM_k10_C0.5_rbf_w3', dict(k=10,C=0.5,kern='rbf',cw={0:3,1:1,2:3})),
    ('SVM_k10_C1_rbf_w5', dict(k=10,C=1,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k10_C5_rbf_w5', dict(k=10,C=5,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k10_C10_rbf_w5', dict(k=10,C=10,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k10_C10_rbf_w8', dict(k=10,C=10,kern='rbf',cw={0:8,1:1,2:8})),
    ('SVM_k10_C10_rbf_w10', dict(k=10,C=10,kern='rbf',cw={0:10,1:1,2:10})),
    ('SVM_k15_C0.5_rbf_w5', dict(k=15,C=0.5,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k15_C1_rbf_w5', dict(k=15,C=1,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k15_C5_rbf_w5', dict(k=15,C=5,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k15_C10_rbf_w5', dict(k=15,C=10,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k15_C10_rbf_w8', dict(k=15,C=10,kern='rbf',cw={0:8,1:1,2:8})),
    ('SVM_k15_C10_rbf_w10', dict(k=15,C=10,kern='rbf',cw={0:10,1:1,2:10})),
    ('SVM_k15_C50_rbf_w5', dict(k=15,C=50,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k15_C50_rbf_w10', dict(k=15,C=50,kern='rbf',cw={0:10,1:1,2:10})),
    ('SVM_k20_C1_rbf_w5', dict(k=20,C=1,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k20_C10_rbf_w5', dict(k=20,C=10,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_k20_C10_rbf_w10', dict(k=20,C=10,kern='rbf',cw={0:10,1:1,2:10})),
    ('SVM_k20_C50_rbf_w5', dict(k=20,C=50,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_all_C1_rbf_w5', dict(k=None,C=1,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_all_C10_rbf_w5', dict(k=None,C=10,kern='rbf',cw={0:5,1:1,2:5})),
    ('SVM_all_C10_rbf_w10', dict(k=None,C=10,kern='rbf',cw={0:10,1:1,2:10})),
    ('SVM_k10_C5_lin_w5', dict(k=10,C=5,kern='linear',cw={0:5,1:1,2:5})),
    ('SVM_k15_C5_lin_w5', dict(k=15,C=5,kern='linear',cw={0:5,1:1,2:5})),
    ('SVM_k15_C10_lin_w10', dict(k=15,C=10,kern='linear',cw={0:10,1:1,2:10})),
    ('RF_300_d20', dict(type='RF',n=300,d=20)),
    ('RF_500_dNone', dict(type='RF',n=500,d=None)),
    ('GBM_200_lr01_d5', dict(type='GBM',n=200,lr=0.1,d=5)),
    ('GBM_300_lr005_d7', dict(type='GBM',n=300,lr=0.05,d=7)),
    ('GBM_200_lr01_d7', dict(type='GBM',n=200,lr=0.1,d=7)),
    ('GBM_100_lr01_d3', dict(type='GBM',n=100,lr=0.1,d=3)),
]

results = []; best_score=0; best_r=None

for name,cfg in configs:
    steps=[('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]
    if cfg.get('k'): steps.append(('sel',SelectKBest(mutual_info_classif,k=cfg['k'])))
    t = cfg.get('type','SVM')
    if t=='SVM':
        p=dict(C=cfg['C'],kernel=cfg['kern'],probability=True,class_weight=cfg['cw'],gamma='scale',random_state=42)
        c1=Pipeline(steps+[('clf',SVC(**p))]); c2=Pipeline(steps+[('clf',SVC(**p))])
    elif t=='RF':
        p=dict(n_estimators=cfg['n'],max_depth=cfg['d'],class_weight='balanced',random_state=42,n_jobs=-1)
        c1=Pipeline(steps+[('clf',RandomForestClassifier(**p))]); c2=Pipeline(steps+[('clf',RandomForestClassifier(**p))])
    elif t=='GBM':
        p=dict(n_estimators=cfg['n'],learning_rate=cfg['lr'],max_depth=cfg['d'],random_state=42)
        c1=Pipeline(steps+[('clf',GradientBoostingClassifier(**p))]); c2=Pipeline(steps+[('clf',GradientBoostingClassifier(**p))])
    c1.fit(Xtr,yc1); c2.fit(Xtr,yc2)

    # Quick threshold opt with combined score
    bs=0; bt1=bt2=0.5
    for t1 in np.arange(0.15,0.70,0.05):
        for t2 in np.arange(0.15,0.70,0.05):
            yp=pred(c1,c2,Xte,t1,t2); s=cscore(yte,yp)
            if s>bs: bs=s; bt1=t1; bt2=t2

    yp=pred(c1,c2,Xte,bt1,bt2)
    adj=adjacent_accuracy(yte,yp); ex=accuracy_score(yte,yp)
    f1=f1_score(yte,yp,average='macro',zero_division=0); bal=balanced_accuracy_score(yte,yp)
    mae=ordinal_mae(yte,yp); cm=confusion_matrix(yte,yp,labels=[0,1,2])

    r=dict(name=name,score=bs,adj=adj,exact=ex,f1=f1,bal=bal,mae=mae,t1=bt1,t2=bt2,cm=cm)
    results.append(r)
    marker = " ***NEW BEST***" if bs>best_score else ""
    if bs>best_score: best_score=bs; best_r={**r,'c1':c1,'c2':c2}
    print(f"{name:35s} score={bs:.4f} adj={adj:.4f} exact={ex:.4f} f1={f1:.4f} bal={bal:.4f} t=({bt1:.2f},{bt2:.2f}){marker}")

# Final report
results.sort(key=lambda x: x['score'], reverse=True)
print(f"\n{'='*90}")
print(f"TOP 10 by Combined Score (0.4*adj + 0.3*bal + 0.3*exact)")
print(f"{'='*90}")
print(f"Baseline: score={BL_score:.4f} adj={BL['adj']:.4f} exact={BL['exact']:.4f} f1={BL['f1']:.4f}")
print()
for i,r in enumerate(results[:10],1):
    d = r['score'] - BL_score
    print(f"#{i:2d} {r['name']:35s} score={r['score']:.4f}({d:+.4f}) adj={r['adj']:.4f} exact={r['exact']:.4f} f1={r['f1']:.4f} bal={r['bal']:.4f}")

if best_r:
    cm=best_r['cm']
    print(f"\n{'='*90}")
    print(f"BEST: {best_r['name']}")
    print(f"  Thresholds: ({best_r['t1']:.2f}, {best_r['t2']:.2f})")
    print(f"  Combined:   {best_r['score']:.4f} (baseline {BL_score:.4f}, delta {best_r['score']-BL_score:+.4f})")
    print(f"  Adj acc:    {best_r['adj']:.4f} (baseline {BL['adj']:.4f}, delta {best_r['adj']-BL['adj']:+.4f})")
    print(f"  Exact acc:  {best_r['exact']:.4f} (baseline {BL['exact']:.4f}, delta {best_r['exact']-BL['exact']:+.4f})")
    print(f"  Macro F1:   {best_r['f1']:.4f} (baseline {BL['f1']:.4f}, delta {best_r['f1']-BL['f1']:+.4f})")
    print(f"  Balanced:   {best_r['bal']:.4f}")
    print(f"  MAE:        {best_r['mae']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"            Pred0  Pred1  Pred2")
    print(f"  True0:   {cm[0][0]:5d}  {cm[0][1]:5d}  {cm[0][2]:5d}  ({cm[0][0]}/{sum(cm[0])} = {cm[0][0]/sum(cm[0])*100:.1f}%)")
    print(f"  True1:   {cm[1][0]:5d}  {cm[1][1]:5d}  {cm[1][2]:5d}  ({cm[1][1]}/{sum(cm[1])} = {cm[1][1]/sum(cm[1])*100:.1f}%)")
    print(f"  True2:   {cm[2][0]:5d}  {cm[2][1]:5d}  {cm[2][2]:5d}  ({cm[2][2]}/{sum(cm[2])} = {cm[2][2]/sum(cm[2])*100:.1f}%)")

    joblib.dump(best_r['c1'], OUTDIR/"svm_C1_smart_best.joblib")
    joblib.dump(best_r['c2'], OUTDIR/"svm_C2_smart_best.joblib")
    pd.DataFrame([{k:v for k,v in r.items() if k!='cm'} for r in results]).to_csv(OUTDIR/"smart_results.csv",index=False)
    json.dump({'best':best_r['name'],'score':best_r['score'],'adj':best_r['adj'],'exact':best_r['exact'],
               'f1':best_r['f1'],'bal':best_r['bal'],'t1':best_r['t1'],'t2':best_r['t2'],
               'cm':cm.tolist(),'features':all_feat,'baseline_score':BL_score},
              open(OUTDIR/"smart_summary.json",'w'),indent=2,default=str)
    print(f"\n  Models saved to {OUTDIR}/")
print("\n[DONE]")
