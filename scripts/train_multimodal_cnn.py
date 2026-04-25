#!/usr/bin/env python3
"""
Multimodal CNN — Audio NI + ESP32 features + Caudal YF-S201.
Arquitectura: CNN-B (frozen extractor) + MLP ESP32 + MLP Caudal -> fusión -> ordinal head.
3 experimentos: A (audio 4381), B (audio subset 510), C (multimodal 510).
"""
import argparse, os, json, time
import numpy as np, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--npz',    default='/storage/mel_dataset.npz')
parser.add_argument('--meta',   default='/storage/mel_dataset_meta.csv')
parser.add_argument('--out',    default='/storage/results_multimodal/')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch',  type=int, default=32)
parser.add_argument('--lr',     type=float, default=1e-3)
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=="cuda" else "none"}')

# ─── load data ────────────────────────────────────────────────────────────────
print('Loading data...')
data  = np.load(args.npz)
X_mel = data['X']   # (N, 64, 512)
y_all = data['y']
meta  = pd.read_csv(args.meta)
assert len(X_mel) == len(meta)
print(f'  Total: {len(X_mel)} samples')

# Load full CSV for ESP32/flow features
csv_path = '/storage/mel_dataset_meta.csv'
feat_csv = '/storage/features_multimodal_merged.csv'

# Try to load multimodal features
ESP32_COLS = ['esp_rms', 'esp_rms_db', 'esp_centroid_mean', 'esp_zcr',
              'esp_crest_factor', 'esp_spectral_contrast_mean']
FLOW_COLS  = ['flow_mean_lmin', 'flow_std_lmin', 'flow_cv', 'flow_duty_pulses']

if os.path.exists(feat_csv):
    df_full = pd.read_csv(feat_csv)
    print(f'  Multimodal CSV: {len(df_full)} rows')
    has_esp32 = df_full[ESP32_COLS[0]].notna() if ESP32_COLS[0] in df_full.columns else pd.Series(False, index=df_full.index)
    has_flow  = df_full[FLOW_COLS[0]].notna()  if FLOW_COLS[0]  in df_full.columns else pd.Series(False, index=df_full.index)
    has_multi = has_esp32 & has_flow
    print(f'  Samples with ESP32+flow: {has_multi.sum()} ({has_multi.mean()*100:.1f}%)')
    multimodal_available = True
else:
    print('  Multimodal CSV not found — running audio-only experiments')
    df_full = meta.copy()
    multimodal_available = False

# ─── model definitions ────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)) if stride != 1 or in_c != out_c else nn.Identity()
    def forward(self, x): return self.net(x) + self.skip(x)

class CnnEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.blocks = nn.Sequential(
            ConvBlock(32,64,2), ConvBlock(64,128,2), ConvBlock(128,256,2))
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        return self.drop(self.gap(x).flatten(1))   # (B, 256)

class MultimodalCNN(nn.Module):
    def __init__(self, n_esp32=6, n_flow=4):
        super().__init__()
        self.audio_enc = CnnEncoder()
        self.esp_mlp = nn.Sequential(
            nn.Linear(n_esp32, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 64),     nn.ReLU())
        self.flow_mlp = nn.Sequential(
            nn.Linear(n_flow, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 32),     nn.ReLU())
        fusion_dim = 256 + 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(128, 64),         nn.ReLU())
        self.head = nn.Linear(64, 2)

    def forward(self, mel, esp_feat, flow_feat):
        a = self.audio_enc(mel)
        e = self.esp_mlp(esp_feat)
        f = self.flow_mlp(flow_feat)
        x = torch.cat([a, e, f], dim=1)
        return self.head(self.fusion(x))

class AudioOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_enc = CnnEncoder()
        self.head = nn.Linear(256, 2)
    def forward(self, mel, *args):
        return self.head(self.audio_enc(mel))

# ─── losses & metrics ────────────────────────────────────────────────────────
def frank_hall_loss(logits, labels, smooth=0.05):
    t1 = (labels >= 1).float() * (1-smooth) + smooth/2
    t2 = (labels >= 2).float() * (1-smooth) + smooth/2
    return (nn.functional.binary_cross_entropy_with_logits(logits[:,0], t1) +
            nn.functional.binary_cross_entropy_with_logits(logits[:,1], t2))

def decode(logits):
    p1 = torch.sigmoid(logits[:,0])
    p2 = torch.minimum(torch.sigmoid(logits[:,1]), p1)
    return torch.stack([1-p1, p1-p2, p2], dim=1).argmax(dim=1)

def adj_acc(yt, yp): return float(np.mean(np.abs(np.array(yt)-np.array(yp)) <= 1))

# ─── dataset ─────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, X, y, esp=None, flow=None, augment=False):
        self.X    = torch.from_numpy(X).unsqueeze(1)
        self.y    = torch.from_numpy(y.astype(np.int64))
        self.esp  = torch.from_numpy(esp.astype(np.float32))  if esp  is not None else torch.zeros(len(X), 6)
        self.flow = torch.from_numpy(flow.astype(np.float32)) if flow is not None else torch.zeros(len(X), 4)
        self.aug  = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.aug:
            if torch.rand(1) > 0.4:
                f0, f = torch.randint(0,20,(1,)).item(), torch.randint(3,16,(1,)).item()
                x[:, f0:min(f0+f,64), :] = 0
            if torch.rand(1) > 0.4:
                t0, t = torch.randint(0,80,(1,)).item(), torch.randint(20,80,(1,)).item()
                x[:, :, t0:min(t0+t,512)] = 0
            x += torch.randn_like(x) * 0.015
        return x, self.esp[i], self.flow[i], self.y[i]

# ─── training function ────────────────────────────────────────────────────────
def train_experiment(name, model, idx_tr, idx_te, X_mel, y_all,
                     esp_arr=None, flow_arr=None, epochs=args.epochs):
    print(f'\n{"="*70}')
    print(f'Experiment {name}: train={len(idx_tr)}, test={len(idx_te)}')

    esp_tr  = esp_arr[idx_tr]  if esp_arr  is not None else None
    esp_te  = esp_arr[idx_te]  if esp_arr  is not None else None
    flow_tr = flow_arr[idx_tr] if flow_arr is not None else None
    flow_te = flow_arr[idx_te] if flow_arr is not None else None

    ds_tr = MelDataset(X_mel[idx_tr], y_all[idx_tr], esp_tr, flow_tr, augment=True)
    ds_te = MelDataset(X_mel[idx_te], y_all[idx_te], esp_te, flow_te, augment=False)

    y_tr = y_all[idx_tr]
    class_counts = np.bincount(y_tr)
    weights = (1.0 / class_counts)[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False,   num_workers=2, pin_memory=True)

    model = model.to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    log_rows = []
    best_f1, best_adj = 0.0, 0.0
    best_path = os.path.join(args.out, f'best_{name}.pt')

    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for mel, esp, flow, yb in dl_tr:
            mel, esp, flow, yb = mel.to(device), esp.to(device), flow.to(device), yb.to(device)
            opt.zero_grad()
            loss = frank_hall_loss(model(mel, esp, flow), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(mel)
        tr_loss /= len(ds_tr)
        sched.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for mel, esp, flow, yb in dl_te:
                p = decode(model(mel.to(device), esp.to(device), flow.to(device))).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())

        adj  = adj_acc(labels, preds)
        ex   = float(np.mean(np.array(preds)==np.array(labels)))
        f1   = f1_score(labels, preds, average='macro', zero_division=0)
        elapsed = time.time() - t0

        flag = ''
        if f1 > best_f1:
            best_f1, best_adj = f1, adj
            torch.save(model.state_dict(), best_path)
            flag = ' <- BEST'

        log_rows.append(dict(epoch=epoch, loss=round(tr_loss,4), adj=round(adj,4),
                             exact=round(ex,4), f1=round(f1,4), t=round(elapsed,1)))
        print(f'  Ep{epoch:03d} loss={tr_loss:.4f} adj={adj:.4f} exact={ex:.4f} F1={f1:.4f} {elapsed:.1f}s{flag}')
        pd.DataFrame(log_rows).to_csv(os.path.join(args.out, f'log_{name}.csv'), index=False)

    # Final eval with best model
    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for mel, esp, flow, yb in dl_te:
            p = decode(model(mel.to(device), esp.to(device), flow.to(device))).cpu().numpy()
            preds.extend(p); labels.extend(yb.numpy())

    cm  = confusion_matrix(labels, preds)
    f1c = f1_score(labels, preds, average=None, zero_division=0)
    result = {
        'experiment': name,
        'n_train': len(idx_tr), 'n_test': len(idx_te),
        'adjacent_accuracy': round(adj_acc(labels, preds), 4),
        'exact_accuracy':    round(float(np.mean(np.array(preds)==np.array(labels))), 4),
        'f1_macro':          round(f1_score(labels, preds, average='macro', zero_division=0), 4),
        'f1_per_class': {'sin_desgaste': round(float(f1c[0]),4) if len(f1c)>0 else 0,
                         'medianamente': round(float(f1c[1]),4) if len(f1c)>1 else 0,
                         'desgastado':   round(float(f1c[2]),4) if len(f1c)>2 else 0},
        'confusion_matrix': cm.tolist(),
    }
    print(f'\n  FINAL {name}: adj={result["adjacent_accuracy"]}  F1={result["f1_macro"]}')
    return result, model, labels, preds

# ─── prepare splits ───────────────────────────────────────────────────────────
# Split for multimodal (Art.2 only — no E3 overlap)
if multimodal_available:
    # Build index mapping from meta to df_full
    # Align by experiment+test_id+label
    esp_cols_avail  = [c for c in ESP32_COLS if c in df_full.columns]
    flow_cols_avail = [c for c in FLOW_COLS  if c in df_full.columns]
    print(f'\nESP32 cols: {esp_cols_avail}')
    print(f'Flow cols:  {flow_cols_avail}')

    # Get multimodal subset indices from meta
    multi_exps = ['art2_test39', 'art2_test50', 'art2_test53']
    multi_mask = meta['experiment'].isin(multi_exps)
    print(f'Multimodal experiments: {meta[multi_mask]["experiment"].value_counts().to_dict()}')

    # Build ESP32 and flow arrays aligned to meta index
    n = len(meta)
    esp_arr_full  = np.zeros((n, len(esp_cols_avail)),  dtype=np.float32)
    flow_arr_full = np.zeros((n, len(flow_cols_avail)), dtype=np.float32)

    # Match meta rows to df_full by experiment+test_id
    for i, (_, row) in enumerate(meta.iterrows()):
        exp = row.get('experiment', '')
        tid = str(row.get('test_id', ''))
        matches = df_full[
            (df_full.get('experiment', pd.Series()) == exp) |
            (df_full.get('test_id', pd.Series()).astype(str) == tid)
        ] if 'experiment' in df_full.columns else pd.DataFrame()

        if len(matches) > 0 and len(esp_cols_avail) > 0:
            vals = matches.iloc[0][esp_cols_avail].values.astype(float)
            esp_arr_full[i] = np.nan_to_num(vals, nan=0.0)
        if len(matches) > 0 and len(flow_cols_avail) > 0:
            vals = matches.iloc[0][flow_cols_avail].values.astype(float)
            flow_arr_full[i] = np.nan_to_num(vals, nan=0.0)

    # Normalize
    imp_esp  = SimpleImputer(strategy='median')
    imp_flow = SimpleImputer(strategy='median')
    scaler_esp  = StandardScaler()
    scaler_flow = StandardScaler()

    esp_arr_full  = scaler_esp.fit_transform(imp_esp.fit_transform(esp_arr_full))
    flow_arr_full = scaler_flow.fit_transform(imp_flow.fit_transform(flow_arr_full))

    esp_arr_full  = esp_arr_full.astype(np.float32)
    flow_arr_full = flow_arr_full.astype(np.float32)
else:
    esp_cols_avail  = ['esp_rms','esp_rms_db','esp_centroid_mean','esp_zcr','esp_crest_factor','esp_spectral_contrast_mean']
    flow_cols_avail = ['flow_mean_lmin','flow_std_lmin','flow_cv','flow_duty_pulses']
    n = len(meta)
    esp_arr_full  = np.zeros((n, len(esp_cols_avail)),  dtype=np.float32)
    flow_arr_full = np.zeros((n, len(flow_cols_avail)), dtype=np.float32)

# Experiment A: Audio only, full dataset, E3 holdout
hold_mask = meta['experiment'] == 'E3'
aug_mask  = meta['mic_type'] == 'augmented_wavs'
train_mask = ~hold_mask & (meta['mic_type'] != 'augmented_wavs')
exp_counts = meta['experiment'].value_counts()
tiny_exps  = exp_counts[exp_counts < 10].index
train_mask &= ~meta['experiment'].isin(tiny_exps)

idx_A_tr = np.where(train_mask | aug_mask)[0]
idx_A_te = np.where(hold_mask)[0]

# Experiments B & C: multimodal subset, leave-one-out by drill
# Use test39 as test, test50+test53 as train (temporal order)
multi_mask = meta['experiment'].isin(['art2_test39', 'art2_test50', 'art2_test53'])
mask_39 = meta['experiment'] == 'art2_test39'
mask_other = meta['experiment'].isin(['art2_test50', 'art2_test53'])

idx_multi_tr = np.where(mask_other)[0]
idx_multi_te = np.where(mask_39)[0]

print(f'\nExperiment A: train={len(idx_A_tr)}, test={len(idx_A_te)} (E3 holdout)')
print(f'Experiments B/C: train={len(idx_multi_tr)}, test={len(idx_multi_te)} (test39)')

# ─── run experiments ─────────────────────────────────────────────────────────
results = {}

# Exp A — Audio only, full dataset
res_A, _, _, _ = train_experiment(
    'A_audio_full', AudioOnlyCNN(),
    idx_A_tr, idx_A_te, X_mel, y_all, epochs=args.epochs)
results['A'] = res_A

# Exp B — Audio only, multimodal subset
res_B, _, _, _ = train_experiment(
    'B_audio_subset', AudioOnlyCNN(),
    idx_multi_tr, idx_multi_te, X_mel, y_all, epochs=args.epochs)
results['B'] = res_B

# Exp C — Full multimodal
res_C, _, _, _ = train_experiment(
    'C_multimodal', MultimodalCNN(len(esp_cols_avail), len(flow_cols_avail)),
    idx_multi_tr, idx_multi_te, X_mel, y_all,
    esp_arr=esp_arr_full, flow_arr=flow_arr_full, epochs=args.epochs)
results['C'] = res_C

# ─── comparison report ────────────────────────────────────────────────────────
print('\n' + '='*70)
print('COMPARISON SUMMARY')
print('='*70)
header = f"{'Experiment':<25} {'Adj Acc':>8} {'Exact':>8} {'F1 Macro':>10} {'F1 deg':>8}"
print(header)
print('-'*60)
for k, r in results.items():
    print(f"  {r['experiment']:<23} {r['adjacent_accuracy']:>8.4f} {r['exact_accuracy']:>8.4f} {r['f1_macro']:>10.4f} {r['f1_per_class']['desgastado']:>8.4f}")

delta_bc_f1  = results['C']['f1_macro']          - results['B']['f1_macro']
delta_bc_adj = results['C']['adjacent_accuracy']  - results['B']['adjacent_accuracy']
delta_bc_deg = results['C']['f1_per_class']['desgastado'] - results['B']['f1_per_class']['desgastado']
print(f'\nDelta C-B (multimodal contribution):')
print(f'  F1 macro:    {delta_bc_f1:+.4f}')
print(f'  Adj acc:     {delta_bc_adj:+.4f}')
print(f'  F1 desgastado: {delta_bc_deg:+.4f}')

final = {
    'experiments': results,
    'multimodal_delta': {
        'f1_macro': round(delta_bc_f1, 4),
        'adj_acc':  round(delta_bc_adj, 4),
        'f1_desgastado': round(delta_bc_deg, 4),
    },
    'svm_baseline_adj': 0.901,
    'conclusion': 'Multimodal (audio+ESP32+flow) vs audio-only in same subset'
}
with open(os.path.join(args.out, 'comparison_report.json'), 'w') as f:
    json.dump(final, f, indent=2)

# ─── comparison plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
names = ['A: Audio\nfull (4381)', 'B: Audio\nsubset (510)', 'C: Multimodal\n(510)']
colors = ['#4C72B0', '#DD8452', '#55A868']

for ax_i, (metric, title) in enumerate([
    ('adjacent_accuracy', 'Adjacent Accuracy'),
    ('f1_macro',          'F1 Macro'),
    ('exact_accuracy',    'Exact Accuracy')]):
    ax = axes[ax_i]
    vals = [results[k][metric] for k in ['A','B','C']]
    bars = ax.bar(names, vals, color=colors, alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.901, color='gray', linestyle='--', alpha=0.7, label='SVM baseline')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2., v+0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if ax_i == 0: ax.legend(fontsize=8)

plt.suptitle('Multimodal CNN Ablation: Audio NI vs Audio+ESP32+Caudal', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(args.out, 'ablation_comparison.png'), dpi=150, bbox_inches='tight')
print(f'\nPlot saved: {args.out}/ablation_comparison.png')
print('ALL DONE.')
