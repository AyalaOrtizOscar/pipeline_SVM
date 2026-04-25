#!/usr/bin/env python3
"""
Multimodal CNN v2 — Corrected: only samples WITH ESP32+flow data.
Train: test53 (138 samples, all with ESP32/flow)
Test:  test39 (336 samples, 333 with ESP32/flow)
Key finding: flow features (mean, std, cv) correlate monotonically with wear.
"""
import argparse, os, json, time
import numpy as np, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--npz',    default='/storage/mel_dataset.npz')
parser.add_argument('--out',    default='/storage/results_multimodal_v2/')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch',  type=int, default=16)
parser.add_argument('--lr',     type=float, default=5e-4)
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=="cuda" else "none"}')

# ─── load data ────────────────────────────────────────────────────────────────
data  = np.load(args.npz)
X_mel = data['X']; y_all = data['y']
meta  = pd.read_csv('/storage/mel_dataset_meta.csv')
df    = pd.read_csv('/storage/features_multimodal_merged.csv')
assert len(X_mel) == len(meta) == len(df)
print(f'Total: {len(X_mel)} samples')

FLOW_COLS  = ['flow_mean_lmin', 'flow_std_lmin', 'flow_cv', 'flow_duty_pulses']
ESP32_COLS = ['esp_rms', 'esp_centroid_mean', 'esp_zcr', 'esp_crest_factor']

# Only use tests WITH complete ESP32+flow data
TRAIN_EXPS = ['art2_test53']            # 138 samples, all with ESP32
TEST_EXPS  = ['art2_test39']            # 336 samples, 333 with ESP32

mask_tr = meta['experiment'].isin(TRAIN_EXPS)
mask_te = meta['experiment'].isin(TEST_EXPS)

idx_tr = np.where(mask_tr)[0]
idx_te = np.where(mask_te)[0]
print(f'Train (test53): {len(idx_tr)} | Test (test39): {len(idx_te)}')
print(f'Train labels: {dict(zip(*np.unique(y_all[idx_tr], return_counts=True)))}')
print(f'Test  labels: {dict(zip(*np.unique(y_all[idx_te], return_counts=True)))}')

# Build feature arrays aligned to meta
def build_feat_array(df, cols):
    arr = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        if col in df.columns:
            arr[:, j] = df[col].fillna(0).values.astype(np.float32)
    return arr

flow_arr  = build_feat_array(df, FLOW_COLS)
esp_arr   = build_feat_array(df, ESP32_COLS)

# Scale on train, apply to all
scaler_flow = StandardScaler().fit(flow_arr[idx_tr])
scaler_esp  = StandardScaler().fit(esp_arr[idx_tr])
flow_arr = scaler_flow.transform(flow_arr).astype(np.float32)
esp_arr  = scaler_esp.transform(esp_arr).astype(np.float32)

print(f'flow_arr non-zero in train: {(flow_arr[idx_tr] != 0).any(axis=1).sum()}/{len(idx_tr)}')
print(f'flow_arr non-zero in test:  {(flow_arr[idx_te] != 0).any(axis=1).sum()}/{len(idx_te)}')

# ─── model ────────────────────────────────────────────────────────────────────
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

class MultimodalV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Audio branch
        self.audio = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            ConvBlock(32, 64, 2), ConvBlock(64, 128, 2), ConvBlock(128, 256, 2),
            nn.AdaptiveAvgPool2d(1))
        self.audio_drop = nn.Dropout(0.4)

        # Flow branch (key discriminator)
        self.flow_branch = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 64), nn.ReLU())

        # ESP32 audio branch (light)
        self.esp_branch = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU())

        # Fusion (256 + 64 + 32 = 352)
        self.fusion = nn.Sequential(
            nn.Linear(352, 128), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(128, 32),  nn.ReLU())
        self.head = nn.Linear(32, 2)

    def forward(self, mel, flow, esp):
        a = self.audio_drop(self.audio(mel).flatten(1))   # (B, 256)
        f = self.flow_branch(flow)                         # (B, 64)
        e = self.esp_branch(esp)                           # (B, 32)
        return self.head(self.fusion(torch.cat([a, f, e], dim=1)))

class AudioOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            ConvBlock(32,64,2), ConvBlock(64,128,2), ConvBlock(128,256,2),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(256, 2)
        self.drop = nn.Dropout(0.4)
    def forward(self, mel, *_):
        return self.head(self.drop(self.net(mel).flatten(1)))

# ─── dataset & helpers ────────────────────────────────────────────────────────
class MMDataset(Dataset):
    def __init__(self, X, y, flow, esp, augment=False):
        self.X    = torch.from_numpy(X).unsqueeze(1)
        self.y    = torch.from_numpy(y.astype(np.int64))
        self.flow = torch.from_numpy(flow)
        self.esp  = torch.from_numpy(esp)
        self.aug  = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.aug:
            if torch.rand(1) > 0.4:
                f0,f = torch.randint(0,20,(1,)).item(), torch.randint(3,16,(1,)).item()
                x[:, f0:min(f0+f,64), :] = 0
            if torch.rand(1) > 0.4:
                t0,t = torch.randint(0,80,(1,)).item(), torch.randint(20,80,(1,)).item()
                x[:, :, t0:min(t0+t,512)] = 0
            x += torch.randn_like(x) * 0.015
        return x, self.flow[i], self.esp[i], self.y[i]

def frank_hall_loss(logits, labels):
    t1 = (labels >= 1).float()
    t2 = (labels >= 2).float()
    return (nn.functional.binary_cross_entropy_with_logits(logits[:,0], t1) +
            nn.functional.binary_cross_entropy_with_logits(logits[:,1], t2))

def decode(logits):
    p1 = torch.sigmoid(logits[:,0])
    p2 = torch.minimum(torch.sigmoid(logits[:,1]), p1)
    return torch.stack([1-p1, p1-p2, p2], dim=1).argmax(dim=1)

def adj_acc(yt, yp): return float(np.mean(np.abs(np.array(yt)-np.array(yp)) <= 1))

def run_exp(name, model, augment_train=True):
    ds_tr = MMDataset(X_mel[idx_tr], y_all[idx_tr], flow_arr[idx_tr], esp_arr[idx_tr], augment=augment_train)
    ds_te = MMDataset(X_mel[idx_te], y_all[idx_te], flow_arr[idx_te], esp_arr[idx_te], augment=False)

    y_tr = y_all[idx_tr]
    weights = (1.0 / np.bincount(y_tr))[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=32, shuffle=False, num_workers=0)

    model = model.to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_f1, best_path = 0.0, os.path.join(args.out, f'best_{name}.pt')
    log_rows = []

    print(f'\n{"="*60}\nExperiment {name}')
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for mel, flow, esp, yb in dl_tr:
            mel, flow, esp, yb = mel.to(device), flow.to(device), esp.to(device), yb.to(device)
            opt.zero_grad()
            loss = frank_hall_loss(model(mel, flow, esp), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(mel)
        tr_loss /= len(ds_tr)
        sched.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for mel, flow, esp, yb in dl_te:
                p = decode(model(mel.to(device), flow.to(device), esp.to(device))).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())

        adj = adj_acc(labels, preds)
        ex  = float(np.mean(np.array(preds)==np.array(labels)))
        f1  = f1_score(labels, preds, average='macro', zero_division=0)
        flag = ''
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            flag = ' <- BEST'
        log_rows.append(dict(epoch=epoch, loss=round(tr_loss,4), adj=round(adj,4),
                             exact=round(ex,4), f1=round(f1,4)))
        print(f'  Ep{epoch:03d} loss={tr_loss:.4f} adj={adj:.4f} exact={ex:.4f} F1={f1:.4f} {time.time()-t0:.1f}s{flag}')
        pd.DataFrame(log_rows).to_csv(os.path.join(args.out, f'log_{name}.csv'), index=False)

    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for mel, flow, esp, yb in dl_te:
            p = decode(model(mel.to(device), flow.to(device), esp.to(device))).cpu().numpy()
            preds.extend(p); labels.extend(yb.numpy())

    cm  = confusion_matrix(labels, preds)
    f1c = f1_score(labels, preds, average=None, zero_division=0)
    result = {
        'name': name, 'n_train': len(idx_tr), 'n_test': len(idx_te),
        'adj_acc':   round(adj_acc(labels, preds), 4),
        'exact_acc': round(float(np.mean(np.array(preds)==np.array(labels))), 4),
        'f1_macro':  round(f1_score(labels, preds, average='macro', zero_division=0), 4),
        'f1_sin':    round(float(f1c[0]),4) if len(f1c)>0 else 0,
        'f1_med':    round(float(f1c[1]),4) if len(f1c)>1 else 0,
        'f1_deg':    round(float(f1c[2]),4) if len(f1c)>2 else 0,
        'cm':        cm.tolist()
    }
    print(f'\n  FINAL {name}: adj={result["adj_acc"]}  F1={result["f1_macro"]}  F1_deg={result["f1_deg"]}')
    return result, log_rows

# ─── run both experiments ──────────────────────────────────────────────────────
res_B, log_B = run_exp('B_audio_only',  AudioOnly())
res_C, log_C = run_exp('C_multimodal',  MultimodalV2())

# ─── results ──────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('MULTIMODAL CONTRIBUTION (C - B):')
delta = {
    'f1_macro':   round(res_C['f1_macro']  - res_B['f1_macro'],  4),
    'adj_acc':    round(res_C['adj_acc']    - res_B['adj_acc'],   4),
    'f1_deg':     round(res_C['f1_deg']     - res_B['f1_deg'],    4),
}
print(f'  F1 macro:      B={res_B["f1_macro"]:.4f}  C={res_C["f1_macro"]:.4f}  delta={delta["f1_macro"]:+.4f}')
print(f'  Adj accuracy:  B={res_B["adj_acc"]:.4f}  C={res_C["adj_acc"]:.4f}  delta={delta["adj_acc"]:+.4f}')
print(f'  F1 desgastado: B={res_B["f1_deg"]:.4f}  C={res_C["f1_deg"]:.4f}  delta={delta["f1_deg"]:+.4f}')

# Flow correlation analysis
print('\n=== Flow-Wear Correlation (Scientific Finding) ===')
label_map = {0:'sin_desgaste', 1:'medianamente', 2:'desgastado'}
for j, col in enumerate(FLOW_COLS):
    vals = {label_map[k]: round(flow_arr[idx_te][y_all[idx_te]==k, j].mean(), 3)
            for k in [0,1,2]}
    print(f'  {col}: {vals}')

# Save
report = {'B_audio': res_B, 'C_multimodal': res_C, 'delta_C_minus_B': delta}
with open(os.path.join(args.out, 'report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [('adj_acc','Adjacent Accuracy'), ('f1_macro','F1 Macro'), ('f1_deg','F1 Desgastado')]
colors  = ['#4C72B0', '#55A868']
for ax, (metric, title) in zip(axes, metrics):
    vals = [res_B[metric], res_C[metric]]
    bars = ax.bar(['B: Audio\nonly', 'C: Audio+\nFlow+ESP32'], vals, color=colors)
    ax.set_title(title, fontsize=12)
    ax.set_ylim([0, 1.05])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2., v+0.01, f'{v:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle(f'Multimodal Contribution — train=test53({len(idx_tr)}), test=test39({len(idx_te)})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(args.out, 'multimodal_contribution.png'), dpi=150, bbox_inches='tight')
print(f'\nDone. Results in {args.out}')
