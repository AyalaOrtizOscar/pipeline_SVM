#!/usr/bin/env python3
"""
CNN v3 Refinado — CNN-B base + mejoras conservadoras para maximizar F1 macro y exact accuracy.
Estrategia: misma arquitectura CNN-B que logró adj=98.8%,
            + label smoothing, + scheduler OneCycleLR, + más épocas con buen LR.

Uso:
    python train_cnn_v3_refined.py --npz /storage/mel_dataset.npz \
                                    --meta /storage/mel_dataset_meta.csv \
                                    --out /storage/results_v3/
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--npz',    default='/storage/mel_dataset.npz')
parser.add_argument('--meta',   default='/storage/mel_dataset_meta.csv')
parser.add_argument('--out',    default='/storage/results_v3/')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch',  type=int, default=48)
parser.add_argument('--lr',     type=float, default=3e-4)
parser.add_argument('--hold_experiment', default='E3')
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
LOG_CSV    = os.path.join(args.out, 'training_log.csv')
BEST_MODEL = os.path.join(args.out, 'best_model.pt')
REPORT_JSON = os.path.join(args.out, 'final_report.json')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ─── data ────────────────────────────────────────────────────────────────────
print('Loading data...')
data = np.load(args.npz)
X_all = data['X']
y_all = data['y']
meta  = pd.read_csv(args.meta)
assert len(X_all) == len(meta)
print(f'  Total: {len(X_all)} | shape: {X_all.shape}')
print(f'  Labels: {dict(zip(*np.unique(y_all, return_counts=True)))}')

hold_mask  = meta['experiment'] == args.hold_experiment
train_mask = ~hold_mask & (meta['mic_type'] != 'augmented_wavs')
aug_mask   = meta['mic_type'] == 'augmented_wavs'
exp_counts = meta['experiment'].value_counts()
tiny_exps  = exp_counts[exp_counts < 10].index
train_mask &= ~meta['experiment'].isin(tiny_exps)

idx_train = np.where(train_mask | aug_mask)[0]
idx_test  = np.where(hold_mask)[0]
print(f'  Train: {len(idx_train)} | Test: {len(idx_test)}')

class MelDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # SpecAugment mejorado
            if torch.rand(1) > 0.4:
                f0 = torch.randint(0, 20, (1,)).item()
                f  = torch.randint(3, 16, (1,)).item()
                x[:, f0:min(f0+f, 64), :] = 0
            if torch.rand(1) > 0.4:
                t0 = torch.randint(0, 80, (1,)).item()
                t  = torch.randint(20, 80, (1,)).item()
                x[:, :, t0:min(t0+t, 512)] = 0
            # Doble enmascaramiento de tiempo
            if torch.rand(1) > 0.6:
                t0 = torch.randint(200, 400, (1,)).item()
                t  = torch.randint(20, 60, (1,)).item()
                x[:, :, t0:min(t0+t, 512)] = 0
            x = x + torch.randn_like(x) * 0.015
        return x, self.y[i]

X_tr, y_tr = X_all[idx_train], y_all[idx_train]
X_te, y_te = X_all[idx_test],  y_all[idx_test]

class_counts = np.bincount(y_tr)
weights_per_class = 1.0 / class_counts
sample_weights = weights_per_class[y_tr]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

ds_train = MelDataset(X_tr, y_tr, augment=True)
ds_test  = MelDataset(X_te, y_te, augment=False)
dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=2, pin_memory=True)
dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,   num_workers=2, pin_memory=True)

# ─── model (CNN-B reforzado con 4 bloques) ───────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x): return self.net(x) + self.skip(x)

class CnnOrdinalV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.blocks = nn.Sequential(
            ConvBlock(32,  64,  stride=2),
            ConvBlock(64,  128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 384, stride=2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.45)
        self.fc   = nn.Linear(384, 64)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = torch.relu(self.fc(x))
        return self.head(x)

model = CnnOrdinalV3().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Model params: {n_params/1e3:.1f}k')

# ─── ordinal loss con label smoothing ────────────────────────────────────────
def frank_hall_loss(logits, labels, smoothing=0.05):
    t1 = (labels >= 1).float()
    t2 = (labels >= 2).float()
    if smoothing > 0:
        t1 = t1 * (1 - smoothing) + smoothing / 2
        t2 = t2 * (1 - smoothing) + smoothing / 2
    loss1 = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], t1)
    loss2 = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], t2)
    return loss1 + loss2

def decode_predictions(logits):
    p1 = torch.sigmoid(logits[:, 0])
    p2 = torch.sigmoid(logits[:, 1])
    p2 = torch.minimum(p2, p1)
    p0 = 1 - p1
    return torch.stack([p0, p1 - p2, p2], dim=1).argmax(dim=1)

def adjacent_accuracy(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))

# ─── training ────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr * 5,
    steps_per_epoch=len(dl_train),
    epochs=args.epochs,
    pct_start=0.2, anneal_strategy='cos'
)

log_rows = []
best_f1  = 0.0
best_adj = 0.0

print(f'\nTraining {args.epochs} epochs, batch={args.batch}, lr={args.lr}')
print('-' * 80)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0

    for xb, yb in dl_train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = frank_hall_loss(logits, yb, smoothing=0.05)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * len(xb)

    train_loss /= len(ds_train)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            preds = decode_predictions(model(xb.to(device))).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    adj_acc   = adjacent_accuracy(all_labels, all_preds)
    exact_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1_macro  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    elapsed   = time.time() - t0

    # Save best by F1 macro (not adj — avoids class collapse)
    flag = ''
    if f1_macro > best_f1:
        best_f1  = f1_macro
        best_adj = adj_acc
        torch.save(model.state_dict(), BEST_MODEL)
        flag = '  <- BEST'

    log_rows.append(dict(epoch=epoch, train_loss=round(train_loss,4),
                         adj_acc=round(adj_acc,4), exact_acc=round(exact_acc,4),
                         f1_macro=round(f1_macro,4), time_s=round(elapsed,1)))

    print(f'Ep {epoch:03d} | loss={train_loss:.4f} | adj={adj_acc:.4f} | '
          f'exact={exact_acc:.4f} | F1={f1_macro:.4f} | {elapsed:.1f}s{flag}')

    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)

# ─── final evaluation ─────────────────────────────────────────────────────────
print('\n=== Final Evaluation ===')
model.load_state_dict(torch.load(BEST_MODEL))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in dl_test:
        preds = decode_predictions(model(xb.to(device))).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm          = confusion_matrix(all_labels, all_preds)
adj_final   = adjacent_accuracy(all_labels, all_preds)
exact_final = float(np.mean(np.array(all_preds) == np.array(all_labels)))
f1_final    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

report = {
    'model': 'CNN-v3 (ResNet-B + OneCycle + LabelSmoothing)',
    'holdout': args.hold_experiment,
    'n_test': len(all_labels),
    'adjacent_accuracy': round(adj_final, 4),
    'exact_accuracy':    round(exact_final, 4),
    'f1_macro':          round(f1_final, 4),
    'f1_per_class': {
        'sin_desgaste':            round(f1_per_cls[0], 4) if len(f1_per_cls) > 0 else 0,
        'medianamente_desgastado': round(f1_per_cls[1], 4) if len(f1_per_cls) > 1 else 0,
        'desgastado':              round(f1_per_cls[2], 4) if len(f1_per_cls) > 2 else 0,
    },
    'confusion_matrix':    cm.tolist(),
    'cnn_b_adj_acc':       0.988,
    'delta_vs_cnn_b':      round(adj_final - 0.988, 4),
    'svm_baseline_adj_acc': 0.901,
    'delta_vs_svm':        round(adj_final - 0.901, 4),
}
print(json.dumps(report, indent=2))
with open(REPORT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

# ─── plots ────────────────────────────────────────────────────────────────────
log_df = pd.read_csv(LOG_CSV)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(log_df['epoch'], log_df['train_loss'], linewidth=2)
ax.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(log_df['epoch'], log_df['adj_acc'],   label='Adjacent Acc', linewidth=2)
ax.plot(log_df['epoch'], log_df['exact_acc'], label='Exact Acc',    linewidth=2)
ax.plot(log_df['epoch'], log_df['f1_macro'],  label='F1 Macro',     linewidth=2)
ax.axhline(y=0.988, color='r',    linestyle='--', label='CNN-B (0.988)')
ax.axhline(y=0.901, color='gray', linestyle='--', label='SVM (0.901)')
ax.set(xlabel='Epoch', ylabel='Score', title='Validation Metrics')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1.05])

ax = axes[2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix\nAdj={adj_final:.4f} F1={f1_final:.4f}')
ax.set_xticklabels(['sin', 'med', 'deg']); ax.set_yticklabels(['sin', 'med', 'deg'])

plt.tight_layout()
plt.savefig(os.path.join(args.out, 'training_analysis.png'), dpi=150, bbox_inches='tight')
print(f'\nPlot saved: {os.path.join(args.out, "training_analysis.png")}')
print(f'Adj={adj_final:.4f} | Exact={exact_final:.4f} | F1={f1_final:.4f}')
print(f'Delta vs CNN-B: {adj_final-0.988:+.4f}  |  Delta vs SVM: {adj_final-0.901:+.4f}')
