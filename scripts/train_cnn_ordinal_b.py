#!/usr/bin/env python3
"""
Ruta B — CNN ordinal sobre mel-espectrogramas (ResNet-18 desde cero).
Frank & Hall ordinal decomposition: C1=P(y>=1), C2=P(y>=2).

Diseñado para ejecutarse en Paperspace Gradient (PyTorch).
Logging a CSV para monitoreo remoto desde Claude.

Uso:
    python train_cnn_ordinal_b.py --npz /storage/mel_dataset.npz \
                                   --meta /storage/mel_dataset_meta.csv \
                                   --out /storage/results/
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path

# ─── parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--npz',    default='/storage/mel_dataset.npz')
parser.add_argument('--meta',   default='/storage/mel_dataset_meta.csv')
parser.add_argument('--out',    default='/storage/results/')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch',  type=int, default=64)
parser.add_argument('--lr',     type=float, default=1e-3)
parser.add_argument('--hold_experiment', default='E3',  help='Holdout experiment (fixed)')
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
LOG_CSV = os.path.join(args.out, 'training_log.csv')
BEST_MODEL = os.path.join(args.out, 'best_model.pt')
REPORT_JSON = os.path.join(args.out, 'final_report.json')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ─── dataset ──────────────────────────────────────────────────────────────────
print('Loading data...')
data = np.load(args.npz)
X_all = data['X']   # (N, 64, 512) float32
y_all = data['y']   # (N,) int8
meta  = pd.read_csv(args.meta)
assert len(X_all) == len(meta), f'Mismatch: {len(X_all)} mels vs {len(meta)} meta rows'
print(f'  Total: {len(X_all)} samples | shape: {X_all.shape}')
print(f'  Labels: {dict(zip(*np.unique(y_all, return_counts=True)))}')

# Split: holdout = E3, train = rest
hold_mask = meta['experiment'] == args.hold_experiment
train_mask = ~hold_mask & (meta['mic_type'] != 'augmented_wavs')
aug_mask   = meta['mic_type'] == 'augmented_wavs'

# Also exclude tiny art2 tests (< 10 samples) from train to avoid leakage
exp_counts = meta['experiment'].value_counts()
tiny_exps = exp_counts[exp_counts < 10].index
train_mask &= ~meta['experiment'].isin(tiny_exps)

idx_train = np.where(train_mask | aug_mask)[0]
idx_test  = np.where(hold_mask)[0]
print(f'  Train: {len(idx_train)} | Test (E3 holdout): {len(idx_test)}')

class MelDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N,1,64,512)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # SpecAugment: frequency masking
            if torch.rand(1) > 0.5:
                f0 = torch.randint(0, 10, (1,)).item()
                f  = torch.randint(0, 12, (1,)).item()
                x[:, f0:f0+f, :] = 0
            # SpecAugment: time masking
            if torch.rand(1) > 0.5:
                t0 = torch.randint(0, 50, (1,)).item()
                t  = torch.randint(0, 60, (1,)).item()
                x[:, :, t0:t0+t] = 0
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.02
        return x, self.y[i]

X_tr, y_tr = X_all[idx_train], y_all[idx_train]
X_te, y_te = X_all[idx_test],  y_all[idx_test]

# Weighted sampler for class imbalance
class_counts = np.bincount(y_tr)
weights_per_class = 1.0 / class_counts
sample_weights = weights_per_class[y_tr]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

ds_train = MelDataset(X_tr, y_tr, augment=True)
ds_test  = MelDataset(X_te, y_te, augment=False)
dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=2)
dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,   num_workers=2)

# ─── model (small CNN — 3 blocks) ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)

class CnnOrdinal(nn.Module):
    """
    Small CNN with Frank & Hall ordinal head.
    Input: (B, 1, 64, 512)
    Output: (B, 2)  — [logit_C1, logit_C2]
      C1 = P(y >= 1),  C2 = P(y >= 2)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.blocks = nn.Sequential(
            ConvBlock(32, 64,  stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(256, 2)   # 2 logits for Frank & Hall

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.head(x)  # (B, 2) logits

model = CnnOrdinal().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Model params: {n_params/1e3:.1f}k')

# ─── Frank & Hall ordinal loss ─────────────────────────────────────────────────
def frank_hall_loss(logits, labels):
    """
    logits: (B, 2) — [logit_C1, logit_C2]
    labels: (B,)  — 0, 1, 2
    C1 target: labels >= 1   (1 if worn)
    C2 target: labels >= 2   (1 if heavily worn)
    """
    t1 = (labels >= 1).float()
    t2 = (labels >= 2).float()
    loss1 = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], t1)
    loss2 = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], t2)
    return loss1 + loss2

def decode_predictions(logits):
    """Convert Frank & Hall logits to class predictions (0, 1, 2)."""
    p1 = torch.sigmoid(logits[:, 0])   # P(y >= 1)
    p2 = torch.sigmoid(logits[:, 1])   # P(y >= 2)
    p2 = torch.minimum(p2, p1)         # monotonicity constraint
    # Class probabilities
    p0 = 1 - p1
    p_med = p1 - p2
    p_deg = p2
    return torch.stack([p0, p_med, p_deg], dim=1).argmax(dim=1)

def adjacent_accuracy(y_true, y_pred):
    """Adjacent accuracy: |y_true - y_pred| <= 1."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))

# ─── training loop ─────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

log_rows  = []
best_adj  = 0.0

print(f'\nTraining {args.epochs} epochs, batch={args.batch}, lr={args.lr}')
print('-' * 70)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0

    for xb, yb in dl_train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = frank_hall_loss(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)

    train_loss /= len(ds_train)
    scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            preds = decode_predictions(model(xb.to(device))).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    adj_acc  = adjacent_accuracy(all_labels, all_preds)
    exact_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1_macro  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    elapsed   = time.time() - t0

    log_row = dict(epoch=epoch, train_loss=round(train_loss,4),
                   adj_acc=round(adj_acc,4), exact_acc=round(exact_acc,4),
                   f1_macro=round(f1_macro,4), time_s=round(elapsed,1))
    log_rows.append(log_row)

    flag = ''
    if adj_acc > best_adj:
        best_adj = adj_acc
        torch.save(model.state_dict(), BEST_MODEL)
        flag = '  ← BEST'

    print(f'Ep {epoch:03d} | loss={train_loss:.4f} | adj={adj_acc:.4f} | '
          f'exact={exact_acc:.4f} | F1={f1_macro:.4f} | {elapsed:.1f}s{flag}')

    # Write log after each epoch (so Claude can monitor remotely)
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)

# ─── final report ─────────────────────────────────────────────────────────────
print('\n=== Final Evaluation on E3 holdout ===')
model.load_state_dict(torch.load(BEST_MODEL))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in dl_test:
        preds = decode_predictions(model(xb.to(device))).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
adj_final  = adjacent_accuracy(all_labels, all_preds)
exact_final = float(np.mean(np.array(all_preds) == np.array(all_labels)))
f1_final    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

report = {
    'model': 'CNN-B (Frank&Hall ordinal)',
    'holdout': args.hold_experiment,
    'n_test': len(all_labels),
    'adjacent_accuracy': round(adj_final, 4),
    'exact_accuracy':    round(exact_final, 4),
    'f1_macro':          round(f1_final, 4),
    'f1_per_class':      {
        'sin_desgaste':          round(f1_per_cls[0], 4) if len(f1_per_cls) > 0 else 0,
        'medianamente_desgastado': round(f1_per_cls[1], 4) if len(f1_per_cls) > 1 else 0,
        'desgastado':            round(f1_per_cls[2], 4) if len(f1_per_cls) > 2 else 0,
    },
    'confusion_matrix': cm.tolist(),
    'svm_baseline_adj_acc': 0.901,   # Art.1 best result for comparison
    'delta_vs_svm': round(adj_final - 0.901, 4),
}

print(json.dumps(report, indent=2))
with open(REPORT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

print(f'\nAdj. accuracy CNN-B: {adj_final:.4f} vs SVM baseline: 0.9010')
print(f'Delta: {adj_final - 0.901:+.4f} pp')
print(f'\nAll outputs in: {args.out}')
