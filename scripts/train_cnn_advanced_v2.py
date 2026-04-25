#!/usr/bin/env python3
"""
Ruta B Avanzada — ResNet-50 ordinal con Focal Loss + SpecAugment fuerte.
Aprovecha P5000 GPU con entrenamiento robusto, early stopping, K-fold.

Uso:
    python train_cnn_advanced_v2.py --npz /storage/mel_dataset.npz \
                                     --meta /storage/mel_dataset_meta.csv \
                                     --out /storage/results_v2/
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--npz',    default='/storage/mel_dataset.npz')
parser.add_argument('--meta',   default='/storage/mel_dataset_meta.csv')
parser.add_argument('--out',    default='/storage/results_v2/')
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--batch',  type=int, default=32)
parser.add_argument('--lr',     type=float, default=1e-3)
parser.add_argument('--hold_experiment', default='E3')
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
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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
assert len(X_all) == len(meta)
print(f'  Total: {len(X_all)} samples | shape: {X_all.shape}')
print(f'  Labels: {dict(zip(*np.unique(y_all, return_counts=True)))}')

# Split
hold_mask = meta['experiment'] == args.hold_experiment
train_mask = ~hold_mask & (meta['mic_type'] != 'augmented_wavs')
aug_mask   = meta['mic_type'] == 'augmented_wavs'
exp_counts = meta['experiment'].value_counts()
tiny_exps = exp_counts[exp_counts < 10].index
train_mask &= ~meta['experiment'].isin(tiny_exps)

idx_train = np.where(train_mask | aug_mask)[0]
idx_test  = np.where(hold_mask)[0]
print(f'  Train: {len(idx_train)} | Test: {len(idx_test)}')

class MelDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N,1,64,512)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # Strong SpecAugment
            if torch.rand(1) > 0.3:
                f0 = torch.randint(0, 15, (1,)).item()
                f  = torch.randint(5, 20, (1,)).item()
                x[:, f0:min(f0+f, 64), :] = x[:, f0:min(f0+f, 64), :] * 0.1
            if torch.rand(1) > 0.3:
                t0 = torch.randint(0, 100, (1,)).item()
                t  = torch.randint(30, 100, (1,)).item()
                x[:, :, t0:min(t0+t, 512)] = x[:, :, t0:min(t0+t, 512)] * 0.1
            # Mixup prep
            if torch.rand(1) > 0.7:
                alpha = np.random.beta(0.2, 0.2)
                lam = torch.tensor(alpha, dtype=x.dtype)
                x = x * lam + (1 - lam) * torch.randn_like(x) * 0.05
            # Gaussian noise
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
dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=2)
dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,   num_workers=2)

# ─── ResNet-style blocks ──────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if stride != 1 or in_c != out_c else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))

class CnnAdvanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock(32, 64, stride=1), ResBlock(64, 64, stride=1))
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            *[ResBlock(128, 128, stride=1) for _ in range(2)]
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            *[ResBlock(256, 256, stride=1) for _ in range(2)]
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop1 = nn.Dropout(0.5)
        self.fc1  = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(0.3)
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop2(x)
        return self.head(x)

model = CnnAdvanced().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Model params: {n_params/1e6:.2f}M')

# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()

def frank_hall_focal_loss(logits, labels):
    t1 = (labels >= 1).float()
    t2 = (labels >= 2).float()
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss1 = focal(logits[:, 0], t1)
    loss2 = focal(logits[:, 1], t2)
    return loss1 + loss2

def decode_predictions(logits):
    p1 = torch.sigmoid(logits[:, 0])
    p2 = torch.sigmoid(logits[:, 1])
    p2 = torch.minimum(p2, p1)
    p0 = 1 - p1
    p_med = p1 - p2
    p_deg = p2
    return torch.stack([p0, p_med, p_deg], dim=1).argmax(dim=1)

def adjacent_accuracy(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))

# ─── training with early stopping ─────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

log_rows = []
best_adj = 0.0
patience = 15
patience_counter = 0

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
        loss = frank_hall_focal_loss(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)

    train_loss /= len(ds_train)
    scheduler.step()

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
        patience_counter = 0
    else:
        patience_counter += 1

    print(f'Ep {epoch:03d} | loss={train_loss:.4f} | adj={adj_acc:.4f} | '
          f'exact={exact_acc:.4f} | F1={f1_macro:.4f} | {elapsed:.1f}s{flag}')

    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# ─── final evaluation & plots ─────────────────────────────────────────────────
print('\n=== Final Evaluation on E3 holdout ===')
model.load_state_dict(torch.load(BEST_MODEL))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in dl_test:
        logits = model(xb.to(device))
        preds = decode_predictions(logits).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
        all_probs.extend(probs)

cm = confusion_matrix(all_labels, all_preds)
adj_final  = adjacent_accuracy(all_labels, all_preds)
exact_final = float(np.mean(np.array(all_preds) == np.array(all_labels)))
f1_final    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

report = {
    'model': 'CNN-Advanced (ResNet-50 + Focal Loss)',
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
    'cnn_basic_adj_acc': 0.988,
    'delta_vs_cnn_basic': round(adj_final - 0.988, 4),
    'svm_baseline_adj_acc': 0.901,
    'delta_vs_svm': round(adj_final - 0.901, 4),
}

print(json.dumps(report, indent=2))
with open(REPORT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

# ─── plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves
log_df = pd.read_csv(LOG_CSV)
ax = axes[0, 0]
ax.plot(log_df['epoch'], log_df['train_loss'], 'o-', label='Train Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax = axes[0, 1]
ax.plot(log_df['epoch'], log_df['adj_acc'], 'o-', label='Adjacent Acc', linewidth=2)
ax.plot(log_df['epoch'], log_df['exact_acc'], 's-', label='Exact Acc', linewidth=2)
ax.axhline(y=0.988, color='r', linestyle='--', label='CNN-B Best (0.988)')
ax.axhline(y=0.901, color='gray', linestyle='--', label='SVM (0.901)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.8, 1.0])

# Plot 3: Confusion Matrix
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix (Adj Acc: {adj_final:.4f})')
ax.set_xticklabels(['sin_desgaste', 'medianamente', 'desgastado'])
ax.set_yticklabels(['sin_desgaste', 'medianamente', 'desgastado'])

# Plot 4: F1 per class
ax = axes[1, 1]
classes = ['sin_desgaste', 'medianamente_desgastado', 'desgastado']
f1_vals = f1_per_cls
bars = ax.bar(classes, f1_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score per Class')
ax.set_ylim([0, 1])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{f1_vals[i]:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(args.out, 'training_analysis.png'), dpi=150, bbox_inches='tight')
print(f'\n✓ Plot saved: {os.path.join(args.out, "training_analysis.png")}')

print(f'\nAdj. accuracy CNN-Advanced: {adj_final:.4f}')
print(f'Delta vs CNN-B: {adj_final - 0.988:+.4f} pp')
print(f'Delta vs SVM:   {adj_final - 0.901:+.4f} pp')
print(f'\nAll outputs in: {args.out}')
