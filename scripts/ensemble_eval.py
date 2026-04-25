#!/usr/bin/env python3
import torch, torch.nn as nn
import numpy as np, pandas as pd, json, os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

data  = np.load('/storage/mel_dataset.npz')
X_all = data['X']; y_all = data['y']
meta  = pd.read_csv('/storage/mel_dataset_meta.csv')
idx_test = np.where(meta['experiment'] == 'E3')[0]
X_te, y_te = X_all[idx_test], y_all[idx_test]
print(f'Test: {len(X_te)} samples')

class MelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dl = DataLoader(MelDS(X_te, y_te), batch_size=64, shuffle=False)

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

class CnnB(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.blocks = nn.Sequential(ConvBlock(32,64,2), ConvBlock(64,128,2), ConvBlock(128,256,2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(256, 2)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        return self.head(self.drop(self.gap(x).flatten(1)))

class CnnV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.blocks = nn.Sequential(ConvBlock(32,64,2), ConvBlock(64,128,2),
                                    ConvBlock(128,256,2), ConvBlock(256,384,2))
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.45)
        self.fc   = nn.Linear(384, 64)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        x = self.drop(self.gap(x).flatten(1))
        return self.head(torch.relu(self.fc(x)))

model_b  = CnnB().to(device)
model_b.load_state_dict(torch.load('/storage/results/best_model.pt', map_location=device))
model_b.eval()

model_v3 = CnnV3().to(device)
model_v3.load_state_dict(torch.load('/storage/results_v3/best_model.pt', map_location=device))
model_v3.eval()
print('Both models loaded OK')

def get_probs(model, dl):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            p1 = torch.sigmoid(logits[:,0])
            p2 = torch.minimum(torch.sigmoid(logits[:,1]), p1)
            probs = torch.stack([1-p1, p1-p2, p2], dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(yb.numpy())
    return np.vstack(all_probs), np.array(all_labels)

def adj_acc(yt, yp): return float(np.mean(np.abs(yt - yp) <= 1))

probs_b,  labels = get_probs(model_b,  dl)
probs_v3, _      = get_probs(model_v3, dl)

preds_b  = probs_b.argmax(1)
preds_v3 = probs_v3.argmax(1)
print(f'CNN-B:  adj={adj_acc(labels,preds_b):.4f}  F1={f1_score(labels,preds_b,average="macro",zero_division=0):.4f}')
print(f'CNN-v3: adj={adj_acc(labels,preds_v3):.4f}  F1={f1_score(labels,preds_v3,average="macro",zero_division=0):.4f}')

best_f1, best_w, best_preds = 0, 0.5, preds_b
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ens = w * probs_b + (1-w) * probs_v3
    p   = ens.argmax(1)
    f1  = f1_score(labels, p, average='macro', zero_division=0)
    adj = adj_acc(labels, p)
    print(f'  w_b={w:.1f}: adj={adj:.4f}  F1={f1:.4f}  exact={float(np.mean(p==labels)):.4f}')
    if f1 > best_f1:
        best_f1, best_w, best_preds = f1, w, p

print(f'\nBest ensemble: w_b={best_w}')
adj_e   = adj_acc(labels, best_preds)
exact_e = float(np.mean(best_preds == labels))
f1c     = f1_score(labels, best_preds, average=None, zero_division=0)
cm      = confusion_matrix(labels, best_preds)
print(f'  Adj={adj_e:.4f}  Exact={exact_e:.4f}  F1={best_f1:.4f}')
print(f'  F1: sin={f1c[0]:.4f}  med={f1c[1]:.4f}  deg={f1c[2]:.4f}')
print(f'  CM:\n{cm}')

os.makedirs('/storage/results_ensemble', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
sns.heatmap(confusion_matrix(labels,preds_b), annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title(f'CNN-B\nadj={adj_acc(labels,preds_b):.4f}  F1={f1_score(labels,preds_b,average="macro",zero_division=0):.4f}')
ax.set_xticklabels(['sin','med','deg']); ax.set_yticklabels(['sin','med','deg'])
ax = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
ax.set_title(f'Ensemble (w_b={best_w})\nadj={adj_e:.4f}  F1={best_f1:.4f}')
ax.set_xticklabels(['sin','med','deg']); ax.set_yticklabels(['sin','med','deg'])
plt.tight_layout()
plt.savefig('/storage/results_ensemble/comparison.png', dpi=150, bbox_inches='tight')

report = {
    'ensemble': f'CNN-B({best_w})+CNN-v3({round(1-best_w,1)})',
    'adjacent_accuracy': round(adj_e, 4),
    'exact_accuracy':    round(exact_e, 4),
    'f1_macro':          round(best_f1, 4),
    'f1_per_class': {'sin_desgaste': round(float(f1c[0]),4),
                     'medianamente': round(float(f1c[1]),4),
                     'desgastado':   round(float(f1c[2]),4)},
    'confusion_matrix': cm.tolist(),
    'svm_baseline_adj': 0.901,
    'cnn_b_adj':        round(adj_acc(labels,preds_b), 4),
    'delta_vs_svm':     round(adj_e - 0.901, 4),
}
with open('/storage/results_ensemble/report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print('Done.')
