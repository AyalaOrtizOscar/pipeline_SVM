#!/usr/bin/env python3
"""
Script de inicio para Paperspace Gradient notebook.
Pegar y ejecutar en la primera celda del notebook para:
1. Instalar dependencias
2. Descargar dataset desde URL (OneDrive/GDrive)
3. Verificar GPU
4. Ejecutar training

REEMPLAZA <ONEDRIVE_URL_NPZ> y <ONEDRIVE_URL_META> con los links de descarga directa.
"""

# ─── CELDA 1: Setup ────────────────────────────────────────────────────────────
SETUP = """
import subprocess, sys

packages = ['librosa', 'soundfile', 'tqdm', 'scikit-learn']
for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

print('Packages installed.')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"""

# ─── CELDA 2: Download data ─────────────────────────────────────────────────────
DOWNLOAD = """
import os, requests
from pathlib import Path

STORAGE = Path('/storage')
STORAGE.mkdir(exist_ok=True)

# REPLACE with your OneDrive direct download URLs:
NPZ_URL  = '<ONEDRIVE_URL_NPZ>'     # mel_dataset.npz (~250 MB)
META_URL = '<ONEDRIVE_URL_META>'    # mel_dataset_meta.csv (~500 KB)

def download(url, dest):
    if Path(dest).exists():
        print(f'Already exists: {dest}')
        return
    print(f'Downloading to {dest}...')
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    done = 0
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'  {done/1e6:.1f} / {total/1e6:.1f} MB', end='\\r')
    print(f'  Done: {Path(dest).stat().st_size/1e6:.1f} MB')

download(NPZ_URL,  '/storage/mel_dataset.npz')
download(META_URL, '/storage/mel_dataset_meta.csv')

# Also download training script
import urllib.request
TRAIN_SCRIPT = 'https://raw.githubusercontent.com/...'   # or upload manually
"""

# ─── CELDA 3: Train ─────────────────────────────────────────────────────────────
TRAIN_CMD = """
!python train_cnn_ordinal_b.py \\
    --npz  /storage/mel_dataset.npz \\
    --meta /storage/mel_dataset_meta.csv \\
    --out  /storage/results/ \\
    --epochs 40 \\
    --batch 64 \\
    --lr 0.001
"""

# ─── CELDA 4: Monitor ───────────────────────────────────────────────────────────
MONITOR = """
import pandas as pd, time

# Live monitoring of training log (run this in a separate cell while training)
while True:
    try:
        log = pd.read_csv('/storage/results/training_log.csv')
        last = log.iloc[-1]
        print(f\"Epoch {int(last.epoch):03d} | adj={last.adj_acc:.4f} | "
              f"exact={last.exact_acc:.4f} | F1={last.f1_macro:.4f} | loss={last.train_loss:.4f}\")
    except:
        print('Waiting for training to start...')
    time.sleep(30)
"""

print("=== PAPERSPACE SETUP SCRIPT ===")
print("\nCopy each cell below into your Paperspace notebook:")
print("\n--- CELL 1 (dependencies) ---")
print(SETUP)
print("\n--- CELL 2 (download data) ---")
print(DOWNLOAD)
print("\n--- CELL 3 (start training) ---")
print(TRAIN_CMD)
print("\n--- CELL 4 (monitor - run separately while training) ---")
print(MONITOR)
print("\n=== DONE === ")
print("After training, download /storage/results/ from Paperspace to review.")
