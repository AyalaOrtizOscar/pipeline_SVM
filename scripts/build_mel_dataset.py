#!/usr/bin/env python3
"""
Builds the unified mel-spectrogram dataset for DL training.
Combines Orejarena (already computed mels) + Art.2 (generated from WAVs).
Output: D:/pipeline_SVM/features/mel_dataset.npz
  X      : (N, 64, 512) float32  — mel spectrograms
  y      : (N,)          int8     — 0=sin_desgaste 1=medianamente 2=desgastado
  meta   : JSON string per sample (experiment, mic_type, test_id, hole_id)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import librosa
import os
import json
from pathlib import Path
from tqdm import tqdm

FEATURES_CSV   = 'D:/pipeline_SVM/features/features_multimodal_merged.csv'
MEL_DIR_OREJ   = 'D:/dataset/mels/'
ART2_SEG_DIR   = 'D:/pipeline_SVM/art2_segments/'
OUT_NPZ        = 'D:/pipeline_SVM/features/mel_dataset.npz'
OUT_META_CSV   = 'D:/pipeline_SVM/features/mel_dataset_meta.csv'

SR       = 44100
N_MELS   = 64
HOP_LEN  = 512
N_FFT    = 2048
FIXED_W  = 512   # fixed width in frames (≈6 s at 44.1kHz/512 hop)

LABEL_MAP = {'sin_desgaste': 0, 'medianamente_desgastado': 1, 'desgastado': 2}

# ─── helpers ──────────────────────────────────────────────────────────────────

def wav_to_mel(wav_path, sr=SR, n_mels=N_MELS, hop=HOP_LEN, n_fft=N_FFT, fixed_w=FIXED_W):
    """Load WAV, compute mel-spectrogram (log scale), pad/crop to fixed width."""
    try:
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              hop_length=hop, n_fft=n_fft,
                                              fmax=sr//2)
        mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)  # z-score per clip
        # Pad or crop to fixed width
        if mel_db.shape[1] < fixed_w:
            mel_db = np.pad(mel_db, ((0, 0), (0, fixed_w - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :fixed_w]
        return mel_db.astype(np.float32)
    except Exception as e:
        print(f'  WARN: failed {wav_path}: {e}')
        return None


def load_or_compute_mel(row, orej_mel_dir):
    """Return mel array for a given metadata row."""
    label_str = str(row.get('label', row.get('label_original', ''))).lower()
    if 'sin' in label_str:
        subfolder = 'sin_desgaste'
    elif 'mediana' in label_str:
        subfolder = 'medianamente_desgastado'
    else:
        subfolder = 'desgastado'

    # Try pre-computed mel path (Orejarena already has these)
    mel_path = str(row.get('mel_path', ''))
    if mel_path and mel_path != 'nan' and os.path.exists(mel_path):
        try:
            mel = np.load(mel_path, allow_pickle=False).astype(np.float32)
            if mel.shape == (N_MELS, FIXED_W):
                return mel
        except Exception:
            pass  # fall through to WAV generation

    # Try filepath first (Art.2 WAVs at art2_segments), then orig_filepath (Orejarena)
    for col in ('filepath', 'orig_filepath'):
        wav_path = str(row.get(col, ''))
        if wav_path and wav_path != 'nan' and os.path.exists(wav_path):
            return wav_to_mel(wav_path)

    return None


# ─── main ─────────────────────────────────────────────────────────────────────

print('Loading metadata CSV...')
df = pd.read_csv(FEATURES_CSV)
print(f'  {len(df)} rows loaded')

# Use 'label' (relabeled with [15/75] threshold) as primary; 'label_original' as fallback
label_col = 'label' if df['label'].notna().sum() > df.get('label_original', pd.Series()).notna().sum() else 'label_original'
print(f'  Using label column: {label_col!r}')
# For rows where primary col is null, fill from the other
if 'label_original' in df.columns and 'label' in df.columns:
    df['label_merged'] = df['label'].fillna(df['label_original'])
    label_col = 'label_merged'
df = df[df[label_col].notna()].copy()
df['label_clean'] = df[label_col].str.lower().str.strip()
df = df[df['label_clean'].isin(LABEL_MAP.keys())]
print(f'  {len(df)} rows with valid labels')

X_list   = []
y_list   = []
meta_rows = []
skipped  = 0

print('Building mel dataset...')
for idx, row in tqdm(df.iterrows(), total=len(df)):
    label_int = LABEL_MAP[row['label_clean']]

    mel = load_or_compute_mel(row, MEL_DIR_OREJ)

    if mel is None:
        # Try art2 segments directory
        test_id = str(row.get('test_id', ''))
        hole_id = str(row.get('hole_id', ''))
        channel = str(row.get('channel', 'ch0'))
        if test_id:
            # Guess path from art2 segments
            prefix = 'C_' if test_id.startswith('test3') or int(test_id.replace('test','')) >= 39 else 'E_'
            wav_guess = Path(ART2_SEG_DIR) / f'{prefix}{test_id}' / 'holes' / f'{hole_id}_{channel}.wav'
            if wav_guess.exists():
                mel = wav_to_mel(wav_guess)

    if mel is None:
        skipped += 1
        continue

    X_list.append(mel)
    y_list.append(label_int)
    meta_rows.append({
        'experiment': row.get('experiment', ''),
        'test_id':    row.get('test_id', ''),
        'mic_type':   row.get('mic_type', ''),
        'group':      row.get('group', ''),
        'split':      row.get('split', ''),
        'label_str':  row['label_clean'],
        'label_int':  label_int,
        'coating':    row.get('coating_coded', 0),
    })

print(f'  Built {len(X_list)} samples, skipped {skipped}')

X = np.stack(X_list, axis=0)
y = np.array(y_list, dtype=np.int8)
print(f'  X shape: {X.shape}  y shape: {y.shape}')
print(f'  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

# Save meta
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUT_META_CSV, index=False)
print(f'  Meta saved → {OUT_META_CSV}')

# Save NPZ
print(f'Saving compressed NPZ to {OUT_NPZ} ...')
np.savez_compressed(OUT_NPZ, X=X, y=y)
size_mb = os.path.getsize(OUT_NPZ) / 1e6
print(f'  Done. File size: {size_mb:.1f} MB')
print(f'\nNext step: upload {OUT_NPZ} and {OUT_META_CSV} to Paperspace notebook.')
