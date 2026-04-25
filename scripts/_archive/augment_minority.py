"""
augment_minority.py
Genera augmentaciones conservadoras para las clases minoritarias.
Salida:
 - WAVs augmentados en D:/pipeline_SVM/augmented/<label>/
 - manifest CSV: D:/pipeline_SVM/augmented/augmented_manifest.csv
"""

import os, json, argparse, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import random
import pandas as pd

# ----------------- CONFIG -----------------
ROOT = Path("D:/pipeline_SVM")
INPUT_FEATURES = ROOT / "features" / "features_svm_baseline.csv"   # archivo con filepath y label_clean
OUT_DIR = ROOT / "augmented"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUG_MANIFEST = OUT_DIR / "augmented_manifest.csv"

SR = 44100  # sample rate to write (use native sr if you prefer)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Aug params (conservadores)
AUG_PER_FILE = 6           # número máximo de augmentaciones por archivo
TARGET_SNR_DB = [20, 25, 30]   # SNRs para ruido aditivo o mezcla con fondo
TIME_STRETCH_FACTORS = [0.98, 0.99, 1.01, 1.02]  # small tempo changes
PITCH_STEPS = [-1.0, -0.5, 0.5, 1.0]  # semitones
GAIN_DB = [-2.0, -1.0, 1.0, 2.0]  # small gain changes

# classes to augment (minorities)
TARGET_LABELS = ["desgastado"]  # usa 'desgastado' según label_clean en tu CSV. añade otras si quieres.
# optional background noise folder (exists in your dataset)
NOISE_FOLDER = Path("D:/dataset/ruidos")  # ajustar si está en otro sitio
# ------------------------------------------

def ensure_dir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def add_noise_at_snr(y, snr_db, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rms_signal = np.sqrt(np.mean(y**2))
    rms_noise = rms_signal / (10**(snr_db / 20.0))
    noise = np.random.normal(0, rms_noise, size=y.shape)
    return y + noise

def apply_gain(y, db):
    factor = 10.0**(db/20.0)
    return y * factor

def time_stretch_safe(y, rate):
    # librosa requires length > frame; handle short files by pad
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        # fallback: simple resample to approximate
        import resampy
        return resampy.resample(y, orig_sr=SR, target_sr=int(SR*rate))

def pitch_shift_safe(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    except Exception:
        return y

def mix_with_noise(y, noise_signals, snr_db, seed=None):
    if seed is not None:
        random.seed(seed)
    noise = random.choice(noise_signals)
    # ensure same length
    if len(noise) < len(y):
        # tile or loop
        reps = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, reps)[:len(y)]
    else:
        start = random.randint(0, len(noise) - len(y))
        noise = noise[start:start+len(y)]
    # scale noise to desired SNR
    rms_signal = np.sqrt(np.mean(y**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    if rms_noise == 0:
        return y
    target_rms_noise = rms_signal / (10**(snr_db / 20.0))
    noise = noise * (target_rms_noise / (rms_noise + 1e-12))
    return y + noise

def load_noise_files(noise_folder, sr):
    noises = []
    if not noise_folder.exists():
        return noises
    for p in noise_folder.rglob("*.wav"):
        try:
            n, _ = librosa.load(str(p), sr=sr, mono=True)
            noises.append(n)
        except Exception:
            continue
    return noises

def main():
    df = pd.read_csv(INPUT_FEATURES, low_memory=False)
    # normalize label column name
    if 'label_clean' in df.columns:
        label_col = 'label_clean'
    else:
        label_col = 'label'
    if 'filepath' not in df.columns:
        raise SystemExit("No encuentro columna 'filepath' en " + str(INPUT_FEATURES))

    noises = load_noise_files(NOISE_FOLDER, SR)
    rows = []
    for idx, row in df.iterrows():
        lab = str(row.get(label_col, "")).strip()
        if lab not in TARGET_LABELS:
            continue
        fp = Path(row['filepath'])
        if not fp.exists():
            print("No existe:", fp)
            continue
        try:
            y, sr = librosa.load(str(fp), sr=SR, mono=True)
        except Exception as e:
            print("error carga:", fp, e)
            continue
        base_name = fp.stem
        out_label_dir = OUT_DIR / lab
        ensure_dir(out_label_dir)
        # decide how many augs for this file
        n_aug = AUG_PER_FILE
        for i in range(n_aug):
            aug_y = y.copy()
            aug_meta = {"orig": str(fp), "label": lab, "aug_idx": i}
            # choose 1-3 transforms
            ops = []
            r = random.random()
            # always at least one transform
            if r < 0.25:
                # time stretch
                f = float(random.choice(TIME_STRETCH_FACTORS))
                aug_y = librosa.effects.time_stretch(aug_y, rate=f)
                ops.append(("time_stretch", f))
            elif r < 0.55:
                # pitch shift
                nst = float(random.choice(PITCH_STEPS))
                aug_y = pitch_shift_safe(aug_y, sr, nst)
                ops.append(("pitch_shift", nst))
            else:
                # small noise or gain and maybe mix
                if random.random() < 0.6:
                    snr = float(random.choice(TARGET_SNR_DB))
                    aug_y = add_noise_at_snr(aug_y, snr, seed=SEED + i)
                    ops.append(("add_noise_snr", snr))
                if random.random() < 0.4:
                    g = float(random.choice(GAIN_DB))
                    aug_y = apply_gain(aug_y, g)
                    ops.append(("gain_db", g))
            # sometimes mix with background machine noise (very conservative)
            if noises and random.random() < 0.3:
                snr_m = float(random.choice(TARGET_SNR_DB))
                aug_y = mix_with_noise(aug_y, noises, snr_m, seed=SEED+i)
                ops.append(("mix_bg_noise_snr", snr_m))
            # clip to [-1,1]
            maxv = np.max(np.abs(aug_y))
            if maxv > 1:
                aug_y = aug_y / (maxv + 1e-12)
            # write file
            out_name = f"{base_name}_aug_{i}.wav"
            out_path = out_label_dir / out_name
            sf.write(str(out_path), aug_y, SR, subtype='PCM_16')
            rows.append({
                "wav_path": str(out_path),
                "orig": str(fp),
                "label": lab,
                "aug_ops": json.dumps(ops),
                "aug_idx": i,
                "duration_s": float(len(aug_y)/SR)
            })
    if rows:
        pd.DataFrame(rows).to_csv(AUG_MANIFEST, index=False)
        print("Augmentaciones guardadas. manifest:", AUG_MANIFEST, "n:", len(rows))
    else:
        print("No se generaron augmentaciones (no hubo archivos de target o error).")

if __name__ == "__main__":
    main()
