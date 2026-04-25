#!/usr/bin/env python3
"""
extract_features_from_splits.py
================================
Extrae 27 features acusticas de todos los WAVs en train/val/test.csv
y genera un CSV unificado compatible con train_svm_ordinal.py.

Salida: D:/pipeline_SVM/features/features_curated_splits.csv
        (sobreescribe el anterior, backup automatico)

Uso:
  python extract_features_from_splits.py
"""

import os
import re
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import pywt

# ── Configuracion ─────────────────────────────────────────────────────

MANIFESTS_DIR = Path("D:/dataset/manifests")
OUTPUT_PATH = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
SR = 44100


# ── Helpers ───────────────────────────────────────────────────────────

def spectral_entropy(S):
    p = S / (np.sum(S) + 1e-12)
    p = np.where(p <= 0, 1e-12, p)
    return -np.sum(p * np.log2(p))


def crest_factor(y):
    peak = np.max(np.abs(y)) + 1e-12
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    return peak / rms


def harmonic_percussive_ratio(y):
    y_h, y_p = librosa.effects.hpss(y)
    e_h = np.sum(y_h**2)
    e_p = np.sum(y_p**2) + 1e-12
    return e_h / e_p


def wavelet_energy(y):
    try:
        coeffs = pywt.wavedec(y, 'db4', level=4)
        return float(sum(np.sum(c**2) for c in coeffs))
    except Exception:
        return 0.0


def extract_features(filepath, sr=SR):
    """Extrae las 27 features del pipeline SVM."""
    y, sr_actual = librosa.load(str(filepath), sr=sr, mono=True)
    duration_s = len(y) / sr_actual

    # RMS
    rms_val = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = float(20 * np.log10(rms_val + 1e-10))
    peak = float(np.max(np.abs(y)))

    # ZCR
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Mel energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=64)
    mel_total_energy = float(np.mean(np.sum(mel, axis=0)))

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr_actual)
    centroid_mean = float(np.mean(sc))
    centroid_std = float(np.std(sc))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_actual)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness_mean = float(np.mean(flatness))
    spectral_flatness_std = float(np.std(flatness))

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr_actual)
    spectral_bandwidth_mean = float(np.mean(bw))
    spectral_bandwidth_std = float(np.std(bw))

    # Spectral entropy
    S = np.abs(librosa.stft(y, n_fft=2048))
    spec_entropy = spectral_entropy(np.mean(S, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr_actual)
    contrast_mean = float(np.mean(contrast))

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=13)
    mfcc_0_mean = float(np.mean(mfcc[0]))
    mfcc_1_mean = float(np.mean(mfcc[1]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr_actual)
    chroma_mean_first = float(np.mean(chroma[0]))
    chroma_mean = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr_actual)
    tonnetz_0_mean = float(np.mean(tonnetz[0]))

    # Harmonic-percussive ratio
    hp_ratio = harmonic_percussive_ratio(y)

    # Tempo & onset rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_actual)
    tempo_arr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr_actual)
    tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_actual)
    onset_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

    # Crest factor
    cr = crest_factor(y)

    # Wavelet energy
    wav_energy = wavelet_energy(y)

    return {
        "duration_s": duration_s,
        "rms": rms_val,
        "rms_db": rms_db,
        "peak": peak,
        "zcr": zcr,
        "mel_total_energy": mel_total_energy,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
        "spectral_flatness_mean": spectral_flatness_mean,
        "spectral_flatness_std": spectral_flatness_std,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_bandwidth_std": spectral_bandwidth_std,
        "spectral_entropy_mean": spec_entropy,
        "spectral_contrast_mean": contrast_mean,
        "mfcc_0_mean": mfcc_0_mean,
        "mfcc_1_mean": mfcc_1_mean,
        "chroma_mean_first": chroma_mean_first,
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
        "tonnetz_0_mean": tonnetz_0_mean,
        "harmonic_percussive_ratio": hp_ratio,
        "tempo": tempo,
        "onset_rate": onset_rate,
        "crest_factor": cr,
        "wavelet_total_energy": wav_energy,
    }


def filepath_to_experiment(filepath):
    fp = str(filepath).replace("\\", "/")
    name = os.path.splitext(os.path.basename(fp))[0].replace("limpio_", "")
    m6 = re.search(r"broc[az]_?(\d+)_(\d+)", name, re.IGNORECASE)
    if m6:
        diam = int(m6.group(1))
        seq = int(m6.group(2))
        if diam == 6:
            return {1: "E3", 2: "E4", 3: "E7"}.get(seq, "unknown")
        elif diam == 8:
            return {1: "E5", 2: "E6"}.get(seq, "unknown")
    m8 = re.search(r"^B0(\d)", name, re.IGNORECASE)
    if m8:
        return {1: "E1", 2: "E2"}.get(int(m8.group(1)), "unknown")
    return "unknown"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXTRACCION DE FEATURES SVM DESDE SPLITS")
    print("=" * 60)

    # Cargar los 3 splits
    splits = {}
    total = 0
    for name in ["train", "val", "test"]:
        path = MANIFESTS_DIR / f"{name}.csv"
        df = pd.read_csv(path)
        splits[name] = df
        total += len(df)
        print(f"  {name}.csv: {len(df)} filas")
    print(f"  Total WAVs a procesar: {total}")

    # Backup del archivo anterior si existe
    if OUTPUT_PATH.exists():
        backup = OUTPUT_PATH.parent / f"features_curated_splits_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(OUTPUT_PATH, backup)
        print(f"\n  Backup: {backup.name}")

    # Procesar todos los WAVs
    all_rows = []
    errors = []
    t0 = time.time()

    for split_name, df_split in splits.items():
        print(f"\n--- Procesando {split_name} ({len(df_split)} WAVs) ---")
        for idx, row in df_split.iterrows():
            filepath = row["filepath"]
            basename = os.path.basename(str(filepath))

            try:
                feats = extract_features(filepath)
                feats["filepath"] = filepath
                feats["label"] = row["label"]
                feats["split"] = split_name
                feats["aug_type"] = row.get("aug_type", "original")
                feats["mic_type"] = row.get("mic_type", "unknown")
                feats["experiment"] = row.get("experiment", filepath_to_experiment(filepath))
                feats["basename"] = basename

                # drill_group for backward compat
                exp = feats["experiment"]
                if "6" in basename or "broc" in basename.lower():
                    feats["drill_group"] = f"broca_6mm_{exp}"
                elif "B0" in basename or "8" in basename:
                    feats["drill_group"] = f"broca_8mm_{exp}"
                else:
                    feats["drill_group"] = f"unknown_{exp}"

                all_rows.append(feats)
            except Exception as e:
                errors.append({"filepath": filepath, "error": str(e)})

            # Progress
            n_done = len(all_rows) + len(errors)
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.1f} WAVs/s, ETA: {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Completado en {elapsed/60:.1f} minutos")
    print(f"  Procesados: {len(all_rows)}, Errores: {len(errors)}")

    if errors:
        print(f"\n  Errores:")
        for e in errors[:10]:
            print(f"    {e['filepath']}: {e['error']}")

    # Guardar
    df_out = pd.DataFrame(all_rows)

    # Reordenar columnas para compatibilidad
    meta_cols = ["filepath", "label", "split", "aug_type", "mic_type", "drill_group", "basename", "experiment"]
    feat_cols = [c for c in df_out.columns if c not in meta_cols]
    df_out = df_out[meta_cols + feat_cols]

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Guardado: {OUTPUT_PATH}")
    print(f"  {len(df_out)} filas x {len(df_out.columns)} columnas")

    # Resumen
    print(f"\n  Split distribution:")
    print(f"    {df_out['split'].value_counts().to_dict()}")
    print(f"\n  Label distribution:")
    print(f"    {df_out['label'].value_counts().to_dict()}")
    print(f"\n  Experiment distribution:")
    print(f"    {df_out['experiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
