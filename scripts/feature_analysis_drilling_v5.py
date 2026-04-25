#!/usr/bin/env python3
"""
feature_analysis_drilling_v5.py

Extrae un conjunto amplio de características acústicas por archivo WAV y guarda
un CSV con agregados por archivo (mean/std, flags básicos). Pensado para el
dataset de taladrado descrito en el proyecto.

Nota: versión sin dependencia de scipy.signal.cwt (usa proxy multiescala RMS).
"""

import argparse
from pathlib import Path
import sys
import os
import json
import math
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------
# Helper functions
# ----------------------------
def safe_load(path, sr=None, mono=True):
    try:
        y, sr_ret = librosa.load(path, sr=sr, mono=mono)
        return y, sr_ret
    except Exception:
        try:
            y, sr_ret = sf.read(path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr is not None and sr_ret != sr:
                y = librosa.resample(y, orig_sr=sr_ret, target_sr=sr)
                sr_ret = sr
            return y, sr_ret
        except Exception as e2:
            raise e2

def spectral_entropy(S, eps=1e-10):
    """Shannon entropy of a power spectrum matrix (bins x frames) -> per-frame entropies"""
    ps = np.abs(S) + eps
    ps_norm = ps / np.sum(ps, axis=0, keepdims=True)
    ent = -np.sum(ps_norm * np.log2(ps_norm + eps), axis=0)
    return ent

def crest_factor(y):
    rms = np.sqrt(np.mean(y**2)) if np.any(y) else 0.0
    peak = np.max(np.abs(y)) if np.any(y) else 0.0
    if rms <= 0:
        return 0.0
    return peak / rms

# --- Reemplazo de CWT por proxy multiescala RMS (robusto) ---
def wavelet_total_energy(y, widths=[256, 512, 1024, 2048]):
    """
    Proxy multiescala de energía: calcula energía RMS en ventanas de distintos tamaños
    y suma las energías. Estable y libre de dependencia de cwt.
    widths: lista de tamaños de ventana en muestras.
    """
    try:
        total = 0.0
        yf = np.asarray(y, dtype=float)
        for w in widths:
            if w <= 0:
                continue
            if len(yf) < w:
                # señal más corta que ventana -> energía global
                frame_rms = np.sqrt(np.mean(yf**2)) if np.any(yf) else 0.0
                total += frame_rms**2
            else:
                hop = max(w // 2, 1)
                en = 0.0
                count = 0
                # framings
                for i in range(0, len(yf) - w + 1, hop):
                    f = yf[i:i + w]
                    if f.size == 0:
                        continue
                    rms = np.sqrt(np.mean(f**2))
                    en += rms**2
                    count += 1
                if count == 0:
                    # fallback to global energy
                    frame_rms = np.sqrt(np.mean(yf**2)) if np.any(yf) else 0.0
                    total += frame_rms**2
                else:
                    total += float(en)
        return float(total)
    except Exception:
        return float(np.sum(yf**2) if np.any(yf) else 0.0)

def agg_stats(arr):
    """Given 1D array -> return mean,std (nan-safe)"""
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan)
    a = np.array(arr, dtype=float)
    return (float(np.nanmean(a)), float(np.nanstd(a, ddof=1)) if a.size > 1 else 0.0)

# ----------------------------
# Feature extraction per file
# ----------------------------
def extract_features_for_file(path: Path, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048) -> Dict:
    row = {}
    row['filepath'] = str(path)
    row['basename'] = path.name
    parts = [p.lower() for p in path.parts]
    row['experiment'] = next((p for p in parts if p.startswith('e') and len(p) <= 5), None)
    mic_candidates = [p for p in parts if 'micro' in p or 'mic' in p or 'micrófono' in p]
    row['mic_type'] = mic_candidates[-1] if mic_candidates else None

    try:
        y, sr_ret = safe_load(str(path), sr=sr, mono=True)
    except Exception as e:
        row['error'] = str(e)
        return row

    if y is None or len(y) == 0:
        row['error'] = "empty"
        return row

    duration_s = float(len(y) / sr_ret)
    row['duration_s'] = duration_s

    try:
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        row['rms_mean'], row['rms_std'] = agg_stats(rms)

        row['peak'] = float(np.max(np.abs(y)))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length)[0]
        row['centroid_mean'], row['centroid_std'] = agg_stats(centroid)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
        row['rolloff_mean'], row['rolloff_std'] = agg_stats(rolloff)

        flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        row['spectral_flatness_mean'], row['spectral_flatness_std'] = agg_stats(flat)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length)
        contrast_mean_per_frame = np.mean(contrast, axis=0)
        row['spectral_contrast_mean'], row['spectral_contrast_std'] = agg_stats(contrast_mean_per_frame)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length)
        chroma_mean_per_frame = np.mean(chroma, axis=0)
        row['chroma_mean'], row['chroma_std'] = agg_stats(chroma_mean_per_frame)

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        row['zcr_mean'], row['zcr_std'] = agg_stats(zcr)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr_ret, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_ret, hop_length=hop_length, units='time')
        row['onset_rate'] = float(len(onsets) / duration_s) if duration_s > 0 else 0.0

        try:
            tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr_ret)[0])
        except Exception:
            tempo = 0.0
        row['tempo'] = tempo

        mfcc = librosa.feature.mfcc(y=y, sr=sr_ret, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        for i in range(n_mfcc):
            m_mean, m_std = agg_stats(mfcc[i])
            row[f"mfcc_{i}_mean"] = m_mean
            row[f"mfcc_{i}_std"] = m_std

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        ent = spectral_entropy(S)
        row['spectral_entropy_mean'], row['spectral_entropy_std'] = agg_stats(ent)

        try:
            # librosa.effects.hpss works on waveform or stft depending on version; try robustly
            try:
                y_h, y_p = librosa.effects.hpss(y)
                harm_energy = np.sum(y_h**2)
                perc_energy = np.sum(y_p**2)
            except Exception:
                # fallback: derive harmonic/percussive from stft magnitude via median filtering not used here
                harm_energy = 0.0
                perc_energy = 0.0
            row['harmonic_percussive_ratio'] = float(harm_energy / (perc_energy + 1e-12)) if (harm_energy + perc_energy) > 0 else 0.0
        except Exception:
            row['harmonic_percussive_ratio'] = np.nan

        row['crest_factor'] = float(crest_factor(y))

        S_mel = librosa.feature.melspectrogram(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length)
        row['mel_total_energy'] = float(np.sum(S_mel))

        # wavelet proxy multiscale energy
        row['wavelet_total_energy'] = float(wavelet_total_energy(y, widths=[256, 512, 1024, 2048]))

    except Exception as e:
        row['error'] = str(e)

    return row

# ----------------------------
# Main CLI
# ----------------------------
def find_wavs(input_dir: Path, exts=(".wav", ".WAV", ".flac", ".FLAC")) -> List[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="Extract audio features per file (drilling dataset).")
    parser.add_argument("--input-dir", "-i", required=True, help="Carpeta raíz con WAVs (recursivo).")
    parser.add_argument("--out-csv", "-o", default="features_per_file.csv", help="CSV de salida.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (0 = mantener original).")
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--manifest", default="", help="Optional CSV manifest que contenga metadata (filepath,label,experiment,mic_type).")
    parser.add_argument("--max-files", type=int, default=0, help="Limitar número de archivos (0 = todos).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv)
    sr = None if args.sr == 0 else args.sr

    if not input_dir.exists():
        print("Input dir no existe:", input_dir)
        sys.exit(1)

    wavs = find_wavs(input_dir)
    if args.max_files and args.max_files > 0:
        wavs = wavs[:args.max_files]

    print(f"Archivos detectados: {len(wavs)} (limit {args.max_files if args.max_files>0 else 'all'})")
    results = []
    errors = []

    manifest_df = None
    if args.manifest:
        try:
            manifest_df = pd.read_csv(args.manifest, low_memory=False)
            if 'filepath' in manifest_df.columns:
                manifest_df['fp_norm'] = manifest_df['filepath'].astype(str).str.replace("\\\\","/").str.strip().str.lower()
        except Exception as e:
            print("Warning: no pude leer manifest:", e)
            manifest_df = None

    for idx, p in enumerate(wavs, start=1):
        try:
            print(f"[{idx}/{len(wavs)}] Procesando: {p}")
            feats = extract_features_for_file(p, sr=sr, n_mfcc=args.n_mfcc, hop_length=args.hop_length, n_fft=args.n_fft)
            if manifest_df is not None and 'fp_norm' in manifest_df.columns:
                fp_norm = str(p).replace("\\\\","/").strip().lower()
                matched = manifest_df[manifest_df['fp_norm'] == fp_norm]
                if matched.empty:
                    matched = manifest_df[manifest_df['filepath'].astype(str).str.lower().str.contains(p.name.lower())]
                if not matched.empty:
                    rowm = matched.iloc[0].to_dict()
                    for k in ['label','experiment','mic_type','duration']:
                        if k in rowm:
                            feats[k] = rowm[k]
            results.append(feats)
        except Exception as e:
            print("ERROR procesando", p, e)
            errors.append({"filepath": str(p), "error": str(e)})

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    ir = {
        "n_files_found": len(wavs),
        "n_processed": len(results),
        "n_errors": len(errors),
        "errors_sample": errors[:10]
    }
    with open(out_csv.with_name(out_csv.stem + "_integrity_report.json"), "w", encoding="utf8") as fh:
        json.dump(ir, fh, indent=2, ensure_ascii=False)

    print("Guardado CSV:", out_csv)
    print("Integrity:", out_csv.with_name(out_csv.stem + "_integrity_report.json"))

if __name__ == "__main__":
    main()
