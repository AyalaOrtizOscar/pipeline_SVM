#!/usr/bin/env python3
"""
extract_features_fast.py

Extrae features de WAVs en paralelo (multiprocessing).
Solo computa las features necesarias para el modelo SVM top-7.

Uso:
    python extract_features_fast.py --source original --output features_original.csv
    python extract_features_fast.py --source cleaned  --output features_cleaned.csv
    python extract_features_fast.py --source both
"""

import sys, os, argparse, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

FEATURES_ORIG = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
CLEAN_DIR = Path("D:/dataset/cleaned_wavs/combinado")
OUTDIR = Path("D:/pipeline_SVM/results/comparison_filtered")
SR = 44100


def extract_top7(wav_path: str) -> dict:
    """Extract only the 7 features needed for comparison (fast)."""
    import librosa
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if len(y) < 2048:
            return None

        duration_s = len(y) / sr

        # Shared STFT (computed once, reused)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

        # 1. spectral_contrast_mean
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        spectral_contrast_mean = float(np.mean(contrast))

        # 2. crest_factor (trivial)
        peak = np.max(np.abs(y)) + 1e-12
        rms = np.sqrt(np.mean(y**2)) + 1e-12
        crest_factor_val = peak / rms

        # 3. chroma_std
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        chroma_std = float(np.std(chroma))

        # 4. zcr
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # 5. spectral_entropy_mean
        S_mean = np.mean(S, axis=1)
        p = S_mean / (np.sum(S_mean) + 1e-12)
        p = np.where(p <= 0, 1e-12, p)
        spectral_entropy_mean = float(-np.sum(p * np.log2(p)))

        # 6. centroid_mean
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        centroid_mean = float(np.mean(centroid))

        # 7. harmonic_percussive_ratio
        y_h, y_p = librosa.effects.hpss(y)
        e_h = np.sum(y_h**2)
        e_p = np.sum(y_p**2) + 1e-12
        hp_ratio = e_h / e_p

        return {
            "spectral_contrast_mean": spectral_contrast_mean,
            "crest_factor": crest_factor_val,
            "chroma_std": chroma_std,
            "zcr": zcr,
            "spectral_entropy_mean": spectral_entropy_mean,
            "centroid_mean": centroid_mean,
            "harmonic_percussive_ratio": hp_ratio,
        }
    except Exception as e:
        print(f"  [ERROR] {wav_path}: {e}", file=sys.stderr)
        return None


def process_one(args):
    """Worker: extract features for a single (filepath, clean_path) pair."""
    idx, filepath, clean_path, source = args
    path = clean_path if source == "cleaned" else filepath
    feats = extract_top7(str(path))
    if feats is not None:
        feats["_idx"] = idx
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="both", choices=["original", "cleaned", "both"])
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load base manifest
    print("Loading features_curated_splits.csv...")
    df = pd.read_csv(FEATURES_ORIG, low_memory=False)
    df_orig = df[df["aug_type"] == "original"].copy().reset_index(drop=True)
    print(f"  {len(df_orig)} original samples")

    # Build clean paths
    clean_paths = []
    for fp in df_orig["filepath"]:
        rel = os.path.relpath(fp, "D:/") if "D:/" in fp.replace("\\", "/") else os.path.basename(fp)
        clean_paths.append(str(CLEAN_DIR / rel))
    df_orig["filepath_clean"] = clean_paths

    sources_to_run = [args.source] if args.source != "both" else ["original", "cleaned"]

    for source in sources_to_run:
        print(f"\n{'='*60}")
        print(f"  Extracting features: {source.upper()}")
        print(f"  Workers: {args.workers}")
        print(f"{'='*60}")

        # Build work items
        work = []
        for i, (_, row) in enumerate(df_orig.iterrows()):
            fp = row["filepath"]
            cp = row["filepath_clean"]
            path = cp if source == "cleaned" else fp
            if not os.path.exists(path):
                continue
            work.append((i, fp, cp, source))

        print(f"  Files to process: {len(work)}")
        t0 = time.time()

        # Process in parallel
        results = []
        with Pool(args.workers) as pool:
            for i, feats in enumerate(pool.imap_unordered(process_one, work, chunksize=10)):
                if feats is not None:
                    results.append(feats)
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(work) - i - 1) / rate / 60
                    print(f"  {i+1}/{len(work)}  {rate:.1f} files/s  ETA {eta:.1f}min")

        elapsed = time.time() - t0
        print(f"  Done: {len(results)}/{len(work)} in {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Merge features back into df
        feat_df = pd.DataFrame(results).set_index("_idx")
        out_df = df_orig.copy()

        # Replace feature columns with new values
        feat_cols = [c for c in feat_df.columns if c != "_idx"]
        for col in feat_cols:
            out_df[col] = np.nan
        for idx in feat_df.index:
            for col in feat_cols:
                out_df.at[idx, col] = feat_df.at[idx, col]

        out_path = OUTDIR / f"features_{source}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        n_valid = out_df[feat_cols[0]].notna().sum()
        print(f"  Valid features: {n_valid}/{len(out_df)}")


if __name__ == "__main__":
    main()
