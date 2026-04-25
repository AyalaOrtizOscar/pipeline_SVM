# extract_features_from_manifests.py
#
# Extrae features acusticas de los WAVs en train/val/test.csv (manifests curados)
# y guarda un CSV listo para train_svm_ordinal.py
#
# Diferencia con extract_baseline_features.py:
#   - Lee directamente los manifests curados (con labels de 3 clases correctos)
#   - Preserva columnas: label, split, drill_group, mic_type, aug_type
#   - Columna 'basename' para GroupShuffleSplit en SVM
#
# Uso:
#   python extract_features_from_manifests.py               # procesa train+val+test
#   python extract_features_from_manifests.py --split train # solo un split
#   python extract_features_from_manifests.py --workers 4   # paralelo (default: 1)
#
# Salida: D:/pipeline_SVM/features/features_curated_splits.csv

import argparse, os, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import librosa

# ── Config ────────────────────────────────────────────────────────────────────
MANIFESTS = {
    "train": "D:/dataset/manifests/train.csv",
    "val":   "D:/dataset/manifests/val.csv",
    "test":  "D:/dataset/manifests/test.csv",
}
OUT_FEATURES = "D:/pipeline_SVM/features/features_curated_splits.csv"
SR           = 44100


# ── Feature extraction ────────────────────────────────────────────────────────
def spectral_entropy(S):
    p = S / (np.sum(S) + 1e-12)
    p = np.where(p <= 0, 1e-12, p)
    return float(-np.sum(p * np.log2(p)))

def crest_factor(y):
    peak = float(np.max(np.abs(y))) + 1e-12
    rms  = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    return peak / rms

def harmonic_percussive_ratio(y):
    y_h, y_p = librosa.effects.hpss(y)
    return float(np.sum(y_h ** 2) / (np.sum(y_p ** 2) + 1e-12))


def extract_features(wav_path: str) -> dict | None:
    """Extrae 14 features acusticas de un WAV. Retorna None si falla."""
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as e:
        return {"_error": str(e)}

    duration_s = len(y) / sr

    # Mel energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_total_energy = float(np.mean(np.sum(mel, axis=0)))

    # Time-domain
    rms   = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = float(20 * np.log10(rms + 1e-12))
    peak  = float(np.max(np.abs(y)))
    zcr   = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_std  = float(np.std(centroid))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean  = float(np.mean(rolloff))
    rolloff_std   = float(np.std(rolloff))
    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness_mean = float(np.mean(flatness))
    spectral_flatness_std  = float(np.std(flatness))
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = float(np.mean(bw))
    spectral_bandwidth_std  = float(np.std(bw))

    # STFT-based
    S = np.abs(librosa.stft(y, n_fft=2048))
    spectral_entropy_mean = spectral_entropy(np.mean(S, axis=1))

    # Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = float(np.mean(contrast))

    # MFCC (first 2 means)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_0_mean = float(np.mean(mfcc[0]))
    mfcc_1_mean = float(np.mean(mfcc[1]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean       = float(np.mean(chroma))
    chroma_mean_first = float(np.mean(chroma[0]))
    chroma_std        = float(np.std(chroma))

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_0_mean = float(np.mean(tonnetz[0]))

    # Onsets
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
    onsets     = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onsets) / duration_s if duration_s > 0 else 0.0

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Harmonic/percussive
    hp_ratio = harmonic_percussive_ratio(y)

    # Crest
    cr = crest_factor(y)

    return {
        "duration_s":               duration_s,
        "rms":                      rms,
        "rms_db":                   rms_db,
        "peak":                     peak,
        "zcr":                      zcr,
        "mel_total_energy":         mel_total_energy,
        "centroid_mean":            centroid_mean,
        "centroid_std":             centroid_std,
        "rolloff_mean":             rolloff_mean,
        "rolloff_std":              rolloff_std,
        "spectral_flatness_mean":   spectral_flatness_mean,
        "spectral_flatness_std":    spectral_flatness_std,
        "spectral_bandwidth_mean":  spectral_bandwidth_mean,
        "spectral_bandwidth_std":   spectral_bandwidth_std,
        "spectral_entropy_mean":    spectral_entropy_mean,
        "spectral_contrast_mean":   spectral_contrast_mean,
        "mfcc_0_mean":              mfcc_0_mean,
        "mfcc_1_mean":              mfcc_1_mean,
        "chroma_mean_first":        chroma_mean_first,
        "chroma_mean":              chroma_mean,
        "chroma_std":               chroma_std,
        "tonnetz_0_mean":           tonnetz_0_mean,
        "harmonic_percussive_ratio": hp_ratio,
        "tempo":                    tempo,
        "onset_rate":               onset_rate,
        "crest_factor":             cr,
        "wavelet_total_energy":     mel_total_energy,  # alias para compatibilidad SVM
    }


def _worker(args):
    """Wrapper para ProcessPoolExecutor."""
    idx, row_dict = args
    wav = row_dict["filepath"]
    feats = extract_features(wav)
    return idx, feats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extrae features desde manifests curados train/val/test"
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="all", help="Qué splits procesar (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Procesos paralelos (default: 1; usa 2-4 si tienes RAM)")
    parser.add_argument("--skip-aug", action="store_true",
                        help="Omitir augmentados (solo originales)")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    dfs = []
    for split in splits:
        df = pd.read_csv(MANIFESTS[split])
        if args.skip_aug:
            df = df[df["aug_type"] == "original"]
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Columna basename para GroupShuffleSplit en SVM
    df_all["basename"] = df_all["filepath"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )

    total = len(df_all)
    print(f"=== extract_features_from_manifests.py ===")
    print(f"  Splits: {splits} | Filas: {total} | workers={args.workers}")
    print(f"  Labels: {df_all['label'].value_counts().to_dict()}")
    if args.skip_aug:
        print("  Modo: solo originales")

    t0 = time.time()
    results = [None] * total
    errors  = 0

    if args.workers == 1:
        for i, (_, row) in enumerate(df_all.iterrows()):
            feats = extract_features(row["filepath"])
            results[i] = feats
            if feats and "_error" in feats:
                print(f"  ERROR [{i}] {row['filepath']}: {feats['_error']}")
                errors += 1
            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] {elapsed:.0f}s transcurridos | "
                      f"{rate:.1f} archivos/s | ETA: {eta/60:.1f} min")
    else:
        rows_list = [(i, row.to_dict()) for i, (_, row) in enumerate(df_all.iterrows())]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, r): r[0] for r in rows_list}
            done = 0
            for fut in as_completed(futures):
                idx, feats = fut.result()
                results[idx] = feats
                done += 1
                if feats and "_error" in feats:
                    errors += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total}] {elapsed:.0f}s | "
                          f"{rate:.1f} arch/s | ETA: {eta/60:.1f} min")

    # Construir DataFrame final
    feat_cols = [r for r in results if r and "_error" not in r]
    if not feat_cols:
        print("ERROR: No se extrajeron features.")
        return

    # Combinar metadata + features
    meta_cols = ["filepath", "label", "split", "aug_type", "mic_type",
                 "drill_group", "basename"]
    meta_cols = [c for c in meta_cols if c in df_all.columns]

    out_rows = []
    for i, (_, row) in enumerate(df_all.iterrows()):
        feats = results[i]
        if feats is None or "_error" in feats:
            continue
        entry = {c: row[c] for c in meta_cols if c in row.index}
        entry.update(feats)
        out_rows.append(entry)

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(OUT_FEATURES), exist_ok=True)
    out_df.to_csv(OUT_FEATURES, index=False)

    elapsed = time.time() - t0
    print(f"\nGuardado: {OUT_FEATURES}")
    print(f"  {len(out_df)} filas x {len(out_df.columns)} columnas")
    print(f"  Labels: {out_df['label'].value_counts().to_dict()}")
    print(f"  Errores: {errors}/{total}")
    print(f"  Tiempo total: {elapsed/60:.1f} min")
    print("\n=== Completado ===")
    print(f"\nSiguiente paso:")
    print(f"  python train_svm_ordinal.py \\")
    print(f"    --input {OUT_FEATURES} \\")
    print(f"    --outdir D:/pipeline_SVM/results/svm_ordinal \\")
    print(f"    --group-col basename")


if __name__ == "__main__":
    main()
