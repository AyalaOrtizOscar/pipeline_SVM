# augment_minority_conservative_fixed.py
# Genera augmentaciones conservadoras para la clase minoritaria 'desgastado'.
# Produce WAVs en D:/pipeline_SVM/augmented/minority_desgastado/ y un CSV mapping.

import os, random, math, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# CONFIG
ROOT = Path("D:/pipeline_SVM")
SRC_FEAT = ROOT / "features" / "features_svm_baseline.cleaned.csv"   # CSV limpio
OUT_DIR = ROOT / "augmented" / "minority_desgastado"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_CSV = OUT_DIR / "augmented_mapping.csv"
SR = 22050
MAX_AUG_PER_SAMPLE = 20    # to cap explosion
MIN_SNR_DB = 30.0
MAX_SNR_DB = 36.0

random.seed(42)
np.random.seed(42)

def add_noise(sig, snr_db):
    # adds gaussian noise at requested SNR (dB)
    rms = np.sqrt(np.mean(sig**2)) if np.any(sig) else 1e-8
    noise_rms = rms / (10**(snr_db/20.0))
    noise = np.random.normal(0, noise_rms, size=sig.shape)
    return sig + noise

def safe_pitch_shift(y, sr, n_steps):
    # use keyword args to be robust across librosa versions
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except TypeError:
        # fallback: try swapping keyword order
        return librosa.effects.pitch_shift(y, n_steps=n_steps, sr=sr)

def safe_time_stretch(y, rate):
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        # If time-stretch fails (too short), return original
        return y

# Load manifest/features
if not SRC_FEAT.exists():
    print("ERROR: no encontré CSV limpio en:", SRC_FEAT)
    sys.exit(1)

df = pd.read_csv(SRC_FEAT, low_memory=False)
label_col = 'label_clean' if 'label_clean' in df.columns else 'label'
minor = df[df[label_col] == 'desgastado'].copy()

n_minority = len(minor)
if n_minority == 0:
    print("No hay muestras 'desgastado' en", SRC_FEAT)
    sys.exit(0)

# Decide target: balancear hasta el máximo de otras clases (o un valor definido)
class_counts = df[label_col].value_counts()
target = int(class_counts.drop(labels=['desgastado'], errors='ignore').max())
print("Clase 'desgastado':", n_minority, "Target por clase:", target)

n_needed_total = max(0, target - n_minority)
if n_needed_total == 0:
    print("No se necesita augment (ya balanceado).")
    sys.exit(0)

# distribuir augment por muestra
n_aug_per_sample = math.ceil(n_needed_total / n_minority)
n_aug_per_sample = min(n_aug_per_sample, MAX_AUG_PER_SAMPLE)
print(f"Se generarán ~{n_aug_per_sample} augment por muestra (total aprox {n_aug_per_sample * n_minority})")

rows_written = []
count = 0
for idx, r in minor.iterrows():
    # elegir ruta WAV preferida: 'wav_path_norm' -> 'filepath' -> 'wav_path'
    wav = None
    for c in ('wav_path_norm','wav_path','filepath'):
        if c in r and pd.notna(r[c]) and str(r[c]).strip() != "":
            wav = str(r[c])
            break
    if not wav or not os.path.isfile(wav):
        # intenta con la versión 'd:' convertida (algunos paths están en H:\ vs d:\)
        if wav:
            alt = wav.replace("H:\\My Drive\\TRABAJO DE GRADO II\\CODIGO PROPIO\\PRUEBA de modelo #\\", "D:\\dataset\\")
            if os.path.isfile(alt):
                wav = alt
    if not wav or not os.path.isfile(wav):
        print("WARNING: no se encuentra WAV para fila index", idx, "ruta original:", r.get('filepath'))
        continue

    try:
        y, sr = librosa.load(wav, sr=SR, mono=True)
    except Exception as e:
        print("ERROR cargando", wav, e)
        continue

    base = Path(wav).stem
    for i in range(n_aug_per_sample):
        y2 = y.copy()

        # 1) pitch shift pequeño: -0.6 .. +0.6 semitones (con keyword args)
        n_steps = random.uniform(-0.6, 0.6)
        try:
            y2 = safe_pitch_shift(y2, sr, n_steps=n_steps)
        except Exception as e:
            # skip pitch if fails
            pass

        # 2) tiny time-stretch 0.985 .. 1.015 (avoid artifacts)
        rate = random.uniform(0.985, 1.015)
        if abs(rate - 1.0) > 1e-6:
            y2 = safe_time_stretch(y2, rate)

        # 3) small gain -1.5 .. +1.5 dB
        gain_db = random.uniform(-1.5, 1.5)
        gain = 10**(gain_db/20.0)
        y2 = y2 * gain

        # 4) add tiny noise SNR 30..36 dB
        y2 = add_noise(y2, random.uniform(MIN_SNR_DB, MAX_SNR_DB))

        # 5) trim/pad to original length
        if len(y2) > len(y):
            y2 = y2[:len(y)]
        elif len(y2) < len(y):
            y2 = np.pad(y2, (0, len(y)-len(y2)))

        out_name = OUT_DIR / f"{base}_aug_auto_{i}.wav"
        try:
            sf.write(out_name, y2, SR, subtype="PCM_16")
            rows_written.append({"orig": wav, "aug": str(out_name)})
            count += 1
        except Exception as e:
            print("ERROR escribiendo", out_name, e)

print("Generadas:", count, "augment files.")
# save mapping CSV
with open(MAPPING_CSV, "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=["orig","aug"])
    w.writeheader()
    for r in rows_written:
        w.writerow(r)

print("Mapping guardado en:", MAPPING_CSV)
print("Augment complete. Ahora: reextrae features de D:/pipeline_SVM/augmented/minority_desgastado y mézclalas con tu features CSV.")
