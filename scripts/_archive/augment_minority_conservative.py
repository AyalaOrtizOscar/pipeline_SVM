# augment_minority_conservative.py (resumen)
import os, random
import numpy as np, soundfile as sf, librosa
import pandas as pd
from pathlib import Path

SRC = "D:/pipeline_SVM/features/features_svm_baseline.cleaned.csv"
OUT = Path("D:/pipeline_SVM/augmented/minority_desgastado")
OUT.mkdir(parents=True, exist_ok=True)
N_AUG = 8
SR = 22050

df = pd.read_csv(SRC, low_memory=False)
label_col = 'label_clean'
rows = df[df[label_col]=='desgastado']
def add_noise(sig, snr_db):
    rms = np.sqrt(np.mean(sig**2))
    noise_rms = rms / (10**(snr_db/20))
    return sig + np.random.normal(0, noise_rms, size=sig.shape)

for _, r in rows.iterrows():
    wav = r.get('wav_path_norm') or r.get('filepath') or r.get('wav_path')
    if pd.isna(wav) or not os.path.isfile(wav): continue
    y, sr = librosa.load(wav, sr=SR, mono=True)
    base = Path(wav).stem
    for i in range(N_AUG):
        y2 = y.copy()
        # pitch shift -0.5..+0.5 semitones
        y2 = librosa.effects.pitch_shift(y2, sr, n_steps=random.uniform(-0.5,0.5))
        # tiny time stretch 0.98..1.02
        ts = random.uniform(0.98,1.02)
        if abs(ts-1.0)>1e-6:
            y2 = librosa.effects.time_stretch(y2, ts)
        # small gain -1.5..+1.5 dB
        g = 10**(random.uniform(-1.5,1.5)/20)
        y2 *= g
        # add noise SNR 30..36
        y2 = add_noise(y2, random.uniform(30,36))
        # fix length
        if len(y2) > len(y): y2 = y2[:len(y)]
        else: y2 = np.pad(y2, (0, max(0,len(y)-len(y2))))
        sf.write(OUT/f"{base}_aug{i}.wav", y2, SR, subtype="PCM_16")
print("Augmentation done. Saved to", OUT)
