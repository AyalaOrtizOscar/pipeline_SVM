# scripts/prepare_features_for_svm.py
import pandas as pd
from pathlib import Path
import argparse


p = argparse.ArgumentParser()
p.add_argument('--src','-s', default=r'D:/pipeline_SVM/features/features_per_file.csv')
p.add_argument('--out','-o', default=r'D:/pipeline_SVM/features/features_svm_baseline.csv')
args = p.parse_args()


src = Path(args.src)
out = Path(args.out)
if not src.exists():
	raise SystemExit(f"No existe features source: {src}")


# columnas de la línea base (ajustar si tus CSV tienen nombres distintos)
baseline = [
'harmonic_percussive_ratio', 'centroid_mean', 'zcr', 'spectral_flatness_mean',
'spectral_entropy_mean', 'onset_rate', 'duration_s', 'crest_factor',
'chroma_std', 'spectral_contrast_mean'
]


df = pd.read_csv(src, low_memory=False)
# asegurar columnas meta
meta_cols = ['filepath','label','experiment','mic_type']
for c in meta_cols:
    if c not in df.columns:
        df[c] = ''


# conservar sólo las columnas disponibles
keep = [c for c in baseline if c in df.columns]
if not keep:
    raise SystemExit('No se encontraron columnas de baseline en ' + str(src))


out_df = df[meta_cols + keep].copy()
out.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out, index=False)
print('Wrote features for SVM:', out, 'rows:', len(out_df))