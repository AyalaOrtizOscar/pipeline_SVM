# D:/pipeline_SVM/scripts/add_basename_and_inspect.py
import pandas as pd
from pathlib import Path

F = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.with_duration.csv")
df = pd.read_csv(F, low_memory=False)

# Crear basename si no existe
if 'basename' not in df.columns:
    # usa filepath si existe, sino wav_path_norm
    src = 'filepath' if 'filepath' in df.columns else ('wav_path_norm' if 'wav_path_norm' in df.columns else None)
    if src is None:
        raise SystemExit("No hay columna filepath ni wav_path_norm para generar basename")
    df['basename'] = df[src].astype(str).apply(lambda p: Path(p).name.lower() if p and p != 'nan' else '')

# Inspect rapido
print("Filas:", len(df))
print("\nClases (label_clean si existe, sino label):")
labcol = 'label_clean' if 'label_clean' in df.columns else 'label'
print(df[labcol].value_counts(dropna=False).to_string())

print("\nMic types:")
if 'mic_type' in df.columns:
    print(df['mic_type'].value_counts().to_string())

print("\nDurations: na count:", df['duration'].isna().sum() if 'duration' in df.columns else "no duration")
if 'duration' in df.columns:
    print(df['duration'].describe().to_string())

# guardar versión con basename
OUT = F.parent / (F.stem + ".with_basename.csv")
df.to_csv(OUT, index=False)
print("\nGuardado preview:", OUT)
