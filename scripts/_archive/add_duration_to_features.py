# add_duration_to_features.py
import pandas as pd
import soundfile as sf
from pathlib import Path
import sys

IN = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.csv")
OUT = Path("D:/pipeline_SVM/inputs/features_svm_baseline_limpios_originals.with_duration.csv")

if not IN.exists():
    print("ERROR: no existe el CSV esperado:", IN)
    sys.exit(1)

df = pd.read_csv(IN, low_memory=False)
if 'filepath' not in df.columns and 'wav_path_norm' not in df.columns:
    print("ERROR: el CSV no tiene columna 'filepath' ni 'wav_path_norm' para localizar archivos WAV.")
    print("Columnas disponibles:", df.columns.tolist())
    sys.exit(1)

# decidir columna con rutas reales
path_col = 'filepath' if 'filepath' in df.columns else 'wav_path_norm'

# si 'wav_path_norm' está pero con rutas normalizadas (lowercase, backslashes),
# y los archivos están en disco con mayúsculas/rel paths, es posible que necesites mapear.
dur = []
for p in df[path_col].astype(str):
    if p == "" or p.lower() == "nan":
        dur.append(None)
        continue
    # si la ruta está normalizada volverla a forma Windows estándar (intenta tanto literal como reemplazo)
    cand = Path(p)
    if not cand.exists():
        # intentar sin drive lowercase/normalizar barras
        p2 = p.replace("\\\\","\\").replace("/", "\\")
        cand = Path(p2)
    try:
        data, sr = sf.read(str(cand))
        dur.append(len(data)/sr)
    except Exception as e:
        dur.append(None)
        # no saturar la salida
        print("No se pudo leer:", p, "->", e)

df['duration'] = dur
df.to_csv(OUT, index=False)
print("Guardado con durations:", OUT, "filas:", len(df))
