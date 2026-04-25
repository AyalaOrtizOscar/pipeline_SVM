# make_svm_manifest.py
import pandas as pd, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, low_memory=False)
# columnas esperadas: filepath,label,mic_type (ajusta si es necesario)
for col in ['filepath','label']:
    if col not in df.columns:
        raise SystemExit(f"Falta columna {col} en {args.input}")
# intenta resolver mic_type si no existe
if 'mic_type' not in df.columns:
    if 'mic' in df.columns:
        df['mic_type'] = df['mic']
    else:
        df['mic_type'] = 'unknown'

# normalizar slashes
df['filepath'] = df['filepath'].astype(str).str.replace("\\\\","/")

df[['filepath','label','mic_type']].to_csv(args.output, index=False, encoding='utf-8')
print("Wrote:", args.output, "rows:", len(df))
