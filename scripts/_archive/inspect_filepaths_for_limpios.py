# inspect_filepaths_for_limpios.py
import pandas as pd
from pathlib import Path
import re

SRC = "D:/pipeline_SVM/inputs/features_svm_baseline_cleaned.csv"  # ajusta si hace falta
df = pd.read_csv(SRC, low_memory=False)

print("Filas total:", len(df))
if 'filepath' not in df.columns:
    print("No veo columna 'filepath' en el CSV. Columnas:", df.columns.tolist())
    raise SystemExit()

# Muestra ejemplos
print("\nPrimeras 20 filepaths:")
for p in df['filepath'].astype(str).head(20):
    print(" ", p)

# Buscar substrings similares a 'limp' (case-insensitive)
candidates = df['filepath'].astype(str).str.lower().unique()
matches = [s for s in candidates if re.search(r"limp|limpios|limpio|clean", s)]
print(f"\nRutas que contienen 'limp' o 'clean' (mostrando hasta 50): {len(matches)}")
for s in matches[:50]:
    print(" ", s)

# Contar parent folders top-level (helpful para ver la estructura)
parents = df['filepath'].astype(str).apply(lambda p: Path(p).parts[-2] if len(Path(p).parts) >= 2 else "")
print("\nTop 30 parent-folder names:")
print(parents.value_counts().head(30).to_string())

# Si existe una columna 'source' o 'clean' mostrarla
for c in ['source','clean','is_clean','set']:
    if c in df.columns:
        print(f"\nColumna '{c}' sample values:")
        print(df[c].value_counts(dropna=False).head(20).to_string())
