# clean_features_csv.py
import pandas as pd, numpy as np, os
SRC = r"D:\pipeline_SVM\inputs\features_svm_baseline_cleaned.csv"  # ajusta si está en otro sitio
OUT = r"D:\pipeline_SVM\inputs\features_svm_baseline_cleaned.for_svm.csv"

print("Leyendo:", SRC)
df = pd.read_csv(SRC, low_memory=False)

# 1) Drop columns totalmente NaN
all_nan = df.columns[df.isna().all()].tolist()
if all_nan:
    print("Columnas completamente NaN (se eliminan):", all_nan)
    df = df.drop(columns=all_nan)

# 2) Drop columnas no numéricas que no son metadata (evitar pasar strings a pipeline)
meta = ['filepath','label','label_clean','mic_type','experiment','duration','source']
meta_present = [c for c in meta if c in df.columns]
# si hay otras columnas objetivas que no quieras como feature (p.ej. 'group_tmp')
to_drop = [c for c in ['group_tmp','group','group_id','index'] if c in df.columns]
if to_drop:
    print("Eliminando cols meta/no-deseadas:", to_drop)
    df = df.drop(columns=to_drop)

# 3) Mostrar columnas numéricas / no numéricas
num = df.select_dtypes(include=[np.number]).columns.tolist()
obj = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numéricas:", len(num), "No-numéricas (ser metadata):", obj)

# 4) Eliminar features con demasiados NaN (umbral configurable)
th = 0.5
nan_frac = df[num].isna().mean()
cols_too_nan = nan_frac[nan_frac > th].index.tolist()
if cols_too_nan:
    print(f"Eliminando {len(cols_too_nan)} columnas con >{int(th*100)}% NaN:", cols_too_nan)
    df = df.drop(columns=cols_too_nan)

# 5) Guardar CSV a usar en SVM
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print("Guardado:", OUT, "filas:", len(df), "cols:", len(df.columns))



