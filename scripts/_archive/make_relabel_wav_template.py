# make_relabel_wav_template.py
import os, csv, sys
from pathlib import Path
import pandas as pd

ROOT = Path("D:/pipeline_SVM")
RUN_DIR = sorted(ROOT.joinpath("results").glob("svm_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
top_errors = RUN_DIR / "top_errors_for_relabel.csv"
OUT = RUN_DIR / "relabel_template_with_wavs.csv"
DATASET_ROOT = Path("D:/dataset")   # donde están las carpetas Con falla, Sin falla, ...

print("Run:", RUN_DIR)
print("Leyendo:", top_errors)
df = pd.read_csv(top_errors)

def find_wav_for_basename(basename, dataset_root=DATASET_ROOT):
    # busca archivos en dataset que contengan el basename (case-insensitive)
    b = basename.lower()
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if b in f.lower():
                return os.path.join(root, f)
    return ""

out_rows = []
for i, row in df.iterrows():
    # intentar derivar basename desde filepath (mel npy o ruta)
    fp = str(row.get("filepath",""))
    basename = Path(fp).name if fp else ""
    wav_path = ""
    if basename:
        # quitar sufijo de mel si aplica
        cand = basename
        for ext in [".npy",".mel.npy","_mel.npy","_mel.npy","_mel.npy",".wav",".WAV"]:
            cand = cand.replace(ext,"")
        wav_path = find_wav_for_basename(cand)
    out_rows.append({
        "mel_path": fp,
        "basename": basename,
        "wav_path": wav_path,
        "true_label": row.get("true_label", ""),
        "pred_label": row.get("pred_label",""),
        "model_top_prob": row.get("model_top_prob", row.get("pred_prob","")),
        "notes": "",
        "reviewer": "",
        "approved": False
    })

out_df = pd.DataFrame(out_rows)
out_df.to_csv(OUT, index=False)
print("Guardado plantilla para review:", OUT)
print("Muestras sin wav encontrado:", (out_df['wav_path']=="").sum(), "de", len(out_df))
