# find_clean_wavs.py
import os
from pathlib import Path
import csv

ROOT = Path("D:/dataset")
clean_dirs = [
    ROOT / "Con falla limpios",
    ROOT / "Sin falla limpios",
    ROOT / "Con falla" / "limpios",   # por si variantes
    ROOT / "Sin falla" / "limpios"
]

out_csv = Path("D:/pipeline_SVM/inputs/clean_wavs_list.csv")
rows = []
for base in [ROOT, ROOT / "Con falla limpios", ROOT / "Sin falla limpios"]:
    # We'll walk all subfolders under D:/dataset and collect anything under folders with 'limp' in name OR exact known paths
    pass

# más robusto: recorrer todo D:/dataset y marcar cualquier WAV cuyo parent-folder contenga 'limp' o 'limios' o 'limpio'
for root, dirs, files in os.walk(str(ROOT)):
    parent = Path(root).name.lower()
    if "limp" in parent or "clean" in parent:
        for f in files:
            if f.lower().endswith(".wav") or f.lower().endswith(".WAV"):
                p = Path(root) / f
                rows.append({"wav_path": str(p), "basename": f.lower(), "parent": parent})

# Si no hay coincidencias, también listamos todos WAVs de D:/dataset para inspección
if not rows:
    for root, dirs, files in os.walk(str(ROOT)):
        for f in files:
            if f.lower().endswith(".wav"):
                p = Path(root) / f
                rows.append({"wav_path": str(p), "basename": f.lower(), "parent": Path(root).name.lower()})

os.makedirs(out_csv.parent, exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["wav_path","basename","parent"])
    writer.writeheader()
    writer.writerows(rows)

print("Guardado listados wavs (primeros 20):", out_csv, "total:", len(rows))
