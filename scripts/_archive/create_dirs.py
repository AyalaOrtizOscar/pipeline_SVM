import os
from pathlib import Path 
BASE = Path(r"D:/pipeline_SVM")
folders = ["data", "manifests", "features", "models", "results", "previews", "relabels", "weights", "notebooks", "scripts"]
for f in folders:
    p = BASE/f
    p.mkdir(parents=True, exist_ok=True)
print("Created fodlers under:", BASE)