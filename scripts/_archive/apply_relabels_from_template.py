# apply_relabels_from_template.py
import pandas as pd, os, sys
from pathlib import Path
import shutil
import datetime

manifest_in = r"D:\pipeline_SVM\inputs\manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.relabel_applied.20250927_160732.no_aug.csv"
template_filled = r"D:\pipeline_SVM\results\relabel_template_for_review_filled.csv"
out_manifest = None  # si None generará con timestamp
backup_dir = r"D:\pipeline_SVM\backups"

def norm(s):
    return str(s).replace("\\","/").strip().lower()

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists(manifest_in):
    print("Manifest no encontrado:", manifest_in); sys.exit(1)
if not os.path.exists(template_filled):
    print("Plantilla rellena no encontrada:", template_filled); sys.exit(1)

m = pd.read_csv(manifest_in, low_memory=False)
tpl = pd.read_csv(template_filled, low_memory=False)

# normalize
m['fp_norm'] = m['filepath'].astype(str).map(norm)
# Prepare template: keep only approved
if 'approved' in tpl.columns:
    approved = tpl[(tpl['approved']==True) | (tpl['approved']=='True') | (tpl['approved']==1) | (tpl['approved']=='1')]
else:
    print("No columna 'approved' en la plantilla: asumir todas como aprobadas")
    approved = tpl.copy()

if approved.empty:
    print("No hay filas aprobadas para aplicar. Salida.")
    sys.exit(0)

# ensure fields present
if 'suggested_new_label' not in approved.columns:
    print("No hay 'suggested_new_label' en la plantilla. Asegúrate de rellenarla."); sys.exit(1)

# create backup
os.makedirs(backup_dir, exist_ok=True)
bak = os.path.join(backup_dir, f"{Path(manifest_in).name}.bak.{ts}.csv")
shutil.copy(manifest_in, bak)
print("Backup creado:", bak)

# build mapping by fp_norm and by basename
approved['fp_norm'] = approved['filepath'].astype(str).map(norm)
approved['bname'] = approved['filepath'].astype(str).map(lambda x: Path(x).name.lower())

map_fp = approved.set_index('fp_norm')['suggested_new_label'].to_dict()
map_bname = approved.dropna(subset=['bname']).set_index('bname')['suggested_new_label'].to_dict()

applied_rows = []
skipped_rows = []

# apply
m_out = m.copy()
m_out['new_label_applied'] = m_out['label']  # default

for idx,row in m_out.iterrows():
    fp = row['fp_norm']
    b = Path(row['filepath']).name.lower()
    if fp in map_fp:
        m_out.at[idx,'new_label_applied'] = map_fp[fp]
        applied_rows.append((row['filepath'], row['label'], map_fp[fp]))
    elif b in map_bname:
        m_out.at[idx,'new_label_applied'] = map_bname[b]
        applied_rows.append((row['filepath'], row['label'], map_bname[b]))
    else:
        skipped_rows.append((row['filepath'], row['label']))

# summary
n_applied = len(applied_rows)
n_skipped = len(skipped_rows)
out_manifest = out_manifest or os.path.splitext(manifest_in)[0] + f".relabel_applied.{ts}.csv"
m_out = m_out.drop(columns=['fp_norm'])
m_out.to_csv(out_manifest, index=False, encoding='utf-8')
print(f"Manifest actualizado guardado en: {out_manifest}")
print("Applied:", n_applied, "Skipped:", n_skipped)

# write details
applied_df = pd.DataFrame(applied_rows, columns=['filepath','old_label','new_label'])
skipped_df = pd.DataFrame(skipped_rows, columns=['filepath','old_label'])
applied_df.to_csv(os.path.join(backup_dir, f"applied_summary_{ts}.csv"), index=False, encoding='utf-8')
skipped_df.to_csv(os.path.join(backup_dir, f"skipped_summary_{ts}.csv"), index=False, encoding='utf-8')
print("Detalles guardados en backup_dir")
