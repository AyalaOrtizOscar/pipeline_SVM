# prepare_relabel_filled_and_previews.py
import pandas as pd, os, shutil, sys
from pathlib import Path

# Rutas (ajusta si cambian)
template_in = r"D:\pipeline_SVM\results\relabel_template_for_review.csv"
template_filled_out = r"D:\pipeline_SVM\results\relabel_template_for_review_filled.csv"
manifest_path = r"D:\pipeline_SVM\inputs\manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.relabel_applied.20250927_160732.no_aug.csv"
previews_dir = Path(r"D:\pipeline_SVM\previews\relabel_review")
previews_dir.mkdir(parents=True, exist_ok=True)

if not os.path.exists(template_in):
    print("Falta plantilla:", template_in); sys.exit(1)
if not os.path.exists(manifest_path):
    print("Falta manifest:", manifest_path); sys.exit(1)

print("Cargando template...", template_in)
df = pd.read_csv(template_in, low_memory=False)

# Asegurar columnas necesarias
if 'suggested_new_label' not in df.columns:
    if 'model_pred' in df.columns:
        df['suggested_new_label'] = df['model_pred']
    else:
        df['suggested_new_label'] = df.get('old_label', '')
if 'approved' not in df.columns:
    df['approved'] = False
if 'notes' not in df.columns:
    df['notes'] = ''

# Normalize column filepath -> basename_id
def basename_id_from_mel(p):
    if pd.isna(p): return ''
    p = str(p).replace("\\","/")
    bn = p.split("/")[-1].lower()
    # remove suffixes like _mel.npy, .npy, .mel.npy
    for suf in ['_mel.npy','.npy','_mel.npy']:
        if bn.endswith(suf):
            bn = bn[: -len(suf)]
            break
    # also strip other suffixes like _aug1, _aug2
    for suf in ['_aug1','_aug2','_aug3','_aug4','_aug5','_aug']:
        if bn.endswith(suf):
            bn = bn[: -len(suf)]
    return bn

df['basename_id'] = df['filepath'].astype(str).map(basename_id_from_mel)

# load manifest to map to wavs
print("Cargando manifest:", manifest_path)
m = pd.read_csv(manifest_path, low_memory=False)
m['fp_norm'] = m['filepath'].astype(str).str.replace("\\","/").str.strip().str.lower()
m['basename'] = m['fp_norm'].map(lambda x: x.split('/')[-1] if isinstance(x,str) else '')
m['basename_id'] = m['basename'].str.lower().str.replace('.wav','').str.replace('.mp3','')
# helper: prefer non _aug
def choose_best_match(cands):
    # cands is a DataFrame slice
    if len(cands)==0: return None
    # prefer entries without '_aug' in path
    c_na = cands[~cands['fp_norm'].str.contains('_aug', na=False)]
    if len(c_na)>0:
        return c_na.iloc[0]
    return cands.iloc[0]

matched_manifest_paths = []
matched_manifest_idx = []
not_matched_idx = []

print("Mapeando mel -> manifest por basename_id (esto puede tardar unos segundos)...")
for i, row in df.iterrows():
    bid = row['basename_id']
    if not bid:
        not_matched_idx.append(i)
        matched_manifest_paths.append(None)
        matched_manifest_idx.append(None)
        continue
    # match any manifest filepath containing the basename id
    cands = m[m['basename_id'].str.contains(bid, na=False)]
    if len(cands)==0:
        # try contains the bid in entire path (extra safety)
        cands = m[m['fp_norm'].str.contains(bid, na=False)]
    if len(cands)==0:
        not_matched_idx.append(i)
        matched_manifest_paths.append(None)
        matched_manifest_idx.append(None)
    else:
        best = choose_best_match(cands)
        matched_manifest_paths.append(best['filepath'])
        matched_manifest_idx.append(int(best.name) if hasattr(best,'name') else None)

df['matched_manifest_filepath'] = matched_manifest_paths
df['matched_manifest_index'] = matched_manifest_idx

print("Matched:", df['matched_manifest_filepath'].notna().sum(), " Unmatched:", df['matched_manifest_filepath'].isna().sum())

# Create preview files copy (only for matched)
preview_rows = []
for i,row in df.iterrows():
    mp = row['matched_manifest_filepath']
    if pd.isna(mp) or mp is None: continue
    src = str(mp).replace("\\","/")
    # if manifest stores relative paths, try to resolve; else assume absolute
    if not os.path.exists(src):
        # try to find relative to dataset root D:/dataset
        candidate = os.path.join(r"D:\dataset", os.path.basename(src))
        if os.path.exists(candidate):
            src = candidate
    if not os.path.exists(src):
        # skip if file not found
        df.at[i,'preview_file'] = None
        df.at[i,'preview_note'] = "manifest_file_not_found"
        continue
    # copy to previews with safe name
    safe_name = f"{i:04d}__{os.path.basename(src)}"
    dst = previews_dir / safe_name
    try:
        shutil.copy(src, dst)
        df.at[i,'preview_file'] = str(dst)
        df.at[i,'preview_note'] = ''
        preview_rows.append(i)
    except Exception as e:
        df.at[i,'preview_file'] = None
        df.at[i,'preview_note'] = f"copy_error:{e}"

print("Previews creados:", len(preview_rows), " en:", previews_dir)

# Save filled template
df.to_csv(template_filled_out, index=False, encoding='utf-8')
print("Plantilla rellena guardada en:", template_filled_out)
# Save unmatched list for manual inspection
unmatched = df[df['matched_manifest_filepath'].isna()]
unmatched[['filepath','basename_id']].to_csv(r"D:\pipeline_SVM\results\relabel_unmatched_list.csv", index=False)
print("Listado unmatched guardado en: D:\\pipeline_SVM\\results\\relabel_unmatched_list.csv")
