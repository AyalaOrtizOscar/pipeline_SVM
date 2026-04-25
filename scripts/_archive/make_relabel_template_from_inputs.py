# make_relabel_template_from_inputs.py (robusto v2)
import pandas as pd
import os, sys, json
from pathlib import Path

manifest_path = r"D:\pipeline_SVM\inputs\manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.relabel_applied.20250927_160732.no_aug.csv"
preds_path = r"D:\pipeline_SVM\inputs\val_predictions.csv"
top_errors = r"D:\pipeline_SVM\inputs\top_errors_sin_by_mic.csv"
out_template = r"D:\pipeline_SVM\results\relabel_template_for_review.csv"

def find_filepath_col(df):
    candidates = ['filepath','file','path','audio_filepath','audio','filename','mel_path']
    lc = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in lc:
            return df.columns[lc.index(cand)]
    # heurístico por contenido (barras/backslash)
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(30).tolist()
        if not sample:
            continue
        n_paths = sum(1 for s in sample if ('\\' in s or '/' in s) and len(s) > 8)
        if n_paths >= max(1, int(0.4 * len(sample))):
            return col
    return None

def norm_fp(s):
    return str(s).replace("\\\\","/").replace("\\","/").strip().lower()

def extract_best_prob_from_cell(cell):
    # si es lista/JSON
    if pd.isna(cell):
        return ''
    try:
        if isinstance(cell, (list,tuple)):
            return max(cell)
        text = str(cell).strip()
        if text.startswith('[') and text.endswith(']'):
            arr = json.loads(text.replace("'", '"'))
            if isinstance(arr, (list,tuple)):
                return max([float(x) for x in arr])
        # si es número
        return float(text)
    except Exception:
        return ''

# Load manifest
if not os.path.exists(manifest_path):
    print("ERROR: manifest no encontrado:", manifest_path); sys.exit(1)
m = pd.read_csv(manifest_path, low_memory=False)
# detect filepath column in manifest
if 'filepath' not in m.columns:
    mp = find_filepath_col(m)
    if mp is None:
        print("ERROR: no se detectó columna de filepath en el manifest. Columnas:", m.columns.tolist()); sys.exit(1)
    else:
        m = m.rename(columns={mp: 'filepath'})
m['fp_norm'] = m['filepath'].astype(str).map(norm_fp)
print("Manifest cargado. Filas:", len(m))

def build_rows_from_preds(p):
    rows = []
    fp_col = find_filepath_col(p)
    if fp_col is None:
        print("-> No se detectó filepath en preds")
        return rows
    p['fp_norm'] = p[fp_col].astype(str).map(norm_fp)
    # detect label/pred cols
    pred_label_col = None
    for c in ['pred_label','pred','label_pred','y_pred','prediction','label']:
        if c in p.columns:
            pred_label_col = c; break
    # detect prob columns
    prob_cols = [c for c in p.columns if c.lower().startswith('prob') or c.lower().startswith('p_') or c.lower().startswith('proba') or c=='probs']
    merged = p.merge(m[['fp_norm'] + ([c for c in ['label','mic_type','duration'] if c in m.columns])].rename(columns={'label':'manifest_label'}), on='fp_norm', how='left', suffixes=('','_m'))
    # if lots of NaN in manifest_label, try joining by basename
    n_missing = merged['manifest_label'].isna().sum()
    if n_missing > 0:
        # try basename matching
        print(f"-> {n_missing} filas no emparejadas por fp_norm. Intentando emparejar por basename...")
        m_bname = m.copy(); m_bname['bname'] = m_bname['filepath'].astype(str).apply(lambda x: Path(x).name.lower())
        p_b = p.copy(); p_b['bname'] = p_b[fp_col].astype(str).apply(lambda x: Path(x).name.lower())
        merged2 = p_b.merge(m_bname[['bname','label','mic_type']].rename(columns={'label':'manifest_label'}), on='bname', how='left', suffixes=('','_m'))
        # use manifest_label from merged2 to fill
        merged['manifest_label'] = merged['manifest_label'].fillna(merged2['manifest_label'])
        if 'mic_type' in merged.columns and 'mic_type' in merged2.columns:
            merged['mic_type'] = merged['mic_type'].fillna(merged2['mic_type'])
    # build rows
    for _,r in merged.iterrows():
        best_prob = ''
        # try prob cols
        if prob_cols:
            # if single column 'probs' might be list-like
            if len(prob_cols)==1:
                best_prob = extract_best_prob_from_cell(r[prob_cols[0]])
            else:
                try:
                    best_prob = max([float(r[c]) for c in prob_cols if pd.notna(r[c])])
                except:
                    best_prob = ''
        else:
            # heurístico buscar columnas con 'score' o similar
            for c in merged.columns:
                if 'score' in c.lower() or 'conf' in c.lower():
                    try:
                        best_prob = float(r[c]); break
                    except:
                        pass
        rows.append({
            'filepath': r.get(fp_col, ''),
            'old_label': r.get('manifest_label',''),
            'model_pred': r.get(pred_label_col,'') if pred_label_col else r.get('label_true',''),
            'pred_prob': best_prob,
            'mic_type': r.get('mic_type',''),
            'duration': r.get('duration','') if 'duration' in merged.columns else '',
            'suggested_new_label': r.get(pred_label_col,'') if pred_label_col else r.get('label_true',''),
            'reviewer': '',
            'notes': '',
            'approved': False
        })
    return rows

rows = []
if os.path.exists(preds_path):
    print("Leyendo predicciones desde:", preds_path)
    p = pd.read_csv(preds_path, low_memory=False)
    print("Columnas en preds:", p.columns.tolist())
    rows = build_rows_from_preds(p)

if (not rows) and os.path.exists(top_errors):
    print("Falling back a top_errors:", top_errors)
    te = pd.read_csv(top_errors, low_memory=False)
    print("Columnas en top_errors:", te.columns.tolist())
    rows = build_rows_from_preds(te)

if not rows:
    print("ERROR: no se generaron filas para la plantilla. Revisa val_predictions.csv y top_errors_sin_by_mic.csv")
    sys.exit(1)

out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(out_template), exist_ok=True)
out_df.to_csv(out_template, index=False, encoding='utf-8')
print("Plantilla escrita en:", out_template, "filas:", len(out_df))
