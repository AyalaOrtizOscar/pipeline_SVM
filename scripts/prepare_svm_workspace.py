# prepare_svm_workspace.py
import os, shutil, sys, numpy as np, pandas as pd
SRC = r"D:\pipeline_tools_for_improvement"
DST = r"D:\pipeline_SVM"
inputs = os.path.join(DST,"inputs")
os.makedirs(inputs, exist_ok=True)

# List of candidate source files to copy if exist (adjust as needed)
candidates = {
    "manifest": os.path.join(SRC,"manifests","manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.csv"),
    "manifest_no_aug": os.path.join(SRC,"manifests","manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.relabel_applied.20250927_160732.no_aug.csv"),
    "val_preds": os.path.join(SRC,"results","val_predictions.csv"),
    "val_preds_cal": os.path.join(SRC,"results","val_predictions_calibrated.csv"),
    "yamnet_npz": os.path.join(SRC,"results","yamnet_embeddings.npz"),
    "svm_prev_preds": os.path.join(SRC,"results","svm_test_preds.csv"),
    "relabel_summary": os.path.join(SRC,"relabels","relabels_applied_summary.csv"),
    "relabel_summary_master": os.path.join(SRC,"relabels","relabels_applied_summary_from_master.csv"),
    "top_errors": os.path.join(SRC,"results","top_errors_sin_by_mic.csv"),
    "classification_report": os.path.join(SRC,"results","plots","classification_report.csv"),
}
copied = []
for k,v in candidates.items():
    if os.path.exists(v):
        dstp = os.path.join(inputs, os.path.basename(v))
        shutil.copy2(v, dstp)
        copied.append((k,dstp))
    else:
        print("No encontrado (se puede ignorar):", v)

# If yamnet npz exists, create X/y numpy files and meta csv
npz = candidates.get("yamnet_npz")
if os.path.exists(npz):
    data = np.load(npz, allow_pickle=True)
    # meta could be stored as arrays or dict, handle commonly used patterns:
    if 'embeddings' in data:
        X = data['embeddings']
        meta_raw = data.get('meta', None)
        if meta_raw is None:
            print("npz tiene embeddings pero sin meta -> creando meta minimal")
            meta = pd.DataFrame({'index': list(range(len(X)))} )
        else:
            try:
                # meta might be an object array with dict inside
                meta_item = meta_raw.item() if hasattr(meta_raw, 'item') else meta_raw
                meta = pd.DataFrame(meta_item)
            except Exception:
                # try converting directly
                try:
                    meta = pd.DataFrame(meta_raw)
                except Exception:
                    meta = pd.DataFrame({'index': list(range(len(X)))})
        # save X/y and meta to inputs
        np.save(os.path.join(inputs,'X_yamnet.npy'), X)
        meta.to_csv(os.path.join(inputs,'yamnet_meta.csv'), index=False)
        print("Saved X_yamnet.npy and yamnet_meta.csv in", inputs)
    else:
        print("NPZ cargado pero no contiene 'embeddings' key. Keys:", list(data.keys()))
else:
    print("No se encontró yamnet npz. Si no existe, puedes generar embeddings con extract script.")

print("Archivos copiados a", inputs)
print("Lista resumida de ficheros copiados:")
for k,p in copied:
    print(" -", k, "->", p)
