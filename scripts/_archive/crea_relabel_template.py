# crea_relabel_template.py
import pandas as pd, os
manifest = r"D:\pipeline_SVM\inputs\manifest_full_with_augment_relabels_applied_full_manual_mic_utf8.with_cond_aug.relabel_applied.20250927_160732.csv"
out = r"D:\pipeline_SVM\results\relabel_review_template.csv"
m = pd.read_csv(manifest, low_memory=False)
m_sel = m[['filepath','label']].rename(columns={'label':'old_label'})
m_sel['new_label'] = m_sel['old_label']   # por defecto
m_sel['reviewer'] = ''
m_sel['review_date'] = ''
m_sel['approved'] = False
m_sel['notes'] = ''
m_sel['source_model_pred'] = ''
m_sel.to_csv(out, index=False)
print("Wrote template:", out)
