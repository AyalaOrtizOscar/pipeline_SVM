# inspect_unmatched.py
import pandas as pd
p = pd.read_csv(r"D:\pipeline_SVM\results\relabel_template_for_review.csv", low_memory=False)
print("Filas:", len(p))
print("\nColumnas:", p.columns.tolist())
# mostrar primeras 20 filas
print("\nPrimeras 20 filas:")
print(p.head(20).to_string(index=False))
# ver basenames y recuento
p['basename'] = p['filepath'].astype(str).map(lambda x: x.replace("\\","/").split("/")[-1].lower())
print("\nTop 30 basenames (freq):")
print(p['basename'].value_counts().head(30).to_string())
# salvar preview para abrir rápido
p.head(200).to_csv(r"D:\pipeline_SVM\results\relabel_template_preview.csv", index=False)
print("\nPreview guardado en relabel_template_preview.csv")
