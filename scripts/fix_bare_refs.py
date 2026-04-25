#!/usr/bin/env python3
"""Add labels to figures + replace bare 'Figura~X.Y' text refs with \ref{}."""
from pathlib import Path
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TEX = Path("D:/pipeline_SVM/informe_proyecto/informe_proyecto.tex")
txt = TEX.read_text(encoding="utf-8")

# (search-for-figure-caption-fragment, label-name, in-text-ref-fragment, replacement)
TARGETS = [
    # Chapter 7 figures
    ("Estructura general del \\textit{dataset} del primer grupo",
     "fig:orejarena-tree", "Figura~7.1", "Figura~\\ref{fig:orejarena-tree}"),
    ("Arquitectura del \\textit{pipeline} de extracci",
     "fig:pipeline-arch", "Figura~7.2", "Figura~\\ref{fig:pipeline-arch}"),
    ("Proyecci\\'{o}n PCA 2D sobre la línea base de diez",
     "fig:pca-baseline", None, None),
    ("Estabilidad de la línea base mediante \\textit{bootstrap}",
     "fig:bootstrap-stability", None, None),
    ("Mapa de calor de la correlaci\\'{o}n de Pearson",
     "fig:corr-heatmap", "Figura~7.4", "Figura~\\ref{fig:corr-heatmap}"),
    ("Proyección PCA 2D de las diez características de la línea base, coloreada",
     "fig:pca-svm", "Figura~7.5", "Figura~\\ref{fig:pca-svm}"),
    ("Mapa de métricas por clase (\\textit{precision}",
     "fig:svm-metrics", None, None),
    ("Resumen visual de las métricas ordinales sobre el conjunto de retenci",
     "fig:ordinal-metrics", None, None),
    ("Importancia por permutación para el \\textit{pipeline} SVM",
     "fig:perm-importance", "Figura~7.8", "Figura~\\ref{fig:perm-importance}"),
    ("Curva de aprendizaje: F1 macro (eje~Y) en funci",
     "fig:learning-cal", None, None),
]

# Apply labels: find \caption{...target...} \end{figure}, insert \label before \end
for frag, label, _, _ in TARGETS:
    # Match the block: \caption[...]?{...frag...} ... \end{figure}
    # Insert \label{label} just before \end{figure} if not already present
    pattern = re.compile(
        r"(\\caption(?:\[[^\]]*\])?\{[^}]*" + re.escape(frag) + r"[^}]*\}\s*)"
        r"(\\end\{figure\})",
        re.DOTALL,
    )
    def repl(m, lbl=label):
        block = m.group(0)
        if "\\label{" + lbl + "}" in block:
            return block
        return m.group(1) + f"\\label{{{lbl}}}\n" + m.group(2)
    new_txt, n = pattern.subn(repl, txt)
    if n == 0:
        print(f"  ✗ frag not found: {frag[:50]!r}")
    else:
        print(f"  ✓ label {label} added ({n} match)")
        txt = new_txt

# Special: Figura 7.3a and Figura 7.3b point to figs pca-baseline & bootstrap
# We'll replace bare refs manually below

# Replace bare text refs to panel-less figures
simple_refs = [
    ("Figura~7.1", "Figura~\\ref{fig:orejarena-tree}"),
    ("Figura~7.2", "Figura~\\ref{fig:pipeline-arch}"),
    ("Figura~7.4", "Figura~\\ref{fig:corr-heatmap}"),
    ("Figura~7.5", "Figura~\\ref{fig:pca-svm}"),
    ("Figura~7.8", "Figura~\\ref{fig:perm-importance}"),
    ("Figura~7.3a", "Figura~\\ref{fig:pca-baseline}"),
    ("Figura~7.3b", "Figura~\\ref{fig:bootstrap-stability}"),
    ("Figura~7.6a", "Figura~\\ref{fig:svm-metrics}a"),
    ("Figura~7.6b", "Figura~\\ref{fig:svm-metrics}b"),
    ("Figura~7.13a", "Figura~\\ref{fig:learning-cal}a"),
    ("Figura~7.13b", "Figura~\\ref{fig:learning-cal}b"),
]

for old, new in simple_refs:
    count = txt.count(old)
    if count:
        txt = txt.replace(old, new)
        print(f"  ✓ {old} → {new} ({count})")

TEX.write_text(txt, encoding="utf-8")
print(f"\nSaved {TEX}")
