#!/usr/bin/env python3
"""Genera img_002.png: estructura general del estudio (tres bloques narrativos).
Izquierda: rediseño del ensayo.
Centro: adquisición y procesamiento de la señal.
Derecha: análisis y clasificación del desgaste.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 7.8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.8)
ax.axis('off')

C_L = '#2980B9'
C_C = '#8E44AD'
C_R = '#E67E22'
C_TITLE = '#2C3E50'
C_EDGE = '#555555'


def block(x, y, w, h, title, items, color):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor='white', edgecolor=color, linewidth=2.2)
    ax.add_patch(rect)
    ax.text(x, y + h/2 - 0.35, title, ha='center', va='center',
            fontsize=13, color=color, fontweight='bold')
    y0 = y + h/2 - 0.95
    for it in items:
        ax.text(x - w/2 + 0.25, y0, "\u2022  " + it, ha='left', va='center',
                fontsize=10.5, color='#222222')
        y0 -= 0.45


def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=2.0,
                                mutation_scale=22))


# Title
ax.text(8, 7.4, "Estructura general del estudio",
        ha='center', va='center', fontsize=16, color=C_TITLE, fontweight='bold')
ax.text(8, 7.0,
        "Del redise\u00f1o del ensayo a la clasificaci\u00f3n ordinal del desgaste",
        ha='center', va='center', fontsize=11, color='#555555', style='italic')

# Left block
block(2.8, 3.6, 4.6, 5.4,
      "Redise\u00f1o del ensayo",
      ["Taladrado CNC (Leadwell V-20)",
       "Herramienta: Dormer A100 / A114",
       "Pieza: disco AISI~4140",
       "Sujeci\u00f3n: copa / bridas",
       "Par\u00e1metros de corte (Cap.\u00a04)",
       "Manifiesto y trazabilidad"],
      C_L)

# Center block
block(8.0, 3.6, 4.6, 5.4,
      "Adquisici\u00f3n y procesamiento",
      ["NI cDAQ-9174 + NI-9234",
       "3 micr\u00f3fonos IEPE (44{,}1 kHz)",
       "ESP32 (INMP441 + YF-S201)",
       "Microscopio digital (flanco)",
       "Segmentaci\u00f3n por agujero",
       "Filtrado espectral y features"],
      C_C)

# Right block
block(13.2, 3.6, 4.6, 5.4,
      "An\u00e1lisis y clasificaci\u00f3n",
      ["Etiquetado ordinal [15/75]\u00a0\\%",
       "SVM Frank \\& Hall (Cap.\u00a08)",
       "Red neuronal convolucional",
       "Validaci\u00f3n LOEO / grupos",
       "M\u00e9tricas: macro\u2011F1, adj. acc.",
       "Interpretabilidad (SHAP)"],
      C_R)

# Arrows
arrow(5.15, 5.65, 3.6)
arrow(10.35, 10.85, 3.6)

# Footer
ax.text(8, 0.35,
        "Salida: base de conocimiento etiquetada \u00b7 modelos desplegados en GUI v5 \u00b7 aprendizaje iterativo por lote",
        ha='center', va='center', fontsize=10.5, color='#333333', style='italic')

plt.tight_layout(pad=0.3)
out = "D:/pipeline_SVM/informe_proyecto/figures/img_002.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
