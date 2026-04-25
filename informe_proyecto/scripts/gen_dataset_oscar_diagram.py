#!/usr/bin/env python3
"""Genera el diagrama de estructura del Lote II (ensayos E8-E14) alineado con el
Capítulo 5 del informe. Sin fechas explícitas; numeración E8-E14 analoga a E1-E7."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

C_ROOT  = '#2C3E50'
C_GRP   = '#2980B9'
C_NI    = '#8E44AD'
C_ESP   = '#E67E22'
C_STATE = '#E74C3C'
C_EDGE  = '#555555'


def box(x, y, w, h, text, color, fontsize=10, text_color='white'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=C_EDGE, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold',
            wrap=True, multialignment='center')


def arrow(x0, y0, x1, y1, lw=1.3, alpha=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=lw, alpha=alpha, mutation_scale=14))


# ── ROOT ─────────────────────────────────────────────────────────────
box(8, 8.3, 8.0, 0.9,
    "Lote II — Ensayos E8 a E14\n7 brocas Dormer A100 · 575 agujeros · 1 725 audios NI",
    C_ROOT, fontsize=13)

# ── GROUPS ───────────────────────────────────────────────────────────
box(4, 6.8, 6.2, 0.9,
    "Ensayos hasta falla (E8, E9, E10)\n3 brocas · 232 agujeros · NI 3 canales",
    C_GRP, fontsize=11)
box(12, 6.8, 6.2, 0.9,
    "Ensayos sin inducción de falla (E11–E14)\n4 brocas · 343 agujeros · NI + ESP32",
    C_GRP, fontsize=11)

arrow(6.4, 7.85, 4.9, 7.3)
arrow(9.6, 7.85, 11.1, 7.3)

# ── NI CHANNELS (banda compartida) ───────────────────────────────────
ax.text(8, 5.8, "Cadena principal NI cDAQ-9174 + NI-9234   (44,1 kHz · 24 bit · IEPE)",
        ha='center', va='center', fontsize=11, color=C_NI, style='italic',
        fontweight='bold')

ni_x = [3.0, 6.5, 10.0]
ni_labels = ["ch0 · MAXLIN UDM-51\n(dinámico)",
             "ch1 · Behringer SL84C\n(dinámico)",
             "ch2 · Behringer C1\n(condensador)"]
for x, lab in zip(ni_x, ni_labels):
    box(x, 5.0, 2.6, 0.8, lab, C_NI, fontsize=10)

# ── ESP32 ────────────────────────────────────────────────────────────
box(13.5, 5.0, 3.4, 0.8,
    "ESP32 (solo E11–E14)\nINMP441 + YF-S201",
    C_ESP, fontsize=10)

# arrows from groups to sensors
for x in ni_x:
    arrow(4.0, 6.35, x, 5.4, lw=1.0, alpha=0.7)
for x in ni_x:
    arrow(12.0, 6.35, x, 5.4, lw=1.0, alpha=0.55)
arrow(12.0, 6.35, 13.5, 5.4, lw=1.3)

# ── DATA LAYER (totales por modalidad) ───────────────────────────────
ax.text(8, 4.1, "Modalidades registradas por ensayo",
        ha='center', va='center', fontsize=11, color='#333333',
        style='italic', fontweight='bold')
mod_x = [3.0, 8.0, 13.0]
mod_labels = ["Audios NI\n1 725 muestras",
              "Imágenes microscopio\n116 registros",
              "Registros ESP32\n(audio + caudal) · 4"]
for x, lab in zip(mod_x, mod_labels):
    box(x, 3.3, 3.8, 0.8, lab, '#34495E', fontsize=10)

for x in ni_x:
    arrow(x, 4.6, 3.0, 3.7, lw=0.6, alpha=0.4)
    arrow(x, 4.6, 8.0, 3.7, lw=0.6, alpha=0.4)
arrow(13.5, 4.6, 13.0, 3.7, lw=1.0)

# ── STATES ───────────────────────────────────────────────────────────
ax.text(8, 2.3, "Etiquetado ordinal por % de vida útil nominal",
        ha='center', va='center', fontsize=11, color=C_STATE,
        style='italic', fontweight='bold')

states = [
    (3.0, 1.4,  "Estado 0 — sin desgaste\n[0 – 15 %]"),
    (8.0, 1.4,  "Estado 1 — medianamente desgastado\n[15 – 75 %]"),
    (13.0, 1.4, "Estado 2 — desgastado\n[75 – 100 %]"),
]
for xs, ys, txt in states:
    box(xs, ys, 4.2, 0.8, txt, C_STATE, fontsize=10)

for x in mod_x:
    for xs, ys, _ in states:
        ax.plot([x, xs], [2.9, 1.82], color=C_EDGE, lw=0.5, alpha=0.3)

ax.text(8, 0.35,
        "Criterio físico: VB = 0{,}3 mm (ISO 3685 / ISO 8688-2) · Umbral [15/75] % consistente con el Lote I",
        ha='center', va='center', fontsize=10, color='#333333', style='italic')

plt.tight_layout(pad=0.2)
out = "D:/pipeline_SVM/informe_proyecto/figures/dataset_oscar_estructura.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
