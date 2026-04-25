#!/usr/bin/env python3
"""Genera figura comparativa: audio_only vs audio+coating vs multimodal."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("D:/pipeline_SVM/results/multimodal_comparison")
reports = sorted(OUT.glob("multimodal_report_*.json"))
r = json.loads(reports[-1].read_text())

variants = list(r['variants'].keys())
labels = {'A_audio_only': 'Solo audio (26)',
          'B_audio_coating': 'Audio + recubrimiento (27)',
          'C_full_multimodal': 'Multimodal completo (38)'}
metrics = ['adjacent_accuracy', 'macro_f1', 'exact_accuracy']
metric_labels = {'adjacent_accuracy': 'Exactitud adyacente',
                 'macro_f1': 'F1 macro',
                 'exact_accuracy': 'Exactitud exacta'}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
x = np.arange(len(variants))
colors = ['#4a6fa5', '#a04a6f', '#6fa54a']
for ax, m in zip(axes, metrics):
    vals = [r['variants'][v]['e3_metrics'][m] for v in variants]
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[v] for v in variants], rotation=15, ha='right', fontsize=9)
    ax.set_title(metric_labels[m], fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15)

fig.suptitle('Ablacion multimodal en el holdout E3 (n=583)', fontsize=12, y=0.99)
plt.tight_layout()
out_png = OUT / 'multimodal_ablation_E3.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')

# Figura 2: LODO A114
fig, ax = plt.subplots(figsize=(7, 4.5))
bits = sorted({b for v in r['variants'].values() for b in v['lodo_a114'].keys()})
for vi, v in enumerate(variants):
    vals = [r['variants'][v]['lodo_a114'].get(b, {}).get('adjacent_accuracy', np.nan)
            for b in bits]
    offset = (vi - 1) * 0.25
    ax.bar(np.arange(len(bits)) + offset, vals, width=0.23,
           label=labels[v], color=colors[vi], edgecolor='black', linewidth=0.8)
ax.set_xticks(np.arange(len(bits)))
ax.set_xticklabels(bits, fontsize=9)
ax.set_ylabel('Exactitud adyacente (LODO)')
ax.set_title('Validacion fuera de dominio: brocas A114 excluidas del entrenamiento')
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
out_png = OUT / 'multimodal_lodo_A114.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')
