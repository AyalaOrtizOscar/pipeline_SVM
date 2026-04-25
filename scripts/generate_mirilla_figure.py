#!/usr/bin/env python3
"""
Genera fig_cap9_mirilla.png para el informe.

Panel A: frame test53 mas nítido con linea de calibración (amarillo) y
         medida de chisel edge wear (rojo) anotadas.
Panel B-D: tres frames de test39 (evt_00, evt_04, evt_08) mostrando
           progresión visual — temprano, medio, tardío.
"""
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

CLIPS   = Path('D:/pipeline_SVM/article2/camera/clips')
EVENTS  = Path('D:/pipeline_SVM/article2/camera/events')
OUT     = Path('D:/pipeline_SVM/informe_proyecto/figures/fig_cap9_mirilla.png')

# ── Panel A: test53 frame 001 con wear medido ────────────────────────────────
# Calibración test53: 164.47 px/mm (mediana de 2 lineas)
# Wear: p1=(135,469), p2=(218,407) → 103.6 px → 0.630 mm

def draw_annotation(img, p1, p2, color, label, offset=(8, -8)):
    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    mid = ((p1[0]+p2[0])//2 + offset[0], (p1[1]+p2[1])//2 + offset[1])
    cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_scalebar(img, px_per_mm, bar_mm=0.5):
    h, w = img.shape[:2]
    bar_px = int(bar_mm * px_per_mm)
    x0, y0 = 20, h - 30
    x1 = x0 + bar_px
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f'{bar_mm} mm', (x0, y0-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)

def load_and_annotate_test53():
    path = CLIPS / 'test53' / '001_t540s_sh327_refl0.007.png'
    img = cv2.imread(str(path))
    if img is None:
        return None
    # scale to 800 wide
    h, w = img.shape[:2]
    scale = 800 / w
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    s = scale

    # Calibration line (from test53/000)
    # cal p1=(226,253), p2=(169,408) scaled
    cal_p1 = (int(226*s), int(253*s))
    cal_p2 = (int(169*s), int(408*s))
    draw_annotation(img, cal_p1, cal_p2, (0,220,220), '1 mm ref', offset=(8, -5))

    # Wear line (from test53/001)
    # vb p1=(135,469), p2=(218,407) scaled
    vb_p1 = (int(135*s), int(469*s))
    vb_p2 = (int(218*s), int(407*s))
    draw_annotation(img, vb_p1, vb_p2, (0,40,255), '0.63 mm', offset=(8, 18))

    # Scale bar using test53 px/mm (164.47), adjusted for scale factor
    draw_scalebar(img, 164.47 * s, bar_mm=0.5)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_event(test, evt_dir_name, label):
    path = EVENTS / test / evt_dir_name / 'best.png'
    img = cv2.imread(str(path))
    if img is None:
        return None, label
    h, w = img.shape[:2]
    scale = 600 / w
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Label overlay
    hh, ww = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, hh-32), (ww, hh), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    cv2.putText(img, label, (8, hh-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,150), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), label


def main():
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a1a')

    # Row 1: Panel A (large, annotated measurement)
    ax_a = fig.add_subplot(2, 4, (1, 2))
    img_a = load_and_annotate_test53()
    if img_a is not None:
        ax_a.imshow(img_a)
    ax_a.set_title('(A) Medición chisel edge wear — test53\nh≈20 agujeros, wear = 0.63 mm',
                   color='white', fontsize=9, pad=4)
    ax_a.axis('off')

    # Row 1 panels B-D: test39 progression
    evt_sequence = [
        ('test39', 'evt_00_t00811s', '(B) test39 evt_00\nt≈13 min  h≈15'),
        ('test39', 'evt_03_t01940s', '(C) test39 evt_03\nt≈32 min  h≈45'),
        ('test39', 'evt_08_t03955s', '(D) test39 evt_08\nt≈66 min  h≈93'),
    ]
    for col, (test, evt, lbl) in enumerate(evt_sequence):
        ax = fig.add_subplot(2, 4, col + 3)
        img_evt, _ = load_event(test, evt, lbl.split('\n')[1])
        if img_evt is not None:
            ax.imshow(img_evt)
        ax.set_title(lbl, color='#ffcc88', fontsize=8, pad=4)
        ax.axis('off')

    # Row 2: test50 and test53 events
    bottom_evts = [
        ('test50', 'evt_00_t00360s', '(E) test50 evt_00\nt≈5 min'),
        ('test50', 'evt_04_t02339s', '(F) test50 evt_04\nt≈38 min'),
        ('test53', 'evt_00_t00549s', '(G) test53 evt_00\nt≈9 min'),
        ('test54', 'evt_01_t02495s', '(H) test54 evt_01\nt≈41 min'),
    ]
    for col, (test, evt, lbl) in enumerate(bottom_evts):
        ax = fig.add_subplot(2, 4, col + 5)
        img_evt, _ = load_event(test, evt, lbl.split('\n')[1])
        if img_evt is not None:
            ax.imshow(img_evt)
        ax.set_title(lbl, color='#aabbff', fontsize=8, pad=4)
        ax.axis('off')

    fig.suptitle(
        'Inspección visual del filo mediante mirilla óptica — detección automática de paradas (test39: 9 eventos)',
        color='white', fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
