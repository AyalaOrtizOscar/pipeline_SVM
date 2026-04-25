#!/usr/bin/env python3
"""
Visualizacion 3D de posiciones de microfonos, broca y pieza de trabajo.
Maquina CNC Leadwell V20.

EJES CNC (Leadwell V20):
  X — horizontal izquierda-derecha
  Y — VERTICAL (eje husillo, positivo = arriba)
  Z — horizontal frente-atras (profundidad)
  Plano de mesa = XZ  |  Vista frontal operario = plano YZ

MAPEO A MATPLOTLIB 3D:
  matplotlib X  =  CNC X    (horizontal izquierda-derecha)
  matplotlib Y  =  CNC Z    (profundidad frente-atras)    <- eje 'depth' en mpl
  matplotlib Z  =  CNC Y    (vertical husillo)            <- eje 'up' en mpl
  → Con este mapeo Z_mpl es el eje visual 'arriba':
    el disco queda plano sobre la mesa y el husillo apunta hacia arriba/abajo.

Coordenadas esfericas (metro laser 4-en-1):
  L = distancia [m], a = azimut en plano horizontal CNC XZ [deg],
  b = elevacion desde horizontal [deg]   (b < 0 => apunta hacia abajo)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Constantes ──
TOOL_CONE_TO_TIP_MM = 70.0
DISC_DIAM_MM        = 152.4          # 6 pulgadas
DISC_HEIGHT_MM      = 50.0
DISC_RADIUS         = DISC_DIAM_MM / 2 / 1000   # metros

# ── Medidas desde punto muerto broca → cada microfono (esfericas) ──
# ch1: medida original L=275mm a=24.3 b=-89.0, corregida via cross-check mat→ch1
MICS = {
    'ch0': {'L': 0.360, 'a': -5.2,  'b': -83.7,
            'name': 'MAXLIN UDM-51',   'type': 'dinamico',   'color': '#e74c3c'},
    'ch1': {'L': 0.275, 'a':  24.3,  'b': -89.0,   # overridden in main()
            'name': 'Behringer SL84C', 'type': 'dinamico',   'color': '#2ecc71'},
    'ch2': {'L': 0.251, 'a':   7.4,  'b': -87.8,
            'name': 'Behringer C1',    'type': 'condensador','color': '#3498db'},
}

# Referencia: pto. muerto broca en HOME → centro material
HOME_TO_MATERIAL = {'L': 0.825, 'a': 57.7, 'b': 1.3}

# Cross-checks: desde centro material → microfono
MAT_TO_CH2 = {'L': 0.362, 'a': 72.1, 'b': -17.0}
MAT_TO_CH1 = {'L': 0.440, 'a': 86.3, 'b': -10.1}


# ─────────────────────────────────────────────
def sph2cart_cnc(L, a_deg, b_deg):
    """Esfericas → cartesiano CNC.
    Retorna [cnc_x, cnc_y_vertical, cnc_z_profundidad]."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    horiz = L * np.cos(b)
    return np.array([horiz * np.cos(a),   # CNC X
                     L     * np.sin(b),   # CNC Y (vertical, + arriba)
                     horiz * np.sin(a)])  # CNC Z (profundidad)


def mpl(p_cnc):
    """CNC [x, y_vert, z_prof] → matplotlib [x, y_mpl, z_mpl].
    Mapeo: mpl_x=cnc_x, mpl_y=cnc_z, mpl_z=cnc_y
    → CNC Y (vertical) se convierte en el eje 'up' (Z) de matplotlib."""
    return np.array([p_cnc[0], p_cnc[2], p_cnc[1]])


def draw_disc(ax, center_cnc, radius, height, n=60):
    """Cilindro con eje a lo largo de CNC Y (vertical husillo).
    Caras planas paralelas al plano de mesa (CNC XZ = matplotlib XY). """
    m      = mpl(center_cnc)
    theta  = np.linspace(0, 2 * np.pi, n)
    z_top  = m[2] + height / 2     # mpl Z superior
    z_bot  = m[2] - height / 2     # mpl Z inferior
    x_r    = m[0] + radius * np.cos(theta)
    y_r    = m[1] + radius * np.sin(theta)   # mpl Y = CNC Z (horizontal)

    top = [list(zip(x_r, y_r, [z_top] * n))]
    ax.add_collection3d(Poly3DCollection(
        top, alpha=0.35, facecolor='#CD853F', edgecolor='#8B4513', linewidth=0.6))
    bot = [list(zip(x_r, y_r, [z_bot] * n))]
    ax.add_collection3d(Poly3DCollection(
        bot, alpha=0.20, facecolor='#A0522D', edgecolor='#8B4513', linewidth=0.4))
    for i in range(n - 1):
        side = [[(x_r[i],   y_r[i],   z_bot),
                 (x_r[i],   y_r[i],   z_top),
                 (x_r[i+1], y_r[i+1], z_top),
                 (x_r[i+1], y_r[i+1], z_bot)]]
        ax.add_collection3d(Poly3DCollection(
            side, alpha=0.15, facecolor='#D2B48C', edgecolor='#8B4513', linewidth=0.2))


# ─────────────────────────────────────────────
def main():
    # ── Posiciones CNC (geometria) ──
    mic_pos = {}
    print("=== Posiciones desde punto muerto broca (medidas directas) ===")
    for ch, mic in MICS.items():
        mic_pos[ch] = sph2cart_cnc(mic['L'], mic['a'], mic['b'])
        p = mic_pos[ch]
        note = "  [pendiente correccion]" if ch == 'ch1' else ""
        print(f"  {ch}: X={p[0]*1000:+.1f}mm  Y={p[1]*1000:+.1f}mm  Z={p[2]*1000:+.1f}mm{note}")

    ch2 = mic_pos['ch2']

    # ── Cross-check ch2 → posicion del material ──
    dy_mat_ch2    = MAT_TO_CH2['L'] * np.sin(np.radians(MAT_TO_CH2['b']))
    y_material    = ch2[1] - dy_mat_ch2          # CNC Y del centro del material
    horiz_mat_ch2 = MAT_TO_CH2['L'] * np.cos(np.radians(MAT_TO_CH2['b']))

    print(f"\n=== Cross-check mat\u2192ch2 ===")
    print(f"  \u0394Y (mat\u2192ch2): {dy_mat_ch2*1000:.1f} mm")
    print(f"  Y material:  {y_material*1000:.1f} mm (bajo pto. muerto broca)")
    print(f"  Altura broca sobre material: {-y_material*1000:.1f} mm")
    print(f"  Dist horiz mat\u2192ch2: {horiz_mat_ch2*1000:.1f} mm")

    mat_dir    = np.radians(210)   # direccion arbitraria ch2→material en plano XZ
    mat_center = np.array([
        ch2[0] + horiz_mat_ch2 * np.cos(mat_dir),
        y_material,
        ch2[2] + horiz_mat_ch2 * np.sin(mat_dir),
    ])
    dist_check = np.linalg.norm(mat_center - ch2)
    print(f"  Mat center CNC: X={mat_center[0]*1000:.0f} Y={mat_center[1]*1000:.0f} Z={mat_center[2]*1000:.0f} mm")
    print(f"  \u2714 Dist mat\u2192ch2 recalc: {dist_check*1000:.1f}mm (medida: {MAT_TO_CH2['L']*1000:.0f}mm)")

    # ── Cross-check mat→ch1 → corregir posicion ch1 ──
    cncaz_mat_ch2 = np.degrees(np.arctan2(ch2[2] - mat_center[2],
                                          ch2[0] - mat_center[0]))
    az_offset  = cncaz_mat_ch2 - MAT_TO_CH2['a']      # offset: CNC az = laser az + offset
    cncaz_ch1  = MAT_TO_CH1['a'] + az_offset
    horiz_ch1  = MAT_TO_CH1['L'] * np.cos(np.radians(MAT_TO_CH1['b']))
    dy_ch1     = MAT_TO_CH1['L'] * np.sin(np.radians(MAT_TO_CH1['b']))
    ch1_corr   = np.array([
        mat_center[0] + horiz_ch1 * np.cos(np.radians(cncaz_ch1)),
        y_material    + dy_ch1,
        mat_center[2] + horiz_ch1 * np.sin(np.radians(cncaz_ch1)),
    ])
    L_c = np.linalg.norm(ch1_corr)
    b_c = np.degrees(np.arcsin(np.clip(ch1_corr[1] / L_c, -1, 1)))
    a_c = np.degrees(np.arctan2(ch1_corr[2], ch1_corr[0]))
    mic_pos['ch1']    = ch1_corr
    MICS['ch1']['L']  = round(L_c, 3)
    MICS['ch1']['a']  = round(a_c, 1)
    MICS['ch1']['b']  = round(b_c, 1)
    print(f"\n=== Cross-check mat\u2192ch1 ===")
    print(f"  Az offset: {az_offset:.1f}\u00b0  CNC az mat\u2192ch1: {cncaz_ch1:.1f}\u00b0")
    print(f"  ch1 corr CNC: X={ch1_corr[0]*1000:+.1f}  Y={ch1_corr[1]*1000:+.1f}  Z={ch1_corr[2]*1000:+.1f} mm")
    print(f"  drill\u2192ch1 esf: L={L_c*1000:.0f}mm  a={a_c:.1f}\u00b0  b={b_c:.1f}\u00b0")
    orig_y = -0.275 * np.sin(np.radians(89.0))
    print(f"  (original: L=275mm  a=24.3\u00b0  b=-89.0\u00b0  \u0394Y={( orig_y - ch1_corr[1])*1000:+.1f}mm)")

    # ═══════════════════════════════════════════════
    # ── Figura ──
    fig = plt.figure(figsize=(16, 12))
    ax  = fig.add_subplot(111, projection='3d')

    # --- Punto muerto broca (origen CNC = 0,0,0) ---
    ax.scatter(0, 0, 0, color='black', s=250, marker='v', zorder=10)
    ax.text(0.02, 0.02, 0.02,
            f"Pto. muerto broca\nBPT40 + {TOOL_CONE_TO_TIP_MM:.0f}mm\n6mm Dormer A100",
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                      edgecolor='black', alpha=0.9))

    # Eje husillo: CNC Y → matplotlib Z (eje vertical en la imagen)
    ax.plot([0, 0], [0, 0], [0,  0.10], 'k-',  linewidth=2,   alpha=0.6)  # arriba
    ax.plot([0, 0], [0, 0], [0, -0.06], 'k--', linewidth=1,   alpha=0.4)  # hacia mesa
    ax.text(0, 0, 0.11, "Husillo \u2191 (+Y_cnc)", fontsize=7,
            ha='center', color='gray', style='italic')

    # --- Vectores mic ---
    # Offsets de etiqueta en coordenadas matplotlib
    label_offsets = {
        'ch0': np.array([ 0.05, -0.03,  0.02]),
        'ch1': np.array([-0.06,  0.03,  0.02]),
        'ch2': np.array([ 0.06,  0.03,  0.02]),
    }

    for ch, mic in MICS.items():
        pos_cnc = mic_pos[ch]
        pm      = mpl(pos_cnc)      # coordenadas matplotlib
        c       = mic['color']

        ax.quiver(0, 0, 0, pm[0], pm[1], pm[2],
                  color=c, arrow_length_ratio=0.06, linewidth=2.5, alpha=0.9)
        ax.scatter(*pm, color=c, s=160, zorder=5,
                   edgecolors='black', linewidths=0.8, marker='o')

        off   = label_offsets[ch]
        L_mm  = mic['L'] * 1000
        ax.text(*(pm + off),
                f"{ch} {mic['name']}\n({mic['type']})\n"
                f"L={L_mm:.0f}mm  a={mic['a']}\u00b0  b={mic['b']}\u00b0",
                fontsize=7, ha='center', color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=c, alpha=0.85))

        # Proyeccion vertical desde mic hasta nivel de mesa (mpl Z = CNC Y = y_material)
        ax.plot([pm[0], pm[0]], [pm[1], pm[1]], [pm[2], y_material],
                color=c, linewidth=0.7, linestyle=':', alpha=0.4)
        ax.scatter(pm[0], pm[1], y_material, color=c, s=25, alpha=0.3)

    # --- Disco de material ---
    draw_disc(ax, mat_center, DISC_RADIUS, DISC_HEIGHT_MM / 1000)

    m_mat = mpl(mat_center)
    ax.scatter(*m_mat, color='#8B4513', s=120, marker='s', zorder=5,
               edgecolors='black', linewidths=1)
    ax.text(m_mat[0], m_mat[1], m_mat[2] + 0.04,
            f"AISI 4140\n\u00d8{DISC_DIAM_MM:.0f}mm \u00d7 {DISC_HEIGHT_MM:.0f}mm\n"
            f"Y={mat_center[1]*1000:.0f}mm (CNC)",
            fontsize=8, ha='center', color='#8B4513', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8DC',
                      edgecolor='#8B4513', alpha=0.9))

    # Linea broca → material
    ax.plot([0, m_mat[0]], [0, m_mat[1]], [0, m_mat[2]],
            color='#8B4513', linewidth=1.5, linestyle='--', alpha=0.5)

    # Cross-check mat→ch2
    ch2_m = mpl(ch2)
    ax.plot([m_mat[0], ch2_m[0]], [m_mat[1], ch2_m[1]], [m_mat[2], ch2_m[2]],
            color='#9B59B6', linewidth=2.5, linestyle='-.', alpha=0.8)
    mid2 = (m_mat + ch2_m) / 2
    ax.text(mid2[0] - 0.04, mid2[1] - 0.03, mid2[2],
            f"Cross-check\n{MAT_TO_CH2['L']*1000:.0f}mm  b={MAT_TO_CH2['b']}\u00b0",
            fontsize=7, color='#9B59B6', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F0E0FF',
                      edgecolor='#9B59B6', alpha=0.85))

    # Cross-check mat→ch1
    ch1_m = mpl(mic_pos['ch1'])
    ax.plot([m_mat[0], ch1_m[0]], [m_mat[1], ch1_m[1]], [m_mat[2], ch1_m[2]],
            color='#27ae60', linewidth=2.5, linestyle='-.', alpha=0.8)
    mid1 = (m_mat + ch1_m) / 2
    ax.text(mid1[0] + 0.04, mid1[1] + 0.03, mid1[2],
            f"Cross-check\n{MAT_TO_CH1['L']*1000:.0f}mm  b={MAT_TO_CH1['b']}\u00b0",
            fontsize=7, color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8FFE8',
                      edgecolor='#27ae60', alpha=0.85))

    # --- Superficie de mesa (plano CNC XZ → matplotlib XY a z_mpl = y_table_cnc) ---
    y_table_cnc = mat_center[1] - DISC_HEIGHT_MM / 2 / 1000   # CNC Y = base del disco
    all_x = [m_mat[0], 0, ch2_m[0], ch1_m[0], mpl(mic_pos['ch0'])[0]]
    all_y = [m_mat[1], 0, ch2_m[1], ch1_m[1], mpl(mic_pos['ch0'])[1]]
    ts    = 0.45
    xx, yy_m = np.meshgrid(
        np.linspace(min(all_x) - ts*0.3, max(all_x) + ts*0.2, 2),
        np.linspace(min(all_y) - ts*0.3, max(all_y) + ts*0.2, 2))
    zz = np.full_like(xx, y_table_cnc)   # mpl Z = CNC Y = altura mesa
    ax.plot_surface(xx, yy_m, zz, alpha=0.07, color='#808080')
    ax.text(xx[0, 0], yy_m[0, 0], y_table_cnc,
            "Mesa CNC (plano XZ)", fontsize=7, color='gray', alpha=0.7)

    # --- Flechas de referencia de ejes CNC (en esquina) ---
    corner_cnc = np.array([-0.35, -0.38, -0.2])
    corner     = mpl(corner_cnc)
    al = 0.06
    ax.quiver(*corner,  al,  0,  0, color='red',   linewidth=1.5, arrow_length_ratio=0.15)  # CNC X → mpl X
    ax.quiver(*corner,  0,   al, 0, color='blue',  linewidth=1.5, arrow_length_ratio=0.15)  # CNC Z → mpl Y
    ax.quiver(*corner,  0,   0,  al, color='green', linewidth=1.5, arrow_length_ratio=0.15) # CNC Y → mpl Z
    ax.text(corner[0]+al+0.01, corner[1],        corner[2],        "X",   color='red',   fontsize=8, fontweight='bold')
    ax.text(corner[0],         corner[1]+al+0.01, corner[2],        "Z",   color='blue',  fontsize=8, fontweight='bold')
    ax.text(corner[0],         corner[1],         corner[2]+al+0.01,"Y\u2191",color='green',fontsize=8, fontweight='bold')

    # ── Etiquetas y formato ──
    ax.set_xlabel('CNC X [m] \u2194', fontsize=10, labelpad=10)
    ax.set_ylabel('CNC Z [m] \u2194 (profundidad)', fontsize=10, labelpad=10)
    ax.set_zlabel('CNC Y [m] \u2195 vertical (husillo)', fontsize=10, labelpad=10)
    ax.set_title(
        'Leadwell V20 — Setup de medicion\n'
        'CNC: X=hor, Y=vertical(husillo\u2193), Z=prof  |  Broca 6mm Dormer A100 | BPT40+70mm | Sujecion: bridas',
        fontsize=10, fontweight='bold', pad=15)

    # Limites de ejes (proporcionales)
    all_mpl = np.array([mpl(p) for p in list(mic_pos.values()) + [mat_center, np.zeros(3)]])
    max_r   = np.abs(all_mpl).max() * 1.3
    ax.set_xlim(-max_r,       max_r)
    ax.set_ylim(-max_r,       max_r)
    ax.set_zlim(-max_r * 1.1, max_r * 0.4)

    # Leyenda
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=m['color'], linewidth=2.5,
                      label=f"{ch} {m['name']} ({m['L']*1000:.0f}mm)")
               for ch, m in MICS.items()]
    handles += [
        Line2D([0], [0], color='#8B4513', linewidth=1.5, linestyle='--',  label="Disco AISI 4140"),
        Line2D([0], [0], color='#9B59B6', linewidth=2,   linestyle='-.',
               label=f"Cross-check mat\u2192ch2 ({MAT_TO_CH2['L']*1000:.0f}mm)"),
        Line2D([0], [0], color='#27ae60', linewidth=2,   linestyle='-.',
               label=f"Cross-check mat\u2192ch1 ({MAT_TO_CH1['L']*1000:.0f}mm)"),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.9)

    # Caja de informacion
    info = [
        f"Cono BPT40 \u2192 pto. muerto: {TOOL_CONE_TO_TIP_MM:.0f} mm (ref. broca 6mm)",
        f"Home \u2192 material: L={HOME_TO_MATERIAL['L']*1000:.0f}mm "
        f"a={HOME_TO_MATERIAL['a']}\u00b0 b={HOME_TO_MATERIAL['b']}\u00b0",
        "",
    ]
    for ch, mic in MICS.items():
        p = mic_pos[ch]
        info.append(f"  {ch} ({mic['name']}): L={mic['L']*1000:.0f}mm "
                    f"a={mic['a']}\u00b0 b={mic['b']}\u00b0  "
                    f"\u2192 X={p[0]*1000:+.0f} Y={p[1]*1000:+.0f} Z={p[2]*1000:+.0f} mm")
    info += [
        "",
        f"Cross-check mat\u2192ch2: L={MAT_TO_CH2['L']*1000:.0f}mm "
        f"a={MAT_TO_CH2['a']}\u00b0 b={MAT_TO_CH2['b']}\u00b0",
        f"Cross-check mat\u2192ch1: L={MAT_TO_CH1['L']*1000:.0f}mm "
        f"a={MAT_TO_CH1['a']}\u00b0 b={MAT_TO_CH1['b']}\u00b0  "
        f"(orig: L=275mm a=24.3\u00b0 b=-89.0\u00b0)",
        f"Altura broca sobre material: {-y_material*1000:.0f} mm",
        f"Sujecion: bridas | Fixture: BPT40",
    ]
    fig.text(0.01, 0.01, "\n".join(info), fontsize=6.5,
             fontfamily='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    plt.tight_layout()

    # ── Guardar vistas ──
    # Vista 1: perspectiva
    ax.view_init(elev=25, azim=-55)
    out1 = "D:/pipeline_SVM/results/mic_vectors_3d.png"
    fig.savefig(out1, dpi=180, bbox_inches='tight')
    print(f"Saved: {out1}")

    # Vista 2: frontal (operario mira a lo largo de CNC Z = mpl Y)
    ax.view_init(elev=10, azim=0)
    out2 = "D:/pipeline_SVM/results/mic_vectors_3d_front_YZ.png"
    fig.savefig(out2, dpi=180, bbox_inches='tight')
    print(f"Saved: {out2}")

    # Vista 3: desde arriba (plano de mesa CNC XZ)
    ax.view_init(elev=90, azim=-90)
    out3 = "D:/pipeline_SVM/results/mic_vectors_3d_top_XZ.png"
    fig.savefig(out3, dpi=180, bbox_inches='tight')
    print(f"Saved: {out3}")

    plt.show()


if __name__ == '__main__':
    main()
