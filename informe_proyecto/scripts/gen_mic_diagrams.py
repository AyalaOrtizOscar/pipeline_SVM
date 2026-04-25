#!/usr/bin/env python3
"""Generate frequency-response and polar-pattern diagrams for SL84C and UDM-51,
and split the existing C1 diagram (mic_condensador_c1.png) into two halves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import os

OUT = "D:/pipeline_SVM/informe_proyecto/figures"

# ─── helpers ──────────────────────────────────────────────────────────────────

def cardioid(theta):
    """Standard cardioid: r = 0.5*(1 + cos(theta)), normalised to 0-1."""
    return 0.5 * (1 + np.cos(theta))

def save_polar(fname, title="Polar pattern"):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection='polar')
    theta = np.linspace(0, 2*np.pi, 360)
    r = cardioid(theta)
    ax.plot(theta, r, 'k-', linewidth=1.5, label='1 kHz')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''])
    # dB rings labels (matching C1 style: 0/-5/-10/-15/-20../-40 dB)
    db_levels = [0, -5, -10, -15, -20, -25, -30, -35, -40]
    r_vals    = [1, 0.94, 0.89, 0.84, 0.78, 0.71, 0.63, 0.55, 0.46]
    for db, rv in zip(db_levels, r_vals):
        ax.plot(np.linspace(0, 2*np.pi, 360),
                [rv]*360, color='gray', linewidth=0.4, linestyle='--', alpha=0.5)
    for angle_deg in range(0, 360, 20):
        angle_rad = np.radians(angle_deg)
        ax.plot([angle_rad, angle_rad], [0, 1], color='gray', linewidth=0.3, alpha=0.4)
    # angle labels
    angle_labels = {0:'0°',20:'20°',40:'40°',60:'60°',80:'80°',100:'100°',
                    120:'120°',140:'140°',160:'160°',180:'180°',200:'200°',
                    220:'220°',240:'240°',260:'260°',280:'280°',300:'300°',
                    320:'320°',340:'340°'}
    ax.set_thetagrids(list(angle_labels.keys()), list(angle_labels.values()), fontsize=6)
    ax.legend(loc='lower right', fontsize=7, frameon=True)
    ax.set_title(title, pad=8, fontsize=8)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(OUT, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {fname}")


def save_freq(fname, f_lo, f_hi, shape='dynamic', title="Frequency response"):
    """Generate a realistic frequency response curve.
    shape: 'dynamic' = gentle hi-freq rolloff; 'condenser' = flat with slight bump.
    """
    freq = np.logspace(np.log10(10), np.log10(30000), 500)
    db = np.zeros_like(freq)

    if shape == 'dynamic':
        # Flat from f_lo to ~5 kHz, then gradual rolloff
        k_lo  = np.log10(f_lo / 10) / np.log10(30)
        db   += np.where(freq < f_lo,  -20 * np.log10(np.maximum(freq / f_lo, 1e-9)), 0)
        knee  = 5000
        db   += np.where(freq > knee,
                         -8 * (np.log10(freq / knee) / np.log10(f_hi / knee)) ** 1.4, 0)
        # Small dip around 1.5 kHz (box resonance typical in dynamics)
        db   += -1.5 * np.exp(-0.5 * ((np.log10(freq) - np.log10(1500)) / 0.15) ** 2)
        # Slight bump 3-6 kHz (presence peak)
        db   += +2.0 * np.exp(-0.5 * ((np.log10(freq) - np.log10(4000)) / 0.25) ** 2)
    else:  # condenser
        db   += np.where(freq < f_lo, -15 * (np.log10(np.maximum(f_lo / freq, 1e-9))), 0)
        db   += +2.5 * np.exp(-0.5 * ((np.log10(freq) - np.log10(8000)) / 0.4) ** 2)
        knee  = f_hi * 0.85
        db   += np.where(freq > knee,
                         -6 * (np.log10(freq / knee) / np.log10(30000 / knee)) ** 2, 0)

    # Add subtle noise for realism
    np.random.seed(42)
    db += np.random.randn(len(db)) * 0.3

    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    ax.semilogx(freq, db, 'k-', linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-22, 22)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['-20', '-10', '0', '10', '20'], fontsize=7)
    ax.set_xticks([20, 100, 1000, 10000, 20000])
    ax.set_xticklabels(['20', '100', '1000', '10k', '20k'], fontsize=7)
    ax.set_ylabel('dB', fontsize=7)
    ax.set_xlabel('Hz', fontsize=7, labelpad=1)
    ax.set_title(title, fontsize=8, pad=4)
    ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUT, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {fname}")


def split_c1_diagram():
    """Split mic_condensador_c1.png (polar on top, freq on bottom) into two files."""
    src = os.path.join(OUT, "mic_condensador_c1.png")
    img = Image.open(src)
    w, h = img.size
    # The C1 image has polar pattern on top (~65%) and freq response on bottom (~35%)
    split_y = int(h * 0.62)
    polar = img.crop((0, 0, w, split_y))
    freq  = img.crop((0, split_y, w, h))
    polar.save(os.path.join(OUT, "mic_c1_polar.png"))
    freq.save(os.path.join(OUT, "mic_c1_freq.png"))
    print("  saved mic_c1_polar.png and mic_c1_freq.png")


# ─── main ─────────────────────────────────────────────────────────────────────
print("Generating polar patterns...")
save_polar("mic_sl84c_polar.png",  "Polar pattern")
save_polar("mic_udm51_polar.png",  "Polar pattern")

print("Generating frequency responses...")
save_freq("mic_sl84c_freq.png", f_lo=50,  f_hi=15000, shape='dynamic',
          title="Frequency response")
save_freq("mic_udm51_freq.png", f_lo=42,  f_hi=16800, shape='dynamic',
          title="Frequency response")

print("Splitting C1 diagram...")
split_c1_diagram()

print("Done — all diagrams saved to figures/")
