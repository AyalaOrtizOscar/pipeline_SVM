#!/usr/bin/env python3
"""
article2_figures_v2.py

Regenera las 3 figuras del Art.2 que quedaron ilegibles en la version 1:
  - fig_feature_correlation.png  -> top-20 features (matriz 20x20 con labels grandes)
  - fig_shap_bar_C1_C2.png       -> top-15 features por panel, layout ancho
  - fig_mi_ranking.png           -> eje tipografia 10pt, figura mas alta

Tambien regenera las figuras cuyo titulo contenia la palabra "umbral",
"falla" o "fallo" para alinear con el vocabulario unificado:
  sin desgaste / medianamente desgastado / desgastado
  region de desgaste / region de etiquetado (no "umbral")

Salida:  D:/pipeline_SVM/results/article2_figures/
"""
import sys, os, warnings
from pathlib import Path
sys.path.insert(0, "D:/pipeline_SVM/scripts")
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC

RANDOM_STATE = 42
MERGED = Path("D:/pipeline_SVM/features/features_multimodal_merged.csv")
OUT = Path("D:/pipeline_SVM/results/article2_figures")
OUT.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = [
    "duration_s", "rms", "rms_db", "peak", "zcr", "mel_total_energy",
    "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_entropy_mean", "spectral_contrast_mean",
    "mfcc_0_mean", "mfcc_1_mean", "chroma_mean_first", "chroma_mean",
    "chroma_std", "tonnetz_0_mean", "harmonic_percussive_ratio",
    "tempo", "onset_rate", "crest_factor",
]
ESP32_FEATURES = ["esp_rms", "esp_rms_db", "esp_centroid_mean", "esp_zcr",
                  "esp_spectral_contrast_mean", "esp_crest_factor"]
FLOW_FEATURES = ["flow_mean_lmin", "flow_std_lmin", "flow_min_lmin",
                 "flow_duty_pulses", "flow_cv"]
COATING_FEATURE = ["coating_coded"]
ALL_MM = AUDIO_FEATURES + ESP32_FEATURES + FLOW_FEATURES + COATING_FEATURE

LABEL_TO_IDX = {"sin_desgaste": 0, "medianamente_desgastado": 1, "desgastado": 2}
IDX_TO_LABEL_ES = {0: "sin desgaste", 1: "medianamente desgastado", 2: "desgastado"}

CLASS_COLORS = {0: "#4a8fd9", 1: "#e09b3d", 2: "#c24a4a"}
BLOCK_COLORS = {"audio": "#4a6fa5", "esp32": "#6fa54a", "flow": "#a04a6f",
                "coating": "#a0884a"}

# Nombres amigables para las figuras (el simbolo tecnico queda en la leyenda)
NICE = {
    "duration_s": "duracion (s)",
    "rms": "RMS",
    "rms_db": "RMS (dB)",
    "peak": "pico",
    "zcr": "ZCR",
    "mel_total_energy": "energia mel",
    "centroid_mean": "centroide",
    "centroid_std": "centroide (std)",
    "rolloff_mean": "rolloff",
    "rolloff_std": "rolloff (std)",
    "spectral_flatness_mean": "planitud",
    "spectral_flatness_std": "planitud (std)",
    "spectral_bandwidth_mean": "ancho banda",
    "spectral_bandwidth_std": "ancho banda (std)",
    "spectral_entropy_mean": "entropia espectral",
    "spectral_contrast_mean": "contraste espectral",
    "mfcc_0_mean": "MFCC 0",
    "mfcc_1_mean": "MFCC 1",
    "chroma_mean_first": "chroma 1",
    "chroma_mean": "chroma (media)",
    "chroma_std": "chroma (std)",
    "tonnetz_0_mean": "tonnetz 0",
    "harmonic_percussive_ratio": "arm/perc",
    "tempo": "tempo",
    "onset_rate": "tasa onsets",
    "crest_factor": "factor cresta",
    "esp_rms": "ESP RMS",
    "esp_rms_db": "ESP RMS (dB)",
    "esp_centroid_mean": "ESP centroide",
    "esp_zcr": "ESP ZCR",
    "esp_spectral_contrast_mean": "ESP contraste",
    "esp_crest_factor": "ESP cresta",
    "flow_mean_lmin": "caudal medio",
    "flow_std_lmin": "caudal (std)",
    "flow_min_lmin": "caudal minimo",
    "flow_duty_pulses": "caudal duty",
    "flow_cv": "caudal CV",
    "coating_coded": "recubrimiento",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


def block_of(feat):
    if feat in AUDIO_FEATURES: return "audio"
    if feat in ESP32_FEATURES: return "esp32"
    if feat in FLOW_FEATURES: return "flow"
    if feat in COATING_FEATURE: return "coating"
    return "audio"


def load_data():
    df = pd.read_csv(MERGED)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["y"] = df["label"].map(LABEL_TO_IDX)
    return df


def make_pipe(k):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("svc", SVC(C=10, kernel="rbf", gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ])


def train_fh(X_tr, y_tr, features, k):
    y1 = (y_tr >= 1).astype(int)
    y2 = (y_tr >= 2).astype(int)
    p1 = make_pipe(k).fit(X_tr[features], y1)
    p2 = make_pipe(k).fit(X_tr[features], y2)
    return p1, p2


def make_split(df):
    train = df[df["experiment"] != "E3"].copy()
    test = df[df["experiment"] == "E3"].copy()
    return train, test


# ====== FIG: MI ranking (v2) ======
def fig_mi_ranking_v2(df):
    X = df[ALL_MM].copy()
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    y = df["y"].values
    mi = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE)
    order = np.argsort(mi)[::-1]
    features = [ALL_MM[i] for i in order]
    values = mi[order]
    labels = [NICE[f] for f in features]
    colors = [BLOCK_COLORS[block_of(f)] for f in features]

    fig, ax = plt.subplots(figsize=(8.8, 10.5))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Informacion mutua (frente al estado de desgaste ordinal)",
                  fontsize=11)
    ax.set_title("Ranking de informacion mutua (38 descriptores multimodales)",
                 fontsize=13, pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.55)
    ax.tick_params(axis="x", labelsize=10)

    # value labels at tip
    for yp, v in zip(y_pos, values):
        ax.text(v + max(values) * 0.01, yp, f"{v:.3f}",
                va="center", fontsize=8.5, color="#444")

    handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=b.upper())
               for b in ["audio", "esp32", "flow", "coating"]]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              fontsize=10, title="Bloque sensor",
              title_fontsize=10)
    plt.tight_layout()
    p = OUT / "fig_mi_ranking.png"
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ====== FIG: Correlation heatmap (v2) ======
def fig_feature_correlation_v2(df, top_n=20):
    """Selecciona top-N por MI y grafica matriz de correlacion NxN."""
    X = df[ALL_MM].copy()
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                         columns=ALL_MM)
    y = df["y"].values
    mi = mutual_info_classif(X_imp.values, y, random_state=RANDOM_STATE)
    top_idx = np.argsort(mi)[::-1][:top_n]
    feats_top = [ALL_MM[i] for i in top_idx]
    # reordena: primero audio, luego esp32, flow, coating para bandas visibles
    order_bucket = {"audio": 0, "esp32": 1, "flow": 2, "coating": 3}
    feats_top.sort(key=lambda f: (order_bucket[block_of(f)], -mi[ALL_MM.index(f)]))

    C = X_imp[feats_top].corr().values
    labels = [NICE[f] for f in feats_top]
    colors = [BLOCK_COLORS[block_of(f)] for f in feats_top]

    fig, ax = plt.subplots(figsize=(12.5, 11.0))
    cmap = LinearSegmentedColormap.from_list(
        "diff", ["#c24a4a", "#ffffff", "#4a6fa5"])
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(feats_top)))
    ax.set_yticks(range(len(feats_top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # color tick labels by block
    for tl, f in zip(ax.get_xticklabels(), feats_top):
        tl.set_color(BLOCK_COLORS[block_of(f)])
    for tl, f in zip(ax.get_yticklabels(), feats_top):
        tl.set_color(BLOCK_COLORS[block_of(f)])

    # draw block separators
    bloques = [block_of(f) for f in feats_top]
    for idx in range(1, len(bloques)):
        if bloques[idx] != bloques[idx - 1]:
            ax.axhline(idx - 0.5, color="black", linewidth=0.9)
            ax.axvline(idx - 0.5, color="black", linewidth=0.9)

    # annotate correlations
    for i in range(len(feats_top)):
        for j in range(len(feats_top)):
            v = C[i, j]
            if abs(v) >= 0.3:
                txt = f"{v:.2f}"
                col = "white" if abs(v) > 0.7 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=col)

    ax.set_title(
        f"Correlacion de Pearson entre los {top_n} descriptores de mayor informacion mutua",
        fontsize=13, pad=14)
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Coeficiente de Pearson", fontsize=11)

    handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=b.upper())
               for b in ["audio", "esp32", "flow", "coating"]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              frameon=True, fontsize=10, title="Bloque", title_fontsize=10)

    plt.tight_layout()
    p = OUT / "fig_feature_correlation.png"
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ====== FIG: SHAP bar v2 (top-15 per panel, friendly names) ======
def fig_shap_bar_v2(train, test, top_n=15):
    import shap
    feats = ALL_MM
    k = 22
    p1, p2 = train_fh(train, train["y"].values, feats, k)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5))
    for ax, (p, title) in zip(axes, [(p1, "C$_1$: transicion a estado desgastado"),
                                      (p2, "C$_2$: decision final (estado desgastado)")]):
        imputer = p.named_steps["imputer"]
        scaler = p.named_steps["scaler"]
        selector = p.named_steps["select"]
        svc = p.named_steps["svc"]

        X_tr_sel = selector.transform(scaler.transform(imputer.transform(train[feats])))
        X_te_sel = selector.transform(scaler.transform(imputer.transform(test[feats])))
        sel_mask = selector.get_support()
        sel_names = [feats[i] for i in range(len(feats)) if sel_mask[i]]

        bg = shap.sample(X_tr_sel, 80, random_state=RANDOM_STATE)
        rng = np.random.RandomState(RANDOM_STATE)
        sample_te = X_te_sel[rng.choice(len(X_te_sel),
                                         size=min(80, len(X_te_sel)),
                                         replace=False)]
        explainer = shap.KernelExplainer(svc.predict_proba, bg)
        sv = explainer.shap_values(sample_te, nsamples=80, silent=True)
        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            sv_pos = sv[..., 1] if sv.ndim == 3 else sv
        mean_abs = np.abs(sv_pos).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        names_ord = [sel_names[i] for i in order]
        nice_ord = [NICE.get(n, n) for n in names_ord]
        vals = mean_abs[order]
        colors = [BLOCK_COLORS[block_of(n)] for n in names_ord]
        y_pos = np.arange(len(names_ord))
        bars = ax.barh(y_pos, vals, color=colors,
                        edgecolor="black", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nice_ord, fontsize=11)
        # color labels by block
        for tl, n in zip(ax.get_yticklabels(), names_ord):
            tl.set_color(BLOCK_COLORS[block_of(n)])
        ax.invert_yaxis()
        ax.set_xlabel("|SHAP| medio", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(axis="x", linestyle=":", alpha=0.55)
        ax.tick_params(axis="x", labelsize=10)

        for yp, v in zip(y_pos, vals):
            ax.text(v + max(vals) * 0.015, yp, f"{v:.3f}",
                    va="center", fontsize=9, color="#333")

    handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=b.upper())
               for b in ["audio", "esp32", "flow", "coating"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.02), fontsize=11,
               title="Bloque sensor", title_fontsize=11)
    fig.suptitle("Importancia SHAP por clasificador Frank-Hall "
                 "(top-15, multimodal de 38 descriptores)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p = OUT / "fig_shap_bar_C1_C2.png"
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("OK", p)


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")
    train, test = make_split(df)
    print(f"Train: {len(train)}  Test (E3): {len(test)}")

    fig_mi_ranking_v2(df)
    fig_feature_correlation_v2(df, top_n=20)
    fig_shap_bar_v2(train, test, top_n=15)
    print("\nAll v2 figures saved to:", OUT)


if __name__ == "__main__":
    main()
