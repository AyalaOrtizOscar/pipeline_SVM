#!/usr/bin/env python3
"""
article2_figures.py

Genera el portafolio de figuras diagnosticas del Art.2 multimodal sobre
el dataset unificado (Orejarena 2014 + Lote II UIS 2026). Las figuras
son analogas a las del Art.1 pero adaptadas al clasificador multimodal
de 38 descriptores.

Figuras producidas (en results/article2_figures/):
  fig_class_distribution.png   — distribucion de clases por fuente
  fig_pca_2d.png               — PCA 2D con 3 clases ordinales
  fig_mi_ranking.png           — ranking MI destacando multimodales
  fig_flow_vs_wear.png         — boxplots de caudal por clase de desgaste
  fig_confusion_comparison.png — 3 matrices de confusion (audio/+coat/MM)
  fig_shap_bar_C1_C2.png       — SHAP |mean| para ambos F&H
  fig_learning_curve.png       — macro F1 vs tamano de entrenamiento
  fig_calibration.png          — calibracion por clase (multimodal)
  fig_cross_experiment.png     — heatmap LOEO por experimento
  fig_feature_correlation.png  — heatmap correlacion features multimodales
"""
import sys, os, json, warnings
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
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score,
                              brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

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
IDX_TO_LABEL_ES = {0: "sin desgaste", 1: "medianamente\ndesgastado", 2: "desgastado"}

CLASS_COLORS = {0: "#4a8fd9", 1: "#e09b3d", 2: "#c24a4a"}
BLOCK_COLORS = {"audio": "#4a6fa5", "esp32": "#6fa54a", "flow": "#a04a6f",
                "coating": "#a0884a"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
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


def frank_hall_decode(p1, p2):
    # monotonicity
    p2 = np.minimum(p1, p2)
    p0 = 1 - p1
    pmid = p1 - p2
    P = np.stack([p0, pmid, p2], axis=1)
    return np.argmax(P, axis=1), P


def train_fh(X_tr, y_tr, features, k):
    y1 = (y_tr >= 1).astype(int)
    y2 = (y_tr >= 2).astype(int)
    p1 = make_pipe(k).fit(X_tr[features], y1)
    p2 = make_pipe(k).fit(X_tr[features], y2)
    return p1, p2


def predict_fh(p1, p2, X, features):
    pp1 = p1.predict_proba(X[features])[:, 1]
    pp2 = p2.predict_proba(X[features])[:, 1]
    yhat, P = frank_hall_decode(pp1, pp2)
    return yhat, P, pp1, pp2


# ========== SPLIT ==========
def make_split(df):
    """Train = todo menos E3; Test = E3 (retencion historica)."""
    train = df[df["experiment"] != "E3"].copy()
    test = df[df["experiment"] == "E3"].copy()
    return train, test


# ========== FIG 1: distribucion de clases ==========
def fig_class_distribution(df):
    df = df.copy()
    df["fuente"] = np.where(df["source"] == "orejarena", "Orejarena (2014)",
                             "Lote II (UIS, 2026)")
    ct = (df.groupby(["fuente", "label"]).size()
            .unstack(fill_value=0)
            .reindex(columns=list(LABEL_TO_IDX.keys())))
    labels_es = [IDX_TO_LABEL_ES[LABEL_TO_IDX[c]].replace("\n", " ") for c in ct.columns]

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(ct.index))
    width = 0.25
    max_val = 0
    for i, (lab, col) in enumerate(zip(ct.columns, labels_es)):
        vals = ct[lab].values
        max_val = max(max_val, vals.max())
        bars = ax.bar(x + (i-1)*width, vals, width,
                      color=CLASS_COLORS[LABEL_TO_IDX[lab]],
                      edgecolor="black", linewidth=0.6, label=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max_val*0.015,
                    str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, fontsize=10)
    ax.set_ylabel("Numero de observaciones")
    ax.set_title("Distribucion de clases ordinales por fuente")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, max_val * 1.15)
    ax.legend(title="Clase ordinal", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    p = OUT / "fig_class_distribution.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 2: PCA 2D ==========
def fig_pca_2d(df):
    X = df[ALL_MM].copy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X)
    y = df["y"].values
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for k in [0, 1, 2]:
        m = (y == k)
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.55,
                   c=CLASS_COLORS[k],
                   label=IDX_TO_LABEL_ES[k].replace("\n", " "),
                   edgecolors="none")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)")
    ax.set_title("PCA 2D sobre el vector multimodal (38 descriptores)")
    ax.legend(title="Clase ordinal", loc="best", frameon=True, markerscale=2)
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    p = OUT / "fig_pca_2d.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 3: MI ranking ==========
def fig_mi_ranking(df):
    X = df[ALL_MM].copy()
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    y = df["y"].values
    mi = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE)
    order = np.argsort(mi)[::-1]
    features = [ALL_MM[i] for i in order]
    values = mi[order]
    colors = [BLOCK_COLORS[block_of(f)] for f in features]

    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Informacion mutua contra la etiqueta ordinal")
    ax.set_title("Ranking de informacion mutua (38 descriptores)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    handles = [mpatches.Patch(color=c, label=b.upper())
               for b, c in BLOCK_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    plt.tight_layout()
    p = OUT / "fig_mi_ranking.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 4: flow vs wear (boxplots) ==========
def fig_flow_vs_wear(df):
    dff = df[df[FLOW_FEATURES].notna().any(axis=1)].copy()
    fig, axes = plt.subplots(1, len(FLOW_FEATURES),
                              figsize=(14, 3.6), sharey=False)
    for ax, feat in zip(axes, FLOW_FEATURES):
        data = [dff.loc[dff["y"] == k, feat].dropna().values
                for k in [0, 1, 2]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                         medianprops=dict(color="black", linewidth=1.2),
                         showfliers=False)
        for patch, k in zip(bp["boxes"], [0, 1, 2]):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        ax.set_xticklabels(["sin", "medio", "desgast."], fontsize=9)
        ax.set_title(feat.replace("flow_", ""), fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Descriptores de caudal YF-S201 por clase ordinal "
                 f"(Lote II, n={len(dff)})", fontsize=11, y=1.02)
    plt.tight_layout()
    p = OUT / "fig_flow_vs_wear.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 5: confusion matrices side-by-side ==========
def fig_confusion_comparison(train, test):
    specs = [
        ("A. Audio (26)", AUDIO_FEATURES, 15),
        ("B. Audio + coating (27)", AUDIO_FEATURES + COATING_FEATURE, 16),
        ("C. Multimodal (38)", ALL_MM, 22),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    ytrue = test["y"].values
    cmap = LinearSegmentedColormap.from_list("bl", ["#ffffff", "#284b82"])
    for ax, (title, feats, k) in zip(axes, specs):
        p1, p2 = train_fh(train, train["y"].values, feats, k)
        yhat, _, _, _ = predict_fh(p1, p2, test, feats)
        cm = confusion_matrix(ytrue, yhat, labels=[0, 1, 2])
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels([IDX_TO_LABEL_ES[i] for i in range(3)], fontsize=8)
        ax.set_yticklabels([IDX_TO_LABEL_ES[i] for i in range(3)], fontsize=8)
        for i in range(3):
            for j in range(3):
                txt = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Verdad")
        # metrics summary
        adj = np.mean(np.abs(ytrue - yhat) <= 1)
        f1m = f1_score(ytrue, yhat, average="macro")
        ax.set_title(f"{title}\nadj={adj:.3f}  F1mac={f1m:.3f}", fontsize=10)
    fig.suptitle("Matrices de confusion normalizadas por fila — retencion E3 (n=583)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    p = OUT / "fig_confusion_comparison.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 6: SHAP bar for C1 and C2 ==========
def fig_shap_bar(train, test):
    import shap
    feats = ALL_MM
    k = 22
    p1, p2 = train_fh(train, train["y"].values, feats, k)

    # Use feature names after selection
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for ax, (p, title) in zip(axes, [(p1, "C1 (cualquier desgaste)"),
                                      (p2, "C2 (desgastado)")]):
        imputer = p.named_steps["imputer"]
        scaler = p.named_steps["scaler"]
        selector = p.named_steps["select"]
        svc = p.named_steps["svc"]

        X_tr_sel = selector.transform(scaler.transform(imputer.transform(train[feats])))
        X_te_sel = selector.transform(scaler.transform(imputer.transform(test[feats])))
        sel_mask = selector.get_support()
        sel_names = [feats[i] for i in range(len(feats)) if sel_mask[i]]

        # use small background for kernel explainer
        bg = shap.sample(X_tr_sel, 80, random_state=RANDOM_STATE)
        sample_te = X_te_sel[np.random.RandomState(RANDOM_STATE).choice(
            len(X_te_sel), size=min(80, len(X_te_sel)), replace=False)]
        explainer = shap.KernelExplainer(svc.predict_proba, bg)
        sv = explainer.shap_values(sample_te, nsamples=80, silent=True)
        # newer shap returns (N, F, C) array
        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            sv_pos = sv[..., 1] if sv.ndim == 3 else sv
        mean_abs = np.abs(sv_pos).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        names_ord = [sel_names[i] for i in order]
        vals = mean_abs[order]
        colors = [BLOCK_COLORS[block_of(n)] for n in names_ord]
        y_pos = np.arange(len(names_ord))
        ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_ord, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|SHAP| medio")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    handles = [mpatches.Patch(color=c, label=b.upper())
               for b, c in BLOCK_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Importancia SHAP por clasificador Frank-Hall (multimodal)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    p = OUT / "fig_shap_bar_C1_C2.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 7: learning curve ==========
def fig_learning_curve(train):
    X = train[ALL_MM]
    y = train["y"].values
    # Use a compound metric via SVC on 3-class directly for learning curve
    pipe = make_pipe(22)
    # learning_curve accepts scoring macro_f1
    train_sizes = np.linspace(0.2, 1.0, 6)
    train_sizes_abs, tr_scores, cv_scores = learning_curve(
        pipe, X, y, cv=4, train_sizes=train_sizes, scoring="f1_macro",
        n_jobs=-1, random_state=RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    tr_mean = tr_scores.mean(axis=1); tr_std = tr_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1); cv_std = cv_scores.std(axis=1)
    ax.plot(train_sizes_abs, tr_mean, "o:", color="#4a6fa5",
            label="Entrenamiento", linewidth=1.8, markersize=6)
    ax.fill_between(train_sizes_abs, tr_mean - tr_std, tr_mean + tr_std,
                     alpha=0.18, color="#4a6fa5")
    ax.plot(train_sizes_abs, cv_mean, "s--", color="#c24a4a",
            label="Validacion (CV=4)", linewidth=1.8, markersize=6)
    ax.fill_between(train_sizes_abs, cv_mean - cv_std, cv_mean + cv_std,
                     alpha=0.18, color="#c24a4a")
    ax.set_xlabel("Numero de muestras de entrenamiento")
    ax.set_ylabel("F1 macro")
    ax.set_title("Curva de aprendizaje — clasificador multimodal")
    ax.grid(linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p = OUT / "fig_learning_curve.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 8: calibration ==========
def fig_calibration(train, test):
    feats = ALL_MM
    k = 22
    p1, p2 = train_fh(train, train["y"].values, feats, k)
    _, P, pp1, pp2 = predict_fh(p1, p2, test, feats)
    ytrue = test["y"].values
    # per-class one-vs-rest probabilities
    probs = {
        0: 1 - pp1,   # P(sin_desgaste)
        1: pp1 - pp2, # P(medianamente)
        2: pp2,       # P(desgastado)
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], "k-", linewidth=1.2, label="Calibracion perfecta")
    for k_, color in CLASS_COLORS.items():
        y_bin = (ytrue == k_).astype(int)
        pr = probs[k_]
        frac_pos, mean_pred = calibration_curve(y_bin, pr, n_bins=10,
                                                 strategy="quantile")
        brier = brier_score_loss(y_bin, pr)
        ax.plot(mean_pred, frac_pos, "o--", color=color, linewidth=1.6,
                markersize=6,
                label=f"{IDX_TO_LABEL_ES[k_].replace(chr(10), ' ')} "
                      f"(Brier={brier:.3f})")
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fraccion observada de positivos")
    ax.set_title("Curva de calibracion por clase — retencion E3")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    p = OUT / "fig_calibration.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


# ========== FIG 9: cross-experiment (LOEO) ==========
def fig_cross_experiment(df):
    """LOEO sobre experimentos Orejarena (E1..E7) — multimodal vs audio."""
    oj = df[df["source"] == "orejarena"].copy()
    exps = sorted(oj["experiment"].dropna().unique())
    feats_audio = AUDIO_FEATURES
    feats_mm = ALL_MM
    M = np.zeros((len(exps), 2))
    for i, test_exp in enumerate(exps):
        tr = oj[oj["experiment"] != test_exp]
        te = oj[oj["experiment"] == test_exp]
        for j, (feats, k) in enumerate([(feats_audio, 15), (feats_mm, 22)]):
            try:
                p1, p2 = train_fh(tr, tr["y"].values, feats, k)
                yhat, _, _, _ = predict_fh(p1, p2, te, feats)
                adj = np.mean(np.abs(te["y"].values - yhat) <= 1)
            except Exception:
                adj = np.nan
            M[i, j] = adj

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    cmap = LinearSegmentedColormap.from_list("gr", ["#c24a4a", "#ffffff", "#4a8fd9"])
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Audio (26)", "Multimodal (38)"])
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps)
    for i in range(len(exps)):
        for j in range(2):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if v < 0.7 else "black")
    ax.set_title("Exactitud adyacente — LOEO sobre experimentos Orejarena")
    plt.colorbar(im, ax=ax, label="Exactitud adyacente")
    plt.tight_layout()
    p = OUT / "fig_cross_experiment.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)
    return pd.DataFrame(M, index=exps, columns=["audio", "multimodal"])


# ========== FIG 10: correlation heatmap ==========
def fig_feature_correlation(df):
    X = df[ALL_MM].copy()
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                         columns=ALL_MM)
    C = X_imp.corr().values

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    cmap = LinearSegmentedColormap.from_list("diff",
                                              ["#c24a4a", "#ffffff", "#4a6fa5"])
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(ALL_MM)))
    ax.set_yticks(range(len(ALL_MM)))
    ax.set_xticklabels(ALL_MM, rotation=90, fontsize=7)
    ax.set_yticklabels(ALL_MM, fontsize=7)
    # block separators
    boundaries = [len(AUDIO_FEATURES),
                   len(AUDIO_FEATURES) + len(ESP32_FEATURES),
                   len(AUDIO_FEATURES) + len(ESP32_FEATURES) + len(FLOW_FEATURES)]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=0.7)
        ax.axvline(b - 0.5, color="black", linewidth=0.7)
    ax.set_title("Correlacion de Pearson entre descriptores multimodales")
    plt.colorbar(im, ax=ax, fraction=0.035)
    plt.tight_layout()
    p = OUT / "fig_feature_correlation.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("OK", p)


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")
    train, test = make_split(df)
    print(f"Train: {len(train)}  Test (E3): {len(test)}")

    fig_class_distribution(df)
    fig_pca_2d(df)
    fig_mi_ranking(df)
    fig_flow_vs_wear(df)
    fig_confusion_comparison(train, test)
    fig_learning_curve(train)
    fig_calibration(train, test)
    loeo = fig_cross_experiment(df)
    loeo.to_csv(OUT / "loeo_results.csv")
    fig_feature_correlation(df)
    # SHAP last because slowest
    fig_shap_bar(train, test)
    print("\nAll figures saved to:", OUT)


if __name__ == "__main__":
    main()
