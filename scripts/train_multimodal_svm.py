#!/usr/bin/env python3
"""
train_multimodal_svm.py — Entrena tres variantes del SVM ordinal Frank & Hall
para cuantificar el aporte de las variables multimodales:

  A) audio_only      — 26 features acústicos NI (línea base, igual al iter_006)
  B) audio+coating   — +1 feature categórica (recubrimiento A100/A114)
  C) full_multimodal — +6 features ESP32 + 5 flow + coating (38 features)

Evalúa en:
  - E3 holdout (Orejarena) — métrica comparable con iteraciones previas
  - Leave-One-Drill-Out (LODO) sobre brocas A114 — test de generalización fuera
    del dominio del corpus heredado
"""
import sys, os, json, time, warnings
sys.path.insert(0, "D:/pipeline_SVM/scripts")
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL,
    ordinal_decode, ordinal_mae, adjacent_accuracy,
    ordinal_confusion_matrix, print_ordinal_report,
)

RANDOM_STATE = 42
MERGED_CSV = Path("D:/pipeline_SVM/features/features_multimodal_merged.csv")
ITER_DIR = Path("D:/pipeline_SVM/results/retrain_iterations")
OUT_DIR = Path("D:/pipeline_SVM/results/multimodal_comparison")
GUI_MODELS_DIR = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/models")

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
ESP32_FEATURES = [
    "esp_rms", "esp_rms_db", "esp_centroid_mean", "esp_zcr",
    "esp_spectral_contrast_mean", "esp_crest_factor",
]
FLOW_FEATURES = [
    "flow_mean_lmin", "flow_std_lmin", "flow_min_lmin",
    "flow_duty_pulses", "flow_cv",
]
COATING_FEATURE = ["coating_coded"]


def make_pipe(k):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(mutual_info_classif, k=k)),
        ('svc', SVC(C=10, kernel='rbf', gamma='scale',
                    probability=True, class_weight='balanced',
                    random_state=RANDOM_STATE)),
    ])


def train_frank_hall(X_train, y_train_int, k):
    y1 = (y_train_int >= 1).astype(int)
    y2 = (y_train_int >= 2).astype(int)
    p1 = make_pipe(k); p1.fit(X_train, y1)
    p2 = make_pipe(k); p2.fit(X_train, y2)
    return p1, p2


def predict(p1, p2, X):
    pr1 = p1.predict_proba(X)[:, 1]
    pr2 = p2.predict_proba(X)[:, 1]
    pr2 = np.minimum(pr2, pr1)
    return ordinal_decode(np.stack([pr1, pr2], axis=1)), pr1, pr2


def eval_split(y_true, y_pred):
    return {
        'n': int(len(y_true)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'exact_accuracy': float(accuracy_score(y_true, y_pred)),
        'adjacent_accuracy': float(adjacent_accuracy(y_true, y_pred)),
        'ordinal_mae': float(ordinal_mae(y_true, y_pred)),
        'confusion_matrix': ordinal_confusion_matrix(y_true, y_pred).tolist(),
    }


def run_variant(df, feat_set, label):
    """Entrena y evalúa una variante de feature set."""
    present = [f for f in feat_set if f in df.columns]
    print(f"\n{'='*72}\n[{label}]  {len(present)} features\n{'='*72}")

    df_train = df[df['split'] != 'test'].copy()
    df_test_e3 = df[df['split'] == 'test'].copy()

    X_tr = df_train[present].values.astype(np.float32)
    y_tr = np.array([LABEL_TO_IDX[l] for l in df_train['label']])
    X_te = df_test_e3[present].values.astype(np.float32)
    y_te = np.array([LABEL_TO_IDX[l] for l in df_test_e3['label']])

    k = min(15 if 'audio' in label.lower() or 'only' in label.lower() else len(present), len(present))
    # Usamos k=15 para la base; para variantes ampliadas, k = min(22, len(present))
    if label == 'A_audio_only':
        k = 15
    elif label == 'B_audio_coating':
        k = 16
    elif label == 'C_full_multimodal':
        k = min(22, len(present))

    t0 = time.time()
    p1, p2 = train_frank_hall(X_tr, y_tr, k=k)
    t_train = time.time() - t0

    # E3 holdout
    y_pred, _, _ = predict(p1, p2, X_te)
    m_e3 = eval_split(y_te, y_pred)

    # LODO sobre brocas A114: hold out cada broca A114 una vez
    lodo_results = {}
    if 'drill_bit' in df.columns:
        a114_bits = [b for b in df['drill_bit'].dropna().unique() if 'A114' in str(b)]
        for bit in a114_bits:
            mask_hold = (df['drill_bit'] == bit)
            if mask_hold.sum() < 10:
                continue
            df_tr = df[~mask_hold & (df['split'] != 'test')]
            df_ho = df[mask_hold]
            Xh_tr = df_tr[present].values.astype(np.float32)
            yh_tr = np.array([LABEL_TO_IDX[l] for l in df_tr['label']])
            Xh_te = df_ho[present].values.astype(np.float32)
            yh_te = np.array([LABEL_TO_IDX[l] for l in df_ho['label']])
            p1h, p2h = train_frank_hall(Xh_tr, yh_tr, k=k)
            yp, _, _ = predict(p1h, p2h, Xh_te)
            lodo_results[bit] = eval_split(yh_te, yp)

    print(f"  Train: {len(X_tr)}  Test E3: {len(X_te)}  k={k}  t_fit={t_train:.1f}s")
    print(f"  E3 -> adj={m_e3['adjacent_accuracy']:.4f}  macro_F1={m_e3['macro_f1']:.4f}  "
          f"exact={m_e3['exact_accuracy']:.4f}  MAE={m_e3['ordinal_mae']:.4f}")
    for bit, m in lodo_results.items():
        print(f"  LODO[{bit}]  n={m['n']}  adj={m['adjacent_accuracy']:.4f}  "
              f"macro_F1={m['macro_f1']:.4f}")

    return {
        'variant': label,
        'feature_set': present,
        'k': k,
        'train_n': int(len(X_tr)),
        'e3_metrics': m_e3,
        'lodo_a114': lodo_results,
        'pipelines': (p1, p2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MERGED_CSV}")
    df = pd.read_csv(MERGED_CSV)
    print(f"  {len(df)} rows, {len(df.columns)} cols")

    variants = {
        'A_audio_only':     AUDIO_FEATURES,
        'B_audio_coating':  AUDIO_FEATURES + COATING_FEATURE,
        'C_full_multimodal': AUDIO_FEATURES + ESP32_FEATURES + FLOW_FEATURES + COATING_FEATURE,
    }

    results = {}
    for name, fs in variants.items():
        results[name] = run_variant(df, fs, name)

    # Tabla comparativa
    print(f"\n{'='*72}\n  RESUMEN COMPARATIVO — E3 HOLDOUT\n{'='*72}")
    print(f"{'Variante':<22} {'n_feat':>7} {'adj_acc':>9} {'macro_F1':>9} "
          f"{'exact':>7} {'MAE':>7}")
    for name, r in results.items():
        m = r['e3_metrics']
        print(f"{name:<22} {len(r['feature_set']):>7} "
              f"{m['adjacent_accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
              f"{m['exact_accuracy']:>7.4f} {m['ordinal_mae']:>7.4f}")

    print(f"\n{'='*72}\n  RESUMEN LODO A114\n{'='*72}")
    print(f"{'Variante':<22} {'broca':<18} {'n':>5} {'adj_acc':>9} {'macro_F1':>9}")
    for name, r in results.items():
        for bit, m in r['lodo_a114'].items():
            print(f"{name:<22} {bit:<18} {m['n']:>5} "
                  f"{m['adjacent_accuracy']:>9.4f} {m['macro_f1']:>9.4f}")

    # Guardar artefactos
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'timestamp': ts,
        'dataset_size': int(len(df)),
        'variants': {
            name: {
                'feature_set': r['feature_set'],
                'k': r['k'],
                'train_n': r['train_n'],
                'e3_metrics': r['e3_metrics'],
                'lodo_a114': r['lodo_a114'],
            } for name, r in results.items()
        }
    }
    report_path = OUT_DIR / f'multimodal_report_{ts}.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")

    # Guardar el mejor modelo multimodal (C) como iter_007
    iter_dir = ITER_DIR / 'iter_007'
    iter_dir.mkdir(parents=True, exist_ok=True)
    p1, p2 = results['C_full_multimodal']['pipelines']
    joblib.dump(p1, iter_dir / 'svm_C1_any_wear_multimodal.joblib')
    joblib.dump(p2, iter_dir / 'svm_C2_severe_wear_multimodal.joblib')
    # También guardamos A_audio_only para comparación directa
    p1a, p2a = results['A_audio_only']['pipelines']
    joblib.dump(p1a, iter_dir / 'svm_C1_any_wear_audio_only.joblib')
    joblib.dump(p2a, iter_dir / 'svm_C2_severe_wear_audio_only.joblib')

    iter_metrics = {
        'iteration': 7,
        'timestamp': ts,
        'trigger_test': 'multimodal_comparison',
        'dataset_size': int(len(df)),
        'variants_compared': list(variants.keys()),
        'best_variant': 'C_full_multimodal',
        'e3_metrics_audio_only': results['A_audio_only']['e3_metrics'],
        'e3_metrics_audio_coating': results['B_audio_coating']['e3_metrics'],
        'e3_metrics_multimodal': results['C_full_multimodal']['e3_metrics'],
        'lodo_a114_audio_only': results['A_audio_only']['lodo_a114'],
        'lodo_a114_multimodal': results['C_full_multimodal']['lodo_a114'],
        'feature_set_multimodal': results['C_full_multimodal']['feature_set'],
    }
    (iter_dir / 'metrics.json').write_text(json.dumps(iter_metrics, indent=2))
    print(f"Iter_007 saved: {iter_dir}")

    # Deploy to GUI (multimodal as primary if it wins)
    m_c = results['C_full_multimodal']['e3_metrics']['adjacent_accuracy']
    m_a = results['A_audio_only']['e3_metrics']['adjacent_accuracy']
    winner = 'C' if m_c >= m_a else 'A'
    print(f"\nMejor variante en E3 (adj_acc): {winner}")
    if GUI_MODELS_DIR.exists():
        target_c1 = GUI_MODELS_DIR / 'svm_C1_multimodal.joblib'
        target_c2 = GUI_MODELS_DIR / 'svm_C2_multimodal.joblib'
        joblib.dump(p1, target_c1)
        joblib.dump(p2, target_c2)
        print(f"Deployed multimodal to {GUI_MODELS_DIR}")


if __name__ == '__main__':
    main()
