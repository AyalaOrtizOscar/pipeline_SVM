#!/usr/bin/env python3
"""
retrain_after_test.py — Iterative retraining pipeline for Art.2 weekend drilling sessions.

After each test:
  1. Segment continuous WAV into individual holes (CutRatio spectral)
  2. Extract 26 acoustic features per channel per hole
  3. Assign wear labels via [15/75] life-fraction threshold
  4. Merge Orejarena (Art.1) + all Art.2 feature CSVs
  5. Retrain SVM (Frank & Hall ordinal, fixed hyperparams → ~30s)
  6. Evaluate on E3 holdout + compare with previous iterations
  7. Deploy updated models to GUI_v5

Usage:
  python retrain_after_test.py --test 40 --drill-bit "6mm#5" --total-holes 95
  python retrain_after_test.py --test 39 --drill-bit "6mm#4" --total-holes 110
  python retrain_after_test.py --reprocess-all
  python retrain_after_test.py --skip-segment --test 39 --drill-bit "6mm#4" --total-holes 110
"""

import sys, os, argparse, json, time, shutil, warnings
sys.path.insert(0, "D:/pipeline_SVM/scripts")
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL,
    ordinal_encode, ordinal_decode, ordinal_proba,
    ordinal_mae, adjacent_accuracy, ordinal_confusion_matrix,
    print_ordinal_report
)
from extract_features_from_manifests import extract_features

# ── Paths ────────────────────────────────────────────────────────────────
OREJARENA_CSV   = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
ART2_FEAT_DIR   = Path("D:/pipeline_SVM/features/art2")
ART2_SEG_DIR    = Path("D:/pipeline_SVM/art2_segments")
ITER_DIR        = Path("D:/pipeline_SVM/results/retrain_iterations")
GUI_MODELS_DIR  = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/models")
C_DATA_BASE     = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados")
E_DATA_BASE     = Path("E:/Datos Generados")

REGISTRY_PATH   = ART2_FEAT_DIR / "drill_bit_registry.json"

# ── Feature order (must match SVM_FEATURE_NAMES in optimized_realtime_prediction.py) ──
SVM_FEATURE_NAMES = [
    "duration_s", "rms", "rms_db", "peak", "zcr", "mel_total_energy",
    "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_entropy_mean", "spectral_contrast_mean",
    "mfcc_0_mean", "mfcc_1_mean", "chroma_mean_first", "chroma_mean",
    "chroma_std", "tonnetz_0_mean", "harmonic_percussive_ratio",
    "tempo", "onset_rate", "crest_factor",
]

# Channel → mic_type mapping for Art.2
CH_MIC_MAP = {
    'ch0': 'dinamico',      # MAXLIN UDM-51
    'ch1': 'dinamico',      # Behringer SL84C
    'ch2': 'condensador',   # Behringer C1
}

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════
# Stage 0: Resolve test paths
# ═══════════════════════════════════════════════════════════════════════════

def resolve_test_paths(test_num):
    """Find test data folder on C: or E: drive."""
    folder_name = f"6mm_test{test_num}"
    # Check C: first (newer tests), then E:
    for base, prefix in [(C_DATA_BASE, 'C'), (E_DATA_BASE, 'E')]:
        test_dir = base / folder_name
        if test_dir.exists():
            # Check for WAV files
            wavs = list(test_dir.glob("ch*.wav"))
            if wavs:
                print(f"  [Stage 0] Found {folder_name} on {prefix}: ({len(wavs)} WAVs)")
                return test_dir, prefix, folder_name
            # Check NI subfolder
            ni_wavs = list((test_dir / "NI").glob("ch*.wav")) if (test_dir / "NI").exists() else []
            if ni_wavs:
                print(f"  [Stage 0] Found {folder_name} on {prefix}: NI/ ({len(ni_wavs)} WAVs)")
                return test_dir, prefix, folder_name
    raise FileNotFoundError(f"No WAV data found for test{test_num} in C: or E:")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Segment test
# ═══════════════════════════════════════════════════════════════════════════

def segment_test(test_num, drill_bit, total_holes, hole_start, test_dir, drive_prefix):
    """Run CutRatio spectral segmentation on a test."""
    import segment_holes_art2 as seg

    # Monkey-patch base path to where the data lives
    if drive_prefix == 'C':
        seg.E_BASE = C_DATA_BASE
    else:
        seg.E_BASE = E_DATA_BASE

    test_id = f"test{test_num}"
    folder_name = f"6mm_test{test_num}"
    notes = f"Broca {drill_bit}, {total_holes} holes, retrain pipeline"

    print(f"\n  [Stage 1] Segmenting {test_id} ({folder_name})...")
    manifest = seg.process_test(test_id, folder_name, drill_bit, hole_start, total_holes, notes)

    if manifest is None:
        raise RuntimeError(f"Segmentation returned None for {test_id}")

    # Rename output to use correct prefix
    src_dir = ART2_SEG_DIR / f"E_{test_id}"
    dst_dir = ART2_SEG_DIR / f"{drive_prefix}_{test_id}"
    if src_dir.exists() and drive_prefix != 'E':
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        src_dir.rename(dst_dir)
        print(f"  Renamed E_{test_id} -> {drive_prefix}_{test_id}")

    n_holes = len(manifest['holes'])
    print(f"  [Stage 1] Done: {n_holes} holes detected (expected ~{total_holes})")
    return manifest


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Extract features
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_for_test(test_num, manifest, drill_bit, total_holes, drive_prefix):
    """Extract 26 features per channel per hole. Cache as CSV."""
    ART2_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ART2_FEAT_DIR / f"test{test_num}_features.csv"

    test_id = f"test{test_num}"
    seg_dir = ART2_SEG_DIR / f"{drive_prefix}_{test_id}"

    rows = []
    n_holes = len(manifest['holes'])
    channels = manifest.get('channels', ['ch0', 'ch1', 'ch2'])

    print(f"\n  [Stage 2] Extracting features for {n_holes} holes x {len(channels)} channels...")
    t0 = time.time()

    for i, hole in enumerate(manifest['holes']):
        hole_id = hole['hole_id']
        hole_num = hole['hole_number']

        for ch in channels:
            wav_key = ch
            if wav_key not in hole.get('files', {}):
                continue

            wav_rel = hole['files'][wav_key]
            wav_path = ART2_SEG_DIR / wav_rel

            # Handle renamed directories (E_testNN -> C_testNN or vice versa)
            if not wav_path.exists():
                # Try replacing prefix in path
                rel_str = str(wav_rel)
                for old_pfx, new_pfx in [('E_test', 'C_test'), ('C_test', 'E_test')]:
                    if old_pfx in rel_str:
                        alt_rel = rel_str.replace(old_pfx, new_pfx, 1)
                        alt_path = ART2_SEG_DIR / alt_rel
                        if alt_path.exists():
                            wav_path = alt_path
                            break

            if not wav_path.exists():
                print(f"    [SKIP] {wav_path} not found")
                continue

            # Skip very short segments
            dur_s = hole.get('duration_s', 0)
            if dur_s < 1.5:
                print(f"    [SKIP] {hole_id}_{ch}: {dur_s:.1f}s < 1.5s minimum")
                continue

            feats = extract_features(str(wav_path))
            if feats is None or '_error' in feats:
                err = feats.get('_error', 'unknown') if feats else 'None returned'
                print(f"    [ERROR] {hole_id}_{ch}: {err}")
                continue

            # Build row with metadata + features
            mic_type = CH_MIC_MAP.get(ch, 'dinamico')
            row = {
                'filepath': str(wav_path),
                'label': '',  # assigned in Stage 3
                'split': 'art2_train',
                'aug_type': 'original',
                'mic_type': mic_type,
                'experiment': f'art2_{test_id}',
                'test_id': test_id,
                'hole_id': hole_id,
                'hole_number': hole_num,
                'channel': ch,
                'drill_bit': drill_bit,
                'total_holes_bit': total_holes,
                'drive': drive_prefix,
            }
            # Add 26 features
            for fname in SVM_FEATURE_NAMES:
                row[fname] = feats.get(fname, np.nan)

            rows.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == n_holes:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{n_holes}] {elapsed:.0f}s elapsed")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0
    print(f"  [Stage 2] Done: {len(df)} samples saved to {csv_path.name} ({elapsed:.0f}s)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Assign wear labels
# ═══════════════════════════════════════════════════════════════════════════

def assign_wear_labels(df, total_holes):
    """Assign labels via [15/75] life-fraction threshold."""
    print(f"\n  [Stage 3] Assigning wear labels (total_holes={total_holes}, threshold=[15/75])...")

    labels = []
    for _, row in df.iterrows():
        h = row['hole_number']
        frac = h / max(total_holes, 1)
        if frac <= 0.15:
            labels.append('sin_desgaste')
        elif frac <= 0.75:
            labels.append('medianamente_desgastado')
        else:
            labels.append('desgastado')
    df['label'] = labels

    counts = df['label'].value_counts().to_dict()
    print(f"  [Stage 3] Labels: {counts}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Merge all features
# ═══════════════════════════════════════════════════════════════════════════

def merge_all_features():
    """Merge Orejarena CSV + all Art.2 CSVs into a single training DataFrame."""
    print(f"\n  [Stage 4] Merging all feature CSVs...")

    # Load Orejarena
    df_ore = pd.read_csv(OREJARENA_CSV, low_memory=False)
    # Keep only needed columns
    meta_keep = ['filepath', 'label', 'split', 'aug_type', 'mic_type', 'experiment']
    ore_meta = [c for c in meta_keep if c in df_ore.columns]
    ore_feats = [c for c in SVM_FEATURE_NAMES if c in df_ore.columns]
    # Also keep wavelet_total_energy if present (alias for mel_total_energy)
    df_ore = df_ore[ore_meta + ore_feats].copy()
    print(f"  Orejarena: {len(df_ore)} rows ({df_ore['label'].value_counts().to_dict()})")

    # Load all Art.2 CSVs
    art2_csvs = sorted(ART2_FEAT_DIR.glob("test*_features.csv"))
    dfs_art2 = []
    for csv_path in art2_csvs:
        df_a = pd.read_csv(csv_path, low_memory=False)
        if len(df_a) == 0:
            continue
        # Ensure it has labels
        if df_a['label'].isna().all() or (df_a['label'] == '').all():
            print(f"  [WARN] {csv_path.name} has no labels — skipping")
            continue
        dfs_art2.append(df_a)
        print(f"  Art.2 {csv_path.name}: {len(df_a)} rows ({df_a['label'].value_counts().to_dict()})")

    if not dfs_art2:
        print("  [WARN] No Art.2 CSVs with labels found — training with Orejarena only")
        return df_ore

    df_art2 = pd.concat(dfs_art2, ignore_index=True)
    # Keep only columns present in Orejarena + features
    common_cols = ore_meta + ore_feats
    art2_cols = [c for c in common_cols if c in df_art2.columns]
    df_art2 = df_art2[art2_cols].copy()

    # Concatenate
    merged = pd.concat([df_ore, df_art2], ignore_index=True)
    print(f"  [Stage 4] Merged: {len(merged)} rows total")
    print(f"    Orejarena: {len(df_ore)} | Art.2: {len(df_art2)}")
    print(f"    Labels: {merged['label'].value_counts().to_dict()}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: Retrain SVM (fixed hyperparams — fast)
# ═══════════════════════════════════════════════════════════════════════════

def retrain_svm(df_merged):
    """Train Frank & Hall SVM with fixed hyperparams. ~30s."""
    print(f"\n  [Stage 5] Retraining SVM (Frank & Hall, C=10, rbf, k=15)...")
    t0 = time.time()

    # Split: train = everything except E3 test
    df_train = df_merged[df_merged['split'] != 'test'].copy()
    df_test = df_merged[df_merged['split'] == 'test'].copy()

    # Art.2 samples go into train
    # (their split is 'art2_train', not 'test')

    print(f"  Train: {len(df_train)} | Test (E3): {len(df_test)}")

    # Prepare X, y
    feat_cols = [c for c in SVM_FEATURE_NAMES if c in df_train.columns]
    X_train = df_train[feat_cols].values.astype(np.float32)
    X_test = df_test[feat_cols].values.astype(np.float32) if len(df_test) > 0 else None

    y_train_int = np.array([LABEL_TO_IDX[l] for l in df_train['label']])
    y_test_int = np.array([LABEL_TO_IDX[l] for l in df_test['label']]) if len(df_test) > 0 else None

    print(f"  Train dist: {dict(zip(*np.unique(y_train_int, return_counts=True)))}")
    if y_test_int is not None:
        print(f"  Test dist:  {dict(zip(*np.unique(y_test_int, return_counts=True)))}")

    # Build fixed pipeline (no grid search)
    def make_pipe():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k=min(15, len(feat_cols)))),
            ('svc', SVC(C=10, kernel='rbf', gamma='scale',
                        probability=True, class_weight='balanced',
                        random_state=RANDOM_STATE)),
        ])

    # C1: P(y >= 1) — any wear
    y_c1 = (y_train_int >= 1).astype(int)
    pipe_c1 = make_pipe()
    pipe_c1.fit(X_train, y_c1)

    # C2: P(y >= 2) — severe wear
    y_c2 = (y_train_int >= 2).astype(int)
    pipe_c2 = make_pipe()
    pipe_c2.fit(X_train, y_c2)

    elapsed = time.time() - t0
    print(f"  [Stage 5] SVM trained in {elapsed:.1f}s")

    # Predict on test
    metrics = {}
    if X_test is not None and len(X_test) > 0:
        p1 = pipe_c1.predict_proba(X_test)[:, 1]
        p2 = pipe_c2.predict_proba(X_test)[:, 1]
        p2 = np.minimum(p2, p1)  # monotonicity
        probs = np.stack([p1, p2], axis=1)
        y_pred = ordinal_decode(probs)

        metrics = {
            'macro_f1': float(f1_score(y_test_int, y_pred, average='macro', zero_division=0)),
            'exact_accuracy': float(accuracy_score(y_test_int, y_pred)),
            'adjacent_accuracy': float(adjacent_accuracy(y_test_int, y_pred)),
            'ordinal_mae': float(ordinal_mae(y_test_int, y_pred)),
        }
        cm = ordinal_confusion_matrix(y_test_int, y_pred)
        metrics['two_step_errors'] = int(cm[0, 2] + cm[2, 0])
        metrics['confusion_matrix'] = cm.tolist()

        print(f"\n  === E3 Holdout Metrics ===")
        print(f"  macro_F1:          {metrics['macro_f1']:.4f}")
        print(f"  exact_accuracy:    {metrics['exact_accuracy']:.4f}")
        print(f"  adjacent_accuracy: {metrics['adjacent_accuracy']:.4f}")
        print(f"  ordinal_MAE:       {metrics['ordinal_mae']:.4f}")
        print(f"  two_step_errors:   {metrics['two_step_errors']}")
        print_ordinal_report(y_test_int, y_pred)

    return pipe_c1, pipe_c2, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: Evaluate and compare iterations
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_and_compare(metrics, df_merged, test_num):
    """Save metrics and compare with previous iterations."""
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    # Find next iteration number
    existing = sorted(ITER_DIR.glob("iter_*"))
    iter_num = len(existing) + 1
    iter_dir = ITER_DIR / f"iter_{iter_num:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Compute dataset stats
    n_orejarena = len(df_merged[df_merged['experiment'].str.startswith('E') |
                                 df_merged['experiment'].str.startswith('art1') |
                                 ~df_merged['experiment'].str.startswith('art2')])
    n_art2 = len(df_merged[df_merged['experiment'].str.startswith('art2')])

    info = {
        'iteration': iter_num,
        'timestamp': datetime.now().isoformat(),
        'trigger_test': f'test{test_num}' if test_num else 'reprocess_all',
        'dataset_size': len(df_merged),
        'n_orejarena': int(n_orejarena),
        'n_art2': int(n_art2),
        'label_distribution': df_merged['label'].value_counts().to_dict(),
        'metrics_E3': metrics,
    }

    # Compare with previous iteration
    if existing:
        prev_metrics_path = existing[-1] / "metrics.json"
        if prev_metrics_path.exists():
            with open(prev_metrics_path) as f:
                prev = json.load(f)
            prev_m = prev.get('metrics_E3', {})
            if prev_m and metrics:
                delta = {k: metrics[k] - prev_m.get(k, 0)
                         for k in ['macro_f1', 'exact_accuracy', 'adjacent_accuracy']
                         if k in metrics}
                delta['ordinal_mae'] = metrics.get('ordinal_mae', 0) - prev_m.get('ordinal_mae', 0)
                info['delta_vs_previous'] = delta

                print(f"\n  === Delta vs iteration {iter_num - 1} ===")
                for k, v in delta.items():
                    direction = "+" if v > 0 else ""
                    better = "BETTER" if (v > 0 and k != 'ordinal_mae') or \
                                         (v < 0 and k == 'ordinal_mae') else \
                             ("WORSE" if v != 0 else "SAME")
                    print(f"  {k:25s}: {direction}{v:.4f} [{better}]")

    # Save
    with open(iter_dir / "metrics.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Generate learning curve data
    _update_learning_curve(ITER_DIR)

    print(f"\n  [Stage 6] Iteration {iter_num} saved to {iter_dir}")
    return iter_num


def _update_learning_curve(iter_dir):
    """Build learning_curve.csv from all iterations."""
    rows = []
    for d in sorted(iter_dir.glob("iter_*")):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            info = json.load(f)
        m = info.get('metrics_E3', {})
        rows.append({
            'iteration': info.get('iteration', 0),
            'timestamp': info.get('timestamp', ''),
            'trigger': info.get('trigger_test', ''),
            'dataset_size': info.get('dataset_size', 0),
            'n_art2': info.get('n_art2', 0),
            'macro_f1': m.get('macro_f1', np.nan),
            'exact_accuracy': m.get('exact_accuracy', np.nan),
            'adjacent_accuracy': m.get('adjacent_accuracy', np.nan),
            'ordinal_mae': m.get('ordinal_mae', np.nan),
        })
    if rows:
        pd.DataFrame(rows).to_csv(iter_dir / "learning_curve.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7: Deploy models
# ═══════════════════════════════════════════════════════════════════════════

def deploy_models(pipe_c1, pipe_c2, iter_num):
    """Deploy SVM models to GUI_v5 and save in results."""
    print(f"\n  [Stage 7] Deploying models...")

    # Save to results
    results_dir = ITER_DIR / f"iter_{iter_num:03d}"
    joblib.dump(pipe_c1, results_dir / "svm_C1_any_wear.joblib")
    joblib.dump(pipe_c2, results_dir / "svm_C2_severe_wear.joblib")
    print(f"  Saved to {results_dir}")

    # Backup existing GUI models
    if GUI_MODELS_DIR.exists():
        for name in ['svm_ordinal_C1.joblib', 'svm_ordinal_C2.joblib']:
            src = GUI_MODELS_DIR / name
            if src.exists():
                bak = GUI_MODELS_DIR / f"{name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(src, bak)

        # Deploy
        joblib.dump(pipe_c1, GUI_MODELS_DIR / "svm_ordinal_C1.joblib")
        joblib.dump(pipe_c2, GUI_MODELS_DIR / "svm_ordinal_C2.joblib")
        print(f"  Deployed to {GUI_MODELS_DIR}")
    else:
        print(f"  [WARN] GUI models dir not found: {GUI_MODELS_DIR}")

    # Also save to svm_ordinal_v2 (legacy location)
    legacy_dir = Path("D:/pipeline_SVM/results/svm_ordinal_v2")
    if legacy_dir.exists():
        joblib.dump(pipe_c1, legacy_dir / "svm_C1_any_wear.joblib")
        joblib.dump(pipe_c2, legacy_dir / "svm_C2_severe_wear.joblib")
        print(f"  Updated legacy {legacy_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Registry — track which tests have been processed
# ═══════════════════════════════════════════════════════════════════════════

def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {'tests': {}}


def save_registry(reg):
    ART2_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(reg, f, indent=2)


def register_test(test_num, drill_bit, total_holes, n_samples, drive_prefix):
    reg = load_registry()
    reg['tests'][str(test_num)] = {
        'drill_bit': drill_bit,
        'total_holes': total_holes,
        'n_samples': n_samples,
        'drive': drive_prefix,
        'processed_at': datetime.now().isoformat(),
    }
    save_registry(reg)


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_single_test(test_num, drill_bit, total_holes, hole_start=1,
                    skip_segment=False, skip_deploy=False):
    """Full pipeline for one test."""
    print("=" * 70)
    print(f"  RETRAIN PIPELINE — test{test_num} ({drill_bit}, {total_holes} holes)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    t_total = time.time()

    # Stage 0: Resolve paths
    test_dir, drive_prefix, folder_name = resolve_test_paths(test_num)

    # Stage 1: Segment
    manifest = None
    seg_dir = ART2_SEG_DIR / f"C_test{test_num}"
    if not seg_dir.exists():
        seg_dir = ART2_SEG_DIR / f"E_test{test_num}"

    if skip_segment and seg_dir.exists():
        manifest_path = seg_dir / "segments_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"\n  [Stage 1] SKIPPED — using existing manifest ({len(manifest['holes'])} holes)")
        else:
            print(f"  [Stage 1] No manifest found, re-segmenting...")
            manifest = segment_test(test_num, drill_bit, total_holes, hole_start,
                                    test_dir, drive_prefix)
    else:
        manifest = segment_test(test_num, drill_bit, total_holes, hole_start,
                                test_dir, drive_prefix)

    # Stage 2: Extract features
    csv_path = ART2_FEAT_DIR / f"test{test_num}_features.csv"
    if csv_path.exists() and skip_segment:
        print(f"\n  [Stage 2] Loading cached features from {csv_path.name}")
        df_test_feats = pd.read_csv(csv_path, low_memory=False)
    else:
        # Refresh drive prefix after potential rename
        dp = drive_prefix
        if (ART2_SEG_DIR / f"C_test{test_num}").exists():
            dp = 'C'
        elif (ART2_SEG_DIR / f"E_test{test_num}").exists():
            dp = 'E'
        df_test_feats = extract_features_for_test(test_num, manifest, drill_bit,
                                                   total_holes, dp)

    # Stage 3: Assign wear labels
    df_test_feats = assign_wear_labels(df_test_feats, total_holes)
    # Re-save with labels
    csv_path = ART2_FEAT_DIR / f"test{test_num}_features.csv"
    df_test_feats.to_csv(csv_path, index=False)

    # Register
    register_test(test_num, drill_bit, total_holes, len(df_test_feats), drive_prefix)

    # Stage 4: Merge
    df_merged = merge_all_features()

    # Stage 5: Retrain SVM
    pipe_c1, pipe_c2, metrics = retrain_svm(df_merged)

    # Stage 6: Evaluate
    iter_num = evaluate_and_compare(metrics, df_merged, test_num)

    # Stage 7: Deploy
    if not skip_deploy:
        deploy_models(pipe_c1, pipe_c2, iter_num)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — iteration {iter_num}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Dataset: {len(df_merged)} samples (Orejarena + Art.2)")
    if metrics:
        print(f"  E3 macro_F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"  E3 adj_acc:  {metrics.get('adjacent_accuracy', 0):.4f}")
    print(f"{'=' * 70}")

    return metrics


def run_reprocess_all(skip_deploy=False):
    """Re-run stages 3-7 for all registered tests without re-segmenting."""
    print("=" * 70)
    print("  REPROCESS ALL — rebuilding from cached features")
    print("=" * 70)

    reg = load_registry()
    if not reg['tests']:
        print("  No tests registered yet. Run with --test first.")
        return

    # Re-label all cached CSVs
    for test_num_str, info in reg['tests'].items():
        csv_path = ART2_FEAT_DIR / f"test{test_num_str}_features.csv"
        if not csv_path.exists():
            print(f"  [WARN] {csv_path.name} not found — skipping")
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        df = assign_wear_labels(df, info['total_holes'])
        df.to_csv(csv_path, index=False)

    # Merge + retrain
    df_merged = merge_all_features()
    pipe_c1, pipe_c2, metrics = retrain_svm(df_merged)
    iter_num = evaluate_and_compare(metrics, df_merged, None)

    if not skip_deploy:
        deploy_models(pipe_c1, pipe_c2, iter_num)

    print(f"\n  REPROCESS COMPLETE — iteration {iter_num}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Iterative retraining pipeline for Art.2 drilling tests"
    )
    parser.add_argument('--test', type=int, help='Test number (e.g. 39, 40)')
    parser.add_argument('--drill-bit', type=str, help='Drill bit ID (e.g. "6mm#4")')
    parser.add_argument('--total-holes', type=int, help='Total holes until fracture')
    parser.add_argument('--hole-start', type=int, default=1,
                        help='Starting hole number (default: 1 for new bit)')
    parser.add_argument('--skip-segment', action='store_true',
                        help='Skip segmentation (use cached)')
    parser.add_argument('--skip-deploy', action='store_true',
                        help='Do not deploy to GUI')
    parser.add_argument('--reprocess-all', action='store_true',
                        help='Re-merge and retrain from all cached features')

    args = parser.parse_args()

    if args.reprocess_all:
        run_reprocess_all(skip_deploy=args.skip_deploy)
    elif args.test:
        if not args.drill_bit:
            parser.error("--drill-bit required with --test")
        if not args.total_holes:
            parser.error("--total-holes required with --test")
        run_single_test(
            test_num=args.test,
            drill_bit=args.drill_bit,
            total_holes=args.total_holes,
            hole_start=args.hole_start,
            skip_segment=args.skip_segment,
            skip_deploy=args.skip_deploy,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python retrain_after_test.py --test 39 --drill-bit "6mm#4" --total-holes 110')
        print('  python retrain_after_test.py --test 40 --drill-bit "6mm#5" --total-holes 95')
        print('  python retrain_after_test.py --reprocess-all')


if __name__ == '__main__':
    main()
