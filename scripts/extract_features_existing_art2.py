#!/usr/bin/env python3
"""
Extract features for all already-segmented Art.2 tests (E: drive tests 15-21).
Populates D:/pipeline_SVM/features/art2/testNN_features.csv for each.
"""
import sys, os, json, time
sys.path.insert(0, "D:/pipeline_SVM/scripts")
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from pathlib import Path

from extract_features_from_manifests import extract_features
from retrain_after_test import (
    SVM_FEATURE_NAMES, CH_MIC_MAP, ART2_FEAT_DIR, ART2_SEG_DIR,
    assign_wear_labels, register_test
)

# Tests and their drill_bit info (from drill history memory)
E_TESTS = {
    15: {'drill_bit': '6mm#1', 'total_holes': 140, 'hole_start': 8},
    16: {'drill_bit': '6mm#1', 'total_holes': 140, 'hole_start': 50},
    17: {'drill_bit': '6mm#1', 'total_holes': 140, 'hole_start': 133},
    18: {'drill_bit': '6mm#2', 'total_holes': 37,  'hole_start': 1},
    19: {'drill_bit': '6mm#3', 'total_holes': 55,  'hole_start': 1},
    20: {'drill_bit': '6mm#3', 'total_holes': 55,  'hole_start': 31},
    21: {'drill_bit': '6mm#3', 'total_holes': 55,  'hole_start': 34},
}

ART2_FEAT_DIR.mkdir(parents=True, exist_ok=True)

for test_num, info in E_TESTS.items():
    csv_path = ART2_FEAT_DIR / f"test{test_num}_features.csv"
    if csv_path.exists() and csv_path.stat().st_size > 100:
        df_check = pd.read_csv(csv_path)
        if len(df_check) > 0 and not (df_check['label'] == '').all():
            print(f"test{test_num}: already cached ({len(df_check)} rows) — skip")
            continue

    test_id = f"test{test_num}"
    seg_dir = ART2_SEG_DIR / f"E_{test_id}"
    manifest_path = seg_dir / "segments_manifest.json"

    if not manifest_path.exists():
        print(f"test{test_num}: no manifest — skip")
        continue

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_holes = len(manifest['holes'])
    channels = manifest.get('channels', ['ch0', 'ch1', 'ch2'])
    drill_bit = info['drill_bit']
    total_holes = info['total_holes']

    print(f"\n{'='*60}")
    print(f"  test{test_num}: {n_holes} holes x {len(channels)} ch, {drill_bit} ({total_holes} total)")
    print(f"{'='*60}")

    rows = []
    t0 = time.time()

    for i, hole in enumerate(manifest['holes']):
        hole_id = hole['hole_id']
        hole_num = hole['hole_number']

        for ch in channels:
            if ch not in hole.get('files', {}):
                continue

            wav_rel = hole['files'][ch]
            wav_path = ART2_SEG_DIR / wav_rel

            if not wav_path.exists():
                continue

            dur_s = hole.get('duration_s', 0)
            if dur_s < 1.5:
                continue

            feats = extract_features(str(wav_path))
            if feats is None or '_error' in feats:
                continue

            mic_type = CH_MIC_MAP.get(ch, 'dinamico')
            row = {
                'filepath': str(wav_path),
                'label': '',
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
                'drive': 'E',
            }
            for fname in SVM_FEATURE_NAMES:
                row[fname] = feats.get(fname, np.nan)
            rows.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == n_holes:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_holes}] {elapsed:.0f}s")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = assign_wear_labels(df, total_holes)
        df.to_csv(csv_path, index=False)
        register_test(test_num, drill_bit, total_holes, len(df), 'E')
        print(f"  Done: {len(df)} samples -> {csv_path.name}")
    else:
        print(f"  [WARN] No samples extracted for test{test_num}")

print("\n=== ALL DONE ===")
