#!/usr/bin/env python3
"""
Segment test39 (broca 6mm#4, 110 holes) and merge into existing
pending_review.json for the labeling GUI.
"""
import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import json
import numpy as np
from pathlib import Path

# Monkey-patch base path before importing
import segment_holes_art2 as seg
seg.E_BASE = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados")

OUT_BASE = seg.OUT_BASE  # D:/pipeline_SVM/art2_segments

# ── test39 config ──
TEST_ID = 'test39'
FOLDER = '6mm_test39'
DRILL_BIT = '6mm#4'
HOLE_START = 1
EXPECTED = 110
NOTES = 'Broca#4 nueva, fractura h=110, caudal bajo ~2.9 L/min, tratamiento termico visible'

print("=" * 70)
print("  Segmenting test39 — Broca 6mm#4 (110 holes)")
print("=" * 70)

m = seg.process_test(TEST_ID, FOLDER, DRILL_BIT, HOLE_START, EXPECTED, NOTES)

if m is None:
    print("[ERROR] No manifest returned")
    sys.exit(1)

# Save per-test manifest
test_out = OUT_BASE / f"C_{TEST_ID}"
manifest_path = test_out / "segments_manifest.json"
# (already saved by process_test but with E_ prefix — move if needed)
actual_out = OUT_BASE / f"E_{TEST_ID}"
if actual_out.exists() and not test_out.exists():
    actual_out.rename(test_out)
    print(f"  Renamed E_{TEST_ID} -> C_{TEST_ID}")
    # Fix paths in manifest
    m['test_id'] = TEST_ID
    manifest_path = test_out / "segments_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(m, f, indent=2)

# ── Merge into pending_review.json ──
pending_path = OUT_BASE / "pending_review.json"
existing = []
if pending_path.exists():
    with open(pending_path) as f:
        existing = json.load(f)

# Remove any old test39 entries
existing = [e for e in existing if e.get('test_id') != TEST_ID]

# Build new entries
new_entries = []
for h in m['holes']:
    new_entries.append({
        'test_id': TEST_ID,
        'segment_id': h['hole_id'],
        'type': 'hole_candidate',
        'start_s': h['start_s'],
        'end_s': h['end_s'],
        'duration_s': h['duration_s'],
        'drill_bit': DRILL_BIT,
        'auto_label': h['label'],
        'confirmed_label': None,
        'method': h['method'],
        'files': h['files'],
        'notes': NOTES,
    })
for n in m['noise']:
    new_entries.append({
        'test_id': TEST_ID,
        'segment_id': n['noise_id'],
        'type': n['auto_type'],
        'start_s': n['start_s'],
        'end_s': n['end_s'],
        'duration_s': n['duration_s'],
        'drill_bit': DRILL_BIT,
        'auto_label': n['label'],
        'confirmed_label': None,
        'method': m.get('detection_method', ''),
        'files': n['files'],
        'notes': NOTES,
    })

merged = existing + new_entries
with open(pending_path, 'w') as f:
    json.dump(merged, f, indent=2)

n_holes = len(m['holes'])
n_noise = len(m['noise'])
avg_dur = np.mean([h['duration_s'] for h in m['holes']]) if m['holes'] else 0

print(f"\n{'=' * 70}")
print(f"  RESULT")
print(f"  Holes: {n_holes} (expected {EXPECTED}), avg {avg_dur:.1f}s/hole")
print(f"  Noise: {n_noise}")
print(f"  ESP32: {'YES' if m.get('has_esp32') else 'NO'}")
print(f"  Flow:  {'YES' if m.get('has_flow') else 'NO'}")
print(f"  Pending review: {len(existing)} old + {len(new_entries)} new = {len(merged)} total")
print(f"{'=' * 70}")
