#!/usr/bin/env python3
"""
compute_wear_from_labels.py

Consume labels.json del mirilla_labelizer y produce medidas de desgaste
en mm (reinterpretacion: chisel edge wear, segun Oscar 2026-04-19).

Estrategia:
  1. Por cada test, calcular px/mm mediana de TODAS las lineas de calibracion
     (Oscar dibuja una por captura que le ensene la mirilla; todas valen 1 mm).
  2. Para frames con linea 'vb' (relabel como 'chisel'), convertir a mm usando
     px/mm del test (no por frame).
  3. Para test sin ninguna calibracion (ej test50), warning + skip.

Output: D:/pipeline_SVM/article2/camera/wear_measurements.csv
Cols: test, frame, t_sec, wear_px, px_per_mm_test, wear_mm, n_cal, quality
"""
import json
import math
import csv
import statistics
from pathlib import Path

BASE = Path('D:/pipeline_SVM/article2/camera/clips')
LABEL_FILE = BASE / 'labels.json'
OUT_CSV = Path('D:/pipeline_SVM/article2/camera/wear_measurements.csv')


def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def extract_t(fname):
    # "NNN_tXXXs_shYY_reflZZ.png" -> XXX
    try:
        return int(fname.split('_t')[1].split('s_')[0])
    except Exception:
        return None


def main():
    labels = json.loads(LABEL_FILE.read_text(encoding='utf-8'))

    per_test_cal = {}
    per_test_frames = {}

    for key, entry in labels.items():
        test, frame = key.split('/', 1)
        per_test_frames.setdefault(test, []).append((frame, entry))
        cal = entry.get('calibration')
        if cal:
            per_test_cal.setdefault(test, []).append(dist(cal['p1'], cal['p2']))

    per_test_px_mm = {}
    print('=== Calibracion por test (px por mm, 1mm referencia) ===')
    for test in sorted(per_test_frames):
        cals = per_test_cal.get(test, [])
        if cals:
            med = statistics.median(cals)
            per_test_px_mm[test] = med
            spread = f'n={len(cals)} min={min(cals):.1f} max={max(cals):.1f}'
            print(f'  {test}: {med:.2f} px/mm  ({spread})')
        else:
            per_test_px_mm[test] = None
            print(f'  {test}: SIN calibracion -> no se podran convertir medidas')

    rows = []
    print('\n=== Medidas de desgaste (chisel edge) ===')
    for test in sorted(per_test_frames):
        px_mm = per_test_px_mm[test]
        for frame, entry in sorted(per_test_frames[test]):
            wear = entry.get('vb')  # labelizer usa 'vb' key, interpretado como chisel
            if not wear:
                continue
            px = dist(wear['p1'], wear['p2'])
            mm = px / px_mm if px_mm else None
            t_sec = extract_t(frame)
            qual = ','.join(entry.get('quality', []))
            rows.append({
                'test': test,
                'frame': frame,
                't_sec': t_sec,
                'wear_px': round(px, 2),
                'px_per_mm_test': round(px_mm, 2) if px_mm else None,
                'wear_mm': round(mm, 4) if mm is not None else None,
                'n_cal': len(per_test_cal.get(test, [])),
                'quality': qual,
            })
            mm_s = f'{mm:.3f} mm' if mm is not None else 'NO CAL'
            print(f'  {test}/{frame}  px={px:.1f}  -> {mm_s}')

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['test', 'frame', 't_sec', 'wear_px',
                            'px_per_mm_test', 'wear_mm', 'n_cal', 'quality'])
        w.writeheader()
        w.writerows(rows)

    print(f'\nWrote {len(rows)} rows -> {OUT_CSV}')

    # resumen por test
    print('\n=== Resumen por test ===')
    by_test = {}
    for r in rows:
        by_test.setdefault(r['test'], []).append(r['wear_mm'])
    for test, vals in sorted(by_test.items()):
        vs = [v for v in vals if v is not None]
        if vs:
            print(f'  {test}: n={len(vs)}  mean={sum(vs)/len(vs):.3f}mm  '
                  f'min={min(vs):.3f}  max={max(vs):.3f}')
        else:
            print(f'  {test}: {len(vals)} medidas sin calibracion')


if __name__ == '__main__':
    main()
