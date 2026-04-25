#!/usr/bin/env python3
"""
extract_mirilla_clips.py

Extrae fotogramas de mirilla de los videos de ensayos Art.2.

Flujo tras parada del husillo (confirmado por Oscar):
  1) limpieza con trapo lente
  2) sopladora sobre broca (elimina taladrina)
  3) rotación manual del husillo para captar ambos flancos

El clip de mirilla dura ~1-3 min. Seleccionamos por nitidez
(Laplacian var), ausencia de reflejos (frac. saturados < umbral)
y brillo medio (evita frames en negro).
"""
import cv2
import numpy as np
import json
from pathlib import Path

VIDEOS = {
    'test30': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test30/video/acquisition.mp4', '1:22:05'),
    'test39': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test39/video/acquisition.mp4', '14:03'),
    'test50': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test50/video/acquisition.mp4', '39:41'),
    'test53': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test53/video/acquisition.mp4', '9:09'),
    'test54': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test54/video/acquisition.mp4', '5:03'),
    'test56': ('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test56/video/acquisition.mp4', '4:41'),
}

OUT_ROOT = Path('D:/pipeline_SVM/article2/camera/clips')
WINDOW_BEFORE_S = 30
WINDOW_AFTER_S  = 180
SAMPLE_EVERY_N  = 8
TOP_N           = 40
REFL_MAX        = 0.12
BRIGHT_MIN      = 25


def parse_ts(s):
    p = [int(x) for x in s.split(':')]
    if len(p) == 2: return p[0]*60 + p[1]
    if len(p) == 3: return p[0]*3600 + p[1]*60 + p[2]
    raise ValueError(s)


def score_frame(gray):
    sh = cv2.Laplacian(gray, cv2.CV_64F).var()
    refl = float((gray > 240).mean())
    bright = float(gray.mean())
    return sh, refl, bright


def process(name, path, ts_str):
    ts = parse_ts(ts_str)
    outdir = OUT_ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'[{name}] FAIL opening {path}')
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    start_f = max(0, int((ts - WINDOW_BEFORE_S) * fps))
    end_f   = min(total, int((ts + WINDOW_AFTER_S) * fps))

    print(f'[{name}] ts={ts}s ({ts_str}) fps={fps:.2f} total={duration:.0f}s  window frames {start_f}..{end_f}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    cands = []
    idx = start_f
    n_read = 0
    while idx < end_f:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_f) % SAMPLE_EVERY_N == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sh, refl, bright = score_frame(gray)
            cands.append({'idx': idx, 't': idx / fps, 'sharp': sh,
                          'refl': refl, 'bright': bright, 'frame': frame})
            n_read += 1
        idx += 1
    cap.release()

    if not cands:
        print(f'[{name}] no candidates read')
        return None

    # Filter by reflection + brightness
    passing = [c for c in cands if c['refl'] < REFL_MAX and c['bright'] > BRIGHT_MIN]
    if len(passing) < 10:
        passing = sorted(cands, key=lambda c: c['refl'])[:50]

    # Rank by sharpness
    passing.sort(key=lambda c: -c['sharp'])
    top = passing[:TOP_N]

    # Save
    meta = []
    for rank, c in enumerate(top):
        fn = outdir / f'{rank:03d}_t{int(c["t"])}s_sh{int(c["sharp"])}_refl{c["refl"]:.3f}.png'
        cv2.imwrite(str(fn), c['frame'])
        meta.append({'rank': rank, 'file': fn.name, 't': c['t'],
                     'idx': c['idx'], 'sharp': c['sharp'],
                     'refl': c['refl'], 'bright': c['bright']})

    (outdir / 'candidates.json').write_text(json.dumps({
        'video': path, 'ts_str': ts_str, 'ts_sec': ts, 'fps': fps,
        'sampled': n_read, 'passing': len(passing),
        'kept_top': len(top), 'candidates': meta
    }, indent=2))

    print(f'[{name}] sampled={n_read} passing={len(passing)} saved_top={len(top)}')
    return meta


def contact_sheet(results):
    """Generate HTML contact sheet for visual review."""
    html = ['<!doctype html><html><head><meta charset=utf-8>',
            '<title>Mirilla candidates</title>',
            '<style>body{font-family:sans-serif;background:#222;color:#ddd}',
            '.test{margin-bottom:2em;border-bottom:1px solid #555;padding-bottom:1em}',
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}',
            '.cell{background:#333;padding:6px;border-radius:4px}',
            '.cell img{width:100%;display:block}',
            '.cell .t{font-size:11px;color:#aaa;margin-top:4px}',
            '</style></head><body>',
            '<h1>Mirilla candidates (ordenados por nitidez descendente)</h1>']

    for name, meta in results.items():
        if meta is None:
            continue
        html.append(f'<div class=test><h2>{name} ({len(meta)} candidatos)</h2><div class=grid>')
        for c in meta[:30]:
            rel = f'{name}/{c["file"]}'
            t = c['t']
            mm = int(t // 60); ss = int(t % 60)
            html.append(
                f'<div class=cell><img src="{rel}" loading=lazy>'
                f'<div class=t>#{c["rank"]:02d} t={mm}:{ss:02d} '
                f'sharp={int(c["sharp"])} refl={c["refl"]:.2f}</div></div>')
        html.append('</div></div>')
    html.append('</body></html>')
    (OUT_ROOT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'Contact sheet: {OUT_ROOT / "index.html"}')


if __name__ == '__main__':
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (path, ts) in VIDEOS.items():
        results[name] = process(name, path, ts)
    contact_sheet(results)
    print('\nDone.')
