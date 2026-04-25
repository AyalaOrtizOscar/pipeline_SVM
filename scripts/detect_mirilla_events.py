#!/usr/bin/env python3
"""
detect_mirilla_events.py

Escanea TODOS los videos de Art.2 buscando todos los eventos de inspección
de mirilla (paradas del husillo cada ~15 agujeros).

Estrategia:
  - Durante taladrado: frames borrosos (vibración), alta variación entre frames
  - Durante inspección de mirilla: drill estático, alta nitidez, baja variación
  → detectar períodos de "alta nitidez sostenida" = eventos de parada

Salida:
  D:/pipeline_SVM/article2/camera/events/
    testNN/
      evt_00_t0000s/  ← primer evento (t=timestamp del mejor frame)
        best.png      ← el mejor frame del evento
        meta.json     ← {t_sec, t_str, sharpness, refl, bright, hole_est}
      evt_01_t0212s/
        ...
    index.html        ← contact sheet con todos los eventos ordenados por test+evento
"""
import cv2
import numpy as np
import json
from pathlib import Path

VIDEOS = {
    'test30': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test30/video/acquisition.mp4',
        'total_holes': None,   # desconocido
    },
    'test39': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test39/video/acquisition.mp4',
        'total_holes': 110,
    },
    'test50': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test50/video/acquisition.mp4',
        'total_holes': None,
    },
    'test53': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test53/video/acquisition.mp4',
        'total_holes': None,
    },
    'test54': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test54/video/acquisition.mp4',
        'total_holes': None,
    },
    'test56': {
        'path': 'C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados/6mm_test56/video/acquisition.mp4',
        'total_holes': None,
    },
}

OUT_ROOT = Path('D:/pipeline_SVM/article2/camera/events')

# --- Parámetros de detección ---
SEEK_INTERVAL  = 5.0     # segundos entre seeks (scan rápido)
SHARP_THRESH   = 80      # nitidez mínima para considerar "candidato a inspección"
REFL_MAX       = 0.15    # reflexión máxima permitida
BRIGHT_MIN     = 30      # brillo mínimo (evita frames en negro)
MIN_EVENT_GAP  = 90      # segundos mínimos entre eventos distintos
MIN_EVENT_LEN  = 5       # segundos mínimos de duración de un evento válido


def fmt_t(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def score_frame(gray):
    sh = cv2.Laplacian(gray, cv2.CV_64F).var()
    refl = float((gray > 240).mean())
    bright = float(gray.mean())
    return sh, refl, bright


def scan_video(name, cfg):
    path = cfg['path']
    total_holes = cfg['total_holes']
    out_test = OUT_ROOT / name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'[{name}] FAIL opening {path}')
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    print(f'[{name}] fps={fps:.2f} duration={fmt_t(duration_s)} ({total_frames} frames)')

    # Escaneo rápido por seek temporal (evita decodificar todo el video)
    candidates = []
    t = 0.0
    n_sampled = 0
    while t < duration_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            t += SEEK_INTERVAL
            continue
        actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh, refl, bright = score_frame(gray)
        n_sampled += 1
        if sh >= SHARP_THRESH and refl <= REFL_MAX and bright >= BRIGHT_MIN:
            candidates.append({'t': actual_t, 'sharp': sh,
                                'refl': refl, 'bright': bright})
        t += SEEK_INTERVAL
    cap.release()
    print(f'[{name}] {n_sampled} seeks, {len(candidates)} candidatos de alta nitidez')

    if not candidates:
        return []

    # Agrupar candidatos en eventos (gap mínimo entre eventos = MIN_EVENT_GAP s)
    events = []
    current_event = [candidates[0]]
    for c in candidates[1:]:
        if c['t'] - current_event[-1]['t'] > MIN_EVENT_GAP:
            events.append(current_event)
            current_event = [c]
        else:
            current_event.append(c)
    events.append(current_event)

    # Filtrar eventos muy cortos (ruido)
    events = [e for e in events if e[-1]['t'] - e[0]['t'] >= MIN_EVENT_LEN or len(e) >= 2]

    print(f'[{name}] {len(events)} eventos de inspección detectados')
    for i, ev in enumerate(events):
        t0, t1 = ev[0]['t'], ev[-1]['t']
        print(f'  evt_{i:02d}: {fmt_t(t0)} – {fmt_t(t1)} ({len(ev)} frames nítidos)')

    # Por cada evento: releer el mejor frame (más nítido) con seek
    evt_metas = []
    cap2 = cv2.VideoCapture(path)
    for i, ev in enumerate(events):
        best = max(ev, key=lambda c: c['sharp'])
        cap2.set(cv2.CAP_PROP_POS_MSEC, best['t'] * 1000)
        ret, frame = cap2.read()
        if not ret:
            continue

        # Estimar hole_number si conocemos total_holes y duración
        hole_est = None
        if total_holes:
            hole_est = round(best['t'] / duration_s * total_holes)

        t_str = fmt_t(best['t'])
        evt_dir = out_test / f'evt_{i:02d}_t{int(best["t"]):05d}s'
        evt_dir.mkdir(parents=True, exist_ok=True)

        img_path = evt_dir / 'best.png'
        cv2.imwrite(str(img_path), frame)

        meta = {
            'event': i,
            't_sec': best['t'],
            't_str': t_str,
            'sharp': best['sharp'],
            'refl': best['refl'],
            'bright': best['bright'],
            'hole_est': hole_est,
            'total_holes': total_holes,
            'n_sharp_frames': len(ev),
            'event_span_s': ev[-1]['t'] - ev[0]['t'],
        }
        (evt_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        evt_metas.append({'test': name, 'dir': evt_dir.name, 'meta': meta})
    cap2.release()

    return evt_metas


def contact_sheet(all_events):
    html = ['<!doctype html><html><head><meta charset=utf-8>',
            '<title>Mirilla events — detección automática</title>',
            '<style>',
            'body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#ddd;margin:0;padding:12px}',
            'h1{margin:0 0 12px 0;font-size:18px}',
            '.test{margin-bottom:24px;border-top:2px solid #555;padding-top:12px}',
            '.test h2{margin:4px 0 8px;font-size:15px;color:#fc9}',
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}',
            '.cell{background:#2a2a2a;padding:6px;border-radius:4px;font-size:11px}',
            '.cell img{width:100%;display:block;border-radius:3px}',
            '.t{color:#aaa;margin-top:4px;line-height:1.5}',
            '.hole{color:#6cf;font-weight:bold}',
            '</style></head><body>',
            '<h1>Eventos de inspección de mirilla — detectados automáticamente</h1>',
            '<p style="font-size:12px;color:#888">Cada tarjeta = 1 parada del husillo. '
            'Los agujeros estimados asumen velocidad uniforme. Usar como guía.</p>']

    for test, evts in sorted(all_events.items()):
        if not evts:
            continue
        html.append(f'<div class="test"><h2>{test} — {len(evts)} eventos</h2><div class="grid">')
        for e in evts:
            m = e['meta']
            rel = f'{test}/{e["dir"]}/best.png'
            hole_s = f'h≈{m["hole_est"]}' if m['hole_est'] else 'h=?'
            html.append(
                f'<div class="cell">'
                f'<img src="{rel}" loading=lazy>'
                f'<div class="t"><b>evt_{m["event"]:02d}</b> {m["t_str"]}<br>'
                f'<span class="hole">{hole_s}</span> &nbsp; '
                f'sh={int(m["sharp"])} refl={m["refl"]:.2f}<br>'
                f'{m["n_sharp_frames"]} frames nítidos en {m["event_span_s"]:.0f}s</div>'
                f'</div>')
        html.append('</div></div>')

    html.append('</body></html>')
    (OUT_ROOT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'\nContact sheet: {OUT_ROOT / "index.html"}')


if __name__ == '__main__':
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_events = {}
    for name, cfg in VIDEOS.items():
        evts = scan_video(name, cfg)
        all_events[name] = evts
    contact_sheet(all_events)
    print('Done.')
