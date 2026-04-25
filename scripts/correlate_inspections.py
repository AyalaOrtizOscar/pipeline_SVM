#!/usr/bin/env python3
"""
correlate_inspections.py
Correlates video, ESP32 flow, and wear predictions to identify drill inspection pauses.
Extracts video frames at each inspection timestamp and generates an HTML report.

Usage:
    python correlate_inspections.py --test 53
    python correlate_inspections.py --test 53 54 56
    python correlate_inspections.py --all
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
DATOS_DIR = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados")
OUTPUT_DIR = Path("D:/pipeline_SVM/results/inspection_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colombia is UTC-5
TZ_COL = timezone(timedelta(hours=-5))

# Feed rate for estimating hole count (from wizard: 86mm/min, depth 25mm → ~17.4s/hole)
DEFAULT_FEED_RATE_MM_MIN = 86.0
DEFAULT_DEPTH_MM = 25.0

LABEL_COLORS = {
    'sin_desgaste': '#27ae60',
    'medianamente_desgastado': '#f39c12',
    'desgastado': '#e74c3c',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_wizard(test_dir: Path) -> dict:
    p = test_dir / "wizard.json"
    return json.loads(p.read_text()) if p.exists() else {}

def load_events(test_dir: Path) -> list:
    p = test_dir / "events.log"
    if not p.exists():
        return []
    events = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events

def get_ni_start_utc(events: list) -> datetime | None:
    for e in events:
        if e.get('event') == 'ni_start_sent':
            return datetime.fromisoformat(e['ts_utc'].replace('Z', '+00:00'))
    return None

def get_cam_offset(events: list) -> float:
    """Seconds between NI start and camera start."""
    ni_start = get_ni_start_utc(events)
    if ni_start is None:
        return 0.0
    for e in events:
        if e.get('event') == 'camera_start_sent':
            cam_t = datetime.fromisoformat(e['ts_utc'].replace('Z', '+00:00'))
            return (cam_t - ni_start).total_seconds()
    return 0.0

def load_flow(test_dir: Path) -> pd.DataFrame | None:
    p = test_dir / "MCU" / "flow.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], utc=True)
    return df

def find_inspection_pauses(df_flow: pd.DataFrame, min_pause_s: float = 30.0,
                            flow_threshold: float = 0.2) -> list[dict]:
    """
    Returns list of dicts: {start_s, end_s, duration_s, preceding_drilling_s}
    representing significant coolant-off pauses (likely drill inspections).
    """
    df = df_flow.copy()
    df['active'] = df['flow_lmin'] > flow_threshold
    df['block'] = (df['active'] != df['active'].shift()).cumsum()

    blocks = df.groupby('block').agg(
        active=('active', 'first'),
        start_s=('elapsed_s', 'min'),
        end_s=('elapsed_s', 'max'),
        duration=('elapsed_s', lambda x: x.max() - x.min()),
        mean_flow=('flow_lmin', 'mean'),
    ).reset_index()

    pauses = []
    # skip first block (always a pause before drilling starts)
    first_drill_idx = blocks[blocks.active].index.min() if blocks.active.any() else None
    for i, row in blocks.iterrows():
        if not row['active'] and row['duration'] >= min_pause_s:
            if first_drill_idx is not None and i <= first_drill_idx:
                continue  # pre-drilling setup pause
            # find how much drilling happened before this pause
            drill_before = blocks[(blocks.index < i) & blocks['active']]['duration'].sum()
            pauses.append({
                'start_s': float(row['start_s']),
                'end_s': float(row['end_s']),
                'duration_s': float(row['duration']),
                'drill_before_s': float(drill_before),
            })
    return pauses

def estimate_holes(drill_before_s: float, feed_rate: float, depth_mm: float) -> int:
    """Estimate cumulative hole count from cumulative drilling seconds."""
    s_per_hole = (depth_mm / feed_rate) * 60  # seconds per hole (pure cutting time)
    # actual cycle is ~2x (includes retract, move, start): rough estimate
    return max(1, round(drill_before_s / (s_per_hole * 2.0)))

def load_predictions(test_dir: Path, ni_start_utc: datetime) -> pd.DataFrame | None:
    csvs = sorted(test_dir.glob("predictions_*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # predictions are in local time (UTC-5)
    df['timestamp_utc'] = df['timestamp'].apply(
        lambda x: x.replace(tzinfo=TZ_COL).astimezone(timezone.utc)
    )
    df['elapsed_s'] = (df['timestamp_utc'] - ni_start_utc).dt.total_seconds()
    return df

def get_wear_at(df_pred: pd.DataFrame, elapsed_s: float,
                mic: str = 'M3_MAXLIN', window_s: float = 60.0) -> dict:
    """Return majority vote wear state in window before elapsed_s."""
    if df_pred is None:
        return {'label': 'N/A', 'prob_desgastado': 0.0}
    window = df_pred[
        (df_pred['mic'] == mic) &
        (df_pred['elapsed_s'] >= elapsed_s - window_s) &
        (df_pred['elapsed_s'] < elapsed_s)
    ]
    if window.empty:
        # try all mics
        window = df_pred[
            (df_pred['elapsed_s'] >= elapsed_s - window_s) &
            (df_pred['elapsed_s'] < elapsed_s)
        ]
    if window.empty:
        return {'label': 'N/A', 'prob_desgastado': 0.0}
    majority = window['etiqueta'].mode().iloc[0]
    mean_deg = window['prob_desgastado'].mean()
    return {'label': majority, 'prob_desgastado': float(mean_deg)}

def extract_video_frame(video_path: Path, t_s: float, output_path: Path) -> bool:
    """Extract a single JPEG frame at t_s seconds."""
    cmd = [
        'ffmpeg', '-y', '-ss', str(t_s), '-i', str(video_path),
        '-frames:v', '1', '-q:v', '2', str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True)
    return output_path.exists()

def extract_video_clip(video_path: Path, start_s: float, end_s: float,
                       output_path: Path, max_dur: float = 20.0) -> bool:
    """Extract a short MP4 clip (capped at max_dur seconds, centered on start_s)."""
    dur = min(end_s - start_s, max_dur)
    # center clip in pause window if possible
    center = (start_s + end_s) / 2
    clip_start = max(0, center - dur / 2)
    cmd = [
        'ffmpeg', '-y', '-ss', str(clip_start), '-i', str(video_path),
        '-t', str(dur), '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
        '-an', str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True)
    return output_path.exists()

# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_flow_with_inspections(df_flow: pd.DataFrame, pauses: list,
                                predictions: pd.DataFrame | None,
                                test_id: str, output_path: Path,
                                total_holes: int | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax_flow, ax_pred = axes

    # Flow
    ax_flow.fill_between(df_flow['elapsed_s'], df_flow['flow_lmin'],
                         alpha=0.6, color='#3498db', label='flow (L/min)')
    ax_flow.set_ylabel('Caudal (L/min)', fontsize=10)
    ax_flow.set_title(f'test{test_id} — Caudal + Predicciones CNN', fontsize=12)

    # Predictions
    if predictions is not None:
        for mic_name, color in [('M3_MAXLIN', '#e74c3c'),
                                 ('M2_SL84C', '#9b59b6'),
                                 ('M1_C1', '#e67e22')]:
            sub = predictions[predictions['mic'] == mic_name].sort_values('elapsed_s')
            if not sub.empty:
                ax_pred.plot(sub['elapsed_s'], sub['prob_desgastado'],
                             label=mic_name, color=color, alpha=0.7, linewidth=1.5)
        ax_pred.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_pred.set_ylabel('P(desgastado)', fontsize=10)
        ax_pred.set_ylim(-0.05, 1.05)

    # Inspection pauses
    for i, p in enumerate(pauses):
        for ax in axes:
            ax.axvspan(p['start_s'], p['end_s'], alpha=0.15, color='#f39c12')
        # Label
        est_holes = estimate_holes(p['drill_before_s'],
                                   DEFAULT_FEED_RATE_MM_MIN, DEFAULT_DEPTH_MM)
        ax_flow.annotate(f'Insp.{i+1}\n~h{est_holes}',
                         xy=((p['start_s'] + p['end_s']) / 2, ax_flow.get_ylim()[1] * 0.8),
                         ha='center', fontsize=8, color='#d35400',
                         arrowprops=dict(arrowstyle='->', color='#d35400'),
                         xytext=((p['start_s'] + p['end_s']) / 2,
                                 ax_flow.get_ylim()[1] * 0.95))

    ax_pred.set_xlabel('Tiempo desde inicio (s)', fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")

# ── Main per-test analysis ────────────────────────────────────────────────────

def analyze_test(test_id: str) -> dict:
    test_dir = DATOS_DIR / f"6mm_test{test_id}"
    if not test_dir.exists():
        print(f"[ERROR] Not found: {test_dir}")
        return {}

    out_dir = OUTPUT_DIR / f"test{test_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS test{test_id}")
    print(f"{'='*60}")

    wizard = load_wizard(test_dir)
    events = load_events(test_dir)
    ni_start = get_ni_start_utc(events)
    cam_offset = get_cam_offset(events)

    print(f"  NI start:    {ni_start}")
    print(f"  Cam offset:  {cam_offset:.2f}s")
    print(f"  Drill bit:   {wizard.get('Referencia_broca', '?')}")
    print(f"  Material:    {wizard.get('material', '?')[:50]}")

    # Load notes
    notes_p = test_dir / "post_test_notes.json"
    notes = json.loads(notes_p.read_text()) if notes_p.exists() else {}
    obs = notes.get('observations', '')
    print(f"  Notas: {obs[:120]}")

    # Predict total holes from notes (e.g. "agujero 44" → 44)
    import re
    m = re.search(r'agujero\s+#?(\d+)', obs, re.IGNORECASE)
    total_holes = int(m.group(1)) if m else None
    if total_holes:
        print(f"  Total holes (from notes): {total_holes}")

    # Flow analysis
    df_flow = load_flow(test_dir)
    has_flow = df_flow is not None
    pauses = []
    if has_flow:
        pauses = find_inspection_pauses(df_flow)
        print(f"  Inspection pauses detected: {len(pauses)}")
        for i, p in enumerate(pauses):
            est = estimate_holes(p['drill_before_s'],
                                  wizard.get('feed_rate', DEFAULT_FEED_RATE_MM_MIN),
                                  float(wizard.get('drilling_depth', '25mm').replace('mm', '')))
            p['est_holes'] = est
            print(f"    Pause {i+1}: {p['start_s']:.0f}s-{p['end_s']:.0f}s "
                  f"({p['duration_s']:.0f}s), after ~{est} holes")
    else:
        print("  No flow.csv — using video duration segments")
        # Estimate pauses every ~15 holes using default timing
        video_p = test_dir / "video" / "acquisition.mp4"
        if video_p.exists():
            r = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', video_p],
                capture_output=True, text=True
            )
            vid_dur = float(json.loads(r.stdout)['format']['duration'])
            s_per_hole = (DEFAULT_DEPTH_MM / DEFAULT_FEED_RATE_MM_MIN) * 60 * 2.0
            n_est = total_holes or int(vid_dur / s_per_hole)
            every = 15
            # Generate synthetic pause timestamps at expected inspection intervals
            for h in range(every, n_est + 1, every):
                approx_s = h * s_per_hole
                pauses.append({
                    'start_s': approx_s,
                    'end_s': approx_s + 60,
                    'duration_s': 60,
                    'drill_before_s': approx_s,
                    'est_holes': h,
                })
            print(f"  Synthetic pauses every {every} holes: {len(pauses)} pauses")

    # Load predictions
    df_pred = None
    if ni_start:
        df_pred = load_predictions(test_dir, ni_start)
        if df_pred is not None:
            print(f"  Predictions loaded: {len(df_pred)} rows")

    # Video analysis
    video_path = test_dir / "video" / "acquisition.mp4"
    has_video = video_path.exists()
    print(f"  Video: {'YES' if has_video else 'NO'} ({video_path.name if has_video else ''})")

    # Generate plot
    if has_flow:
        plot_path = out_dir / "flow_predictions_timeline.png"
        plot_flow_with_inspections(df_flow, pauses, df_pred, test_id, plot_path, total_holes)

    # Extract frames and clips at each inspection pause
    inspection_data = []
    for i, p in enumerate(pauses):
        # Video timestamp = elapsed_s - cam_offset (cam starts cam_offset seconds after NI)
        # Video playback position = p['start_s'] - cam_offset
        vid_t = max(0, p['start_s'] - cam_offset)
        wear = get_wear_at(df_pred, p['start_s']) if df_pred is not None else {'label': 'N/A', 'prob_desgastado': 0.0}
        flow_before = None
        if has_flow:
            window = df_flow[(df_flow['elapsed_s'] >= p['start_s'] - 60) &
                             (df_flow['elapsed_s'] < p['start_s'])]
            flow_before = float(window['flow_lmin'].mean()) if not window.empty else None

        insp = {
            'pause_num': i + 1,
            'est_holes': p.get('est_holes', '?'),
            'pause_start_s': p['start_s'],
            'pause_end_s': p['end_s'],
            'pause_duration_s': p['duration_s'],
            'video_t_s': vid_t,
            'video_t_fmt': f"{int(vid_t//60):02d}:{int(vid_t%60):02d}",
            'wear_label': wear['label'],
            'wear_prob_deg': wear['prob_desgastado'],
            'mean_flow_before_lmin': flow_before,
            'frame_path': None,
            'clip_path': None,
        }

        if has_video:
            frame_path = frames_dir / f"inspection_{i+1:02d}_h{insp['est_holes']}.jpg"
            if extract_video_frame(video_path, vid_t, frame_path):
                insp['frame_path'] = str(frame_path.relative_to(out_dir))
                print(f"  Frame {i+1}: {frame_path.name} @ {insp['video_t_fmt']} "
                      f"-> wear={wear['label']} ({wear['prob_desgastado']:.2f})")
            else:
                print(f"  [WARN] Frame extraction failed for pause {i+1}")

            # Extract short clip (up to 15s inside the pause)
            clip_path = clips_dir / f"inspection_{i+1:02d}_h{insp['est_holes']}.mp4"
            clip_end_vid = max(0, p['end_s'] - cam_offset)
            if extract_video_clip(video_path, vid_t, clip_end_vid, clip_path, max_dur=15.0):
                insp['clip_path'] = str(clip_path.relative_to(out_dir))
                print(f"    Clip: {clip_path.name}")

        inspection_data.append(insp)

    # Save JSON summary
    summary = {
        'test_id': test_id,
        'ni_start_utc': ni_start.isoformat() if ni_start else None,
        'cam_offset_s': cam_offset,
        'drill_bit': wizard.get('Referencia_broca', '?'),
        'total_holes_from_notes': total_holes,
        'has_flow': has_flow,
        'has_video': has_video,
        'has_predictions': df_pred is not None,
        'observations': obs,
        'inspections': inspection_data,
    }
    json_path = out_dir / "inspection_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Summary saved: {json_path}")

    # Generate HTML report
    _write_html_report(summary, out_dir)

    return summary

# ── HTML report ───────────────────────────────────────────────────────────────

def _write_html_report(summary: dict, out_dir: Path):
    test_id = summary['test_id']
    has_plot = (out_dir / "flow_predictions_timeline.png").exists()

    rows = ""
    for insp in summary['inspections']:
        color = LABEL_COLORS.get(insp['wear_label'], '#95a5a6')
        frame_html = ''
        if insp.get('frame_path'):
            frame_html = (f'<img src="{insp["frame_path"]}" '
                          f'style="max-width:280px;max-height:180px;border-radius:4px">')
        clip_html = ''
        if insp.get('clip_path'):
            clip_html = (f'<video controls width="280" style="border-radius:4px">'
                         f'<source src="{insp["clip_path"]}" type="video/mp4"></video>')

        flow_txt = (f'{insp["mean_flow_before_lmin"]:.2f} L/min'
                    if insp.get('mean_flow_before_lmin') is not None else 'N/A')
        rows += f"""
        <tr>
          <td style="text-align:center;font-weight:bold">#{insp['pause_num']}</td>
          <td style="text-align:center">~{insp['est_holes']}</td>
          <td style="text-align:center">{insp['video_t_fmt']}<br>
              <small>({insp['pause_start_s']:.0f}s elap.)</small></td>
          <td style="text-align:center">{insp['pause_duration_s']:.0f}s</td>
          <td style="text-align:center">
              <span style="background:{color};color:white;padding:3px 8px;border-radius:4px;font-size:0.85em">
                {insp['wear_label']}
              </span><br>
              <small>P(deg)={insp['wear_prob_deg']:.2f}</small>
          </td>
          <td style="text-align:center">{flow_txt}</td>
          <td>{frame_html}</td>
          <td>{clip_html}</td>
        </tr>"""

    plot_section = ""
    if has_plot:
        plot_section = '<img src="flow_predictions_timeline.png" style="max-width:100%;margin:16px 0">'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Inspecciones — test{test_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #2c3e50; }}
  .meta {{ background: #fff; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 6px; overflow: hidden; }}
  th {{ background: #2c3e50; color: white; padding: 10px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #dee2e6; vertical-align: middle; }}
  tr:hover {{ background: #f1f3f5; }}
</style>
</head>
<body>
<h1>Reporte de Inspecciones — test{test_id}</h1>
<div class="meta">
  <strong>Broca:</strong> {summary['drill_bit']} &nbsp;|&nbsp;
  <strong>Agujeros totales (notas):</strong> {summary['total_holes_from_notes'] or 'N/A'} &nbsp;|&nbsp;
  <strong>Flow:</strong> {'SI' if summary['has_flow'] else 'NO'} &nbsp;|&nbsp;
  <strong>Predicciones:</strong> {'SI' if summary['has_predictions'] else 'NO'}<br>
  <strong>Observaciones:</strong> {summary['observations']}
</div>
{plot_section}
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Aguj. est.</th>
      <th>Video timestamp</th>
      <th>Durac. pausa</th>
      <th>Estado desgaste</th>
      <th>Caudal previo</th>
      <th>Frame</th>
      <th>Clip</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</body>
</html>"""

    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding='utf-8')
    print(f"  HTML report: {html_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Correlate drill inspections')
    parser.add_argument('--test', nargs='+', type=str,
                        help='Test IDs to analyze (e.g. --test 53 54 56)')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all tests with video')
    args = parser.parse_args()

    if args.all:
        tests = sorted([d.name.replace('6mm_test', '')
                        for d in DATOS_DIR.iterdir()
                        if (d / 'video' / 'acquisition.mp4').exists()])
    elif args.test:
        tests = args.test
    else:
        parser.print_help()
        return

    print(f"Analyzing tests: {tests}")
    summaries = []
    for t in tests:
        s = analyze_test(t)
        if s:
            summaries.append(s)

    # Cross-test summary
    if len(summaries) > 1:
        _write_cross_test_summary(summaries)

    print(f"\nDone. Reports in: {OUTPUT_DIR}")

def _write_cross_test_summary(summaries: list):
    out = OUTPUT_DIR / "cross_test_summary.html"
    rows = ""
    for s in summaries:
        for insp in s['inspections']:
            color = LABEL_COLORS.get(insp['wear_label'], '#95a5a6')
            flow_txt = (f'{insp["mean_flow_before_lmin"]:.2f}'
                        if insp.get('mean_flow_before_lmin') is not None else '—')
            rows += f"""
            <tr>
              <td>{s['test_id']}</td>
              <td>{s['drill_bit']}</td>
              <td>#{insp['pause_num']}</td>
              <td>~{insp['est_holes']}</td>
              <td><span style="background:{color};color:white;padding:2px 6px;border-radius:3px">
                  {insp['wear_label']}</span></td>
              <td>{flow_txt}</td>
              <td>{insp['wear_prob_deg']:.2f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Cross-Test Inspection Summary</title>
<style>
  body{{font-family:Arial,sans-serif;margin:24px;background:#f8f9fa}}
  h1{{color:#2c3e50}}
  table{{border-collapse:collapse;width:100%;background:#fff}}
  th{{background:#2c3e50;color:white;padding:8px}}
  td{{padding:7px 10px;border-bottom:1px solid #dee2e6}}
  tr:hover{{background:#f1f3f5}}
</style></head>
<body>
<h1>Resumen de Inspecciones — Multi-Test</h1>
<table>
  <thead><tr>
    <th>Test</th><th>Broca</th><th>Insp.</th><th>Aguj.est.</th>
    <th>Estado</th><th>Caudal (L/min)</th><th>P(deg)</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>"""
    out.write_text(html, encoding='utf-8')
    print(f"\nCross-test summary: {out}")

if __name__ == '__main__':
    main()
