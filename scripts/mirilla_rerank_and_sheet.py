#!/usr/bin/env python3
"""
Re-rank los candidatos de mirilla ponderando baja reflexion.
Genera contact sheet HTML agrupado por test.
"""
import json
from pathlib import Path

BASE = Path('D:/pipeline_SVM/article2/camera/clips')


def composite(c, sh_max):
    """Score 0-100: nitidez normalizada + bonus por baja reflexion y brillo razonable."""
    sh = c['sharp'] / max(sh_max, 1) * 60        # hasta 60 pts
    refl_pen = max(0, 30 * (1 - c['refl'] / 0.10))  # 30 pts si refl<=0 a 0 si refl>=0.10
    brt = c['bright']
    bright_bonus = 10 if 40 <= brt <= 180 else max(0, 10 - abs((brt-110)/10))
    return sh + refl_pen + bright_bonus


def process_test(testdir):
    jp = testdir / 'candidates.json'
    if not jp.exists():
        return None
    data = json.loads(jp.read_text())
    cands = data.get('candidates', [])
    if not cands:
        return data
    sh_max = max(c['sharp'] for c in cands)
    for c in cands:
        c['score'] = composite(c, sh_max)
    cands.sort(key=lambda c: -c['score'])
    data['candidates'] = cands
    jp.write_text(json.dumps(data, indent=2))
    return data


def contact_sheet(all_data):
    html = ['<!doctype html><html><head><meta charset=utf-8>',
            '<title>Mirilla candidates (ranked)</title>',
            '<style>',
            'body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#ddd;margin:0;padding:12px}',
            'h1{margin:0 0 12px 0;font-size:18px}',
            '.test{margin-bottom:20px;border-top:1px solid #444;padding-top:12px}',
            '.test h2{margin:4px 0 8px 0;font-size:15px;color:#fc9}',
            '.meta{font-size:12px;color:#888;margin-bottom:8px}',
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}',
            '.cell{background:#2a2a2a;padding:4px;border-radius:3px;font-size:10px}',
            '.cell img{width:100%;display:block;border-radius:2px}',
            '.t{color:#aaa;margin-top:3px;line-height:1.3}',
            '.good{border:2px solid #5a5}',
            '.meh{border:1px solid #555}',
            '.bad{opacity:.6}',
            '</style></head><body>',
            '<h1>Mirilla candidates — ranked por score compuesto (nitidez + anti-reflejo + brillo)</h1>',
            '<div class="meta">Borde verde = score>70; borde gris = 50-70; atenuado = &lt;50.</div>']

    for test, data in sorted(all_data.items()):
        if not data:
            continue
        cands = data['candidates']
        ts = data.get('ts_str', '?')
        nc = len(cands)
        html.append(f'<div class="test"><h2>{test} — mirilla @ {ts}</h2>')
        html.append(f'<div class="meta">{nc} candidatos (ordenados por score compuesto)</div>')
        html.append('<div class="grid">')
        for c in cands[:30]:
            sc = c.get('score', 0)
            klass = 'good' if sc > 70 else ('meh' if sc > 50 else 'bad')
            rel = f'{test}/{c["file"]}'
            t = c['t']
            if t >= 3600:
                tt = f'{int(t//3600)}:{int((t%3600)//60):02d}:{int(t%60):02d}'
            else:
                tt = f'{int(t//60)}:{int(t%60):02d}'
            html.append(
                f'<div class="cell {klass}"><img src="{rel}" loading=lazy>'
                f'<div class=t>#{c["rank"]:02d} t={tt}<br>'
                f'sc={sc:.0f} sh={int(c["sharp"])} refl={c["refl"]:.2f}</div></div>')
        html.append('</div></div>')
    html.append('</body></html>')
    (BASE / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'Contact sheet written: {BASE / "index.html"}')


if __name__ == '__main__':
    BASE.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for td in sorted(BASE.iterdir()):
        if td.is_dir():
            all_data[td.name] = process_test(td)
    contact_sheet(all_data)
