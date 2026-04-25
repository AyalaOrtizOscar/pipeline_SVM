#!/usr/bin/env python3
"""
mirilla_events_labelizer.py

Labelizer para eventos de inspeccion detectados automaticamente.
Estructura: D:/pipeline_SVM/article2/camera/events/testNN/evt_XX_tYYYYYs/best.png

Workflow:
  1. Drag linea de calibracion (regla 1mm) -> C para confirmar
  2. Drag linea de chisel edge wear -> V para confirmar
  3. N / P = siguiente / previo evento
  4. T / B = siguiente / previo test
  5. S = guardar, Q = salir, R = limpiar frame actual

Nota: la calibracion se hereda automaticamente del evento previo del mismo test
si no se dibuja una nueva (ahorra tiempo).

Labels guardados en: D:/pipeline_SVM/article2/camera/events/labels.json
"""
import cv2
import json
import math
from pathlib import Path

EVENTS_ROOT = Path('D:/pipeline_SVM/article2/camera/events')
LABEL_FILE  = EVENTS_ROOT / 'labels.json'
DEFAULT_CAL_MM = 1.0


class EventsLabelizer:
    def __init__(self):
        # Recoger todos los eventos en orden: test -> evt
        self.entries = []   # list of (test, evt_dir, meta)
        for test_dir in sorted(EVENTS_ROOT.iterdir()):
            if not test_dir.is_dir():
                continue
            for evt_dir in sorted(test_dir.iterdir()):
                if not evt_dir.is_dir():
                    continue
                img = evt_dir / 'best.png'
                meta_f = evt_dir / 'meta.json'
                if not img.exists():
                    continue
                meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
                self.entries.append((test_dir.name, evt_dir.name, img, meta))

        if not self.entries:
            raise RuntimeError(f'No events found in {EVENTS_ROOT}')
        print(f'Loaded {len(self.entries)} events across '
              f'{len(set(e[0] for e in self.entries))} tests')

        self.labels = {}
        if LABEL_FILE.exists():
            self.labels = json.loads(LABEL_FILE.read_text(encoding='utf-8'))
            print(f'Loaded {len(self.labels)} existing labels')

        self.idx = 0
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        self.last_line = None

    def fkey(self):
        t, d, _, _ = self.entries[self.idx]
        return f'{t}/{d}'

    def cur_test(self):
        return self.entries[self.idx][0]

    def inherited_cal(self):
        """Busca calibracion del evento previo del mismo test."""
        test = self.cur_test()
        for i in range(self.idx - 1, -1, -1):
            if self.entries[i][0] != test:
                break
            k = f'{self.entries[i][0]}/{self.entries[i][1]}'
            cal = self.labels.get(k, {}).get('calibration')
            if cal:
                return cal
        return None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            self.is_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            self.drag_end = (x, y)
            self.is_dragging = False
            if self.drag_start != self.drag_end:
                self.last_line = (self.drag_start, self.drag_end)

    def draw(self, img):
        k = self.fkey()
        e = self.labels.get(k, {})

        # calibration (yellow) — propia o heredada
        cal = e.get('calibration') or self.inherited_cal()
        if cal:
            inherited = 'calibration' not in e
            color = (0, 200, 200) if inherited else (0, 255, 255)
            cv2.line(img, tuple(cal['p1']), tuple(cal['p2']), color, 2)
            mid = ((cal['p1'][0]+cal['p2'][0])//2, (cal['p1'][1]+cal['p2'][1])//2)
            label = f'cal {cal["mm"]}mm' + (' (heredada)' if inherited else '')
            cv2.putText(img, label, (mid[0]+5, mid[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # chisel wear (red)
        vb = e.get('vb')
        if vb:
            cv2.line(img, tuple(vb['p1']), tuple(vb['p2']), (0, 0, 255), 2)
            mm = vb.get('mm')
            lbl = f'wear {mm:.3f}mm' if mm else 'wear (no cal)'
            mid = ((vb['p1'][0]+vb['p2'][0])//2, (vb['p1'][1]+vb['p2'][1])//2)
            cv2.putText(img, lbl, (mid[0]+5, mid[1]+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # drag in-progress (green)
        if self.is_dragging and self.drag_start and self.drag_end:
            cv2.line(img, self.drag_start, self.drag_end, (0, 255, 0), 1)
        elif self.last_line:
            cv2.line(img, self.last_line[0], self.last_line[1], (0, 200, 100), 2)

        # HUD
        test, evt_dir, _, meta = self.entries[self.idx]
        hole_s = f'h≈{meta["hole_est"]}' if meta.get('hole_est') else 'h=?'
        qual = ','.join(e.get('quality', []) or [])
        total = meta.get('total_holes', '?')

        hud = [
            f'[{self.idx+1}/{len(self.entries)}] {test}  {evt_dir}',
            f'{meta.get("t_str","?")}  {hole_s}/{total}  Quality: {qual or "-"}',
            'DRAG -> C=cal  V=chisel_wear  | 1=refl 2=viruta 3=taladrina 4=humedad 5=OK',
            'N/P=evento  T/B=test  R=clear  S=save  Q=quit',
        ]
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 76), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for row, text in enumerate(hud):
            cv2.putText(img, text, (6, 16 + row*16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    def compute_mm(self, pts, cal):
        if not cal:
            return None
        return math.dist(pts[0], pts[1]) / math.dist(cal['p1'], cal['p2']) * cal['mm']

    def save(self):
        LABEL_FILE.write_text(
            json.dumps(self.labels, indent=2, default=list), encoding='utf-8')
        print(f'Saved {len(self.labels)} labels -> {LABEL_FILE}')

    def confirm_cal(self):
        if not self.last_line:
            print('  (no line drawn)')
            return
        k = self.fkey()
        self.labels.setdefault(k, {})['calibration'] = {
            'p1': list(self.last_line[0]),
            'p2': list(self.last_line[1]),
            'mm': DEFAULT_CAL_MM,
        }
        print(f'  CAL set: {self.last_line}')
        self.last_line = None
        vb = self.labels[k].get('vb')
        if vb:
            cal = self.labels[k]['calibration']
            vb['mm'] = self.compute_mm((vb['p1'], vb['p2']), cal)

    def confirm_wear(self):
        if not self.last_line:
            print('  (no line drawn)')
            return
        k = self.fkey()
        e = self.labels.setdefault(k, {})
        cal = e.get('calibration') or self.inherited_cal()
        mm = self.compute_mm(self.last_line, cal)
        e['vb'] = {'p1': list(self.last_line[0]),
                   'p2': list(self.last_line[1]), 'mm': mm}
        print(f'  chisel wear: {mm:.3f} mm' if mm else '  chisel wear (no cal yet)')
        self.last_line = None

    def toggle_quality(self, label):
        k = self.fkey()
        q = self.labels.setdefault(k, {}).setdefault('quality', [])
        if label in q:
            q.remove(label)
        else:
            q.append(label)

    def clear_current(self):
        k = self.fkey()
        if k in self.labels:
            del self.labels[k]
        self.last_line = None
        print(f'  cleared {k}')

    def jump_to_test(self, direction):
        cur = self.cur_test()
        if direction > 0:
            for i in range(self.idx + 1, len(self.entries)):
                if self.entries[i][0] != cur:
                    self.idx = i
                    self.last_line = None
                    print(f'  -> {self.entries[i][0]}')
                    return
        else:
            start = self.idx
            while start > 0 and self.entries[start-1][0] == cur:
                start -= 1
            if start == 0:
                return
            prev = self.entries[start-1][0]
            i = start - 1
            while i > 0 and self.entries[i-1][0] == prev:
                i -= 1
            self.idx = i
            self.last_line = None
            print(f'  -> {prev}')

    def run(self):
        QUALITY_KEYS = {
            ord('1'): 'reflejo', ord('2'): 'viruta_atrapada',
            ord('3'): 'taladrina_excesiva', ord('4'): 'humedad',
            ord('5'): 'clip_valido',
        }
        cv2.namedWindow('Mirilla events', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mirilla events', 960, 720)
        cv2.setMouseCallback('Mirilla events', self.on_mouse)

        while True:
            _, _, img_path, _ = self.entries[self.idx]
            img = cv2.imread(str(img_path))
            if img is None:
                self.idx = min(self.idx + 1, len(self.entries) - 1)
                continue
            h, w = img.shape[:2]
            if w < 800:
                scale = 800 / w
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            self.draw(img)
            cv2.imshow('Mirilla events', img)
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            if k == ord('q'):
                self.save()
                break
            elif k == ord('s'):
                self.save()
            elif k == ord('n'):
                self.last_line = None
                self.idx = min(self.idx + 1, len(self.entries) - 1)
            elif k == ord('p'):
                self.last_line = None
                self.idx = max(self.idx - 1, 0)
            elif k == ord('t'):
                self.jump_to_test(+1)
            elif k == ord('b'):
                self.jump_to_test(-1)
            elif k == ord('c'):
                self.confirm_cal()
            elif k == ord('v'):
                self.confirm_wear()
            elif k == ord('r'):
                self.clear_current()
            elif k in QUALITY_KEYS:
                self.toggle_quality(QUALITY_KEYS[k])

        cv2.destroyAllWindows()


if __name__ == '__main__':
    EventsLabelizer().run()
