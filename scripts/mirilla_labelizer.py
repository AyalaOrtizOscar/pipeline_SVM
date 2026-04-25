#!/usr/bin/env python3
"""
mirilla_labelizer.py

Herramienta interactiva para medir desgaste de flanco (VB) en los clips
de mirilla extraidos.

Workflow por cada imagen:
  1. Click-arrastra para dibujar la linea de calibracion (referencia 1 mm por defecto)
  2. Presiona C para confirmar calibracion
  3. Click-arrastra para dibujar la linea de desgaste (chisel edge wear)
  4. Presiona V para confirmar desgaste
  5. Presiona teclas de calidad: 1=reflejo 2=viruta 3=taladrina 4=humedad 5=valido
  6. Presiona N siguiente, P previo, T siguiente_test, B test_previo, S guardar, Q salir

Las medidas se guardan en D:/pipeline_SVM/article2/camera/clips/labels.json
"""
import cv2
import json
import math
from pathlib import Path

BASE = Path('D:/pipeline_SVM/article2/camera/clips')
LABEL_FILE = BASE / 'labels.json'
DEFAULT_CAL_MM = 1.0

QUALITY_KEYS = {
    ord('1'): 'reflejo',
    ord('2'): 'viruta_atrapada',
    ord('3'): 'taladrina_excesiva',
    ord('4'): 'humedad',
    ord('5'): 'clip_valido',
}


class Labelizer:
    def __init__(self):
        self.frames = []
        for td in sorted(BASE.iterdir()):
            if td.is_dir():
                for f in sorted(td.glob('*.png')):
                    self.frames.append((td.name, f))
        print(f'Loaded {len(self.frames)} frames across {len(set(f[0] for f in self.frames))} tests')

        self.labels = {}
        if LABEL_FILE.exists():
            self.labels = json.loads(LABEL_FILE.read_text(encoding='utf-8'))
            print(f'Loaded existing labels: {len(self.labels)}')

        self.idx = 0
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        self.last_line = None

    def fkey(self):
        t, f = self.frames[self.idx]
        return f'{t}/{f.name}'

    def jump_to_test(self, direction):
        """Avanza (+1) o retrocede (-1) al primer frame del siguiente/previo test."""
        cur_test = self.frames[self.idx][0]
        if direction > 0:
            for i in range(self.idx + 1, len(self.frames)):
                if self.frames[i][0] != cur_test:
                    self.idx = i
                    self.last_line = None
                    print(f'  -> test {self.frames[i][0]}')
                    return
            print('  (ya en ultimo test)')
        else:
            # encuentra el primer frame del test actual, retrocede uno mas para entrar al previo
            start_of_cur = self.idx
            while start_of_cur > 0 and self.frames[start_of_cur - 1][0] == cur_test:
                start_of_cur -= 1
            if start_of_cur == 0:
                print('  (ya en primer test)')
                return
            prev_test = self.frames[start_of_cur - 1][0]
            # ir al primer frame del prev_test
            i = start_of_cur - 1
            while i > 0 and self.frames[i - 1][0] == prev_test:
                i -= 1
            self.idx = i
            self.last_line = None
            print(f'  -> test {prev_test}')

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

        # existing calibration (yellow)
        cal = e.get('calibration')
        if cal:
            cv2.line(img, tuple(cal['p1']), tuple(cal['p2']), (0, 255, 255), 2)
            mid = ((cal['p1'][0]+cal['p2'][0])//2, (cal['p1'][1]+cal['p2'][1])//2)
            cv2.putText(img, f'cal {cal["mm"]}mm', (mid[0]+5, mid[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # existing wear line (red) — key 'vb' retained for backwards compat with labels.json
        vb = e.get('vb')
        if vb:
            cv2.line(img, tuple(vb['p1']), tuple(vb['p2']), (0, 0, 255), 2)
            vb_mm = vb.get('mm', None)
            label = f'wear {vb_mm:.3f}mm' if vb_mm else 'wear (no cal)'
            mid = ((vb['p1'][0]+vb['p2'][0])//2, (vb['p1'][1]+vb['p2'][1])//2)
            cv2.putText(img, label, (mid[0]+5, mid[1]+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # in-progress drag (green)
        if self.is_dragging and self.drag_start and self.drag_end:
            cv2.line(img, self.drag_start, self.drag_end, (0, 255, 0), 1)
        elif self.last_line:
            cv2.line(img, self.last_line[0], self.last_line[1], (0, 200, 100), 2)

        # HUD
        t, f = self.frames[self.idx]
        qual = e.get('quality', [])
        hud1 = f'[{self.idx+1}/{len(self.frames)}] {t}/{f.name}'
        hud2 = f'Quality: {",".join(qual) if qual else "-"}'
        hud3 = 'DRAG line, then C=cal V=chisel_wear  |  1=refl 2=viruta 3=taladrina 4=humedad 5=OK'
        hud4 = 'N/P=frame  T/B=test jump  R=clear S=save Q=quit'
        h, w = img.shape[:2]
        # Background strip
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 76), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        cv2.putText(img, hud1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(img, hud2, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,255,220), 1)
        cv2.putText(img, hud3, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
        cv2.putText(img, hud4, (6, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,255), 1)

    def compute_mm(self, pts, cal):
        if not cal:
            return None
        cal_len = math.dist(cal['p1'], cal['p2'])
        vb_len = math.dist(pts[0], pts[1])
        return vb_len / cal_len * cal['mm']

    def save(self):
        LABEL_FILE.write_text(json.dumps(self.labels, indent=2, default=list), encoding='utf-8')
        print(f'Saved {len(self.labels)} labels -> {LABEL_FILE}')

    def confirm_cal(self):
        if not self.last_line:
            print('  (no line drawn)')
            return
        k = self.fkey()
        self.labels.setdefault(k, {})['calibration'] = {
            'p1': list(self.last_line[0]),
            'p2': list(self.last_line[1]),
            'mm': DEFAULT_CAL_MM
        }
        print(f'  CAL set: {self.last_line} = {DEFAULT_CAL_MM}mm')
        self.last_line = None
        # recompute VB mm if present
        vb = self.labels[k].get('vb')
        if vb:
            mm = self.compute_mm((vb['p1'], vb['p2']), self.labels[k]['calibration'])
            vb['mm'] = mm

    def confirm_vb(self):
        if not self.last_line:
            print('  (no line drawn)')
            return
        k = self.fkey()
        e = self.labels.setdefault(k, {})
        cal = e.get('calibration')
        mm = self.compute_mm(self.last_line, cal)
        e['vb'] = {
            'p1': list(self.last_line[0]),
            'p2': list(self.last_line[1]),
            'mm': mm
        }
        print(f'  chisel wear set: {mm:.3f} mm' if mm else '  chisel wear set (no cal yet)')
        self.last_line = None

    def toggle_quality(self, label):
        k = self.fkey()
        e = self.labels.setdefault(k, {})
        q = e.setdefault('quality', [])
        if label in q:
            q.remove(label)
            print(f'  {label} OFF')
        else:
            q.append(label)
            print(f'  {label} ON')

    def clear_current(self):
        k = self.fkey()
        if k in self.labels:
            del self.labels[k]
            print(f'  cleared {k}')
        self.last_line = None

    def run(self):
        cv2.namedWindow('Mirilla chisel wear', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mirilla chisel wear', 960, 720)
        cv2.setMouseCallback('Mirilla chisel wear', self.on_mouse)

        while True:
            t, fpath = self.frames[self.idx]
            img = cv2.imread(str(fpath))
            if img is None:
                print(f'FAIL read {fpath}')
                self.idx = min(self.idx+1, len(self.frames)-1)
                continue
            # upscale small images for easier clicks
            h, w = img.shape[:2]
            if w < 800:
                scale = 800 / w
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            self.draw(img)
            cv2.imshow('Mirilla chisel wear', img)
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
                self.idx = min(self.idx+1, len(self.frames)-1)
            elif k == ord('p'):
                self.last_line = None
                self.idx = max(self.idx-1, 0)
            elif k == ord('t'):
                self.jump_to_test(+1)
            elif k == ord('b'):
                self.jump_to_test(-1)
            elif k == ord('c'):
                self.confirm_cal()
            elif k == ord('v'):
                self.confirm_vb()
            elif k == ord('r'):
                self.clear_current()
            elif k in QUALITY_KEYS:
                self.toggle_quality(QUALITY_KEYS[k])

        cv2.destroyAllWindows()


if __name__ == '__main__':
    Labelizer().run()
