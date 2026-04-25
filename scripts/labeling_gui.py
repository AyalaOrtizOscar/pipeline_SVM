#!/usr/bin/env python3
"""
Art.2 Labeling GUI — v4: Auto-labeling + Custom Labels + Validation.

Workflow:
  1. Arrastra en el waveform para seleccionar una franja de tiempo.
  2. Presiona una tecla (1-9) para asignar un label a esa franja.
  3. Repite para cada fenomeno acustico en el segmento.
  4. Enter = confirmar todas las regiones y avanzar al siguiente pendiente.

  AUTO-ETIQUETADO:
  - Boton "AUTO-ETIQUETAR": etiqueta automaticamente todos los segmentos pendientes
    usando KNN sobre los ya etiquetados + reglas por tipo de segmento.
  - Filtro "auto_pending": muestra solo los auto-etiquetados pendientes de validar.
  - Tecla [A] o boton: acepta el auto-label actual y avanza.

  LABEL PERSONALIZADO:
  - Boton "+ NUEVO LABEL": agrega un tipo de audio nuevo en tiempo de ejecucion.
  - Los labels custom se guardan en custom_labels.json y se recargan al inicio.

Cada region guardada: {start_s, end_s, label, duration_s, source}
"""
import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import soundfile as sf
import json, os, time, threading
from pathlib import Path
from datetime import datetime
from scipy.signal import resample as scipy_resample

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.widgets import SpanSelector
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

# ── Config ──────────────────────────────────────────────────────────────
SEGMENTS_DIR      = Path("D:/pipeline_SVM/art2_segments")
PENDING_FILE      = SEGMENTS_DIR / "pending_review.json"
LABELS_FILE       = SEGMENTS_DIR / "labels.json"
AUTOLABELS_FILE   = SEGMENTS_DIR / "autolabels.json"
CUSTOM_LABELS_FILE= SEGMENTS_DIR / "custom_labels.json"

# Palette for new custom labels (cycled through when user doesn't pick one)
_CUSTOM_PALETTE = ['#e84393','#00b894','#fd79a8','#6c5ce7','#fdcb6e',
                   '#e17055','#74b9ff','#a29bfe','#55efc4','#ffeaa7']

LABEL_COLORS = {
    'drilling':       '#27ae60',
    'post_fracture':  '#e67e22',
    'taladrina':      '#16a085',
    'machine_noise':  '#7f8c8d',
    'sopladora':      '#f39c12',
    'door_ambient':   '#c0392b',
    'repositioning':  '#8e44ad',
    'unknown':        '#2980b9',
    'skip':           '#636e72',
}
ALL_LABELS = list(LABEL_COLORS.keys())

LABEL_NOTES = {
    'drilling':      'Maquina ON + corte activo (husillo avanzando en material)',
    'post_fracture': 'Ciclo G81 ejecutado con broca rota (sin corte real)',
    'taladrina':     'Maquina ON + flujo refrigerante activo (sin cortar)',
    'machine_noise': 'Maquina ON + mov. no-mecanizado (home, MPG, cambio herramienta)',
    'sopladora':     'Limpieza con aire comprimido sobre flanco',
    'door_ambient':  'Apertura de puerta CNC / ruido ambiental',
    'repositioning': 'Reposicionamiento rapido XY entre agujeros (sin corte)',
    'unknown':       'No identificado — revisar despues',
    'skip':          'Excluir del dataset',
}

CONTEXT_NOTES = {
    'test15': 'Broca#1 (h.8-50). Mirilla+endoscopio entre agujeros. 872s.',
    'test16': 'Broca#1 (h.50-132). 604s. "100% real".',
    'test17': 'Broca#1 FIN (~133-140). Solo 30s. FRACTURA al final.',
    'test18': 'Broca#2 (h.1-37). 54min. FRACTURA en h=37. Segmentos 38+ = post_fracture.',
    'test19': 'Broca#3 (h.1-30). 16min. Pinza ER32. "FINALIZA cara 1 disco 2".',
    'test20': 'Broca#3 cont. (h.31-53). Solo 46s. Sujecion copa.',
    'test21': 'Broca#3 FIN (h.54-55). 24min. FRACTURA broca#3 en h=55.',
}

# Alpha values for region overlays on waveform
REGION_ALPHA = 0.35


def _load_custom_labels():
    """Load custom labels from disk and merge into globals."""
    if not CUSTOM_LABELS_FILE.exists():
        return
    try:
        with open(CUSTOM_LABELS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            name  = entry['name']
            color = entry.get('color', '#95a5a6')
            note  = entry.get('note', '')
            if name not in LABEL_COLORS:
                LABEL_COLORS[name] = color
                ALL_LABELS.append(name)
                LABEL_NOTES[name] = note
    except Exception as e:
        print(f"[custom_labels] Error cargando: {e}")

_load_custom_labels()


class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Art.2 — Etiquetador Temporal v3")
        self.root.geometry("1500x920")
        self.root.configure(bg='#2c3e50')

        self.segments = []
        self.labels = {}           # key -> {regions: [...], timestamp, ...}
        self.autolabels = {}       # key -> {label, confidence, source:'auto'}
        self.current_idx = 0
        self.playing = False
        self.paused  = False
        self._label_frame = None   # ref to label panel frame for rebuild

        # Playback position tracking for playhead cursor
        self.play_channel       = 'ch2'   # channel currently loaded for playback
        self.play_data          = None    # numpy array of currently playing audio
        self.play_sr            = None
        self.play_offset_s      = 0.0     # seconds already played before last play()
        self._play_start_wall   = None    # wall-clock when sd.play() was last called (set in thread)
        self._play_frame_count  = 0       # unused tracker (kept for future use)
        self._play_total_frames = 0

        # Playhead line objects (set after first draw)
        self.playhead_wave = None
        self.playhead_rms  = None

        # Temporal annotation state
        self.current_regions = []     # [{start_s, end_s, label}, ...]
        self.pending_selection = None  # (start_s, end_s) from SpanSelector
        self.audio_data = None
        self.audio_sr   = None

        self.label_buttons = {}       # label -> Button widget
        self.span_selector = None

        self.load_data()
        self.build_ui()
        self.show_segment(0)
        self._tick_playhead()   # start playhead update loop

    # ── Data I/O ─────────────────────────────────────────────────────────

    def load_data(self):
        if PENDING_FILE.exists():
            with open(PENDING_FILE) as f:
                self.segments = json.load(f)
        else:
            messagebox.showerror("Error", f"No se encontro pending_review.json en:\n{PENDING_FILE}")
            sys.exit(1)

        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                self.labels = json.load(f)

        if AUTOLABELS_FILE.exists():
            try:
                with open(AUTOLABELS_FILE, encoding='utf-8') as f:
                    self.autolabels = json.load(f)
            except Exception:
                self.autolabels = {}

        self.n_total = len(self.segments)
        self.n_holes = sum(1 for s in self.segments if 'hole' in s.get('segment_id', ''))
        self.n_noise = self.n_total - self.n_holes

    def save_labels(self):
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

    def save_autolabels(self):
        with open(AUTOLABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.autolabels, f, indent=2, ensure_ascii=False)

    # ── UI Construction ───────────────────────────────────────────────────

    def build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self.root, bg='#1a252f', height=36)
        top.pack(fill='x', padx=4, pady=2)

        self.progress_label = tk.Label(top, text="", font=('Consolas', 10, 'bold'),
                                        bg='#1a252f', fg='#f1c40f')
        self.progress_label.pack(side='left', padx=10)

        self.filter_var = tk.StringVar(value='all')
        for filt in ['all', 'pending', 'labeled', 'holes_only', 'noise_only', 'auto_pending']:
            tk.Radiobutton(top, text=filt, variable=self.filter_var, value=filt,
                           command=self.apply_filter, bg='#1a252f', fg='#ecf0f1',
                           selectcolor='#2c3e50', font=('Consolas', 8)).pack(side='left', padx=2)

        self._auto_btn_var = tk.StringVar(value="AUTO-ETIQUETAR")
        tk.Button(top, textvariable=self._auto_btn_var,
                  font=('Consolas', 9, 'bold'),
                  bg='#d35400', fg='white', command=self.run_autolabeler
                  ).pack(side='right', padx=3)
        tk.Button(top, text="STATS", font=('Consolas', 9, 'bold'),
                  bg='#8e44ad', fg='white', command=self.show_stats
                  ).pack(side='right', padx=6)

        # ── Main PanedWindow ──
        main = tk.PanedWindow(self.root, orient='horizontal', bg='#2c3e50', sashwidth=5)
        main.pack(fill='both', expand=True, padx=4, pady=2)

        # Left: segment list
        left = tk.Frame(main, bg='#1a252f', width=260)
        main.add(left, width=260)
        tk.Label(left, text="Segmentos", font=('Consolas', 10, 'bold'),
                 bg='#1a252f', fg='white').pack(pady=2)
        lf = tk.Frame(left, bg='#1a252f')
        lf.pack(fill='both', expand=True)
        self.seg_listbox = tk.Listbox(lf, font=('Consolas', 8), bg='#0d1b2a', fg='#ecf0f1',
                                       selectbackground='#2980b9', width=33, height=50)
        sb = tk.Scrollbar(lf, command=self.seg_listbox.yview)
        self.seg_listbox.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.seg_listbox.pack(side='left', fill='both', expand=True)
        self.seg_listbox.bind('<<ListboxSelect>>', self.on_list_select)
        self.populate_list()

        # Center: waveform + controls
        center = tk.Frame(main, bg='#2c3e50')
        main.add(center, width=780)
        self._build_waveform_panel(center)
        self._build_label_panel(center)
        self._build_nav_panel(center)

        # Right: context + regions
        right = tk.Frame(main, bg='#1a252f', width=340)
        main.add(right, width=340)
        tk.Label(right, text="Contexto & Regiones", font=('Consolas', 10, 'bold'),
                 bg='#1a252f', fg='white').pack(pady=2)
        self.info_text = tk.Text(right, font=('Consolas', 8), bg='#0d1b2a',
                                  fg='#ecf0f1', wrap='word', height=55, width=42)
        self.info_text.pack(fill='both', expand=True, padx=3, pady=3)

        # ── Keyboard shortcuts ──
        self.root.bind('<space>',     lambda e: self.play_audio('ch2'))
        self.root.bind('<Right>',     lambda e: self.next_segment())
        self.root.bind('<Left>',      lambda e: self.prev_segment())
        self.root.bind('<Return>',    lambda e: self.confirm_and_advance())
        self.root.bind('<BackSpace>', lambda e: self.undo_last_region())
        self.root.bind('<Delete>',    lambda e: self.clear_regions())
        self.root.bind('s', lambda e: self.stop_audio())
        self.root.bind('p', lambda e: self.toggle_pause())
        self.root.bind('a', lambda e: self.accept_autolabel())
        for i, label in enumerate(ALL_LABELS[:9]):
            key = str(i + 1)
            self.root.bind(key, lambda e, l=label: self.assign_label_to_selection(l))

    def _build_waveform_panel(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="matplotlib no disponible",
                     bg='#2c3e50', fg='#e74c3c').pack(pady=20)
            return

        self.fig = Figure(figsize=(8.5, 4.0), dpi=95, facecolor='#1a252f')
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_rms  = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.45, left=0.06, right=0.98, top=0.93, bottom=0.1)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=2)

        # Playback controls row
        ctrl = tk.Frame(parent, bg='#1a252f')
        ctrl.pack(fill='x', padx=4, pady=2)

        tk.Button(ctrl, text="PLAY ch2", font=('Consolas', 10, 'bold'),
                  bg='#27ae60', fg='white', width=10,
                  command=lambda: self.play_audio('ch2')).pack(side='left', padx=3, pady=3)
        for ch in ['ch0', 'ch1', 'esp32']:
            tk.Button(ctrl, text=ch, font=('Consolas', 8),
                      bg='#2471a3', fg='white', width=6,
                      command=lambda c=ch: self.play_audio(c)).pack(side='left', padx=2, pady=3)
        self.btn_pause = tk.Button(ctrl, text="PAUSA [P]", font=('Consolas', 9, 'bold'),
                  bg='#d4ac0d', fg='#1a252f', width=10,
                  command=self.toggle_pause)
        self.btn_pause.pack(side='left', padx=3, pady=3)
        tk.Button(ctrl, text="STOP [S]", font=('Consolas', 9, 'bold'),
                  bg='#922b21', fg='white', width=9,
                  command=self.stop_audio).pack(side='left', padx=4, pady=3)

        # Selection status
        self.selection_var = tk.StringVar(value="Arrastra en el waveform para seleccionar una franja")
        tk.Label(ctrl, textvariable=self.selection_var,
                 font=('Consolas', 9), bg='#1a252f', fg='#f39c12').pack(side='left', padx=8)

    def _build_label_panel(self, parent):
        lf = tk.LabelFrame(parent,
                           text="  Asignar label a seleccion (tecla 1-9 o click):  ",
                           font=('Consolas', 9, 'bold'),
                           bg='#2c3e50', fg='#ecf0f1', labelanchor='nw')
        lf.pack(fill='x', padx=4, pady=2)
        self._label_frame = lf

        self._rebuild_label_buttons(lf)

        # Action row
        action = tk.Frame(lf, bg='#2c3e50')
        action.pack(fill='x', padx=4, pady=3)
        tk.Button(action, text="Deshacer  [Backspace]",
                  font=('Consolas', 8), bg='#7f8c8d', fg='white', width=20,
                  command=self.undo_last_region).pack(side='left', padx=3)
        tk.Button(action, text="Borrar todo  [Del]",
                  font=('Consolas', 8), bg='#922b21', fg='white', width=14,
                  command=self.clear_regions).pack(side='left', padx=3)
        tk.Button(action, text="+ NUEVO LABEL",
                  font=('Consolas', 8, 'bold'), bg='#6c5ce7', fg='white', width=14,
                  command=self._add_custom_label).pack(side='left', padx=3)
        tk.Button(action, text="ACEPTAR AUTO  [A]",
                  font=('Consolas', 8, 'bold'), bg='#00b894', fg='white', width=16,
                  command=self.accept_autolabel).pack(side='left', padx=3)
        tk.Button(action, text="CONFIRMAR Y SIGUIENTE  [Enter]",
                  font=('Consolas', 10, 'bold'), bg='#27ae60', fg='white',
                  command=self.confirm_and_advance).pack(side='right', padx=5)

    def _rebuild_label_buttons(self, parent_frame=None):
        """(Re)create label buttons — called at startup and when a new label is added."""
        lf = parent_frame or self._label_frame
        if lf is None:
            return
        # Destroy existing button rows (not the action row, which is always last)
        for widget in lf.winfo_children():
            if isinstance(widget, tk.Frame) and widget != getattr(self, '_action_row', None):
                widget.destroy()

        self.label_buttons = {}
        labels_per_row = 5
        rows_needed = (len(ALL_LABELS) + labels_per_row - 1) // labels_per_row
        btn_rows = []
        for _ in range(rows_needed):
            row = tk.Frame(lf, bg='#2c3e50')
            row.pack(pady=1, before=lf.winfo_children()[-1] if lf.winfo_children() else None)
            btn_rows.append(row)

        for i, label in enumerate(ALL_LABELS):
            color = LABEL_COLORS[label]
            row   = btn_rows[i // labels_per_row]
            key_n = str(i + 1) if i < 9 else ('0' if i == 9 else '-')
            btn = tk.Button(row,
                            text=f"[{key_n}] {label.upper()}",
                            font=('Consolas', 8, 'bold'),
                            bg='#2d3436', fg='#b2bec3',
                            activebackground=color,
                            width=15, height=1, relief='flat', bd=2,
                            command=lambda l=label: self.assign_label_to_selection(l))
            btn.pack(side='left', padx=2, pady=1)
            self.label_buttons[label] = (btn, color)

        # Rebind keyboard shortcuts (only keys 1-9)
        for i, label in enumerate(ALL_LABELS[:9]):
            key = str(i + 1)
            try:
                self.root.unbind(key)
            except Exception:
                pass
            self.root.bind(key, lambda e, l=label: self.assign_label_to_selection(l))

    def _build_nav_panel(self, parent):
        nav = tk.Frame(parent, bg='#2c3e50')
        nav.pack(fill='x', padx=4, pady=2)
        tk.Button(nav, text="<< Anterior", font=('Consolas', 9),
                  bg='#636e72', fg='white', width=14,
                  command=self.prev_segment).pack(side='left', padx=4)
        tk.Button(nav, text="Siguiente >>", font=('Consolas', 9),
                  bg='#636e72', fg='white', width=14,
                  command=self.next_segment).pack(side='left', padx=4)
        tk.Button(nav, text="Siguiente pendiente  >>", font=('Consolas', 9, 'bold'),
                  bg='#d35400', fg='white', width=20,
                  command=self.next_pending).pack(side='left', padx=4)

    # ── List management ───────────────────────────────────────────────────

    def populate_list(self):
        self.seg_listbox.delete(0, tk.END)
        self.filtered_indices = []
        filt = self.filter_var.get()

        for i, seg in enumerate(self.segments):
            key = f"{seg['test_id']}_{seg['segment_id']}"
            labeled    = key in self.labels
            has_auto   = key in self.autolabels
            if filt == 'pending'      and labeled:                        continue
            if filt == 'auto_pending' and (labeled or not has_auto):      continue
            if filt == 'labeled' and not labeled: continue
            if filt == 'holes_only' and 'hole' not in seg['segment_id']: continue
            if filt == 'noise_only' and 'noise' not in seg['segment_id']: continue

            entry = self.labels.get(key, {})
            regions = entry.get('regions', [])
            if regions:
                unique_labels = list(dict.fromkeys(r['label'] for r in regions))
                status = '+'.join(unique_labels[:3])
            elif has_auto:
                al = self.autolabels[key]
                status = f"AUTO:{al['label']}({int(al.get('confidence',0)*100)}%)"
            else:
                status = '?'
            dur = seg.get('duration_s', 0)
            if labeled:
                prefix = "[OK]"
            elif has_auto:
                prefix = "[~A]"
            else:
                prefix = "[  ]"
            display = f"{prefix} {seg['test_id']}/{seg['segment_id']} ({dur:.0f}s) {status}"
            self.seg_listbox.insert(tk.END, display)

            if labeled and regions:
                first_label = regions[0]['label']
                self.seg_listbox.itemconfig(
                    self.seg_listbox.size() - 1,
                    fg=LABEL_COLORS.get(first_label, '#95a5a6'))
            elif has_auto and not labeled:
                self.seg_listbox.itemconfig(
                    self.seg_listbox.size() - 1, fg='#fd79a8')

            self.filtered_indices.append(i)

    def apply_filter(self):
        self.populate_list()

    def on_list_select(self, event):
        sel = self.seg_listbox.curselection()
        if sel and sel[0] < len(self.filtered_indices):
            self.show_segment(self.filtered_indices[sel[0]])

    # ── Segment display ───────────────────────────────────────────────────

    def show_segment(self, idx):
        if idx < 0 or idx >= len(self.segments):
            return
        self.current_idx = idx
        seg = self.segments[idx]
        key = f"{seg['test_id']}_{seg['segment_id']}"

        # Stop audio and reset playback state
        self._stop_sd()
        self.playing = False
        self.paused  = False
        self.play_offset_s  = 0.0
        self._play_start_wall = None

        # Restore saved regions (manual) or pre-fill autolabel if none
        saved = self.labels.get(key, {})
        if saved.get('regions'):
            self.current_regions = list(saved['regions'])
        elif key in self.autolabels:
            al = self.autolabels[key]
            dur = seg.get('duration_s', 30.0)
            self.current_regions = [{
                'start_s':    0.0,
                'end_s':      round(dur, 3),
                'duration_s': round(dur, 3),
                'label':      al['label'],
                'source':     'auto',
                'confidence': al.get('confidence', 0.0),
            }]
        else:
            self.current_regions = []
        self.pending_selection = None

        # Update progress bar
        n_lab = sum(1 for s in self.segments
                    if f"{s['test_id']}_{s['segment_id']}" in self.labels)
        pct = int(100 * n_lab / self.n_total) if self.n_total else 0
        self.progress_label.config(
            text=f"[{pct:3d}%] {n_lab}/{self.n_total}  |  "
                 f"{seg['test_id']}/{seg['segment_id']}  ({seg.get('duration_s',0):.1f}s)")

        self.selection_var.set("Arrastra en el waveform para seleccionar una franja")
        self._refresh_label_buttons_dim()

        if HAS_MPL:
            self._load_audio(seg)
            self.draw_waveform()

        self.update_context(seg)

    def _load_audio(self, seg):
        """Load audio into self.audio_data / self.audio_sr."""
        files = seg.get('files', {})
        rel = files.get('ch2', files.get('ch0', ''))
        path = SEGMENTS_DIR / rel if rel else None
        if path and path.exists():
            try:
                self.audio_data, self.audio_sr = sf.read(str(path), dtype='float32')
            except Exception:
                self.audio_data, self.audio_sr = None, None
        else:
            self.audio_data, self.audio_sr = None, None

    def draw_waveform(self):
        """Redraw waveform + RMS with current_regions overlaid."""
        self.ax_wave.clear()
        self.ax_rms.clear()

        if self.audio_data is not None:
            data = self.audio_data
            sr   = self.audio_sr
            t    = np.arange(len(data)) / sr
            dur  = t[-1]

            # Waveform — envelope render (min/max per window), max 1200 columns
            TARGET_COLS = 1200
            if len(data) > TARGET_COLS * 2:
                seg_len = max(1, len(data) // TARGET_COLS)
                n_segs  = len(data) // seg_len
                chunks  = data[:n_segs * seg_len].reshape(n_segs, seg_len)
                env_max = chunks.max(axis=1)
                env_min = chunks.min(axis=1)
                t_env   = np.arange(n_segs) * seg_len / sr
                self.ax_wave.fill_between(t_env, env_min, env_max,
                                          color='#2e86de', alpha=0.75)
                self.ax_wave.plot(t_env, env_max, color='#5dade2', linewidth=0.4, alpha=0.6)
                self.ax_wave.plot(t_env, env_min, color='#5dade2', linewidth=0.4, alpha=0.6)
            else:
                self.ax_wave.plot(t, data, color='#2e86de', linewidth=0.5)
            self.ax_wave.set_xlim(0, dur)
            self.ax_wave.set_ylim(max(-1.05, data.min() * 1.1),
                                  min( 1.05, data.max() * 1.1))
            seg = self.segments[self.current_idx]
            self.ax_wave.set_title(
                f"ch2 — {seg['segment_id']}  ({dur:.1f}s)  "
                f"| {len(self.current_regions)} region(es)",
                color='white', fontsize=8, pad=3)

            # RMS envelope
            win  = min(2048, len(data) // 20)
            hop  = win // 4
            n_st = max(1, (len(data) - win) // hop)
            rms  = np.array([np.sqrt(np.mean(data[i*hop:i*hop+win]**2))
                             for i in range(n_st)])
            t_rms = np.arange(n_st) * hop / sr
            self.ax_rms.plot(t_rms, rms, color='#e74c3c', linewidth=0.8)
            self.ax_rms.fill_between(t_rms, rms, alpha=0.25, color='#e74c3c')
            self.ax_rms.set_xlim(0, dur)
            self.ax_rms.set_title("RMS Envelope", color='white', fontsize=7, pad=2)
            self.ax_rms.set_xlabel("Tiempo (s)", color='#aaa', fontsize=7)

            # Draw saved regions
            for reg in self.current_regions:
                color   = LABEL_COLORS.get(reg['label'], '#95a5a6')
                is_auto = reg.get('source') == 'auto'
                alpha   = 0.18 if is_auto else REGION_ALPHA
                hatch   = '//' if is_auto else None
                self.ax_wave.axvspan(reg['start_s'], reg['end_s'],
                                     alpha=alpha, color=color, zorder=2,
                                     hatch=hatch, linewidth=0)
                conf_str = f" ~{int(reg.get('confidence',0)*100)}%" if is_auto else ""
                self.ax_wave.text(
                    (reg['start_s'] + reg['end_s']) / 2,
                    self.ax_wave.get_ylim()[1] * 0.82,
                    reg['label'][:8] + conf_str,
                    color='white', fontsize=6,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc=color,
                              alpha=0.85 if is_auto else 0.7))
                self.ax_rms.axvspan(reg['start_s'], reg['end_s'],
                                    alpha=0.15 if is_auto else 0.25,
                                    color=color, zorder=2)

            # Draw pending selection
            if self.pending_selection:
                s, e = self.pending_selection
                self.ax_wave.axvspan(s, e, alpha=0.25, color='#f1c40f', zorder=3,
                                     hatch='//', linewidth=0)
                self.ax_rms.axvspan(s, e, alpha=0.25, color='#f1c40f', zorder=3)

            # Playhead cursor lines (drawn last so they're on top)
            self.playhead_wave = self.ax_wave.axvline(
                x=-1, color='#ffffff', linewidth=1.2, alpha=0.85, zorder=10, linestyle='--')
            self.playhead_rms = self.ax_rms.axvline(
                x=-1, color='#ffffff', linewidth=1.2, alpha=0.85, zorder=10, linestyle='--')

        else:
            self.ax_wave.text(0.5, 0.5, "Sin audio (ch2 no disponible)",
                              transform=self.ax_wave.transAxes,
                              color='#e74c3c', ha='center', fontsize=10)

        for ax in [self.ax_wave, self.ax_rms]:
            ax.set_facecolor('#0d1b2a')
            ax.tick_params(colors='#aaa', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a4a4a')
        self.fig.set_facecolor('#1a252f')

        self.canvas.draw()

        # (Re)install SpanSelector after every redraw
        self._install_span_selector()

    def _install_span_selector(self):
        """Install matplotlib SpanSelector on the waveform axis."""
        if not HAS_MPL:
            return
        if self.span_selector:
            try:
                self.span_selector.disconnect_events()
            except Exception:
                pass
        self.span_selector = SpanSelector(
            self.ax_wave,
            self.on_span_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='#f1c40f'),
            interactive=False,
            drag_from_anywhere=True,
        )

    def on_span_select(self, xmin, xmax):
        """Called when user finishes dragging on waveform."""
        if xmax - xmin < 0.05:
            return
        dur = self.audio_data.shape[0] / self.audio_sr if self.audio_data is not None else 1e9
        xmin = max(0.0, xmin)
        xmax = min(dur, xmax)
        self.pending_selection = (round(xmin, 3), round(xmax, 3))
        self.selection_var.set(
            f"Seleccion: {xmin:.2f}s — {xmax:.2f}s  ({xmax-xmin:.2f}s)  →  presiona tecla 1-9")
        self._refresh_label_buttons_active()
        self.draw_waveform()

    # ── Label assignment ──────────────────────────────────────────────────

    def assign_label_to_selection(self, label):
        """Assign a label to the pending selection and add it as a region."""
        if self.pending_selection is None:
            return
        s, e = self.pending_selection
        self.current_regions.append({
            'start_s':    s,
            'end_s':      e,
            'duration_s': round(e - s, 3),
            'label':      label,
        })
        self.pending_selection = None
        self.selection_var.set(
            f"Region añadida: [{s:.2f}s — {e:.2f}s] = {label.upper()}  "
            f"| {len(self.current_regions)} region(es) total")
        self._refresh_label_buttons_dim()
        self.draw_waveform()
        self.update_context(self.segments[self.current_idx])

    def undo_last_region(self):
        if self.current_regions:
            removed = self.current_regions.pop()
            self.selection_var.set(f"Deshecho: {removed['label']} [{removed['start_s']:.2f}—{removed['end_s']:.2f}s]")
            self.draw_waveform()
            self.update_context(self.segments[self.current_idx])

    def clear_regions(self):
        if self.current_regions and messagebox.askyesno(
                "Confirmar", "Borrar TODAS las regiones de este segmento?"):
            self.current_regions = []
            self.pending_selection = None
            self.selection_var.set("Regiones borradas.")
            self.draw_waveform()
            self.update_context(self.segments[self.current_idx])

    def confirm_and_advance(self):
        """Save current regions and go to next pending."""
        if not self.current_regions:
            messagebox.showwarning("Sin regiones", "Añade al menos una region antes de confirmar.\n"
                                   "(Si quieres excluir el segmento, añade una region completa con label 'skip')")
            return
        seg = self.segments[self.current_idx]
        key = f"{seg['test_id']}_{seg['segment_id']}"
        self.labels[key] = {
            'regions':    self.current_regions,
            'timestamp':  datetime.now().isoformat(),
            'test_id':    seg['test_id'],
            'segment_id': seg['segment_id'],
            'duration_s': seg.get('duration_s', 0),
        }
        self.save_labels()
        self.current_regions = []
        self.pending_selection = None
        self._refresh_label_buttons_dim()
        self.populate_list()
        self.next_pending()

    def _refresh_label_buttons_active(self):
        """Brighten all buttons to show they can be clicked now."""
        for label, (btn, color) in self.label_buttons.items():
            btn.config(bg=color, fg='white', relief='raised')

    def _refresh_label_buttons_dim(self):
        """Dim buttons when no selection is active."""
        for label, (btn, color) in self.label_buttons.items():
            btn.config(bg='#2d3436', fg='#636e72', relief='flat')

    # ── Context panel ─────────────────────────────────────────────────────

    def update_context(self, seg):
        self.info_text.delete('1.0', tk.END)
        key = f"{seg['test_id']}_{seg['segment_id']}"
        saved = self.labels.get(key, {})

        lines = []
        lines += [
            f"{'='*38}",
            f"  {seg['segment_id'].upper()}",
            f"  Test: {seg['test_id']}  |  Broca: {seg.get('drill_bit','?')}",
            f"  {seg.get('start_s',0):.2f}s — {seg.get('end_s',0):.2f}s  ({seg.get('duration_s',0):.1f}s)",
            f"  Auto: {seg.get('auto_label','?')}  |  Metodo: {seg.get('method','?')}",
            f"{'='*38}", "",
        ]

        # Saved status
        if saved:
            lines.append(f"  [GUARDADO {saved.get('timestamp','')[:16]}]")
        else:
            lines.append(f"  [PENDIENTE]")
        lines.append("")

        # Current regions (in-progress or saved)
        regions_to_show = self.current_regions or saved.get('regions', [])
        if regions_to_show:
            lines.append(f"  REGIONES ({len(regions_to_show)}):")
            for i, r in enumerate(regions_to_show):
                lines.append(
                    f"  {i+1:2d}. {r['start_s']:6.2f}s — {r['end_s']:6.2f}s"
                    f"  ({r['duration_s']:.2f}s)  {r['label'].upper()}")
        else:
            lines.append("  Sin regiones aun.")
        lines.append("")

        # Context notes
        if seg['test_id'] in CONTEXT_NOTES:
            lines += [f"  --- CONTEXTO ---", f"  {CONTEXT_NOTES[seg['test_id']]}", ""]

        # Notes
        if seg.get('notes'):
            lines += [f"  Notas: {seg['notes']}", ""]

        # Label definitions
        lines += [f"  --- LABELS ---"]
        for i, lbl in enumerate(ALL_LABELS):
            key_n = str(i + 1) if i < 9 else '0'
            lines.append(f"  [{key_n}] {lbl}: {LABEL_NOTES[lbl]}")
        lines += [
            "",
            f"  --- ATAJOS ---",
            f"  Arrastra waveform = seleccionar franja",
            f"  1-9 = asignar label a seleccion",
            f"  Backspace = deshacer ultima region",
            f"  Delete = borrar todo",
            f"  Enter = CONFIRMAR y siguiente",
            f"  Space = Play ch2   P = Pausa   S = Stop",
            f"  Flechas = navegar segmentos",
        ]

        self.info_text.insert('1.0', '\n'.join(lines))

    # ── Playback ──────────────────────────────────────────────────────────

    PLAYBACK_SR = 48000  # resample to standard rate for clean playback

    def _load_play_data(self, channel):
        """Load audio data for playback, resampled to 48kHz."""
        seg = self.segments[self.current_idx]
        files = seg.get('files', {})
        rel = files.get(channel, '')
        if not rel and channel == 'ch0':
            rel = files.get('ch2', '')
        path = SEGMENTS_DIR / rel if rel else None
        if path and path.exists():
            try:
                data, sr = sf.read(str(path), dtype='float32')
                # Ensure mono
                if data.ndim > 1:
                    data = data[:, 0]
                # Resample to 48kHz for clean Bluetooth/USB playback
                if sr != self.PLAYBACK_SR:
                    n_new = int(round(len(data) * self.PLAYBACK_SR / sr))
                    data = scipy_resample(data, n_new).astype('float32')
                self.play_data = np.ascontiguousarray(data, dtype='float32')
                self.play_sr   = self.PLAYBACK_SR
                self.play_channel = channel
                return True
            except Exception:
                pass
        self.selection_var.set(f"No disponible: {channel}")
        return False

    def play_audio(self, channel='ch2'):
        """Start playback from beginning (or current offset if same channel)."""
        if channel != self.play_channel:
            self.play_offset_s = 0.0
        self._stop_sd()
        self.paused = False
        self.btn_pause.config(text="PAUSA [P]", bg='#d4ac0d', fg='#1a252f')

        if not self._load_play_data(channel):
            return
        if not HAS_SD:
            self.selection_var.set("sounddevice no disponible")
            return

        start_sample = int(self.play_offset_s * self.play_sr)
        data_to_play = self.play_data[start_sample:]

        # Reset frame counter; play_start_wall set inside thread to avoid
        # thread-startup offset accumulating against the cap
        self._play_frame_count  = 0
        self._play_total_frames = len(data_to_play)
        self._play_start_wall   = None
        self.playing = True

        def _play():
            try:
                self._play_start_wall = time.time()
                sd.play(data_to_play, self.play_sr)
                sd.wait()
            except Exception:
                pass
            finally:
                self.playing = False

        threading.Thread(target=_play, daemon=True).start()

    def toggle_pause(self):
        """Pause or resume playback."""
        if not HAS_SD:
            return
        if not self.playing and not self.paused:
            return

        if self.paused:
            # Resume from saved offset
            self.paused = False
            self.btn_pause.config(text="PAUSA [P]", bg='#d4ac0d', fg='#1a252f')

            start_sample = int(self.play_offset_s * self.play_sr)
            data_to_play = self.play_data[start_sample:] if self.play_data is not None else np.array([])

            self._play_frame_count  = 0
            self._play_total_frames = len(data_to_play)
            self._play_start_wall   = None
            self.playing = True

            def _resume():
                try:
                    self._play_start_wall = time.time()
                    sd.play(data_to_play, self.play_sr)
                    sd.wait()
                except Exception:
                    pass
                finally:
                    self.playing = False

            import threading
            threading.Thread(target=_resume, daemon=True).start()
        else:
            # Pause: save current position
            t0 = self._play_start_wall
            elapsed = (time.time() - t0) if t0 is not None else 0.0
            self.play_offset_s = min(
                self.play_offset_s + elapsed,
                len(self.play_data) / self.play_sr if self.play_data is not None else 0)
            self._stop_sd()
            self.playing = False
            self.paused = True
            self.btn_pause.config(text="REANUDAR [P]", bg='#27ae60', fg='white')

    def _stop_sd(self):
        if HAS_SD:
            try:
                sd.stop()
            except Exception:
                pass

    def stop_audio(self):
        self._stop_sd()
        self.playing = False
        self.paused  = False
        self.play_offset_s    = 0.0
        self._play_start_wall = None
        self.btn_pause.config(text="PAUSA [P]", bg='#d4ac0d', fg='#1a252f')

    # ── Playhead cursor update loop ───────────────────────────────────────

    def _tick_playhead(self):
        """Called every 40ms to update the playhead line position."""
        try:
            if self.playing and not self.paused and self._play_start_wall is not None:
                elapsed = time.time() - self._play_start_wall
                pos_s   = self.play_offset_s + elapsed
                # Cap at audio_data duration (matches waveform x-axis)
                dur = (len(self.audio_data) / self.audio_sr
                       if self.audio_data is not None else 0)
                pos_s = min(pos_s, dur)
                self._move_playhead(pos_s)
            elif not self.playing and not self.paused:
                self._move_playhead(-1)   # hide off-screen
        except Exception:
            pass
        self.root.after(40, self._tick_playhead)

    def _move_playhead(self, pos_s):
        """Update the vertical playhead lines without full redraw."""
        try:
            if self.playhead_wave and self.playhead_rms:
                self.playhead_wave.set_xdata([pos_s, pos_s])
                self.playhead_rms.set_xdata([pos_s, pos_s])
                self.canvas.draw_idle()
        except Exception:
            pass

    # ── Navigation ────────────────────────────────────────────────────────

    def next_segment(self):
        self.show_segment(min(self.current_idx + 1, len(self.segments) - 1))

    def prev_segment(self):
        self.show_segment(max(self.current_idx - 1, 0))

    def next_pending(self):
        start = self.current_idx + 1
        for i in list(range(start, len(self.segments))) + list(range(0, start)):
            key = f"{self.segments[i]['test_id']}_{self.segments[i]['segment_id']}"
            if key not in self.labels:
                self.show_segment(i)
                return
        messagebox.showinfo("Completo",
                            "Todos los segmentos han sido etiquetados.\n"
                            "Presiona STATS para ver el resumen.")

    # ── Stats ─────────────────────────────────────────────────────────────

    def show_stats(self):
        from collections import Counter
        total   = self.n_total
        labeled = len(self.labels)
        pct     = int(100 * labeled / total) if total else 0

        n_holes_done = sum(1 for s in self.segments
                           if 'hole' in s.get('segment_id', '')
                           and f"{s['test_id']}_{s['segment_id']}" in self.labels)
        n_noise_done = sum(1 for s in self.segments
                           if 'noise' in s.get('segment_id', '')
                           and f"{s['test_id']}_{s['segment_id']}" in self.labels)

        counter = Counter()
        total_regions = 0
        for entry in self.labels.values():
            for r in entry.get('regions', []):
                counter[r['label']] += 1
                total_regions += 1

        lines = [
            f"PROGRESO",
            f"  Segmentos etiquetados: {labeled}/{total} ({pct}%)",
            f"  Pendientes:            {total - labeled}",
            f"  Holes:  {n_holes_done}/{self.n_holes}",
            f"  Noise:  {n_noise_done}/{self.n_noise}",
            f"  Regiones totales:      {total_regions}",
            f"",
            f"DISTRIBUCION DE LABELS (en regiones):",
        ]
        for lbl, cnt in counter.most_common():
            pct_l = int(100 * cnt / total_regions) if total_regions else 0
            lines.append(f"  {lbl:<20} {cnt:>4} regiones  ({pct_l}%)")

        if labeled == total:
            lines += ["", "*** ETIQUETADO COMPLETO ***",
                      f"Drilling confirmados: {counter.get('drilling', 0)} regiones",
                      f"Post-fractura:        {counter.get('post_fracture', 0)} regiones",
                      f"Labels guardados en:  {LABELS_FILE}"]

        messagebox.showinfo("Estadisticas", '\n'.join(lines))


    # ── Nuevo label personalizado ─────────────────────────────────────────

    def _add_custom_label(self):
        """Dialog to create a new audio label type."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Nuevo Label")
        dlg.geometry("400x260")
        dlg.configure(bg='#2c3e50')
        dlg.grab_set()

        tk.Label(dlg, text="Nombre del label (sin espacios, minusculas):",
                 bg='#2c3e50', fg='white', font=('Consolas', 9)).pack(pady=(14, 2))
        name_var = tk.StringVar()
        name_entry = tk.Entry(dlg, textvariable=name_var, font=('Consolas', 11),
                              bg='#0d1b2a', fg='#ecf0f1', insertbackground='white', width=28)
        name_entry.pack(pady=2)
        name_entry.focus()

        tk.Label(dlg, text="Descripcion (que tipo de sonido es):",
                 bg='#2c3e50', fg='white', font=('Consolas', 9)).pack(pady=(10, 2))
        note_var = tk.StringVar()
        tk.Entry(dlg, textvariable=note_var, font=('Consolas', 10),
                 bg='#0d1b2a', fg='#ecf0f1', insertbackground='white', width=36).pack(pady=2)

        # Color selector
        tk.Label(dlg, text="Color (dejar vacio = auto):",
                 bg='#2c3e50', fg='white', font=('Consolas', 9)).pack(pady=(10, 2))
        color_var = tk.StringVar()
        color_row = tk.Frame(dlg, bg='#2c3e50')
        color_row.pack()
        for c in _CUSTOM_PALETTE[:8]:
            b = tk.Button(color_row, bg=c, width=3, height=1,
                          command=lambda col=c: color_var.set(col))
            b.pack(side='left', padx=1)

        status_var = tk.StringVar()
        tk.Label(dlg, textvariable=status_var, bg='#2c3e50', fg='#e74c3c',
                 font=('Consolas', 8)).pack(pady=4)

        def _confirm():
            name = name_var.get().strip().lower().replace(' ', '_')
            note = note_var.get().strip()
            if not name:
                status_var.set("El nombre no puede estar vacio.")
                return
            if name in LABEL_COLORS:
                status_var.set(f"'{name}' ya existe.")
                return
            # Pick color
            col = color_var.get() or _CUSTOM_PALETTE[len(ALL_LABELS) % len(_CUSTOM_PALETTE)]
            # Add to globals
            LABEL_COLORS[name] = col
            ALL_LABELS.append(name)
            LABEL_NOTES[name] = note or f'Label personalizado: {name}'
            # Save to disk
            existing = []
            if CUSTOM_LABELS_FILE.exists():
                try:
                    with open(CUSTOM_LABELS_FILE, encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    pass
            existing.append({'name': name, 'color': col, 'note': note or ''})
            with open(CUSTOM_LABELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            # Rebuild buttons
            self._rebuild_label_buttons()
            dlg.destroy()
            self.selection_var.set(f"Nuevo label creado: '{name.upper()}'")

        tk.Button(dlg, text="CREAR LABEL", font=('Consolas', 10, 'bold'),
                  bg='#6c5ce7', fg='white', command=_confirm).pack(pady=8)
        dlg.bind('<Return>', lambda e: _confirm())

    # ── Auto-etiquetado ───────────────────────────────────────────────────

    def _extract_audio_features(self, seg):
        """Extract simple features from a segment's audio file for KNN."""
        files = seg.get('files', {})
        rel   = files.get('ch2', files.get('ch0', files.get('ch1', '')))
        path  = SEGMENTS_DIR / rel if rel else None
        if not path or not path.exists():
            return None
        try:
            data, sr = sf.read(str(path), dtype='float32')
            if data.ndim > 1:
                data = data[:, 0]
            if len(data) < 100:
                return None
            rms      = float(np.sqrt(np.mean(data ** 2)))
            rms_std  = float(np.std([np.sqrt(np.mean(data[i:i+sr] ** 2))
                                     for i in range(0, max(1, len(data)-sr), sr // 2)]))
            zcr      = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)
            dur      = len(data) / sr
            # Spectral centroid (cheap version)
            spectrum = np.abs(np.fft.rfft(data[:min(len(data), 4096)]))
            freqs    = np.fft.rfftfreq(min(len(data), 4096), 1.0 / sr)
            sc       = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-9))
            return np.array([rms, rms_std, zcr, sc / 1000.0, dur], dtype=np.float32)
        except Exception:
            return None

    def run_autolabeler(self):
        """Auto-label all unlabeled segments using KNN + segment-type rules."""
        unlabeled = [s for s in self.segments
                     if f"{s['test_id']}_{s['segment_id']}" not in self.labels
                     and f"{s['test_id']}_{s['segment_id']}" not in self.autolabels]
        if not unlabeled:
            messagebox.showinfo("Auto-etiquetado",
                                "No hay segmentos pendientes sin auto-label.\n"
                                "Usa el filtro 'auto_pending' para validarlos.")
            return

        self._auto_btn_var.set("Procesando...")
        self.root.update()

        # ── Build training set from manually labeled segments ──
        train_X, train_y = [], []
        for seg in self.segments:
            key = f"{seg['test_id']}_{seg['segment_id']}"
            if key not in self.labels:
                continue
            regions = self.labels[key].get('regions', [])
            if not regions:
                continue
            # Majority label for this segment
            from collections import Counter
            maj = Counter(r['label'] for r in regions).most_common(1)[0][0]
            feat = self._extract_audio_features(seg)
            if feat is not None:
                train_X.append(feat)
                train_y.append(maj)

        use_knn = len(train_X) >= 3
        if use_knn:
            train_arr = np.array(train_X)
            # Z-score normalize
            mu  = train_arr.mean(axis=0)
            std = train_arr.std(axis=0) + 1e-9

        n_auto = 0
        for seg in unlabeled:
            key      = f"{seg['test_id']}_{seg['segment_id']}"
            seg_id   = seg.get('segment_id', '')
            is_hole  = 'hole' in seg_id
            dur      = seg.get('duration_s', 30.0)

            # Rule 1: hole segments → drilling (high confidence)
            if is_hole:
                label, conf = 'drilling', 0.92
            else:
                # Try KNN on noise segments
                if use_knn:
                    feat = self._extract_audio_features(seg)
                    if feat is not None:
                        feat_norm = (feat - mu) / std
                        train_norm = (train_arr - mu) / std
                        dists = np.linalg.norm(train_norm - feat_norm, axis=1)
                        k = min(3, len(train_y))
                        nn_idx = np.argsort(dists)[:k]
                        nn_labels = [train_y[i] for i in nn_idx]
                        from collections import Counter
                        most_common = Counter(nn_labels).most_common(1)[0]
                        label = most_common[0]
                        conf  = most_common[1] / k
                    else:
                        label, conf = 'unknown', 0.0
                else:
                    # No training data — use duration heuristics for noise
                    if dur < 5:
                        label, conf = 'repositioning', 0.55
                    elif dur < 15:
                        label, conf = 'taladrina', 0.50
                    else:
                        label, conf = 'machine_noise', 0.50

            self.autolabels[key] = {
                'label':      label,
                'confidence': round(conf, 3),
                'source':     'auto',
                'segment_id': seg_id,
                'test_id':    seg.get('test_id', ''),
            }
            n_auto += 1

        self.save_autolabels()
        self._auto_btn_var.set(f"AUTO-ETIQUETAR ({n_auto} nuevos)")
        self.populate_list()
        messagebox.showinfo(
            "Auto-etiquetado completo",
            f"{n_auto} segmentos auto-etiquetados.\n"
            f"Usa el filtro 'auto_pending' para validarlos uno por uno.\n"
            f"Tecla [A] o boton 'ACEPTAR AUTO' para confirmar cada uno.")

    def accept_autolabel(self):
        """Accept the current auto-label as a confirmed manual label."""
        seg = self.segments[self.current_idx]
        key = f"{seg['test_id']}_{seg['segment_id']}"

        # Find auto regions in current_regions
        auto_regions = [r for r in self.current_regions if r.get('source') == 'auto']
        manual_regions = [r for r in self.current_regions if r.get('source') != 'auto']

        if not auto_regions and key not in self.autolabels:
            self.selection_var.set("No hay auto-label para este segmento.")
            return

        # Convert auto regions to confirmed (remove 'source'='auto' flag)
        confirmed = []
        for r in auto_regions:
            c = dict(r)
            c['source'] = 'validated'
            confirmed.append(c)
        confirmed += manual_regions

        if not confirmed:
            self.selection_var.set("Sin regiones para confirmar.")
            return

        self.labels[key] = {
            'regions':    confirmed,
            'timestamp':  datetime.now().isoformat(),
            'test_id':    seg.get('test_id', ''),
            'segment_id': seg.get('segment_id', ''),
            'duration_s': seg.get('duration_s', 0),
            'source':     'validated_auto',
        }
        self.save_labels()
        # Remove from autolabels
        self.autolabels.pop(key, None)
        self.save_autolabels()
        self.current_regions = []
        self.pending_selection = None
        self.populate_list()
        self.next_pending()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
