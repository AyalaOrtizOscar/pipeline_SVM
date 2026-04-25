#!/usr/bin/env python3
"""
Segment Art.2 continuous WAV recordings into individual drill holes.
V3: Spectral CutRatio method — detects individual G81 cycles by
    the high-frequency spike during rapid retract/reposition between holes.

Key insight: During actual cutting, spectral energy is broadband (chip
formation noise). During rapid retract + XY reposition + rapid approach,
the spindle-in-air sound shifts energy to higher frequencies, creating
a detectable spike in the CutRatio (3-12kHz / 200-1500Hz).

Output: D:/pipeline_SVM/art2_segments/E_testNN/
"""
import sys
sys.path.insert(0, "D:/pipeline_SVM/scripts")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import soundfile as sf
import json, csv, shutil
from pathlib import Path
from scipy.signal import find_peaks, medfilt
from scipy.ndimage import gaussian_filter1d

# ── Config ──────────────────────────────────────────────────────────────
E_BASE = Path("E:/Datos Generados")
OUT_BASE = Path("D:/pipeline_SVM/art2_segments")
OUT_BASE.mkdir(parents=True, exist_ok=True)

SR_NI = 51200
FRAME_DUR = 0.5       # seconds per FFT frame
HOP_DUR = 0.25        # seconds hop between frames
MIN_PEAK_DIST = 8.0   # seconds minimum between hole transitions
MIN_BLOCK_DUR = 5.0   # seconds minimum drilling block
BLOCK_MERGE_GAP = 8.0 # seconds to merge close blocks
HOLE_PADDING = 0.5    # seconds padding before/after each hole

TESTS_6MM = {
    # test_id: (folder, drill_bit, hole_start, expected_holes, notes)
    'test15': ('6mm_test15', '6mm#1', 8,   42,  'Broca#1 cont., mirilla+endoscopio, 872s'),
    'test16': ('6mm_test16', '6mm#1', 50,  82,  'Broca#1 cont., 604s'),
    'test17': ('6mm_test17', '6mm#1', 133, 7,   'Broca#1 FIN ~7 holes, 30s, SATURADO'),
    'test18': ('6mm_test18', '6mm#2', 1,   37,  'Broca#2, 3267s=54min, FRACTURA h=37'),
    'test19': ('6mm_test19', '6mm#3', 1,   30,  'Broca#3, 984s=16min, cara1 disco2'),
    'test20': ('6mm_test20', '6mm#3', 31,  3,   'Broca#3 cont., 46s corto, SATURADO'),
    'test21': ('6mm_test21', '6mm#3', 34,  21,  'Broca#3 FIN, FRACTURA h=55, 1462s'),
}


# ── Spectral analysis ──────────────────────────────────────────────────
def compute_cut_ratio(data, sr):
    """Compute CutRatio: energy(3-12kHz) / energy(200-1500Hz) per frame.

    The ratio spikes during rapid retract/reposition (spindle-in-air sound)
    and drops during actual cutting (broadband chip formation noise).
    """
    frame_size = int(FRAME_DUR * sr)
    hop_size = int(HOP_DUR * sr)
    n_frames = max(0, (len(data) - frame_size) // hop_size + 1)

    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    band_cut = (freqs >= 3000) & (freqs < 12000)
    band_spn = (freqs >= 200) & (freqs < 1500)

    cut_ratio = np.zeros(n_frames, dtype=np.float32)
    frame_times = np.zeros(n_frames, dtype=np.float32)
    window = np.hanning(frame_size).astype(np.float32)

    for i in range(n_frames):
        start = i * hop_size
        frame = data[start:start + frame_size] * window
        fft_mag = np.abs(np.fft.rfft(frame))
        e_cut = np.sum(fft_mag[band_cut] ** 2)
        e_spn = np.sum(fft_mag[band_spn] ** 2)
        cut_ratio[i] = e_cut / (e_spn + 1e-10)
        frame_times[i] = start / sr

    return frame_times, cut_ratio


def is_saturated(data):
    """Check if signal is clipped/saturated (CutRatio won't work)."""
    peak = np.max(np.abs(data))
    # If >5% of samples are near peak, it's saturated
    near_peak = np.mean(np.abs(data) > 0.95 * peak)
    return peak > 0.9 and near_peak > 0.01


def detect_holes_spectral(frame_times, cut_ratio):
    """Detect individual drill holes using peak-first CutRatio method.

    Strategy (works for both high and low baseline signals):
    1. Find ALL transition peaks in CutRatio (spikes during retract/reposition)
    2. Cluster nearby peaks into drilling blocks
    3. Segments between peaks within a cluster = individual holes

    Returns: holes, blocks, threshold
    """
    dt = frame_times[1] - frame_times[0] if len(frame_times) > 1 else HOP_DUR

    # Smooth CutRatio
    k = max(3, int(1.0 / dt))
    if k % 2 == 0:
        k += 1
    cr_med = medfilt(cut_ratio, kernel_size=k)
    cr_smooth = gaussian_filter1d(cr_med, sigma=2)

    # --- Step 1: Find ALL transition peaks ---
    # Try progressively lower thresholds until we get a reasonable count
    min_dist_frames = max(1, int(MIN_PEAK_DIST / dt))
    median_cr = np.median(cr_smooth)

    # Try multiple threshold levels: p90, p80, p70, p60
    peaks = np.array([], dtype=int)
    used_threshold = 0.0
    for pct in [90, 80, 70, 60]:
        p_val = np.percentile(cr_smooth, pct)
        prom = max((p_val - median_cr) * 0.3, 0.3)
        candidates, _ = find_peaks(cr_smooth,
                                   distance=min_dist_frames,
                                   prominence=prom,
                                   height=max(p_val, 0.3))
        if len(candidates) >= len(peaks):
            peaks = candidates
            used_threshold = p_val
        # Stop if we found a reasonable number of peaks
        if len(peaks) >= 5:
            break

    peak_times = frame_times[peaks]

    if len(peak_times) == 0:
        return [], [], float(used_threshold)

    # --- Step 2: Cluster peaks into drilling blocks ---
    # Peaks within MAX_INTER_HOLE gap are in the same block
    MAX_INTER_HOLE = 25.0  # seconds max between consecutive holes
    clusters = []
    current_cluster = [peak_times[0]]

    for i in range(1, len(peak_times)):
        if peak_times[i] - peak_times[i - 1] <= MAX_INTER_HOLE:
            current_cluster.append(peak_times[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [peak_times[i]]
    clusters.append(current_cluster)

    # --- Step 3: Build blocks and holes ---
    blocks = []
    holes = []
    MARGIN = 5.0  # seconds margin before first and after last peak

    for cluster in clusters:
        bs = max(0, cluster[0] - MARGIN)
        be = min(frame_times[-1], cluster[-1] + MARGIN)
        blocks.append((bs, be))

        if len(cluster) == 1:
            # Single peak = likely 1-2 holes around it
            holes.append((bs, be))
        else:
            # First hole: before first peak
            if cluster[0] - bs > 2.0:
                holes.append((bs, cluster[0]))
            # Between consecutive peaks
            for j in range(len(cluster) - 1):
                s, e = cluster[j], cluster[j + 1]
                if e - s > 2.0:
                    holes.append((float(s), float(e)))
            # Last hole: after last peak
            if be - cluster[-1] > 2.0:
                holes.append((cluster[-1], be))

    return holes, blocks, float(used_threshold)


def detect_holes_rms(data, sr):
    """Fallback for saturated signals: use RMS envelope with Otsu threshold."""
    hop = int(0.1 * sr)
    win = int(0.2 * sr)
    n = (len(data) - win) // hop + 1

    rms = np.zeros(n)
    times = np.zeros(n)
    for i in range(n):
        chunk = data[i * hop:i * hop + win]
        rms[i] = np.sqrt(np.mean(chunk ** 2))
        times[i] = i * hop / sr

    # For saturated signals, look for dips in RMS (transitions between holes)
    rms_smooth = gaussian_filter1d(rms, sigma=5)
    thr = _otsu_threshold(rms_smooth)

    is_active = rms_smooth > thr
    blocks = _find_contiguous(times, is_active, 3.0, 2.0)

    # Within blocks, find valleys using inverted RMS
    holes = []
    for bs, be in blocks:
        mask = (times >= bs) & (times <= be)
        t_block = times[mask]
        r_block = rms_smooth[mask]
        block_dur = be - bs

        if block_dur < 20:
            holes.append((bs, be))
            continue

        # Find valleys (inverted peaks) with min_distance
        inv = -r_block
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        min_dist = max(1, int(8.0 / dt))
        valleys, _ = find_peaks(inv, distance=min_dist, prominence=np.std(r_block) * 0.5)

        if len(valleys) < 1:
            holes.append((bs, be))
            continue

        valley_times = t_block[valleys]
        # Build holes from valleys
        boundaries = [bs] + list(valley_times) + [be]
        for j in range(len(boundaries) - 1):
            s, e = boundaries[j], boundaries[j + 1]
            if e - s > 2.0:
                holes.append((s, e))

    return holes, blocks


def _find_contiguous(times, active, min_dur, merge_gap):
    """Find contiguous active regions, merge close ones, filter by min_dur."""
    blocks = []
    in_block = False
    for i in range(len(active)):
        if active[i] and not in_block:
            in_block = True
            bs = times[i]
        elif not active[i] and in_block:
            in_block = False
            be = times[i]
            blocks.append((bs, be))
    if in_block:
        blocks.append((bs, times[-1]))

    # Merge
    merged = []
    for b in blocks:
        if merged and (b[0] - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)

    return [(s, e) for s, e in merged if (e - s) >= min_dur]


def _otsu_threshold(values, n_bins=256):
    """Otsu's method for optimal bimodal threshold."""
    hist, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return np.median(values)

    best_thr, best_var = centers[0], 0
    cum_sum, cum_mean = 0, 0
    global_mean = np.sum(hist * centers)

    for i in range(len(hist)):
        cum_sum += hist[i]
        if cum_sum == 0 or cum_sum == total:
            continue
        cum_mean += hist[i] * centers[i]
        w0 = cum_sum / total
        m0 = cum_mean / cum_sum
        m1 = (global_mean - cum_mean) / (total - cum_sum)
        bv = w0 * (1 - w0) * (m0 - m1) ** 2
        if bv > best_var:
            best_var = bv
            best_thr = centers[i]
    return best_thr


# ── WAV/CSV extraction ─────────────────────────────────────────────────
def save_segment_wav(src_path, out_path, start_sec, end_sec, sr=SR_NI):
    """Extract WAV segment with bounds checking."""
    info = sf.info(str(src_path))
    max_frames = info.frames
    start_frame = max(0, int(start_sec * sr))
    end_frame = min(max_frames, int(end_sec * sr))
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        return False
    with sf.SoundFile(str(src_path)) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype='float32')
    sf.write(str(out_path), data, sr, subtype='FLOAT')
    return True


def save_flow_segment(flow_path, out_path, start_sec, end_sec):
    """Extract flow.csv rows within time range."""
    if not flow_path or not flow_path.exists():
        return
    rows_out = []
    header = None
    with open(flow_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        time_col = 0
        for i, h in enumerate(header):
            if 'time' in h.lower() or 'elapsed' in h.lower():
                time_col = i
                break
        for row in reader:
            try:
                t = float(row[time_col])
                if start_sec <= t <= end_sec:
                    rows_out.append(row)
            except (ValueError, IndexError):
                continue
    if rows_out:
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows_out)


def find_aux_files(test_dir):
    """Find ESP32 audio and flow.csv files."""
    esp32_path = None
    for name in ['esp32_audio.wav', 'esp_part_fixed.wav', 'rec_from_esp.wav']:
        p = test_dir / "MCU" / name
        if p.exists():
            esp32_path = p
            break
    flow_path = None
    for name in ['flow.csv', 'flow_from_esp.csv']:
        p = test_dir / "MCU" / name
        if p.exists():
            flow_path = p
            break
    return esp32_path, flow_path


# ── Pause classification ───────────────────────────────────────────────
def classify_pauses(blocks, total_dur):
    """Classify gaps between drilling blocks."""
    pauses = []
    prev_end = 0.0
    for s, e in blocks:
        if s - prev_end >= 2.0:
            dur = s - prev_end
            if dur <= 4:
                ptype = 'sopladora'
            elif dur <= 30:
                ptype = 'repositioning'
            else:
                ptype = 'measurement_pause'
            pauses.append((prev_end, s, ptype))
        prev_end = e
    if total_dur - prev_end >= 2.0:
        pauses.append((prev_end, total_dur, 'post_drilling'))
    return pauses


# ── Main pipeline ──────────────────────────────────────────────────────
def process_test(test_id, folder_name, drill_bit, hole_start, expected, notes):
    """Process one test: spectral segmentation into individual holes."""
    test_dir = E_BASE / folder_name
    out_dir = OUT_BASE / f"E_{test_id}"
    holes_dir = out_dir / "holes"
    noise_dir = out_dir / "noise"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    holes_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  {test_id} ({folder_name}) | {drill_bit} | start={hole_start} | ~{expected} holes")
    print(f"  {notes}")
    print(f"{'=' * 70}")

    # Find channels (check test_dir directly, then NI/ subfolder for newer tests)
    ch_paths = {}
    for f in sorted(test_dir.glob("ch*.wav")):
        ch_paths[f.stem.split()[0]] = f
    if not ch_paths:
        ni_dir = test_dir / "NI"
        for f in sorted(ni_dir.glob("ch*.wav")) if ni_dir.exists() else []:
            ch_paths[f.stem.split()[0]] = f
    if not ch_paths:
        print("  [SKIP] No WAV files")
        return None

    ref_ch = 'ch2' if 'ch2' in ch_paths else max(ch_paths.keys())
    ref_path = ch_paths[ref_ch]
    print(f"  Ref: {ref_ch} ({ref_path.name})")

    # Load reference channel
    with sf.SoundFile(str(ref_path)) as f:
        sr = f.samplerate
        data = f.read(dtype='float32')
    total_dur = len(data) / sr
    print(f"  Duration: {total_dur:.1f}s ({total_dur/60:.1f}min)")

    # Choose detection method
    saturated = is_saturated(data)
    if saturated:
        print("  [!] Signal SATURATED - using RMS fallback")
        holes, blocks = detect_holes_rms(data, sr)
        method = 'rms_fallback'
        threshold = 0
    else:
        print("  Computing CutRatio (3-12kHz / 200-1500Hz)...")
        frame_times, cut_ratio = compute_cut_ratio(data, sr)
        p50 = np.median(cut_ratio)
        p90 = np.percentile(cut_ratio, 90)
        p99 = np.percentile(cut_ratio, 99)
        print(f"  CutRatio: p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} max={cut_ratio.max():.2f}")
        holes, blocks, threshold = detect_holes_spectral(frame_times, cut_ratio)
        method = 'spectral_cutratio'

    print(f"  Blocks: {len(blocks)}, Holes detected: {len(holes)} (expected ~{expected})")
    for i, (s, e) in enumerate(blocks):
        n_in = sum(1 for hs, he in holes if s <= hs and he <= e + 1)
        print(f"    Block {i+1}: {s:.1f}s-{e:.1f}s ({e-s:.1f}s) -> {n_in} holes")

    # Classify pauses
    pauses = classify_pauses(blocks, total_dur)

    # Find auxiliary files
    esp32_path, flow_path = find_aux_files(test_dir)

    # ── Save hole segments ──
    print(f"  Saving {len(holes)} hole segments...")
    manifest = {
        'test_id': test_id,
        'folder': folder_name,
        'drill_bit': drill_bit,
        'hole_start_estimate': hole_start,
        'expected_holes': expected,
        'total_duration_s': round(total_dur, 3),
        'sr_ni': sr,
        'reference_channel': ref_ch,
        'detection_method': method,
        'threshold': float(threshold),
        'n_blocks': len(blocks),
        'n_holes_detected': len(holes),
        'notes': notes,
        'channels': sorted(ch_paths.keys()),
        'has_esp32': esp32_path is not None,
        'has_flow': flow_path is not None,
        'blocks': [{'start_s': round(float(s), 3), 'end_s': round(float(e), 3),
                     'duration_s': round(float(e - s), 3)} for s, e in blocks],
        'holes': [],
        'noise': [],
    }

    for i, (s, e) in enumerate(holes):
        hole_num = hole_start + i
        hole_id = f"hole_{hole_num:03d}"
        dur = e - s

        hole_meta = {
            'hole_id': hole_id,
            'hole_number': hole_num,
            'start_s': round(float(s), 3),
            'end_s': round(float(e), 3),
            'duration_s': round(float(dur), 3),
            'label': 'drilling_candidate',
            'confidence': 'auto_v3',
            'method': method,
            'files': {},
        }

        # Save all NI channels
        for ch_name, ch_path in sorted(ch_paths.items()):
            out_wav = holes_dir / f"{hole_id}_{ch_name}.wav"
            if save_segment_wav(ch_path, out_wav, s, e, sr):
                hole_meta['files'][ch_name] = str(out_wav.relative_to(OUT_BASE))

        # ESP32
        if esp32_path:
            try:
                esp_info = sf.info(str(esp32_path))
                esp_sr = esp_info.samplerate
                esp_dur = esp_info.frames / esp_sr
                if s < esp_dur:
                    out_esp = holes_dir / f"{hole_id}_esp32.wav"
                    if save_segment_wav(esp32_path, out_esp, s, min(e, esp_dur), esp_sr):
                        hole_meta['files']['esp32'] = str(out_esp.relative_to(OUT_BASE))
            except Exception:
                pass

        # Flow
        if flow_path:
            out_flow = holes_dir / f"{hole_id}_flow.csv"
            save_flow_segment(flow_path, out_flow, s, e)
            if out_flow.exists() and out_flow.stat().st_size > 10:
                hole_meta['files']['flow'] = str(out_flow.relative_to(OUT_BASE))

        # Metadata
        meta_path = holes_dir / f"{hole_id}_meta.json"
        with open(meta_path, 'w') as mf:
            json.dump(hole_meta, mf, indent=2)

        manifest['holes'].append(hole_meta)

    # ── Save noise segments ──
    print(f"  Saving {len(pauses)} noise segments...")
    for i, (s, e, ptype) in enumerate(pauses):
        noise_id = f"noise_{i+1:03d}"
        noise_meta = {
            'noise_id': noise_id,
            'start_s': round(float(s), 3),
            'end_s': round(float(e), 3),
            'duration_s': round(float(e - s), 3),
            'auto_type': ptype,
            'label': ptype,
            'confidence': 'auto_v3',
            'files': {},
        }
        if ref_ch in ch_paths:
            out_wav = noise_dir / f"{noise_id}_{ref_ch}.wav"
            if save_segment_wav(ch_paths[ref_ch], out_wav, s, e, sr):
                noise_meta['files'][ref_ch] = str(out_wav.relative_to(OUT_BASE))
        manifest['noise'].append(noise_meta)

    # Save manifest
    manifest_path = out_dir / "segments_manifest.json"
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)

    ratio = len(holes) / max(expected, 1) * 100
    print(f"  DONE: {len(holes)}/{expected} holes ({ratio:.0f}%), {len(pauses)} noise")
    return manifest


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  ART.2 -- Spectral Hole Segmentation v3")
    print("  Method: CutRatio (3-12kHz / 200-1500Hz) peak detection")
    print("  Fallback: RMS valley detection for saturated signals")
    print("=" * 70)

    all_manifests = []
    for tid, (folder, bit, h_start, expected, notes) in TESTS_6MM.items():
        try:
            m = process_test(tid, folder, bit, h_start, expected, notes)
            if m:
                all_manifests.append(m)
        except Exception as ex:
            print(f"  [ERROR] {tid}: {ex}")
            import traceback
            traceback.print_exc()

    # Build pending_review.json
    pending = []
    for m in all_manifests:
        for h in m['holes']:
            pending.append({
                'test_id': m['test_id'],
                'segment_id': h['hole_id'],
                'type': 'hole_candidate',
                'start_s': h['start_s'],
                'end_s': h['end_s'],
                'duration_s': h['duration_s'],
                'drill_bit': m['drill_bit'],
                'auto_label': h['label'],
                'confirmed_label': None,
                'method': h['method'],
                'files': h['files'],
                'notes': m['notes'],
            })
        for n in m['noise']:
            pending.append({
                'test_id': m['test_id'],
                'segment_id': n['noise_id'],
                'type': n['auto_type'],
                'start_s': n['start_s'],
                'end_s': n['end_s'],
                'duration_s': n['duration_s'],
                'drill_bit': m['drill_bit'],
                'auto_label': n['label'],
                'confirmed_label': None,
                'method': m.get('detection_method', ''),
                'files': n['files'],
                'notes': m['notes'],
            })

    pending_path = OUT_BASE / "pending_review.json"
    with open(pending_path, 'w') as f:
        json.dump(pending, f, indent=2)

    # Summary
    total_holes = sum(len(m['holes']) for m in all_manifests)
    total_noise = sum(len(m['noise']) for m in all_manifests)
    total_expected = sum(v[3] for v in TESTS_6MM.values())

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"  Tests: {len(all_manifests)} | Holes: {total_holes} (exp ~{total_expected})")
    print(f"  Noise segments: {total_noise} | Pending review: {len(pending)}")
    print(f"{'=' * 70}")
    for m in all_manifests:
        exp = TESTS_6MM[m['test_id']][3]
        n = len(m['holes'])
        pct = n / max(exp, 1) * 100
        status = 'OK' if 50 < pct < 200 else 'CHECK'
        avg_dur = np.mean([h['duration_s'] for h in m['holes']]) if m['holes'] else 0
        print(f"  {m['test_id']}: {n:3d}/{exp:3d} ({pct:5.0f}%) "
              f"avg={avg_dur:.1f}s/hole  [{m['detection_method']}] [{status}]")
    print("[DONE]")
