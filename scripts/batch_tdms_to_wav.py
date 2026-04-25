#!/usr/bin/env python3
"""
Batch TDMS → WAV converter for Art. 2 dataset.
Standalone CLI version (no PySide6 dependency).

Uses streaming (30s chunks) to handle files up to 25+ GB.
Two-pass: normalization then writing.

Usage:
    python batch_tdms_to_wav.py --source "C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados"
    python batch_tdms_to_wav.py --source "C:/..." --tests test6 test7 test9
    python batch_tdms_to_wav.py --source "C:/..." --all
"""

import argparse
import os
import sys
import time
import wave
import numpy as np
from nptdms import TdmsFile

CHUNK_SAMPLES = 1_536_000  # 30s at 51200 Hz
SAMPLE_RATE = 51200
BIT_DEPTH = 32
TARGET_PEAK = 0.9
ADDITIONAL_GAIN = 1.0


def convert_tdms(tdms_path, output_dir, sample_rate=SAMPLE_RATE):
    """Convert a single TDMS file to per-channel WAVs using streaming."""
    print(f"\n{'='*60}")
    print(f"Converting: {tdms_path}")
    print(f"Output to:  {output_dir}")

    file_size_mb = os.path.getsize(tdms_path) / (1024 * 1024)
    print(f"File size:  {file_size_mb:.0f} MB")

    t0 = time.time()

    with TdmsFile.open(tdms_path) as tdms:
        group = next((g for g in tdms.groups() if g.name == "NI_Acq"), None)
        if group is None:
            print("  ERROR: No 'NI_Acq' group found. Skipping.")
            return False

        channels = group.channels()
        if not channels:
            print("  ERROR: No channels in 'NI_Acq'. Skipping.")
            return False

        ch_names = [ch.name for ch in channels]
        total_samples = len(channels[0])
        duration_min = total_samples / sample_rate / 60
        print(f"  Channels: {ch_names}")
        print(f"  Samples:  {total_samples:,} ({duration_min:.1f} min)")

        # Pass 1: normalization
        print("  Pass 1/2: computing normalization...")
        max_val = 0.0
        offset = 0
        while offset < total_samples:
            length = min(CHUNK_SAMPLES, total_samples - offset)
            for ch in channels:
                chunk = ch.read_data(offset=offset, length=length)
                chunk_max = float(np.max(np.abs(chunk)))
                if chunk_max > max_val:
                    max_val = chunk_max
            offset += length

        norm = (TARGET_PEAK / max_val) if max_val > 0 else 1.0
        scale = norm * ADDITIONAL_GAIN
        print(f"  max_val={max_val:.6f}, scale={scale:.4f}")

        # Pass 2: write WAVs
        print("  Pass 2/2: writing WAVs...")
        samp_width = 4 if BIT_DEPTH == 32 else 2
        int_max = (2**31 - 1) if BIT_DEPTH == 32 else (2**15 - 1)
        int_dtype = np.int32 if BIT_DEPTH == 32 else np.int16

        wav_files = {}
        try:
            for ch in channels:
                wav_path = os.path.join(output_dir, f"{ch.name}.wav")
                wf = wave.open(wav_path, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(samp_width)
                wf.setframerate(sample_rate)
                wav_files[ch.name] = wf

            offset = 0
            total_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
            chunks_done = 0
            while offset < total_samples:
                length = min(CHUNK_SAMPLES, total_samples - offset)
                for ch in channels:
                    chunk = ch.read_data(offset=offset, length=length)
                    amp = np.clip(chunk * scale, -1.0, 1.0)
                    amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
                    intdata = (amp * int_max).astype(int_dtype)
                    wav_files[ch.name].writeframes(intdata.tobytes())
                offset += length
                chunks_done += 1
                if chunks_done % 20 == 0 or chunks_done == total_chunks:
                    pct = int(100 * chunks_done / total_chunks)
                    elapsed = time.time() - t0
                    print(f"    {pct}% ({offset/sample_rate/60:.1f} min written, {elapsed:.0f}s elapsed)")
        finally:
            for wf in wav_files.values():
                try:
                    wf.close()
                except Exception:
                    pass

    elapsed = time.time() - t0
    print(f"  DONE in {elapsed:.0f}s — {len(channels)} WAVs written to {output_dir}")
    return True


def find_tests_needing_conversion(source_dir):
    """Find test folders that have TDMS but no ch0.wav."""
    needs_conversion = []
    for entry in sorted(os.listdir(source_dir)):
        test_dir = os.path.join(source_dir, entry)
        if not os.path.isdir(test_dir):
            continue
        tdms_path = os.path.join(test_dir, "NI", "datos.tdms")
        ch0_path = os.path.join(test_dir, "ch0.wav")
        if os.path.isfile(tdms_path) and not os.path.isfile(ch0_path):
            size_mb = os.path.getsize(tdms_path) / (1024 * 1024)
            needs_conversion.append((entry, test_dir, tdms_path, size_mb))
    return needs_conversion


def main():
    parser = argparse.ArgumentParser(description="Batch TDMS→WAV converter")
    parser.add_argument("--source", required=True, help="Base dir with test folders")
    parser.add_argument("--tests", nargs="*", help="Specific test names (e.g. test6 test7)")
    parser.add_argument("--all", action="store_true", help="Convert all tests needing WAV")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted")
    args = parser.parse_args()

    source = args.source
    if not os.path.isdir(source):
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    pending = find_tests_needing_conversion(source)

    if args.tests:
        filter_set = set(f"6mm_{t}" if not t.startswith("6mm_") else t for t in args.tests)
        pending = [(name, d, t, s) for name, d, t, s in pending if name in filter_set]

    if not pending:
        print("No tests need conversion.")
        return

    total_mb = sum(s for _, _, _, s in pending)
    print(f"\nTests to convert: {len(pending)} ({total_mb:.0f} MB total TDMS)")
    print("-" * 60)
    for name, _, _, size_mb in pending:
        print(f"  {name:30s} {size_mb:>8.0f} MB")

    if args.dry_run:
        print("\n[DRY RUN] No conversion performed.")
        return

    print(f"\nStarting conversion...")
    success = 0
    failed = 0
    t_total = time.time()

    for name, test_dir, tdms_path, _ in pending:
        try:
            ok = convert_tdms(tdms_path, test_dir, SAMPLE_RATE)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {name} — {e}")
            failed += 1

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {success} converted, {failed} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
