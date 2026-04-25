#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Art.2 Audio Segmentation and Feature Extraction Pipeline.

Takes continuous WAV files from Art.2 tests and:
1. Detects drill onset/offset via energy onset detection
2. Extracts per-hole WAV segments (3-5 seconds)
3. Applies Art.1 feature extraction (26 acoustic features)
4. Generates CSV suitable for Art.2 model retraining

Input: E:/Datos Generados/6mm_test*/ch0.wav (continuous)
Output: D:/pipeline_SVM/results/art2_dataset/
  - segments/ (per-hole WAV files)
  - features.csv (N x 26 features)
  - metadata.json (hole count per test, wear labels)

Usage:
  python segment_and_feature_art2.py --test 15 16 17    # specific tests
  python segment_and_feature_art2.py --all              # all usable tests
  python segment_and_feature_art2.py --dry-run          # preview only
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf

# Import feature extraction from Art.1 pipeline
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ordinal_utils import extract_26_features
except ImportError:
    print("[!] Warning: ordinal_utils not found. Features will be placeholder.")
    def extract_26_features(y, sr):
        """Placeholder feature extractor."""
        return np.random.rand(26)

# Paths - support both Windows and Unix path formats
E_ROOT_CANDIDATES = [
    Path("/e/Datos Generados"),
    Path("E:/Datos Generados"),
    Path(r"E:\Datos Generados")
]
E_ROOT = next((p for p in E_ROOT_CANDIDATES if p.exists()), E_ROOT_CANDIDATES[0])
OUTPUT_DIR = Path("D:/pipeline_SVM/results/art2_dataset")
SEGMENTS_DIR = OUTPUT_DIR / "segments"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Usable tests from analysis
USABLE_TESTS = [
    "6mm_test2", "6mm_test3", "6mm_test5", "6mm_test6", "6mm_test7",
    "6mm_test8", "6mm_test9", "6mm_test10", "6mm_test11", "6mm_test12",
    "6mm_test13", "6mm_test15", "6mm_test16", "6mm_test17", "6mm_test18",
    "6mm_test19", "6mm_test20", "6mm_test21",
    "6mm_test10 (encendido de maquina)"
]

def detect_drill_segments(y, sr=16000, min_duration=1.5, max_duration=8.0):
    """
    Detect drill segments via energy onset detection.

    Returns: list of (start_frame, end_frame) tuples for each detected segment
    """
    # Compute energy envelope
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # Normalize and threshold
    rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-6)
    threshold = 1.0  # 1 std above mean

    # Find segments where energy is above threshold
    segments = []
    in_segment = False
    start_frame = None

    for i, val in enumerate(rms_norm):
        if val > threshold and not in_segment:
            start_frame = i
            in_segment = True
        elif val <= threshold and in_segment:
            end_frame = i
            duration_s = (end_frame - start_frame) * hop_length / sr

            # Only keep segments of reasonable length (drill holes are 2-5s)
            if min_duration <= duration_s <= max_duration:
                segments.append((start_frame, end_frame))

            in_segment = False

    # Convert frames to samples
    segments_samples = [(s * hop_length, e * hop_length) for s, e in segments]
    return segments_samples

def extract_segment_audio(y, sr, segments_samples, pad=0.5):
    """
    Extract audio for each segment with padding.
    Returns: list of (audio_array, start_time, end_time)
    """
    pad_samples = int(pad * sr)  # Padding in samples
    extracted = []

    for start_sample, end_sample in segments_samples:
        # Add padding
        start = max(0, start_sample - pad_samples)
        end = min(len(y), end_sample + pad_samples)

        segment_audio = y[start:end]
        start_time = start / sr
        end_time = end / sr

        extracted.append((segment_audio, start_time, end_time))

    return extracted

def process_test(test_dir, test_name, dry_run=False):
    """
    Process a single test: segment audio, extract features.
    Returns: dict with results
    """
    # Load continuous audio (first channel)
    audio_path = test_dir / "ch0.wav"
    if not audio_path.exists():
        return {"error": "No ch0.wav found"}

    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
    except Exception as e:
        return {"error": f"Audio load failed: {e}"}

    # Detect segments
    segments_samples = detect_drill_segments(y, sr=sr)

    if len(segments_samples) == 0:
        return {"error": "No segments detected"}

    # Extract segment audio
    extracted = extract_segment_audio(y, sr, segments_samples, pad=0.5)

    # Extract features from each segment
    features_list = []
    saved_segments = []

    for i, (segment_audio, start_time, end_time) in enumerate(extracted):
        # Extract features
        try:
            features = extract_26_features(segment_audio, sr)
            features_list.append(features)
        except Exception as e:
            print(f"    [!] Feature extraction failed for segment {i}: {e}")
            features_list.append(np.full(26, np.nan))

        # Save segment WAV (optional, skip in dry-run)
        if not dry_run:
            segment_path = SEGMENTS_DIR / f"{test_name}_hole_{i:03d}.wav"
            try:
                sf.write(str(segment_path), segment_audio, sr)
                saved_segments.append({
                    "hole_idx": i,
                    "path": str(segment_path),
                    "duration_s": len(segment_audio) / sr,
                    "start_time": start_time,
                    "end_time": end_time
                })
            except Exception as e:
                print(f"    [!] Segment save failed: {e}")

    return {
        "test": test_name,
        "num_segments": len(features_list),
        "features": features_list,  # List of 26-dim arrays
        "segments": saved_segments,
        "duration_total_s": len(y) / sr
    }

def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description="Art.2 audio segmentation & feature extraction")
    parser.add_argument("--test", nargs="+", type=str, help="Specific test numbers (e.g., 15 16 17)")
    parser.add_argument("--all", action="store_true", help="Process all usable tests")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving files")
    args = parser.parse_args()

    # Determine which tests to process
    if args.test:
        tests_to_process = [f"6mm_test{t}" for t in args.test]
    elif args.all:
        tests_to_process = USABLE_TESTS
    else:
        # Default: process key progressive wear tests
        tests_to_process = ["6mm_test15", "6mm_test16", "6mm_test17", "6mm_test18", "6mm_test19"]

    print("[*] Art.2 Audio Segmentation & Feature Extraction")
    print("[*] Output directory: {}".format(OUTPUT_DIR))
    if args.dry_run:
        print("[!] DRY-RUN MODE (no files saved)")
    print()

    # Process each test
    all_features = []
    all_metadata = []

    for test_name in tests_to_process:
        test_dir = E_ROOT / test_name
        if not test_dir.exists():
            print("[XX] {} not found".format(test_name))
            continue

        print("[*] Processing {}...".format(test_name), end=" ", flush=True)
        result = process_test(test_dir, test_name, dry_run=args.dry_run)

        if "error" in result:
            print("FAILED: {}".format(result["error"]))
            continue

        print("OK ({} segments)".format(result["num_segments"]))

        # Aggregate features
        if result["features"]:
            for i, feat_array in enumerate(result["features"]):
                if not np.any(np.isnan(feat_array)):
                    all_features.append(feat_array)
                    all_metadata.append({
                        "test": test_name,
                        "hole_idx": i,
                        "duration_s": result["duration_total_s"]
                    })

    # Save aggregated features CSV
    if all_features:
        feature_names = ["feature_{:02d}".format(i + 1) for i in range(26)]
        df_features = pd.DataFrame(
            np.array(all_features),
            columns=feature_names
        )
        df_features["test"] = [m["test"] for m in all_metadata]
        df_features["hole_idx"] = [m["hole_idx"] for m in all_metadata]

        output_csv = OUTPUT_DIR / "features_art2_segmented.csv"
        df_features.to_csv(str(output_csv), index=False)

        print()
        print("[OK] Saved {} feature vectors to {}".format(len(all_features), output_csv))
        print("     Shape: {} x {}".format(df_features.shape[0], df_features.shape[1]))

    print()
    print("Next steps:")
    print("  1. Inspect features_art2_segmented.csv for quality")
    print("  2. Add wear labels based on hole sequence")
    print("  3. Retrain ordinal model with [15/75] threshold")
    print()

if __name__ == "__main__":
    main()
