#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Art.2 Dataset Analysis and Segmentation Strategy.

Problem: WAVs are continuous (~872s), but no per-hole timestamps in events.log.
Solution: Use onset detection + predictions CSV as temporal anchors.

Generates:
  1. Dataset inventory (which tests have complete data)
  2. Audio segmentation attempt (drill onsets)
  3. Usability assessment for retraining
  4. Recommendations for data pipeline
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
E_ROOT = Path("E:/Datos Generados")
C_ROOT = Path("C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados")
RESULTS_DIR = Path("D:/pipeline_SVM/results/art2_analysis")
RESULTS_DIR.mkdir(exist_ok=True)

def load_test_metadata(test_root):
    """Load wizard.json and manifest.csv from test."""
    wizard_path = test_root / "wizard.json"
    manifest_path = test_root / "manifest.csv"
    events_path = test_root / "events.log"

    metadata = {
        "test_root": str(test_root),
        "wizard": None,
        "manifest": None,
        "events": None,
        "audio_files": []
    }

    # Load wizard.json
    if wizard_path.exists():
        with open(wizard_path, 'r') as f:
            metadata["wizard"] = json.load(f)

    # Load manifest.csv
    if manifest_path.exists():
        metadata["manifest"] = pd.read_csv(manifest_path)

    # Load events.log
    if events_path.exists():
        events = []
        with open(events_path, 'r') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    pass
        metadata["events"] = events

    # List audio files
    for audio in test_root.glob("ch*.wav"):
        metadata["audio_files"].append(str(audio))

    return metadata

def analyze_continuous_audio(wav_path, sr=None):
    """
    Analyze continuous audio for drill onsets.
    Returns: dict with duration, RMS envelope, detected onsets, etc.
    """
    try:
        y, sr = librosa.load(wav_path, sr=sr)
    except Exception as e:
        return {"error": str(e)}

    duration_s = len(y) / sr

    # Compute RMS energy envelope using lower resolution to avoid memory issues
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, n_fft=1024)
        log_S = librosa.power_to_db(S, ref=np.max)
        rms = np.sqrt(np.mean(log_S**2, axis=0))
    except (MemoryError, np.core._exceptions.MemoryError):
        # Fallback: use simpler envelope computation
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Detect onset times (drill starts when energy jumps up)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Filter onsets: keep only those separated by >0.5s (avoid noise spikes)
    filtered_onsets = []
    if len(onset_times) > 0:
        filtered_onsets.append(onset_times[0])
        for t in onset_times[1:]:
            if t - filtered_onsets[-1] > 0.5:
                filtered_onsets.append(t)

    return {
        "duration_s": duration_s,
        "sr": sr,
        "num_frames": len(rms),
        "rms_env": rms.tolist()[:100],  # First 100 for preview
        "all_onsets": onset_times.tolist(),
        "filtered_onsets": filtered_onsets,
        "num_filtered_onsets": len(filtered_onsets)
    }

def load_predictions_csv(test_root):
    """Load predictions CSV and extract timestamps as temporal markers."""
    pred_files = list(test_root.glob("predictions_*.csv"))
    if not pred_files:
        return None

    try:
        df = pd.read_csv(pred_files[0])
        return {
            "file": str(pred_files[0]),
            "rows": len(df),
            "columns": df.columns.tolist(),
            "sample": df.head(3).to_dict()
        }
    except Exception as e:
        return {"error": str(e)}

def assess_test_usability(metadata, audio_analysis, predictions):
    """
    Assess whether test data is usable for Art.2 retraining.
    Returns: usability score (0-100) and recommendation.
    """
    score = 0
    issues = []

    # Check completeness
    if metadata.get("wizard"):
        score += 20
    else:
        issues.append("Missing wizard.json")

    if metadata.get("audio_files"):
        score += 20
    else:
        issues.append("No audio files")

    if predictions:
        score += 15
    else:
        issues.append("No predictions.csv")

    # Check audio quality
    if audio_analysis and "error" not in audio_analysis:
        score += 20
        if audio_analysis.get("num_filtered_onsets", 0) >= 3:
            score += 15  # Good segmentability
        else:
            issues.append(f"Only {audio_analysis.get('num_filtered_onsets', 0)} onsets detected")
    else:
        issues.append("Audio analysis failed")

    # Recommendation
    if score >= 70:
        recommendation = "USABLE -- good candidate for dataset"
    elif score >= 50:
        recommendation = "PARTIAL -- may need manual review"
    else:
        recommendation = "POOR -- skip for now"

    return {
        "score": score,
        "issues": issues,
        "recommendation": recommendation
    }

def main():
    """Main analysis."""
    print("[*] Art.2 Dataset Analysis -- Continuous Audio Segmentation")
    print()

    # Find all E: tests (6mm, complete WAV)
    tests = []
    if E_ROOT.exists():
        for test_dir in sorted(E_ROOT.glob("6mm_test*")):
            if test_dir.is_dir():
                tests.append(("E", test_dir))

    # Find C: tests with WAV
    if C_ROOT.exists():
        for test_dir in sorted(C_ROOT.glob("test*")):
            if test_dir.is_dir() and any(test_dir.glob("ch*.wav")):
                tests.append(("C", test_dir))

    print("[OK] Found {} tests with audio files".format(len(tests)))
    print()

    # Analyze each test
    results = []
    for source, test_path in tests:
        test_name = test_path.name
        print("[*] Analyzing {}:{}...".format(source, test_name), end=" ", flush=True)

        # Load metadata
        metadata = load_test_metadata(test_path)

        # Analyze first audio channel
        audio_analysis = None
        if metadata["audio_files"]:
            audio_analysis = analyze_continuous_audio(metadata["audio_files"][0], sr=16000)

        # Load predictions
        predictions = load_predictions_csv(test_path)

        # Assess usability
        usability = assess_test_usability(metadata, audio_analysis, predictions)

        result = {
            "source": source,
            "test": test_name,
            "metadata": metadata,
            "audio_analysis": audio_analysis,
            "predictions": predictions,
            "usability": usability
        }
        results.append(result)

        print("Score: {}/100 -- {}".format(usability['score'], usability['recommendation'][:20]))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Aggregate results
    usable_tests = [r for r in results if r["usability"]["score"] >= 70]
    partial_tests = [r for r in results if 50 <= r["usability"]["score"] < 70]
    poor_tests = [r for r in results if r["usability"]["score"] < 50]

    print("\nUsable ({}):".format(len(usable_tests)))
    for r in usable_tests:
        onsets = r.get("audio_analysis", {}).get("num_filtered_onsets", "?")
        print("  OK {}:{} -- {} onsets".format(r['source'], r['test'], onsets))

    if partial_tests:
        print("\nPartial ({}):".format(len(partial_tests)))
        for r in partial_tests:
            issues = ", ".join(r["usability"]["issues"][:2])
            print("  ~ {}:{} -- {}".format(r['source'], r['test'], issues))

    if poor_tests:
        print("\nPoor ({}):".format(len(poor_tests)))
        for r in poor_tests:
            issues = ", ".join(r["usability"]["issues"][:1])
            print("  XX {}:{} -- {}".format(r['source'], r['test'], issues))

    # Save results JSON
    output_file = RESULTS_DIR / "art2_inventory.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        for r in results:
            if r.get("audio_analysis") and "rms_env" in r["audio_analysis"]:
                r["audio_analysis"]["rms_env"] = "[RMS envelope preview]"
        json.dump(results, f, indent=2, default=str)

    print("\n[OK] Results saved to {}".format(output_file))

    # Recommendations
    print()
    print("=" * 80)
    print("RECOMMENDATIONS FOR RETRAINING")
    print("=" * 80)
    print()
    print("1. SEGMENTATION APPROACH:")
    print("   - Use onset detection (librosa) to segment continuous audio")
    print("   - Cross-validate against predictions_*.csv timestamps")
    print("   - Fallback: manual annotation for edge cases")
    print()
    print("2. LABELING STRATEGY:")
    print("   - Assign cumulative wear label based on hole sequence within test")
    print("   - 6mm tests range 15-140 holes (progressive wear)")
    print("   - Use duty_cycle from flow.csv as auxiliary feature")
    print()
    print("3. DATASET COMPOSITION:")
    print("   - Core: {} fully usable tests".format(len(usable_tests)))
    print("   - Supplement: {} partial tests with cleanup".format(len(partial_tests)))
    print("   - Skip: {} poor tests".format(len(poor_tests)))
    print()
    print("4. PIPELINE:")
    print("   - Extract per-hole WAV segments (3-5 seconds each)")
    print("   - Re-extract 26 acoustic features from Art.1 pipeline")
    print("   - Train new ordinal model with [15/75] threshold")
    print("   - Compare CNN/AST baseline against SVM on same data")
    print()

if __name__ == "__main__":
    main()
