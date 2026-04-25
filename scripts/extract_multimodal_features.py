#!/usr/bin/env python3
"""
extract_multimodal_features.py

Amplía los CSV de features existentes con:
 - 6 features agregadas del audio ESP32 (INMP441, I2S):
     esp_rms, esp_rms_db, esp_centroid_mean, esp_zcr,
     esp_spectral_contrast_mean, esp_crest_factor
 - 5 features del caudal (flow.csv del ESP32) para la ventana temporal del hole:
     flow_mean_lmin, flow_std_lmin, flow_min_lmin,
     flow_duty_pulses, flow_cv (coeficiente de variación)
 - 1 feature categórica de recubrimiento:
     coating_coded  (1 si A100 con TiN, 0 si A114 sin recubrimiento, NaN si desconocido)

Todas las filas de Orejarena quedan con NaN en las columnas multimodales
(se imputan con mediana durante el entrenamiento).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

ART2_DIR = Path('D:/pipeline_SVM/art2_segments')
FEAT_DIR = Path('D:/pipeline_SVM/features/art2')
DATA_ROOT = Path('C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados')
OREJARENA_CSV = Path('D:/pipeline_SVM/features/features_curated_splits.csv')
OUT_MERGED = Path('D:/pipeline_SVM/features/features_multimodal_merged.csv')

ESP32_COLS = ['esp_rms', 'esp_rms_db', 'esp_centroid_mean', 'esp_zcr',
              'esp_spectral_contrast_mean', 'esp_crest_factor']
FLOW_COLS = ['flow_mean_lmin', 'flow_std_lmin', 'flow_min_lmin',
             'flow_duty_pulses', 'flow_cv']
COATING_COL = 'coating_coded'
ALL_MM = ESP32_COLS + FLOW_COLS + [COATING_COL]

COATING_MAP = {
    # A100 tienen recubrimiento de nitruro de titanio (TiN)
    'DORMER_A100#5': 1,
    '6mm#1': 1,  # broca heredada, asumimos TiN por defecto
    '6mm#2': 1,
    '6mm#3': 1,
    '6mm#4': 1,
    '6mm#5': 1,
    # A114 sin recubrimiento
    'DORMER_A114#1': 0,
    'DORMER_A114#2': 0,
    'DORMER_A114#3': 0,
}

SR_ESP = 16000  # INMP441 via I2S


def extract_esp_features(wav_path):
    try:
        y, sr = librosa.load(str(wav_path), sr=SR_ESP, mono=True)
        if len(y) < SR_ESP // 4:  # <0.25 s
            return {c: np.nan for c in ESP32_COLS}
        rms = float(np.sqrt(np.mean(y ** 2)))
        rms_db = 20 * np.log10(rms + 1e-12)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
        peak = float(np.max(np.abs(y)))
        crest = peak / (rms + 1e-12)
        return {
            'esp_rms': rms,
            'esp_rms_db': rms_db,
            'esp_centroid_mean': float(centroid),
            'esp_zcr': float(zcr),
            'esp_spectral_contrast_mean': float(spec_contrast),
            'esp_crest_factor': float(crest),
        }
    except Exception as e:
        print(f'  [warn] esp feature fail {wav_path.name}: {e}')
        return {c: np.nan for c in ESP32_COLS}


def build_flow_index(flow_csv):
    if not flow_csv.exists():
        return None
    try:
        df = pd.read_csv(flow_csv)
        if 'elapsed_s' not in df.columns or 'flow_lmin' not in df.columns:
            return None
        return df
    except Exception:
        return None


def flow_features_for_hole(flow_df, start_s, end_s):
    if flow_df is None:
        return {c: np.nan for c in FLOW_COLS}
    win = flow_df[(flow_df['elapsed_s'] >= start_s) &
                  (flow_df['elapsed_s'] <= end_s)]
    if len(win) == 0:
        return {c: np.nan for c in FLOW_COLS}
    flow = win['flow_lmin'].values.astype(float)
    pulses = win['pulses_s'].values.astype(float) if 'pulses_s' in win.columns else np.zeros_like(flow)
    mean = float(np.mean(flow))
    std = float(np.std(flow))
    mn = float(np.min(flow))
    duty = float(np.mean(pulses > 0))
    cv = std / (mean + 1e-6)
    return {
        'flow_mean_lmin': mean,
        'flow_std_lmin': std,
        'flow_min_lmin': mn,
        'flow_duty_pulses': duty,
        'flow_cv': cv,
    }


def process_test(test_id, art2_prefix='C', drill_bit=None):
    """Annade columnas multimodal a un CSV existente de features."""
    feat_csv = FEAT_DIR / f'test{test_id}_features.csv'
    if not feat_csv.exists():
        print(f'[skip] test{test_id}: CSV faltante')
        return None
    df = pd.read_csv(feat_csv)
    if df.empty:
        return None
    test_folder = DATA_ROOT / f'6mm_test{test_id}'
    flow_df = build_flow_index(test_folder / 'MCU' / 'flow.csv')
    prefix = art2_prefix
    seg_dir = ART2_DIR / f'{prefix}_test{test_id}' / 'holes'

    # cargamos los meta.json para saber start_s/end_s
    meta_map = {}
    for mj in seg_dir.glob('hole_*_meta.json'):
        try:
            meta = json.loads(mj.read_text())
            meta_map[meta['hole_id']] = meta
        except Exception:
            pass

    rows_new = []
    for _, row in df.iterrows():
        hole_id = row.get('hole_id')
        meta = meta_map.get(hole_id)
        # ESP32 audio
        if meta and meta.get('files', {}).get('esp32'):
            # Los paths en meta usan backslash y prefijo heredado posiblemente
            esp_path = seg_dir / f'{hole_id}_esp32.wav'
            esp_feats = extract_esp_features(esp_path) if esp_path.exists() else {c: np.nan for c in ESP32_COLS}
        else:
            esp_feats = {c: np.nan for c in ESP32_COLS}
        # Flow
        if meta:
            flow_feats = flow_features_for_hole(flow_df, meta['start_s'], meta['end_s'])
        else:
            flow_feats = {c: np.nan for c in FLOW_COLS}
        rows_new.append({**esp_feats, **flow_feats})

    df_mm = pd.DataFrame(rows_new, index=df.index)
    df = pd.concat([df, df_mm], axis=1)
    # coating
    bit = df['drill_bit'].iloc[0] if 'drill_bit' in df.columns else drill_bit
    df[COATING_COL] = COATING_MAP.get(bit, np.nan)
    # Guardamos CSV expandido
    out = FEAT_DIR / f'test{test_id}_features_mm.csv'
    df.to_csv(out, index=False)
    n_esp = df['esp_rms'].notna().sum()
    n_flow = df['flow_mean_lmin'].notna().sum()
    print(f'test{test_id} [{bit}]: n={len(df)}  esp_ok={n_esp}  flow_ok={n_flow}  coating={df[COATING_COL].iloc[0]}')
    return df


def merge_all():
    # Orejarena base: annade columnas NaN para todas las multimodales
    oj = pd.read_csv(OREJARENA_CSV)
    for c in ALL_MM:
        if c not in oj.columns:
            oj[c] = np.nan
    # Aseguramos columnas Art.2 que no existen en Orejarena
    for c in ['test_id', 'hole_id', 'hole_number', 'channel',
              'drill_bit', 'total_holes_bit', 'drive']:
        if c not in oj.columns:
            oj[c] = np.nan
    oj['source'] = 'orejarena'

    art2_frames = []
    registry_path = FEAT_DIR / 'drill_bit_registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'tests': {}}
    for tid, info in registry['tests'].items():
        prefix = info.get('drive', 'C')
        bit = info.get('drill_bit')
        mm_csv = FEAT_DIR / f'test{tid}_features_mm.csv'
        if mm_csv.exists():
            df = pd.read_csv(mm_csv)
            df['source'] = f'art2_test{tid}'
            art2_frames.append(df)
    art2 = pd.concat(art2_frames, ignore_index=True) if art2_frames else pd.DataFrame()

    # Alinea columnas
    all_cols = set(oj.columns) | set(art2.columns)
    for c in all_cols:
        if c not in oj.columns:
            oj[c] = np.nan
        if c not in art2.columns:
            art2[c] = np.nan
    merged = pd.concat([oj[list(all_cols)], art2[list(all_cols)]], ignore_index=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f'\nMerged CSV: {OUT_MERGED}')
    print(f'  Rows: {len(merged)}  (orejarena={len(oj)}  art2={len(art2)})')
    print('  Label distribution:')
    print(merged['label'].value_counts().to_string())
    # cobertura multimodal
    for c in ALL_MM:
        pct = 100 * merged[c].notna().sum() / len(merged)
        print(f'  {c}: {pct:.1f}% non-null')
    return merged


def main():
    registry = json.loads((FEAT_DIR / 'drill_bit_registry.json').read_text())
    for tid, info in registry['tests'].items():
        process_test(int(tid), art2_prefix=info['drive'],
                     drill_bit=info['drill_bit'])
    merge_all()


if __name__ == '__main__':
    main()
