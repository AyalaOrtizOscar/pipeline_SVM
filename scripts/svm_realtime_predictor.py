#!/usr/bin/env python3
"""
SVM Real-time Predictor para laboratorio Art.2
Carga modelo Frank & Hall entrenado y predice clase de desgaste en tiempo real
desde segmentos de audio WAV o características pre-calculadas.

Uso:
  python svm_realtime_predictor.py --audio agujero_001.wav --features lista.csv
  python svm_realtime_predictor.py --batch --csv resultados.csv
"""

import joblib
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# Cargar modelos entrenados
# ─────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("D:/pipeline_SVM/results/svm_ordinal_v2")

def load_models():
    """Carga los dos clasificadores binarios y el scaler."""
    c1_model = joblib.load(MODEL_DIR / "svm_C1_top15_orig.joblib")
    c2_model = joblib.load(MODEL_DIR / "svm_C2_top15_orig.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler_top15_orig.joblib")

    with open(MODEL_DIR / "metrics_top15_orig.json") as f:
        metrics = json.load(f)

    return c1_model, c2_model, scaler, metrics

# ─────────────────────────────────────────────────────────────────────
# Frank & Hall Ordinal Decoding
# ─────────────────────────────────────────────────────────────────────

def frank_hall_decode(c1_pred, c2_pred):
    """
    Decodifica predicciones binarias Frank & Hall a clase ordinal.

    C1: any wear (0 = sin desgaste, 1 = con desgaste)
    C2: severe wear (0 = desgaste leve, 1 = desgaste severo)

    Clases ordinales:
      0 = sin desgaste       (C1=0, C2=0)
      1 = medianamente gastado (C1=1, C2=0)
      2 = severamente gastado   (C1=1, C2=1)
    """
    if c1_pred == 0:
        return 0, "sin_desgaste"
    elif c1_pred == 1 and c2_pred == 0:
        return 1, "desgaste_leve"
    else:  # c1_pred == 1 and c2_pred == 1
        return 2, "desgaste_severo"

def predict_single_sample(features, c1_model, c2_model, scaler):
    """
    Predice clase ordinal para un vector de 15 features.

    Args:
        features: array (15,) con features normalizadas
        c1_model, c2_model: clasificadores entrenados
        scaler: StandardScaler para normalizar features

    Returns:
        clase_ord: int (0, 1, 2)
        clase_nombre: str
        confianza: float (media de probabilidades)
    """
    # Normalizar
    features_scaled = scaler.transform([features])

    # Predicciones binarias
    c1_pred = c1_model.predict(features_scaled)[0]
    c2_pred = c2_model.predict(features_scaled)[0]

    # Probabilidades
    c1_proba = c1_model.predict_proba(features_scaled)[0]
    c2_proba = c2_model.predict_proba(features_scaled)[0]

    # Decodificar a ordinal
    clase_ord, clase_nombre = frank_hall_decode(c1_pred, c2_pred)

    # Confianza = promedio de máx probabilidades
    confianza = (max(c1_proba) + max(c2_proba)) / 2.0

    return clase_ord, clase_nombre, confianza

# ─────────────────────────────────────────────────────────────────────
# CLI para laboratorio
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SVM Real-time Predictor — Validación en vivo en el laboratorio"
    )
    parser.add_argument(
        "--csv",
        help="CSV con columnas: feature_01,feature_02,...,feature_15,n_agujero (opcional)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo: pide 15 features por consola"
    )
    parser.add_argument(
        "--output",
        help="Archivo de salida para registros (opcional)"
    )
    args = parser.parse_args()

    # Cargar modelos
    print("[*] Cargando modelos...")
    c1_model, c2_model, scaler, metrics = load_models()
    print(f"[✓] Modelos cargados. Acc. test: {metrics.get('adj_acc', 'N/A')}")
    print()

    output_file = None
    if args.output:
        output_file = open(args.output, "a")
        output_file.write(f"\n# Sesión iniciada {datetime.now().isoformat()}\n")
        output_file.write("n_agujero,clase_ord,clase_nombre,confianza,timestamp\n")

    if args.interactive:
        print("Modo INTERACTIVO — Ingresa 15 features separadas por comas")
        print("(O escribe 'salir' para terminar)\n")

        n_agujero = 1
        while True:
            try:
                inp = input(f"Agujero #{n_agujero} [15 features]: ").strip()
                if inp.lower() == "salir":
                    break

                features = np.array([float(x.strip()) for x in inp.split(",")])
                if len(features) != 15:
                    print(f"  ✗ Error: ingresaste {len(features)} valores, se esperaban 15")
                    continue

                clase_ord, clase_nombre, confianza = predict_single_sample(
                    features, c1_model, c2_model, scaler
                )

                print(f"  ✓ Clase: {clase_ord} ({clase_nombre}), Confianza: {confianza:.2%}")

                if output_file:
                    ts = datetime.now().isoformat()
                    output_file.write(f"{n_agujero},{clase_ord},{clase_nombre},{confianza:.3f},{ts}\n")
                    output_file.flush()

                n_agujero += 1

            except ValueError:
                print("  ✗ Error: ingreso inválido. Usa números separados por comas.")
            except KeyboardInterrupt:
                break

    elif args.csv:
        print(f"[*] Leyendo CSV: {args.csv}")
        try:
            df = pd.read_csv(args.csv)

            # Columnas esperadas: feature_01 a feature_15
            feature_cols = [f"feature_{i:02d}" for i in range(1, 16)]
            missing = set(feature_cols) - set(df.columns)
            if missing:
                print(f"[✗] Columnas faltantes: {missing}")
                sys.exit(1)

            results = []
            for idx, row in df.iterrows():
                features = row[feature_cols].values
                clase_ord, clase_nombre, confianza = predict_single_sample(
                    features, c1_model, c2_model, scaler
                )

                n_agujero = row.get("n_agujero", idx + 1)
                print(
                    f"  Agujero {n_agujero}: Clase {clase_ord} ({clase_nombre}), "
                    f"Confianza {confianza:.2%}"
                )

                results.append({
                    "n_agujero": n_agujero,
                    "clase_ord": clase_ord,
                    "clase_nombre": clase_nombre,
                    "confianza": confianza
                })

            if output_file:
                for r in results:
                    ts = datetime.now().isoformat()
                    output_file.write(
                        f"{r['n_agujero']},{r['clase_ord']},"
                        f"{r['clase_nombre']},{r['confianza']:.3f},{ts}\n"
                    )

        except Exception as e:
            print(f"[✗] Error al procesar CSV: {e}")
            sys.exit(1)

    else:
        parser.print_help()

    if output_file:
        output_file.close()
        print(f"\n[✓] Resultados guardados en {args.output}")

if __name__ == "__main__":
    main()
