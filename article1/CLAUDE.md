# Pipeline SVM - Article 1

## Project Overview
Ordinal classification of drill bit wear in CNC drilling via acoustic analysis.
Journal target: RCTA (Revista Colombiana de Tecnologias de Avanzada).
Authors: Meneses, Orejarena, Ayala.

## Key Concepts
- **Frank & Hall ordinal decomposition**: 2 binary classifiers (C1: any wear, C2: severe wear) → 3 ordinal classes
- **26 acoustic features**: RMS, MFCC, spectral entropy, crest factor, harmonic/percussive ratio, etc.
- **Threshold sensitivity**: sweep from 60% to 97% of tool life definition
- **Microphones**: E1-E2 used TXL 235 TOPP AUDIO (dynamic), E3-E7 used PG81 SHURE (condenser)
- **Spectral gating**: noisereduce library, improved adj. accuracy from 84.7% → 90.1%

## Directory Structure
```
D:/pipeline_SVM/
├── article1/          ← YOU ARE HERE. LaTeX article + PDF
├── cv/                ← Modular CV system (3 profiles)
├── scripts/           ← 34 Python scripts (analysis, training, figures)
├── features/          ← Extracted feature CSVs (108 MB)
├── manifests/         ← Train/val/test split definitions
├── results/           ← Trained models (.joblib), figures, reports
│   ├── svm_ordinal_v2/    ← Best ordinal models
│   ├── article1_figures/  ← Publication figures
│   └── comparison_filtered/ ← Original vs denoised comparison
├── augmented/         ← Augmented WAV files (1 GB)
├── noise_profiles/    ← Spectral gating noise profiles
└── previews/          ← Audio preview files
```

## Build Commands

### Compile article
```bash
cd "D:/pipeline_SVM/article1"
CLEAN_PATH="/c/Users/ayala/AppData/Local/Programs/MiKTeX/miktex/bin/x64:/c/Windows/system32:/c/Windows"
PATH="$CLEAN_PATH" pdflatex -interaction=nonstopmode articulo_rcta.tex
```

### Compile CV (any profile)
```bash
cd "D:/pipeline_SVM/cv"
# Edit oscar_ayala_cv.tex line ~11: \ProfileAtrue, \ProfileBtrue, or \ProfileCtrue
CLEAN_PATH="/c/Users/ayala/AppData/Local/Programs/MiKTeX/miktex/bin/x64:/c/Windows/system32:/c/Windows"
PATH="$CLEAN_PATH" pdflatex -interaction=nonstopmode oscar_ayala_cv.tex
```

### IMPORTANT: MiKTeX PATH
Always use CLEAN_PATH to avoid conflict with `C:\Users\ayala\.local\bin\claude.exe\` being treated as a directory by MiKTeX.

## Python Environment
- Python 3.10.0
- Key deps: scikit-learn, librosa, soundfile, shap, noisereduce, pandas, numpy, matplotlib, seaborn, joblib, umap-learn
- No requirements.txt exists (TODO)
- random_state=42 used consistently

## Article Scripts (in ../scripts/)
These generate the article's figures and tables:
- `ablation_features_article.py` — Feature ablation study
- `bootstrap_metrics_article.py` — Confidence intervals
- `cross_experiment_article.py` — Cross-experiment generalization (LOEO)
- `format_figures_article.py` — Publication-ready figures
- `generate_shap_article.py` — SHAP summary plots
- `threshold_sensitivity_article.py` — Threshold sweep analysis

## Key Results
- Best adjacent accuracy: 90.1% (with spectral gating)
- Adjacent accuracy without filtering: 84.7%
- Errors never exceed one ordinal step (98.6% adjacent accuracy at best threshold)
- Top-7 features outperform top-10 (finding from session 8)

## Dataset
- 3,075 audio samples from 7 experiments (E1-E7)
- Retagged with threshold [15/75] (15% fresh, 75% worn)
- Original experiments by Orejarena (2014)

## Conventions
- Spanish for article text, English for code comments
- Hard-coded paths use D:/pipeline_SVM/ (forward slashes in Python)
- All scripts use `#!/usr/bin/env python3`
- Feature CSVs follow manifest format: filepath, label, mic_type

---

# Article 2 Context (CNN/AST — DEADLINE 15 abril 2026)

## Diferencias clave Art.1 vs Art.2
| Aspecto | Art.1 (SVM) | Art.2 (CNN/AST) |
|---------|------------|-----------------|
| Datos | Orejarena (2014), E1-E7 | Oscar Ayala (2025-2026), C: + E: |
| Microfonos | TXL 235 + PG81 SHURE | MAXLIN UDM-51 + Behringer SL84C + Behringer C1 |
| Features | 26 hand-crafted | CNN end-to-end + features |
| Multimodal | Solo audio NI | Audio NI + ESP32 (INMP441 + **caudal YF-S201**) + video microscopio |
| Etiquetado | Vida util progresiva | Audio + flanco VB + caudal (fusion multimodal) |
| Modelo | SVM Frank & Hall | CNN ordinal + AST |

## Dataset Art.2 — Fuente canonica
Leer: C:\Users\ayala\Obsidian\OscarVault\01-Tesis\Art2-Dataset-Inventario.md

### Resumen rapido
- 12 tests multimodal prioritarios (~62.7 GB)
- 12 tests parciales (~15.1 GB)
- 6 perfiles de ruido (~34.8 GB)
- Secuencia clave: E:test15→16→17→18→19→20→21 (desgaste progresivo, 140 agujeros)
- C: tests 6,7,9,14,15,16: WAV convertido (sesion 10)
- E: todos: WAV ya existia
- D: todas las carpetas son DUPLICADOS de C: o E: — NO USAR

## Problemas conocidos en datos Art.2
- Audio con saltos/gaps (especialmente ensayos 8mm)
- Audio contiene: taladrado + ruido maquina reposicionamiento + sopladora taladrina
- Videos: algunos con flanco util, otros oscuridad — clasificar
- E:test13 usa ACEITE (unico test con refrigerante diferente)
- C:test25 tiene 2x Behringer C1 (config diferente a los demas)
- C:test17: NI desconectado en audio 591, TDMS parcial 12GB

## Conversion TDMS→WAV
```bash
python "D:/pipeline_SVM/scripts/batch_tdms_to_wav.py" --source "C:/Users/ayala/Documents Thesis/GUI_v5/app/Datos Generados" --all
python "D:/pipeline_SVM/scripts/batch_tdms_to_wav.py" --source "C:/..." --tests test6 test7 --dry-run
```

## ESP32 — Variable clave del Art.2
El sensor de caudal YF-S201 (flow.csv) es la variable diferenciadora vs Art.1:
- Mide flujo de taladrina (refrigerante) en tiempo real
- Permite inferir capacidad de enfriamiento material+broca por sesion
- Correlacion caudal↔desgaste: fenomeno termico cambia con wear progresivo
- Cada ensayo tiene perfil de caudal unico → variable explicativa fisica
- Audio INMP441 (baja calidad) aporta diversidad de sensor, no es fuente primaria
- Archivos: MCU/flow.csv (nuevo) o audio/flow.csv (legacy I:)

## Ensayos nuevos planeados (miercoles 2 abril 2026)
- Taladrado real con datos completos (NI + ESP32 caudal + video microscopio)
- Probar modelo SVM Art.1 en prediccion real
- Nuevas imagenes flanco de broca
- Registrar caudal por ensayo para correlacion termica
