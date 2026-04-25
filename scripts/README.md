# pipeline_SVM — Scripts Definitivos

## Flujo de trabajo

```
1. prepare_svm_workspace.py       → Crea estructura de carpetas y verifica insumos
2. extract_baseline_features.py   → Extrae 37 features acústicas por WAV → features/merged_features_raw.csv
3. make_svm_manifest.py           → Genera manifest SVM con rutas y etiquetas
4. prepare_features_for_svm.py    → Limpia, imputa y normaliza features para entrenamiento
5. train_svm_final.py             → Entrena SVM con StratifiedGroupKFold + GridSearch
6. train_rf_baseline_fixed.py     → Entrena RandomForest como baseline comparativo
```

## Análisis e interpretabilidad

```
feature_analysis_drilling_v5.py   → ANOVA, Kruskal-Wallis, Cohen's d por feature
kruskal_cohen_mutual.py           → Tests estadísticos + mutual information
pca_and_pairplot_baseline.py      → Visualización PCA del espacio de features
umap_inspect.py                   → Proyección UMAP para inspección visual
shap_summary.py                   → SHAP values para interpretabilidad del modelo
permutation_importance.py         → Importancia por permutación
separability_stats.py             → Métricas de separabilidad entre clases
```

## Evaluación y calibración

```
evaluate_model_and_save_report.py         → Reporte de clasificación + confusion matrix
per_mic_confusion.py                       → Breakdown de errores por tipo de micrófono
calibrate_prefit_and_threshold_fixed.py   → Calibración de probabilidades post-entrenamiento
generate_calibration_curve.py             → Curva de calibración visual
```

## _archive/
Contiene 78 scripts de iteraciones anteriores. No eliminar.
