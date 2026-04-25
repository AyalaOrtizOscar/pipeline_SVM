# Guía de estudio para la sustentación del informe

Documento de apoyo construido a partir de `informe_proyecto.tex` (141 páginas). Concentra **conceptos poco comunes en ingeniería mecánica** (estadística, ML, DSP, programación) explicados con el lenguaje que un mecánico domina: analogías físicas, parámetros y unidades. Cada bloque termina con una "**respuesta corta para jurado**" — lo que debes decir si te preguntan en 30 segundos.

---

## Parte 1 · Clasificación ordinal y métricas

### 1.1 Qué es una clasificación ordinal
Predecir una etiqueta que **tiene orden** pero no es numérica continua. Acá: `sin_desgaste < medianamente_desgastado < desgastado`. No es regresión (no hay "2.5"), pero tampoco es nominal (no confundas los extremos como si fueran equivalentes).

**Analogía mecánica:** tolerancias H7/H8/H9 — hay orden natural, pero la diferencia entre H7 y H8 no es la misma "distancia" que entre H8 y H9.

> **Respuesta corta:** "Es una clasificación con clases ordenadas, donde equivocarse por un escalón es menos grave que equivocarse por dos."

### 1.2 Descomposición Frank & Hall (ordinal → dos binarios)
Convierte el problema de 3 clases ordenadas en **2 clasificadores binarios**:
- **C1:** ¿hay algún desgaste? → $P(y \geq 1)$
- **C2:** ¿el desgaste es severo? → $P(y \geq 2)$

La clase final se decide por margen máximo, con la restricción de monotonicidad $p_2 \leq p_1$ (no puede ser más probable "desgastado" que "algún desgaste").

> **Respuesta corta:** "En lugar de un clasificador de 3 clases, entreno dos preguntas binarias más fáciles y las combino respetando el orden."

### 1.3 Macro F1 (la métrica estrella del artículo)
- **Precisión** = de lo que predije como clase X, cuánto era realmente clase X.
- **Recall (sensibilidad)** = de lo que era realmente clase X, cuánto logré detectar.
- **F1** = media armónica de precisión y recall. Penaliza fuerte si una de las dos está baja.
- **Macro F1** = promedio simple de F1 por clase. **No pondera por tamaño**, así que una clase chica pesa igual que una grande.

**Por qué macro y no accuracy:** si 80 % de los datos son "sin desgaste", un modelo tonto que siempre diga "sin desgaste" tiene 80 % de accuracy pero es inútil.

> **Respuesta corta:** "Uso macro F1 porque las clases están desbalanceadas y me interesa detectar el desgaste severo, que es la minoritaria."

### 1.4 Adjacent accuracy (tolerancia ordinal)
Porcentaje de predicciones que están **en la clase correcta o en una adyacente**. Tolera errores de un escalón, castiga errores de dos escalones.

En el trabajo: 90.1 % adjacent accuracy tras filtrado espectral. Esto significa que 9 de cada 10 predicciones o aciertan o fallan sólo por un nivel.

> **Respuesta corta:** "Confundir 'medianamente desgastada' con 'desgastada' es menos grave que confundir 'nueva' con 'desgastada'. Adjacent accuracy captura esa asimetría."

### 1.5 Matriz de confusión
Tabla `K×K` con clases reales en filas y predichas en columnas. La diagonal son aciertos; fuera de diagonal son errores. Útil para ver **qué par de clases se confunde**.

En el informe: los errores están concentrados en la frontera 15–25 % y 65–80 % de vida útil, nunca saltan dos clases.

### 1.6 Validación cruzada (CV)
Dividir el dataset en K partes, entrenar con K-1 y validar con la restante; rotar. Variantes:

| Variante | Qué garantiza |
|----------|---------------|
| `KFold` | K particiones aleatorias |
| `StratifiedKFold` | Preserva la proporción de clases en cada fold |
| `GroupKFold` | Nunca mezcla muestras del mismo "grupo" (p.ej. misma broca) |
| `Leave-One-Experiment-Out (LOEO)` | Deja fuera experimentos completos — test más duro |

**Por qué LOEO en este trabajo:** si entreno con audios del ensayo E3 y valido con otros audios del mismo E3, el modelo memoriza la firma del mic y del día. LOEO fuerza que el modelo generalice a sesiones nuevas.

> **Respuesta corta:** "LOEO es la validación más honesta porque simula cómo se comporta el modelo en un ensayo que nunca vio."

### 1.7 Class weight balanced
Truco anti-desbalance: pondera cada clase inversamente a su frecuencia. Clase minoritaria = peso alto → el optimizador le da más importancia.

```python
SVC(class_weight='balanced', ...)
```

### 1.8 Calibración de probabilidades
Un modelo está **calibrado** si cuando dice "65 % de probabilidad", en 65 de cada 100 casos efectivamente la clase ocurre. Métodos: Platt scaling, isotonic regression.

El SHAP y la matriz de confusión dan información complementaria: las probabilidades pueden estar mal calibradas aunque el ranking de clases esté bien.

### 1.9 Bootstrap y intervalos de confianza
Remuestreo con reemplazo: de un dataset de N, se toman N muestras al azar con posibilidad de repetir. Al repetir el proceso 1 000 veces se obtiene una **distribución empírica** del estimador (p.ej. macro F1), de donde se sacan percentiles 2.5 y 97.5 como IC 95 %.

> **Respuesta corta:** "No asumí ninguna distribución teórica para los errores; el bootstrap me dio los intervalos de confianza usando los datos."

---

## Parte 2 · SVM y pipeline de features (Art.1)

### 2.1 Máquina de Vectores de Soporte (SVM)
Encuentra el **hiperplano** que separa dos clases con el **margen máximo**. Los puntos que definen el margen son los *support vectors*; el resto no afecta.

- Frontera lineal para datos linealmente separables.
- **Kernel trick**: proyecta los datos a dimensión superior para que sean separables sin calcular explícitamente esa proyección.

**Analogía mecánica:** es como elegir el plano de corte en CAD/CAM que deja la máxima holgura entre dos geometrías contrapuestas.

### 2.2 Kernel RBF (Radial Basis Function)
$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$. Mide similitud basada en distancia euclídea. `gamma` alta → fronteras muy locales (riesgo de overfit). `gamma` baja → fronteras suaves.

### 2.3 Hiperparámetro C
Controla el **trade-off entre margen y clasificación correcta**. C alto = el modelo prefiere no equivocarse aunque el margen sea estrecho. C bajo = margen más amplio pero tolera errores.

En el trabajo: `C = 10`, `gamma = 'scale'`.

### 2.4 Pipeline de scikit-learn
Encadena pasos de preprocesamiento + modelo como una sola entidad. Garantiza que ningún paso vea el test durante el entrenamiento (evita *data leakage*).

```python
Pipeline([
  ('imputer', SimpleImputer(strategy='median')),   # rellena NaN con la mediana
  ('scaler',  StandardScaler()),                   # media 0, var 1
  ('select',  SelectKBest(mutual_info_classif, k=15)),
  ('svc',     SVC(C=10, kernel='rbf', class_weight='balanced'))
])
```

### 2.5 Imputación
Qué hacer con valores faltantes. Estrategias: media, mediana, moda, KNN, model-based. **Mediana** es robusta a outliers.

### 2.6 StandardScaler
Resta media, divide por desviación. Necesario para SVM porque el kernel RBF depende de distancias; sin escalar, features con mayor rango dominan el cálculo.

### 2.7 SelectKBest + Mutual Information
Selecciona las K features con mayor información mutua respecto a la etiqueta. **Información mutua** = cuánto se reduce la incertidumbre sobre Y si conozco X. Captura dependencias **no lineales** (a diferencia de la correlación de Pearson).

### 2.8 SHAP (Shapley Additive Explanations)
Asigna a cada feature una **contribución marginal** a una predicción específica, basada en teoría de juegos cooperativos (valor de Shapley). Garantiza:
- **Consistencia** (si un feature contribuye más, su SHAP no baja).
- **Aditividad** (las contribuciones suman la predicción).

Uso en el trabajo: identificar qué features explican las predicciones de C1 y C2 por separado.

> **Respuesta corta:** "SHAP dice, para cada predicción, cuánto empujó cada feature hacia arriba o hacia abajo. Es interpretabilidad a nivel de muestra."

---

## Parte 3 · Señal acústica y DSP

### 3.1 Frecuencia de muestreo (sample rate)
Cuántas muestras por segundo se digitalizan. En el trabajo: **44.1 kHz** y **51.2 kS/s** (NI).

**Teorema de Nyquist:** para reconstruir una señal sin aliasing, hay que muestrear a al menos **2×** la máxima frecuencia de interés. A 44.1 kHz cubre hasta ~22 kHz (oído humano).

### 3.2 IEPE (Integrated Electronics Piezo-Electric)
Estándar de acondicionamiento donde el sensor incluye un **pre-amplificador** alimentado por **corriente constante** (~4 mA) enviada por el mismo cable coaxial que lleva la señal. Elimina la necesidad de cables separados de alimentación.

El NI-9234 es un módulo **IEPE nativo** de 4 canales, ±5 V, 24 bits.

### 3.3 Phantom power 48 V
Alimentación DC enviada por el cable XLR (pines 2 y 3 a +48 V respecto a pin 1) para micrófonos de condensador. El C-1 Behringer la necesita; el SL84C dinámico **no** — pero tolera que esté presente.

### 3.4 XLR balanceado vs. BNC
- **XLR balanceado:** tres hilos (señal+, señal-, shield). El receptor resta las dos señales, cancelando ruido en modo común. Estándar en audio profesional.
- **BNC coaxial:** un hilo central + malla. Más común en instrumentación (NI-9234 tiene entradas BNC).
- El informe documenta un **adaptador XLR→BNC** en el rack para interconectar ambos mundos.

### 3.5 Pre-énfasis
Filtro pasa-altos suave ($y_t = x_t - \alpha x_{t-1}$ con $\alpha \approx 0.97$) que amplifica altas frecuencias. Usado en audio/voz para compensar la caída espectral típica.

### 3.6 Spectral gating (noisereduce)
Filtrado no lineal: primero se estima el **perfil de ruido** en una ventana silenciosa; luego se resta ese perfil en el dominio de la frecuencia. Subió adjacent accuracy de 84.7 % → 90.1 % en el Art.1.

> **Respuesta corta:** "Aprendo la huella espectral del ruido de fondo y la atenúo selectivamente en cada banda, dejando pasar la señal de interés."

### 3.7 MFCC (Mel-Frequency Cepstral Coefficients)
1. FFT → espectro.
2. Filtros en escala **Mel** (logarítmica, imita la percepción humana).
3. Log de la energía en cada banda.
4. DCT (transformada coseno discreta) → los primeros coeficientes son los MFCC.

MFCC₀ correlaciona con energía global; MFCC₁ con balance brillante/oscuro. Son la representación estándar en reconocimiento de voz y música.

### 3.8 Descriptores espectrales (26 features)
| Descriptor | Qué mide |
|------------|----------|
| RMS | Energía promedio |
| Peak | Amplitud máxima |
| Crest factor | Peak / RMS (cuán "picudo" es el pico) |
| ZCR | Cruces por cero — proxy de frecuencia dominante |
| Spectral centroid | "Centro de masa" del espectro (brillo) |
| Spectral rolloff | Frecuencia bajo la cual está el 85 % de la energía |
| Spectral bandwidth | Ancho de banda efectivo |
| Spectral flatness | Qué tan parecido a ruido blanco |
| Spectral contrast | Diferencia pico-valle por banda |
| Spectral entropy | Aleatoriedad del espectro |
| Chroma | Perfil de 12 clases de pitch |
| Tonnetz | Relaciones tonales armónicas |
| Harmonic/percussive ratio | Separación entre componente tonal y transitoria |
| Wavelet energy | Energía en la descomposición wavelet |

### 3.9 Wavelets
Descomposición en tiempo-frecuencia con **resolución variable**: alta resolución temporal en altas frecuencias, alta resolución frecuencial en bajas. Mejor que FFT para señales no estacionarias (como los transitorios del corte).

### 3.10 Onset detection
Detecta **inicios** de eventos sonoros (golpes, notas, entradas del filo). En taladrado, un onset por agujero. Se calculan sobre la envolvente espectral.

---

## Parte 4 · CNN y espectrogramas Mel (Art.2)

### 4.1 Espectrograma Mel
Matriz (tiempo × frecuencia-Mel) donde cada celda tiene la energía en dB. En el Art.2: `64 bandas × 512 ventanas × 1 canal`, entrada de la CNN.

### 4.2 Capas convolucionales
Un filtro (kernel) pequeño (3×3, 5×5) se desliza sobre la imagen multiplicando-sumando. Cada filtro aprende a detectar un patrón (bordes, texturas, transitorios). La salida es un **mapa de activación**.

**Analogía mecánica:** es como pasar un palpador por toda la superficie buscando rugosidades de cierta longitud de onda.

### 4.3 Pooling
Reduce la dimensión quedándose con el máximo (MaxPooling) o el promedio (AvgPooling) de cada región. Da invariancia a pequeños desplazamientos.

### 4.4 Softmax + label smoothing
Softmax convierte logits en probabilidades que suman 1. Label smoothing reemplaza el one-hot `[0,0,1]` por `[ε/2, ε/2, 1-ε]` (ε=0.1). Evita overconfidence y mejora calibración.

### 4.5 Fine-tuning
Cargar un modelo preentrenado y **reentrenar sólo las últimas capas** con datos del dominio nuevo, usando learning rate bajo. Mucho más barato que entrenar desde cero.

### 4.6 AST (Audio Spectrogram Transformer)
Transformer aplicado al espectrograma como si fuera una imagen (ViT). Divide el espectrograma en *patches*, los proyecta a embeddings y aplica self-attention. Estado del arte en clasificación de audio desde 2021.

---

## Parte 5 · Ecosistema de programación

### 5.1 Stack Python del proyecto
| Librería | Uso |
|----------|-----|
| `scikit-learn` | SVM, pipeline, CV, métricas |
| `librosa` | Audio → features (MFCC, chroma, espectrograma) |
| `soundfile` | I/O de WAV |
| `nptdms` | Lectura de archivos TDMS de LabVIEW/NI |
| `noisereduce` | Spectral gating |
| `shap` | Explicabilidad |
| `numpy` / `pandas` | Operaciones numéricas y tabulares |
| `matplotlib` / `seaborn` | Figuras |
| `joblib` | Serialización de modelos entrenados (`.joblib`) |
| `tensorflow` / `keras` | CNN ordinal |

### 5.2 Manifest (JSON/CSV)
Archivo plano que acompaña cada captura con **metadatos**: timestamp UTC, SR, mic, ganancia, operador, broca, agujero N°, etc. Es la columna vertebral de la trazabilidad FAIR.

### 5.3 Principios FAIR
- **Findable** (encontrable — DOI, metadatos ricos)
- **Accessible** (abierto o con permiso claro)
- **Interoperable** (formatos estándar: WAV, CSV, JSON)
- **Reusable** (licencia clara, manifest completo)

### 5.4 Random state
Semilla fija (aquí `random_state=42`) que hace que procesos estocásticos (shuffle, splits, inicialización) produzcan resultados **reproducibles**. Sin semilla, cada corrida da un número ligeramente distinto.

### 5.5 Data leakage
Contaminación del test con información del train. Causas típicas:
- Escalar con `fit` sobre todo el dataset antes de partir.
- Seleccionar features usando la etiqueta de test.
- Usar datos del futuro para predecir el pasado.

Se evita con el uso riguroso de `Pipeline` de sklearn y CV correcto.

---

## Parte 6 · Conceptos de manufactura (repaso rápido)

### 6.1 Parámetros de corte
- **Velocidad de corte** $v_c$ [m/min]: velocidad tangencial del filo.
- **RPM** $N$: $N = \dfrac{1000\, v_c}{\pi\, d}$.
- **Avance por revolución** $f_r$ [mm/rev]: profundidad que avanza por vuelta.
- **Avance lineal** $v_f = f_r \cdot N$ [mm/min].
- **Tiempo por agujero** $t_h = L / v_f$.
- **Torque** $T = 60\,000\, P / (2\pi N)$.

### 6.2 Criterio VB (flank wear)
Desgaste del **flanco** de la herramienta medido en mm. Normas:
- **ISO 3685** (torneado)
- **ISO 8688-2** (fresado)
- Umbral operativo: **VB = 0.3 mm** (0.6 mm máximo localizado).

### 6.3 Mecanismos de desgaste
- **Abrasión:** partículas duras de la pieza rayan la herramienta.
- **Adhesión:** soldadura fría pieza-filo (típica a baja velocidad).
- **Difusión:** migración atómica (alta temperatura).
- **Deformación plástica:** el filo cede bajo carga.

Cada uno deja una **firma acústica** distinta porque cambian las frecuencias de la interacción.

### 6.4 Dormer A100 vs. A114
- **A100:** HSS con recubrimiento brillante, punta 118°, serie corta.
- **A114:** HSS sin recubrimiento, geometría similar, más disponible localmente.

Para mantener comparabilidad con Orejarena (2014), ambas usan $v_c = 20$ m/min y $f_r$ del selector.

---

## Parte 7 · Números clave del informe (memorízalos)

| Dato | Valor |
|------|-------|
| Muestras totales Art.1 | 3 075 audios (Orejarena 2014) |
| Experimentos Art.1 | 7 (E1–E7) |
| Muestras Lote II (Art.2) | 1 446 |
| Experimentos Lote II | 7 brocas (E8–E14) |
| Agujeros Lote II | 575 |
| Dataset unificado | 4 381 muestras |
| Features SVM | 26 → top 15 por MI |
| Hiperparámetros | C=10, RBF, gamma=scale |
| Adjacent accuracy baseline | 84.7 % |
| Adjacent accuracy con filtrado | **90.1 %** |
| Sample rate NI | 44.1 kHz (24 bit IEPE) |
| Canales | 3 (MAXLIN UDM-51, SL84C, C1) |
| ESP32 sample rate | 16 kHz |
| Caudalímetro | YF-S201 (pulsos binarios) |
| Umbral etiquetado | [15 / 75] % de vida útil |
| VB de referencia | 0.3 mm (ISO 3685 / 8688-2) |

---

## Parte 8 · Preguntas probables del jurado (y cómo responderlas)

**P: ¿Por qué SVM y no una red neuronal para el Art.1?**
R: Con 3 000 muestras y 26 features hand-crafted, la SVM con kernel RBF tiene mejor sesgo-varianza que una CNN. La CNN se justifica en el Art.2 porque hay más datos, multimodalidad y representación tiempo-frecuencia que explotar.

**P: ¿Qué pasa si el sensor cambia?**
R: El modelo es sensible al tipo de micrófono (dinámico vs. condensador) — por eso en el Art.2 se incluye `mic_type` como one-hot feature. La calibración por sesión (tono de referencia) mitiga el drift instrumental.

**P: ¿Cómo sabes que no hay overfitting?**
R: Uso LOEO (leave-one-experiment-out) como validación, StratifiedGroupKFold en CV interna, y reporto intervalos bootstrap. Además, la métrica se mantiene estable al variar el umbral de etiquetado entre 60 y 97 % en el análisis de sensibilidad.

**P: ¿Por qué [15/75] como umbral?**
R: Coincide con el criterio físico $VB = 0.3$ mm (ISO 3685), validado con las imágenes de microscopio. Por debajo de 15 % la broca aún no muestra huella medible; por encima de 75 % ya hay cráter y mayor riesgo de fractura.

**P: ¿Qué aporta el ESP32 si el NI ya captura audio de mejor calidad?**
R: El INMP441 no es la fuente primaria de audio; el aporte real es el **caudalímetro YF-S201** — variable termo-hidráulica que correlaciona con desgaste progresivo. El ESP32 diversifica la redundancia y permite un despliegue industrial económico.

**P: ¿Por qué 44.1 kHz y no 96 kHz?**
R: Los transitorios de corte relevantes están por debajo de 20 kHz. 44.1 kHz cumple Nyquist con margen, es el estándar de audio profesional y el hardware IEPE del NI-9234 ya opera ahí.

**P: ¿Qué limitaciones tiene el trabajo?**
R: (1) Un solo material (SAE 4140), (2) un solo centro de mecanizado (Leadwell V-20), (3) etiquetas ordinales de 3 niveles — se podrían afinar a más clases con más datos, (4) no se midió vibración mecánica directa (sólo acústica + caudal).

**P: ¿Qué sigue?**
R: Publicar el dataset con DOI, ampliar a más materiales, aplicar AST (transformer) al Art.2, integrar vibración con acelerómetro triaxial, y construir un sistema de alerta en tiempo real sobre la GUI v5.

---

## Atajos de pronunciación y términos en inglés

| Término | Pronunciación orientativa | Significado |
|---------|---------------------------|-------------|
| Macro F1 | /ˈmakro-efe-uno/ | Media de F1 por clase |
| Kernel | /ˈkernel/ | Función de similitud |
| Overfitting | /overˈfitin/ | Memorización |
| Underfitting | /underˈfitin/ | Subajuste |
| Bootstrap | /butˈstrap/ | Remuestreo |
| Pipeline | /ˈpaiplain/ | Cadena de procesamiento |
| Feature | /ˈfitʃer/ | Característica |
| Cross-validation | /kros validˈeishon/ | Validación cruzada |
| Phantom power | /ˈfantom ˈpauer/ | Alimentación fantasma |
| Spectral centroid | /ˈspektral ˈsentroid/ | Centroide espectral |
| Label smoothing | /ˈleibel ˈsmudin/ | Suavizado de etiquetas |
| Fine-tuning | /fain ˈtiunin/ | Reajuste fino |

---

## Cierre

Si solo tienes tiempo para memorizar **3 cosas**, que sean:

1. **Frank & Hall + 90.1 % adjacent accuracy** — el corazón del Art.1.
2. **LOEO + StratifiedGroupKFold** — por qué la evaluación es honesta.
3. **Caudal YF-S201** — la variable diferenciadora del Art.2.

Con estas tres el jurado ve que dominaste el método, la validación y la innovación.
