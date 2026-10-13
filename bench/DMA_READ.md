# Techo de DMA de lectura pura (Fase 2, 13 oct 2026)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**.

## 0. Pregunta y criterio de decisión, fijados ANTES de medir (commit anterior a cualquier resultado)

**Pregunta.** ¿El decode (`src/kernels/matvec_i8.py` + `mv_i8.cc`, 4 columnas, pesos int8) está limitado por el DMA de lectura o por el cálculo? La Fase 0 midió que el `wait` del decode rinde 17.1 y 21.1 GB/s, en el techo de una copia memcpy (17-22 GB/s por dirección), pero esa copia lee y escribe a la vez y el decode solo lee [medido, `PERF.md` §2-§3]. El encargo decía "el kernel no está limitado por DMA" (suelo de MAC ~250 µs a 1.5x). Una de las dos cosas es falsa y esta medición lo decide.

**Diseños comparados (misma topología de DMA, misma transformación del DMA, mismos tamaños de tile):**
- `real`: el kernel de decode actual (`matvec_i8.py` con `mv_i8.cc`).
- `ro` (lectura pura): `bench/dma_read_only.py`, copia de `matvec_i8.py` en la que el núcleo solo adquiere y libera cada tile de A (y B) sin calcular. Mide el tiempo de mover los pesos por shim -> memtile -> núcleo con la transformación "quad-transpuesta", sin MAC. Variante `ro_plain`: sin la transformación, para aislar su coste.

**Barrido:** columnas (n_cores) 1, 2 y 4; K = 2048; M = 2048, 8192 y 32768 (A = 4, 16 y 64 MB de int8). Para `real` se miden M = 2048 y 8192 en 1, 2 y 4 columnas.

**Repeticiones:** n >= 3 procesos independientes por configuración (cada uno registra el xclbin de cero), 30 iteraciones cronometradas tras 5 de calentamiento; por proceso se toma la mediana, y se reporta media ± desviación de las 3 medianas. Se descartan sin sustituir solo las que fallen la verificación de datos.

**Métricas:** BW_ro = bytes de A / T_ro (GB/s, 10^9); r = T_ro / T_real a igual configuración; escala con columnas (BW por columna).

**Contención anotada (en cada proceso, antes y después):** `/proc/loadavg`, gobernador, `powerprofilesctl`, los 5 procesos con más CPU, y estado de la GPU (reposo, P-state). Se reporta el coeficiente de variación entre procesos; si supera el 10 % la configuración se marca como ruidosa y se repite en otro momento. Se repite además una configuración (4 columnas, 16 MB) con carga de CPU sintética (varios procesos ocupando núcleos) para ver cuánto mueve la contención de memoria.

**Criterio de decisión (con r a 4 columnas y A >= 16 MB, media de los 3 procesos, y la misma tendencia en 2 columnas):**
- **r >= 0.85** -> el decode está limitado por el DMA de lectura. Mejorar el cálculo (int8×int8, `mv_i8_i8`) puede dar como máximo 1/r <= 1.18x en un token y **no se justifica para decode**. La fase se reduce a costes fijos por llamada; el int8×int8 solo se retoma si se quiere lote (NV > 1) gratis.
- **r <= 0.60** -> el kernel está limitado por cálculo con >= 1.67x de margen: **`mv_i8_i8` se justifica** (ganancia máxima 1/r).
- **0.60 < r < 0.85** -> zona intermedia (ganancia máxima entre 1.18x y 1.67x): no se construye nada; se reporta y decides.
- Independiente de r, si BW_ro (4 columnas) supera 1.25x el BW del decode real, hay margen de DMA que el diseño actual no aprovecha y se investiga (tamaño de tile, canales por columna) antes de tocar el cálculo.

**Qué NO se concluye de esto:** el efecto sobre tok/s del modelo completo (hay ~170 llamadas, GPU, CPU); solo si el kernel está limitado por DMA o por cálculo.

## 1. Resultados [medido, 13 oct 2026, desktop abierto: carga media 1.7-2.6; GPU en reposo P8]
Datos crudos: `bench/dma_read_runs.jsonl` (cada línea = un proceso: mediana de 30 llamadas tras 5 de calentamiento, y la contención anotada). Reproducir: `python bench/dma_read_build.py && python bench/dma_read_sweep.py --procs 3`. Medias ± desviación de las 3 medianas (3 procesos independientes); GB/s = 10^9 bytes de A por segundo.

| columnas | A (MB) | lectura pura `ro` (ms · GB/s) | `ro` sin transformación (GB/s) | decode real `real` (ms · GB/s) | r = T_ro / T_real |
|---:|---:|---|---:|---|---:|
| 1 | 4 | 0.688 ± 0.008 · 6.10 ± 0.07 | 6.15 | 0.695 ± 0.005 · 6.03 | 0.990 |
| 1 | 17 | 2.530 ± 0.013 · 6.63 ± 0.03 | 6.61 | 2.549 ± 0.006 · 6.58 | 0.993 |
| 1 | 34 | 5.065 ± 0.021 · 6.62 ± 0.03 | 6.64 | — | — |
| 2 | 4 | 0.403 ± 0.002 · 10.40 ± 0.06 | 10.44 | 0.417 ± 0.023 · 10.07 | 0.967 |
| 2 | 17 | 1.368 ± 0.018 · 12.27 ± 0.16 | 12.46 | 1.356 ± 0.020 · 12.38 | 1.009 |
| 2 | 34 | 2.744 ± 0.014 · 12.23 ± 0.06 | 12.35 | — | — |
| 4 | 4 | 0.268 ± 0.013 · 15.67 ± 0.76 | 16.46 | 0.260 ± 0.001 · 16.16 | 1.033 |
| 4 | 17 | 0.774 ± 0.014 · 21.69 ± 0.39 | 21.76 | 0.769 ± 0.016 · 21.83 | 1.006 |
| 4 | 34 | 1.519 ± 0.027 · 22.10 ± 0.39 | 21.77 | — | — |
| 4 | 67 | 3.080 ± 0.044 · 21.79 ± 0.31 | 21.86 | — | — |

- La variación entre procesos es <=2.5 % (5 % solo en los 4 MB de 4 columnas, donde el tiempo es 0.26 ms). Ninguna configuración supera el 10 %.
- Los tamaños de 64 MB con 1-2 columnas no compilan (la descripción del DMA del shim no admite ese patrón); se cubre hasta 34 MB.
- El resultado del kernel real se verificó contra numpy en cada proceso (`True`).
- **Transformación "quad-transpuesta" del DMA: coste nulo** (`ro` frente a `ro` sin transformación, dentro del 1-2 %).

### Contención de memoria [medido, `bench/dma_read_contention.py`, 4 columnas, 17 MB, 3 procesos por celda]
| condición | diseño | tiempo ms | GB/s | carga media |
|---|---|---:|---:|---|
| escritorio abierto | `ro` | 0.818 ± 0.011 | 20.50 | 2.6 |
| escritorio abierto | `real` | 0.802 ± 0.035 | 20.93 | 2.6 |
| 8 procesos copiando memoria | `ro` | 1.404 ± 0.067 | 11.95 | 3.6-6.7 |
| 8 procesos copiando memoria | `real` | 1.396 ± 0.058 | 12.01 | 3.6-6.7 |
La contención de DDR baja el ancho de banda de lectura de la NPU un **42 %** y afecta igual a `ro` y a `real`; el escritorio abierto cuesta ~5 % respecto a la tanda del barrido (20.5-20.9 frente a 21.7 GB/s).

### Canales por columna [medido, `bench/dma_channels.py`, memcpy 34 MB, GB/s por dirección, 3 medianas]
| columnas | 1 canal | 2 canales |
|---:|---:|---:|
| 1 | 6.72 ± 0.05 | 12.43 ± 0.12 |
| 2 | 12.49 ± 0.10 | 19.69 ± 0.53 |
| 4 | 20.23 ± 0.85 | 21.02 ± 1.52 |
Con 4 columnas el segundo canal no aporta: el techo (~21-22 GB/s de lectura) es del chip, no de la topología del diseño.

## 2. Decisión según el criterio fijado en §0
- **r = 0.97-1.03 en todas las configuraciones, con 4 columnas y >= 16 MB r = 1.006** (>= 0.85): **el decode de un token está limitado por el DMA de lectura**, no por el cálculo. El tiempo del kernel real es el de mover los pesos (±3 %).
- Consecuencia del criterio: **`mv_i8_i8` no se justifica para el decode de un token** (ganancia máxima 1/r ≈ 1.0x). No se escribe.
- BW_ro a 4 columnas (21.8 GB/s) frente al del decode real (21.8 GB/s): no hay margen de DMA sin aprovechar.
- Cuadra con la Fase 0 [medido, `PERF.md` §3]: el `wait` por token del 1.7B (81.5 ms para 1.72 GB) está a 3 % del suelo de lectura a 21.8 GB/s (78.9 ms); en el 0.6B (34.8 ms para 0.60 GB, suelo 27.3 ms) quedan ~7.5 ms (22 %) de costes fijos por llamada.
- **Lo que la medición NO dice y es importante:** el suelo de cálculo existe y es menor que el de DMA solo mientras NV=1. Con los datos de `mv_i8_nv` (4864x896, 2 columnas: 448 / 571 / 1069 µs para NV = 1 / 2 / 4) el cálculo por vector ronda **~270 µs, ~60 % del tiempo de DMA** [estimado]; con NV >= 2 el cálculo pasa a ser el límite. Por eso la verificación en lote cuesta 1.25x (NV=2) y 2.3x (NV=4). Un núcleo con `mmul` int8×int8 (256 MAC por instrucción frente a los 32 actuales) bajaría ese techo de cálculo muy por debajo del de DMA: **solo sirve para lotes** (verificación especulativa, resúmenes en paralelo), no para el token único. Estimación [no medida]: NV=4 pasaría de ~2.3x a ~1.0-1.2x.
- Qué sí puede acelerar el token único: menos bytes por token (int4, cabeza de salida reducida), reparto con la GPU (la GPU lee sus filas de la VRAM, fuera de este DDR) y evitar contención de DDR durante el decode.

## 3. Paso 1 de la Fase 2: bytes por token frente a lo medido [medido, 13 oct 2026]
`bench/bytes_floor.py` (un proceso por modelo; 3 corridas de 48 tokens tras 1 de calentamiento; contexto corto, ~16-64 tokens). Los bytes se cuentan de los objetos `NpuLin` que construye el motor (incluye relleno y la cabeza de salida), no de la configuración. Suelo de DMA = bytes / 22.3 GB/s (pendiente del ajuste de 4 columnas, `§1`); el límite superior suma 0.080 ms por llamada (ordenada del mismo ajuste, llamada síncrona: el motor envía juntas las llamadas de una matriz, así que lo real queda entre los dos). Crudos: `bench/bytes_floor.jsonl`. Las filas "assisted" usan la GPU y se midieron con la **GPU degradada** [medido, GPU degradada].

| modelo | MB int8 por token | llamadas NPU por token | suelo DMA ms (solo bytes - con llamadas) | espera NPU medida ms | espera / suelo | GB/s efectivos | submit + prep + post ms | resto (CPU/GPU) ms | paso total ms | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B (solo NPU) | 596 | 212 | 26.7 - 43.7 | 36.2 ± 0.1 | 1.36 | 16.5 | 8.8 | 6.5 | 51.5 ± 0.3 | 19.4 |
| Qwen3-1.7B (solo NPU) | 1720 | 184 | 77.2 - 91.9 | 83.4 ± 0.2 | 1.08 | 20.6 | 9.8 | 7.2 | 100.4 ± 0.2 | 9.96 |
| Qwen3.5-0.8B | 766 | 218 | 34.4 - 51.8 | 42.0 ± 0.0 | 1.22 | 18.3 | 10.2 | 27.2 | 79.4 ± 2.6 | 12.6 |
| Qwen3.5-2B | 1885 | 170 | 84.5 - 98.1 | 89.8 ± 2.0 | 1.06 | 21.0 | 11.2 | 39.6 | 140.6 ± 3.2 | 7.11 |
| Qwen3.5-2B, reparto GPU 50 % | 940 | 157 | 42.2 - 54.7 | 49.0 ± 0.0 | 1.16 | 19.2 | 84.6 | 24.1 | 157.7 ± 0.1 | 6.34 |
| Qwen3.5-4B | 4376 | 282 | 196.3 - 218.8 | 196.9 ± 2.5 | 1.00 | 22.2 | 19.7 | 80.2 | 296.7 ± 2.6 | 3.37 |

Lectura:
- **La espera de la NPU está en el suelo de DMA** salvo en los modelos pequeños: 1.00-1.08x en 1.7B, 2B y 4B; 1.22-1.36x en 0.8B y 0.6B, donde las matrices son más pequeñas (menos GB/s efectivos: 4 MB rinden ~16 GB/s en `§1`) y pesan más las llamadas. Los bytes contados coinciden con la Fase 0 (596 y 1720 MB). Cuadra con `§2`: no hay margen de cálculo ni de topología que ganar en la espera.
- **Qué queda fuera de la espera:** 17 % del paso en el 1.7B (17 ms), 30 % en el 0.6B, **36 % en el 2B (51 ms) y 34 % en el 4B (100 ms)** [medido, GPU degradada]: GPU (DeltaNet, atención, lectura), CPU y sincronización. Ahí está la otra mitad de la mejora posible, pero con la GPU como está hoy no es comparable con el CHANGELOG (2B: 112 ms en una GPU sana).
- **El reparto 50 % de la GPU no ayuda con esta GPU** (157.7 ms frente a 140.6): la NPU lee la mitad de los bytes (940 MB, espera 49 ms), pero la parte de la GPU (`submit` 84.6 ms) la retrasa. Es el síntoma de la GPU degradada, no del diseño.
- **Techo para un token, solo por bytes [estimado]:** halvar los bytes (int4) bajaría la espera a ~42 / 45 / 98 ms en 1.7B / 2B / 4B, o sea, paso total ~58 / 96 / 198 ms (1.7x / 1.46x / 1.50x) manteniendo el resto igual. El 27 % de los bytes del 2B es la cabeza de salida (509 MB): `NPU_FAST_VOCAB` es el paso 2.
