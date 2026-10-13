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
