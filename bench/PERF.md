# Rendimiento de decode y prefill (rama `perf/medicion`)

Etiquetas: **[medido]** salió de un script de este repo; **[leído]** viene del código; **[estimado]** cálculo a partir de medidas; **[hipótesis]** sin comprobar; **[pendiente]**.

## 0. Entorno [medido]
Ryzen 5 8645HS (NPU XDNA1, 4 columnas), CachyOS, kernel 7.2.9-2-cachyos, XRT 2.21.75 (`xrt-smi examine`), mlir-aie 1.4.4, governor y `powerprofilesctl` en `performance`, enchufado. Todo **solo NPU** (`NPU_GPU=0`, sin GPU). Había fondo animado y shell de escritorio consumiendo CPU, lo que añade ruido.

**Desviaciones del encargo:** el repo no contiene Qwen2.5-0.5B/1.5B; se midieron **Qwen3-0.6B** (`models/qwen3s`) y **Qwen3-1.7B** (`models/qwen3`). La 1.7B se midió con 3 corridas tras 1 de calentamiento y prefill a 256 y 1024 tokens (la corrida completa tardó más de 20 min sin terminar); la 0.6B con el protocolo completo (5 tras 2).

## 1. Línea base [medido] (`bench/perf_baseline.py`)
| métrica | Qwen3-0.6B (5+2) | Qwen3-1.7B (3+1) |
|---|---|---|
| decode tok/s (32 pasos, contexto ≈16) | 19.83 ± 0.27 | 10.10 ± 0.14 |
| prefill tok/s, 256 tokens | 187.17 ± 22.06 | 108.28 ± 6.72 |
| prefill tok/s, 1024 tokens | 177.53 ± 21.98 | 52.11 ± 0.84 |
| prefill tok/s, 2048 tokens | 140.92 ± 4.32 | no medido |
| tiempo al primer token (prompt de chat ~40 tok) | 405 ± 8 ms | 847 ± 1 ms |

## 2. Techo de DMA [medido] (`bench/dma_ceiling.py`, ejemplo `memcpy`, ruta shim→memtile→shim, int32, datos verificados)
Cada ejecución lee y escribe el buffer a la vez. "Por dirección" = tamaño/tiempo.
| columnas | MB | ms (mediana) | GB/s por dirección (mediana) | mejor | tráfico DDR total (mediana) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 / 32 / 128 | 1.33 / 5.07 / 20.38 | 6.33 / 6.62 / 6.59 | 6.51–6.90 | 12.7–13.2 |
| 2 | 8 / 32 / 128 | 0.74 / 2.78 / 10.91 | 11.34 / 12.06 / 12.30 | 11.8–13.2 | 22.7–24.6 |
| 4 | 8 / 32 / 128 | 0.48 / 1.64 / 7.38 | 17.32 / 20.41 / 18.18 | 18.5–21.9 | 34.6–40.8 |

Escala casi lineal con las columnas (≈6.5 GB/s por columna). **Salvedad:** el test mueve lectura y escritura simultáneas; el decode solo lee pesos, así que su techo de lectura pura podría ser distinto **[no medido]**.

## 3. Desglose por token [medido] (`python src/qwen_npu.py --model DIR --profile`, contexto ≈60)
`--profile` no cambia el comportamiento por defecto; su coste es ≈0–3 % (16.4 tok/s sin él, 15.8–16.7 con él, 0.6B).
| componente | 0.6B ms (%) | 1.7B ms (%) |
|---|---:|---:|
| NPU: esperar kernels (`wait`), suma de las 5 matrices | 34.8 (56 %) | 81.5 (73 %) |
| NPU: enviar despachos (`submit`) | 5.7 (9 %) | 5.7 (5 %) |
| NPU: preparar entrada/cuantizar/sync (`host_prep`) | 2.0 (3 %) | 2.2 (2 %) |
| NPU: leer salida/acumular/escalar (`post`) | 3.6 (6 %) | 4.6 (4 %) |
| **Atención + norm q/k + RoPE (CPU numpy)** | **13.1 (21 %)** | **13.9 (12.5 %)** |
| Resto CPU (rms, silu, embed, residuos) | 2.6 (4 %) | 3.2 (3 %) |
| Muestreo, penalización, detokenizar (fuera de `step`) | 0.14 | 0.18 |
| **Paso total** | **61.8** | **111.2** |
Despachos por token: 212 (0.6B), 184 (1.7B); llamadas a `NpuLin`: 113 por token en ambas.

**Cuello de botella principal [estimado a partir de medidas]:** los bytes int8 por token son 596 MB (0.6B) y 1720 MB (1.7B) (calculados de `config.json`). Dividiendo por el tiempo de `wait`: 17.1 GB/s (0.6B) y 21.1 GB/s (1.7B), es decir, **el tramo NPU del decode ya está en el techo de DMA medido (≈18–22 GB/s con 4 columnas)**. La hipótesis del encargo de "~9–10 GB/s efectivos" no se sostiene en estos modelos. Lo que queda por ganar sin cambiar el modelo ni la cuantización está fuera de la NPU: ≈13 ms de atención en CPU por token y ≈10 ms de overhead de despacho/host.

## 4. Contexto largo, decode NPU-only [medido] (`bench/attn_vs_ctx.py`, Qwen3-0.6B, 3 corridas, media ± desviación, 13 oct 2026)
Atención con caché head-major y consultas float32 (commit `f0e81bd`). "Antes" = una sola corrida de la sesión anterior (atención antigua), sin repetir.
| contexto | decode tok/s | paso ms | atención ms | NPU `wait` ms | antes: paso ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 18.4 ± 0.1 | 54.5 ± 0.3 | 5.8 ± 0.2 | 33.9 ± 0.1 | 52.0 |
| 256 | 17.5 ± 0.2 | 57.2 ± 0.6 | 8.8 ± 0.3 | 34.2 ± 0.2 | 98.7 |
| 1024 | 12.5 ± 0.2 | 79.9 ± 1.2 | 31.7 ± 0.4 | 34.3 ± 0.2 | 290.3 |
| 2048 | 9.4 ± 0.1 | 106.5 ± 1.2 | 57.2 ± 0.8 | 34.3 ± 0.1 | 556.1 |
| 3500 | 6.8 ± 0.1 | 146.3 ± 1.8 | 95.1 ± 1.1 | 34.1 ± 0.1 | 1056.3 |
La atención sigue creciendo con el contexto (95 ms a 3500) pero 10x menos que antes. **Mismos tokens** que la atención antigua [medido]: ver §8.

## 5. xclbins y reconfiguración
- **[leído]** El decode usa **un único xclbin y un único `hw_context` de 4 columnas** por modelo (`Npu.kernel`, `BUILD="build/c4"`). Todos los kernels comparten K (`k0()`); cambiar de matriz solo cambia el buffer de instrucciones. No alterna xclbins entre tokens.
- **[leído]** `scripts/build_matvec.py --shared` es el motor **antiguo de 2 columnas** (`build/c2`); el xclbin depende solo de K y varía el `.bin` por M. El decode actual usa `--shared4` (`build/c4`, 4 columnas).
- **[leído]** El prefill sí crea un contexto por forma (K,N) de matriz (`PrefillNpu.kernel`), con límite de 6 contextos hardware (`MAX_CONTEXTS`).
- Coste medido de cambiar el buffer de instrucciones dentro de un contexto, y de cambiar de contexto: ver §8 (1-5 µs frente a ~400 µs).

## 6. Causa del prefill lento (NPU-only, Qwen3-0.6B) [medido]
Es la **atención numpy de un solo hilo**, cuadrática en la longitud del prompt. Ver §8, pregunta 2.

## 7. Mejora de bajo riesgo (`--fuse-proj`)
**[leído]** La bandera no existe en el código (solo en el encargo original). En el decode, qkv y gate|up **ya están concatenados** (`np.concatenate` en `Qwen.__init__`, `src/qwen_npu.py` ~l. 351-356); `o` y `down` son matrices sueltas sin hermana con la que fusionar. Decisión con datos en §8, pregunta 4: no hace falta.

## 8. Fase 0: las cuatro preguntas abiertas, cerradas con número (13 oct 2026)
Todo **solo NPU** (`NPU_GPU=0`), un proceso a la vez, 3 corridas por medida salvo que se indique. Las desviaciones del encargo (modelos Qwen3-0.6B/1.7B, no Qwen2.5) siguen igual que en §0.

### 8.1 ¿Mismos tokens con la atención NPU-only nueva? **Sí** [medido]
`bench/token_equiv.py`: 5 prompts fijos, 40 tokens, decodificación voraz (el `stream()` del motor, penalización de repetición incluida), comparados con la salida del commit anterior al arreglo (`f0e81bd^`, atención antigua, ejecutado desde un worktree temporal con la misma `.venv`/modelos).
| modelo | prompts idénticos (texto y tokens) | corridas |
|---|---|---|
| Qwen3-0.6B (`models/qwen3s`) | 5 / 5 | 3 / 3 |
| Qwen3-1.7B (`models/qwen3`) | 5 / 5 | 3 / 3 |
Incluye el prompt con ~600 tokens de documento (atención sobre caché más larga). No cubre contextos de 3500 tokens ni el perfil Asistido.

### 8.2 Causa del prefill NPU-only (187 → 141 tok/s en la línea base): atención numpy [medido]
`bench/prefill_breakdown.py` (3 corridas por tamaño; tiempos totales) y `bench/prefill_attn.py` (el bloque de atención de `Qwen.prefill`, mismo código, aislado, sin NPU; 3 corridas), Qwen3-0.6B:
| tokens | prefill total s | tok/s | NPU lineales (wait+copias+despacho) s | atención numpy s | % atención | resto (norms, RoPE, SiLU) s |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.17 ± 0.02 | 220 ± 4 | 0.57 | 0.53 ± 0.03 | 45 % | 0.07 |
| 1024 | 6.51 ± 0.03 | 157.3 ± 0.6 | 1.75 | 4.14 ± 0.01 | 64 % | 0.62 |
| 2048 | 18.65 ± 0.14 | 109.8 ± 0.8 | 3.26 | 14.42 ± 0.08 | 77 % | 1.0 |
- La parte NPU crece casi lineal con los tokens (0.57 → 1.75 → 3.26 s); la atención crece ×7.8 al ×4 de tokens y ×3.5 al ×2: cuadrática. Con 2048 tokens, `wait` del kernel es solo el 12 % del tiempo; copias (convertir a bf16, rellenar a 128 filas, escribir/leer) ≤ 4 %; despacho ≤ 2 %.
- **[estimado]** ~19 GFLOP de atención por capa a 2048 tokens en 0.52 s → ~37 GFLOP/s: del orden de un solo núcleo de CPU en fp32 (el motor fija `OMP_NUM_THREADS=1`). No es DMA ni el padding.
- Consecuencia **[propuesto]**: más hilos para la atención del prefill, tras una bandera apagada por defecto (ROADMAP §2). No aplicado en esta fase.
- Aviso de método: las cifras absolutas difieren de la línea base de §1 (256 → 187 ± 22, 2048 → 141 ± 4; otro arnés, el actual da 220 y 110). La tendencia (cae con el tamaño) es la misma. Una corrida de la sesión anterior (`bench/.pf_06.md`: 1024 → 29 s, 2048 → 160 s) no se reprodujo en ninguna de las 3+3 corridas de ahora; se atribuye a carga ajena en ese momento **[no verificado]** y no se usa.

### 8.3 Coste de cambiar el buffer de instrucciones: ~1-5 µs, no ~410 µs [medido]
`bench/insts_switch_npu.py` (nuevo): un solo xclbin/contexto, dos flujos de instrucciones (matrices de distinto M), llamada síncrona, 300 llamadas × 5 repeticiones (µs por llamada, media ± desviación).
| par (M×K) | A sola | B sola | A+B alternando (cambia el buffer) | control A+A' (mismo buffer) | penalización por cambio |
|---|---:|---:|---:|---:|---:|
| 1024×1024 / 2560×1024 | 147.7 ± 1.3 | 211.0 ± 0.7 | 364.1 ± 1.5 | 298.3 ± 4.5 | **+5.4 µs** (control +2.9) |
| 512×1024 / 768×1024 | 125.0 ± 5.4 | 135.0 ± 4.6 | 261.4 ± 10.2 | 250.7 ± 6.4 | **+1.3 µs** (control +0.6) |
A 113 llamadas por token, el cambio de instrucciones cuesta como mucho ~0.6 ms por token (<1 % de 54 ms). Coste fijo de una llamada síncrona ≈ 105 µs (ajuste lineal de los tiempos de arriba: ≈ 0.039 µs por fila de 1024 B).
El **~410 µs citado en el código es el cambio de CONTEXTO (xclbin distinto), y se confirma** [medido, `bench/ctx_switch_npu.py`, 3 corridas, K=1024 frente a K=2048, ambos en 4 columnas]: penalización 789 / 800 / 807 µs por par = **~400 µs por cambio**. Por eso el motor usa un solo contexto con K común: lo que evita es 400 µs, no 5.

### 8.4 ¿Hace falta `--fuse-proj` en el decode? **No** [leído + medido]
- Ya fusionado donde se puede: qkv y gate|up van concatenados; quedan `o` y `down` (cada una depende de la salida anterior) y la cabeza de salida en trozos. No hay proyecciones hermanas que fundir.
- Techo teórico de todo el coste fijo de llamadas: 113 llamadas × ~105 µs ≈ 11.9 ms por token, el 22 % de los 54.5 ms del 0.6B y el 11 % de los 111 ms del 1.7B **[estimado]**; en la práctica los envíos se solapan (`submit` medido: 5.7 ms) y las llamadas restantes no se pueden fundir por dependencia de datos. Fundir la cabeza de salida (16 trozos) es otro asunto, ya en el ROADMAP ("Fewer launches").
- En el prefill las llamadas son grandes: el envío (`submit`) es el 1-2 % del tiempo (§8.2), así que fundir gate y up allí ganaría <2 % **[estimado a partir de §8.2]**.
- Decisión: no se implementa; queda registrado en `docs/DECISIONS.md`.
