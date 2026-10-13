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

## 4. Contexto largo [pendiente de pegar]: ver `bench/attn_vs_ctx.py`.

## 5. xclbins y reconfiguración
- **[leído]** El decode usa **un único xclbin y un único `hw_context` de 4 columnas** por modelo (`Npu.kernel`, `BUILD="build/c4"`). Todos los kernels comparten K (`k0()`); cambiar de matriz solo cambia el buffer de instrucciones. No alterna xclbins entre tokens.
- **[leído]** `scripts/build_matvec.py --shared` es el motor **antiguo de 2 columnas** (`build/c2`); el xclbin depende solo de K y varía el `.bin` por M. El decode actual usa `--shared4` (`build/c4`, 4 columnas).
- **[leído]** El prefill sí crea un contexto por forma (K,N) de matriz (`PrefillNpu.kernel`), con límite de 6 contextos hardware (`MAX_CONTEXTS`).
- Coste medido de cambiar el buffer de instrucciones: **[pendiente]**.

## 6. Causa del prefill lento [pendiente]
Hipótesis a medir: atención en numpy (el prefill de 1.7B cae de 108 a 52 tok/s al pasar de 256 a 1024 tokens, patrón compatible con coste O(T²) en CPU), conversión a bf16 y copias por llamada de 128 filas, padding. Nada confirmado.

## 7. Mejora de bajo riesgo (`--fuse-proj`)
**[leído]** En el decode, qkv y gate|up **ya están concatenados** (`np.concatenate` en `Qwen.__init__`), así que la fusión propuesta ya existe ahí y no hay ganancia que obtener con esa bandera. No implementada.
