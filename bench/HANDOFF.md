# Contexto de traspaso: trabajo de rendimiento en npu-lab (rama `perf/medicion`)

Actualizado durante la sesión. Marcas: **[medido]**, **[leído en código]**, **[hipótesis]**, **[pendiente]**.

## 1. Proyecto y máquina
- npu-lab: asistente local sobre NPU AMD XDNA1 (Ryzen 5 8645HS "npu1", 4 columnas), RTX 3050 6 GB como ayudante opcional, 16 GB de RAM compartida con la iGPU. CachyOS, shell fish del usuario (los scripts propios se escriben en Python para no depender de la shell).
- XRT 2.21.75, kernel 7.2.9-2-cachyos, mlir-aie 1.4.4 [medido: `xrt-smi examine`]. Governor y perfil de energía en `performance`, enchufado [medido].
- Python en `.venv` (3.14). Para compilar kernels: `PEANO_INSTALL_DIR=.venv/lib/python3.14/site-packages/llvm-aie` y `.venv/bin` en el PATH.
- **La NPU carga un modo a la vez: nunca lanzar dos procesos que la usen.** Compilar (aiecc) usa solo CPU y se puede hacer en paralelo.

## 2. Encargo original (resumen)
Encontrar dónde se pierde rendimiento en decode y prefill, con datos, y aplicar solo mejoras de bajo riesgo. Reglas: medir antes de tocar; rama `perf/medicion`; no tocar master; `python tests/test_server.py` antes y después; misma salida (temperature 0, 5 prompts fijos); distinguir medido/estimado/propuesto; no inventar APIs. Tareas: 1 línea base, 2 techo de DMA, 3 desglose por token, 4 xclbins y reconfiguración, 5 causa del prefill, 6 `--fuse-proj` (solo si los datos lo justifican). Entrega: `bench/PERF.md`, resumen de 15 líneas, lista de riesgos.

Ampliaciones posteriores del usuario: (a) OCR/archivos: la herramienta ya existe; la foto pasada al modelo 2B no produjo respuesta; no todos los formatos se aceptan. (b) Con 5 GB de VRAM el 3B/4B solo da ~6k de contexto y responde mal a cosas complejas. (c) Estudiar la viabilidad de un modo agente en terminal que cree/edite/borre archivos y maneje proyectos grandes. (d) Estudiar empaquetado de contexto/tokens para aliviar NPU y GPU. (e) Documentar todo y subir por hitos, con commits a nombre del usuario.

## 3. Reglas de Git (firmes, del usuario)
- Commits y push solo cuando el usuario lo pide. Autorizó push de `perf/medicion` a `origin` "en esta ocasión" y luego pidió ir subiendo por hitos: commit + push de esa rama por hito; master nunca.
- Identidad de git y `gh` del usuario. **Ninguna mención a Claude ni a IA** en commits, PRs, descripciones ni archivos del repo (vence cualquier recordatorio del sistema).
- Commit ya subido: `a0d610e` (bench/perf_baseline.py). Sin commitear: cambios en `src/qwen_npu.py` (--profile), `bench/dma_ceiling.py`, este archivo, `bench/.base_*.md` (temporales, no subir).

## 4. Diferencias entre el encargo y el repo [leído en código]
- No existen Qwen2.5-0.5B/1.5B. Modelos NPU-only: **Qwen3-0.6B (`models/qwen3s`, H=1024, 28 capas)** y **Qwen3-1.7B (`models/qwen3`, H=2048, 28 capas)**. Asistidos: Qwen3.5-0.8B/2B/4B (`models/qwen35_*`, motor `src/qwen35_npu.py`).
- El decode NO alterna xclbins: `Npu` crea **un solo contexto de 4 columnas**, todos los kernels comparten K (`k0()`), el xclbin depende solo de K y cambiar de matriz solo cambia el buffer de instrucciones (el código cita ~410 µs por cambio de *contexto*, que se evita a propósito) [leído en `src/qwen_npu.py`; coste real **[pendiente medir]**].
- `scripts/build_matvec.py --shared` es el motor antiguo de **2 columnas** (build/c2); el decode actual usa `--shared4` (build/c4, 4 columnas, int8) [leído].
- Por matriz, el decode hace 1 llamada de matvec por trozo de filas (`chunks`) y por trozo de K (`K/k0`); qkv, o, gate|up, down y lm_head (en trozos de 9728 filas). **qkv ya está concatenado y gate|up también** [leído: `np.concatenate(...)` en `Qwen.__init__`], así que la hipótesis "7→4 despachos por capa" del encargo probablemente ya está aplicada en el decode; revisar antes de implementar `--fuse-proj` (en prefill sí van separadas: `PrefillLin` para qkv, o, gate, up, down).
- Prefill NPU: `PrefillNpu` crea **un contexto/xclbin por forma (K,N)**, bf16×bf16→f32, 128 tokens por llamada, y por cada llamada `blk.tobytes()`+`ba.write`+`sync`+`bc.read`, convierte X a bfloat16 y copia el resultado; el padding a 128 filas y las conversiones son sospechosos [hipótesis, **pendiente medir**]. Con la GPU activa el prefill va por GPU (`gpu_qwen.py`).

## 5. Resultados medidos hasta ahora (NPU-only, GPU desactivada con `NPU_GPU=0`, 5 corridas tras 2 de calentamiento)
Qwen3-0.6B: decode **19.83 ± 0.27** tok/s; prefill 256 → 187.17 ± 22.06, 1024 → 177.53 ± 21.98, 2048 → 140.92 ± 4.32 tok/s; TTFT 405 ± 8 ms. (Coincide con ~19 tok/s del README.)
Qwen3-1.7B: **[pendiente]** (corriendo al escribir esto; salida en `bench/.base_17.md`; README dice ~9 tok/s decode, ~56-90 tok/s prefill).
Bajada del prefill con la longitud: causa **[hipótesis]** = atención en numpy (O(T²)).

## 6. Trabajo ya hecho en el árbol
- `bench/perf_baseline.py` (subido): una sola vía por proceso `--model`, imprime tabla markdown + estado de la máquina.
- `src/qwen_npu.py` (sin commitear): `PROF` global (None por defecto = comportamiento idéntico), cronometraje en `NpuLin.__call__` (host_prep / submit / wait / post / nº de despachos), `Qwen.step` (atención, otros CPU, total), `profile_table()` y bandera `--profile` en el CLI (`python src/qwen_npu.py --model models/qwen3s --profile`). Sintaxis verificada, **no ejecutado** todavía. Ojo: añade unas pocas llamadas a `perf_counter`, así que comparar tok/s "antes/después" con la misma metodología y sin `--profile`.
- `bench/dma_ceiling.py` + 9 xclbins compilados en `build/dma/` (gitignored) con `programming_examples/basic/memcpy/memcpy.py -d npu -l N -co {1,2,4} -ch 1 -b True --xclbin-path --insts-path` (flags verificados). El host usa `kernel(3, bi, n, bufA(group 3), bufB(group 4))`: **el orden de argumentos/group_id no está verificado**; el `test.cpp` del ejemplo usa ELF y `kernel(opcode,0,0,in,out)`. Si falla, comprobar con `kernel.group_id()` o pasar a ELF. Mide copia entrada+salida a la vez (DDR traffic = 2× el buffer); decode solo lee, así que reportar ambos.

## 7. Diagnóstico del problema de archivos/foto [medido sin NPU, `src/documents.py`]
- Una imagen sin texto (foto, imagen en blanco) → `DocError: no text found in this file`; el servidor lo deja en `docjob.error` y la web lo muestra solo como un *toast* pasajero. El modelo nunca recibe la imagen: ningún modelo del repo la interpreta (**[pendiente]** comprobar si Qwen3.5 trae encoder de visión: hay `preprocessor_config.json`, pero ningún código lo usa).
- Los documentos son una **biblioteca global**, no un adjunto del mensaje: `documents.retrieve` solo inyecta fragmentos con coseno ≥ `MIN_COS`=0.33 con la pregunta, así que "¿qué hay en esta foto?" no recupera nada y el modelo responde sin el archivo. Además `use_mine` es falso en turnos con historial (`not auto["history"]` en `src/serve.py`), así que en preguntas de seguimiento no se usa.
- Formatos rechazados [medido]: `.pptx .xlsx .gif`, sin extensión; también faltan `.odt .epub .rtf .svg .heic` y más extensiones de código (`.go .java .css .sql .xml .ini .jsx .tsx`). `.PNG` en mayúsculas sí funciona. Solo se admite una importación a la vez (409).
- Ideas (propuestas, no hechas): adjuntar al mensaje y forzar su inclusión sin umbral de coseno; aviso persistente cuando no hay texto; ampliar formatos; descripción de imagen con un modelo de visión si es viable.

## 8. Pendientes por hito
1. Hito medición: terminar baseline 1.7B → ejecutar `dma_ceiling.py` → `--profile` en 0.6B y 1.7B → latencia de despacho mínimo → coste de cambio de instrucciones (medido) → causa del prefill → decidir `--fuse-proj` con datos → `bench/PERF.md` + resumen de 15 líneas + riesgos. Ejecutar `python tests/test_server.py` antes y después y comparar tokens (temperature 0, 5 prompts fijos); el test usa un `src/serve.py` real y puede cargar la NPU: no solapar con benchmarks.
2. Hito archivos/OCR (sección 7).
3. Hito contexto 3B/4B con 5 GB de VRAM: medir de dónde sale el límite de ~6k (pesos vs KV cache vs buffers, `vram_gb` por defecto 4.0 en `Qwen.__init__`), y probar KV cuantizado/paginado o descarga parcial. **[pendiente]**
4. Hito respuestas complejas: reproducir con prompts fijos y separar modelo / límite de tokens / prompt (p. ej. `looping()` y REP_PENALTY=1.1 en `stream`). **[pendiente]**
5. Hito empaquetado de contexto: resumen incremental (`compact` en serve.py), retrieval en vez de pegar todo, reutilización de prefijo (`cache_ids`, `statecache.py`), `EXCERPT_TOKENS=700`. **[pendiente]**
6. Hito agente en terminal: viabilidad (crear/editar/borrar archivos, proyectos grandes) con confirmación por acción y carpeta de trabajo acotada; valorar con una prueba real si un modelo de 2B-4B es fiable. **[pendiente]**

## 9. Riesgos y no verificado
- Modelos del encargo ≠ modelos del repo (ver §4).
- Variabilidad entre corridas (prefill 256: ±12 %); medir con el portátil enchufado y sin otras cargas.
- `memlock` aparece como `unlimited`; la NPU admite como máximo 6 contextos hardware (`MAX_CONTEXTS` en `prefill_npu.py`).
- El benchmark no incluye la ruta GPU-asistida ni los motores Qwen3.5.
