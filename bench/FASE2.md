# Fase 2, paso 2: lotes (13 oct 2026)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**. Todo lo de Qwen3.5 que use la GPU: **[medido, GPU degradada]** (`bench/ENERGY.md`: 2.5 TFLOPS, memoria a 780 MHz). `mv_i8_i8` sigue en espera y no se escribe. Cada lote fija su criterio de éxito ANTES de medir (commit previo a los resultados), n >= 3 y cambios de comportamiento tras una bandera apagada por defecto.

## Lote A. Descomposición de lo que queda fuera de la espera de la NPU (Qwen3.5-2B y 4B)
**Qué se mide:** por token, el tiempo del paso (`Qwen35.step`) repartido en: espera de la NPU, resto del coste de las llamadas NPU (preparar, enviar, recoger), pasos de la GPU (`linear_step` de las capas DeltaNet y `attn_step` de las de atención completa, que incluyen la subida/bajada de datos, porque devuelven NumPy), cabeza de salida (`lm`), y el pegamento de CPU (embedding, normas, SiLU, sumas residuales). Instrumentación en el propio script (copia del paso con relojes), sin tocar `src/`; el reparto por GPU queda a 0 %.
**Criterio de éxito:** (1) la copia instrumentada devuelve exactamente los mismos logits que `Qwen35.step` sobre 5 pasos fijos (diferencia 0); (2) las partes suman el paso total con un error <= 5 %; (3) 3 corridas de 48 tokens por modelo, media ± desviación, desviación del total <= 5 %; (4) se identifica la parte mayor fuera de la espera y se estima su cota inferior (qué parte es lanzamiento/sincronización y qué parte cálculo) con una medida, no con una suposición.
**Qué NO se concluye:** nada sobre una GPU sana; las cifras de GPU son de una GPU degradada.

### Resultados del lote A [medido, GPU degradada; `bench/step_breakdown.py`, 3 corridas x 48 tokens, reparto de GPU 0 %, crudos en `bench/step_breakdown.jsonl`]
Criterios: (1) logits idénticos en 5 pasos fijos: **sí** en los dos modelos; (2) las partes suman el paso con error <= 0.1 %; (3) desviación del total 1.7 % (2B) y 0.3 % (4B); (4) cota inferior medida (abajo).

| parte del paso | 2B ms (%) | 4B ms (%) |
|---|---:|---:|
| NPU: espera del flujo de pesos | 88.0 ± 0.4 (65 %) | 195.5 ± 0.2 (67 %) |
| NPU: coste de las llamadas (preparar, enviar, recoger; capas + cabeza) | 10.4 (8 %) | 20.3 (7 %) |
| GPU: pasos DeltaNet (18 / 24 por token) | 28.0 ± 1.1 (21 %) | 64.2 ± 0.3 (22 %) |
| GPU: pasos de atención (6 / 8 por token) | 7.7 ± 0.2 (6 %) | 9.9 ± 0.3 (3 %) |
| CPU: pegamento (embedding, normas, SiLU, residuales) | 1.7 (1 %) | 3.0 (1 %) |
| paso total (con relojes) | 136.0 ± 2.3 | 293.1 ± 0.9 |

**Lo que hay fuera de la espera (36 % en el 2B, 33 % en el 4B):**
- **El mayor trozo son los pasos DeltaNet de la GPU (21-22 %), y casi todo es tiempo de núcleos en una GPU degradada:** repetir los grafos CUDA de las capas DeltaNet sin copias tarda 16.6 ms (2B; 921 µs por capa) y 50.8 ms (4B; 2117 µs por capa), el 59 % y el 79 % del tiempo medido de esos pasos. Con la GPU sana el CHANGELOG da 3.9 ms por token para el 2B, así que esta parte hoy está limitada por la GPU, no por el diseño [medido, GPU degradada].
- **Lanzamiento, copias y sincronización de la GPU (cota inferior del ahorro por fusionar):** un viaje GPU vacío (copiar, repetir un grafo mínimo, sincronizar, copiar) cuesta 22-23 µs, es decir 0.5-0.7 ms por token: el coste fijo por sí solo es despreciable. Lo que sobra entre el tiempo de núcleos (perfil de un token: 18.1 ms en el 2B con 1152 núcleos y 306 lanzamientos; 54.1 ms en el 4B con 1584 núcleos y 408 lanzamientos) y el tiempo de los pasos de GPU (35.7 y 74.1 ms) son **~17.6 ms (13 % del paso) en el 2B y ~20 ms (7 %) en el 4B** entre copias, conversiones a NumPy, sincronización y huecos entre lanzamientos. Es el techo de lo que se ganaría fusionando lanzamientos y copias, y solo existe con GPU; con la GPU sana la parte de núcleos se encoge y este resto pesaría más.
- **Costes fijos de la NPU: 10.4 ms (8 %) en el 2B y 20.3 ms (7 %) en el 4B.** Es la parte accionable sin GPU: lote C.
- El pegamento de CPU es 1 %; no merece trabajo.
- No hay solapamiento posible dentro de un token (cada capa depende de la anterior): la GPU espera a la NPU y al revés.

## Lote B. `NPU_FAST_VOCAB` en Qwen3.5 (bandera apagada por defecto)
**Qué cambia (tras bandera `NPU_FAST_VOCAB=1` o `fast_head=True`, solo con reparto de GPU 0 %):** la cabeza de salida de la NPU se calcula sobre 38 912 de los 248 320 tokens (4 trozos de 9 728 filas, ya compilados) en lugar de todo el vocabulario: ~429 MB menos por token en el 2B (la cabeza pesa 509 MB de 1885 MB). El subconjunto se elige con la misma función que el motor denso (`fast_vocab`: frecuencia en el índice de documentos y en las traducciones del sistema, todos los símbolos y los 256 bytes) pero con **caché propia por tokenizador**, porque Qwen3 y Qwen3.5 solo comparten 281 ids de token [medido] y la caché antigua se valida solo por longitud (con 38 912 entradas una caché de Qwen3 pasaría por válida para Qwen3.5: error silencioso). Sin bandera, el código es el de antes.
**Criterio de éxito (fijado antes de medir):** se recomienda pasar a "opcional visible" si se cumplen TODOS:
1. **Velocidad** [medido, GPU degradada], Qwen3.5-2B, bandera encendida frente a apagada en procesos alternados, 3 + 3, 48 tokens, contexto corto: la media de tok/s mejora **>= 10 %** y la mejora supera 2 desviaciones de la diferencia (los intervalos no se solapan). Estimación previa: 429 MB / 21.9 GB/s = 19.6 ms de 136 ms = +16 %.
2. **Calidad:** en 10 prompts fijos (los 5 de `bench/token_equiv.py` más 5 de español, código y charla), decodificación voraz de 80 tokens: (a) la fracción de tokens de la salida con cabeza completa que NO están en el subconjunto (los que la cabeza reducida no puede escribir) **<= 0.5 %**; (b) texto idéntico a la cabeza completa en **>= 8 de 10** prompts; en los demás se cuantifica el índice del primer token distinto y la coincidencia por token.
3. Sin la bandera, la salida y el rendimiento son los de antes (los tests offline pasan y, en el modo apagado, los mismos logits en 5 pasos fijos).
Si se cumple 1 pero falla 2: "opcional con aviso" (qué se pierde); si falla 1: no se recomienda.

### Resultados del lote B [medido, GPU degradada; Qwen3.5-2B; `bench/fast_vocab_eval.py`, crudos en `bench/fast_vocab_runs.jsonl`]
**1. Velocidad (3 + 3 procesos alternados, 48 tokens, contexto corto): CUMPLE.** Bandera apagada: 139.1 ± 0.9 ms/token (7.19 tok/s; 140.0, 138.9, 138.3). Bandera encendida: 117.8 ± 0.5 ms/token (8.49 tok/s; 117.9, 117.3, 118.3). **+18.1 %**, intervalos separados por más de 2 desviaciones; la estimación previa era +16 %. Anchura de los logits: 248 320 -> 38 912.
**2. Calidad: NO CUMPLE.** (a) De los 639 tokens que genera la cabeza completa en los 10 prompts, **57 (8.92 %)** no están en el subconjunto (criterio <= 0.5 %). (b) Texto idéntico en **1 de 10** prompts (criterio >= 8): el primer token distinto cae entre el 1 y el 55; la coincidencia por token va de 0.03 a 1.00. Ejemplos: "París" se escribe "**la República Francesa es París**"; en el prompt del JSON la cabeza reducida inventó el autor ("Antón de Sarmiento" en lugar de "Antoine de Saint-Exupéry"), porque no puede escribir los trozos de ese nombre.
**3. Sin la bandera: CUMPLE.** Logits idénticos al commit anterior en 5 pasos fijos (diferencia máxima 0.0); la velocidad coincide con el paso 1 (139-141 ms).

**Causa de la calidad [medido]:** los tokens que faltan no son raros: son palabras españolas comunes (" Francia", " París", " cielo", " cerebro", " humano", " aprender", " receta", " harina", " maíz", " Aquí") y trozos ingleses (" mol", "customer", " atm"). El subconjunto se elige con la frecuencia en el índice de documentos y en las traducciones del sistema, y el vocabulario de Qwen3.5 (248k, con muchas palabras españolas enteras en los ids altos) queda mal cubierto: 38 912 filas son el 15.7 % del vocabulario, frente al 25.6 % en el Qwen3 (151k) para el que se diseñó.

**Veredicto según la regla fijada:** velocidad sí, calidad no -> **"opcional con aviso", NO "opcional visible".** Queda tras la bandera `NPU_FAST_VOCAB=1` (Qwen3.5, reparto 0 %), apagada por defecto, con el aviso de que cambia el texto en 9 de cada 10 respuestas de estos prompts y puede inventar nombres propios.

**Propuesta (sin hacer; necesita tu OK):** lote B2: elegir el subconjunto por frecuencia en texto generado por el propio modelo con prompts de selección distintos de los 10 de evaluación (español e inglés, código), manteniendo todos los símbolos y bytes; mismo criterio, mismas 10 preguntas. Coste: ~15 min de generación con la GPU degradada. Solo tiene sentido si merece el +18 % (que sigue disponible a otro precio de calidad) frente al riesgo de cambiar la salida.

## Lote C. Costes fijos de las llamadas a la NPU (bandera apagada por defecto)
**Punto de partida [medido, lote A]:** en Qwen3.5 el coste de las llamadas (preparar la entrada, enviar, recoger) es 10.4 ms/token en el 2B (170 llamadas, ~61 µs por llamada) y 20.3 ms/token en el 4B (282 llamadas); en los Qwen3 de solo NPU, 8.8 ms (0.6B) y 9.8 ms (1.7B). Cada llamada de `NpuLin.__call__` (`src/qwen_npu.py`) hace: cuantizar la entrada (`abs().max()`, división, redondeo, `astype`), escribir y sincronizar cada trozo de entrada, **crear una ejecución nueva por trozo** (`npu.kern(3, bi, n, bo, bx, bc)`), esperar, `bc.sync` de cada salida, convertir a int64 y escalar, concatenar. Candidatos, cada uno medido por separado en un microbenchmark antes de tocar el motor: (V1) ejecuciones preconfiguradas y reutilizadas (`pyxrt.run` con `set_arg` una vez; por llamada solo `start` y `wait`); (V2) sin el `bc.sync` de salida si los resultados no cambian; (V3) postproceso con búferes preasignados.
**Qué se mide primero (microbenchmark, `bench/call_overhead.py`, NPU libre, n >= 3 procesos):** microsegundos por llamada de una matriz de 1 trozo y de 3 trozos (como `down`), y su variante tamaño pequeño (donde el coste fijo domina), para V0 (actual) y V1-V3 acumulados. Solo se pasa al motor si el microbenchmark da **>= 20 µs menos por llamada con resultados idénticos bit a bit**.
**Criterio de éxito en el motor (tras bandera `NPU_RUN_REUSE=1`, apagada por defecto):** (1) logits idénticos (diferencia 0) a la bandera apagada en 5 pasos fijos, en un Qwen3 y un Qwen3.5; (2) el coste de las llamadas en Qwen3.5-2B baja **>= 20 %** (de 10.4 a <= 8.3 ms/token) o, equivalentemente, el paso baja **>= 2 %** (>= 2.8 ms), medido en procesos alternados 3 + 3 de 48 tokens [medido, GPU degradada]; (3) lo mismo en Qwen3-1.7B de solo NPU (sin GPU): el paso baja >= 2 %. Si (1) falla no se integra; si (1) se cumple y (2) y (3) no, la bandera se descarta y se documenta el techo medido.
**Fuera de este lote:** fusionar los 26 trozos de la cabeza de salida (necesita xclbins nuevos), que se evalúa solo si V1-V3 no bastan y con otro criterio.

### Resultados del lote C [medido; microbenchmark en NPU libre, motor con Qwen3.5-2B [medido, GPU degradada] y Qwen3-1.7B de solo NPU]
**Microbenchmark** (`bench/call_overhead.py`, crudos en `bench/call_overhead.jsonl`; 3 procesos, mediana de 300 llamadas por proceso, resultado comparado con V0 bit a bit): reutilizar las ejecuciones (V1) ahorra **19-39 µs por llamada** (M = 1024: 29 ± 19 y 35 ± 20 µs con 1 y 3 trozos; M = 2048: 20 ± 1 y 39 ± 10; M = 12288: 21 ± 33 y 19 ± 12), con resultados idénticos. **Quitar la sincronización de la salida (V2) cambia los resultados: esa sincronización es necesaria.** El postproceso sin temporal int64 (V3) no aporta de forma consistente (11-50 µs con desviaciones iguales o mayores). El microbenchmark cumple el umbral (>= 20 µs, idéntico) solo por poco.
**Motor con la bandera `NPU_RUN_REUSE=1` (implementada y probada; ya revertida, ver abajo), procesos alternados 3 + 3, 48 tokens, ms por token:**

| modelo | | espera NPU | submit | prep | post | paso total |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B [GPU degradada] | reutilización apagada | 88.5 ± 0.1 | 4.4 ± 0.0 | 1.9 ± 0.0 | 4.0 ± 0.0 | 138.3 ± 0.9 |
| | encendida | 89.4 ± 0.1 | 1.2 ± 0.0 | 1.8 ± 0.0 | 3.9 ± 0.0 | 136.6 ± 0.4 |
| Qwen3-1.7B (solo NPU) | apagada | 83.0 ± 0.7 | 4.5 ± 0.3 | 1.7 ± 0.1 | 3.2 ± 0.4 | 99.5 ± 1.9 |
| | encendida | 85.0 ± 1.0 | 1.5 ± 0.1 | 1.8 ± 0.1 | 3.7 ± 0.3 | 99.2 ± 1.6 |

- (1) Logits idénticos con la bandera encendida y apagada en 5 pasos fijos: **sí** (Qwen3-0.6B y Qwen3.5-2B, diferencia 0.0).
- (2) Coste de llamada en el host (submit + prep + post): 10.3 -> 6.9 ms (**-33 %**), pero el paso total solo baja **1.8 ms (1.3 %)** en el 2B. (3) En el 1.7B el paso baja 0.3 ms (**0.3 %, dentro del ruido**): **no se cumple el criterio de paso (>= 2 %)**.
- **Por qué no se traduce en velocidad:** el envío de una matriz se solapa con el DMA de los trozos ya lanzados; al hacerlo más barato, ese tiempo reaparece como espera (+0.9 ms en el 2B, +2.0 ms en el 1.7B). Lo que queda realmente expuesto del coste de llamada es lo que no se puede solapar: preparar la entrada (prep, 1.7-1.9 ms) y recoger y escalar la salida (post, 3.2-4.0 ms): **~5.9 ms (4.3 % del paso) en el 2B y ~4.9 ms (4.9 %) en el 1.7B**. Ese es el techo medido de los "costes fijos por llamada" con este diseño; ninguna de las variantes probadas lo reduce de forma que se note.

**Veredicto:** el criterio se cumple en (1) y en el coste de llamada, y falla en el paso (2B 1.3 %, 1.7B 0.3 %). La bandera `NPU_RUN_REUSE` **se descarta**: el cambio en `src/qwen_npu.py` se revierte (no queda código muerto); el microbenchmark y los números quedan como registro. Fusionar los 26 trozos de la cabeza de salida (necesita xclbins nuevos) no se evalúa: su techo es el mismo prep/post, ya acotado en 4-5 % del paso.
