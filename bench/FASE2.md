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
