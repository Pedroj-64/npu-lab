# Viabilidad de int4 simulada en CPU (Fase 2, paso 3, 13 oct 2026)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**. Todo en CPU con `torch`/`transformers` ya instalados, sin tocar NPU ni GPU; pesos solo de `models/` (ya descargados). Los cálculos son deterministas: se reportan una vez (el ruido de reloj no entra en calidad).

## 0. Criterio y protocolo fijados ANTES de medir (commit previo a los resultados)
**Qué se simula.** Cada matriz que hoy va a la NPU (todas las `nn.Linear` de las capas y la cabeza de salida) se cuantiza y se descuantiza ("cuantización falsa") a bf16, que es lo que ya hace el camino de la GPU con int8. La tabla de embeddings de entrada no se cuantiza (se lee por filas en la CPU, no se transmite por token). Referencia de producción: **int8 por fila simétrico** (como `qmat` de `src/qwen_npu.py`). Referencia sin cuantizar: bf16.
**Barrido.** Cuantización por grupos a lo largo de la dimensión de entrada K, escala por grupo: tamaño de grupo **32, 64 y 128**; **simétrica** (rango -8..7, escala = max|w|/7) y **asimétrica** (escala = (max-min)/15, cero por grupo). **Variante A:** todo en int4, cabeza de salida incluida. **Variante B:** cabeza de salida en int8 por fila, el resto en int4. 12 configuraciones + int8 + bf16.
**Modelos:** Qwen3-1.7B y Qwen3.5-2B. El 4B no se carga: bf16 son ~8.8 GB y la RAM disponible (≈8.9 GB con 2.4 GB de swap ya ocupados por el sistema) no alcanza sin tocar el swap.
**Texto fijo para coseno y perplejidad:** dos segmentos de 384 tokens: los primeros caracteres de `README.md` (inglés) y de `bench/PERF.md` (español técnico), tal como están en este commit; perplejidad por token sobre el texto con la teacher forcing. **Prompts fijos** para la comparación de tokens: los 5 de `bench/token_equiv.py` con la plantilla de chat (sin pensamiento), decodificación voraz de 32 tokens, comparados con la salida de int8.
**Criterio de calidad [propuesto por el usuario, adoptado tal cual]:** una configuración PASA si cumple las tres: (1) **coseno medio de logits >= 0.99** frente a bf16 (media sobre todas las posiciones de los dos segmentos); (2) **perplejidad <= +3 % relativo frente a la perplejidad de int8** en el mismo texto; (3) **tokens idénticos a los de int8 en >= 4 de 5 prompts** (o diferencia cuantificada: primer token distinto y coincidencia por token). (3) solo se evalúa en las configuraciones que pasan (1) y (2) y en int8 como control (coste de generar en CPU).
**Bytes por token (incluidas escalas).** int8 por fila: 1 byte por peso + 4 bytes de escala por fila. int4 por grupos: 0.5 byte por peso + 2 bytes (escala fp16) por grupo (+ 0.5 byte por grupo del cero si es asimétrica). Se suman por todas las matrices que se transmiten por token (las de las capas y la cabeza). El techo por DMA se recalcula con **21.8 GB/s** medidos (4 columnas, `bench/DMA_READ.md`): tok/s alcanzable **solo por bytes** [estimado] frente al límite que pondría el cálculo (~270 µs por vector con 2 columnas, ~135 µs con 4) [estimado], y qué pasa con el desempaquetado int4 -> int8 en el núcleo (sin medirlo).
**Si pasa:** no se escribe el kernel; como mucho se estima en ciclos desde las cabeceras de `aie_api` el desempaquetado dentro de un tile y se deja como propuesta con criterio de lote <= 1.3x. **Si no pasa:** se documenta qué variante se acercó más y qué falló, y se cierra la vía int4.
**Límites declarados:** la simulación mide calidad, no velocidad; texto de 768 tokens y 5 prompts son una muestra corta; la cuantización por grupos no calibra (sin GPTQ/AWQ): es el peor caso razonable de int4 y el techo de calidad podría ser mejor con calibración; el 4B no se evalúa.


## 1. Resultados [medido, CPU, 13 oct 2026; deterministas: una sola evaluación por configuración]
`bench/int4_sim.py` (crudos en `bench/int4_sim_qwen3.jsonl` y `bench/int4_sim_qwen35_2b.jsonl`). Coseno: media sobre las 766 posiciones frente al modelo bf16 sin cuantizar (entre paréntesis, el peor). Perplejidad sobre los dos segmentos de 384 tokens (README en inglés, PERF.md en español). "MB/token" incluye escalas (int8 por fila: 4 bytes por fila; int4: escala fp16 por grupo, y 0.5 byte por grupo del cero si es asimétrica). Comprobación: int8 da **1723 MB/token en el 1.7B y 1885 MB/token en el 2B**, igual que los bytes que cuenta el motor (1720 y 1885, `bench/DMA_READ.md` §3).

### Qwen3-1.7B
| config | MB/token | coseno medio (mínimo) | perplejidad | frente a int8 | pasa |
|---|---:|---:|---:|---:|---|
| bf16 | 3441 | 1.0000 (1.00) | 38.92 | +0.8 % | — |
| int8 | 1723 | 0.9991 (0.97) | 38.63 | +0.0 % | — |
| A_g32_sym | 968 | 0.9624 (0.06) | 46.66 | +20.8 % | no |
| A_g32_asym | 995 | 0.9773 (0.39) | 40.97 | +6.1 % | no |
| A_g64_sym | 914 | 0.9525 (-0.22) | 49.34 | +27.7 % | no |
| A_g64_asym | 927 | 0.9666 (0.13) | 44.30 | +14.7 % | no |
| A_g128_sym | 887 | 0.9437 (-0.18) | 50.69 | +31.2 % | no |
| A_g128_asym | 894 | 0.9595 (-0.00) | 45.88 | +18.8 % | no |
| B_g32_sym | 1104 | 0.9661 (0.06) | 44.82 | +16.0 % | no |
| B_g32_asym | 1127 | 0.9800 (0.39) | 40.40 | +4.6 % | no |
| B_g64_sym | 1060 | 0.9569 (-0.22) | 46.59 | +20.6 % | no |
| B_g64_asym | 1071 | 0.9698 (0.13) | 42.63 | +10.4 % | no |
| B_g128_sym | 1038 | 0.9487 (-0.18) | 47.56 | +23.1 % | no |
| B_g128_asym | 1044 | 0.9633 (-0.00) | 44.16 | +14.3 % | no |

### Qwen3.5-2B
| config | MB/token | coseno medio (mínimo) | perplejidad | frente a int8 | pasa |
|---|---:|---:|---:|---:|---|
| bf16 | 3763 | 1.0000 (1.00) | 27.04 | +0.6 % | — |
| int8 | 1885 | 0.9994 (0.99) | 26.88 | +0.0 % | — |
| A_g32_sym | 1058 | 0.9607 (0.84) | 32.15 | +19.6 % | no |
| A_g32_asym | 1088 | 0.9735 (0.77) | 30.30 | +12.7 % | no |
| A_g64_sym | 999 | 0.9475 (0.68) | 32.72 | +21.7 % | no |
| A_g64_asym | 1014 | 0.9656 (0.86) | 30.72 | +14.3 % | no |
| A_g128_sym | 970 | 0.9343 (0.74) | 34.51 | +28.4 % | no |
| A_g128_asym | 977 | 0.9572 (0.78) | 30.36 | +13.0 % | no |
| B_g32_sym | 1282 | 0.9657 (0.85) | 31.07 | +15.6 % | no |
| B_g32_asym | 1303 | 0.9768 (0.77) | 29.47 | +9.6 % | no |
| B_g64_sym | 1239 | 0.9538 (0.69) | 31.43 | +16.9 % | no |
| B_g64_asym | 1250 | 0.9700 (0.86) | 30.17 | +12.2 % | no |
| B_g128_sym | 1217 | 0.9421 (0.75) | 33.22 | +23.6 % | no |
| B_g128_asym | 1223 | 0.9627 (0.79) | 29.97 | +11.5 % | no |

**Criterio (§0): ninguna de las 24 configuraciones lo cumple.** La más cercana en los dos modelos es **B_g32_asym** (cabeza en int8, resto en int4 por grupos de 32 asimétrico): 1.7B coseno 0.9800 y perplejidad +4.6 % (le falta 0.010 de coseno y 1.6 puntos de perplejidad); 2B coseno 0.9768 y +9.6 %. Todas las simétricas y todos los grupos de 64 y 128 quedan peor; la cuantización asimétrica gana siempre a la simétrica, y el grupo de 32 al de 128. La variante A (todo int4) es peor que la B en 12 de 12 comparaciones, pero por poco (la cabeza pesa menos en el 1.7B).
**(3) Tokens a temperatura 0 [medido, 5 prompts, 32 tokens], la más cercana frente a int8:** idénticos en **1 de 5** prompts en el 1.7B y en **1 de 5** en el 2B (primer token distinto: 2, 2, 5 y 0 en el 1.7B; 0, 7, 2 y 0 en el 2B; coincidencia por token 0.00-0.25 en los textos que divergen). Falla también.

## 2. Bytes por token y techo por DMA [estimado a partir de los bytes medidos y de los 21.8 GB/s de `bench/DMA_READ.md`]
Suelo por DMA = MB/token / 21.8 GB/s. Si solo se encoge la espera de la NPU y el resto del paso se mantiene (medido en `bench/FASE2.md`, lote A):

| modelo | config | MB/token | suelo DMA ms | paso ms | tok/s solo por bytes |
|---|---|---:|---:|---:|---:|
| 1.7B (solo NPU) | int8 (hoy) | 1723 | 79.1 | 100.4 | 9.96 |
| | B_g32_asym | 1127 | 51.7 | 71.5 | 13.98 |
| | A_g32_asym | 995 | 45.6 | 65.1 | 15.35 |
| | A_g128_sym | 887 | 40.7 | 59.9 | 16.69 |
| 2B [GPU degradada] | int8 (hoy) | 1885 | 86.4 | 140.6 | 7.11 |
| | B_g32_asym | 1303 | 59.8 | 112.9 | 8.86 |
| | A_g32_asym | 1088 | 49.9 | 102.6 | 9.74 |
| | A_g128_sym | 970 | 44.5 | 97.0 | 10.31 |

**Límite que pondría el cálculo [estimado]:** el suelo de cálculo del kernel actual sale de ~270 µs por vector para 4.36 M MAC con 2 columnas (`bench/DMA_READ.md` §2) = ~32 MAC/ns con 4 columnas, es decir ~53 ms por token para los 1.72 G de pesos del 1.7B y ~58 ms para el 2B. Con int4 el suelo de DMA (41-60 ms) queda **igual o por debajo del suelo de cálculo (53-58 ms) más el coste de desempaquetar int4 a int8, que no he medido**. O sea, aunque la calidad pasara, el beneficio real quedaría por debajo de la columna "tok/s solo por bytes" y haría falta el núcleo `mmul` int8×int8 (lo que `mv_i8_i8` habilitaría) para no pasar del cuello de DMA al de cálculo.

## 3. Resultado y límites
**Resultado: int4 por grupos sin calibración NO es viable** con este criterio: ni en perplejidad (+4.6 % la mejor en el 1.7B, +9.6 % en el 2B), ni en coseno (0.980 y 0.977 frente a 0.99), ni en tokens (1 de 5). Se cierra la vía "int4 por redondeo al más cercano". El ahorro de bytes que prometía (hasta 1.4-1.7x en el paso) no se materializa.
**Límites:** (a) es cuantización por redondeo sin calibración (RTN), el peor caso razonable; los métodos con calibración (GPTQ, AWQ) suelen perder menos pero no se han medido aquí **[reportado por terceros, sin verificar]** y necesitan datos de calibración y mucho más cálculo en CPU; (b) el texto son 768 tokens y 5 prompts: una muestra corta; (c) solo se probó una escala por grupo (fp16) y el desempaquetado no se midió; (d) el 4B no se evaluó (RAM); (e) mide calidad, no velocidad.
**Aviso de memoria:** durante las evaluaciones el swap del sistema subió de 2.2 a ~4.0 GB (sin escrituras sostenidas, 9-10 GB de RAM disponibles); es el sistema empujando aplicaciones inactivas, pero queda anotado.

## 4. Barrido por tipo de matriz (13 oct 2026) — criterio fijado ANTES de medir (commit previo a los resultados)
**Idea.** El paso 3 cuantizó todas las matrices igual y falló. Pero no todas pesan lo mismo ni son igual de sensibles. Se prueba int4 solo en algunas clases y el resto en int8 por fila (como hoy).
**Clases** (por nombre de la `nn.Linear`): `GU` = gate_proj + up_proj del MLP; `DOWN` = down_proj; `ATT` = todas las proyecciones de atención o DeltaNet (q, k, v, o; in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj); `HEAD` = cabeza de salida. Partes aproximadas del byte-por-token del 1.7B: GU 38 %, DOWN 19 %, ATT 19 %, HEAD 18 % (resto: ninguno; las fracciones se calculan y se reportan por modelo).
**Combinaciones:** las 4 clases sueltas; los pares GU+DOWN, GU+ATT, DOWN+ATT; y GU+DOWN+ATT+HEAD ya medidas (variante A) y GU+DOWN+ATT (variante B) quedan como referencia. Cuantización **asimétrica** (la simétrica fue peor en 12 de 12 comparaciones del paso 3) y grupos de **32, 64 y 128**. Total: 7 combinaciones x 3 grupos = 21 configuraciones por modelo, en Qwen3-1.7B y Qwen3.5-2B. Mismo texto fijo y mismos 5 prompts que el paso 3.
**Criterio (mismo umbral que el paso 3, más una condición de utilidad):** una combinación PASA si cumple (1) coseno medio de logits >= 0.99 frente a bf16; (2) perplejidad <= +3 % frente a int8; (3) texto idéntico al de int8 en >= 4 de 5 prompts (solo se evalúa en las que pasan (1) y (2)); **y** (4) **ahorra al menos un 15 % de bytes por token frente a int8** (sin ahorro apreciable no vale la pena el cambio). Se exige que pase en **los dos modelos**; si pasa solo en uno, se reporta pero no se cierra como viable.
**Tope:** si ninguna combinación pasa, la vía "int4 sin calibración" se cierra definitivamente y GPTQ/AWQ queda solo anotado en el ROADMAP. Si alguna pasa, **no se escribe ningún kernel**: se documenta (y el límite de cálculo del paso 3 sigue aplicando: `mv_i8_i8` en espera).

### Resultados del barrido por clase [medido, CPU, determinista; `bench/int4_sim.py sweep --only combos`, crudos en los mismos `.jsonl`]
Cuotas de bytes por token con todo en int8 (calculadas, no supuestas): **1.7B** GU 41 %, ATT 20 %, DOWN 20 %, HEAD 18 % (1723 MB); **2B** GU 32 %, ATT 25 %, DOWN 16 %, HEAD 27 % (1885 MB). Regresión: la combinación GU+DOWN+ATT en g32 reproduce exactamente la variante B del paso 3 (perplejidad 40.397, coseno 0.98001, 1127 MB/token). Todas con cuantización asimétrica. Perplejidad y ahorro frente a int8 (ppl 38.634 en el 1.7B, 26.878 en el 2B).

| clases en int4 | grupo | 1.7B: coseno | ppl | ahorro | 2B: coseno | ppl | ahorro | cumple (1)(2)(4) en 1.7B / 2B |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GU | 32 | 0.9913 | +3.0 % | 17.3 % | 0.9937 | +1.1 % | 13.6 % | no / no |
| GU | 64 | 0.9904 | +3.6 % | 18.9 % | 0.9918 | +2.9 % | 14.8 % | no / no |
| GU | 128 | 0.9878 | +7.5 % | 19.7 % | 0.9903 | +1.7 % | 15.5 % | no / sí |
| DOWN | 32 | 0.9921 | +0.9 % | 8.6 % | 0.9945 | +2.6 % | 6.8 % | no / no |
| DOWN | 64 | 0.9892 | +2.7 % | 9.4 % | 0.9930 | +2.1 % | 7.4 % | no / no |
| DOWN | 128 | 0.9871 | +4.9 % | 9.8 % | 0.9917 | +1.0 % | 7.7 % | no / no |
| ATT | 32 | 0.9914 | +0.3 % | 8.7 % | 0.9872 | +6.4 % | 10.5 % | no / no |
| ATT | 64 | 0.9870 | +1.3 % | 9.5 % | 0.9838 | +6.6 % | 11.5 % | no / no |
| ATT | 128 | 0.9839 | +0.3 % | 9.9 % | 0.9797 | +6.3 % | 11.9 % | no / no |
| HEAD | 32 | 0.9963 | +1.6 % | 7.7 % | 0.9960 | +2.9 % | 11.4 % | no / no |
| HEAD | 64 | 0.9956 | +4.4 % | 8.4 % | 0.9949 | +1.5 % | 12.5 % | no / no |
| HEAD | 128 | 0.9948 | +3.4 % | 8.7 % | 0.9937 | +2.1 % | 13.0 % | no / no |
| GU+DOWN | 32 | 0.9860 | +3.7 % | 26.0 % | 0.9888 | +3.6 % | 20.4 % | no / no |
| GU+DOWN | 64 | 0.9816 | +7.4 % | 28.4 % | 0.9852 | +5.2 % | 22.2 % | no / no |
| GU+DOWN | 128 | 0.9788 | +12.2 % | 29.6 % | 0.9824 | +2.7 % | 23.2 % | no / no |
| GU+ATT | 32 | 0.9865 | +2.6 % | 26.0 % | 0.9812 | +7.1 % | 24.1 % | no / no |
| GU+ATT | 64 | 0.9808 | +6.5 % | 28.4 % | 0.9764 | +9.3 % | 26.3 % | no / no |
| GU+ATT | 128 | 0.9743 | +7.4 % | 29.6 % | 0.9706 | +8.4 % | 27.4 % | no / no |
| DOWN+ATT | 32 | 0.9842 | +0.7 % | 17.3 % | 0.9826 | +9.1 % | 17.3 % | no / no |
| DOWN+ATT | 64 | 0.9755 | +5.6 % | 18.9 % | 0.9781 | +9.1 % | 18.9 % | no / no |
| DOWN+ATT | 128 | 0.9708 | +6.8 % | 19.7 % | 0.9722 | +8.5 % | 19.7 % | no / no |

**Cuenta de combinaciones que cumplen (1) coseno >= 0.99, (2) perplejidad <= +3 % y (4) ahorro >= 15 %:** en el 1.7B, **0 de 21**; en el 2B, **1 de 21** (solo gate+up en int4, grupo 128: coseno 0.9903, +1.7 %, ahorro 15.5 %); en los dos modelos a la vez, **0 de 21**. El criterio (3) (tokens) no llegó a evaluarse porque nada pasa en ambos modelos.

**Casi-aciertos, anotados con sus cifras exactas (no se reinterpreta el umbral):**
- **Solo gate+up en int4, grupo 32:** 1.7B coseno 0.9913, ahorro 17.3 %, perplejidad **+3.02 %** (falla el 3 % por 0.02 puntos); 2B coseno 0.9937, perplejidad +1.1 %, ahorro **13.6 %** (falla el 15 % por 1.4 puntos). Con grupo 64: 1.7B +3.6 %, 2B ahorro 14.8 %.
- **Solo `down` o solo `ATT` o solo la cabeza** en int4 pasan calidad en el 1.7B (grupo 32) pero ahorran 8-9 %, y en el 2B `ATT` falla calidad (+6.3 % a +6.6 %).
- **Cualquier par** (el ahorro llega a 20-30 %) falla calidad en el 2B y en casi todos los del 1.7B (el mejor par: GU+ATT g32, 1.7B +2.6 % pero coseno 0.9865).
- La sensibilidad por clase difiere entre modelos: la atención (`ATT`) es lo más sensible en el 2B (DeltaNet/atención híbrida), mientras que en el 1.7B lo más sensible es gate+up en grupos grandes.

**Resultado: ninguna combinación pasa en los dos modelos -> por el tope fijado, la vía "int4 sin calibración" queda CERRADA.** El máximo que ofrece sin romper la calidad según el criterio es ~15-17 % menos de bytes (solo gate+up), que daría como mucho ~1.1x en el paso [estimado] y que además tropezaría con el límite del cálculo (paso 3). GPTQ/AWQ queda solo anotado en el ROADMAP.
