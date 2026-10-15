# Energía y eficiencia (Fase 1, 13 oct 2026, rama `perf/medicion`; repetida con la GPU sana el mismo día)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**.

## Método [medido con `bench/energy.py`]
- Un proceso por configuración (la NPU admite uno), modelo cargado, 1 corrida de calentamiento + **3 corridas** medidas; las tablas dan media ± desviación entre esas 3 corridas (dentro de una sesión). Excepción, Qwen3.5-2B con la GPU sana: 2 corridas en un proceso + 1 en otro (dos procesos por reparto: uno con 2 corridas y otro con 1) agrupadas a mano con `n = 3` por configuración.
- Decode: 64 pasos tras un prompt de 16 tokens (contexto corto). Prefill: el prompt se repite hasta sumar ≥6 s (4096 tokens en el perfil Asistido, 512 en NPU-only, ids aleatorios con semilla fija).
- Potencia, sin root: **APU** (CPU + NPU + iGPU; la NPU sola NO tiene sensor) = `power1_input` del hwmon `amdgpu` (PPT, 20 Hz); **GPU NVIDIA** = `nvidia-smi --query-gpu=power.draw.instant` cada 100 ms (precisión documentada ±5 W). Total = APU + GPU. RAPL (`/sys/class/powercap/.../energy_uj`) es solo root [leído: permisos `-r--------`] y la batería marca 0 W en corriente, por eso no se usan. No cuentan pantalla, SSD, RAM fuera del paquete ni pérdidas del cargador.
- tokens/J = tok/s ÷ W totales. "Neto" = tok/s ÷ (W − W en reposo con el modelo cargado, medido 20 s en el mismo proceso): coste marginal de generar. El reposo del escritorio cambia ±2-3 W entre procesos (shell, navegador, visualizador de audio abiertos), así que **el neto es orientativo; el bruto es la cifra principal**.
- Condiciones: enchufado (`ACAD` online), `powerprofilesctl` = `performance` (estaba en `balanced`; lo dejé de nuevo en `balanced` al terminar), governor `performance`, kernel 7.2.9-2-cachyos, `llm_vram_gb` = 4.5 (el de `rules.json`; VRAM máxima vista abajo, siempre ≥1.4 GB libres de la tarjeta de 5.72 GiB).

## ⚠ Condición de la GPU en la primera tanda [medido, GPU degradada]
La GPU no rindió lo documentado: **2.4-2.5 TFLOPS fp16** en un matmul 4096² (el CHANGELOG cita 11.5), estado P5 con **reloj de memoria 780 MHz** y `SW Power Cap` activo con solo ~23 W de 40 W de límite (`nvidia-smi -q`, `nvidia-powerd` activo). Efecto: prefill de Qwen3.5-2B ≈320 tok/s (documentado ~1700) y el reparto del 50 % **no acelera** (6.1 tok/s, el CHANGELOG da 13.8). Repetido con `balanced` y con `performance`: igual (7.1 / 6.2 tok/s a 0 % / 50 %). No pude averiguar la causa sin root **[no verificado]**. Consecuencia: **las filas con GPU de este documento son válidas para esta máquina en este estado y no reproducen el CHANGELOG**; conviene repetir `bench/energy.py` cuando la GPU vuelva a rendir (reinicio, otro cargador, `nvidia-smi -q -d PERFORMANCE`).

## Condición de la GPU en la tanda sana [medido]
Tras reiniciar **con el cargador enchufado** (kernel: `ACAD (on-line)` al arrancar, 13 oct 12:48) la GPU rinde: matmul fp16 4096² = **15.3 TFLOPS**, P0, memoria a 5486 MHz, 40 W. Mismo perfil (`performance`), governor y kernel. La causa raíz de la degradación NO está determinada: arrancar sin cargador deja la GPU en P5 y con cargador queda sana (3 arranques con datos, correlación, sin contraprueba). Las tablas de abajo son de esta tanda; las de la tanda degradada se conservan más abajo.

**Comprobación de la potencia [leído en código + medido]:** en cada fila `total W = APU W + GPU W` (`energy.py`, `measure`; ninguna fila con diferencia > 1e-9) y `tokens/J = tok/s ÷ total W`. La GPU NO se descuenta nunca del total: en el reparto 0 % la GPU sigue encendida con el modelo cargado y su gasto (~11 W en decode, ~10 W en reposo con el modelo cargado frente a ~4.5-6 W sin modelo) está dentro del total. El reposo solo se resta en la columna "net".

## Resultados con la GPU sana [medido, n = 3 corridas por fila; crudos en `bench/energy_runs.jsonl` con `gpu_state`]
| modelo / perfil | fase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B Asistido, reparto 0 % | decode | 9.29 ± 0.17 | 24.9 ± 2.9 | 11.1 ± 0.0 | 36.0 ± 3.0 | 0.26 ± 0.03 | 0.82 ± 0.14 |
| | prefill 4096 | 2069.7 ± 4.2 | 23.5 ± 2.0 | 46.8 ± 1.8 | 70.3 ± 2.0 | 29.5 ± 0.8 | 45.4 ± 3.3 |
| Qwen3.5-2B Asistido, reparto 25 % | decode | 10.95 ± 0.03 | 23.9 ± 0.3 | 13.1 ± 0.2 | 37.1 ± 0.2 | 0.30 ± 0.00 | 0.80 ± 0.01 |
| | prefill 4096 | 2078.1 ± 5.3 | 22.3 ± 1.4 | 46.1 ± 1.7 | 68.3 ± 3.1 | 30.5 ± 1.3 | 46.3 ± 2.9 |
| Qwen3.5-2B Asistido, reparto 50 % | decode | **14.22 ± 0.05** | 24.8 ± 0.5 | 16.0 ± 0.4 | 40.9 ± 0.9 | **0.35 ± 0.01** | 0.80 ± 0.05 |
| | prefill 4096 | 2093.9 ± 5.0 | 22.0 ± 2.3 | 48.0 ± 0.7 | 70.0 ± 1.6 | 29.9 ± 0.7 | 44.7 ± 1.8 |
| Qwen3.5-4B Asistido, reparto 0 % (`--vram-gb 5.0`) | decode | 4.23 ± 0.20 | 25.4 ± 5.0 | 12.0 ± 0.3 | 37.4 ± 4.9 | 0.11 ± 0.02 | 0.33 ± 0.11 |
| | prefill 4096 | 781.7 ± 2.1 | 23.2 ± 2.2 | 48.7 ± 0.3 | 71.8 ± 2.1 | 10.9 ± 0.3 | 16.2 ± 0.7 |
| Qwen3-1.7B Asistido | decode | 10.07 ± 0.01 | 22.8 ± 0.3 | 10.9 ± 0.2 | 33.7 ± 0.4 | 0.30 ± 0.00 | 1.33 ± 0.08 |
| | prefill 4096 | 2402.0 ± 0.6 | 23.9 ± 2.6 | 47.3 ± 0.5 | 71.2 ± 2.7 | 33.8 ± 1.3 | 53.4 ± 3.2 |
| Qwen3-1.7B solo NPU (GPU sin usar) | decode | 9.10 ± 0.04 | 23.2 ± 1.0 | 4.6 ± 0.0 | 27.8 ± 1.0 | 0.33 ± 0.01 | 1.78 ± 0.31 |
| | prefill **512** | 111.6 ± 1.1 | 24.7 ± 1.2 | 4.6 ± 0.0 | 29.3 ± 1.2 | 3.82 ± 0.19 | 16.9 ± 3.0 |

Notas: el 4B solo se midió con reparto 0 % (con la tarjeta de 5.72 GiB los pesos int8 + KV ya ocupan 5057 MiB; un reparto mayor no se probó **[no medido]**); la desviación grande de su APU (±5 W) tiene causa no determinada [no verificado], por eso su 0.11 ± 0.02 tok/J es menos fiable. El solo-NPU lee prompts de 512 tokens y el Asistido de 4096 (no son comparables 1 a 1 en tok/J de prompt). Los W de reposo del escritorio cambian ±3 W entre procesos; el "net" es orientativo.

## Lectura por perfil con la GPU sana [medido salvo lo marcado]
**Qwen3-1.7B (mismo modelo en los dos perfiles)**
- *Decode:* NPU-only 9.10 tok/s a 27.8 W = 0.33 tok/J; Asistido 10.07 tok/s (+11 %) a 33.7 W = 0.30 tok/J. **NPU-only es ~10 % más eficiente** (la GPU solo añade ~6 W y ~1 tok/s); el Asistido es ~11 % más rápido.
- *Lectura de prompt:* NPU-only 112 tok/s; Asistido 2402 tok/s (**21x más rápido**), 33.8 tok/J frente a 3.8 (**~9x más eficiente**, aun gastando 71 W frente a 29 W).
- Conviene: NPU-only para chat corto con prompts de unas decenas de tokens (misma velocidad, menos W, GPU y 4 GB de VRAM libres); Asistido en cuanto el prompt pase de ~100-200 tokens (documentos, historial largo, código pegado) **[umbral estimado, no medido]**.

**Qwen3.5-2B (solo Asistido; la GPU es obligatoria)**
- *Decode:* subir el reparto de GPU **sí acelera y sí ahorra energía por token**: 0 % 9.29 tok/s, 0.26 tok/J; 25 % 10.95 (+18 %), 0.30; 50 % 14.22 (+53 %), 0.35 (+35 %). La potencia sube (36 -> 41 W) menos de lo que sube la velocidad.
- *Prompt (4096):* ~2070-2094 tok/s en los tres repartos (el reparto no cambia la lectura de prompt), ~30 tok/J.
- Conviene: **50 % como valor por defecto del decode** con la GPU sana (más rápido y más eficiente); 0 % solo si necesitas la GPU libre (juegos, otro modelo) o la menor VRAM. Valores mayores del 50 % están por barrer **[propuesto, sin medir]**.

**Qwen3.5-4B:** 4.23 tok/s en decode y 782 tok/s leyendo prompt con reparto 0 %; ~0.11 tok/J. Es el modo "calidad": la mitad de rápido que el 2B en decode y 2.6x más lento leyendo prompt.

## Qué conclusión sale y qué no (GPU sana)
- Con el escritorio abierto el suelo de ~23-26 W domina: generar añade ~13-18 W sobre el reposo en decode, así que acabar antes ahorra más energía que bajar la potencia instantánea.
- En prompts largos la GPU gasta ~47 W pero termina 20x antes: **leer con GPU es siempre más eficiente por token**; la decisión real entre perfiles es la latencia y la VRAM, no los julios.
- La NPU sola no se puede aislar (su aporte está dentro del APU, ~23 W incluyendo CPU). No afirmo W de la NPU.
- No se midió: `balanced`/`power-saver`, batería, Qwen3.5-0.8B, 4B con reparto > 0 %. Una sesión (13 oct) por configuración; el reinicio con cargador es el único cambio entre tandas.

## Segunda sesión sana (bloque 1, 13 oct 13:22-13:44) [medido, n = 3 corridas por fila; crudos en `bench/energy_runs.jsonl`]
**Qwen3.5-4B con reparto de GPU, `--vram-gb 5.0`** (el 0 % viene de la primera sesión sana: 4.23 ± 0.20 tok/s, 0.11 ± 0.02 tok/J, pico 5057 MiB):

| reparto | decode tok/s | total W (GPU W) | tok/J | prefill 4096 tok/s | VRAM pico (nvidia-smi) | hueco libre contra 6144 MiB (sobrestimado, ver corrección) |
|---|---:|---:|---:|---:|---:|---:|
| 0 % | 4.23 ± 0.20 | 37.4 (12.0) | 0.11 ± 0.02 | 781.7 | 5057 MiB | 1087 MiB (cumple >= 1024) |
| 25 % | **4.99 ± 0.00** (+18 %) | 46.2 (17.7) | 0.11 ± 0.00 | 780.4 | **5277 MiB** | 867 MiB (**no cumple**) |
| 50 % | **6.76 ± 0.07** (+60 %) | 63.2 (36.9) | 0.11 ± 0.00 | 782.6 | **5755 MiB** | 389 MiB (**no cumple**; el asignador de CUDA registró un OOM recuperado al pedir 608 MiB con 548 MiB libres) |

**Corrección [13 oct 15:50, medida con el rechazo real de la CLI y con el mensaje de OOM del asignador de CUDA]:** la columna anterior restaba el pico de los 6144 MiB de la tarjeta, pero CUDA solo puede usar **5857 MiB** (el asignador de CUDA lo dice: `total: 6141509632`) y `nvidia-smi` reporta 5830 MiB libres con el escritorio abierto. El hueco real para el escritorio es **5857 − pico = ~800 MiB (0 %), ~580 MiB (25 %) y ~102 MiB (50 %)**, unos 287 MiB menos de lo que decía la tabla (1087 / 867 / 389). Los picos medidos no cambian; cambia lo que significan para el margen.

Lectura: el reparto sube la velocidad del 4B (+18 % y +60 %) pero **no mejora tokens/J** (la GPU gasta lo que ahorra) y **rompe el hueco de 1 GB para el escritorio** desde el 25 %. Por la regla escrita (pico <= 5120 MiB) el 4B se queda en 0 % con 5.0 GB; los repartos 0.40 y 0.30 del plan no se probaron porque el pico crece con el reparto (+220 MiB del 0 al 25 %, +478 más hasta el 50 %) y ya el 25 % se pasa [inferido de 3 puntos, no medido]. Decisión pendiente tuya: 25 % (+18 %, 867 MiB libres) o 0 %. El prefill no cambia con el reparto (782 tok/s), igual que en el 2B.

**Qwen3-1.7B, ¿la diferencia 0.33 frente a 0.30 tok/J es ruido? No** (regla escrita antes: real si las medias de 3 procesos difieren > 2 desviaciones entre procesos y en la misma dirección en los 3 pares). Procesos alternados solo NPU / Asistido, decode, 3 corridas cada uno:

| perfil | tok/s (media de 3 procesos) | total W | GPU W | tok/J por proceso |
|---|---:|---:|---:|---|
| solo NPU | 8.80 ± 0.02 | 31.5 ± 0.2 | 4.6 | 0.279, 0.278, 0.281 |
| Asistido | 9.53 ± 0.08 (+8 %) | 39.2 ± 0.8 | 11.0 | 0.248, 0.246, 0.236 |

Diferencia de tok/J 0.036 frente a 2 x la mayor desviación entre procesos 0.014, y NPU-only gana en los 3 pares: **real, NPU-only ~13 % más eficiente en decode**. Viene de los W (la GPU con el modelo cargado gasta 11.0 W frente a 4.6 W, +6.5 W) y no de tok/s (el Asistido es +8 % más rápido). **Pero el nivel absoluto cambia entre sesiones:** el solo-NPU pasó de 9.10 tok/s a 27.8 W (0.33) a 8.80 tok/s a 31.5 W (0.28) en 40 min (el APU subió de 23.2 a 27.0 W con el mismo estado de energía): los tok/J absolutos de sesiones distintas no se comparan; la razón NPU/Asistido dentro de una sesión (1.10 y 1.15) sí.

**Qwen3.5-2B, barrido 60-75 %:** ver `bench/PROPUESTA_GPU_SANA.md` (75 % = 18.19 tok/s, +34 % sobre el 50 %, tok/J igual).

**Qwen3.5-2B, control de la sesión:** reparto 50 % = 14.13 ± 0.03 tok/s, 0.33 ± 0.01 tok/J (hoy a las 13:27; 14.22 ± 0.05 y 0.35 a las 12:59: misma cifra dentro de ~1 %, W algo mayores).

# Tanda anterior con la GPU degradada (P5, memoria 780 MHz, ~2.5 TFLOPS) [medido, GPU degradada]
Conservada tal cual como referencia; NO describe el rendimiento normal de la máquina.

## Resultados [medido, GPU degradada]
## qwen35_2b assisted share 0%  (performance, AC=1, governor=performance, kernel=7.2.9-2-cachyos, max VRAM seen 3401 MiB)
idle, no model: APU 17.44 W + GPU 4.69 W; idle, model loaded: APU 19.79 W + GPU 6.08 W = 25.87 W
| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 7.08 ± 0.17 | 28.45 ± 1.69 | 9.18 ± 0.16 | 37.63 ± 1.83 | 0.19 ± 0.01 | 0.61 ± 0.10 | 88 |
| prefill 4096 | 321.33 ± 0.51 | 26.90 ± 1.17 | 22.59 ± 0.04 | 49.49 ± 1.13 | 6.49 ± 0.14 | 13.62 ± 0.66 | 115 |
## qwen35_2b assisted share 25%  (performance, AC=1, governor=performance, kernel=7.2.9-2-cachyos, max VRAM seen 3583 MiB)
idle, no model: APU 15.98 W + GPU 5.13 W; idle, model loaded: APU 20.96 W + GPU 5.96 W = 26.92 W
| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 8.92 ± 0.13 | 28.61 ± 0.52 | 19.46 ± 0.09 | 48.07 ± 0.45 | 0.19 ± 0.00 | 0.42 ± 0.02 | 70 |
## qwen35_2b assisted share 50%  (performance, AC=1, governor=performance, kernel=7.2.9-2-cachyos, max VRAM seen 3825 MiB)
idle, no model: APU 18.01 W + GPU 5.12 W; idle, model loaded: APU 23.18 W + GPU 6.37 W = 29.55 W
| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 6.10 ± 0.01 | 29.62 ± 0.22 | 20.64 ± 0.14 | 50.26 ± 0.34 | 0.12 ± 0.00 | 0.29 ± 0.01 | 104 |
## qwen3 npu  (performance, AC=1, governor=performance, kernel=7.2.9-2-cachyos, max VRAM seen 28 MiB)
idle, no model: APU 19.61 W + GPU 5.20 W; idle, model loaded: APU 21.35 W + GPU 4.76 W = 26.11 W
| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 8.91 ± 0.22 | 27.95 ± 2.11 | 4.79 ± 0.02 | 32.74 ± 2.10 | 0.27 ± 0.02 | 1.44 ± 0.44 | 70 |
| prefill 512 | 76.92 ± 2.49 | 28.16 ± 2.01 | 4.76 ± 0.06 | 32.92 ± 2.05 | 2.35 ± 0.22 | 12.17 ± 4.40 | 64 |
## qwen3 assisted  (performance, AC=1, governor=performance, kernel=7.2.9-2-cachyos, max VRAM seen 4243 MiB)
idle, no model: APU 20.25 W + GPU 4.67 W; idle, model loaded: APU 18.47 W + GPU 5.88 W = 24.35 W
| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 8.54 ± 0.10 | 26.17 ± 0.89 | 8.58 ± 0.28 | 34.75 ± 0.81 | 0.25 ± 0.01 | 0.82 ± 0.06 | 74 |
| prefill 4096 | 399.34 ± 0.90 | 26.88 ± 0.67 | 22.67 ± 0.14 | 49.55 ± 0.80 | 8.06 ± 0.13 | 15.86 ± 0.50 | 87 |

Reposo: la máquina con el escritorio abierto gasta ~19-20 W (APU) + ~5 W (GPU) sin hacer nada, 24-30 W con el modelo cargado.

## Lectura por perfil [medido, GPU degradada]
**Qwen3-1.7B (mismo modelo en los dos perfiles)**
- *Decode:* NPU-only 8.91 ± 0.22 tok/s a 32.7 W = 0.27 tok/J; Asistido 8.54 ± 0.10 a 34.8 W = 0.25 tok/J. Prácticamente empate; **NPU-only es ligeramente mejor** (menos W, la GPU solo añade ~4 W de reposo activo).
- *Lectura de prompt:* NPU-only 76.9 tok/s = 2.35 tok/J; Asistido 399 tok/s = 8.06 tok/J. **El Asistido es 5.2x más rápido y 3.4x más eficiente** aunque consume 49.5 W frente a 32.9 W, porque acaba antes.
- Conviene: NPU-only si el contexto es corto y el prompt breve (decode igual de eficiente, GPU libre y sin 4 GB de VRAM); Asistido en cuanto haya que leer más de unos cientos de tokens.

**Qwen3.5-2B (solo existe Asistido; la GPU es obligatoria)**
- *Decode:* 0 %: 7.08 tok/s, 37.6 W, 0.19 tok/J. 25 %: 8.92 tok/s (+26 %), 48.1 W, 0.19 tok/J (misma eficiencia, más rápido). 50 %: 6.10 tok/s, 50.3 W, 0.12 tok/J (**peor en velocidad y en eficiencia** con esta GPU limitada).
- *Lectura de prompt (4096 tokens, reparto 0 %):* 321 tok/s, 49.5 W, 6.5 tok/J.
- Conviene: 25 % si quieres velocidad (+26 % por la misma energía por token); 0 % si quieres la GPU libre/silencio; 50 % solo si la GPU rinde lo documentado **[no verificable ahora]**. En esta máquina la regla "más reparto = más rápido" no se cumple.

## Qué conclusión salía y qué no [medido, GPU degradada]
- Con el escritorio abierto, el suelo de ~25 W domina: generar añade solo ~7-12 W sobre el reposo **[medido]**, así que acabar antes (más tok/s) ahorra más energía que bajar la potencia instantánea ("correr y dormir").
- La NPU sola no se puede aislar: su aporte está dentro de los 28 W del APU (la CPU ya gasta en glue de Python). No afirmo W de la NPU.
- No se midió: perfil `balanced`/`power-saver` completo, batería, Qwen3.5-0.8B/4B, el mismo reparto con la GPU "sana". Una sesión por configuración (no se repitió entre días).

## Reproducir
```
powerprofilesctl set performance      # y enchufado
.venv/bin/python bench/energy.py --model models/qwen35_2b --share 0.25 --phases decode --json salida.jsonl
.venv/bin/python bench/energy.py --model models/qwen3 --engine npu --phases decode,prefill
.venv/bin/python bench/energy.py --model models/qwen3 --engine assisted --phases decode,prefill
```
Crudos de esta tanda: `bench/energy_runs.jsonl`.

Tanda sana: `--model models/qwen35_2b --share {0,0.25,0.5} --runs 3`; `--model models/qwen3 --engine assisted --runs 3`; `--model models/qwen35_4b --share 0 --vram-gb 5.0 --runs 3`; `--model models/qwen3 --engine npu --runs 3`.


## Línea base solo NPU, 15 oct 2026 (tarea T0.1) / NPU-only baseline
Enchufado (AC=1), gobernador y perfil `performance`, kernel 7.2.10-1-cachyos, GPU en reposo (P8, ~4.5 W), NPU libre, 3 corridas por fase [medido]. Comando: `bench/energy.py --model models/<m> --engine npu --runs 3`. Crudos añadidos a `bench/energy_runs.jsonl`.

| Modelo | Fase | tok/s | W totales (APU+GPU) | tokens/J | tokens/J netos |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | decode | 18.49 ± 0.04 | 30.46 ± 2.68 | 0.61 ± 0.05 | 1.71 ± 0.41 |
| Qwen3-0.6B | prefill 512 | 213.03 ± 0.81 | 30.48 ± 0.39 | 6.99 ± 0.06 | 18.92 ± 0.59 |
| Qwen3-1.7B | decode | 9.19 ± 0.01 | 27.24 ± 0.48 | 0.34 ± 0.01 | 1.20 ± 0.08 |
| Qwen3-1.7B | prefill 512 | 99.15 ± 1.70 | 28.18 ± 0.99 | 3.52 ± 0.17 | 11.64 ± 1.57 |

Reposo con el modelo cargado: 19.2 W (0.6B) y 19.6 W (1.7B), de los que ~4.5 W son de la GPU inactiva. Con `bench/watch.py` (perf_baseline, NPU sola): decode 20.49 y 20.42 tok/s, prefill 256 de 240.7 y 241.4 tok/s, TTFT 391 ms. Cifras coherentes con `bench/PERF.md` (decode 19.83, prefill 187 a 256 tokens; el prefill salió ~28 % más rápido que entonces, causa sin investigar). **Conclusión para el plan:** el consumo marginal de la NPU sola es ~11 W sobre el reposo; la comparación contra CPU/GPU para tareas pequeñas queda para T3.1.
