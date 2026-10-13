# Energía y eficiencia (Fase 1, 13 oct 2026, rama `perf/medicion`)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**.

## Método [medido con `bench/energy.py`]
- Un proceso por configuración (la NPU admite uno), modelo cargado, 1 corrida de calentamiento + **3 corridas** medidas; las tablas dan media ± desviación entre esas 3 corridas (dentro de una sesión).
- Decode: 64 pasos tras un prompt de 16 tokens (contexto corto). Prefill: el prompt se repite hasta sumar ≥6 s (4096 tokens en el perfil Asistido, 512 en NPU-only, ids aleatorios con semilla fija).
- Potencia, sin root: **APU** (CPU + NPU + iGPU; la NPU sola NO tiene sensor) = `power1_input` del hwmon `amdgpu` (PPT, 20 Hz); **GPU NVIDIA** = `nvidia-smi --query-gpu=power.draw.instant` cada 100 ms (precisión documentada ±5 W). Total = APU + GPU. RAPL (`/sys/class/powercap/.../energy_uj`) es solo root [leído: permisos `-r--------`] y la batería marca 0 W en corriente, por eso no se usan. No cuentan pantalla, SSD, RAM fuera del paquete ni pérdidas del cargador.
- tokens/J = tok/s ÷ W totales. "Neto" = tok/s ÷ (W − W en reposo con el modelo cargado, medido 20 s en el mismo proceso): coste marginal de generar. El reposo del escritorio cambia ±2-3 W entre procesos (shell, navegador, visualizador de audio abiertos), así que **el neto es orientativo; el bruto es la cifra principal**.
- Condiciones: enchufado (`ACAD` online), `powerprofilesctl` = `performance` (estaba en `balanced`; lo dejé de nuevo en `balanced` al terminar), governor `performance`, kernel 7.2.9-2-cachyos, `llm_vram_gb` = 4.5 (el de `rules.json`; VRAM máxima vista abajo, siempre ≥1.4 GB libres de la tarjeta de 5.72 GiB).

## ⚠ Condición de la GPU durante esta medición [medido]
La GPU no rindió lo documentado: **2.4-2.5 TFLOPS fp16** en un matmul 4096² (el CHANGELOG cita 11.5), estado P5 con **reloj de memoria 780 MHz** y `SW Power Cap` activo con solo ~23 W de 40 W de límite (`nvidia-smi -q`, `nvidia-powerd` activo). Efecto: prefill de Qwen3.5-2B ≈320 tok/s (documentado ~1700) y el reparto del 50 % **no acelera** (6.1 tok/s, el CHANGELOG da 13.8). Repetido con `balanced` y con `performance`: igual (7.1 / 6.2 tok/s a 0 % / 50 %). No pude averiguar la causa sin root **[no verificado]**. Consecuencia: **las filas con GPU de este documento son válidas para esta máquina en este estado y no reproducen el CHANGELOG**; conviene repetir `bench/energy.py` cuando la GPU vuelva a rendir (reinicio, otro cargador, `nvidia-smi -q -d PERFORMANCE`).

## Resultados [medido]
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

## Lectura por perfil [medido salvo lo marcado]
**Qwen3-1.7B (mismo modelo en los dos perfiles)**
- *Decode:* NPU-only 8.91 ± 0.22 tok/s a 32.7 W = 0.27 tok/J; Asistido 8.54 ± 0.10 a 34.8 W = 0.25 tok/J. Prácticamente empate; **NPU-only es ligeramente mejor** (menos W, la GPU solo añade ~4 W de reposo activo).
- *Lectura de prompt:* NPU-only 76.9 tok/s = 2.35 tok/J; Asistido 399 tok/s = 8.06 tok/J. **El Asistido es 5.2x más rápido y 3.4x más eficiente** aunque consume 49.5 W frente a 32.9 W, porque acaba antes.
- Conviene: NPU-only si el contexto es corto y el prompt breve (decode igual de eficiente, GPU libre y sin 4 GB de VRAM); Asistido en cuanto haya que leer más de unos cientos de tokens.

**Qwen3.5-2B (solo existe Asistido; la GPU es obligatoria)**
- *Decode:* 0 %: 7.08 tok/s, 37.6 W, 0.19 tok/J. 25 %: 8.92 tok/s (+26 %), 48.1 W, 0.19 tok/J (misma eficiencia, más rápido). 50 %: 6.10 tok/s, 50.3 W, 0.12 tok/J (**peor en velocidad y en eficiencia** con esta GPU limitada).
- *Lectura de prompt (4096 tokens, reparto 0 %):* 321 tok/s, 49.5 W, 6.5 tok/J.
- Conviene: 25 % si quieres velocidad (+26 % por la misma energía por token); 0 % si quieres la GPU libre/silencio; 50 % solo si la GPU rinde lo documentado **[no verificable ahora]**. En esta máquina la regla "más reparto = más rápido" no se cumple.

## Qué conclusión sale y qué no
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
