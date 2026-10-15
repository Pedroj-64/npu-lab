# Project map: tokens per answer / Mapa de proyecto: tokens por respuesta

Measured 2026-10-15 with `bench/graph_tokens.py` on copies of three real projects (code only, `graphify update`, no model, no network). Token counts are **estimates (4 characters per token)**, not tokenizer counts. / Medido el 2026-10-15 con `bench/graph_tokens.py` sobre copias de tres proyectos reales. Los tokens son **estimaciones (4 caracteres por token)**.

| Project | Files | Nodes | Edges | All parsed source | Median answer (budget 1500) | Median size of the files the answer cites |
|---|---|---|---|---|---|---|
| inklus (Flutter) | 131 | 4,737 | 5,718 | ~338,000 | ~1,245 | ~12,600 |
| Gradify_App (Kotlin) | 145 | 1,839 | 4,831 | ~224,000 | ~1,219 | ~35,300 |
| npu-lab (Python) | 141 | 1,399 | 2,462 | ~257,000 | ~1,258 | ~37,500 |

3 questions per project, budgets 800 / 1500 / 3000 (full table: run the script). / 3 preguntas por proyecto.

## How to read it / Cómo leerlo
- The answer is a **map** (symbol names, files, lines, connections), not code. It says where to look; to edit, the assistant still reads the few files involved. Fair comparison: map answer + the 1–3 files that matter, versus reading every file the question touches. / La respuesta es un **mapa**, no código: dice dónde mirar; para editar hay que leer igualmente los pocos archivos implicados.
- Compared with the whole source the answer is ~99 % smaller; compared with the files it cites it is 67–99 % smaller. The first number is an upper bound, the second is the realistic one. / Frente a todo el código, ~99 % menos; frente a los archivos que cita, 67–99 % menos (la segunda cifra es la realista).
- **Not measured:** whether the answer is the *right* one (relevance). Needs a small question set with known answers before claiming quality. / **No medido:** si la respuesta es la correcta (relevancia).
- Answers are capped at 20,000 characters (~5,000 tokens); budget 3000 hit that cap on 3 of 9 npu-lab/Gradify runs. / El tope de 20 000 caracteres (~5 000 tokens) se alcanzó en varios casos con presupuesto 3000.
- Small projects gain little: for the 6-node test project the answer was larger than the source. / En proyectos pequeños el ahorro es casi nulo.
