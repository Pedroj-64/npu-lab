# Decisions

Why things are built the way they are, and what was looked at and NOT built. Moved verbatim from the old `ROADMAP.md` (Oct 2026), plus the
Phase 0 decisions at the top. Labels in the old text: measured = in `bench/RESULTS.md` / `bench/PERF.md`; estimate = extrapolated; proposal = does not exist.

## Decisiones del paso 2 de la Fase 2 (13 oct 2026)
- **`NPU_FAST_VOCAB` en Qwen3.5: opcional con aviso, no visible.** Velocidad +18 %, pero el 8.9 % de los tokens de la salida completa no están en el subconjunto y 9 de 10 respuestas cambian; puede inventar nombres propios. Queda tras la bandera (apagada, solo con reparto de GPU 0 %). La caché del subconjunto es propia por tokenizador porque `fast_vocab` solo validaba la longitud.
- **No se reutilizan las ejecuciones de XRT (`NPU_RUN_REUSE`):** el envío ya estaba escondido bajo el DMA; ahorrarlo no acelera el paso (0.3-1.3 %). La sincronización de la salida (`bc.sync`) es necesaria: sin ella los resultados cambian.
- **No se persigue más la reducción de costes fijos de llamada ni del pegamento de CPU:** techo medido 4-5 % y 1 % del paso.
- **Fusión de lanzamientos de la GPU (DeltaNet):** el ahorro posible (13 % en el 2B, 7 % en el 4B) solo puede juzgarse con una GPU sana; queda pendiente de diagnosticar la GPU degradada.

## Corrección de la Fase 2 sobre el DMA (13 oct 2026)
- **El decode de un token está limitado por el DMA de lectura, no por el cálculo** [medido, `bench/DMA_READ.md`]: un diseño de solo lectura con la misma topología tarda lo mismo que el kernel real (r = 0.97-1.03; con 4 columnas y >= 16 MB, r = 1.006) y rinde 6.6 / 12.3 / 21.8 GB/s con 1 / 2 / 4 columnas. La frase "el kernel está a ~1.5x de su suelo de MAC int16, no limitado por DMA" (CHANGELOG, backlog; encargo) era falsa para NV=1: el suelo de MAC (~270 µs por vector con 2 columnas) es ~60 % del tiempo de DMA y solo domina con NV >= 2, que es lo que midió `mv_i8_nv`.
- **No se escribe `mv_i8_i8` para el decode de un token** (criterio r >= 0.85 fijado antes de medir). Sigue en pie "no construir decodificación especulativa sobre este kernel", con un matiz: un núcleo `mmul` int8×int8 quitaría el límite de cálculo con lotes y podría hacerla viable; es una propuesta aparte, pendiente de tu OK.
- Los números de DMA de la Fase 0 (memcpy) y de la Fase 2 (lectura pura) coinciden: no hay "margen de DMA sin aprovechar" que justificar con otro diseño de transferencia.

## Decisiones de la Fase 4 (13 oct 2026)
- **Capa de decisión: `logprob` con el 4B por defecto** (sin VRAM extra, probabilidades de las opciones sin muestrear). `embed` no se adopta (empate en zona gris, IC solapados) pero queda como alternativa cuando solo hay un 2B cargado; el 2B no decide con palabras (sesgo hacia "None").
- **Guard de `DONE` apagado por defecto** (`agent_guard`). El guard objetivo manda cuando hay check; el modelo es señal secundaria y se reporta como curva de τ, sin umbral recomendado (τ=0.678 se calibró con datos sintéticos y no vale con trazas reales; con 1-2 negativos no se puede recalibrar).
- **Laya, NanoJev y Kev: no se descargan ni instalan.** Motivos [reportado por terceros, sin verificar]: Laya multilingüe necesita torch/transformers, contexto de 1024 tokens y sin ONNX int8 verificado del multilingüe; NanoJev solo muestra benchmarks de juegos con 8-10 episodios; Kev exige fusionar el LoRA, convertir a int8 y construir la cabeza "pointer". Jev (propietario, solo API) queda fuera por la regla de privacidad.
- **El agente no es autónomo:** 22/25 con `DONE` falso 0.08 [0.00, 0.20] (n=1); sigue siendo un ayudante que pide confirmación en cada escritura y en `RUN`.
- **Idioma de la documentación (decidido en un paso):** los informes de medición (`bench/*.md`) y los bloques nuevos de cada fase van en español, como pide el encargo; el README y el texto histórico movido del ROADMAP se quedan en inglés (documento público y registro literal). No se traduce el texto movido para no alterar lo que se midió con esas palabras.

## Decisiones de la Fase 0 (13 oct 2026)
- **No se implementa `--fuse-proj`** (ni en decode ni en prefill). Datos: en decode ya están fusionadas qkv y gate|up; las demás matrices dependen de la salida anterior; el coste fijo total de las llamadas es como mucho ~11.9 ms por token y no es fusionable. En prefill el despacho es el 1-2 % del tiempo. Se reabre solo si cambia el motor (por ejemplo un kernel que acepte varias K). Evidencia: `bench/PERF.md` §8.3-§8.4.
- **El motor sigue con un único contexto de 4 columnas y K común:** cambiar de contexto cuesta ~400 µs (medido), cambiar el buffer de instrucciones 1-5 µs.
- **La atención de prefill NPU-only se deja como está por ahora:** el cuello de botella es la atención numpy de un hilo (45-77 % del prefill). La mejora (más hilos) cambia el comportamiento, así que irá tras una bandera apagada por defecto y con equivalencia de tokens comprobada; propuesta registrada en `ROADMAP.md` §2.
- **Idioma:** el texto movido del ROADMAP antiguo se conserva literal en inglés; los bloques nuevos de la Fase 0 van en español. Pendiente de decisión del usuario: traducir todo o mantener el inglés en los documentos públicos del repo.


## Speculative decoding: not on this kernel (old backlog 7 and section 6b)
7. [ ] **Speculative decoding with the model's own MTP head** (Qwen3.5 ships one extra layer for this): the batched-vector kernel measured NV=2 at 1.25x the cost of one token, so ~1.3-1.4x with typical acceptance. Medium effort.
  - [x] **Measured: batched verification is NOT nearly free with this kernel.** `src/kernels/mv_i8_nv.cc` (one weight load feeds NV accumulators; correct against numpy), 4864x896, 2 cores: **NV=1 448 us, NV=2 571 us, NV=4 1069 us** (per vector 448 / 286 / 267 us). Hoisting the activation patterns out of the row loop changed nothing, so the limit is the int16 multiply-accumulate rate (~250 us per vector floor), not memory. [Matiz de la Fase 2: cierto para NV >= 2; con NV=1 el tiempo es el de DMA, ver la corrección arriba.] Consequence: verifying k drafted tokens costs ~1.25x (k=2) to ~2.3x (k=4) one decode step, so even with 60% draft acceptance the gain is ~1.1-1.3x, far below the 1.7-2x estimated below. Do not build speculative decoding on this kernel; a faster inner loop (int8 x int8 activations, or a better mac schedule) would be needed first.

## Not now (old section 4b)
- Not now: **token eviction** (StreamingLLM, H2O, SnapKV: third-party LongBench scores 27 / 33 / 35 vs 43 for full attention, [paper](https://arxiv.org/pdf/2604.19351)); losing a token that is needed later breaks an edit. Last resort. **Prompt compression** (LLMLingua-2: [repo](https://github.com/microsoft/LLMLingua)): gains are reported for RAG text, nothing for code, and removing tokens from code breaks it; try on documents only. **Speculative decoding with MTP**: reported 92/86/71% acceptance on code, but this kernel measured NV=2 at 1.25x the cost of one token (section 6b), so ~1.1-1.3x; revisit after a faster inner loop.

## Autonomous agent: no, a confirm-each-step helper (old section 4d)
- **Verdict:** an autonomous agent is not viable with the 4B as is; an agent that proposes ONE action at a time and waits for the user's confirmation (as the Apply button does) is, because every failure here was a refusal that is visible and harmless, and the wrong outcome is caught when the user looks at the result. The 0.8B and the 2B were not tried (the edit benchmark already shows the 2B misses imports and logic).
- **Second version of the harness (same model, same 5 tasks, one run; changed after seeing the first result):** protocol example with a real path, "read before editing" stated, the real file text returned when a SEARCH does not match, `DONE` refused after a failed action or when a file no longer compiles (`--check` verifies each of these without a model). **Valid calls 18/19 = 94% (was 65%); solved 3/5 (was 4/5).** So the call-format problem is fixed by the harness. What remains is the model's reasoning, with valid actions: task 3 wrote `item.value` (an attribute `Item` does not have; it tried to edit `models.py`, got refused, and did not read it), task 5 renamed the function in `report.py` and the import in `cli.py` but left a call site, then said `DONE`. Both compile, so the compile check cannot catch them; a run of the project's tests would. Task 3 passed in the first run and failed in this one with the same model, so one sample per task is noise-level (+-1 task): do not read 3/5 vs 4/5 as a regression.
- **Verdict (unchanged, now better supported):** valid calls pass the bar (94% >= 80%) but tasks solved do not (3-4 of 5) and a false `DONE` still happens twice in ten tasks: no autonomous agent with the 4B; an agent that proposes one action at a time and waits for the user's confirmation is viable. A false `DONE` is only caught by running something: in a real project, the project's own test command (set by the user, run only after confirmation, output shown) before accepting `DONE`. Not done: the fixture has no tests, so it cannot be measured here.

### 4f. Decision models (Jev / "System One") and int2: looked at, nothing built (Oct 2026, web check only, not measured here)
- **Jev** (TypeSafe AI, released 2026-09-15): a model that returns a choice from a fixed set, a score or a yes/no probability in one pass instead of generating text (routing, triage, reranking, guards). Per the sources it is proprietary, hosted-API only, early-access waitlist, no published weights: it does not fit a project that runs locally. Vendor speed/cost claims are unverified; a paper (JevOut) says added context can flip a correct answer. Open look-alikes exist (SemIf, ex OpenJev), not reviewed.
- **Idea worth trying without Jev:** the agent's weak spot is the false `DONE`. Ask the 4B a closed yes/no question ("is the task finished?") and score it by the logprob of "yes" vs "no" as a guard before accepting `DONE`; a few lines in `src/agent.py`, measured with `bench/agent_eval.py` (noise is +-1 task per run, so 2-3 samples). The same closed-choice trick could pick the model for 3b (model mixer). Not started.
- **JEPA** world models (H-JEPA, Qantara, EB-JEPA) are for robotics/navigation with vision and actions; no source combines them with an LLM. Not useful for the code agent.
- **int2 weights:** exist, but lose a lot: Qwen3-8B post-training 2-bit drops from 68.3 to ~48-54 average (D2Quant); QAT starting from an int4 model recovers part of it and still stays below plain int4. It would need training or calibration plus a new NPU kernel (today only int8 exists, int4 is pending). Not worth it before KV int8 and int4 (4e).

## 8. Models and GPU helpers (decision record)
Labels: **measured** = in `bench/RESULTS.md`; **estimate** = extrapolated, not run; **proposal** = does not exist yet.
Hardware of the test machine: Ryzen 5 8645HS (XDNA1), RTX 3050 6 GB laptop (also drives the desktop, so keep VRAM modest, `gpu_vram_gb` default 1.5), 16 GB shared RAM.

**Measured:** Qwen2.5-0.5B int8 ~15.7 tok/s decode, ~170 tok/s prompt; 1.5B ~6.3 tok/s decode, ~56 tok/s prompt; embeddings ~1100 chunks/s on GPU vs ~20 on NPU.
Decode reaches only ~6-8 GB/s effective, far below DRAM, so the limit is the kernels/DMA, not memory.

**Decision: Qwen2.5-3B is not the default; at most an optional "quality (slow)" profile.**
- [x] **Measured (Qwen2.5-3B, int8, prompt reading token by token)**: **3.58 tok/s** decode (1.5B: 6.62), same 8 prompts. Needed 6 new matvec kernels (all tile without padding) and sharded-safetensors support in `Weights`. The fast prompt reader does not fit: its bf16 copy is ~6 GB and only ~7 GB of RAM were free, so prompt reading stays at ~3 tok/s (a 1000-token RAG prompt would take minutes). Answers: clearly better on instruction following (translated to English where the 1.5B just repeated the Spanish; explained `chmod 755` correctly where the 1.5B got it wrong), equal on facts (Bogota), and both invented the meaning of "NPU". Verdict: usable as the optional "quality (slow)" profile for short questions; not the default. To make it practical, prefill would have to read bf16 weights from disk per layer or use int8 (accuracy check needed).

**Candidate NPU models** (decode tok/s are estimates unless marked):

| Model | Decode | Licence | Role / notes |
|---|---|---|---|
| Qwen2.5-0.5B | 15.7 (measured) | Apache-2.0 | fast mode / draft |
| Qwen3-0.6B | ~13 | Apache-2.0 | newer draft |
| Llama-3.2-1B | ~7-8 | Llama community | fast; intermediate 8192 (clean tiling), no bias (small change), Spanish supported |
| Qwen2.5-1.5B | 6.3 (measured) | Apache-2.0 | current baseline |
| Qwen3-1.7B | ~5.5-6 | Apache-2.0 | best quality/cost; needs QK-norm and no qkv bias in `qwen_npu.py`; use `/no_think` |
| Qwen3-4B | ~2-3 | Apache-2.0 | quality profile, only with speculative decoding |

**Measured (8 prompts, 120 tokens, greedy; same engine):**

| Model | Decode | Prompt reading | Notes |
|---|---|---|---|
| Qwen2.5-1.5B | 6.62 tok/s | 74 tok/s | invents things; fails the translation prompt |
| **Qwen3-1.7B** (ported: QK-norm, no bias, thinking off) | **5.21 tok/s** | **76 tok/s** | coherent, translates and explains `chmod` correctly; as router classifier 83% alone, **100% (24/24) after the rules**, 2.8 s per question |
| Qwen2.5-3B | 3.58 tok/s | ~3 tok/s (no RAM for the fast reader) | best instruction following of the Qwen2.5 family, too slow; classifier 92% at 40 s per question |

Verdict: **Qwen3-1.7B becomes the recommended main model** (about 20% slower decode than the 1.5B for clearly better answers); 1.5B stays as the fast option, 3B as the optional slow profile. The 24-question set is small and the rules were tuned on it, so the 100% is a ceiling, not a promise.

**Which Qwen fits this machine (checked on the Hugging Face model cards, Oct 2026).** Budget: ~21 GB/s of NPU weight streaming (measured), ~7 GB of free RAM, int8 weights resident plus a bf16 copy for fast prompt reading. Estimated tok/s = 21 GB/s over the int8 bytes read per token + ~25 ms of host work (the formula reproduces the measured 9.3 tok/s of Qwen3-1.7B):

| Model | int8 GB/token | est. tok/s | bf16 prefill copy | Fits? |
|---|---|---|---|---|
| Qwen3-0.6B | 0.60 | ~19 | 0.9 GB | yes, works today (fast mode) |
| **Qwen3-1.7B** | 1.72 | **9.3 (measured)** | 2.8 GB | **yes, works today (main)** |
| Qwen3.5-0.8B | 0.75 | ~16 | 1.0 GB | needs port |
| **Qwen3.5-2B** | 1.88 | ~8.7 | 2.7 GB | needs port; best target |
| Qwen3.5-4B / Qwen3-4B | 4.2 / 4.0 | ~4.5 | 7.1-7.3 GB | RAM only without fast prompt reading |
| Qwen3.5-9B, 27B, MoE (35B-A3B...), Coder-Next 80B | 8+ | <2.5 | >13 GB | no |

Qwen3.5 (Feb-Mar 2026, Apache-2.0, 201 languages, 262k native context) is a hybrid: 3 Gated DeltaNet (linear attention) layers per 1 gated-attention layer, head_dim 256, partial RoPE (25%), output gate on attention, vocabulary 248k (the output head is 27% of the bytes of the 2B), a built-in multi-token-prediction head and thinking on by default (`enable_thinking=false` through the chat template). Why the 2B is the target: same speed class as Qwen3-1.7B, newer and stronger, and the hybrid design makes long context cheap: only 6 of its 24 layers keep a KV cache (~12 KB/token, i.e. ~0.2 GB at 16k and ~1.5 GB at 128k in fp16) while the DeltaNet layers keep a fixed-size state. That directly answers the long-context need above. Cost: the engine needs a new Gated DeltaNet layer (short conv + delta-rule recurrence + gating, in numpy on the CPU; its projections still run on the NPU), gated attention, partial RoPE and the bigger vocabulary; validate against the reference implementation (needs `transformers` installed for a numerical check). The 4B is the later "quality" profile once prompt reading no longer needs a bf16 copy in RAM.

Leaning (earlier note): main = Qwen3-1.7B (or keep 1.5B at zero cost); fast = Llama-3.2-1B or 0.5B.

**GPU as a helper** (small VRAM, short bursts), by value per VRAM:
  Risk: only pays off if the bf16 prompt kernels are efficient at tiny M (4-6 tokens); benchmark that first.
- [x] Embeddings for big indexes on GPU (`--device gpu`).
- Not now: prompt on GPU + decode on NPU (moving the KV cache between backends costs too much to integrate).

**Order of work:**
1. Reranker on GPU; measure prompt-size reduction and time to first token.
2. Port Qwen3-1.7B or Llama-3.2-1B; compare with Qwen2.5-1.5B on ~20 real Arch/paru/pacman questions, with and without docs.
3. Benchmark the verification kernel at small M to decide if speculative decoding is viable.
4. Check int8 does not hurt quality (cosine or perplexity vs bf16).
5. If 3B/4B clearly helps, add it as the "quality (slow)" profile in the resource selector, never as a replacement.

## Principles
1. Decide per question; show the decision; let the user override it.
2. Never inject text that is not clearly relevant; "no source" is a valid result.
3. Everything private stays local; anything that leaves the PC is opt-in, per message and labelled.
4. Measure before and after, keep the numbers in `bench/RESULTS.md`.

