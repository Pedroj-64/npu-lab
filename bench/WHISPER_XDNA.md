# Whisper encoder on the first-generation NPU: reproduced here / reproducido aquí (2026-10-15)

Project reproduced: [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna) (MIT). Machine: Ryzen 5 8645HS (XDNA1), plugged in, `performance` profile, GPU hidden (`CUDA_VISIBLE_DEVICES=`), NPU alone. 30-second window (1500 frames), the JFK clip. Tooling: `bench/power_of_command.py` (whole-machine power: APU package + NVIDIA board, idle measured just before). Raw logs stayed in a temporary folder; every figure below is from those runs.

## Deviations from the upstream recipe (declared)
No Docker (the daemon is off here): the project's own code ran in this repo's venv (Python 3.14, mlir-aie **1.4.4**, upstream pins 1.3.4/Python 3.13), with `MLIR_AIE_DIR` pointed at a shim and a stub `numba` (openai-whisper imports it only for word timestamps). **The on-NPU GELU/softmax kernels did not compile** on 1.4.4 (`TensorTiler2D` no longer exists in `aie.helpers.taplib`), so this is the upstream "GEMMs on the NPU, GELU/softmax/layer-norm in numpy" configuration (all GEMMs bf16 with fp32 accumulation on the AIE array). The upstream large-v3 numbers (on-NPU row ops) were NOT reproduced.

## Correctness (matches the upstream claim)
| Model | Layers x width | Cosine vs torch fp32 | Transcription (JFK) |
|---|---|---:|---|
| tiny | 4 x 384 | 0.99988 | MATCH, character for character |
| base | 6 x 512 | 0.99994 | MATCH |
| small | 12 x 768 | 0.99994 | MATCH |

## Speed and energy (encoder only, 3 to 5 timed passes; energy above idle, mean over many passes)
| Model | NPU path wall | CPU torch fp32, 6 threads | NPU / CPU time | NPU energy per pass | CPU energy per pass | NPU / CPU energy | cores busy |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny | 432 ms | 85 ms | 5.1x slower | 5.1 J | 1.9 J | 2.6x | 0.8 vs 6 |
| base | 836 ms | 205 ms | 4.1x slower | not measured | not measured | – | 0.8 vs 6 |
| small | 2264 ms | 775 ms | 2.9x slower | 29.1 J | 16.4 J | 1.8x | 0.8 vs 6 |

## Reading
- It **works and is correct**, and real-time capable (small: the 30 s window encodes in 2.3 s, 13x real time), but **on this machine it is slower and uses more energy than the CPU** at these sizes. The gap narrows as the model grows (5.1x -> 2.9x slower; 2.6x -> 1.8x energy), which fits upstream's claim that the NPU only pays off at large-v3 size, but that is a trend, not a measurement of large-v3.
- About 55 percent of the NPU path's time is host numpy (softmax + GELU, ~620 + ~560 ms of 2264 at small). Getting the on-NPU row kernels to build on mlir-aie 1.4.4 is the most promising next step.
- What the NPU does give: it leaves 5 to 6 CPU cores free (0.8 cores busy against all 6).
- Plan criterion T4.1 ("at least 3x real time AND fewer watts than the CPU"): real time **met**, watts **not met** at tiny/small. Decision: not integrated into npu-lab; see `docs/DECISIONS.md`.
