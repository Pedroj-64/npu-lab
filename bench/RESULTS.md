# Matmul, Ryzen 5 8645HS (RyzenAI-npu1, Hawk Point), XRT 2.21.75, mlir-aie 1.4.4

mlir-aie `whole_array` example, 4 columns, tiles 32x32x32, i16 in / i32 out,
run with `bench/matmul_npu.py`. CPU column: numpy float32 `A @ B` (all cores).
Ops counted as 2*M*K*N; NPU is integer ops, CPU is float ops (not the same unit).

| M=K=N | NPU ms | NPU GOPS | CPU ms | CPU GFLOPS |
|------:|-------:|---------:|-------:|-----------:|
| 512   | 0.287  | 935      | 0.651  | 413        |
| 1024  | 1.496  | 1435     | 4.631  | 464        |
| 2048  | 34.312 | 501      | 46.427 | 370        |

Repeated runs (3x, desktop apps open): 1024^3 NPU 1413-1480 GOPS; 2048^3 NPU 563-625 GOPS
(CPU numpy itself varied 370-649 GFLOPS, so 2048^3 is noisy: roughly a tie with the CPU).

All results verified equal to numpy int32 matmul.
The drop at 2048 is unexplained yet (likely data movement / tiling, not compute).

## Reproduce
    make -C third_party/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array \
      M=1024 K=1024 N=1024 build/final_1024x1024x1024_32x32x32_4c.xclbin
    # needs PEANO_INSTALL_DIR=.venv/lib/python3.14/site-packages/llvm-aie and .venv/bin on PATH
    python bench/matmul_npu.py <xclbin> <insts.bin> 1024 1024 1024 20

## MiniLM-L6 embeddings (64 short texts)

Linear layers on the NPU (int16 values quantized to +-511, i32 accumulate), everything
else (attention, softmax, LayerNorm, GELU) in numpy on the CPU.
Run: `python bench/minilm_npu_vs_cpu.py`

| backend | time | note |
|---|---|---|
| CPU float32 | 185 ms | |
| NPU linears | 181 ms | cosine vs CPU: min 0.996, mean 0.997 |

Roughly a tie at this model size. The win to measure next is CPU load / power, not latency.

## Matrix-vector (LLM decode shape), mlir-aie `matrix_vector`, 1 core, i16
`python bench/matvec_npu.py <xclbin> <insts> M K` (y = A @ x, A streamed from RAM every call)

| M x K | us/call | weights | GB/s |
|------:|--------:|--------:|-----:|
| 128 x 896  | 138  | 0.23 MB | 1.7 |
| 896 x 896  | 352  | 1.61 MB | 4.6 |
| 4864 x 896 | 1438 | 8.72 MB | 6.1 |

Fit: ~100 us fixed cost per call + ~6.5 GB/s streaming on one core.
Rough Qwen2.5-0.5B decode estimate with this kernel as-is (i16, 1 core): ~200+ ms/token (~4 tok/s).
Levers: more cores, int8 weights, fewer launches (fuse layers).

## Multi-core matvec (`src/kernels/matvec.py --n-cores N`, device `from_name("npu", n_cols=N)`)
| M x K | cores | us/call | GB/s |
|------:|------:|--------:|-----:|
| 896 x 896  | 1 | 346 | 4.6 |
| 896 x 896  | 2 | 287 | 5.6 |
| 896 x 896  | 4 | 206 | 7.8 |
| 4864 x 896 | 1 | 1438 | 6.1 |
| 4864 x 896 | 4 | 633 | 13.8 |

(The default device is a 1-column partition; with >1 core the design only fits if the
device is built with as many columns as cores.)

## Qwen2.5-0.5B-Instruct decode (`src/qwen_npu.py`)
All dense layers on the NPU (4-core i16 matvec, qkv and gate+up fused, lm_head in 16 chunks),
the rest in numpy. Greedy decoding, same answers as the float32 numpy version on the test prompts.

| backend | decode speed |
|---|---|
| numpy float32 (CPU) | 2.2 tok/s |
| NPU | 8.0 tok/s (80 tokens, steady) |

~92% of wall time is spent inside NPU calls (~97 per token). Next levers: 8-bit weights
(half the bytes), more cores per column, fewer/larger launches.

## int8 weights (`src/kernels/mv_i8.cc`, `matvec_i8.py`) and tile width
Custom AIE2 kernel: int8 weights x int16 activations -> int32. Each core widens 32 weights
(8 rows x 4 cols) to int16 and multiplies them with the matching b values; weights reach the
core "quad-transposed" via the DMA. `python bench/matvec8_npu.py <xclbin> <insts> M K`.

Finding: with 32x32 tiles int8 was NOT faster (662 us vs 659 us for 4864x896): each tile row is only
32-64 contiguous bytes in RAM and short bursts cost the same either way. Wider tiles fix it
(tile k = columns per tile, `-k`), m stays 32 (rows must divide into m*4 per call):

| 4864 x 896, 4 cores | k=32 | k=64 | k=128 |
|---|---:|---:|---:|
| int16 weights | 659 us | 495 us | 463 us |
| int8 weights  | 662 us | 377 us | **289 us** |

Other shapes with int8, k=128: 9728x896 479 us (int16/k=32: 1178 us), 896x4864 290 us, 6016x896 352 us.

Qwen2.5-0.5B-Instruct decode, steady: **11.5 tok/s** with int8 weights (7.9-8.0 tok/s with int16, 2.2 on CPU).
Weight quantization error is ~1% (per-row int8). Switch back with `NPU_INT8=0`.
Build: `python scripts/build_matvec.py --i8 1152x896 896x896 9728x896 896x4864 6016x896`.

## Context switching was the real cost (found by profiling inside the LLM loop)
Each matvec measured in isolation vs inside the Qwen loop (int8, 4 columns, one xclbin per shape):

| shape | isolated | in the LLM loop |
|---|---:|---:|
| 9728x896 | 479 us | 918 us |
| 896x4864 | 290 us | 743 us |
| 1152x896 | 165 us | 616 us |
| 896x896 | 151 us | 604 us |

A constant ~450 us per call = the NPU reconfiguring whenever the next call uses a different xclbin.
Also: more cores helps little (4864x896 int8: 1 core 746 us, 2 cores 448 us, 4 cores 291 us, i.e. ~150 us
fixed + 596 us / cores), so more columns is not the lever.

Two facts that fix it (`scripts/build_matvec.py --shared`):
1. The xclbin does not depend on M: one built for M=1152 runs M=9728 and M=896 with only the instruction stream swapped.
2. Two 2-column contexts stay resident at once: alternating A/B costs 1270 us = 821 + 450, no switch penalty.

Qwen2.5-0.5B with one shared 2-column xclbin per K (896 and 4864): **15.6 tok/s** (was 11.5), same answers.
