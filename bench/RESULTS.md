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
