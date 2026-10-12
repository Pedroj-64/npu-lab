# Licensing

The original code in this repository is released under the **MIT License** (see `LICENSE`).

## Files derived from other projects
These keep their original license and copyright header; our modifications to them are offered under the same terms.

| File | Derived from | License |
|---|---|---|
| `src/kernels/matvec.py`, `src/kernels/matvec_i8.py` | [mlir-aie](https://github.com/Xilinx/mlir-aie) `programming_examples/basic/matrix_multiplication/matrix_vector` | Apache-2.0 WITH LLVM-exception, (c) Advanced Micro Devices, Inc. |
| `src/kernels/mv_i8.cc` | mlir-aie `aie_kernels/linalg/mv_i16.cc` | Apache-2.0 WITH LLVM-exception, (c) Advanced Micro Devices, Inc. |

## Not included, downloaded by the user
Nothing below is redistributed in this repository; check each license before reuse.

- mlir-aie / IRON and the `llvm-aie` (Peano) compiler: Apache-2.0 with LLVM exception.
- XRT and the `amdxdna` driver: Apache-2.0 / GPL-2.0 (kernel module).
- Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct: Apache-2.0 (the 3B and 72B sizes use other licenses).
- paraphrase-multilingual-MiniLM-L12-v2 and all-MiniLM-L6-v2: Apache-2.0.
