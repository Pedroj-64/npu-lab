#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""int8 matrix x NV vectors (batched verification of drafted tokens), based on matvec_i8.py.

B is passed as (K/k, NV, k) int16 (host arranges it) and C comes back as (M/m, NV, m) int32: block g holds, for each of the
NV vectors, the m rows g*m..g*m+m-1. The weights are streamed once for all NV vectors.
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.kernels._common import _make_extern
import os
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


@iron.jit
def matrix_vector(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    vectorized: CompileTime[bool] = True,
    use_chess: CompileTime[bool] = False,
    n_cores: CompileTime[int] = 1,
    NV: CompileTime[int] = 4,
):
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    matvec_kernel = _make_extern(
        "matvec_i8_nv_i16_i32",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mv_i8_nv.cc"),
        [np.ndarray[(m * k,), np.dtype[np.int8]], np.ndarray[(NV * k,), np.dtype[np.int16]],
         np.ndarray[(NV * m,), np.dtype[np.int32]]],
        compile_flags=[f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DNV={NV}"],
        use_chess=use_chess,
    )
    zero_kernel = kernels.zero(NV * m, np.int32, use_chess=use_chess)

    dtype_in = np.dtype[np.int8]       # weights
    dtype_b = np.dtype[np.int16]       # activations
    dtype_out = np.dtype[np.int32]
    A_ty = np.ndarray[(M, K), dtype_in]
    B_ty = np.ndarray[(1, K * NV), dtype_b]
    C_ty = np.ndarray[(1, M * NV), dtype_out]
    inA_ty = np.ndarray[(m, k), dtype_in]
    inB_ty = np.ndarray[(NV * k,), dtype_b]
    outC_ty = np.ndarray[(NV * m,), dtype_out]

    # The vectorized mv kernel reads A in a "32-bit-word transposed" layout
    # (see aie_kernels/linalg/mv_i16.cc); the kernel's contract declares the DMA
    # transform on A's layout, so the design applies what the kernel wants.
    a_from_stream = TensorAccessPattern.full((k // 4, m, 4)).permute((1, 0, 2))

    def core_fn(of_a, of_b, of_c, zero, matvec):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matvec(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(inB_ty)
    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward(from_stream=a_from_stream))
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        workers.append(
            Worker(
                core_fn,
                [
                    coreA_fifos[i].cons(),
                    B_fifo.cons(),
                    outC_fifos[i].prod(),
                    zero_kernel,
                    matvec_kernel,
                ],
            )
        )

    A_taps = TensorAccessPattern.full((M, K)).tile((m, k)).split(0, M_div_m_div_n_cores)
    C_taps = TensorAccessPattern.full((M * NV,)).partition(n_cores)
    b_tap = TensorAccessPattern.full((1, K * NV)).repeat(M_div_m_div_n_cores)

    memA_prods = [f.prod() for f in memA_fifos]
    outC_cons = [f.cons() for f in outC_fifos]

    def sequence(a_in, b_in, c_out, b_h, memA_hs, outC_hs):
        b_h.fill(b_in, b_tap)
        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            memA_hs[i].fill(a_in, a_tap)
            outC_hs[i].drain(c_out, c_tap, wait=True)

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, B_fifo.prod(), memA_prods, outC_cons],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix-Vector Multiplication")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=288)
    p.add_argument("-K", type=int, default=288)
    p.add_argument("-N", type=int, default=4, help="NV: number of vectors processed per weight read")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("--dtype_in", type=str, default="i16")
    p.add_argument("--dtype_out", type=str, default="i32")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--scalar", action="store_true", help="use scalar mv kernel")
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    add_benchmark_args(p)
    return p


def _run_and_verify(opts):
    raise SystemExit("verify with bench/matvec8nv_npu.py (needs the batched B/C layouts)")


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        m=opts.m,
        k=opts.k,
        vectorized=not opts.scalar,
        use_chess=bool(opts.use_chess),
        n_cores=opts.n_cores,
        NV=opts.N,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        matrix_vector,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=o.n_cores if o.dev == "npu" else None),
    )


if __name__ == "__main__":
    main()
