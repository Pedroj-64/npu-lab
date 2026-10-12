"""Run a mlir-aie whole_array matmul (i16 in, i32 out) on the NPU via pyxrt,
check it against numpy, and report NPU vs CPU time.

usage: python bench/matmul_npu.py <xclbin> <insts.bin> [M K N] [iters]
"""
import sys, time
import numpy as np
import pyxrt

xclbin_path, insts_path = sys.argv[1:3]
M, K, N = (int(x) for x in sys.argv[3:6]) if len(sys.argv) >= 6 else (512, 512, 512)
iters = int(sys.argv[6]) if len(sys.argv) >= 7 else 50

dev = pyxrt.device(0)
xb = pyxrt.xclbin(xclbin_path)
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kernel = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

insts = np.fromfile(insts_path, dtype=np.uint32)
rng = np.random.default_rng(0)
A = rng.integers(-8, 8, (M, K), dtype=np.int16)
B = rng.integers(-8, 8, (K, N), dtype=np.int16)

flags = pyxrt.bo.host_only
bo_i = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
bo_a = pyxrt.bo(dev, A.nbytes, flags, kernel.group_id(3))
bo_b = pyxrt.bo(dev, B.nbytes, flags, kernel.group_id(4))
bo_c = pyxrt.bo(dev, M * N * 4, flags, kernel.group_id(5))
bo_i.write(insts.tobytes(), 0); bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_a.write(A.tobytes(), 0); bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_b.write(B.tobytes(), 0); bo_b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def run_once():
    r = kernel(3, bo_i, len(insts), bo_a, bo_b, bo_c)
    r.wait()


run_once()  # warmup
bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
C = np.frombuffer(bo_c.read(M * N * 4, 0), dtype=np.int32).reshape(M, N)
ref = A.astype(np.int32) @ B.astype(np.int32)
print("correct:", np.array_equal(C, ref))

t = time.perf_counter()
for _ in range(iters):
    run_once()
npu = (time.perf_counter() - t) / iters

Af, Bf = A.astype(np.float32), B.astype(np.float32)
t = time.perf_counter()
for _ in range(iters):
    Af @ Bf
cpu = (time.perf_counter() - t) / iters

ops = 2 * M * K * N
print(f"{M}x{K}x{N}  NPU {npu*1e3:.3f} ms ({ops/npu/1e9:.1f} GOPS)  "
      f"CPU numpy f32 {cpu*1e3:.3f} ms ({ops/cpu/1e9:.1f} GFLOPS)")
