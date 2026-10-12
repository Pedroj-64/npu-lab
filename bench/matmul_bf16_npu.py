"""Check/time a bf16 -> f32 whole_array matmul (prefill building block): C = A(MxK) @ B(KxN).
usage: python bench/matmul_bf16_npu.py <xclbin> <insts.bin> M K N [iters]"""
import sys, time
import numpy as np, pyxrt
from ml_dtypes import bfloat16
xp, ip = sys.argv[1:3]
M, K, N = (int(v) for v in sys.argv[3:6])
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20
dev = pyxrt.device(0); xb = pyxrt.xclbin(xp); dev.register_xclbin(xb)
k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
ins = np.fromfile(ip, dtype=np.uint32)
rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(bfloat16); B = rng.standard_normal((K, N)).astype(bfloat16)
TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
ba = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3)); ba.write(A.tobytes(), 0); ba.sync(TO)
bb = pyxrt.bo(dev, B.nbytes, pyxrt.bo.host_only, k.group_id(4)); bb.write(B.tobytes(), 0); bb.sync(TO)
bc = pyxrt.bo(dev, M * N * 4, pyxrt.bo.host_only, k.group_id(5))
def run(): k(3, bi, len(ins), ba, bb, bc).wait()
run(); bc.sync(FROM)
C = np.frombuffer(bc.read(M * N * 4, 0), dtype=np.float32).reshape(M, N)
ref = A.astype(np.float32) @ B.astype(np.float32)
print(f"rel err vs float32: {np.linalg.norm(C-ref)/np.linalg.norm(ref):.2e}")
t = time.perf_counter()
for _ in range(iters): run()
dt = (time.perf_counter() - t) / iters
print(f"{M}x{K}x{N}: {dt*1e3:.2f} ms, {2*M*K*N/dt/1e9:.0f} GFLOPS")
