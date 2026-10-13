"""Check and time the int8 matrix x NV vectors kernel (src/kernels/matvec_i8_nv.py).
usage: python bench/matvec8nv_npu.py <xclbin> <insts.bin> M K NV [n_cores=2] [k=128] [m=32] [iters=200]
Key number: us/call as NV grows. If NV=4 costs about the same as NV=1, verifying 4 drafted tokens per weight read is almost free."""
import sys, time
import numpy as np
import pyxrt

xp, ip = sys.argv[1:3]
M, K, NV = (int(a) for a in sys.argv[3:6])
cores = int(sys.argv[6]) if len(sys.argv) > 6 else 2
k_t = int(sys.argv[7]) if len(sys.argv) > 7 else 128
m_t = int(sys.argv[8]) if len(sys.argv) > 8 else 32
iters = int(sys.argv[9]) if len(sys.argv) > 9 else 200
dev = pyxrt.device(0); xb = pyxrt.xclbin(xp); dev.register_xclbin(xb)
k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
ins = np.fromfile(ip, dtype=np.uint32)
rng = np.random.default_rng(0)
A = rng.integers(-127, 128, (M, K), dtype=np.int8); X = rng.integers(-2047, 2048, (NV, K), dtype=np.int16)
B = np.ascontiguousarray(X.reshape(NV, K // k_t, k_t).transpose(1, 0, 2))          # (K/k, NV, k)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
ba = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3)); ba.write(A.tobytes(), 0); ba.sync(TO)
bx = pyxrt.bo(dev, B.nbytes, pyxrt.bo.host_only, k.group_id(4)); bx.write(B.tobytes(), 0); bx.sync(TO)
bc = pyxrt.bo(dev, M * NV * 4, pyxrt.bo.host_only, k.group_id(5))
def run(): k(3, bi, len(ins), ba, bx, bc).wait()
run(); bc.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
y = np.frombuffer(bc.read(M * NV * 4, 0), dtype=np.int32).reshape(M // m_t, NV, m_t).transpose(1, 0, 2).reshape(NV, M)
want = X.astype(np.int32) @ A.astype(np.int32).T
print("correct:", np.array_equal(y, want), "" if np.array_equal(y, want) else f"(max diff {np.abs(y - want).max()})")
t = time.perf_counter()
for _ in range(iters): run()
dt = (time.perf_counter() - t) / iters
print(f"{M}x{K} NV={NV}: {dt*1e6:.0f} us/call, weights {A.nbytes/1e6:.2f} MB -> {A.nbytes/dt/1e9:.2f} GB/s, {dt*1e6/NV:.0f} us per vector")
