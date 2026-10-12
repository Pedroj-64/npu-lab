"""Time a mlir-aie matrix_vector kernel (i16, y = A @ x) on the NPU.
usage: python bench/matvec_npu.py <xclbin> <insts.bin> M K [iters]
"""
import sys, time
import numpy as np
import pyxrt

xp, ip = sys.argv[1:3]
M, K = int(sys.argv[3]), int(sys.argv[4])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 200
dev = pyxrt.device(0); xb = pyxrt.xclbin(xp); dev.register_xclbin(xb)
k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
ins = np.fromfile(ip, dtype=np.uint32)
rng = np.random.default_rng(0)
A = rng.integers(-100, 100, (M, K), dtype=np.int16); x = rng.integers(-100, 100, K, dtype=np.int16)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
ba = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3)); ba.write(A.tobytes(), 0); ba.sync(TO)
bx = pyxrt.bo(dev, x.nbytes, pyxrt.bo.host_only, k.group_id(4)); bx.write(x.tobytes(), 0); bx.sync(TO)
bc = pyxrt.bo(dev, M * 4, pyxrt.bo.host_only, k.group_id(5))
def run(): k(3, bi, len(ins), ba, bx, bc).wait()
run(); bc.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
y = np.frombuffer(bc.read(M * 4, 0), dtype=np.int32)
print("correct:", np.array_equal(y, A.astype(np.int32) @ x.astype(np.int32)))
t = time.perf_counter()
for _ in range(iters): run()
dt = (time.perf_counter() - t) / iters
print(f"{M}x{K}: {dt*1e6:.0f} us/call, weights {A.nbytes/1e6:.2f} MB -> {A.nbytes/dt/1e9:.2f} GB/s")
