"""How much does it cost to alternate between two hardware contexts? usage: python bench/ctx_switch_npu.py A.xclbin A.bin MxK  B.xclbin B.bin MxK
Runs each alone, then alternating A,B,A,B. If alternating costs ~ the sum of the isolated times, contexts coexist without a switch penalty."""
import sys, time
import numpy as np
import pyxrt

dev = pyxrt.device(0)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE


def setup(xp, ip, shape):
    M, K = (int(v) for v in shape.split("x"))
    xb = pyxrt.xclbin(xp); dev.register_xclbin(xb)
    k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
    ins = np.fromfile(ip, dtype=np.uint32)
    rng = np.random.default_rng(0)
    A = rng.integers(-127, 128, (M, K), dtype=np.int8); x = rng.integers(-2047, 2048, K, dtype=np.int16)
    bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
    ba = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3)); ba.write(A.tobytes(), 0); ba.sync(TO)
    bx = pyxrt.bo(dev, x.nbytes, pyxrt.bo.host_only, k.group_id(4)); bx.write(x.tobytes(), 0); bx.sync(TO)
    bc = pyxrt.bo(dev, M * 4, pyxrt.bo.host_only, k.group_id(5))
    return lambda: k(3, bi, len(ins), ba, bx, bc).wait()


a = setup(*sys.argv[1:4])
try:
    b = setup(*sys.argv[4:7])
except Exception as e:
    print("second context could NOT be created next to the first:", str(e)[:100]); sys.exit(0)


def timeit(f, n=200):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e6


ta, tb = timeit(a), timeit(b)
tab = timeit(lambda: (a(), b()))
print(f"A alone {ta:.0f} us, B alone {tb:.0f} us, alternating A+B {tab:.0f} us -> switch penalty {tab - ta - tb:.0f} us per pair")
