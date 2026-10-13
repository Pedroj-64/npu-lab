"""What does it cost to change the instruction buffer inside ONE hardware context (what the decode does between matrices)?
One xclbin / one context (the one of A, K shared), two instruction streams A and B (different M), each with its own weights and output.
Times, per call and synchronous (submit + wait): A alone, B alone, A,B alternating (instruction buffer changes every call), and a control
A,A' alternating with the SAME instruction stream and different buffers (changes everything except the instructions).
usage: .venv/bin/python bench/insts_switch_npu.py build/c4/c4_1024x1024 build/c4/c4_3072x1024 [--n 300] [--reps 5]   (prefixes without extension; same K)"""
import argparse, statistics as st, time
import numpy as np
import pyxrt

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
dev = pyxrt.device(0)
xb = pyxrt.xclbin(a.a + ".xclbin"); dev.register_xclbin(xb)
kern = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
rng = np.random.default_rng(0)


def setup(prefix):
    M, K = (int(v) for v in prefix.rsplit("_", 1)[1].split("x"))
    ins = np.fromfile(prefix + ".bin", dtype=np.uint32)
    bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
    W = rng.integers(-127, 128, (M, K), dtype=np.int8); x = rng.integers(-2047, 2048, K, dtype=np.int16)
    bw = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(3)); bw.write(W.tobytes(), 0); bw.sync(TO)
    bx = pyxrt.bo(dev, x.nbytes, pyxrt.bo.host_only, kern.group_id(4)); bx.write(x.tobytes(), 0); bx.sync(TO)
    bc = pyxrt.bo(dev, M * 4, pyxrt.bo.host_only, kern.group_id(5))
    return lambda: kern(3, bi, len(ins), bw, bx, bc).wait(), M, K


fa, Ma, Ka = setup(a.a); fb, Mb, Kb = setup(a.b); fa2, _, _ = setup(a.a)
assert Ka == Kb, "A and B must share K (one xclbin)"


def us(f):
    f(); t = time.perf_counter()
    for _ in range(a.n): f()
    return (time.perf_counter() - t) / a.n * 1e6


rows = {k: [] for k in ("A alone", "B alone", "A+B (insts change)", "A+A' (same insts)")}
for _ in range(a.reps):
    rows["A alone"].append(us(fa)); rows["B alone"].append(us(fb))
    rows["A+B (insts change)"].append(us(lambda: (fa(), fb()))); rows["A+A' (same insts)"].append(us(lambda: (fa(), fa2())))
ms = lambda k: (st.mean(rows[k]), st.stdev(rows[k]))
print(f"## {a.a.split('/')[-1]} ({Ma}x{Ka}) vs {a.b.split('/')[-1]} ({Mb}x{Kb}), {a.n} calls x {a.reps} reps, microseconds per call (pair for the last two)")
for k in rows:
    m, s = ms(k); print(f"- {k}: {m:.1f} +- {s:.1f}")
pen = ms("A+B (insts change)")[0] - ms("A alone")[0] - ms("B alone")[0]
ctl = ms("A+A' (same insts)")[0] - 2 * ms("A alone")[0]
print(f"- penalty of changing the instruction stream: {pen:+.1f} us per switch (control, same insts: {ctl:+.1f} us per pair)")
