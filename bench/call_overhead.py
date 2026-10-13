"""Fixed cost of one NPU linear call, variants (Phase 2 step 2, batch C; criteria in bench/FASE2.md). The cycle is NpuLin.__call__ of src/qwen_npu.py: quantise the input, write + sync the input
pieces, submit one run per piece, wait, sync the output, int32 -> scaled float. Same xclbin / instruction streams / buffers for every variant; outputs must be IDENTICAL bit for bit.
 V0 current: a new run object per piece per call (kern(3, ...)), bc.sync(from device), post with int64 temporaries
 V1 : preconfigured runs (pyxrt.run + set_arg once; per call only start() and wait())
 V2 : V1 without the output sync (INVALID: the outputs then differ from V0, so the sync from the device is needed; kept to show it)
 V3 : V1 + post-processing without the int64 temporary
usage: .venv/bin/python bench/call_overhead.py M NK [--calls 300]   (K = 2048, NK pieces of K0 = 2048 as 'down' of Qwen3.5-2B uses 3)   one process = one measurement; NPU free
Prints a JSON line (appended to bench/call_overhead.jsonl)."""
import json, os, sys, time
import numpy as np
import pyxrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M, NK = int(sys.argv[1]), int(sys.argv[2])
calls = int(sys.argv[sys.argv.index("--calls") + 1]) if "--calls" in sys.argv else 300
K0, XMAX = 2048, 2047
TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
dev = pyxrt.device(0)
xb = pyxrt.xclbin(f"{ROOT}/build/c4/c4_2048x2048.xclbin"); dev.register_xclbin(xb)
kern = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
ins = np.fromfile(f"{ROOT}/build/c4/c4_{M}x2048.bin", dtype=np.uint32)
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
rng = np.random.default_rng(0)
bw, bc, bcv, bx, bxv = [], [], [], [], []
for j in range(NK):
    W = rng.integers(-127, 128, (M, K0), dtype=np.int8)
    b = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(3)); b.write(W.tobytes(), 0); b.sync(TO); bw.append(b)
    c = pyxrt.bo(dev, M * 4, pyxrt.bo.host_only, kern.group_id(5)); bc.append(c); bcv.append(np.frombuffer(c.map(), dtype=np.int32))
    x = pyxrt.bo(dev, K0 * 2, pyxrt.bo.host_only, kern.group_id(4)); bx.append(x); bxv.append(np.frombuffer(x.map(), dtype=np.int16))
s = np.float32(0.00123)
runs = []
for j in range(NK):
    r = pyxrt.run(kern); r.set_arg(0, 3); r.set_arg(1, bi); r.set_arg(2, len(ins)); r.set_arg(3, bw[j]); r.set_arg(4, bx[j]); r.set_arg(5, bc[j]); runs.append(r)
xs = [rng.standard_normal(NK * K0).astype(np.float32) for _ in range(8)]


def quant(x):
    sx = np.abs(x).max() / XMAX or 1.0
    return np.round(x / sx).astype(np.int16), sx


def v0(x):
    xq, sx = quant(x)
    for j in range(NK):
        bxv[j][:] = xq[j * K0:(j + 1) * K0]; bx[j].sync(TO)
    rs = [kern(3, bi, len(ins), bw[j], bx[j], bc[j]) for j in range(NK)]
    acc = None
    for j, r in enumerate(rs):
        r.wait(); bc[j].sync(FROM)
        acc = bcv[j].astype(np.int64) if acc is None else acc + bcv[j]
    return acc * s * sx


def v1(x, sync_out=True, light=False):
    xq, sx = quant(x)
    for j in range(NK):
        bxv[j][:] = xq[j * K0:(j + 1) * K0]; bx[j].sync(TO)
    for r in runs: r.start()
    acc = None
    for j, r in enumerate(runs):
        r.wait()
        if sync_out: bc[j].sync(FROM)
        if light:
            acc = bcv[j] * 1 if acc is None else acc + bcv[j]
        else:
            acc = bcv[j].astype(np.int64) if acc is None else acc + bcv[j]
    return acc * s * sx


V = {"V0": v0, "V1": v1, "V2": lambda x: v1(x, sync_out=False), "V3": lambda x: v1(x, sync_out=True, light=True)}
ref = {k: [f(x).copy() for x in xs[:3]] for k, f in V.items()}
same = {k: all(np.array_equal(a, b) for a, b in zip(ref[k], ref["V0"])) for k in V}
res = {}
for k, f in V.items():
    for _ in range(20): f(xs[0])
    ts = []
    for i in range(calls):
        t = time.perf_counter(); f(xs[i % 8]); ts.append(time.perf_counter() - t)
    res[k] = float(np.median(ts) * 1e6)
print(json.dumps(dict(M=M, NK=NK, us=res, identical=same)))
open(os.path.join(ROOT, "bench", "call_overhead.jsonl"), "a").write(json.dumps(dict(M=M, NK=NK, us=res, identical=same)) + "\n")
