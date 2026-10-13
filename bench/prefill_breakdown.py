"""Where does prompt reading (prefill, NPU only) spend its time? Splits one prefill into: host work before the NPU call (bf16 conversion, padding to 128 rows,
buffer write + sync), waiting for the kernel, reading the result back, and everything else (attention, norms, RoPE, SiLU in numpy).
The timed PrefillLin.__call__ below repeats src/prefill_npu.py line by line, only with clocks around each phase.
usage: .venv/bin/python bench/prefill_breakdown.py [--model models/qwen3s] [--tokens 256,1024,2048]"""
import argparse, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["NPU_GPU"] = "0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from ml_dtypes import bfloat16
import prefill_npu as P
import qwen_npu as Q

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen3s")
ap.add_argument("--tokens", default="256,1024,2048")
a = ap.parse_args()
T_ = {}
pc = time.perf_counter


def timed_call(self, X):
    px, T = self.pf.npu.px, X.shape[0]
    t0 = pc(); Xb = X.astype(bfloat16); T_["convert"] += pc() - t0
    out = np.empty((T, self.N), np.float32)
    for r in range(0, T, P.CH):
        rows = min(P.CH, T - r)
        t0 = pc(); blk = np.zeros((P.CH, self.K), bfloat16); blk[:rows] = Xb[r:r + rows]; T_["pad"] += pc() - t0
        c0 = 0
        for K, n, bo in self.parts:
            k, bi, ni, ba, bc = self.pf.kernel(K, n)
            t0 = pc(); ba.write(blk.tobytes(), 0); ba.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE); T_["write+sync in"] += pc() - t0
            t0 = pc(); run = k(3, bi, ni, ba, bo, bc); T_["submit"] += pc() - t0
            t0 = pc(); run.wait(); T_["wait (kernel)"] += pc() - t0
            t0 = pc(); bc.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            out[r:r + rows, c0:c0 + n] = np.frombuffer(bc.read(P.CH * n * 4, 0), dtype=np.float32).reshape(P.CH, n)[:rows]; T_["sync+read out"] += pc() - t0
            c0 += n
            T_["dispatches"] += 1
    return out if self.bias is None else out + self.bias


P.PrefillLin.__call__ = timed_call
m = Q.Qwen(a.model, gpu=False)
assert m.pf, "prefill kernels missing"
rng = np.random.default_rng(0)
print(f"## {a.model}\n| tokens (rep) | total s | tok/s | wait (kernel) | write+sync in | sync+read out | convert+pad | submit | NPU-linear total | attention+norms+rope+silu (rest) | dispatches |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
for n in map(int, a.tokens.split(",")):
    ids = rng.integers(1000, 100000, n).tolist()
    for rep in range(3):                                   # every repetition is printed: this phase is not stable run to run
        T_.clear(); T_.update({k: 0.0 for k in ("convert", "pad", "write+sync in", "submit", "wait (kernel)", "sync+read out", "dispatches")})
        m.cache_ids = []; t = pc(); m.prefill(ids, 0); tot = pc() - t
        lin = sum(T_[k] for k in ("convert", "pad", "write+sync in", "submit", "wait (kernel)", "sync+read out"))
        pct = lambda x: f"{x:.2f}s ({x / tot * 100:.0f}%)"
        print(f"| {n} ({rep}) | {tot:.2f} | {n / tot:.0f} | {pct(T_['wait (kernel)'])} | {pct(T_['write+sync in'])} | {pct(T_['sync+read out'])} | {pct(T_['convert'] + T_['pad'])} | {pct(T_['submit'])} | {pct(lin)} | {pct(tot - lin)} | {int(T_['dispatches'])} |", flush=True)
