"""Time ONE configuration of the pure-read DMA sweep in this process (bench/DMA_READ.md). Synchronous calls like the decode (submit + wait), weights int8 (M x 2048),
vector int16, output int32. Prints one JSON line. usage: .venv/bin/python bench/dma_read_bench.py KIND COLS M [--iters 30] [--warmup 5]
KIND = ro | plain | real  (build them with bench/dma_read_build.py)"""
import json, os, subprocess, sys, time
import numpy as np
import pyxrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kind, cols, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
iters, warm, K = arg("--iters", 30), arg("--warmup", 5), 2048
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
env0 = dict(load0=open("/proc/loadavg").read().split()[0], governor=sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), profile=sh("powerprofilesctl get"),
            top=sh("ps -eo pcpu,comm --sort=-pcpu | head -6 | tail -5 | tr -s ' ' | tr '\\n' ';'"), gpu=sh("nvidia-smi --query-gpu=pstate,utilization.gpu --format=csv,noheader"))
name = os.path.join(ROOT, f"build/dmaro/{kind}_c{cols}_{M}x{K}")
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
dev = pyxrt.device(0)
xb = pyxrt.xclbin(name + ".xclbin"); dev.register_xclbin(xb)
k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
ins = np.fromfile(name + ".bin", dtype=np.uint32)
rng = np.random.default_rng(0)
A = rng.integers(-127, 128, (M, K), dtype=np.int8); b = rng.integers(-2047, 2048, K, dtype=np.int16)
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
ba = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, k.group_id(3)); ba.write(A.tobytes(), 0); ba.sync(TO)
bb = pyxrt.bo(dev, b.nbytes, pyxrt.bo.host_only, k.group_id(4)); bb.write(b.tobytes(), 0); bb.sync(TO)
bc = pyxrt.bo(dev, M * 4, pyxrt.bo.host_only, k.group_id(5))
ts = []
for i in range(warm + iters):
    t = time.perf_counter(); k(3, bi, len(ins), ba, bb, bc).wait(); ts.append(time.perf_counter() - t)
ok = None
if kind == "real":
    bc.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    ok = bool(np.array_equal(np.frombuffer(bc.read(M * 4, 0), dtype=np.int32), A.astype(np.int32) @ b.astype(np.int32)))
ts = np.array(ts[warm:])
env1 = dict(load1=open("/proc/loadavg").read().split()[0])
print(json.dumps(dict(kind=kind, cols=cols, M=M, bytes=int(A.nbytes), median_s=float(np.median(ts)), min_s=float(ts.min()), p90_s=float(np.percentile(ts, 90)), cv_in=float(ts.std() / ts.mean()), ok=ok, **env0, **env1)))
