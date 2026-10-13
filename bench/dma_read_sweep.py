"""Run the pure-read DMA sweep: every configuration in N independent processes, interleaved (round r runs all configurations once), results in
bench/dma_read_runs.jsonl (appended; resumable) and a markdown summary. usage: .venv/bin/python bench/dma_read_sweep.py [--procs 3] [--report-only]
Run it with the NPU free. Contention is recorded per process (load average, top CPU processes, GPU state) and the spread between processes is reported."""
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
OUT = "bench/dma_read_runs.jsonl"
procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
CONF = [(kind, c, M) for c in (1, 2, 4) for M in (2048, 8192, 16384, 32768) for kind in ("ro", "plain") if os.path.exists(f"build/dmaro/{kind}_c{c}_{M}x2048.xclbin")] + \
       [("real", c, M) for c in (1, 2, 4) for M in (2048, 8192)]


def runs():
    return [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []


if "--report-only" not in sys.argv:
    assert subprocess.run("fuser /dev/accel/* 2>/dev/null", shell=True, capture_output=True, text=True).stdout.strip() == "", "the NPU is in use"
    for r in range(procs):
        for kind, c, M in CONF:
            have = sum(1 for x in runs() if (x["kind"], x["cols"], x["M"]) == (kind, c, M))
            if have > r:
                continue
            p = subprocess.run([sys.executable, "bench/dma_read_bench.py", kind, str(c), str(M)], capture_output=True, text=True, timeout=300)
            if p.returncode:
                print("FAILED", kind, c, M, p.stderr[-200:], flush=True); continue
            open(OUT, "a").write(p.stdout.strip().splitlines()[-1] + "\n")
        print("round", r + 1, "done", flush=True)
R = runs()
key = lambda x: (x["kind"], x["cols"], x["M"])
agg = {}
for x in R:
    agg.setdefault(key(x), []).append(x)
print("\n| design | cols | A (MB) | n procs | time ms (mean ± sd of the medians) | BW GB/s (mean ± sd) | CV between procs | load avg (min-max) | data check |\n|---|---:|---:|---:|---:|---:|---:|---|---|")
for k in sorted(agg, key=lambda k: (k[1], k[2], k[0])):
    xs = agg[k]; t = np.array([x["median_s"] for x in xs]) * 1e3; bw = np.array([x["bytes"] / x["median_s"] / 1e9 for x in xs])
    la = [float(x["load0"]) for x in xs] + [float(x["load1"]) for x in xs]
    print(f"| {k[0]} | {k[1]} | {xs[0]['bytes'] / 1e6:.0f} | {len(xs)} | {t.mean():.3f} ± {t.std(ddof=1) if len(t) > 1 else 0:.3f} | {bw.mean():.2f} ± {bw.std(ddof=1) if len(bw) > 1 else 0:.2f} | "
          f"{(t.std(ddof=1) / t.mean() * 100) if len(t) > 1 else 0:.1f} % | {min(la):.1f}-{max(la):.1f} | {xs[0]['ok'] if xs[0]['ok'] is not None else '-'} |")
print("\nr = T_ro / T_real (same cols and size):")
for c in (1, 2, 4):
    for M in (2048, 8192):
        a, b = agg.get(("ro", c, M)), agg.get(("real", c, M))
        if a and b:
            ra = [x["median_s"] for x in a]; rb = [x["median_s"] for x in b]
            print(f"- {c} col, {M * 2048 / 1e6:.0f} MB: r = {np.mean(ra) / np.mean(rb):.3f}  (ro {np.mean(ra) * 1e3:.3f} ms, real {np.mean(rb) * 1e3:.3f} ms)")
