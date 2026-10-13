"""Does contention for DDR move the pure-read ceiling? 4 columns, 17 MB (M=8192): ro and real, 3 processes each, first on the quiet desktop and then while N processes copy
large arrays (DDR bandwidth + CPU load). usage: .venv/bin/python bench/dma_read_contention.py [--stress 8]   (NPU free)"""
import json, multiprocessing as mp, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
N = int(sys.argv[sys.argv.index("--stress") + 1]) if "--stress" in sys.argv else 8


def burn(stop):
    a = np.ones(16 * 1024 * 1024); b = np.empty_like(a)                      # 128 MB per array, 256 MB per process (8 processes = 2 GB): far beyond any cache, safe for the shared 16 GB
    while not stop.is_set():
        np.copyto(b, a); a += 1


def trial(kind):
    p = subprocess.run([sys.executable, "bench/dma_read_bench.py", kind, "4", "8192"], capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def block(label):
    res = {"ro": [], "real": []}
    for _ in range(3):
        for k in res:
            res[k].append(trial(k))
    for k, xs in res.items():
        t = np.array([x["median_s"] for x in xs]) * 1e3
        print(f"| {label} | {k} | {t.mean():.3f} ± {t.std(ddof=1):.3f} | {16.777216 / t.mean():.2f} | {min(float(x['load0']) for x in xs):.1f}-{max(float(x['load1']) for x in xs):.1f} |", flush=True)
    return res


def main():
    assert subprocess.run("fuser /dev/accel/* 2>/dev/null", shell=True, capture_output=True, text=True).stdout.strip() == "", "the NPU is in use"
    print("| condition | design | time ms (mean ± sd of 3 medians) | BW GB/s | load avg |\n|---|---|---:|---:|---|")
    block("quiet desktop")
    stop = mp.Event(); ps = [mp.Process(target=burn, args=(stop,)) for _ in range(N)]
    [p.start() for p in ps]; time.sleep(3)
    try:
        block(f"{N} memory-copy processes")
    finally:
        stop.set(); [p.join(5) for p in ps]; [p.terminate() for p in ps]


if __name__ == "__main__":
    main()
