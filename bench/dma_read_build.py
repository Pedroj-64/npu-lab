"""Build the xclbins/instruction streams of the pure-read DMA ceiling sweep (CPU only, parallel; aiecc may run next to an NPU job).
  ro    : bench/dma_read_only.py          (same data movement as the decode kernel, no multiply-accumulate)
  plain : the same with --plain           (no access-pattern transform on A)
  real  : src/kernels/matvec_i8.py        (the decode kernel with mv_i8.cc)
Names: build/dmaro/<kind>_c<cols>_<M>x2048.{xclbin,bin}.  usage: .venv/bin/python bench/dma_read_build.py"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
env = {**os.environ, "PEANO_INSTALL_DIR": f"{ROOT}/.venv/lib/python3.14/site-packages/llvm-aie", "PATH": f"{ROOT}/.venv/bin:" + os.environ["PATH"]}
os.makedirs("build/dmaro", exist_ok=True)
jobs = []
for cols in (1, 2, 4):
    for M in (2048, 8192, 16384, 32768):
        for kind in ("ro", "plain"):
            jobs.append((kind, cols, M))
    for M in (2048, 8192):
        jobs.append(("real", cols, M))


def build(j):
    kind, cols, M = j
    out = f"build/dmaro/{kind}_c{cols}_{M}x2048"
    if os.path.exists(out + ".xclbin"):
        return j, "cached"
    script = "src/kernels/matvec_i8.py" if kind == "real" else "bench/dma_read_only.py"
    cmd = [sys.executable, script, "--dev", "npu", "-M", str(M), "-K", "2048", "--n-cores", str(cols), "-k", "128", f"--xclbin-path={out}.xclbin", f"--insts-path={out}.bin"]
    if kind == "plain":
        cmd.append("--plain")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return j, "ok" if r.returncode == 0 and os.path.exists(out + ".xclbin") else "FAILED " + (r.stdout + r.stderr)[-300:]


with ThreadPoolExecutor(6) as ex:
    for j, st in ex.map(build, jobs):
        print(j, st, flush=True)
