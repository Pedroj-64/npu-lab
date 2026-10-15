"""Energy per token: tok/s, mean watts and tokens per joule for decode and prompt reading, one configuration per process (the NPU takes one process).
Power sources (no root needed):
  - APU package ("PPT": CPU + NPU + iGPU, one number, the NPU alone has no sensor): /sys/class/hwmon/<amdgpu>/power1_input (microwatts, read at 20 Hz)
  - NVIDIA board: `nvidia-smi --query-gpu=power.draw.instant` (documented accuracy +-5 W), streamed every 100 ms
  Not counted: screen, SSD, RAM outside the package, fans, charger losses (RAPL energy_uj is root-only; the battery reports 0 W on AC).
Total = APU + GPU. "idle" = the same process with the model loaded and nothing running, so net J/token = (P - P_idle) / (tok/s) is the marginal cost.
usage: .venv/bin/python bench/energy.py --model models/qwen3 [--share 0.5] [--engine assisted|npu] [--phases decode,prefill] [--runs 3] [--json out.jsonl]
  --engine npu: Qwen3 NPU-only profile (gpu=False, prompts read by the NPU); assisted (default): GPU reads prompts and attends."""
import argparse, glob, json, os, statistics as st, subprocess, sys, threading, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--share", type=float, default=0.0)
ap.add_argument("--engine", default="assisted", choices=["assisted", "npu"])
ap.add_argument("--phases", default="decode,prefill")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--tokens", type=int, default=64, help="decode steps per run")
ap.add_argument("--prefill-tokens", type=int, default=0, help="0 = 512 for --engine npu, 4096 otherwise")
ap.add_argument("--min-seconds", type=float, default=6.0, help="a prefill run repeats the prompt until it lasted this long")
ap.add_argument("--vram-gb", type=float, default=4.5)
ap.add_argument("--idle-seconds", type=float, default=20.0)
ap.add_argument("--json")
ap.add_argument("--force", action="store_true", help="run even if not on AC / not in the performance power profile")
a = ap.parse_args()
if a.engine == "npu":
    os.environ["NPU_GPU"] = "0"
import numpy as np


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


ppt = next((d + "/power1_input" for d in glob.glob("/sys/class/hwmon/hwmon*") if open(d + "/name").read().strip() == "amdgpu" and os.path.exists(d + "/power1_input")), None)
assert ppt, "no amdgpu hwmon power1_input (APU package power)"
state = dict(profile=sh("powerprofilesctl get"), ac=sh("cat /sys/class/power_supply/A*/online"), governor=sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), kernel=sh("uname -r"))
if not a.force and (state["profile"] != "performance" or state["ac"] != "1"):
    sys.exit(f"conditions not met: {state} (plug in, `powerprofilesctl set performance`, or --force)")


class Sampler:
    """Time-stamped samples of both sources; energy of an interval = trapezoid over the samples inside it."""
    def __init__(self):
        self.apu, self.gpu, self.run = [], [], True
        threading.Thread(target=self._apu, daemon=True).start()
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant,memory.used", "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
        self.vram = 0.0
        threading.Thread(target=self._gpu, daemon=True).start()

    def _apu(self):
        f = open(ppt)
        while self.run:
            f.seek(0); self.apu.append((time.perf_counter(), int(f.read()) / 1e6)); time.sleep(0.05)

    def _gpu(self):
        for line in self.p.stdout:
            w, mem = (float(x) for x in line.split(","))
            self.gpu.append((time.perf_counter(), w)); self.vram = max(self.vram, mem)

    def mean(self, series, t0, t1):
        xs = [w for t, w in series if t0 <= t <= t1]
        return sum(xs) / len(xs) if xs else float("nan"), len(xs)

    def watts(self, t0, t1):
        (pa, na), (pg, ng) = self.mean(self.apu, t0, t1), self.mean(self.gpu, t0, t1)
        return pa, pg, min(na, ng)

    def stop(self):
        self.run = False; self.p.terminate()


S = Sampler()
time.sleep(1.0)
t0 = time.perf_counter(); time.sleep(a.idle_seconds)
nomodel = S.watts(t0, time.perf_counter())

import qwen_npu as Q
if a.engine == "npu":
    m = Q.Qwen(a.model, gpu=False)
else:
    import qwen_npu as Q35
    m = Q35.Qwen(a.model, gpu=True, vram_gb=a.vram_gb, max_ctx=8192)
t0 = time.perf_counter(); time.sleep(a.idle_seconds)
idle = S.watts(t0, time.perf_counter())
rng = np.random.default_rng(0)
out = []


def measure(name, work, tokens):
    """work() -> number of tokens processed; run once to warm up, then --runs timed runs."""
    work()
    rows = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); n = work(); t1 = time.perf_counter()
        pa, pg, ns = S.watts(t0, t1)
        tps = n / (t1 - t0)
        rows.append(dict(phase=name, tok_s=tps, apu_w=pa, gpu_w=pg, total_w=pa + pg, tok_per_j=tps / (pa + pg), net_tok_per_j=tps / max(pa + pg - idle[0] - idle[1], 1e-6), samples=ns))
    return rows


def decode():
    ids = rng.integers(1000, 100000, 16).tolist(); m.cache_ids = []
    logits = m.prefill(ids, 0); pos = len(ids)
    t = time.perf_counter()
    for _ in range(a.tokens):
        logits = m.step(int(np.argmax(logits)), pos); pos += 1
    return a.tokens


n_pf = a.prefill_tokens or (512 if a.engine == "npu" else 4096)
ids_pf = rng.integers(1000, 100000, n_pf).tolist()


def prefill():
    done, t = 0, time.perf_counter()
    while time.perf_counter() - t < a.min_seconds:
        m.cache_ids = []; m.prefill(ids_pf, 0); done += n_pf
    return done


if "decode" in a.phases:
    out += measure("decode", decode, a.tokens)
if "prefill" in a.phases:
    out += measure(f"prefill {n_pf}", prefill, n_pf)
S.stop()
label = f"{os.path.basename(a.model)} {a.engine}"
ms = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs) if len(xs) > 1 else 0:.2f}"
print(f"## {label}  ({state['profile']}, AC={state['ac']}, governor={state['governor']}, kernel={state['kernel']}, max VRAM seen {S.vram:.0f} MiB)")
print(f"idle, no model: APU {nomodel[0]:.2f} W + GPU {nomodel[1]:.2f} W; idle, model loaded: APU {idle[0]:.2f} W + GPU {idle[1]:.2f} W = {idle[0] + idle[1]:.2f} W")
print("| phase | tok/s | APU W | GPU W | total W | tokens/J | net tokens/J | samples/run |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for ph in dict.fromkeys(r["phase"] for r in out):
    rs = [r for r in out if r["phase"] == ph]
    print(f"| {ph} | {ms([r['tok_s'] for r in rs])} | {ms([r['apu_w'] for r in rs])} | {ms([r['gpu_w'] for r in rs])} | {ms([r['total_w'] for r in rs])} | "
          f"{ms([r['tok_per_j'] for r in rs])} | {ms([r['net_tok_per_j'] for r in rs])} | {min(r['samples'] for r in rs)} |")
if a.json:
    with open(a.json, "a") as f:
        f.write(json.dumps(dict(label=label, state=state, idle=idle, nomodel=nomodel, runs=out)) + "\n")
