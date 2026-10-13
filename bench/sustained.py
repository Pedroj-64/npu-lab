"""Sustained decode at a GPU share: tok/s per 30 s window, whole-machine power (APU hwmon + NVIDIA), GPU temperature, SM and memory clocks, throttle reasons, CPU temperature, fan.
Decode runs continuously for --seconds (several 64-token generations from a fixed prompt). usage: .venv/bin/python bench/sustained.py --model models/qwen35_2b --share 0.75 [--seconds 300] [--json out]   (NPU + GPU: run alone)"""
import argparse, glob, json, os, statistics as st, subprocess, sys, threading, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/qwen35_2b"); ap.add_argument("--share", type=float, required=True)
ap.add_argument("--seconds", type=float, default=300); ap.add_argument("--vram-gb", type=float, default=4.5); ap.add_argument("--json"); a = ap.parse_args()
hw = {open(d + "/name").read().strip(): d for d in glob.glob("/sys/class/hwmon/hwmon*")}
rd = lambda p: int(open(p).read())
apu = hw["amdgpu"] + "/power1_input"; cput = hw["k10temp"] + "/temp1_input"
fans = glob.glob(hw["hp"] + "/fan*_input") if "hp" in hw else []
Q = "temperature.gpu,clocks.sm,clocks.mem,power.draw.instant,clocks_throttle_reasons.active,pstate"
rows, run = [], True


def sample():
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits", "-lms", "1000"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if not run: break
        t, sm, mem, w, thr, ps = [x.strip() for x in line.split(",")]
        rows.append(dict(t=time.perf_counter(), gpu_c=float(t), sm=float(sm), mem=float(mem), gpu_w=float(w), thr=int(thr, 16), pstate=ps, apu_w=rd(apu) / 1e6, cpu_c=rd(cput) / 1000,
                         fan=[rd(f) for f in fans], freq=st.mean(rd(f) for f in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")) / 1e3))
    p.terminate()


import numpy as np
import qwen35_npu as Q35
m = Q35.load_engine(a.model, gpu=True, gpu_share=a.share, vram_gb=a.vram_gb, max_ctx=4096)
ids = m.tok.encode("Explica en detalle cómo funciona un motor de combustión interna.", add_special_tokens=False).ids
threading.Thread(target=sample, daemon=True).start(); time.sleep(3)
t0 = time.perf_counter(); wins, n_tok, w0 = [], 0, t0
while time.perf_counter() - t0 < a.seconds:
    m.cache_ids = []; lg = m.prefill(ids, 0); pos = len(ids)
    for _ in range(64):
        lg = m.step(int(np.argmax(lg)), pos); pos += 1; n_tok += 1
        if time.perf_counter() - w0 >= 30:
            wins.append((w0, time.perf_counter(), n_tok)); w0, n_tok = time.perf_counter(), 0
run = False; time.sleep(1.2)
res = []
for (s, e, n) in wins:
    r = [x for x in rows if s <= x["t"] <= e]
    res.append(dict(t=round(s - t0), tok_s=n / (e - s), total_w=st.mean(x["apu_w"] + x["gpu_w"] for x in r), gpu_w=st.mean(x["gpu_w"] for x in r), gpu_c=max(x["gpu_c"] for x in r), sm=st.mean(x["sm"] for x in r),
                    mem=min(x["mem"] for x in r), thr=sorted({hex(x["thr"]) for x in r}), cpu_c=max(x["cpu_c"] for x in r), freq=st.mean(x["freq"] for x in r), fan=[max(x["fan"][i] for x in r) for i in range(len(fans))]))
print(f"model {os.path.basename(a.model)} share {a.share:.0%}, {a.seconds:.0f} s")
print("| t s | tok/s | total W | GPU W | GPU max C | SM MHz | mem MHz min | throttle | CPU max C | CPU MHz | fan |\n|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
for r in res:
    print(f"| {r['t']} | {r['tok_s']:.2f} | {r['total_w']:.1f} | {r['gpu_w']:.1f} | {r['gpu_c']:.0f} | {r['sm']:.0f} | {r['mem']:.0f} | {','.join(r['thr'])} | {r['cpu_c']:.0f} | {r['freq']:.0f} | {r['fan']} |")
k = max(1, len(res) // 5); f, l = res[:k], res[-k:]
print(f"first {k} windows vs last {k}: tok/s {st.mean(x['tok_s'] for x in f):.2f} -> {st.mean(x['tok_s'] for x in l):.2f} ({(st.mean(x['tok_s'] for x in l) / st.mean(x['tok_s'] for x in f) - 1) * 100:+.1f} %); total W {st.mean(x['total_w'] for x in f):.1f} -> {st.mean(x['total_w'] for x in l):.1f}; max GPU C {max(x['gpu_c'] for x in res):.0f}; max CPU C {max(x['cpu_c'] for x in res):.0f}")
if a.json:
    open(a.json, "a").write(json.dumps(dict(model=a.model, share=a.share, seconds=a.seconds, windows=res)) + "\n")
