"""Cost of the sentinel: APU package power with the service stopped vs running (idle, no model loaded), plus the energy of one scan.
Rule written before measuring (plan T3.1): average power added by the sentinel <= 1 W. The service sleeps 30 minutes between scans, so the average added power is
(extra joules of one scan) / 1800 s; the idle windows only show that sleeping costs nothing.
usage: .venv/bin/python bench/sentinel_power.py [--seconds 45] [--rounds 2] [--roots ~/Documentos/Codigo]     (leave the PC alone while it runs: ~4 minutes)
Spanish: costo del centinela: potencia en reposo con el servicio parado y activo, y la energía de un escaneo; criterio: <= 1 W añadido en promedio."""
import argparse, glob, os, statistics as st, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sentinel
ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=int, default=45); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--roots", nargs="+", default=["~/Documentos/Codigo"]); a = ap.parse_args()
ppt = next((d + "/power1_input" for d in glob.glob("/sys/class/hwmon/hwmon*") if open(d + "/name").read().strip() == "amdgpu" and os.path.exists(d + "/power1_input")), None)
assert ppt, "no amdgpu hwmon power1_input"
def watts(seconds=None, work=None):
    xs, t0, f = [], time.perf_counter(), open(ppt)
    while True:
        f.seek(0); xs.append(int(f.read()) / 1e6)
        if work is None and time.perf_counter() - t0 >= seconds: break
        if work is not None and work.poll() is not None: break
        time.sleep(0.05)
    return sum(xs) / len(xs), time.perf_counter() - t0
svc = lambda c: subprocess.run(["systemctl", "--user", c, "npu-sentinel.service"], capture_output=True)
was = subprocess.run(["systemctl", "--user", "is-active", "npu-sentinel.service"], capture_output=True, text=True).stdout.strip() == "active"
off, on = [], []
try:
    for r in range(a.rounds):
        svc("stop"); time.sleep(5); off.append(watts(a.seconds)[0])
        svc("start"); time.sleep(5); on.append(watts(a.seconds)[0])
finally:
    svc("start" if was else "stop")
# one scan, sampled while it runs (the same function the service calls)
code = f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); import sentinel; sentinel.events({a.roots!r})"
base = st.mean(off)
scans = []
for _ in range(3):
    p = subprocess.Popen([sys.executable, "-c", code]); w, t = watts(work=p); scans.append((w, t))
extra_j = st.mean((w - base) * t for w, t in scans)
print(f"idle, service stopped: {[round(x, 2) for x in off]} W   running: {[round(x, 2) for x in on]} W   (APU package, no model)")
print(f"one scan: {st.mean(t for _, t in scans):.2f} s at {st.mean(w for w, _ in scans):.2f} W -> {extra_j:+.1f} J over idle")
print(f"average added power at one scan per 30 min: {extra_j / 1800 * 1000:+.2f} mW  (limit 1000 mW)")
