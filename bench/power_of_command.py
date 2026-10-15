"""Whole-machine power while a command runs: mean watts (APU package from hwmon + NVIDIA board) and joules, with the idle level measured just before.
usage: .venv/bin/python bench/power_of_command.py [--idle 15] -- COMMAND ARGS...      (env vars are inherited; leave the PC alone while it runs)
Used for the Whisper-on-NPU comparison (docs/DECISIONS.md, T4.1/T4.2). Not counted: screen, SSD, RAM outside the package, fans, charger losses.
Spanish: potencia de toda la máquina mientras corre un comando: vatios medios, julios y nivel en reposo medido justo antes."""
import argparse, glob, subprocess, sys, threading, time
ap = argparse.ArgumentParser(); ap.add_argument("--idle", type=float, default=15); ap.add_argument("cmd", nargs=argparse.REMAINDER); a = ap.parse_args()
cmd = [c for c in a.cmd if c != "--"]
assert cmd, "give a command after --"
ppt = next((d + "/power1_input" for d in glob.glob("/sys/class/hwmon/hwmon*") if open(d + "/name").read().strip() == "amdgpu" and open(d + "/power1_input")), None)
apu, gpu, run = [], [], True
def s_apu():
    f = open(ppt)
    while run:
        f.seek(0); apu.append((time.perf_counter(), int(f.read()) / 1e6)); time.sleep(0.05)
def s_gpu():
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        gpu.append((time.perf_counter(), float(l)))
        if not run: p.terminate(); break
threading.Thread(target=s_apu, daemon=True).start(); threading.Thread(target=s_gpu, daemon=True).start()
time.sleep(1)
mean = lambda s, t0, t1: (lambda xs: sum(xs) / len(xs) if xs else float("nan"))([w for t, w in s if t0 <= t <= t1])
t0 = time.perf_counter(); time.sleep(a.idle); t1 = time.perf_counter()
idle = mean(apu, t0, t1) + mean(gpu, t0, t1)
t2 = time.perf_counter(); r = subprocess.run(cmd); t3 = time.perf_counter(); run = False
p = mean(apu, t2, t3) + mean(gpu, t2, t3)
print(f"POWER idle {idle:.2f} W | running {p:.2f} W | {t3 - t2:.1f} s | total {p * (t3 - t2):.0f} J | above idle {(p - idle) * (t3 - t2):.0f} J | exit {r.returncode}")
