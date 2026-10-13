#!/bin/sh
# Block 1 pre-flight (bench/PROPUESTA_GPU_SANA.md): exits non-zero, printing why, unless the machine is in the healthy state. Needs nothing installed.
cd "$(dirname "$0")/.." || exit 2
fail() { echo "PREFLIGHT FAIL: $1"; exit 1; }
[ "$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1)" = 1 ] || fail "charger not online"
[ "$(powerprofilesctl get)" = performance ] || fail "power profile is not performance"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || fail "another process is using the GPU"
[ -z "$(pgrep -f 'energy.py|step_breakdown|prefix_reuse|serve.py')" ] || fail "another bench or the server is running"
.venv/bin/python - <<'P' || exit 1
import torch, time, subprocess, sys
a = torch.randn(4096, 4096, device="cuda", dtype=torch.half); b = a.clone()
for _ in range(10): a @ b
torch.cuda.synchronize(); t = time.time()
for _ in range(200): a @ b
torch.cuda.synchronize(); tf = 2 * 4096 ** 3 * 200 / (time.time() - t) / 1e12
q = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.mem", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
print(f"preflight: {tf:.1f} TFLOPS, {q[0]}, mem {q[1]} MHz")
sys.exit(0 if tf >= 10 and q[0] == "P0" and float(q[1]) > 5000 else (print("PREFLIGHT FAIL: GPU not healthy") or 1))
P
