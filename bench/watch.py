"""Continuous measurement: short NPU-only benchmark appended to bench/history.jsonl, with regression check.
usage: .venv/bin/python bench/watch.py [--model models/qwen3s] [--threshold 0.10] [--simulate-drop 0.2] [--selftest]
Exit codes: 0 ok, 2 regression vs the median of the last 3 runs of the same model, 3 skipped (NPU busy).
Wraps bench/perf_baseline.py (one process, NPU only). Never run it while the server or another bench uses the NPU.
Spanish: medición continua; añade una corrida corta a bench/history.jsonl y marca regresiones."""
import argparse, json, os, re, statistics as st, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "bench", "history.jsonl")
PY = os.path.join(ROOT, ".venv", "bin", "python")
BUSY = r"src/(serve|chat_cli|qwen|qwen35)|bench/(perf_baseline|energy)"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def parse(table):
    """perf_baseline markdown rows -> {'decode': x, 'prefill_256': y, 'ttft_ms': z} (means)."""
    out = {}
    for line in table.splitlines():
        m = re.match(r"\|\s*(decode tok/s|prefill tok/s, (\d+) tokens|time to first token, ms)[^|]*\|\s*([\d.]+)", line)
        if m:
            key = "decode" if m.group(1).startswith("decode") else f"prefill_{m.group(2)}" if m.group(2) else "ttft_ms"
            out[key] = float(m.group(3))
    return out


def state():
    return {"kernel": sh("uname -r"), "ac": sh("cat /sys/class/power_supply/A*/online"),
            "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "gpu_pstate": sh("nvidia-smi --query-gpu=pstate --format=csv,noheader"),
            "commit": sh(f"git -C {ROOT} rev-parse --short HEAD")}


def last(model, n=3):
    rows = []
    if os.path.exists(HIST):
        for l in open(HIST):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("model") == model and r.get("decode"):
                rows.append(r)
    return rows[-n:]


def regression(new, old, thr):
    """True if decode or prefill_256 fell more than thr below the median of old (higher is better)."""
    for k in ("decode", "prefill_256"):
        base = [r[k] for r in old if k in r]
        if base and k in new and new[k] < st.median(base) * (1 - thr):
            return True
    return False


def selftest():
    t = "| decode tok/s (16 steps) | 19.83 ± 0.27 |\n| prefill tok/s, 256 tokens | 187.10 ± 2.00 |\n| time to first token, ms (chat) | 405.0 ± 3 |"
    assert parse(t) == {"decode": 19.83, "prefill_256": 187.10, "ttft_ms": 405.0}, parse(t)
    assert regression({"decode": 15, "prefill_256": 187}, [{"decode": 20, "prefill_256": 187}] * 3, 0.10)
    assert not regression({"decode": 19, "prefill_256": 187}, [{"decode": 20, "prefill_256": 187}] * 3, 0.10)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3s")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--simulate-drop", type=float, default=0.0, help="multiply measured numbers by (1 - x) to test the regression check; nothing is written to history")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if sh(f"pgrep -af '{BUSY}' | grep -v pgrep | grep -v watch.py"):
        print("NPU busy, skipped", file=sys.stderr); sys.exit(3)
    r = subprocess.run([PY, os.path.join(ROOT, "bench", "perf_baseline.py"), "--model", a.model, "--runs", "3", "--warmup", "1",
                        "--tokens", "16", "--prefill", "256"], capture_output=True, text=True, cwd=ROOT)
    res = parse(r.stdout)
    if "decode" not in res:
        print(r.stdout, r.stderr, file=sys.stderr); sys.exit(1)
    if a.simulate_drop:
        res = {k: (v * (1 - a.simulate_drop) if k != "ttft_ms" else v) for k, v in res.items()}
    prev = last(a.model)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": a.model, **res, **state()}
    row["regression"] = regression(res, prev, a.threshold)
    if not a.simulate_drop:                                  # a simulated number must never become a baseline
        with open(HIST, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    sys.exit(2 if row["regression"] else 0)


main()
