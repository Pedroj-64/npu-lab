"""Decode / prefill / time-to-first-token baseline, NPU only (GPU helpers disabled), one model per process.
usage: .venv/bin/python bench/perf_baseline.py --model models/qwen3s [--runs 5 --warmup 2 --tokens 32]
Prints a markdown row block (mean +- std over --runs, after --warmup discarded runs) plus the machine state."""
import argparse, os, subprocess, sys, time
os.environ["NPU_GPU"] = "0"                      # NPU only: no GPU attention / prefill, so the numbers are the NPU's own
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import qwen_npu as Q


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def stat(xs):
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"


def repeat(fn, runs, warmup):
    return [fn() for i in range(warmup + runs)][warmup:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3s")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=32, help="decode steps per run")
    ap.add_argument("--prefill", default="256,1024,2048")
    a = ap.parse_args()
    m = Q.Qwen(a.model, gpu=False)
    rng = np.random.default_rng(0)
    print(f"## {a.model}  prefill kernels: {'yes' if m.pf else 'NO (token-by-token fallback)'}", flush=True)
    print(f"governor={sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')} power-profile={sh('powerprofilesctl get')} "
          f"AC={sh('cat /sys/class/power_supply/A*/online')} kernel={sh('uname -r')}")

    def decode():                                # a short real prompt is read first so attention has a realistic (tiny) context
        ids = rng.integers(1000, 100000, 16).tolist()
        m.cache_ids = []
        logits = m.prefill(ids, 0)
        pos, t = len(ids), time.perf_counter()
        for _ in range(a.tokens):
            logits = m.step(int(np.argmax(logits)), pos); pos += 1
        return a.tokens / (time.perf_counter() - t)

    def prefill(n):
        ids = rng.integers(1000, 100000, n).tolist()
        def run():
            m.cache_ids = []
            t = time.perf_counter(); m.prefill(ids, 0)
            return n / (time.perf_counter() - t)
        return run

    def ttft():
        m.cache_ids = []
        t = time.perf_counter()
        next(m.stream("What is the capital of France? Answer in one sentence.", 4))
        return (time.perf_counter() - t) * 1000

    print("| metric | mean ± std |\n|---|---|", flush=True)
    print(f"| decode tok/s ({a.tokens} steps) | {stat(repeat(decode, a.runs, a.warmup))} |", flush=True)
    for n in map(int, a.prefill.split(",")):
        print(f"| prefill tok/s, {n} tokens | {stat(repeat(prefill(n), a.runs, a.warmup))} |", flush=True)
    print(f"| time to first token, ms (chat prompt, ~40 tokens) | {stat(repeat(ttft, a.runs, a.warmup))} |", flush=True)


main()
