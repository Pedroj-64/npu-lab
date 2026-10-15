"""Continuous measurement from the UI: runs bench/watch.py (short NPU-only benchmark) in the background and reads its log, bench/history.jsonl.
Spanish: medición continua desde la interfaz: lanza bench/watch.py en segundo plano y lee bench/history.jsonl."""
import json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "bench", "history.jsonl")
job = {"state": "idle"}          # idle | running | done | skipped | regression | error
_lock = threading.Lock()


class BenchError(Exception):
    pass


def history(limit=60):
    rows = []
    if os.path.exists(HIST):
        for l in open(HIST):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("decode"):
                rows.append({k: r.get(k) for k in ("ts", "model", "decode", "prefill_256", "ttft_ms", "regression", "kernel", "ac", "gpu_pstate", "commit")})
    return rows[-limit:]


def state():
    return {"job": dict(job), "history": history()}


def start():
    with _lock:
        if job["state"] == "running":
            raise BenchError("a measurement is already running")
        job.clear(); job.update(state="running", started=int(time.time()))

    def work():
        try:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "watch.py")], capture_output=True, text=True, timeout=900, cwd=ROOT)
            st = {0: "done", 2: "regression", 3: "skipped"}.get(r.returncode, "error")
            with _lock:
                job.update(state=st, seconds=int(time.time() - job["started"]), msg=(r.stderr or r.stdout).strip()[-300:])
        except Exception as e:
            with _lock:
                job.update(state="error", msg=f"{type(e).__name__}: {e}")
    threading.Thread(target=work, daemon=True).start()
    return dict(job)
