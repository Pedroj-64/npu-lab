"""Offline test for src/bench_job.py: the history is read from the log, skipping broken lines and rows without a decode figure. usage: python tests/test_bench_job.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import bench_job
f = tempfile.mktemp(suffix=".jsonl"); bench_job.HIST = f
assert bench_job.history() == [], "missing log -> empty"
open(f, "w").write(json.dumps({"ts": "2026-10-15T08:40:09", "model": "m", "decode": 20.5, "prefill_256": 240.7, "x": 1}) + "\n" + "not json\n" + json.dumps({"ts": "t", "model": "m"}) + "\n"
                   + "".join(json.dumps({"ts": f"t{i}", "model": "m", "decode": 20 + i}) + "\n" for i in range(80)))
h = bench_job.history(60)
assert len(h) == 60 and h[-1]["decode"] == 99 and "x" not in h[0], h[-1]
assert bench_job.state()["job"]["state"] == "idle"
os.remove(f); print("bench_job ok")
