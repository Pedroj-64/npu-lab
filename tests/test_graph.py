"""Offline test for src/graph.py (graphify adapter): folder rule, build, status and a compact query. Needs the `graphify` command; skips without it.
usage: python tests/test_graph.py"""
import os, shutil, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import graph

for bad in ("/tmp", "", "~/no-such-folder-xyz"):
    try:
        graph.check_root(bad); raise SystemExit(f"accepted {bad!r}")
    except graph.GraphError:
        pass
if not graph.available():
    print("ok (graphify not installed: only the folder rule was checked)"); sys.exit(0)

root = tempfile.mkdtemp(prefix=".graph-test-", dir=graph.HOME)           # inside the home, as the rule requires
try:
    open(os.path.join(root, "a.py"), "w").write("import json\n\ndef load(path):\n    return json.load(open(path))\n\ndef save(path, obj):\n    json.dump(obj, open(path, 'w'))\n")
    open(os.path.join(root, "b.py"), "w").write("from a import load\n\ndef count(path):\n    return len(load(path))\n")
    assert graph.status(root)["built"] is False
    try:
        graph.query(root, "who loads json"); raise SystemExit("query worked before a build")
    except graph.GraphError:
        pass
    graph.start_update(root)
    for _ in range(120):
        if graph.job["state"] != "running": break
        time.sleep(0.5)
    assert graph.job["state"] == "done", graph.job
    st = graph.status(root)
    assert st["built"] and st["nodes"] > 0 and st["files"] >= 2 and st["source_tokens"] > 0, st
    r = graph.query(root, "which function loads json", 800)
    assert r["text"] and r["tokens"] > 0 and 0 <= r["saved_pct"] <= 100, r
    print("ok", {k: st[k] for k in ("nodes", "edges", "files")}, "query tokens", r["tokens"], "source tokens", r["source_tokens"])
finally:
    shutil.rmtree(root, ignore_errors=True)

# fallback without graphify: the built-in Python map answers, flagged as a fallback
import shutil as _sh
root2 = tempfile.mkdtemp(prefix=".graph-test-", dir=graph.HOME)
try:
    open(os.path.join(root2, "m.py"), "w").write("def alpha(x):\n    '''Adds one.'''\n    return x + 1\n")
    real = graph.available; graph.available = lambda: False
    r = graph.query(root2, "what does alpha do")
    graph.available = real
    assert r["fallback"] and "alpha(x)" in r["text"] and r["source_tokens"] > 0, r
    print("fallback ok")
finally:
    _sh.rmtree(root2, ignore_errors=True)
