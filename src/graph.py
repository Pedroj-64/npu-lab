"""Project map with graphify: build or refresh a code graph (AST only, no model, no network) and answer compact queries from it.
Why: a short graph answer costs far fewer tokens than sending whole files to a small local model or to any other assistant.
The graph lives in <project>/graphify-out/ (graph.json, graph.html, GRAPH_REPORT.md). Needs the `graphify` command on PATH.
Spanish: mapa del proyecto con graphify (solo AST, sin modelo ni red); una consulta corta gasta muchos menos tokens que enviar archivos."""
import json, os, shutil, subprocess, threading, time

HOME = os.path.expanduser("~")
BUILD_TIMEOUT, QUERY_TIMEOUT, MAX_OUT = 600, 60, 20000
job = {"state": "idle"}            # idle | running | done | error (one build at a time; polled through GET /project/graph)
_lock = threading.Lock()


class GraphError(Exception):
    pass


def available():
    return shutil.which("graphify") is not None


def check_root(root):
    """Existing folder inside the user's home (same rule as the chat project folder); returns the real path."""
    if not str(root or "").strip():                      # realpath("") would be the server's own folder
        raise GraphError("set a project folder inside your home first")
    real = os.path.realpath(os.path.expanduser(str(root)))
    if not real.startswith(HOME + os.sep) or not os.path.isdir(real):
        raise GraphError("set a project folder inside your home first")
    return real


def graph_file(root):
    return os.path.join(root, "graphify-out", "graph.json")


def _tokens(chars):
    return max(1, round(chars / 4))      # rough: ~4 characters per token; an estimate, not a tokenizer count


def status(root):
    """Facts about the current graph, or {'built': False}."""
    root = check_root(root)
    g = graph_file(root)
    out = {"root": root, "available": available(), "built": os.path.isfile(g), "job": dict(job)}
    if out["built"]:
        d = json.load(open(g))
        srcs = {n.get("source_file") for n in d.get("nodes", []) if n.get("source_file")}
        size = 0
        for s in srcs:
            p = s if os.path.isabs(s) else os.path.join(root, s)
            try:
                size += os.path.getsize(p)
            except OSError:
                pass
        out.update(nodes=len(d.get("nodes", [])), edges=len(d.get("links", d.get("edges", []))), files=len(srcs),
                   source_tokens=_tokens(size), updated=int(os.path.getmtime(g)),
                   communities=len({n.get("community") for n in d.get("nodes", []) if n.get("community") is not None}))
    return out


def _run(args, timeout):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GraphError("graphify took too long")
    except OSError as e:
        raise GraphError(f"cannot run graphify: {e}")
    return r.returncode, (r.stdout + r.stderr).strip()


def start_update(root):
    """Build or refresh the graph in the background (code only: deterministic, free, offline)."""
    root = check_root(root)
    if not available():
        raise GraphError("graphify is not installed (the `graphify` command was not found)")
    with _lock:
        if job["state"] == "running":
            raise GraphError("a graph build is already running")
        job.clear(); job.update(state="running", root=root, started=int(time.time()))

    def work():
        rc, out = _run(["graphify", "update", root], BUILD_TIMEOUT)
        with _lock:
            job.update(state="done" if rc == 0 else "error", seconds=int(time.time() - job["started"]), msg=out[-400:])
    threading.Thread(target=lambda: _guard(work), daemon=True).start()
    return dict(job)


def _guard(fn):
    try:
        fn()
    except GraphError as e:
        with _lock:
            job.update(state="error", msg=str(e))
    except Exception as e:
        with _lock:
            job.update(state="error", msg=f"{type(e).__name__}: {e}")


def query(root, question, budget=1500):
    """Compact answer from the graph. Returns {text, tokens, source_tokens, saved_pct}; tokens are estimates (chars / 4)."""
    root = check_root(root)
    q = " ".join(str(question).split())[:300]
    if not q:
        raise GraphError("write a question")
    if not os.path.isfile(graph_file(root)):
        raise GraphError("build the graph first")
    budget = max(200, min(int(budget), 8000))
    rc, out = _run(["graphify", "query", q.lstrip("-"), "--budget", str(budget), "--graph", graph_file(root)], QUERY_TIMEOUT)
    if rc != 0:
        raise GraphError(out[-300:] or "graphify query failed")
    text = out.replace(root + os.sep, "")[:MAX_OUT]               # no local paths in what the user pastes elsewhere
    st = status(root)
    t = _tokens(len(text))
    return {"text": text, "tokens": t, "source_tokens": st["source_tokens"],
            "saved_pct": max(0, round(100 * (1 - t / st["source_tokens"]))) if st["source_tokens"] else 0}
