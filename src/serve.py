"""Local engine server for the launcher (one mode loaded at a time: LLM chat or semantic search).

  NPU_LAB_TOKEN=... python src/serve.py        # prints {"port": N} on stdout when ready

Serves the web UI (web/index.html) at / and a JSON API under the same origin.
Security model: binds 127.0.0.1 on a random port; every API request needs `Authorization: Bearer <token>`
(token from the NPU_LAB_TOKEN env var, or generated and printed; the launcher hands it to the page in the
URL fragment, which never reaches the server); the Host header must be loopback; an Origin header is only
accepted if it is this server's own origin; no CORS headers; only whitelisted models; folders only under $HOME.

API (JSON):  POST /login (one-time code -> token)   GET /status /rules /doctor   POST /mode /profile /rules /chat (ndjson stream) /search /index /index/stop
"""
import gc, hmac, json, os, secrets, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = {"qwen": "models/qwen", "qwen15": "models/qwen15"}   # whitelist: the API never takes a path
PROFILES = ("npu_only", "assisted")                           # assisted: GPU/CPU may help bulk indexing
MAX_BODY, MAX_PROMPT, MAX_TOKENS = 64 * 1024, 8000, 1024
TOKEN = os.environ.get("NPU_LAB_TOKEN") or secrets.token_urlsafe(32)
CODE = os.environ.get("NPU_LAB_CODE")      # one-time login code the launcher puts in the URL (swapped for TOKEN once)
_code_lock = threading.Lock()


class State:
    def __init__(self):
        self.mode, self.model, self.engine = "none", None, None
        self.profile = "npu_only"
        self.npu_lock = threading.Lock()          # one NPU job at a time
        self.idx = None
        self.job = {"state": "idle"}              # index job status
        self.stop = threading.Event()
        self.docs, self.embedder = None, None     # local documentation index + a CPU query embedder (~110 ms per question)

    def unload(self):
        self.engine, self.mode, self.model = None, "none", None
        gc.collect()                              # drops pyxrt contexts/buffers -> NPU is free again

    def load_index(self):
        import search
        self.idx = search.load()

    def docs_info(self):
        import docs
        if self.docs is None:
            self.docs = docs.load()
        return None if not self.docs else {"chunks": len(self.docs["text"]), "built": self.docs["built"]}

    def index_info(self):
        if self.idx is None:
            self.load_index()
        return {"files": sum(1 for v in self.idx.values() if len(v["snip"])),
                "chunks": sum(len(v["snip"]) for v in self.idx.values())}


S = State()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_rules(d):
    """Only two knobs are editable from the UI: which folders may be read (under $HOME) and the GPU VRAM cap."""
    import search
    r = search.rules()
    home = os.path.realpath(os.path.expanduser("~"))
    if "roots" in d:
        roots = d["roots"]
        if not isinstance(roots, list) or not 1 <= len(roots) <= 20:
            raise Bad(400, "roots: 1..20 folders")
        clean = []
        for x in roots:
            real = os.path.realpath(os.path.expanduser(str(x)))
            if real == home or not real.startswith(home + os.sep) or not os.path.isdir(real):
                raise Bad(400, f"not an existing folder inside your home (and not the whole home): {x}")
            clean.append("~" + real[len(home):])
        r["roots"] = clean
    if "gpu_vram_gb" in d:
        g = float(d["gpu_vram_gb"])
        if not 0.5 <= g <= 4:
            raise Bad(400, "gpu_vram_gb must be between 0.5 and 4")
        r["gpu_vram_gb"] = g
    os.makedirs(os.path.dirname(search.CFG), exist_ok=True)
    json.dump(r, open(search.CFG, "w"), indent=2, ensure_ascii=False)
    return r


def doctor():
    """First-run checklist (like an onboarding wizard): what works, what is missing, how to fix it."""
    import importlib.util, platform
    out = []

    def add(id, ok, detail, fix=""):
        out.append({"id": id, "ok": bool(ok), "detail": detail, "fix": "" if ok else fix})
    linux = platform.system() == "Linux"
    add("os", linux, platform.system(), "NPU backends are Linux-only for now; the web UI and CPU search still work.")
    dev = linux and os.path.exists("/dev/accel/accel0")
    add("npu_device", dev, "/dev/accel/accel0" if dev else "no NPU device node",
        "Needs the amdxdna kernel driver (Linux 6.14+ ships it) and an XDNA1 NPU.")
    px = importlib.util.find_spec("pyxrt") is not None
    add("xrt", px, "pyxrt importable" if px else "pyxrt not found", "sudo pacman -S xrt xrt-plugin-amdxdna, then link pyxrt into the venv (see README).")
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        add("memlock", soft == resource.RLIM_INFINITY, "unlimited" if soft == resource.RLIM_INFINITY else f"{soft} bytes",
            "XRT needs unlimited locked memory: add '* soft memlock unlimited' and '* hard memlock unlimited' to /etc/security/limits.d/99-npu.conf, then log out and in.")
    except ImportError:
        add("memlock", False, "not applicable here", "")
    from qwen_npu import shapes
    import prefill_npu
    for name, d in MODELS.items():
        has_model = os.path.exists(os.path.join(ROOT, d, "model.safetensors"))
        add(f"model_{name}", has_model, d if has_model else f"{d}/model.safetensors missing",
            "Download Qwen/Qwen2.5-" + ("0.5B" if name == "qwen" else "1.5B") + f"-Instruct (model.safetensors, tokenizer.json, config.json) into {d}/.")
        if has_model:
            need = [f"c2_{m}x{k}" for m, k in shapes(json.load(open(os.path.join(ROOT, d, "config.json"))))]
            miss = [n for n in need if not os.path.exists(os.path.join(ROOT, "build/c2", n + ".xclbin"))]
            add(f"kernels_{name}", not miss, "all built" if not miss else f"{len(miss)} missing",
                f"python scripts/build_matvec.py --shared $(python src/qwen_npu.py --model {d} --shapes)")
            cfg = json.load(open(os.path.join(ROOT, d, "config.json")))
            missp = [x for x in prefill_npu.shapes(cfg) if not os.path.exists(os.path.join(ROOT, prefill_npu.xclbin_name(*x) + ".xclbin"))]
            add(f"prefill_{name}", not missp, "all built" if not missp else f"{len(missp)} missing (prompts will be read slowly)",
                f"python scripts/build_prefill.py {d}")
    import docs
    add("docs_index", os.path.exists(docs.IDX), "built" if os.path.exists(docs.IDX) else "not built (chat cannot use local documentation)",
        "python src/docs.py --device gpu build   (needs the tldr cache: `tldr --update`; optional: sudo pacman -S arch-wiki-docs). Without a GPU use --device npu (slower).")
    emb = os.path.exists(os.path.join(ROOT, "models/multi/model.safetensors"))
    add("model_search", emb, "models/multi" if emb else "models/multi missing",
        "Download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (model.safetensors, tokenizer.json, config.json) into models/multi/.")
    import search
    add("rules", os.path.exists(search.CFG), search.CFG.replace(os.path.expanduser("~"), "~"), "Created on first use; open the Search page to review it.")
    add("gpu_helper", importlib.util.find_spec("torch") is not None, "torch present (optional)" if importlib.util.find_spec("torch") else "optional, not installed",
        "Optional: pip install torch for the fast first-time indexing helper.")
    return out


class Bad(Exception):
    def __init__(self, code, msg):
        self.code, self.msg = code, msg


def set_mode(mode, model=None, device="npu"):
    if mode not in ("none", "llm", "search"):
        raise Bad(400, "mode must be none|llm|search")
    if S.job.get("state") == "running":
        raise Bad(409, "an index job is running")
    if not S.npu_lock.acquire(blocking=False):
        raise Bad(409, "busy")
    try:
        S.unload()
        if mode == "llm":
            if model not in MODELS:
                raise Bad(400, f"model must be one of {sorted(MODELS)}")
            from qwen_npu import Qwen
            S.engine, S.mode, S.model = Qwen(MODELS[model]), "llm", model
        elif mode == "search":
            if device not in ("npu", "gpu", "cpu"):
                raise Bad(400, "device must be npu|gpu|cpu")
            if device != "npu" and S.profile == "npu_only":
                raise Bad(403, "profile npu_only: GPU/CPU helpers are disabled")
            import search
            S.engine, S.mode, S.model = search.make_model(device), "search", device
            S.load_index()
    finally:
        S.npu_lock.release()


def start_index(device):
    import search
    if S.job.get("state") == "running":
        raise Bad(409, "an index job is already running")
    if device not in ("npu", "gpu", "cpu"):
        raise Bad(400, "device must be npu|gpu|cpu")
    if device != "npu" and S.profile == "npu_only":
        raise Bad(403, "profile npu_only: GPU/CPU helpers are disabled")
    if device == "npu" and S.mode != "search":
        raise Bad(409, "switch to search mode first (it holds the NPU embedding model)")
    if device == "npu" and not S.npu_lock.acquire(blocking=False):
        raise Bad(409, "busy")
    S.stop.clear()
    S.job = {"state": "running", "device": device, "files_done": 0, "files_total": 0, "chunks": 0, "rate": 0.0}

    def work():
        try:
            def prog(i):
                S.job.update(i)
            r = search.run_index(device, progress=prog, stop=S.stop.is_set,
                                 model=S.engine if device == "npu" else None)
            S.load_index()
            S.job = {"state": "aborted" if r["aborted"] else "done", **r}
        except Exception as e:  # report, never crash the server
            S.job = {"state": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            if device == "npu":
                S.npu_lock.release()

    threading.Thread(target=work, daemon=True).start()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # close after each response: streaming needs no chunking

    def log_message(self, *a):         # no access log (paths and prompts are private)
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _check_host(self):
        port = self.server.server_address[1]
        if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            raise Bad(403, "bad host")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            raise Bad(403, "foreign origin")

    def _auth(self):
        self._check_host()
        got = self.headers.get("Authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {TOKEN}".encode()):
            raise Bad(401, "unauthorized")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise Bad(413, "body too large")
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise Bad(400, "invalid JSON")
        if not isinstance(d, dict):
            raise Bad(400, "JSON object expected")
        return d

    def _handle(self, fn):
        try:
            self._auth()
            fn()
        except Bad as e:
            self._send(e.code, {"error": e.msg})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                self._check_host()
                page = open(os.path.join(ROOT, "web", "index.html"), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                 "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            except Bad as e:
                self._send(e.code, {"error": e.msg})
            return
        self._handle(self._get)

    def do_POST(self):
        if self.path == "/login":
            return self._login()
        self._handle(self._post)

    def _login(self):
        global CODE
        try:
            self._check_host()
            d = self._body()
            with _code_lock:
                ok = CODE is not None and hmac.compare_digest(str(d.get("code", "")).encode(), CODE.encode())
                if ok:
                    CODE = None                  # single use
            if not ok:
                raise Bad(401, "invalid or already used code")
            self._send(200, {"token": TOKEN})
        except Bad as e:
            self._send(e.code, {"error": e.msg})

    def _get(self):
        import importlib.util
        if self.path == "/status":
            self._send(200, {"mode": S.mode, "model": S.model, "profile": S.profile,
                             "busy": S.npu_lock.locked(), "index": S.index_info(), "docs": S.docs_info(), "job": S.job,
                             "gpu_helper": importlib.util.find_spec("torch") is not None,
                             "models": sorted(MODELS)})
        elif self.path == "/rules":
            import search
            self._send(200, search.rules())
        elif self.path == "/doctor":
            self._send(200, {"checks": doctor()})
        else:
            raise Bad(404, "not found")

    def _post(self):
        d = self._body()
        if self.path == "/mode":
            set_mode(d.get("mode"), d.get("model"), d.get("device", "npu"))
            self._send(200, {"mode": S.mode, "model": S.model})
        elif self.path == "/profile":
            if d.get("profile") not in PROFILES:
                raise Bad(400, f"profile must be one of {PROFILES}")
            S.profile = d["profile"]
            self._send(200, {"profile": S.profile})
        elif self.path == "/rules":
            self._send(200, set_rules(d))
        elif self.path == "/chat":
            self._chat(d)
        elif self.path == "/search":
            self._search(d)
        elif self.path == "/index":
            start_index(d.get("device", "npu"))
            self._send(202, {"started": True})
        elif self.path == "/index/stop":
            S.stop.set()
            self._send(200, {"stopping": True})
        else:
            raise Bad(404, "not found")

    def _chat(self, d):
        prompt, n, use_docs = str(d.get("prompt", "")), int(d.get("tokens", 256)), bool(d.get("docs", False))
        if S.mode != "llm":
            raise Bad(409, "not in llm mode")
        if not prompt.strip() or len(prompt) > MAX_PROMPT or not 1 <= n <= MAX_TOKENS:
            raise Bad(400, f"prompt 1..{MAX_PROMPT} chars, tokens 1..{MAX_TOKENS}")
        if not S.npu_lock.acquire(blocking=False):
            raise Bad(409, "busy")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                ctx = None
                if use_docs:
                    import docs, search
                    if S.docs_info() is None:
                        raise RuntimeError("docs index not built: run `python src/docs.py --device gpu build` (or --device npu)")
                    if S.embedder is None:
                        S.embedder = search.make_model("cpu")
                    ctx = docs.retrieve(S.embedder, prompt, 4, S.docs)
                    self.wfile.write((json.dumps({"sources": [{"src": c["src"], "title": c["title"], "score": round(c["score"], 3)} for c in ctx]}) + "\n").encode())
                    self.wfile.flush()
                for piece in S.engine.stream(prompt, n, ctx):
                    self.wfile.write((json.dumps({"token": piece}) + "\n").encode()); self.wfile.flush()
                self.wfile.write((json.dumps({"done": True, "stats": S.engine.stats}) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                pass                      # client left: stop generating
            except Exception as e:
                self.wfile.write((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode())
        finally:
            S.npu_lock.release()

    def _search(self, d):
        import search
        q = str(d.get("query", "")).strip()
        if S.mode != "search":
            raise Bad(409, "not in search mode")
        if not q or len(q) > 500:
            raise Bad(400, "query 1..500 chars")
        if not S.npu_lock.acquire(blocking=False):
            raise Bad(409, "busy")
        try:
            self._send(200, {"results": search.run_search(S.engine, q, min(int(d.get("n", 8)), 50), S.idx)})
        finally:
            S.npu_lock.release()


def watch_parent():
    """The launcher keeps our stdin open; when it dies for any reason (crash, kill -9, closed terminal) the pipe
    closes, stdin hits EOF and we exit, so an orphaned server never keeps the NPU locked."""
    def run():
        try:
            sys.stdin.read()
        finally:
            os._exit(0)
    threading.Thread(target=run, daemon=True).start()


def main():
    if os.environ.get("NPU_LAB_WATCH_STDIN") == "1":
        watch_parent()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    print(json.dumps({"port": srv.server_address[1], **({} if os.environ.get("NPU_LAB_TOKEN") else {"token": TOKEN})}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
