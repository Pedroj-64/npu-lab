"""Local engine server for the launcher (one mode loaded at a time: LLM chat or semantic search).

  NPU_LAB_TOKEN=... python src/serve.py        # prints {"port": N} on stdout when ready

Serves the web UI (web/index.html) at / and a JSON API under the same origin.
Security model: binds 127.0.0.1 on a random port; every API request needs `Authorization: Bearer <token>`
(token from the NPU_LAB_TOKEN env var, or generated and printed; the launcher hands it to the page in the
URL fragment, which never reaches the server); the Host header must be loopback; an Origin header is only
accepted if it is this server's own origin; no CORS headers; only whitelisted models; folders only under $HOME.

API (JSON):  POST /login (one-time code -> token)   GET /status /rules /routing /documents /chats /facts /doctor /settings   POST /routing /facts /chats/save /chats/load /chats/delete /documents/upload /documents/delete /settings /settings/models /settings/test_web /mode /profile /rules /chat (ndjson stream) /search /index /index/stop
"""
import gc, hmac, json, os, secrets, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Qwen3.5 (hybrid) needs the GPU: Assisted profile only. Qwen3 (dense) also runs NPU only.
MODELS = {"qwen35s": "models/qwen35_08b", "qwen35m": "models/qwen35_2b", "qwen35l": "models/qwen35_4b", "qwen3s": "models/qwen3s", "qwen3": "models/qwen3"}
NEEDS_GPU = ("qwen35s", "qwen35m", "qwen35l")   # whitelist: the API never takes a path
PROFILES = ("npu_only", "assisted")                           # assisted: GPU/CPU may help bulk indexing
MAX_BODY, CHAT_BODY, MAX_TOKENS = 64 * 1024, 16 * 1024 * 1024, 4096     # a chat request may carry a whole long conversation or document
DEFAULT_CONTEXT = 80000                                                    # tokens; the window actually used is cut to what fits in the GPU memory limit
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
        self.emb_lock = threading.Lock()          # the CPU embedder is shared by chat retrieval and document import
        self.docjob = {"state": "idle"}           # import status of "my documents"

    def cpu_embedder(self):
        import search

        class Locked:                             # encode() one caller at a time
            def __init__(s, m): s.m = m
            def encode(s, texts):
                with self.emb_lock:
                    return s.m.encode(texts)
        if self.embedder is None:
            self.embedder = Locked(search.make_model("cpu"))
        return self.embedder

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
    """Only two knobs are editable from the UI: which folders may be read (your home or inside it) and the GPU VRAM cap."""
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
            if not (real == home or real.startswith(home + os.sep)) or not os.path.isdir(real):
                raise Bad(400, f"must be an existing folder inside your home (use ~ for the whole home): {x}")
            clean.append("~" + real[len(home):])
        r["roots"] = clean
    if "state_cache_gb" in d:
        v = float(d["state_cache_gb"])
        if not 0 <= v <= 100:
            raise Bad(400, "state_cache_gb must be between 0 (off) and 100")
        r["state_cache_gb"] = v
    if "gpu_share" in d:
        v = float(d["gpu_share"])
        if v not in (0.0, 0.25, 0.5):
            raise Bad(400, "gpu_share must be 0 (NPU only), 0.25 or 0.5")
        r["gpu_share"] = v
    if "compact_at" in d:
        a = float(d["compact_at"])
        if not (a == 0 or 0.3 <= a <= 0.95):
            raise Bad(400, "compact_at must be 0 (off) or between 0.3 and 0.95")
        r["compact_at"] = a
    if "llm_vram_gb" in d:
        g = float(d["llm_vram_gb"])
        if not 1.0 <= g <= 6.0:
            raise Bad(400, "llm_vram_gb must be between 1 and 6")
        r["llm_vram_gb"] = g
    if "context_tokens" in d:
        n = int(d["context_tokens"])
        if not 2048 <= n <= 262144:
            raise Bad(400, "context_tokens must be between 2048 and 262144")
        r["context_tokens"] = n
    if "save_chats" in d:
        r["save_chats"] = bool(d["save_chats"])
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
    HF = {"qwen3": "Qwen/Qwen3-1.7B", "qwen3s": "Qwen/Qwen3-0.6B", "qwen35s": "Qwen/Qwen3.5-0.8B", "qwen35m": "Qwen/Qwen3.5-2B", "qwen35l": "Qwen/Qwen3.5-4B"}
    for name, d in MODELS.items():
        has_model = os.path.isdir(os.path.join(ROOT, d)) and any(f.endswith(".safetensors") for f in os.listdir(os.path.join(ROOT, d)))
        if not has_model and name not in ("qwen3", "qwen35m"):
            continue                                  # optional models: nothing to report until you download them
        add(f"model_{name}", has_model, d if has_model else f"{d}/model.safetensors missing",
            f"Download {HF[name]} (safetensors, tokenizer.json, config.json) into {d}/.")
        if has_model:
            cfg = json.load(open(os.path.join(ROOT, d, "config.json")))
            import qwen35_npu
            import search as _s
            share = float(_s.rules().get("gpu_share", 0.0))
            need = [f"c4_{m}x{k}" for m, k in (qwen35_npu.shapes(qwen35_npu.text_config(cfg), share) if name in NEEDS_GPU else shapes(cfg))]
            miss = [n for n in need if not os.path.exists(os.path.join(ROOT, "build/c4", n + ".xclbin"))]
            add(f"kernels_{name}", not miss, "all built" if not miss else f"{len(miss)} missing",
                f"python scripts/build_matvec.py --shared4 $(python src/qwen35_npu.py --model {d} --shapes" + (f" --share {share}" if name in NEEDS_GPU and share else "") + ")")
            if name not in NEEDS_GPU:                 # Qwen3.5 reads prompts on the GPU: no NPU prefill kernels
                missp = [x for x in prefill_npu.shapes(cfg) if not os.path.exists(os.path.join(ROOT, prefill_npu.xclbin_name(*x) + ".xclbin"))]
                add(f"prefill_{name}", not missp, "all built" if not missp else f"{len(missp)} missing (prompts will be read slowly)",
                    f"python scripts/build_prefill.py {d}")
    import shutil
    add("documents_pdf", bool(shutil.which("pdftotext")), "pdftotext found" if shutil.which("pdftotext") else "pdftotext missing (PDF files cannot be read)", "sudo pacman -S poppler")
    add("documents_ocr", bool(shutil.which("tesseract")), "tesseract found" if shutil.which("tesseract") else "optional: tesseract missing (scanned PDFs and images cannot be read)",
        "sudo pacman -S tesseract tesseract-data-spa tesseract-data-eng")
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


SUMMARY_SYSTEM = ("You compress a conversation into notes for yourself. Keep every name, number, file name, command, code identifier, decision, "
                  "constraint and open question; drop greetings and repetition. Write in the language of the conversation, as plain bullet points, "
                  "at most 300 words.")


def compact(history, window, at, keep, phase):
    """When the conversation passes `at` of the context window, summarise all but the last `keep` turns with the loaded model.
    Returns (history, None) untouched, or (notes + recent turns, {"summary", "kept"}). at = 0 disables it."""
    tok = S.engine.tok
    count = lambda t: len(tok.encode(t, add_special_tokens=False).ids)
    if not at or len(history) <= keep * 2 or sum(count(m["content"]) + 8 for m in history) < window * at:
        return history, None
    old, recent = history[:-keep * 2], history[-keep * 2:]
    ids = tok.encode("\n".join(f"{m['role']}: {m['content']}" for m in old), add_special_tokens=False).ids
    step = max(512, int(window * 0.4))                    # pieces that fit comfortably next to the notes and the instructions
    pieces = [tok.decode(ids[i:i + step]) for i in range(0, len(ids), step)]
    notes = ""
    for k, piece in enumerate(pieces):
        phase(done=k, of=len(pieces))
        notes = S.engine.complete(f"Previous notes:\n{notes or '(none)'}\n\nNext part of the conversation:\n{piece}\n\nWrite the updated notes.",
                                  SUMMARY_SYSTEM, 400).strip()
    import pins
    notes = pins.block(pins.pin(old)) + notes                   # the user's own fact lines word for word: a summary may drop or change a digit
    pair = [{"role": "user", "content": "[Notes of the earlier conversation]\n" + notes}, {"role": "assistant", "content": "Understood, I will use those notes."}]
    return pair + recent, {"summary": notes, "kept": len(recent)}


class Bad(Exception):
    def __init__(self, code, msg):
        self.code, self.msg = code, msg


def set_mode(mode, model=None, device="npu"):
    if mode not in ("none", "llm", "search"):
        raise Bad(400, "mode must be none|llm|search")
    j = S.job
    if j.get("state") == "running" and j.get("device") == "npu":   # a GPU/CPU index job does not hold the NPU
        done = f" ({j.get('files_done', 0)}/{j.get('files_total') or '?'} files)" if j.get("files_total") else ""
        raise Bad(409, f"Indexing is using the NPU{done}. Wait for it to finish, or press Stop in the Search tab.")
    if not S.npu_lock.acquire(blocking=False):
        raise Bad(409, "The NPU is busy answering a request. Try again in a moment.")
    try:
        S.unload()
        if mode == "llm":
            if model not in MODELS:
                raise Bad(400, f"model must be one of {sorted(MODELS)}")
            from qwen35_npu import load_engine
            import search
            if model in NEEDS_GPU and S.profile != "assisted":
                raise Bad(409, "Qwen3.5 models use the GPU for prompt reading and attention: switch to the Assisted profile (or pick a Qwen3 model for NPU only)")
            rl = search.rules()
            # assisted: the GPU reads prompts and holds the KV cache / DeltaNet state; the NPU still generates every token
            S.engine, S.mode, S.model = load_engine(MODELS[model], gpu=S.profile == "assisted", max_ctx=int(rl.get("context_tokens", DEFAULT_CONTEXT)), vram_gb=float(rl.get("llm_vram_gb", 4.0)), gpu_share=float(rl.get("gpu_share", 0.0)), cache_gb=float(rl.get("state_cache_gb", 8.0))), "llm", model
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

    def _body(self, limit=MAX_BODY):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
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

    def _upload(self):
        """POST /documents/upload?name=file.pdf with the raw file as body: imported in the background (poll GET /documents)."""
        import documents, urllib.parse
        n = int(self.headers.get("Content-Length") or 0)
        if n > documents.MAX_BYTES:
            raise Bad(413, f"file too large (max {documents.MAX_BYTES // 2**20} MB)")
        name = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("name", ["document"])[0]
        data = self.rfile.read(n)
        if S.docjob["state"] == "running":
            raise Bad(409, "another document is still being imported")
        S.docjob = {"state": "running", "name": os.path.basename(name)[:120], "msg": "starting"}

        def work():
            try:
                documents.add(S.cpu_embedder(), name, data, lambda m: S.docjob.update(msg=m))
                S.docjob = {"state": "done", "name": os.path.basename(name)[:120]}
            except documents.DocError as e:
                S.docjob = {"state": "error", "name": os.path.basename(name)[:120], "error": str(e)}
            except Exception as e:
                S.docjob = {"state": "error", "name": os.path.basename(name)[:120], "error": f"{type(e).__name__}: {e}"}
        threading.Thread(target=work, daemon=True).start()
        self._send(202, S.docjob)

    def do_POST(self):
        if self.path == "/login":
            return self._login()
        if self.path.startswith("/documents/upload"):
            return self._handle(self._upload)
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
                             "gpu_helper": importlib.util.find_spec("torch") is not None, "gpu_assist": S.engine is not None and getattr(S.engine, "g", None) is not None,
                             "context": getattr(S.engine, "max_ctx", None), "context_limited": getattr(S.engine, "ctx_limited", False),
                             "cloud": bool(__import__("settings").load()["cloud"]["active"]),
                             "models": [m for m in MODELS if os.path.isdir(os.path.join(ROOT, MODELS[m]))]})
        elif self.path == "/routing":
            import router
            self._send(200, {**router.load(), "defaults": router.DEFAULTS})
        elif self.path == "/documents":
            import documents
            self._send(200, {"documents": documents.listing(), "job": S.docjob})
        elif self.path == "/rules":
            import search
            self._send(200, search.rules())
        elif self.path == "/facts":
            import facts
            self._send(200, {"facts": facts.load()})
        elif self.path == "/chats":
            import chats
            self._send(200, {"chats": chats.listing()})
        elif self.path == "/doctor":
            self._send(200, {"checks": doctor()})
        elif self.path == "/settings":
            import settings
            self._send(200, settings.masked(settings.load()))
        else:
            raise Bad(404, "not found")

    def _post(self):
        d = self._body(CHAT_BODY if self.path == "/chat" else MAX_BODY)
        if self.path == "/mode":
            set_mode(d.get("mode"), d.get("model"), d.get("device", "npu"))
            self._send(200, {"mode": S.mode, "model": S.model})
        elif self.path == "/profile":
            if d.get("profile") not in PROFILES:
                raise Bad(400, f"profile must be one of {PROFILES}")
            S.profile = d["profile"]
            gpu_now = S.engine is not None and getattr(S.engine, "g", None) is not None
            self._send(200, {"profile": S.profile, "reload": S.mode == "llm" and gpu_now != (S.profile == "assisted")})   # the loaded model keeps its old setup until reloaded
        elif self.path == "/rules":
            self._send(200, set_rules(d))
        elif self.path == "/documents/delete":
            import documents
            try:
                documents.delete(str(d.get("id", "")))
            except documents.DocError as e:
                raise Bad(400, str(e))
            self._send(200, {"documents": documents.listing()})
        elif self.path == "/facts":
            import facts
            try:
                self._send(200, {"facts": facts.save(d.get("facts"))})
            except ValueError as e:
                raise Bad(400, str(e))
        elif self.path in ("/chats/save", "/chats/load", "/chats/delete"):
            import chats
            try:
                i = str(d.get("id", ""))
                if self.path == "/chats/save":
                    self._send(200, chats.save(i, d.get("title", ""), d.get("history") or []))
                elif self.path == "/chats/load":
                    self._send(200, chats.load(i))
                else:
                    chats.delete(i)
                    self._send(200, {"chats": chats.listing()})
            except chats.ChatError as e:
                raise Bad(400, str(e))
        elif self.path == "/routing":
            import router
            try:
                self._send(200, router.save(d))
            except ValueError as e:
                raise Bad(400, str(e))
        elif self.path == "/settings":
            import settings
            try:
                self._send(200, settings.masked(settings.update(d)))
            except ValueError as e:
                raise Bad(400, str(e))
        elif self.path == "/settings/models":       # list the models a configured provider offers
            import settings, cloud, web
            p = settings.load()["cloud"]["providers"].get(str(d.get("provider", "")))
            if not p:
                raise Bad(404, "unknown provider")
            try:
                self._send(200, {"models": cloud.list_models(p)})
            except web.WebError as e:
                raise Bad(502, str(e))
        elif self.path == "/settings/test_web":
            import settings, web
            try:
                r = web.search(str(d.get("query", "npu"))[:200], settings.load()["web"], 3, 0)
                self._send(200, {"results": [{"title": x["title"], "url": x["url"]} for x in r]})
            except web.WebError as e:
                raise Bad(502, str(e))
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

    def _line(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def _chat(self, d):
        """ndjson stream: {phase}, {sources}, {warning}, {think}, {token}, {done, stats}, {error}.
        backend "local": the NPU model (needs llm mode). backend "cloud": your configured provider (needs no NPU)."""
        import settings
        prompt, n = str(d.get("prompt", "")), int(d.get("tokens", 400))
        cloud = d.get("backend") == "cloud"
        history = [{"role": m.get("role"), "content": str(m.get("content", ""))[:400000]} for m in (d.get("history") or [])[-400:] if isinstance(m, dict)]
        max_prompt = max(8000, 4 * int(getattr(S.engine, "max_ctx", 4096)))          # about 4 characters per token
        if not prompt.strip() or len(prompt) > max_prompt or not 1 <= n <= MAX_TOKENS:
            raise Bad(400, f"prompt 1..{max_prompt} characters (4 per token of the context window), tokens 1..{MAX_TOKENS}")
        cfg = settings.load()
        provider = cfg["cloud"]["providers"].get(cfg["cloud"]["active"]) if cloud else None
        if cloud and not provider:
            raise Bad(409, "no cloud provider is configured (Settings)")
        if not cloud and S.mode != "llm":
            raise Bad(409, "not in llm mode")
        if not cloud and not S.npu_lock.acquire(blocking=False):
            raise Bad(409, "busy")
        import router
        rcfg = router.load()
        auto = router.decide(prompt, rcfg)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                if not cloud and S.engine is not None:
                    import search
                    rl = search.rules()
                    history, info = compact(history, S.engine.max_ctx, float(rl.get("compact_at", 0.7)), int(rl.get("keep_turns", 3)),
                                            lambda **kw: self._line({"phase": "compacting", **kw}))
                    if info:
                        self._line({"compacted": info})
                ctx = []
                if not auto["sure"] and rcfg["classifier"] and not cloud and "auto" in (d.get("docs", "auto"), d.get("web", "auto")):
                    self._line({"phase": "routing"})        # no rule was sure: ask the local model (one word, 3 tokens)
                    auto = router.from_label(S.engine.classify(prompt, router.CLASSIFY_SYSTEM, list(router.LABELS)))
                use_docs, use_web = router.resolve(d.get("docs", "auto"), auto["docs"]), router.resolve(d.get("web", "auto"), auto["web"])
                import documents
                use_mine = router.resolve(d.get("mine", "auto"), bool(documents.listing()) and (not auto["history"] or documents.refers(prompt, documents.listing())))   # "and the second page of this PDF?" is a follow-up that still needs the file
                self._line({"route": {"docs": use_docs, "web": use_web, "mine": use_mine, "reason": auto["reason"], "escalate": router.escalate(prompt)}})
                if use_docs:
                    self._line({"phase": "docs"})
                    import docs
                    if S.docs_info() is None:
                        if d.get("docs") != "auto":
                            self._line({"warning": "docs index not built: run `python src/docs.py --device gpu build`"})
                    else:
                        ctx += [{**c, "kind": "docs"} for c in docs.retrieve(S.cpu_embedder(), prompt, 4, S.docs)]
                if use_mine:
                    import documents
                    self._line({"phase": "mine"})
                    ctx = documents.retrieve(S.cpu_embedder(), prompt, 4) + ctx          # your own documents rank first
                if use_web:
                    self._line({"phase": "web"})
                    import web
                    try:
                        ctx += web.as_context(router.clean_web(web.search(router.search_query(prompt, history), cfg["web"], cfg["web"]["results"] + 2, cfg["web"]["read"]))[:cfg["web"]["results"]])
                    except web.WebError as e:
                        self._line({"warning": f"web search failed: {e}"})
                if ctx:
                    self._line({"sources": [{"src": c["src"], "title": c.get("title", c["src"]), "url": c.get("url", ""), "kind": c["kind"]} for c in ctx]})
                if cloud:
                    import cloud as cl
                    self._line({"phase": "asking", "provider": cfg["cloud"]["active"]})
                    send = prompt
                    if ctx and d.get("cloud_context"):    # excerpts only leave the PC if the user ticked "include excerpts"
                        send = "Reference excerpts:\n\n" + "\n\n".join(f"[{c['src']}]\n{c['text']}" for c in ctx) + f"\n\nQuestion: {prompt}"
                    t0, count = time.time(), 0
                    for kind, piece in cl.stream(provider, history + [{"role": "user", "content": send}], n):
                        count += 1
                        self._line({kind: piece})
                    self._line({"done": True, "stats": {"backend": "cloud", "seconds": round(time.time() - t0, 1)}})
                else:
                    for piece in S.engine.stream(prompt, n, ctx or None, history, lambda name, **kw: self._line({"phase": name, **kw}), system=__import__("facts").as_system(__import__("qwen_npu").SYSTEM)):
                        self._line({"token": piece})
                    self._line({"done": True, "stats": {"backend": "local", **S.engine.stats}})
            except (BrokenPipeError, ConnectionResetError):
                pass                      # client left: stop generating
            except Exception as e:
                try:
                    self._line({"error": f"{type(e).__name__}: {e}"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            if not cloud:
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
