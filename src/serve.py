"""Local engine server for the launcher (one mode loaded at a time: LLM chat or semantic search).

  NPU_LAB_TOKEN=... [NPU_LAB_PORT=8765] python src/serve.py        # prints {"port": N} on stdout when ready

Serves the web UI (web/index.html) at / and a JSON API under the same origin.
Security model: binds 127.0.0.1 on a random port; every API request needs `Authorization: Bearer <token>`
(token from the NPU_LAB_TOKEN env var, or generated and printed; the launcher hands it to the page in the
URL fragment, which never reaches the server); the Host header must be loopback; an Origin header is only
accepted if it is this server's own origin; no CORS headers; only whitelisted models; folders only under $HOME.

OpenAI-compatible (for editors, loaded model only): GET /v1/models   POST /v1/chat/completions (stream or not; no tools).
API (JSON):  POST /login (one-time code -> token)   GET /status /rules /routing /documents /chats /facts /doctor /settings   POST /routing /facts /agent/run /project/apply /chats/save /chats/load /chats/delete /documents/upload /documents/delete /settings /settings/models /settings/test_web /mode /profile /rules /chat (ndjson stream) /search /index /index/stop
"""
import itertools, gc, hmac, json, os, re, secrets, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Qwen3.5 (hybrid) needs the GPU: Assisted profile only. Qwen3 (dense) also runs NPU only.
MODELS = {"qwen35s": "models/qwen35_08b", "qwen35m": "models/qwen35_2b", "qwen35l": "models/qwen35_4b", "qwen3s": "models/qwen3s", "qwen3": "models/qwen3"}
NEEDS_GPU = ("qwen35s", "qwen35m", "qwen35l")   # whitelist: the API never takes a path
SHARES = {"qwen35s": (0.0, 0.25, 0.5), "qwen35m": (0.0, 0.25, 0.5, 0.75), "qwen35l": (0.0, 0.25)}   # GPU shares with a measured fit per model (bench/ENERGY.md); the NPU kernels are built per share
DEFAULT_SHARE = {"qwen35m": 0.5}                                                                       # everything else: 0 (NPU only)


# Peak GPU memory (MiB, nvidia-smi) per model and share, measured with llm_vram_gb 5.0 and a 12000-token window (bench/ENERGY.md, bench/PROPUESTA_GPU_SANA.md). Missing entry: only the check after loading applies.
PEAK_MIB = {("qwen35m", 0.0): 3401, ("qwen35m", 0.25): 3583, ("qwen35m", 0.5): 3825, ("qwen35m", 0.75): 4067, ("qwen35l", 0.0): 5057, ("qwen35l", 0.25): 5277, ("qwen35l", 0.5): 5755}
ACT_MIB = 64                     # extra GPU memory seen while reading a long prompt after loading: 22 MiB for a 10 000-token prompt on the 4B (the 1024-token chunk buffers, 355 MiB, are already inside the memory limit)
DEFAULT_VRAM_MARGIN_MIB = 700    # GPU memory kept free for the desktop; the user decides the final value (Settings)
BUILD_CMD = "python scripts/build_matvec.py --shared4 $(python src/qwen35_npu.py --model {dir} --shapes --share {share})"     # ~80 s on this machine


def missing_kernels(model, share, build_dir=None):
    """Names of the NPU kernels that still have to be built for `model` at GPU share `share` ([] = ready). Nothing is run."""
    import qwen35_npu
    cfg = json.load(open(os.path.join(ROOT, MODELS[model], "config.json")))
    bd = build_dir or os.path.join(ROOT, "build/c4")
    return [n for n in (f"c4_{m}x{k}" for m, k in qwen35_npu.shapes(qwen35_npu.text_config(cfg), share)) if not os.path.exists(os.path.join(bd, n + ".xclbin"))]


def free_vram_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None               # no NVIDIA tools: nothing to check


def own_vram_mib():
    """GPU memory this very process still holds (the CUDA context stays after a model is unloaded). A new model reuses it, so it does not count against the new model."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return sum(int(m.split(",")[1]) for m in out.strip().splitlines() if m.split(",")[0].strip() == str(os.getpid()))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


def vram_refusal(model, share, margin_mib, free_mib):
    """Message when loading `model` at `share` would leave less than `margin_mib` free for the desktop (from the measured peak), else None."""
    pk = PEAK_MIB.get((model, share))
    if free_mib is None or pk is None or free_mib - pk >= margin_mib:
        return None
    return (f"Not loaded: {model} at a {share:.0%} GPU share peaks at ~{pk} MiB of GPU memory (measured) and only {free_mib} MiB are free, which would leave {free_mib - pk} MiB for the desktop "
            f"(your margin is {margin_mib:.0f} MiB). Pick a lower GPU share, close programs that use the GPU, or lower the margin in Settings.")


def rules_view():
    """What GET /rules returns: the rules plus, per model, the effective share, the allowed shares and which shares still need NPU kernels (so the page can warn)."""
    import search
    rl = search.rules(); kern = {}
    for m, sh in SHARES.items():
        if os.path.exists(os.path.join(ROOT, MODELS[m], "config.json")):
            kern[m] = {str(x): len(missing_kernels(m, x)) for x in sh}
    return {**rl, "gpu_share_effective": {m: share_of(m, rl) for m in SHARES}, "gpu_share_allowed": SHARES, "gpu_share_kernels_missing": kern,
            "gpu_share_build_cmd": {m: BUILD_CMD.replace("{dir}", MODELS[m]) for m in SHARES}, "vram_margin_mib": rl.get("vram_margin_mib", DEFAULT_VRAM_MARGIN_MIB)}


def share_of(model, rl):
    """GPU share of decoding for `model`: the user's choice in rules["gpu_share"][model], else the measured default. A scalar from an older rules file is ignored (it used to apply to every model)."""
    v = rl.get("gpu_share")
    v = v.get(model) if isinstance(v, dict) else None
    return float(DEFAULT_SHARE.get(model, 0.0) if v is None else v)
PROFILES = ("npu_only", "assisted")                           # assisted: GPU/CPU may help bulk indexing
READ_TOOLS = ("read_file", "list_dir", "file_search", "grep_search", "view", "grep", "glob")                   # editor tools offered to the model over /v1 (chat agent and CLI-style agent names)
WRITE_TOOLS = ("create_file", "replace_string_in_file", "create_directory", "create", "edit")                  # the editor asks the user before each write
AGENT_TOOLS = READ_TOOLS + WRITE_TOOLS
def fmt_call(name, args):
    """Qwen's native tool format: raw values, nothing to escape (code with quotes and newlines breaks JSON in small models)."""
    return f"<tool_call>\n<function={name}>\n" + "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>"


ARG_ALIASES = {"filePath": ("path", "file_path", "filepath", "file", "filename"), "dirPath": ("path", "dir", "directory"), "path": ("filePath", "file_path", "dirPath", "dir"),
               "oldString": ("old_string", "old_str", "old"), "newString": ("new_string", "new_str", "new"), "content": ("file_text", "text", "code", "contents")}
ARG_DEFAULTS = {"startLine": 1, "endLine": 300, "explanation": "requested by the user"}


def fix_args(args, tools_spec):
    """Make the model's arguments match the tool schema: rename common aliases, drop unknown keys, fill defaults for plain required fields."""
    props = (tools_spec.get("parameters") or {}).get("properties") or {}
    if not props:
        return args
    out = {}
    for k, v in args.items():
        if k in props:
            out[k] = v
        else:
            for real, alts in ARG_ALIASES.items():
                if real in props and real not in args and k in alts:
                    out[real] = v
                    break
    for k in (tools_spec.get("parameters") or {}).get("required") or []:
        if k not in out and k in ARG_DEFAULTS:
            out[k] = ARG_DEFAULTS[k]
    return out


def parse_call(out, tools):
    """First tool call in the model's reply -> (name, args, start index of the call), or None. Accepts the XML form and, as a fallback, <tool_call>{json}</tool_call> (arguments nested or flat; trailing junk such as an extra brace is ignored)."""
    m = re.search(r"<function=([\w.-]+)>([\s\S]*?)(</function>|</tool_call>|$)", out)
    if m and m.group(1) in tools:
        props = (tools[m.group(1)].get("parameters") or {}).get("properties") or {}
        args = {}
        for k, v in re.findall(r"<parameter=([\w.-]+)>\n?([\s\S]*?)\n?</parameter>", m.group(2)):
            t = (props.get(k) or {}).get("type")
            try:
                args[k] = int(v) if t == "integer" else float(v) if t == "number" else json.loads(v) if t in ("boolean", "array", "object") else v
            except ValueError:
                args[k] = v
        st = out.rfind("<tool_call>", 0, m.start() + 1)
        return m.group(1), fix_args(args, tools[m.group(1)]), st if st >= 0 else m.start()
    m = re.search(r"<tool_call>\s*(?=\{)", out)
    if m:
        try:
            j, _ = json.JSONDecoder(strict=False).raw_decode(out, m.end())
        except ValueError:
            return None
        if isinstance(j, dict) and j.get("name") in tools:
            a = j.get("arguments")
            return j["name"], fix_args(a if isinstance(a, dict) else {k: v for k, v in j.items() if k != "name"}, tools[j["name"]]), m.start()


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
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()   # a failed or unloaded model must give its VRAM back to the driver

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
        g = d["gpu_share"]
        if not isinstance(g, dict) or any(k not in SHARES or v not in SHARES[k] for k, v in g.items()):
            raise Bad(400, "gpu_share must be an object {model: share}; allowed: " + ", ".join(f"{k} {list(v)}" for k, v in SHARES.items()))
        r["gpu_share"] = {**(r["gpu_share"] if isinstance(r.get("gpu_share"), dict) else {}), **{k: float(v) for k, v in g.items()}}
    if "vram_margin_mib" in d:
        v = d["vram_margin_mib"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 3000:
            raise Bad(400, "vram_margin_mib must be between 0 and 3000")
        r["vram_margin_mib"] = float(v)
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
    if "project_root" in d:
        x = str(d["project_root"]).strip()
        if x:
            real = os.path.realpath(os.path.expanduser(x))
            if not (real.startswith(home + os.sep)) or not os.path.isdir(real):
                raise Bad(400, f"the project must be an existing folder inside your home: {x}")
            x = real
        r["project_root"] = x
    if "agent_test_command" in d:                     # the only command the agent's RUN may start (after your confirmation); empty = no RUN
        x = " ".join(str(d["agent_test_command"]).split())
        if len(x) > 300 or "\n" in str(d["agent_test_command"]):
            raise Bad(400, "agent_test_command must be one line of at most 300 characters")
        r["agent_test_command"] = x
    if "agent_guard" in d:
        if d["agent_guard"] not in ("off", "objective", "model"):
            raise Bad(400, "agent_guard must be off, objective or model")
        r["agent_guard"] = d["agent_guard"]
    if "agent_guard_tau" in d:
        v = float(d["agent_guard_tau"])
        if not 0.0 <= v <= 1.0:
            raise Bad(400, "agent_guard_tau must be between 0 and 1")
        r["agent_guard_tau"] = v
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
            share = share_of(name, _s.rules())
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
                  "at most 150 words.")
NOTES_HEAD, NOTES_TOKENS = "[Notes of the earlier conversation]\n", 250      # short notes: writing them is what a compaction costs (decode speed)


def compact(history, window, at, keep, phase):
    """When the conversation passes `at` of the context window, summarise all but the last `keep` turns with the loaded model.
    Incremental: the notes of an earlier compaction are the starting point and only the turns that aged out since are summarised.
    Returns (history, None) untouched, or (notes + recent turns, {"summary", "kept"}). at = 0 disables it."""
    tok = S.engine.tok
    count = lambda t: len(tok.encode(t, add_special_tokens=False).ids)
    if not at or len(history) <= keep * 2 or sum(count(m["content"]) + 8 for m in history) < window * at:
        return history, None
    old, recent = history[:-keep * 2], history[-keep * 2:]
    notes, body = "", old
    if len(old) >= 2 and old[0]["content"].startswith(NOTES_HEAD):         # earlier notes: keep them (minus their pinned block, which is rebuilt) and read only what is new
        notes, body = old[0]["content"][len(NOTES_HEAD):].split("Summary:\n", 1)[-1].strip(), old[2:]
        if not body:
            return history, None
    ids = tok.encode("\n".join(f"{m['role']}: {m['content']}" for m in body), add_special_tokens=False).ids
    step = max(1024, int(window * 0.6))                   # one piece in the usual case: every extra piece is another round of writing notes
    pieces = [tok.decode(ids[i:i + step]) for i in range(0, len(ids), step)]
    for k, piece in enumerate(pieces):
        phase(step=k, of=len(pieces))          # not "done": the page reads {"done": true} as the end of the answer
        notes = S.engine.complete(f"Previous notes:\n{notes or '(none)'}\n\nNext part of the conversation:\n{piece}\n\nWrite the updated notes.",
                                  SUMMARY_SYSTEM, NOTES_TOKENS).strip()
    import pins
    notes = pins.block(pins.pin(old)) + notes                   # the user's own fact lines word for word: a summary may drop or change a digit
    pair = [{"role": "user", "content": NOTES_HEAD + notes}, {"role": "assistant", "content": "Understood, I will use those notes."}]
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
            margin = float(rl.get("vram_margin_mib", DEFAULT_VRAM_MARGIN_MIB))
            if model in NEEDS_GPU:
                share = share_of(model, rl); miss = missing_kernels(model, share)
                if miss:
                    raise Bad(409, f"{len(miss)} NPU kernels are missing for {model} at a {share:.0%} GPU share (~80 s to build): " + BUILD_CMD.replace("{dir}", MODELS[model]).replace("{share}", str(share)))
                free = free_vram_mib()
                msg = vram_refusal(model, share, margin, None if free is None else free + own_vram_mib())
                if msg:
                    raise Bad(409, msg)
            # assisted: the GPU reads prompts and holds the KV cache / DeltaNet state; the NPU still generates every token
            S.engine, S.mode, S.model = load_engine(MODELS[model], gpu=S.profile == "assisted", max_ctx=int(rl.get("context_tokens", DEFAULT_CONTEXT)), vram_gb=float(rl.get("llm_vram_gb", 4.0)), gpu_share=share_of(model, rl), cache_gb=float(rl.get("state_cache_gb", 8.0))), "llm", model
            left = free_vram_mib() if model in NEEDS_GPU else None
            if left is not None and left - ACT_MIB < margin:                # the failure path below unloads what was just loaded
                raise Bad(409, f"Unloaded: after loading {model} only {left} MiB of GPU memory are free and reading a long prompt needs ~{ACT_MIB} MiB more; your margin for the desktop is {margin:.0f} MiB. Lower the GPU share, the memory limit or the context window.")
        elif mode == "search":
            if device not in ("npu", "gpu", "cpu"):
                raise Bad(400, "device must be npu|gpu|cpu")
            if device != "npu" and S.profile == "npu_only":
                raise Bad(403, "profile npu_only: GPU/CPU helpers are disabled")
            import search
            S.engine, S.mode, S.model = search.make_model(device), "search", device
            S.load_index()
    except BaseException as e:
        import traceback; traceback.clear_frames(e.__traceback__)      # frames of the failed load hold its tensors
        S.unload()                                    # a half-loaded model (e.g. CUDA out of memory) must not keep the GPU
        raise
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
            if os.environ.get("NPU_LAB_DEBUG"):       # shape only, never the value
                with open(os.environ["NPU_LAB_DEBUG"], "a") as f:
                    f.write(json.dumps({"t": int(time.time()), "auth_fail": {"len": len(got), "starts": got[:7], "expected_len": len(TOKEN) + 7, "headers": sorted(k for k in self.headers.keys())}}) + "\n")
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
            self._dbg(e.code, e.msg)
            self._send(e.code, {"error": e.msg})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._dbg(500, f"{type(e).__name__}: {e}")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _dbg(self, code, msg):
        if os.environ.get("NPU_LAB_DEBUG") and self.path.startswith("/v1"):
            with open(os.environ["NPU_LAB_DEBUG"], "a") as f:
                f.write(json.dumps({"t": int(time.time()), "error": code, "msg": str(msg)[:300]}) + "\n")

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
        elif self.path == "/v1/models":          # OpenAI-compatible, for VS Code Copilot and other clients
            self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "npu-lab"} for m in MODELS if os.path.isdir(os.path.join(ROOT, MODELS[m]))]})
        elif self.path == "/routing":
            import router
            self._send(200, {**router.load(), "defaults": router.DEFAULTS})
        elif self.path == "/documents":
            import documents
            self._send(200, {"documents": documents.listing(), "job": S.docjob})
        elif self.path == "/rules":
            self._send(200, rules_view())
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
        elif self.path == "/project/graph":      # state of the graphify map of the project folder
            import graph, search
            try:
                self._send(200, graph.status(search.rules().get("project_root", "")))
            except graph.GraphError as e:
                raise Bad(400, str(e))
        else:
            raise Bad(404, "not found")

    def _post(self):
        d = self._body(CHAT_BODY if self.path in ("/chat", "/v1/chat/completions") else MAX_BODY)
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
        elif self.path in ("/project/graph", "/project/graph/query"):      # graphify map: code only, no model, no network
            import graph, search
            root = search.rules().get("project_root", "")
            try:
                if self.path == "/project/graph":
                    self._send(202, graph.start_update(root))
                else:
                    self._send(200, graph.query(root, d.get("q", ""), int(d.get("budget") or 1500)))
            except (graph.GraphError, ValueError) as e:
                raise Bad(400, str(e))
        elif self.path == "/agent/run":        # one action of the agent; anything that writes only runs with confirm=true (the page sends it after the user pressed OK)
            import agent, search
            root = search.rules().get("project_root", "")
            if not root or not os.path.isdir(root):
                raise Bad(409, "no project folder is set (Settings)")
            rl = search.rules(); guard = rl.get("agent_guard", "off"); backend = None
            if guard == "model":                  # the loaded model answers one closed question about the evidence; it needs the NPU/GPU for a moment and overwrites its KV cache
                import decide
                if S.engine is None or not hasattr(S.engine, "option_probs"):
                    guard = "objective"
                elif not S.npu_lock.acquire(blocking=False):
                    raise Bad(409, "The NPU is busy answering a request. Try again in a moment.")
                else:
                    backend = decide.LogprobBackend(S.engine, limit=12000)
            try:
                self._send(200, agent.run(root, str(d.get("text", ""))[:400000], confirmed=d.get("confirm") is True, last_ok=d.get("last_ok") is not False,
                                          test_command=rl.get("agent_test_command", ""), guard=guard, backend=backend, task=str(d.get("task", ""))[:2000],
                                          tau=float(rl.get("agent_guard_tau", 0.5))))
            finally:
                if backend is not None:
                    S.npu_lock.release()
        elif self.path == "/project/apply":      # writes files: only inside the chosen project, all or nothing, and only when the user pressed Apply
            import project, search
            root = search.rules().get("project_root", "")
            if not root or not os.path.isdir(root):
                raise Bad(409, "no project folder is set (Settings)")
            self._send(200, {"results": project.apply(root, str(d.get("text", ""))[:400000], dry=bool(d.get("dry")))})
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
        elif self.path == "/v1/chat/completions":
            self._openai(d)
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
                    pieces = []
                    system, root, ask = __import__("facts").as_system(__import__("qwen_npu").SYSTEM), rl.get("project_root", ""), prompt
                    if root and os.path.isdir(root):          # project mode: the map in the (cached) system prompt, the files the question names in the question
                        if d.get("agent") is True:            # terminal agent: one action per reply (the page runs it through /agent/run)
                            import agent
                            system += agent.system_block(root, rl.get("agent_test_command", ""))
                        else:
                            import project
                            system += project.system_block(root)
                            ask = project.attach(prompt, root)    # `prompt` stays the plain question: it is what the history keeps
                    for piece in S.engine.stream(ask, n, ctx or None, history, lambda name, **kw: self._line({"phase": name, **kw}), system=system):
                        pieces.append(piece)
                        self._line({"token": piece})
                    self._line({"done": True, "stats": {"backend": "local", **S.engine.stats}})
                    # tidy up AFTER the answer, while the user reads it: the notes arrive on this same stream and the page applies them before the next question
                    answer = re.sub(r"<think>[\s\S]*?(</think>|$)", "", "".join(pieces)).strip()
                    if answer:
                        try:
                            new, info = compact(history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], S.engine.max_ctx, float(rl.get("compact_at", 0.7)),
                                                int(rl.get("keep_turns", 3)), lambda **kw: self._line({"phase": "compacting", **kw}))
                            if info:
                                self._line({"compacted": info})
                        except (BrokenPipeError, ConnectionResetError):
                            raise
                        except Exception as e:
                            self._line({"warning": f"could not compact the conversation: {type(e).__name__}: {e}"})
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

    def _openai(self, d):
        """POST /v1/chat/completions (OpenAI format, stream or not) on the loaded local model: same engine, no routing/web/docs.
        With `tools`, a read-only "archaic agent": only AGENT_TOOLS are offered to the model, which asks for one with <tool_call>{json}</tool_call>; we turn that into
        an OpenAI tool_call and the editor runs it and sends the result back. The editor's own (huge) system prompt is dropped in that mode.
        ponytail: one call per reply, no parallel calls, text format instead of the model's native template; the `model` field is ignored (the loaded model answers)."""
        if S.mode != "llm":
            raise Bad(409, "not in llm mode: load a model first")
        text = lambda c: c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
        cut = lambda t, n: t if len(t) <= n else t[:n // 4] + "\n[...]\n" + t[-3 * n // 4:]
        tools = {f["name"]: f for t in d.get("tools") or [] if isinstance(t, dict) for f in [t.get("function") or {}] if f.get("name") in AGENT_TOOLS}
        raw = [m for m in d.get("messages") or [] if isinstance(m, dict)]
        if os.environ.get("NPU_LAB_DEBUG"):           # metadata only (no prompt text): what the editor really sends
            with open(os.environ["NPU_LAB_DEBUG"], "a") as f:
                f.write(json.dumps({"t": int(time.time()), "stream": d.get("stream"), "ntools": len(d.get("tools") or []), "names": [(t.get("function") or {}).get("name") for t in d.get("tools") or [] if isinstance(t, dict)], "offered": list(tools),
                                    "msgs": [(m.get("role"), len(text(m.get("content"))), len(m.get("tool_calls") or [])) for m in raw]}) + "\n")
        if tools:
            names = {tc.get("id"): tc["function"]["name"] for m in raw for tc in m.get("tool_calls") or [] if tc.get("function")}
            msgs = []
            for m in raw:
                r, c = m.get("role"), text(m.get("content"))
                if r == "assistant" and m.get("tool_calls"):
                    c += "\n".join(fmt_call(tc["function"]["name"], json.loads(tc["function"].get("arguments") or "{}")) for tc in m["tool_calls"][:1] if tc.get("function"))
                elif r == "tool":
                    r, c = "user", f'<tool_response name="{names.get(m.get("tool_call_id"), "?")}">\n{cut(c, 8000)}\n</tool_response>'
                elif r == "user":
                    c = cut(c, 14000)
                msgs.append((r, c))
            spec = "\n".join(f"- {n}({', '.join(k + ('*' if k in (f.get('parameters') or {}).get('required', []) else '') for k in (f.get('parameters') or {}).get('properties', {}))}): {(f.get('description') or '')[:160]}"
                             for n, f in tools.items())
            last_user = max((i for i, m in enumerate(raw) if m.get("role") == "user"), default=0)
            final = sum(m.get("role") == "tool" for m in raw[last_user:]) >= 3     # 3 reads for one question is enough: force an answer (a 2B model loops otherwise)
            ex = next(iter(tools)); exk = next(iter((tools[ex].get("parameters") or {}).get("properties") or {}), "path")
            system = ("You are a coding assistant inside an editor and you CAN read and edit the user's project: never say you cannot access files. Your tools (* = required argument):\n" + spec +
                      f'\n\nWhen you need a file or folder, reply with ONLY a call in this exact format and nothing else, e.g.\n{fmt_call(ex, {exk: "README.md"})}\n'
                      "Values are written raw (no quotes or escaping), also file contents. "
                      "The result comes back in a <tool_response> message. One tool per reply. Use a tool before answering about any file you have not seen. "
                      "You can create and edit files ONLY inside the open workspace (use the workspace folder path shown in the editor context if a tool wants absolute paths; never '..'; the editor asks the user to confirm each change). You cannot run commands or delete. To create a file use create_file with the full content. "
                      "Never repeat a tool call you already made. When you have what you need, answer in the user's language without a tool_call and do not invent contents.")
            if final:
                system += "\n\nNo more tool calls are allowed now: answer the user with what you already have."
        else:
            msgs = [(m.get("role"), text(m.get("content"))) for m in raw]
            system = "\n\n".join(c for r, c in msgs if r in ("system", "developer") and c) or None
        turns = []
        for r, c in msgs:
            if r in ("user", "assistant") and c:
                if turns and turns[-1]["role"] == r:
                    turns[-1]["content"] += "\n\n" + c
                else:
                    turns.append({"role": r, "content": c})
        while tools and sum(len(t["content"]) for t in turns) > 36000 and len(turns) > 1:   # keep the newest turns that fit (~16k tokens of context)
            turns.pop(0)
            while turns and turns[0]["role"] != "user":
                turns.pop(0)
        if not turns or turns[-1]["role"] != "user":
            raise Bad(400, "the last message must be from the user")
        n = max(1, min(int(d.get("max_tokens") or d.get("max_completion_tokens") or 1024), 1024))   # the editor's summaries ask for far more and would hold the NPU for minutes
        if not S.npu_lock.acquire(timeout=300):      # the editor fires several requests at once (chat + title...): queue them instead of failing
            raise Bad(409, "busy")
        try:
            gen = S.engine.stream(turns[-1]["content"], n, None, turns[:-1], None, system=system)
            cid, name, t0 = "chatcmpl-" + secrets.token_hex(8), S.model, int(time.time())
            chunk = lambda delta, fin=None: {"id": cid, "object": "chat.completion.chunk", "created": t0, "model": name, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            if tools:                                   # collect the whole reply, then answer with text or one tool_call
                def run(g):
                    o = ""
                    for piece in g:
                        o += piece
                        if "</tool_call>" in o or "</function>" in o:
                            break
                    return re.sub(r"<think>[\s\S]*?(</think>|$)", "", o).strip()
                try:
                    out = run(gen)
                    pc = parse_call(out, tools)
                    if not final and not pc and ("<tool_call>" in out or "<function=" in out):      # malformed call: one retry showing the exact format
                        ex = next(iter(tools)); exk = next(iter((tools[ex].get("parameters") or {}).get("properties") or {}), "path")
                        hint = f"That tool call could not be read. Reply again with ONLY a call in this exact format (raw values, nothing else):\n{fmt_call(ex, {exk: '...'})}"
                        out = run(S.engine.stream(hint, n, None, turns + [{"role": "assistant", "content": out}], None, system=system))
                        pc = parse_call(out, tools)
                except ValueError as e:
                    raise Bad(400, f"{e} (the prompt does not fit the loaded model's context)")
                call, content = None, out
                if final:
                    content = re.sub(r"<tool_call>[\s\S]*", "", out).strip() or "(sin respuesta: el modelo no pudo contestar con lo que leyó)"
                elif pc:
                    nm, a, st = pc
                    if not (nm in WRITE_TOOLS and any(".." in str(a.get(k) or "").replace("\\", "/").split("/") for k in ("filePath", "dirPath", "path", "file_path"))):
                        call, content = {"id": "call_" + secrets.token_hex(8), "type": "function", "function": {"name": nm, "arguments": json.dumps(a)}}, out[:st].strip()
                if os.environ.get("NPU_LAB_DEBUG"):
                    with open(os.environ["NPU_LAB_DEBUG"], "a") as f:
                        f.write(json.dumps({"t": int(time.time()), "reply_chars": len(out), "tool_call": call and call["function"]["name"]}) + "\n")
                if d.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    ev = [chunk({"role": "assistant"})] + ([chunk({"content": content})] if content else []) + \
                         ([chunk({"tool_calls": [{"index": 0, **call}]})] if call else []) + [chunk({}, "tool_calls" if call else "stop")]
                    try:
                        for o in ev:
                            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(200, {"id": cid, "object": "chat.completion", "created": t0, "model": name, "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop",
                                     "message": {"role": "assistant", "content": content or None, **({"tool_calls": [call]} if call else {})}}]})
                return
            try:
                first = next(gen, "")                  # prompt too long etc. fail here, before the stream starts: a clean 400 instead of a broken stream
            except ValueError as e:
                raise Bad(400, f"{e} (the editor's prompt does not fit the loaded model's context; lower maxInputTokens or raise the context in Settings)")
            gen = itertools.chain([first], gen)
            if d.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                send = lambda o: (self.wfile.write(b"data: " + (json.dumps(o) if o != "[DONE]" else o).encode() + b"\n\n"), self.wfile.flush())
                try:
                    send(chunk({"role": "assistant"}))
                    for piece in gen:
                        send(chunk({"content": piece}))
                    send(chunk({}, "stop"))
                    send("[DONE]")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                out = "".join(gen)
                self._send(200, {"id": cid, "object": "chat.completion", "created": t0, "model": name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}]})
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
    root = os.environ.get("NPU_LAB_STATE") or os.path.expanduser("~/.cache/npu-lab/state")
    if os.path.isdir(root):                                    # states saved by an older version had the default permissions
        import statecache
        statecache.lock_down(root)
    if os.environ.get("NPU_LAB_WATCH_STDIN") == "1":
        watch_parent()
    srv = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("NPU_LAB_PORT", 0))), H)   # fixed port lets VS Code point at it
    srv.daemon_threads = True
    print(json.dumps({"port": srv.server_address[1], **({} if os.environ.get("NPU_LAB_TOKEN") else {"token": TOKEN})}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
