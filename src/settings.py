"""User settings: web search provider and cloud-assist providers (with API keys).

Stored in ~/.config/npu-lab/providers.json, created with mode 0600 (only your user can read it).
Keys are write-only from the UI: they are never returned by the API, only `has_key` flags.
"""
import json, os, tempfile

PATH = os.environ.get("NPU_LAB_SETTINGS") or os.path.expanduser("~/.config/npu-lab/providers.json")   # env override is for tests
DEFAULTS = {
    "web": {"provider": "duckduckgo", "searxng_url": "", "brave_key": "", "wiki_lang": "es", "results": 4, "read": 2},
    "cloud": {"active": "", "providers": {}},
}
KINDS = ("openai", "gemini")          # "openai" = any OpenAI-compatible endpoint (NVIDIA NIM, OpenRouter, Ollama, ...)


def load():
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        user = json.load(open(PATH))
    except (OSError, json.JSONDecodeError):
        return cfg
    cfg["web"].update(user.get("web", {}))
    cfg["cloud"]["active"] = user.get("cloud", {}).get("active", "")
    cfg["cloud"]["providers"] = user.get("cloud", {}).get("providers", {})
    return cfg


def save(cfg):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(PATH))
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(tmp, 0o600)                       # before it replaces the real file
    os.replace(tmp, PATH)


def masked(cfg):
    """What the browser may see: everything except secrets."""
    w = {k: v for k, v in cfg["web"].items() if k != "brave_key"}
    w["has_brave_key"] = bool(cfg["web"].get("brave_key"))
    provs = {n: {"kind": p.get("kind"), "base_url": p.get("base_url", ""), "model": p.get("model", ""), "has_key": bool(p.get("key"))}
             for n, p in cfg["cloud"]["providers"].items()}
    return {"web": w, "cloud": {"active": cfg["cloud"]["active"], "providers": provs}}


def update(patch):
    """Apply a UI patch. A missing/empty secret field keeps the stored secret; {"remove": true} deletes a provider."""
    cfg = load()
    w = patch.get("web")
    if isinstance(w, dict):
        if w.get("provider") in ("duckduckgo", "wikipedia", "searxng", "brave"):
            cfg["web"]["provider"] = w["provider"]
        for k in ("searxng_url", "wiki_lang"):
            if isinstance(w.get(k), str):
                cfg["web"][k] = w[k].strip()[:200]
        if w.get("brave_key"):
            cfg["web"]["brave_key"] = str(w["brave_key"]).strip()[:300]
        for k, lo, hi in (("results", 1, 8), ("read", 0, 4)):
            if isinstance(w.get(k), int):
                cfg["web"][k] = max(lo, min(hi, w[k]))
    c = patch.get("cloud")
    if isinstance(c, dict):
        for name, p in (c.get("providers") or {}).items():
            name = str(name).strip()[:40]
            if not name or not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("provider name: letters, digits, - and _ only")
            if p.get("remove"):
                cfg["cloud"]["providers"].pop(name, None)
                continue
            if p.get("kind") not in KINDS:
                raise ValueError(f"kind must be one of {KINDS}")
            cur = cfg["cloud"]["providers"].get(name, {})
            new = {"kind": p["kind"], "base_url": str(p.get("base_url", cur.get("base_url", ""))).strip()[:300],
                   "model": str(p.get("model", cur.get("model", ""))).strip()[:120], "key": str(p.get("key") or cur.get("key", "")).strip()[:400]}
            cfg["cloud"]["providers"][name] = new
        if "active" in c:
            if c["active"] and c["active"] not in cfg["cloud"]["providers"]:
                raise ValueError("active provider does not exist")
            cfg["cloud"]["active"] = c["active"]
    save(cfg)
    return cfg
