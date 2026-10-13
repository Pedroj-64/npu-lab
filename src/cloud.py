"""Optional cloud assist: stream an answer from a provider YOU configured (your own API key).

Kinds: "openai" (any OpenAI-compatible /chat/completions: NVIDIA NIM, OpenRouter, Ollama, LM Studio...) and "gemini".
This is opt-in per message. What is sent: the conversation and, only if you tick it, the docs/web excerpts.
Never your indexed files. Keys never appear in errors or logs.
"""
import json, re, urllib.error, urllib.parse, urllib.request
import web

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
SYSTEM = ("You are a helpful assistant. Answer in Spanish unless the user writes in another language. "
          "Use Markdown with fenced code blocks for code. Be accurate and say when you are unsure.")


def _local(url):
    return (urllib.parse.urlsplit(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def _redact(msg, key):
    return msg.replace(key, "***") if key else msg


def _open(url, body, headers, key, method="POST"):
    allow_local = _local(url)
    if urllib.parse.urlsplit(url).scheme != "https" and not allow_local:
        raise web.WebError("cloud endpoints must use https (plain http only for localhost)")
    web.check_url(url, allow_local)
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": web.UA, **headers})
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        detail = _redact(e.read(400).decode("utf-8", "ignore"), key)
        raise web.WebError(f"provider answered HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise web.WebError(_redact(f"request failed: {e}", key))


def _sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


def _need(p):
    if not p.get("model"):
        raise web.WebError("this provider has no model set (use 'List models' in Settings)")
    if not p.get("key") and not _local(p.get("base_url", "")):
        raise web.WebError("this provider has no API key")


def stream(p, messages, max_tokens=1024):
    """Yield ("think"|"token", text). `messages`: [{role: user|assistant, content}] (system prompt is added here)."""
    _need(p)
    key = p.get("key", "")
    if p["kind"] == "openai":
        base = (p.get("base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
        body = {"model": p["model"], "stream": True, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}] + messages}
        resp = _open(base + "/chat/completions", body, {"Authorization": f"Bearer {key}"} if key else {}, key)
        for ev in _sse(resp):
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if d.get("reasoning_content") or d.get("reasoning"):
                yield "think", d.get("reasoning_content") or d.get("reasoning")
            if d.get("content"):
                yield "token", d["content"]
    elif p["kind"] == "gemini":
        base = (p.get("base_url") or GEMINI).rstrip("/")
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        resp = _open(f"{base}/models/{urllib.parse.quote(p['model'], safe='')}:streamGenerateContent?alt=sse", body, {"x-goog-api-key": key}, key)
        for ev in _sse(resp):
            for part in ((ev.get("candidates") or [{}])[0].get("content") or {}).get("parts", []):
                if part.get("text"):
                    yield ("think" if part.get("thought") else "token"), part["text"]
    else:
        raise web.WebError("unknown provider kind")


def list_models(p):
    """Model ids the provider offers, so you pick a real one instead of guessing a name."""
    key = p.get("key", "")
    if p["kind"] == "openai":
        base = (p.get("base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
        resp = _open(base + "/models", None, {"Authorization": f"Bearer {key}"} if key else {}, key, method="GET")
        return sorted(m["id"] for m in json.load(resp).get("data", []))
    base = (p.get("base_url") or GEMINI).rstrip("/")
    resp = _open(base + "/models?pageSize=200", None, {"x-goog-api-key": key}, key, method="GET")
    return sorted(m["name"].split("/", 1)[-1] for m in json.load(resp).get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])
                  and not re.search(r"tts|image|embedding|computer-use|deep-research|antigravity|live|audio|robotics", m["name"]))   # chat models only
