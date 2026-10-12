"""Offline checks for settings/cloud/web helpers (a fake provider on localhost, no real keys, no internet).
usage: .venv/bin/python tests/test_cloud.py"""
import json, os, stat, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import settings, cloud, web

settings.PATH = os.path.join(tempfile.mkdtemp(), "providers.json")     # never touch the real config
SECRET = "sk-test-SECRET-1234567890"
seen = {}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        seen["auth"] = self.headers.get("Authorization") or self.headers.get("x-goog-api-key")
        body = json.dumps({"data": [{"id": "m-b"}, {"id": "m-a"}], "models": [{"name": "models/gem-1", "supportedGenerationMethods": ["generateContent"]}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); seen["body"] = json.loads(self.rfile.read(n))
        seen["auth"] = self.headers.get("Authorization") or self.headers.get("x-goog-api-key")
        if "bad" in self.path or seen["body"].get("model") == "bad":
            self.send_response(401); self.end_headers(); self.wfile.write(b'{"error":"bad key ' + SECRET.encode() + b'"}'); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if "chat/completions" in self.path:
            evs = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}, {"choices": [{"delta": {"content": "Hola "}}]}, {"choices": [{"delta": {"content": "mundo"}}]}]
        else:
            evs = [{"candidates": [{"content": {"parts": [{"text": "plan", "thought": True}]}}]}, {"candidates": [{"content": {"parts": [{"text": "Hola gemini"}]}}]}]
        for e in evs: self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


srv = HTTPServer(("127.0.0.1", 0), Fake); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

# settings: secrets are write-only and the file is private
settings.update({"cloud": {"providers": {"fake": {"kind": "openai", "base_url": f"http://127.0.0.1:{port}/v1", "model": "m", "key": SECRET}}, "active": "fake"},
                 "web": {"provider": "brave", "brave_key": "BRAVE-SECRET-999"}})
m = settings.masked(settings.load())
check("config file is mode 0600", stat.S_IMODE(os.stat(settings.PATH).st_mode) == 0o600)
check("masked config has no secrets", SECRET not in json.dumps(m) and "BRAVE-SECRET" not in json.dumps(m) and m["cloud"]["providers"]["fake"]["has_key"] and m["web"]["has_brave_key"])
settings.update({"cloud": {"providers": {"fake": {"kind": "openai", "model": "m2"}}}})
check("editing without a key keeps the stored key", settings.load()["cloud"]["providers"]["fake"]["key"] == SECRET and settings.load()["cloud"]["providers"]["fake"]["model"] == "m2")
for bad in ({"cloud": {"providers": {"../x": {"kind": "openai"}}}}, {"cloud": {"providers": {"ok": {"kind": "nope"}}}}, {"cloud": {"active": "ghost"}}):
    try: settings.update(bad); check("invalid settings rejected", False)
    except ValueError: check("invalid settings rejected", True)

# openai-compatible streaming + reasoning + auth header
p = settings.load()["cloud"]["providers"]["fake"]
out = list(cloud.stream(p, [{"role": "user", "content": "hi"}]))
check("openai stream: think + tokens", out == [("think", "hmm"), ("token", "Hola "), ("token", "mundo")])
check("openai: bearer key sent, system prompt added", seen["auth"] == f"Bearer {SECRET}" and seen["body"]["messages"][0]["role"] == "system" and seen["body"]["stream"] is True)
check("list models (openai)", cloud.list_models(p) == ["m-a", "m-b"])
# gemini
g = {"kind": "gemini", "base_url": f"http://127.0.0.1:{port}/v1beta", "model": "gem-1", "key": SECRET}
out = list(cloud.stream(g, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}]))
check("gemini stream: thought part separated", out == [("think", "plan"), ("token", "Hola gemini")])
check("gemini: key in header (not URL), roles mapped", seen["auth"] == SECRET and [c["role"] for c in seen["body"]["contents"]] == ["user", "model", "user"])
check("list models (gemini)", cloud.list_models(g) == ["gem-1"])
# errors never leak the key
try: list(cloud.stream({**p, "model": "bad"}, [{"role": "user", "content": "x"}])); check("error path", False)
except web.WebError as e: check("provider error does not leak the key", SECRET not in str(e) and "401" in str(e))
try: list(cloud.stream({"kind": "openai", "base_url": "http://example.com/v1", "model": "m", "key": "k"}, [])); check("plain http to a remote host refused", False)
except web.WebError as e: check("plain http to a remote host refused", "https" in str(e))
try: list(cloud.stream({"kind": "openai", "base_url": "https://192.168.1.5/v1", "model": "m", "key": "k"}, [])); check("private https host refused", False)
except web.WebError as e: check("private https host refused", "non-public" in str(e))
# web helpers (offline): result parsing + context shaping
page = ('<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa">Title <b>A</b></a><a class="result__snippet">Snip A &amp; more</a>'
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fb">Title B</a><a class="result__snippet">Snip B</a>')
web.http = lambda *a, **k: (200, "text/html", page.encode())
r = web.ddg("q", 5, {})
check("duckduckgo parser", [x["url"] for x in r] == ["https://example.org/a", "https://example.net/b"] and r[0]["title"] == "Title A" and r[0]["snippet"] == "Snip A & more")
ctx = web.as_context([dict(r[0], text="")], 50)
check("web context shaping", ctx[0]["src"] == "web:example.org" and ctx[0]["kind"] == "web")
srv.shutdown()
sys.exit(1 if fails else 0)
