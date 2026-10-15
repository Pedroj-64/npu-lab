"""Offline test for bridge/telegram_bridge.py: allow-list, commands and error messages, with a fake npu-lab server and a fake Telegram, all on localhost.
usage: python tests/test_bridge.py"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge"))
import telegram_bridge as B

fails = 0
def check(n, ok):
    global fails; fails += not ok; print(("PASS " if ok else "FAIL ") + n)

sent, calls = [], []
EVENTS = [{"name": "inklus", "repo": "~/c/inklus", "dirty": 13, "ahead": 0}, {"name": "mango", "repo": "~/c/mango", "dirty": 0, "ahead": 1}]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        calls.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/status": return self._send(200, {"model": "qwen3s", "profile": "npu_only", "busy": False})
        if self.path == "/sentinel": return self._send(200, {"events": EVENTS})
        if self.path == "/bench": return self._send(200, {"history": [{"ts": "2026-10-15T08:40:09", "decode": 20.4, "prefill_256": 241.0}]})
        self._send(404, {"error": "nf"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}"); calls.append(("POST", self.path, body))
        if self.path.startswith("/bot"):                                    # fake Telegram
            if self.path.endswith("sendMessage"): sent.append(body)
            return self._send(200, {"ok": True, "result": []})
        if self.path == "/v1/chat/completions": return self._send(200, {"choices": [{"message": {"content": "<think>x</think>Rome."}}]})
        if self.path == "/helpers/commit": return self._send(200, {"repo": "inklus", "files": ["a", "b"], "secrets": [{"kind": "Hugging Face token", "line": 2}], "secrets_untracked": [], "message": ""})
        if self.path == "/mode" and body.get("model") == "boom": return self._send(409, {"error": "The NPU is busy"})
        if self.path == "/mode": return self._send(200, {"mode": body.get("mode")})
        self._send(404, {"error": "nf"})
srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
local = B.Local(port=port, token="t" * 40)
hist = []
check("status", "qwen3s" in B.handle("/status", local, hist) and "2" in B.handle("/status", local, hist))
check("repos", "inklus: 13 uncommitted" in B.handle("/repos", local, hist) and "mango: 1 unpushed" in B.handle("/repos", local, hist))
r = B.handle("/commit inklus", local, hist)
check("commit reports secrets without their value", "POSSIBLE SECRETS" in r and "Hugging Face token at line 2" in r and "hf_" not in r)
check("commit of an unknown repo", "nothing pending" in B.handle("/commit nope", local, hist))
check("plain text goes to the model, thinking removed, history kept", B.handle("capital of Italy?", local, hist) == "Rome." and hist[-1]["content"] == "Rome." and len(hist) == 2)
for i in range(10): B.handle(f"q{i}", local, hist)
check("history keeps the last 6 messages", len(hist) <= 6)
check("unknown command", "Unknown" in B.handle("/rm -rf", local, hist))
check("server error is explained", "busy" in B.handle("/load boom", local, hist) or "loaded" in B.handle("/load qwen3", local, hist))
import socket
fam = [a[0] for a in socket.getaddrinfo("localhost", 80, 0, socket.SOCK_STREAM)]
check("IPv4 addresses are tried first", fam == sorted(fam, key=lambda f: f != socket.AF_INET))
said = []
B.handle("/load qwen3s", local, [], say=said.append)
check("a load announces itself before the wait", len(said) == 1 and "Loading qwen3s" in said[0])
dead = B.Local(port=1, token="t" * 40)
check("server down message", "not running" in B.handle("/status", dead, []))
check("token is sent as bearer", any(c[0] == "GET" and c[2] == "Bearer " + "t" * 40 for c in calls))
bridge = B.Bridge("SECRETTOKEN", [111], local, base=f"http://127.0.0.1:{port}")
check("stranger gets no answer", bridge.process({"update_id": 1, "message": {"from": {"id": 999}, "chat": {"id": 999}, "text": "/status"}}) is None and not sent)
check("allowed user gets an answer", bridge.process({"update_id": 2, "message": {"from": {"id": 111}, "chat": {"id": 111}, "text": "/repos"}}) and sent and sent[-1]["chat_id"] == 111)
check("offset advances", bridge.offset == 3)
srv.shutdown(); sys.exit(1 if fails else 0)
