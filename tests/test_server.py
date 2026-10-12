"""End-to-end check of src/serve.py. Security checks always run; the NPU cycle runs when an NPU + models exist.
usage: .venv/bin/python tests/test_server.py"""
import http.client, json, os, subprocess, sys, time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "sk-test-SECRET-1234567890"


class Fake(BaseHTTPRequestHandler):          # an OpenAI-compatible provider on localhost that streams a reasoning part + 2 tokens
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        ok = self.headers.get("Authorization") == f"Bearer {KEY}"
        self.send_response(200 if ok else 401); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if ok:
            for e in ({"choices": [{"delta": {"reasoning_content": "hmm"}}]}, {"choices": [{"delta": {"content": "```py\nprint(1)\n```"}}]}):
                self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")


fake = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=fake.serve_forever, daemon=True).start()
TOKEN = "t" * 43
p = subprocess.Popen([sys.executable, "-u", "src/serve.py"], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                     env={**os.environ, "NPU_LAB_TOKEN": TOKEN, "NPU_LAB_CODE": "c" * 43,
                                    "NPU_LAB_SETTINGS": os.path.join(__import__("tempfile").mkdtemp(), "providers.json")})
port = json.loads(p.stdout.readline())["port"]


def call(method, path, body=None, headers=None, token=TOKEN, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    if token:
        h["Authorization"] = "Bearer " + token
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    data = r.read()
    return r.status, data


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed = True
check.failed = False

try:
    check("login with a wrong code -> 401", call("POST", "/login", {"code": "nope"}, token=None)[0] == 401)
    s1, b1 = call("POST", "/login", {"code": "c" * 43}, token=None)
    check("login code gives the token once", s1 == 200 and json.loads(b1)["token"] == TOKEN)
    check("login code is single use", call("POST", "/login", {"code": "c" * 43}, token=None)[0] == 401)
    check("login refused for a foreign Origin", call("POST", "/login", {"code": "c" * 43}, token=None, headers={"Origin": "https://evil.example"})[0] == 403)
    check("page served without token", call("GET", "/", token=None)[0] == 200 and b"npu-lab" in call("GET", "/", token=None)[1])
    check("page refused for a foreign Host (DNS rebinding)", call("GET", "/", token=None, host="evil.example")[0] == 403)
    check("API without token -> 401", call("GET", "/status", token=None)[0] == 401)
    check("API with wrong token -> 401", call("GET", "/status", token="x" * 43)[0] == 401)
    check("API with foreign Host -> 403", call("GET", "/status", host="evil.example")[0] == 403)
    check("foreign Origin -> 403", call("GET", "/status", headers={"Origin": "https://evil.example"})[0] == 403)
    check("own Origin accepted", call("POST", "/profile", {"profile": "npu_only"}, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200)
    check("status ok", call("GET", "/status")[0] == 200)
    check("doctor ok", call("GET", "/doctor")[0] == 200)
    check("unknown model refused (no paths from the API)", call("POST", "/mode", {"mode": "llm", "model": "../../etc"})[0] == 400)
    check("GPU/CPU refused under npu_only", call("POST", "/mode", {"mode": "search", "device": "gpu"})[0] == 403)
    check("roots outside $HOME refused", call("POST", "/rules", {"roots": ["/etc"]})[0] == 400)
    check("whole $HOME refused as root", call("POST", "/rules", {"roots": ["~"]})[0] == 400)
    # settings: secrets are write-only
    s, b = call("POST", "/settings", {"cloud": {"providers": {"fake": {"kind": "openai", "base_url": f"http://127.0.0.1:{fake.server_address[1]}/v1", "model": "m", "key": KEY}}, "active": "fake"},
                                      "web": {"provider": "brave", "brave_key": "BRAVE-SECRET-1"}})
    check("settings saved, response has no secrets", s == 200 and KEY.encode() not in b and b"BRAVE-SECRET" not in b and json.loads(b)["cloud"]["providers"]["fake"]["has_key"])
    check("GET /settings has no secrets", KEY.encode() not in call("GET", "/settings")[1] and b"BRAVE-SECRET" not in call("GET", "/settings")[1])
    check("status never contains a key", KEY.encode() not in call("GET", "/status")[1])
    check("bad provider kind refused", call("POST", "/settings", {"cloud": {"providers": {"x": {"kind": "evil"}}}})[0] == 400)
    s, body = call("POST", "/chat", {"prompt": "hola", "backend": "cloud", "history": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
    lines = [json.loads(x) for x in body.decode().splitlines() if x]
    check("cloud chat streams think + token + done without any local model", s == 200 and any(l.get("think") == "hmm" for l in lines)
          and "print(1)" in "".join(l.get("token", "") for l in lines) and lines[-1].get("done") and KEY not in body.decode())
    check("chat without a loaded model -> 409", call("POST", "/chat", {"prompt": "hi"})[0] == 409)
    check("oversized body -> 413", call("POST", "/profile", None, headers={"Content-Length": "9999999"})[0] in (413, 400))

    if os.path.exists("/dev/accel/accel0") and os.path.exists(os.path.join(ROOT, "models/qwen/model.safetensors")):
        t = time.time()
        check("load LLM", call("POST", "/mode", {"mode": "llm", "model": "qwen"})[0] == 200)
        s, body = call("POST", "/chat", {"prompt": "What is the capital of France? One sentence.", "tokens": 30})
        lines = [json.loads(x) for x in body.decode().splitlines() if x]
        text = "".join(x.get("token", "") for x in lines)
        check(f"chat streams an answer ({text.strip()[:40]!r})", s == 200 and "Paris" in text and lines[-1].get("done"))
        if os.path.exists(os.path.expanduser("~/.local/share/npu-lab/docs.pkl")):
            s, body = call("POST", "/chat", {"prompt": "¿Cómo instalo un paquete del AUR usando paru?", "tokens": 25, "docs": True})
            lines = [json.loads(x) for x in body.decode().splitlines() if x]
            srcs = [x["src"] for x in next((l["sources"] for l in lines if "sources" in l), [])]
            check(f"chat with local docs reports its sources {srcs[:2]}", s == 200 and any("paru" in x for x in srcs) and lines[-1].get("done"))
        s, body = call("POST", "/chat", {"prompt": "What is my name?", "tokens": 20, "docs": False,
                                         "history": [{"role": "user", "content": "My name is Zed."}, {"role": "assistant", "content": "Nice to meet you, Zed."}]})
        text = "".join(json.loads(x).get("token", "") for x in body.decode().splitlines() if x)
        check(f"local chat remembers earlier turns ({text.strip()[:40]!r})", s == 200 and "Zed" in text)
        check("switch to search frees the NPU and loads the embedder", call("POST", "/mode", {"mode": "search", "device": "npu"})[0] == 200)
        s, body = call("POST", "/search", {"query": "servidor PHP base de datos", "n": 3})
        res = json.loads(body).get("results", [])
        check(f"search returns results ({len(res)})", s == 200 and len(res) > 0)
        check("unload", call("POST", "/mode", {"mode": "none"})[0] == 200)
        print(f"NPU cycle took {time.time()-t:.0f}s")
    else:
        print("SKIP NPU cycle (no NPU or model)")
finally:
    fake.shutdown()
    p.terminate()
    p.wait(10)
sys.exit(1 if check.failed else 0)
