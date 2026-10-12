"""End-to-end check of src/serve.py. Security checks always run; the NPU cycle runs when an NPU + models exist.
usage: .venv/bin/python tests/test_server.py"""
import http.client, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "t" * 43
p = subprocess.Popen([sys.executable, "-u", "src/serve.py"], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                     env={**os.environ, "NPU_LAB_TOKEN": TOKEN, "NPU_LAB_CODE": "c" * 43})
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
    check("chat without a loaded model -> 409", call("POST", "/chat", {"prompt": "hi"})[0] == 409)
    check("oversized body -> 413", call("POST", "/profile", None, headers={"Content-Length": "9999999"})[0] in (413, 400))

    if os.path.exists("/dev/accel/accel0") and os.path.exists(os.path.join(ROOT, "models/qwen/model.safetensors")):
        t = time.time()
        check("load LLM", call("POST", "/mode", {"mode": "llm", "model": "qwen"})[0] == 200)
        s, body = call("POST", "/chat", {"prompt": "What is the capital of France? One sentence.", "tokens": 30})
        lines = [json.loads(x) for x in body.decode().splitlines() if x]
        text = "".join(x.get("token", "") for x in lines)
        check(f"chat streams an answer ({text.strip()[:40]!r})", s == 200 and "Paris" in text and lines[-1].get("done"))
        check("switch to search frees the NPU and loads the embedder", call("POST", "/mode", {"mode": "search", "device": "npu"})[0] == 200)
        s, body = call("POST", "/search", {"query": "servidor PHP base de datos", "n": 3})
        res = json.loads(body).get("results", [])
        check(f"search returns results ({len(res)})", s == 200 and len(res) > 0)
        check("unload", call("POST", "/mode", {"mode": "none"})[0] == 200)
        print(f"NPU cycle took {time.time()-t:.0f}s")
    else:
        print("SKIP NPU cycle (no NPU or model)")
finally:
    p.terminate()
    p.wait(10)
sys.exit(1 if check.failed else 0)
