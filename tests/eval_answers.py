"""Answer quality on checkable facts, without vs with local docs. Needs a running server with an LLM already loaded (it uses the NPU: run it alone).
usage: NPU_LAB_TOKEN=... python tests/eval_answers.py [port] [--tokens 200]      (exit 1 when any case fails)"""
import http.client, json, os, sys

PORT = int(next((a for a in sys.argv[1:] if a.isdigit()), 8765))
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 200
TOKEN = os.environ.get("NPU_LAB_TOKEN", "")

# (question, must contain any of, must contain none of); matching is case-insensitive
CASES = [
    ("How do I install an AUR package with paru?", ["paru -s"], ["app repository"]),
    ("What is the capital of Italy? One sentence.", ["roma", "rome"], []),
    ("¿Qué hace el comando chmod 755 archivo?", ["755", "rwx"], []),
    ("Translate to English: 'El gato duerme en el sofá'", ["cat"], ["gato duerme"]),
]


def ask(q, docs):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    c.request("POST", "/chat", json.dumps({"prompt": q, "tokens": TOKENS, "docs": docs, "web": "off", "mine": "off"}), {"Authorization": "Bearer " + TOKEN})
    r = c.getresponse()
    if r.status != 200:
        sys.exit(f"/chat -> {r.status}: {r.read()[:200]!r}")
    out = ""
    for line in r:
        m = json.loads(line)
        out += m.get("token", "")
        if "error" in m:
            sys.exit("engine error: " + str(m["error"]))
    return out.lower()


def ok(ans, want, forbid):
    return (not want or any(w in ans for w in want)) and not any(f in ans for f in forbid)


fails = 0
for q, want, forbid in CASES:
    res = {docs: ok(ask(q, docs), want, forbid) for docs in ("off", "on")}
    fails += not all(res.values())
    print(f"{'PASS' if all(res.values()) else 'FAIL'} {q!r}  without docs: {res['off']}  with docs: {res['on']}")
print(f"{len(CASES) * 2 - fails}/{len(CASES) * 2} ok" if not fails else f"{fails} case(s) failed")
sys.exit(1 if fails else 0)
