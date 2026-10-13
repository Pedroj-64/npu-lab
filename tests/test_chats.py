"""Offline test of saved chats (no model, no NPU, ~0.1 s). usage: python tests/test_chats.py"""
import os, sys, tempfile
os.environ["NPU_LAB_CHATS"] = os.path.join(tempfile.mkdtemp(), "chats")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import chats

H = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "¡Hola!"}]
assert chats.listing() == []
m = chats.save("abc123", "", H)
assert m["title"] == "hola" and m["messages"] == 2
assert chats.load("abc123")["history"] == H
chats.save("def456", "segundo", H + H)
assert [c["id"] for c in chats.listing()] in (["def456", "abc123"], ["abc123", "def456"]) and len(chats.listing()) == 2
for bad in ("../x", "a/b", "ab", "x" * 65, "", None):
    try:
        chats.save(bad, "t", H); raise SystemExit(f"accepted bad id {bad!r}")
    except chats.ChatError:
        pass
for bad in ([], [{"role": "system", "content": "x"}], [{"role": "user", "content": ""}]):
    try:
        chats.save("ok1234", "t", bad); raise SystemExit("accepted an empty chat")
    except chats.ChatError:
        pass
chats.delete("abc123")
try:
    chats.load("abc123"); raise SystemExit("loaded a deleted chat")
except chats.ChatError:
    pass
assert [c["id"] for c in chats.listing()] == ["def456"]
print("chats ok")

# the same through the HTTP API (server started without loading any model, so the NPU is untouched)
import http.client, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = "t" * 43
tmp = tempfile.mkdtemp()
p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                     env={**os.environ, "NPU_LAB_TOKEN": T, "NPU_LAB_RULES": tmp + "/rules.json", "NPU_LAB_CHATS": tmp + "/c", "NPU_LAB_FACTS": tmp + "/f.json", "NPU_LAB_DOCS": tmp + "/d", "NPU_LAB_ROUTING": tmp + "/r.json", "NPU_LAB_SETTINGS": tmp + "/s.json"})
try:
    port = json.loads(p.stdout.readline())["port"]
    def call(path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None, {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
        r = c.getresponse(); return r.status, json.loads(r.read())
    assert call("/chats") == (200, {"chats": []})
    s, m = call("/chats/save", {"id": "web001", "title": "t", "history": H}); assert s == 200 and m["messages"] == 2, (s, m)
    assert call("/chats/load", {"id": "web001"})[1]["history"] == H
    assert call("/chats/save", {"id": "../../etc", "history": H})[0] == 400
    assert call("/chats/load", {"id": "nope00"})[0] == 400
    assert call("/rules", {"save_chats": True})[1]["save_chats"] is True
    assert call("/chats/delete", {"id": "web001"}) == (200, {"chats": []})
    assert call("/facts") == (200, {"facts": []}) and call("/facts", {"facts": [" a  b ", ""]}) == (200, {"facts": ["a b"]}) and call("/facts", {"facts": "x"})[0] == 400
    print("chats api ok")
finally:
    p.terminate()
