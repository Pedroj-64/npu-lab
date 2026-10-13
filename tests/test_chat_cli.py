"""The terminal chat against a real server (no model loaded) with a scripted model: streaming, history, the agent loop with its y/N gate, edit preview.
usage: .venv/bin/python tests/test_chat_cli.py"""
import builtins, io, json, os, shutil, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src"))
import chat_cli as C

tmp = tempfile.mkdtemp()
for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items(): os.environ[f"NPU_LAB_{k}"] = f"{tmp}/{v}"
os.environ["NPU_LAB_TRASH"] = tmp + "/trash"
proj = tempfile.mkdtemp(dir=os.path.expanduser("~"))
open(proj + "/a.py", "w").write("x = 1\n"); open(proj + "/old.py", "w").write("y = 2\n")
srv = C.Server()


class Fake:                         # stands in for the /chat stream: replays scripted replies as the server would send them
    def __init__(self, replies): self.replies, self.prompts = list(replies), []
    def __call__(self, body):
        self.prompts.append(body["prompt"]); text = self.replies.pop(0)
        lines = [json.dumps({"phase": "reading", "tokens": 5}), json.dumps({"think": "hm"})] + [json.dumps({"token": t}) for t in (text[:3], text[3:])]
        lines.append(json.dumps({"done": True, "stats": {"decode_tokens": 4, "decode_s": 1.0, "prefill_tokens": 5, "reused_tokens": 0}}))
        return type("C", (), {"close": lambda s: None})(), [l.encode() for l in lines]


def say(inputs):                    # answer the y/N questions in order
    it = iter(inputs); builtins.input = lambda *_: next(it)


try:
    chat = C.Chat(srv, 50); assert srv.call("/rules", {"project_root": proj})["project_root"]
    chat.project = proj; srv.stream = Fake(["hello there"])
    assert chat.ask("hi") == "hello there" and chat.history[-2:] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]
    # agent: read (runs alone), delete (asks), done
    chat.agent = True; f = srv.stream = Fake(["READ a.py", "DELETE old.py", "DONE removed"]); say(["y"])
    chat.agent_loop(chat.ask("remove old.py"))
    assert not os.path.exists(proj + "/old.py") and len(f.prompts) == 3 and f.prompts[1].startswith("Result:"), f.prompts
    assert any(n == "old.py" for _, _, fs in os.walk(tmp + "/trash") for n in fs)
    # answering n changes nothing
    open(proj + "/old.py", "w").write("y = 2\n"); srv.stream = Fake(["DELETE old.py"]); say(["n"])
    chat.agent_loop(chat.ask("again")); assert os.path.exists(proj + "/old.py")
    # edit preview outside agent mode
    chat.agent = False; edit = "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
    say(["n"]); chat.edits(edit); assert open(proj + "/a.py").read() == "x = 1\n"
    say(["y"]); chat.edits(edit); assert open(proj + "/a.py").read() == "x = 2\n"
    print("chat cli ok")
finally:
    srv.close(); shutil.rmtree(proj, ignore_errors=True)
