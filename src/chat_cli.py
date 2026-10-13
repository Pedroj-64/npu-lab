"""Terminal chat for the local models. Starts its own server (so close the web launcher first: one program at a time on the NPU/GPU), loads a model and talks to
the same API as the web page. With a project folder and /agent on, the model proposes one action per reply: reads run alone, anything that writes asks y/N first.
usage: python src/chat_cli.py [--model qwen35l] [--project ~/code/app] [--agent] [--tokens 600]
       python src/chat_cli.py --check        offline self-check of the helpers (no model, no server)
In the chat: /help /new /project DIR /agent on|off /model NAME /quit   (Ctrl+C stops an answer; Ctrl+D leaves)"""
import http.client, json, os, re, subprocess, sys, tempfile, threading, time
try:
    import readline  # noqa: F401  (line editing and arrow-key history for input())
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
TTY = sys.stdout.isatty() and "NO_COLOR" not in os.environ
paint = lambda code, t: f"\033[{code}m{t}\033[0m" if TTY else t
dim, bold, red, green, cyan, yellow = (lambda t, c=c: paint(c, t) for c in ("2", "1", "31", "32", "36", "33"))
THINK = re.compile(r"<think>[\s\S]*?(</think>|$)")
MAX_STEPS = 12


def colour_diff(text):
    """+/- lines green/red, hunk headers cyan, the rest dim."""
    out = []
    for l in text.splitlines():
        out.append(green(l) if l.startswith("+") and not l.startswith("+++") else red(l) if l.startswith("-") and not l.startswith("---") else cyan(l) if l.startswith("@@") else dim(l))
    return "\n".join(out)


class Server:
    def __init__(self):
        self.token = "c" * 43
        self.proc = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                                     env={**os.environ, "NPU_LAB_TOKEN": self.token})
        self.port = json.loads(self.proc.stdout.readline())["port"]

    def conn(self, timeout=600):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def call(self, path, body=None, timeout=600):
        c = self.conn(timeout)
        c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None, {"Authorization": "Bearer " + self.token, "Host": f"127.0.0.1:{self.port}"})
        r = c.getresponse(); data = json.loads(r.read())
        if r.status != 200:
            raise RuntimeError(data.get("error", f"{path} -> {r.status}"))
        return data

    def stream(self, body):
        c = self.conn()
        c.request("POST", "/chat", json.dumps(body), {"Authorization": "Bearer " + self.token, "Host": f"127.0.0.1:{self.port}"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(json.loads(r.read()).get("error", f"/chat -> {r.status}"))
        return c, r

    def close(self):
        try:
            self.call("/mode", {"mode": "none"}, 30)
        except Exception:
            pass
        self.proc.terminate()


class Spinner:
    """One status line that rewrites itself while the model reads or thinks."""
    def __init__(self):
        self.text, self.stop, self.t = "", threading.Event(), None

    def set(self, text):
        self.text = text
        if TTY and self.t is None:
            self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        for i in range(10 ** 9):
            if self.stop.wait(0.12): break
            if self.text: sys.stdout.write(f"\r\033[K{dim('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[i % 10] + ' ' + self.text)}"); sys.stdout.flush()

    def clear(self):
        self.text = ""
        if TTY: sys.stdout.write("\r\033[K"); sys.stdout.flush()


class Chat:
    def __init__(self, srv, tokens):
        self.srv, self.tokens, self.history, self.agent, self.project, self.model = srv, tokens, [], False, "", ""

    def ask(self, prompt):
        """One model reply, streamed. -> the answer without its <think> part, or None when stopped/failed."""
        sp, text, shown, stats = Spinner(), "", 0, {}
        sp.set("thinking")
        c = None
        try:
            c, r = self.srv.stream({"prompt": prompt, "tokens": self.tokens, "history": self.history, "docs": "off", "web": "off", "mine": "off", "agent": self.agent})
            for line in r:
                m = json.loads(line)
                if m.get("phase") == "reading": sp.set(f"reading {m['tokens']} tokens")
                elif m.get("phase") == "compacting": sp.set("summarising old turns")
                if m.get("warning"): sp.clear(); print(yellow("! " + m["warning"]))
                if m.get("error"): sp.clear(); print(red("error: " + str(m["error"]))); return None
                if m.get("think"): sp.set("thinking")
                if m.get("token"):
                    text += m["token"]
                    visible = THINK.sub("", text) if "<think>" in text else text
                    if "<think>" in text and "</think>" not in text: continue
                    if visible[shown:]: sp.clear(); sys.stdout.write(visible[shown:]); sys.stdout.flush(); shown = len(visible)
                if m.get("compacted"):
                    s = m["compacted"]
                    self.history = [{"role": "user", "content": "[Notes of the earlier conversation]\n" + s["summary"]}, {"role": "assistant", "content": "Understood, I will use those notes."}, *self.history[-s["kept"]:]]
                    print(dim("\n(older turns were summarised)"), end="")
                if m.get("done") is True:
                    stats = m.get("stats", {})
                    answer = THINK.sub("", text).strip()
                    self.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                    # keep reading: notes from a compaction may follow on the same stream
        except KeyboardInterrupt:
            sp.clear(); print(dim("\n(stopped)")); return None
        except (RuntimeError, OSError) as e:
            sp.clear(); print(red(f"error: {e}")); return None
        finally:
            sp.stop.set(); sp.clear()
            if c: c.close()
        if stats:
            print(dim(f"\n{stats.get('decode_tokens', 0) / max(stats.get('decode_s', 1e-9), 1e-9):.1f} tok/s · {stats.get('prefill_tokens', 0)} tokens read ({stats.get('reused_tokens', 0)} reused)"))
        return THINK.sub("", text).strip()

    def edits(self, answer):
        """An answer with SEARCH/REPLACE blocks outside agent mode: preview, then write only after y."""
        if not self.project or "<<<<<<< SEARCH" not in answer: return
        try:
            d = self.srv.call("/project/apply", {"text": answer, "dry": True})
            bad = [r for r in d["results"] if r["action"] in ("failed", "none")]
            for r in d["results"]: print(("  " + red("✗ ") if r in bad else "  " + green("✓ ")) + f"{r['action']}  {r.get('file', '')}  {r.get('error', '')}")
            if bad or input(bold("Apply these edits? [y/N] ")).strip().lower() != "y": return
            print(green("applied: ") + ", ".join(r["file"] for r in self.srv.call("/project/apply", {"text": answer})["results"]))
        except (RuntimeError, EOFError) as e:
            print(red(str(e)))

    def agent_loop(self, answer, task=""):
        ok = True
        for step in range(MAX_STEPS):
            try:
                r = self.srv.call("/agent/run", {"text": answer, "last_ok": ok, "task": task})
                if r["needs_confirm"]:
                    print(colour_diff(r["shown"]))
                    if input(bold("Do it? [y/N] ")).strip().lower() != "y": print(dim("agent stopped; nothing was changed")); return
                    r = self.srv.call("/agent/run", {"text": answer, "confirm": True, "last_ok": ok, "task": task})
                    print(green("done: ") + dim(r["shown"][:300]) if r["ok"] else red("failed: ") + r["shown"][:300])
                else:
                    print(dim(("→ " if r["ok"] else "✗ ") + r["shown"][:300].replace("\n", "\n  ")))
                ok = r["ok"]
                if r["done"]: print(green("✓ ") + "agent finished"); return
            except (RuntimeError, EOFError, KeyboardInterrupt) as e:
                print(red(str(e) or "agent stopped")); return
            print(cyan("\nagent ▸ "), end="")
            answer = self.ask(("Result: " if ok else "Error: ") + r["shown"] + "\nNext action?")
            if not answer: return
        print(yellow(f"agent paused after {MAX_STEPS} steps; write a message to continue"))

    def command(self, line):
        cmd, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        if cmd in ("quit", "exit", "q"): raise EOFError
        if cmd == "new": self.history = []; print(dim("new conversation"))
        elif cmd == "project":
            try: self.project = self.srv.call("/rules", {"project_root": os.path.expanduser(rest)}).get("project_root", ""); print(dim(f"project: {self.project or '(none)'}"))
            except RuntimeError as e: print(red(str(e)))
        elif cmd == "agent":
            self.agent = rest == "on" and bool(self.project)
            print(dim("agent mode " + ("on" if self.agent else "off" + ("" if self.project or rest != "on" else " (set a /project folder first)"))))
        elif cmd == "model": self.load(rest or self.model)
        else: print(__doc__.split("In the chat:")[1].strip() if "In the chat:" in __doc__ else "")

    def load(self, model):
        sp = Spinner(); sp.set(f"loading {model} (the 4B takes ~40 s)")
        try:
            self.srv.call("/mode", {"mode": "llm", "model": model}); self.model = model; sp.clear(); print(green("✓ ") + f"{model} loaded")
        except RuntimeError as e:
            sp.clear(); print(red(f"could not load {model}: {e}")); return False
        finally:
            sp.stop.set(); sp.clear()
        return True


def main():
    srv = Server()
    try:
        srv.call("/profile", {"profile": "assisted"})
        chat = Chat(srv, arg("--tokens", 600))
        if not chat.load(arg("--model", "qwen35l")): return 1
        if "--project" in sys.argv: chat.command("/project " + arg("--project", ""))
        if "--agent" in sys.argv: chat.command("/agent on")
        print(bold("npu-lab chat") + dim("  /help for commands, Ctrl+C stops an answer, Ctrl+D leaves"))
        while True:
            try:
                line = input((cyan("agent") if chat.agent else cyan("you")) + " ▸ ").strip()
                if not line: continue
                if line.startswith("/"): chat.command(line); continue
                print(green("\nmodel ▸ "), end="")
                answer = chat.ask(line)
                if answer: chat.agent_loop(answer, line) if chat.agent else chat.edits(answer)
                print()
            except KeyboardInterrupt:
                print()
    except (EOFError, KeyboardInterrupt):
        print(dim("\nbye"))
    finally:
        srv.close()
    return 0


def check():
    assert colour_diff("+a\n-b\n@@ x\n c").count("\n") == 3
    assert THINK.sub("", "<think>hmm</think>hello") == "hello" and THINK.sub("", "<think>still thinking") == ""
    print("chat_cli helpers ok")


if __name__ == "__main__":
    check() if "--check" in sys.argv else sys.exit(main())
