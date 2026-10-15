"""Phone bridge: a Telegram bot that talks to YOUR npu-lab server on this PC. Standard library only.
  python bridge/telegram_bridge.py            (long polling; run it as the user service tools/systemd/npu-telegram.service)
Config: ~/.config/npu-lab/telegram.json  {"token": "...", "allowed_user_ids": [123456789]}   (mode 600, never in the repo)
The server is reached at 127.0.0.1:$NPU_LAB_PORT (default 8765) with the token in ~/.config/npu-lab/token, like an editor would.
Safety: only the listed Telegram user ids get an answer (everyone else is ignored, nothing is sent back); no command reads a file or runs a shell command;
the bot never receives file contents, only short model answers and counts. Telegram's servers do see the messages (bots are not end-to-end encrypted).
Spanish: puente al celular: bot de Telegram que habla con el servidor npu-lab de este PC; solo responde a los IDs permitidos y no lee archivos ni ejecuta comandos."""
import json, os, re, socket, sys, time, urllib.error, urllib.request

CONF = os.path.expanduser("~/.config/npu-lab/telegram.json")
TOKEN_FILE = os.path.expanduser("~/.config/npu-lab/token")
HELP = ("npu-lab on your PC. Messages pass through Telegram's servers.\n\n"
        "/status  what is loaded and how many repos have pending work\n/repos  repos with uncommitted or unpushed work\n/summary  two lines about them, from the local model\n"
        "/commit NAME  secret check + a draft commit message for that repo\n/explain COMMAND  what a shell command does\n/log UNIT  what a user service logged\n"
        "/bench  last speed measurement\n/load [qwen3|qwen3s]  load a model on the NPU\n/unload  free the NPU\n\nAnything else is a question for the local model.")


_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, *args, **kw):
    """Try IPv4 addresses before IPv6: on a network whose IPv6 route to Telegram hangs, the first connect attempt would otherwise wait for its whole timeout."""
    return sorted(_getaddrinfo(host, port, family, *args, **kw), key=lambda a: a[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first


class LocalError(Exception):
    pass


class Local:
    """The npu-lab server of this PC."""
    def __init__(self, port=None, token=None):
        self.base = f"http://127.0.0.1:{port or os.environ.get('NPU_LAB_PORT') or 8765}"
        try:
            self.token = token or open(TOKEN_FILE).read().strip()
        except OSError:
            raise LocalError("no server token found: start npu-lab once (it creates ~/.config/npu-lab/token)")

    def call(self, path, body=None, timeout=330):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except ValueError:
                msg = ""
            raise LocalError(msg or f"server answered {e.code}")
        except (urllib.error.URLError, OSError):
            raise LocalError("the npu-lab server is not running on this PC (open npu-lab from the tray)")

    def chat(self, messages, n=300):
        r = self.call("/v1/chat/completions", {"messages": messages, "max_tokens": n})
        return re.sub(r"<think>[\s\S]*?(</think>|$)", "", r["choices"][0]["message"]["content"]).strip()


def _clip(t, n=3500):
    return t if len(t) <= n else t[:n] + "\n[...]"


def handle(text, local, history, say=None):
    """Reply text for one allowed user's message. `history`: this user's list of {role, content} (edited in place, last 6 turns kept). `say`: optional function that sends an interim message (a model load takes about a minute)."""
    text = (text or "").strip()
    if not text:
        return "Send text."
    cmd, _, arg = text.partition(" ")
    arg = arg.strip()
    cmd = cmd.split("@")[0].lower()
    try:
        if cmd in ("/start", "/help"):
            return HELP
        if cmd == "/status":
            s = local.call("/status")
            ev = local.call("/sentinel")["events"]
            return (f"Model: {s['model'] or 'none loaded'} ({s['profile']}){' - busy' if s.get('busy') else ''}\nRepos with pending work: {len(ev)}")
        if cmd == "/repos":
            ev = local.call("/sentinel")["events"]
            if not ev:
                return "Everything is committed and pushed."
            return "\n".join(f"{e['name']}: " + ", ".join(x for x in (f"{e['dirty']} uncommitted" if e["dirty"] else "", f"{e['ahead']} unpushed" if e["ahead"] else "") if x) for e in ev[:20])
        if cmd == "/summary":
            ev = local.call("/sentinel")["events"]
            if not ev:
                return "Nothing pending."
            lst = "; ".join(f"{e['name']} ({e['dirty']} uncommitted, {e['ahead']} unpushed)" for e in ev[:20])
            return _clip(local.chat([{"role": "user", "content": "In two short sentences, which of these projects should I look at first and why? " + lst}], 120))
        if cmd == "/commit":
            if not arg:
                return "Usage: /commit NAME   (one of the repos from /repos)"
            ev = {e["name"].lower(): e for e in local.call("/sentinel")["events"]}
            e = ev.get(arg.lower())
            if not e:
                return "That repo has nothing pending (or is not in the watched folders)."
            r = local.call("/helpers/commit", {"repo": e["repo"]})
            lines = [f"{r['repo']}: {len(r['files'])} changed files"]
            if r["secrets"] or r["secrets_untracked"]:
                lines.append("POSSIBLE SECRETS (values not shown): " + ", ".join(f"{x['kind']} at line {x['line']}" for x in r["secrets"]) + ", ".join(f"{x['kind']} in {x['file']}" for x in r["secrets_untracked"]))
            lines.append(r["message"] or r.get("note", ""))
            return _clip("\n".join(l for l in lines if l))
        if cmd == "/explain":
            return _clip(local.call("/helpers/explain", {"command": arg})["answer"]) if arg else "Usage: /explain COMMAND"
        if cmd == "/log":
            return _clip(local.call("/helpers/triage", {"unit": arg})["answer"]) if arg else "Usage: /log UNIT   (a user service, e.g. npu-sentinel.service)"
        if cmd == "/bench":
            h = local.call("/bench")["history"]
            if not h:
                return "No measurements yet."
            r = h[-1]
            return f"{r['ts'].replace('T', ' ')}: decode {r['decode']} tok/s, 256-token prompt {r.get('prefill_256')} tok/s" + (" - REGRESSION" if r.get("regression") else "")
        if cmd == "/load":
            m = arg if arg in ("qwen3", "qwen3s") else "qwen3s"
            if say:
                say(f"Loading {m} on the NPU, about a minute (the 1.7B can take longer)...")
            local.call("/mode", {"mode": "llm", "model": m})
            return f"{m} loaded."
        if cmd == "/unload":
            local.call("/mode", {"mode": "none"})
            return "NPU free."
        if cmd.startswith("/"):
            return "Unknown command. /help"
        history.append({"role": "user", "content": text[:2000]})
        del history[:-5]                                   # room for the answer: 6 messages at most
        while history and history[0]["role"] != "user":
            history.pop(0)
        ans = local.chat(history)
        history.append({"role": "assistant", "content": ans})
        return _clip(ans) or "(no answer)"
    except LocalError as e:
        if history and history[-1]["role"] == "user":
            history.pop()
        return f"Could not do that: {e}"


class Bridge:
    def __init__(self, token, allowed, local, base="https://api.telegram.org"):
        self.token, self.allowed, self.local, self.base, self.hist, self.offset = token, set(allowed), local, base, {}, 0

    def api(self, method, params, timeout=70):
        req = urllib.request.Request(f"{self.base}/bot{self.token}/{method}", data=json.dumps(params).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"telegram {method}: HTTP {e.code}")           # never the URL: it contains the token
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(f"telegram {method}: {type(e).__name__}")

    def process(self, upd):
        """Handle one update; returns the reply that was sent or None (ignored)."""
        self.offset = max(self.offset, upd.get("update_id", 0) + 1)
        m = upd.get("message") or {}
        uid, chat, text = (m.get("from") or {}).get("id"), (m.get("chat") or {}).get("id"), m.get("text")
        if uid not in self.allowed or chat is None or not text:
            return None                                                          # strangers get no answer at all
        self.api("sendChatAction", {"chat_id": chat, "action": "typing"})
        reply = handle(text, self.local, self.hist.setdefault(uid, []), say=lambda t: self.api("sendMessage", {"chat_id": chat, "text": t}))
        self.api("sendMessage", {"chat_id": chat, "text": reply})
        return reply

    def run(self):
        print("npu-lab bridge running for", len(self.allowed), "allowed user(s)", flush=True)
        while True:
            try:
                for upd in self.api("getUpdates", {"timeout": 50, "offset": self.offset, "allowed_updates": ["message"]}, 70).get("result", []):
                    try:
                        self.process(upd)
                    except RuntimeError as e:
                        print(e, file=sys.stderr, flush=True)
            except RuntimeError as e:
                print(e, file=sys.stderr, flush=True)
                time.sleep(10)


def main():
    try:
        c = json.load(open(CONF))
        token, allowed = c["token"], [int(x) for x in c["allowed_user_ids"]]
    except (OSError, ValueError, KeyError):
        sys.exit(f"missing or invalid {CONF} (needs token and allowed_user_ids)")
    if not allowed:
        sys.exit("allowed_user_ids is empty: the bot would answer nobody")
    Bridge(token, allowed, Local()).run()


if __name__ == "__main__":
    main()
