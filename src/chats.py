"""Saved chats (opt-in, local): one JSON file per chat in ~/.local/share/npu-lab/chats. The browser sends a chat only when the user turned saving on.
  save(id, title, history) -> meta     listing() -> [meta] newest first     load(id) -> chat     delete(id)"""
import json, os, re, time

DIR = os.environ.get("NPU_LAB_CHATS") or os.path.expanduser("~/.local/share/npu-lab/chats")
ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")          # the id becomes a file name: nothing else gets through
MAX_MESSAGES, MAX_CHARS = 2000, 4_000_000


class ChatError(Exception):
    pass


def _path(i):
    if not isinstance(i, str) or not ID.match(i):
        raise ChatError("invalid chat id")
    return os.path.join(DIR, i + ".json")


def save(i, title, history):
    p = _path(i)
    msgs = [{"role": m["role"], "content": str(m["content"])} for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    if not msgs or len(msgs) > MAX_MESSAGES or sum(len(m["content"]) for m in msgs) > MAX_CHARS:
        raise ChatError(f"a chat needs 1..{MAX_MESSAGES} messages and at most {MAX_CHARS} characters")
    meta = {"id": i, "title": (str(title).strip() or msgs[0]["content"])[:80], "updated": int(time.time()), "messages": len(msgs)}
    os.makedirs(DIR, exist_ok=True)
    tmp = p + ".tmp"
    json.dump({**meta, "history": msgs}, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, p)
    return meta


def load(i):
    try:
        return json.load(open(_path(i)))
    except FileNotFoundError:
        raise ChatError("no such chat")


def listing():
    out = []
    for f in os.listdir(DIR) if os.path.isdir(DIR) else []:
        if f.endswith(".json"):
            try:
                c = json.load(open(os.path.join(DIR, f)))
                out.append({k: c[k] for k in ("id", "title", "updated", "messages")})
            except (OSError, ValueError, KeyError):
                pass
    return sorted(out, key=lambda c: -c["updated"])


def delete(i):
    try:
        os.remove(_path(i))
    except FileNotFoundError:
        raise ChatError("no such chat")
