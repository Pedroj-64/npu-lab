"""Project mode: the model sees the map of a folder the user picked and proposes edits as SEARCH/REPLACE blocks; the user applies them with a button.
  system_block(root) -> str        text added to the system prompt (format rules + the repo map): stable between turns, so it stays in the cache
  attach(prompt, root) -> str      the question with the full text of the files it names (max 3)
  parse(text) -> [(path, search, replace)]
  apply(root, text, dry=False) -> [{"file", "action", "error"}]   all or nothing: nothing is written if any block fails"""
import os, re
import repomap

MAX_FILE, MAX_ATTACH = 1_000_000, 6000
DENY = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ssh", ".gnupg"}
BLOCK = re.compile(r"(?:^|\n)[ \t]*`*([^\n<>`]+?)`*[ \t]*\n<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
FORMAT = ("You are editing the user's project. When a change is needed, answer with one block per change, exactly:\n"
          "path/relative/to/the/project.py\n<<<<<<< SEARCH\n(lines copied exactly from the current file; empty to create a new file)\n=======\n(the new lines)\n>>>>>>> REPLACE\n"
          "To add code at the end of a file, SEARCH for its last lines and repeat them in REPLACE followed by the new code. Do not rewrite what does not change. "
          "Explain briefly outside the blocks. If you have not seen a file in full, ask for it by name instead of guessing its contents.")


def system_block(root):
    return "\n\n" + FORMAT + "\n\nMap of the project (signatures only):\n" + repomap.build(root, 3000)


def _safe(root, rel, must_exist):
    """-> absolute path inside root, or raise ValueError."""
    rel = rel.strip().replace("\\", "/")
    if not rel or os.path.isabs(rel) or ".." in rel.split("/") or any(p in DENY for p in rel.split("/")):
        raise ValueError(f"path not allowed: {rel!r}")
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, rel))
    if not (path == base or path.startswith(base + os.sep)):
        raise ValueError(f"path leaves the project: {rel!r}")
    if must_exist and not os.path.isfile(path):
        raise ValueError(f"no such file: {rel}")
    return path


def attach(prompt, root):
    names = {}
    for d, subs, files in os.walk(root):
        subs[:] = [s for s in subs if s not in DENY and not s.startswith(".")]
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if re.search(r"(?<![\w./-])" + re.escape(rel) + r"(?![\w])|(?<![\w./-])" + re.escape(f) + r"(?![\w])", prompt):
                names[rel] = os.path.join(d, f)
        if len(names) > 20:
            break
    out = []
    for rel in sorted(names)[:3]:
        try:
            t = open(names[rel], encoding="utf-8").read()[:MAX_ATTACH]
        except (OSError, UnicodeDecodeError):
            continue
        out.append(f"### {rel}\n```\n{t}\n```")
    return ("Files you asked about:\n\n" + "\n\n".join(out) + "\n\nQuestion: " + prompt) if out else prompt


def parse(text):
    return [(m.group(1).strip(), m.group(2).strip("\n"), m.group(3).strip("\n")) for m in BLOCK.finditer(text)]


def apply(root, text, dry=False):
    blocks = parse(text)
    if not blocks:
        return [{"file": "", "action": "none", "error": "no edit block in the answer"}]
    new, res = {}, []
    for rel, a, b in blocks:
        try:
            path = _safe(root, rel, must_exist=bool(a))
            cur = new[path] if path in new else (open(path, encoding="utf-8").read() if os.path.isfile(path) else "")
            if len(cur) > MAX_FILE:
                raise ValueError("file too large")
            if a:
                if cur.count(a) != 1:
                    raise ValueError(f"SEARCH text found {cur.count(a)} times (must be exactly once)")
                cur = cur.replace(a, b, 1)
            elif os.path.exists(path) or path in new:
                raise ValueError("file already exists: give the lines to search for")
            else:
                cur = b
            new[path] = cur if cur.endswith("\n") else cur + "\n"
            res.append({"file": rel, "action": "create" if not a else "edit", "error": ""})
        except (ValueError, OSError, UnicodeDecodeError) as e:
            return res + [{"file": rel, "action": "failed", "error": str(e)}]        # all or nothing: nothing was written
    if not dry:
        for path, t in new.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            open(tmp, "w", encoding="utf-8").write(t)
            os.replace(tmp, path)
    return res
