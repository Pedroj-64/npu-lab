"""Terminal agent, server side: the model answers with ONE action per reply; this module checks and runs it inside the project folder.
Reading and listing run at once. Anything that writes (edit, create, delete) is only previewed until the caller passes confirmed=True.
Deleting moves the file to a trash folder outside the project (recoverable). DONE is not taken on the model's word.
  run(root, reply, confirmed=False, last_ok=True, test_command="", guard="off", backend=None, task="", tau=0.5) -> {"kind", "ok", "executed", "needs_confirm", "shown", "done"}
RUN (only when the user configured a test command): runs exactly that command inside the project, after the same confirmation as a write, and shows its output.
guard (off by default): "objective" = DONE is also refused when nothing changed, a file does not compile, or the configured tests were not run after the last change / fail;
"model" = in addition, the decision backend (src/decide.py) must give p(done) >= tau. The model's own claim is never the only evidence."""
import os, re, shlex, shutil, subprocess, time
import project, repomap, decide

TRASH = os.environ.get("NPU_LAB_TRASH") or os.path.expanduser("~/.local/share/npu-lab/trash")
PROTOCOL = """You are an agent that changes the files of a small Python project, one action per reply. Reply with exactly ONE action and nothing else:
READ path                  show a file
LIST                       list the project files
DELETE path                delete a file
DONE one-line summary      you are finished
or to edit or create a file, a block (the path on the line before it, written as in the project map, for example models.py):
models.py
<<<<<<< SEARCH
(lines copied exactly from the current file; leave empty to create a new file)
=======
(the new lines)
>>>>>>> REPLACE
You will see the result of each action. Read a file before editing it: the lines you search for must be copied from what READ showed. Edit one place per reply. DONE is refused if your last action failed or a file no longer compiles."""
PROTOCOL_RUN = "\nRUN                        run the project's tests (the user confirms first); run them after your last change and before DONE"
VERB = re.compile(r"\s*(READ|LIST|DELETE|DONE|RUN)\b[ \t]*([^\n]*)")
_STATE = {}                 # project root -> {"before": files at the first action, "tests": None | (passed, output) of the last RUN since the last change}
MAX_SNAPSHOT = 200_000


def snapshot(root):
    """{relative path: text} of the project's text files (denied folders and big or binary files skipped): the evidence the DONE guard compares."""
    out = {}
    for d, subs, fs in os.walk(root):
        subs[:] = [x for x in subs if x not in project.DENY and not x.startswith(".") and x != "__pycache__"]
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) <= MAX_SNAPSHOT and not f.endswith(".pyc"):
                    out[os.path.relpath(p, root)] = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass
    return out


def system_block(root, test_command=""):
    return "\n\n" + PROTOCOL + (PROTOCOL_RUN if test_command else "") + "\n\nProject map:\n" + repomap.build(root, 2500)


def listing(root):
    return "\n".join(sorted(os.path.relpath(os.path.join(d, f), root) for d, subs, fs in os.walk(root)
                            if not any(p in project.DENY for p in os.path.relpath(d, root).split(os.sep)) for f in fs if not f.endswith(".pyc")))


def compile_error(root):
    """First syntax error among the project's .py files, or ''."""
    for d, subs, fs in os.walk(root):
        subs[:] = [s for s in subs if s not in project.DENY and not s.startswith(".")]
        for f in sorted(fs):
            if f.endswith(".py"):
                try:
                    compile(open(os.path.join(d, f), encoding="utf-8").read(), f, "exec")
                except (SyntaxError, ValueError, UnicodeDecodeError) as e:
                    return f"{f} does not compile: {e}"
    return ""


def _r(kind, ok, shown, executed=False, needs_confirm=False, done=False):
    return {"kind": kind, "ok": ok, "executed": executed, "needs_confirm": needs_confirm, "shown": shown, "done": done}


def run(root, reply, confirmed=False, last_ok=True, test_command="", guard="off", backend=None, task="", tau=0.5):
    st = _STATE.setdefault(root, {"before": snapshot(root), "tests": None}) if (guard != "off" or test_command) else {"before": None, "tests": None}
    r = _run(root, reply, confirmed, last_ok, test_command, guard, backend, task, tau, st)
    if r["kind"] in ("edit", "delete") and r["executed"]:
        st["tests"] = None                              # the last test result is stale once a file changed
    if r["done"]:
        _STATE.pop(root, None)                          # the next task starts from a new baseline
    return r


def _run(root, reply, confirmed, last_ok, test_command, guard, backend, task, tau, st):
    text = reply.split("</think>")[-1].strip()
    if "<<<<<<< SEARCH" in text:
        res = project.apply(root, text, dry=not confirmed)
        bad = [r for r in res if r["action"] in ("failed", "none")]
        shown = "; ".join(f"{r['action']} {r['file']}" + (f" ({r['error']})" if r["error"] else "") for r in res)
        if bad and bad[-1]["error"].startswith("SEARCH text found"):          # show the real file, so the next reply can copy the lines exactly
            try:
                shown += f"\nThe file {bad[-1]['file']} currently contains:\n" + open(project._safe(root, bad[-1]["file"], True), encoding="utf-8").read()[:3000]
            except (ValueError, OSError, UnicodeDecodeError):
                pass
        return _r("edit", not bad, shown, executed=not bad and confirmed, needs_confirm=not bad and not confirmed)
    m = VERB.match(text)
    if not m:
        return _r("invalid", False, "not an action: reply with READ, LIST, DELETE, DONE or an edit block")
    verb, rest = m.group(1), m.group(2).strip().strip("`")
    if verb == "RUN":
        norm = lambda c: " ".join(c.split())
        if not test_command:
            return _r("run", False, "RUN is not available: no test command is configured")
        if rest and norm(rest) != norm(test_command):
            return _r("run", False, f"RUN only runs the configured test command ({test_command}); write RUN alone")
        if not confirmed:
            return _r("run", True, f"run the tests: {test_command}", needs_confirm=True)
        try:
            p = subprocess.run(shlex.split(test_command), cwd=root, capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            out = (p.stdout + p.stderr).strip()
            st["tests"] = (p.returncode == 0, out)
            return _r("run", True, f"exit code {p.returncode}\n" + out[-3000:], executed=True)
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            return _r("run", False, f"RUN failed: {e}")
    if verb == "DONE":
        why = "" if last_ok else "your last action failed; fix it first"
        why = why or compile_error(root)
        p = None
        if not why and guard != "off":
            why, p = _guard(root, rest, test_command, guard, backend, task, tau, st)
        res = _r("done", False, f"DONE refused: {why}") if why else _r("done", True, rest, executed=True, done=True)
        if p is not None:
            res["p_done"] = p
        return res
    if verb == "LIST":
        return _r("list", True, listing(root), executed=True)
    try:
        path = project._safe(root, rest, must_exist=True)
        if verb == "READ":
            return _r("read", True, open(path, encoding="utf-8").read()[:4000], executed=True)
        if not confirmed:
            return _r("delete", True, f"delete {rest} (moved to the trash, not erased)", needs_confirm=True)
        dest = os.path.join(TRASH, time.strftime("%Y%m%d-%H%M%S"), rest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
        return _r("delete", True, f"deleted {rest} (in the trash: {dest})", executed=True)
    except (ValueError, OSError, UnicodeDecodeError) as e:
        return _r(verb.lower(), False, f"{verb} failed: {e}")


def _guard(root, claim, test_command, guard, backend, task, tau, st):
    """-> (reason to refuse DONE or '', p(done) or None). Objective checks first; the model's probability is only a secondary signal."""
    after = snapshot(root)
    tests = st["tests"]
    if test_command and tests is None and after != st["before"]:
        return "run the project's tests (RUN) after your last change, then DONE", None
    state = decide.done_state(task or "(not given)", claim, st["before"], after, "", tests[1] if tests else None)
    ok, reason, p = decide.done_guard(st["before"], after, "", tests[0] if tests else None, backend if guard == "model" else None, state, tau)
    return ("" if ok else reason), p
