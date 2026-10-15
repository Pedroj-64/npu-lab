"""Sentinel: watches your git repos and tells you what was left half done (uncommitted files, unpushed commits). No model, no network, no NPU:
plain `git` calls, so it costs almost nothing while it waits. Each repo is announced once per change; a new desktop notice only when its state changes.
usage: python src/sentinel.py --roots ~/Documentos/Codigo [--once | --every 900] [--no-notify]
Spanish: centinela de repos: avisa de trabajo sin commitear o sin subir; solo git, sin modelo ni red."""
import argparse, json, os, shutil, subprocess, sys, time

STATE = os.path.expanduser("~/.local/share/npu-lab/sentinel.json")


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=20)
    return r.stdout.strip() if r.returncode == 0 else None


def repos(roots, depth=2):
    """Folders with a .git inside the roots (depth levels down); hidden folders and build output are skipped."""
    found = []
    for root in roots:
        root = os.path.realpath(os.path.expanduser(root))
        base = root.count(os.sep)
        for d, dirs, _ in os.walk(root):
            if ".git" in dirs:
                found.append(d); dirs[:] = []; continue
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("node_modules", "build", "target", ".venv", "venv")]
            if d.count(os.sep) - base >= depth:
                dirs[:] = []
    return sorted(set(found))


def inspect(repo):
    """{'repo', 'dirty', 'ahead'} or None when git cannot read it."""
    st = git(repo, "status", "--porcelain")
    if st is None:
        return None
    ahead = git(repo, "rev-list", "--count", "@{u}..HEAD")      # None without an upstream: nothing to say about pushing
    return {"repo": repo, "dirty": len([l for l in st.splitlines() if l]), "ahead": int(ahead) if ahead and ahead.isdigit() else 0}


def events(roots):
    return [e for e in map(inspect, repos(roots)) if e and (e["dirty"] or e["ahead"])]


def text(e):
    parts = ([f"{e['dirty']} uncommitted"] if e["dirty"] else []) + ([f"{e['ahead']} unpushed"] if e["ahead"] else [])
    return f"{os.path.basename(e['repo'])}: " + ", ".join(parts)


def new_only(evs, seen):
    """Events whose state differs from the last time they were announced; returns (to_announce, updated_seen)."""
    now = {e["repo"]: (e["dirty"], e["ahead"]) for e in evs}
    fresh = [e for e in evs if list(seen.get(e["repo"], [])) != list(now[e["repo"]])]
    return fresh, {k: list(v) for k, v in now.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=0, help="seconds between scans (0 or --once: one scan)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            seen = json.load(open(STATE))
        except (OSError, ValueError):
            seen = {}
        fresh, seen = new_only(events(a.roots), seen)
        for e in fresh:
            print(text(e), flush=True)
            if not a.no_notify and shutil.which("notify-send"):
                subprocess.run(["notify-send", "-a", "npu-lab", "Left half done", text(e)])
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        json.dump(seen, open(STATE, "w"))
        if a.once or not a.every:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
