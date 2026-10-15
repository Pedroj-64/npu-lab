"""Developer helpers that need no cloud: look at a repo's pending diff, check it for secrets, and build the prompts a small local model answers.
Everything here is plain git and regular expressions, so the checks are exact and cost nothing; only the wording of a commit message or an explanation is left to the model.
Spanish: ayudas de programación: diff pendiente, revisión de secretos (exacta, sin modelo) y prompts para el modelo local."""
import os, re, subprocess

SKIP = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "uv.lock", "*.min.js", "*.min.css", "*.svg", "*.png", "*.jpg", "*.pdf")
SECRET_PATTERNS = [
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("Hugging Face token", r"\bhf_[A-Za-z0-9]{20,}"),
    ("GitHub token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}"),
    ("Telegram bot token", r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}"),
    ("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("API key (sk-...)", r"\bsk-[A-Za-z0-9_-]{30,}"),
    ("Google API key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("password or key assigned in code", r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[=:]\s*[\"'][^\"'\s]{8,}[\"']"),
]
SECRET_RE = [(n, re.compile(p)) for n, p in SECRET_PATTERNS]


class HelperError(Exception):
    pass


def repo_ok(repo, roots):
    """Real path of a git repo that lives inside one of the watched `roots`; refuses anything else."""
    real = os.path.realpath(os.path.expanduser(str(repo or "")))
    inside = any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(os.path.expanduser(x)) for x in roots))
    if not inside or not os.path.isdir(os.path.join(real, ".git")):
        raise HelperError("not a git repo inside the watched folders")
    return real


def _git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise HelperError(r.stderr.strip()[:200] or "git failed")
    return r.stdout


def pending_diff(repo, limit=6000):
    """(files, diff text) of everything not yet committed (tracked changes; new untracked files are listed by name only). The diff is cut to `limit` characters."""
    ex = [f":(exclude){p}" for p in SKIP]
    files = [l[3:] for l in _git(repo, "status", "--porcelain").splitlines() if l.strip()]
    diff = _git(repo, "diff", "HEAD", "--", ".", *ex)
    if len(diff) > limit:
        diff = diff[:limit // 3] + "\n[... cut ...]\n" + diff[-(2 * limit) // 3:]
    return files, diff


def secret_findings(text):
    """[{'kind', 'line'}]: where something that looks like a secret appears in the ADDED lines of a diff. The value itself is never returned."""
    out, line = [], 0
    for raw in text.splitlines():
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line = int(m.group(1)) - 1 if m else line
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            line += 1
            for kind, rx in SECRET_RE:
                if rx.search(raw):
                    out.append({"kind": kind, "line": line}); break
        elif not raw.startswith("-"):
            line += 1
    return out


def scan_untracked(repo, limit_bytes=200_000):
    """The same secret check on new, untracked text files (git diff does not show them)."""
    out = []
    for l in _git(repo, "status", "--porcelain").splitlines():
        if l.startswith("??"):
            p = os.path.join(repo, l[3:].strip().rstrip("/"))
            if os.path.isfile(p) and os.path.getsize(p) <= limit_bytes:
                try:
                    text = open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                for kind, rx in SECRET_RE:
                    if rx.search(text):
                        out.append({"kind": kind, "file": l[3:].strip()}); break
    return out


def commit_prompt(files, diff):
    return ("Write a git commit message for this change: one summary line of at most 70 characters in the imperative mood, then if needed a blank line and up to 3 short bullet points. "
            "Plain text only, no quotes, no code fences, no mention of tools or assistants.\n\nFiles: " + ", ".join(files[:25]) + "\n\nDiff:\n" + diff)


def explain_prompt(command, reference=""):
    return ("Explain what this shell command does, in at most 4 short lines, and say if it is destructive.\n\nCommand: " + command[:300]
            + (f"\n\nReference excerpts (newer than your memory):\n{reference[:1500]}" if reference else ""))


def triage_prompt(unit, log):
    return (f"These are the last lines of the log of the service {unit}. In at most 5 short lines say whether anything failed, the likely cause, and the first thing to check. "
            "If everything looks fine, say so in one line.\n\n" + log[-5000:])


def unit_log(unit, lines=80):
    """Last journal lines of a USER systemd unit (the name is validated: letters, digits, dots, dashes, underscores)."""
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,80}", unit or ""):
        raise HelperError("unit names use letters, digits and . _ - @ only")
    r = subprocess.run(["journalctl", "--user", "-u", unit, "-n", str(lines), "--no-pager"], capture_output=True, text=True, timeout=20)
    if r.returncode or not r.stdout.strip():
        raise HelperError("no log for that user unit")
    return r.stdout
