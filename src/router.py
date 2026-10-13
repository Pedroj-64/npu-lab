"""Per-question routing: which sources (if any) are worth consulting. Pure rules, no model, no I/O.

  decide(prompt)         -> {"docs", "web", "history": bool, "reason", "sure"}   (what "Auto" does; criteria in ~/.config/npu-lab/routing.json)
  from_label(text)       -> the same, from the local model's one-word reply (used when no rule is sure and the classifier is on)
  resolve(mode, auto)    -> bool       mode is "auto" | "on" | "off" (or a bool from older clients)
  clean_web(results)     -> results without search-engine pages, duplicate domains or empty text
  search_query(prompt, history) -> standalone web query
"""
import datetime, json, os, re, urllib.parse

PATH = os.environ.get("NPU_LAB_ROUTING") or os.path.expanduser("~/.config/npu-lab/routing.json")   # env override is for tests
# Editable criteria: each list is regex fragments joined with | and matched as whole words, case-insensitive.
DEFAULTS = {
    "history": ["above", "earlier", "you said", "you told", "we (talked|discussed|were)", "this (chat|conversation)", "repeat (that|it)",
                "summari[sz]e (the|this|our|that)", "we talk(ed)?", "my (first|last|previous) (question|message)", "(translate|shorten|rewrite|expand) (that|it|this)", "arriba", "antes", "dijiste", "hablamos", "de qu[eé] (hablamos|conversamos)", "este chat",
                "esta conversaci[oó]n", "repite", "resume (lo|esto|la conversaci)"],
    "fresh": ["today", "yesterday", "tonight", "latest", "news", "current(ly)?", "right now", "price of", "weather", "score", "who (is|won)",
              "what happened", "hoy", "ayer", "anoche", "[uú]ltim[oa]s?", "noticias", "actual(mente)?", "ahora mismo", "precio", "clima",
              "resultado", "qui[eé]n (es|gan[oó])", "qu[eé] pas[oó]"],
    "terminal": ["terminal", "command", "comando", "shell", "bash", "fish", "zsh", "sudo", "pacman", "paru", "yay", "systemctl", "journalctl",
                 "git", "ssh", "grep", "chmod", "mount", "ls", "cd", "cp", "mv", "rm", "tar", "curl", "wget", "docker", "aur", "arch", "linux",
                 "install", "instalar", "package", "paquete", "flag", "option", "opci[oó]n"],
    "programming": ["typescript", "javascript", "python", "rust", "java", "golang", "kotlin", "swift", "ruby", "php", "html", "css", "sql",
                    "react", "vue", "class", "clase", "function", "funci[oó]n", "variable", "array", "loop", "bucle", "algorithm", "algoritmo",
                    "regex", "api"],
}
FOLLOWUP = re.compile(r"^\s*(and|y|what about|y si|y en|and in|ahora|now|also|tambi[eé]n)\b", re.I)


def load():
    """Effective criteria: the defaults with whatever ~/.config/npu-lab/routing.json replaces. Bad entries are ignored."""
    cfg = {**{k: list(v) for k, v in DEFAULTS.items()}, "classifier": False}
    try:
        user = json.load(open(PATH))
    except (OSError, json.JSONDecodeError):
        return cfg
    for k in DEFAULTS:
        if isinstance(user.get(k), list):
            cfg[k] = [x for x in user[k] if isinstance(x, str) and 0 < len(x) <= 80][:200]
    cfg["classifier"] = bool(user.get("classifier"))
    return cfg


def validate(d):
    """Return the cleaned config or raise ValueError (lists of regex fragments that compile; classifier on/off)."""
    out = {"classifier": bool(d.get("classifier"))}
    for k in DEFAULTS:
        items = d.get(k, DEFAULTS[k])
        if not isinstance(items, list) or len(items) > 200:
            raise ValueError(f"{k}: a list of up to 200 words")
        for x in items:
            if not isinstance(x, str) or not 0 < len(x) <= 80:
                raise ValueError(f"{k}: entries are 1..80 characters")
            try:
                re.compile(x)
            except re.error as e:
                raise ValueError(f"{k}: {x!r} is not a valid pattern ({e})")
        out[k] = items
    return out


def save(d):
    d = validate(d)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump(d, open(PATH, "w"), indent=2, ensure_ascii=False)
    return d


def _rx(items):
    return re.compile(r"\b(?:" + "|".join(items) + r")\b", re.I) if items else None


def decide(prompt, cfg=None):
    """Rule layer. `sure` is False when no rule matched and the answer is only the default (the classifier may refine it)."""
    cfg = cfg or load()
    hist, fresh, term, prog = (_rx(cfg[k]) for k in ("history", "fresh", "terminal", "programming"))
    m = lambda rx: bool(rx and rx.search(prompt))
    r = lambda docs, web, history, why, sure=True: {"docs": docs, "web": web, "history": history, "reason": why, "sure": sure}
    if m(hist) or (FOLLOWUP.search(prompt) and len(prompt) < 60):
        return r(False, False, True, "refers to the conversation")
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", prompt)]
    if m(fresh) or any(y >= datetime.date.today().year for y in years):
        return r(False, True, False, "time-sensitive")
    if m(prog) and not m(term):
        return r(False, False, False, "general programming: answer directly")
    if m(term):
        return r(True, False, False, "names a tool or asks for a command")
    return r(True, False, False, "no rule matched; only relevant excerpts are used", sure=False)


LABELS = {"NONE": (False, False, False, "classifier: answer directly"), "DOCS": (True, False, False, "classifier: local docs"),
          "WEB": (False, True, False, "classifier: web"), "HISTORY": (False, False, True, "classifier: refers to the conversation")}
CLASSIFY_SYSTEM = ("You route questions to a source. Do not answer them.\n"
                   "NONE: general knowledge, programming, science, writing, concepts. Example: 'explain what a closure is'.\n"
                   "DOCS: how to use a Linux command, package manager or installed tool. Example: 'how do I delete a folder with rm'.\n"
                   "WEB: recent events, news, prices, versions, anything that changes over time. Example: 'who is the current champion'.\n"
                   "HISTORY: refers to what was said earlier in this conversation. Example: 'make that shorter'.")


def from_label(label):
    """Map a classifier label to a route."""
    hit = LABELS[label]
    return {"docs": hit[0], "web": hit[1], "history": hit[2], "reason": hit[3], "sure": True}


def escalate(prompt):
    """Hint (never automatic): a long or multi-part question is where a small local model struggles."""
    return len(prompt) > 400 or prompt.count("?") > 2


def resolve(mode, auto):
    if mode in (True, "on"):
        return True
    return bool(auto) if mode == "auto" else False


ENGINES = ("duckduckgo.com", "google.com", "bing.com", "yahoo.com", "baidu.com", "yandex.com", "ecosia.org", "startpage.com", "search.brave.com")


def _host(url):
    h = (urllib.parse.urlsplit(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def clean_web(results):
    out, seen = [], set()
    for r in results:
        h = _host(r["url"])
        if not h or h in seen or any(h == e or h.endswith("." + e) for e in ENGINES) or not (r.get("text") or r.get("snippet")):
            continue
        seen.add(h)
        out.append(r)
    return out


def search_query(prompt, history=()):
    """A short follow-up ("and in Python?") borrows the previous user question; otherwise the prompt is the query."""
    prev = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    return (prev[:120] + " " + prompt if prev and len(prompt) < 40 else prompt)[:200]
