"""Local documentation index: man pages + tldr pages (+ Arch wiki if installed).

Grounds the LLM in CURRENT docs instead of what it memorised: the text is what is installed on this machine.
  python src/docs.py build [--device npu|gpu|cpu]     index everything (re-run after big upgrades)
  python src/docs.py ask "how do I install an AUR package" [-k 4]
Nothing here reads your personal files. Index: ~/.local/share/npu-lab/docs.pkl
"""
import argparse, glob, html, html.parser, os, pickle, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IDX = os.path.expanduser("~/.local/share/npu-lab/docs.pkl")
TLDR = os.path.expanduser("~/.cache/tealdeer/tldr-pages")
MAN_DIRS = [f"/usr/share/man/man{s}" for s in (1, 5, 8)]
WIKI = "/usr/share/doc/arch-wiki/html/en"
CHUNK, MAX_MAN_CHUNKS, MAX_WIKI_CHUNKS = 550, 10, 12


def chunk(text, size=CHUNK, max_n=None):
    """Split on paragraph/line boundaries into pieces of about `size` characters."""
    out, cur = [], ""
    for line in text.splitlines():
        if len(cur) + len(line) > size and cur.strip():
            out.append(cur.strip())
            cur = ""
        cur += line + "\n"
        if max_n and len(out) >= max_n:
            return out
    if cur.strip():
        out.append(cur.strip())
    return out[:max_n] if max_n else out


def tldr_docs():
    for lang in ("es", "en"):
        for sub in ("common", "linux"):
            for p in sorted(glob.glob(f"{TLDR}/pages.{lang}/{sub}/*.md")):
                name = os.path.basename(p)[:-3]
                t = open(p, encoding="utf-8", errors="ignore").read()
                t = re.sub(r"\{\{|\}\}", "", t)
                yield f"tldr:{lang}/{name}", name, chunk(t, 700, 3)


def render_man(path):
    try:
        r = subprocess.run(["man", "-P", "cat", "-l", path], capture_output=True, text=True, timeout=20,
                           env={**os.environ, "MANWIDTH": "100", "LC_ALL": "C.UTF-8"})
    except subprocess.TimeoutExpired:
        return ""
    t = re.sub(r".\x08", "", r.stdout)                       # strip overstrike bold/underline
    t = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", t))
    return t


def man_docs():
    seen, paths = set(), []
    for d in MAN_DIRS:                                       # section 1 first: it wins name clashes
        for p in sorted(glob.glob(d + "/*")):
            name = os.path.basename(p).rsplit(".", 2)[0] if p.endswith(".gz") else os.path.basename(p).rsplit(".", 1)[0]
            if name not in seen and not name.startswith("_"):
                seen.add(name)
                paths.append((name, p))
    with ThreadPoolExecutor(3) as ex:             # modest: man rendering is the only CPU-heavy part
        for (name, p), t in zip(paths, ex.map(lambda x: render_man(x[1]), paths)):
            cs = chunk(t, CHUNK, MAX_MAN_CHUNKS)
            if cs:
                yield f"man:{name}", name, cs


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "head"):
            self.skip += 1
        if tag in ("p", "li", "pre", "h1", "h2", "h3", "h4", "tr", "br", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "head") and self.skip:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip:
            self.parts.append(d)


def wiki_docs():
    for p in sorted(glob.glob(WIKI + "/*.html")):
        raw = open(p, encoding="utf-8", errors="ignore").read()
        t = _Text()
        t.feed(raw)
        txt = re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", "".join(t.parts)))
        m = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = re.sub(r"\s*[-–]\s*ArchWiki\s*$", "", html.unescape(m.group(1)).strip()) if m else os.path.basename(p)[:-5].replace("_", " ")
        cs = chunk(txt, 700, MAX_WIKI_CHUNKS)
        if cs:
            yield f"archwiki:{os.path.basename(p)[:-5]}", title, cs


def build(device="npu", progress=print):
    import search
    model = search.make_model(device)
    bs = 256 if device == "gpu" else 32
    srcs, titles, texts = [], [], []
    for gen in (tldr_docs, man_docs, wiki_docs):
        n0 = len(texts)
        for src, title, cs in gen():
            for c in cs:
                srcs.append(src); titles.append(title); texts.append(c)
        progress(f"{gen.__name__}: {len(texts) - n0} chunks")
    t0, embs = time.perf_counter(), []
    order = np.argsort([len(t) for t in texts])             # similar lengths together = less padding
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        embs.append(model.encode([f"{titles[j]}\n{texts[j]}" for j in idx]))
        if (i // bs) % 20 == 0:
            progress(f"  embedded {i + len(idx)}/{len(texts)}  ({(i + len(idx)) / (time.perf_counter() - t0):.0f} chunks/s)")
    E = np.zeros((len(texts), embs[0].shape[1]), np.float16)
    E[order] = np.concatenate(embs).astype(np.float16)
    os.makedirs(os.path.dirname(IDX), exist_ok=True)
    pickle.dump({"src": srcs, "title": titles, "text": texts, "emb": E, "built": time.time()}, open(IDX + ".tmp", "wb"))
    os.replace(IDX + ".tmp", IDX)
    progress(f"done: {len(texts)} chunks in {time.perf_counter() - t0:.0f}s")


def load():
    return pickle.load(open(IDX, "rb")) if os.path.exists(IDX) else None


STOP = {"como", "para", "que", "los", "las", "una", "con", "del", "por", "the", "and", "for", "how", "can", "install", "instalar"}


def retrieve(model, query, k=4, idx=None):
    """Best chunk per source for `query`: [{score, src, title, text}]."""
    idx = idx or load()
    if not idx:
        return []
    q = model.encode([query])[0].astype(np.float32)
    sc = idx["emb"].astype(np.float32) @ q
    qw = {w for w in re.findall(r"[a-z0-9][a-z0-9+._-]{2,}", query.lower()) if w not in STOP}
    if qw:                                                    # a command name in the query should find its own page
        title_hit = np.array([t.lower() in qw for t in idx["title"]], dtype=np.float32)
        sc = sc + 0.30 * title_hit
    sc = sc + 0.04 * np.array([x.startswith("tldr:") for x in idx["src"]], dtype=np.float32)   # short, practical pages first
    res, seen = [], set()
    for i in np.argsort(-sc):
        s = idx["src"][i]
        if s in seen:
            continue
        seen.add(s)
        res.append({"score": float(sc[i]), "src": s, "title": idx["title"][i], "text": idx["text"][i]})
        if len(res) >= k:
            break
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    a_ = sp.add_parser("ask"); a_.add_argument("query"); a_.add_argument("-k", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.device)
    else:
        import search
        for r in retrieve(search.make_model(a.device), a.query, a.k):
            print(f"\n{r['score']:.3f}  [{r['src']}]\n{r['text'][:400]}")
