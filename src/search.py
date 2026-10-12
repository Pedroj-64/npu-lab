"""Local semantic file search; embeddings run on the NPU.

  python src/search.py index [--dry]   (global: --device npu|gpu|cpu, default npu)     (re)index the folders in the rules file
  python src/search.py search "query" [-n 8]     search by meaning

Rules (what may be read) live in ~/.config/npu-lab/rules.json, created on first use.
The index lives in ~/.local/share/npu-lab/ and never leaves the machine or the repo.
"""
import argparse, fnmatch, json, os, pickle, re, sys, time, unicodedata
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from minilm import MiniLM

MODEL = "models/multi"
CFG = os.path.expanduser("~/.config/npu-lab/rules.json")
IDX = os.path.expanduser("~/.local/share/npu-lab/index.pkl")
DEFAULT_RULES = {
    "roots": ["~/Documentos"],
    "exts": [".md", ".txt", ".rst", ".org", ".tex", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h",
             ".cpp", ".hpp", ".cs", ".go", ".rs", ".kt", ".php", ".sql", ".sh", ".fish", ".html", ".css",
             ".json", ".yml", ".yaml", ".toml", ".xml", ".csv"],
    "deny_dirs": ["node_modules", ".git", ".venv", "venv", "env", "__pycache__", "build", "dist", "target",
                  "vendor", "site-packages", "claves", ".ssh", ".gnupg", "secrets"],
    "deny_names": [".env*", "*secret*", "*password*", "*passwd*", "*token*", "*credential*", "*claves*",
                   "*.pem", "*.key", "id_rsa*", "*.kdbx", "*.p12", "*.pfx", "package-lock.json",
                   "yarn.lock", "*.min.js", "*.min.css", "*.lock"],
    "max_bytes": 300000,
}
SECRET = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY|(?i:password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S{8,}")


def rules():
    if not os.path.exists(CFG):
        os.makedirs(os.path.dirname(CFG), exist_ok=True)
        json.dump(DEFAULT_RULES, open(CFG, "w"), indent=2, ensure_ascii=False)
    return json.load(open(CFG))


def files(r):
    exts, dd, dn = set(r["exts"]), set(r["deny_dirs"]), r["deny_names"]
    for root in r["roots"]:
        for d, subs, names in os.walk(os.path.expanduser(root)):
            subs[:] = [s for s in subs if s not in dd and not s.startswith(".")]
            for n in names:
                low = n.lower()
                if os.path.splitext(low)[1] not in exts or any(fnmatch.fnmatch(low, p) for p in dn):
                    continue
                p = os.path.join(d, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if 0 < st.st_size <= r["max_bytes"]:
                    yield p, st.st_mtime


def chunks(path, size=500, overlap=80, max_chunks=6):
    try:
        t = open(path, "rb").read()
        if b"\0" in t:
            return []
        t = t.decode("utf-8", "ignore")
    except OSError:
        return []
    out, i = [], 0
    while i < len(t):
        c = t[i:i + size].strip()
        if len(c) > 40 and not SECRET.search(c):
            out.append(c)
        i += size - overlap
        if len(out) >= max_chunks:
            break
    return out


def load():
    return pickle.load(open(IDX, "rb")) if os.path.exists(IDX) else {}


def make_model(device):
    """device: npu (default, low power) | gpu (fast bulk, VRAM-capped, needs torch+CUDA) | cpu."""
    if device == "gpu":
        from gpu_embed import GpuEmbedder
        return GpuEmbedder(MODEL, max_len=128, vram_gb=rules().get("gpu_vram_gb", 1.5))
    if device == "cpu":
        return MiniLM(d=MODEL, max_len=128, fast_gelu=True)
    from npu_linear import NpuLinear
    return MiniLM(linear=NpuLinear(), d=MODEL, max_len=128, fast_gelu=True)


def run_index(device="npu", dry=False, progress=None, stop=None, model=None):
    """(Re)index the folders allowed by the rules. progress(dict) is called every ~100 files;
    stop() -> True aborts (what is done so far is saved). Returns a summary dict."""
    r = rules()
    found = dict(files(r))
    idx = load()
    for p in list(idx):
        if p not in found:
            del idx[p]
    todo = [p for p, m in found.items() if p not in idx or idx[p]["mtime"] != m]
    out = {"matching": len(found), "todo": len(todo), "chunks_new": 0, "seconds": 0.0, "chunks_total": 0, "aborted": False}
    if dry:
        return out
    model = model or make_model(device)
    bs = 256 if device == "gpu" else 32
    t0, n = time.perf_counter(), 0
    batch_txt, batch_ref = [], []

    def save():
        os.makedirs(os.path.dirname(IDX), exist_ok=True)
        tmp = IDX + ".tmp"
        pickle.dump(idx, open(tmp, "wb"))
        os.replace(tmp, IDX)

    def flush():
        nonlocal n
        if not batch_txt:
            return
        order = np.argsort([len(x) for x in batch_txt])
        emb = model.encode([batch_txt[i] for i in order])
        for rank, i in enumerate(order):
            p, k = batch_ref[i]
            idx[p]["emb"][k] = emb[rank]
        n += len(batch_txt)
        batch_txt.clear(); batch_ref.clear()

    for j, p in enumerate(todo):
        if stop and stop():
            out["aborted"] = True
            break
        cs = chunks(p)
        idx[p] = {"mtime": found[p], "snip": [c[:300] for c in cs],
                  "emb": np.zeros((len(cs), 384), np.float32)}
        for k, c in enumerate(cs):
            batch_txt.append("/".join(p.split("/")[-3:]) + "\n" + c)
            batch_ref.append((p, k))
            if len(batch_txt) >= bs:
                flush()
        if j % 500 == 499:
            flush(); save()
        if progress and j % 100 == 99:
            progress({"files_done": j + 1, "files_total": len(todo), "chunks": n, "rate": n / (time.perf_counter() - t0)})
    flush()
    if out["aborted"]:  # drop entries that were registered but not embedded
        for p in todo:
            if p in idx and not idx[p]["emb"].any() and len(idx[p]["snip"]):
                del idx[p]
    save()
    out.update(chunks_new=n, seconds=time.perf_counter() - t0, chunks_total=sum(len(v["snip"]) for v in idx.values()))
    return out


def cmd_index(a):
    def show(i):
        print(f"  {i['files_done']}/{i['files_total']} files, {i['chunks']} chunks, {i['rate']:.0f} chunks/s", flush=True)
    r = run_index(a.device, a.dry, show if not a.dry else None)
    if a.dry:
        print(f"{r['matching']} files match the rules, {r['todo']} new/changed")
    else:
        print(f"done: {r['chunks_new']} chunks in {r['seconds']:.0f}s; index has {r['chunks_total']} chunks")


def words(t):
    """lowercase, accent-free word set; splits CamelCase and path separators."""
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = unicodedata.normalize("NFD", t.lower())
    return {w for w in re.findall(r"[a-z0-9]{3,}", "".join(c for c in t if unicodedata.category(c) != "Mn"))
            if w not in STOP}


STOP = {"los", "las", "del", "con", "para", "por", "que", "una", "uno", "the", "and", "for", "de", "la", "el"}


def run_search(model, query, n=8, idx=None):
    """Top-n files for `query`: [{score, path, snippet}]. `model` is anything with .encode([text])."""
    idx = idx if idx is not None else load()
    idx = {p: v for p, v in idx.items() if len(v["snip"])}
    if not idx:
        return []
    q = model.encode([query])[0]
    paths = [p for p, v in idx.items() for _ in v["snip"]]
    snips = [s for v in idx.values() for s in v["snip"]]
    sc = np.concatenate([v["emb"] for v in idx.values()]) @ q
    qw = words(query)
    if qw:  # lexical boost: share of query words that appear in the file path
        pw = {p: len(qw & words(p.replace(os.path.expanduser("~"), ""))) / len(qw) for p in set(paths)}
        sc = sc + 0.25 * np.array([pw[p] for p in paths])
    res, seen = [], set()
    for i in np.argsort(-sc):
        if paths[i] in seen:
            continue
        seen.add(paths[i])
        res.append({"score": float(sc[i]), "path": paths[i], "snippet": " ".join(snips[i][:200].split())})
        if len(res) >= n:
            break
    return res


def cmd_search(a):
    res = run_search(make_model(a.device), a.query, a.n)
    if not res:
        sys.exit("empty index: run `index` first")
    for r in res:
        print(f"\n{r['score']:.3f}  {r['path']}\n    {r['snippet']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("index"); i.add_argument("--dry", action="store_true"); i.set_defaults(f=cmd_index)
    b = sp.add_parser("search"); b.add_argument("query"); b.add_argument("-n", type=int, default=8); b.set_defaults(f=cmd_search)
    a = ap.parse_args()
    a.f(a)
