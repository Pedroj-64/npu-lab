"""Hybrid retrieval over the local docs index: BM25 (ES+EN tokenizer) fused with the cosine score of `docs.retrieve`.
Candidate for the retrieval flag, OFF by default: nothing imports this module yet, `docs.retrieve` is untouched.
  bm = build(idx, stem=True)            # once per index, scipy sparse, ~seconds
  hits = retrieve(model, query, idx=idx, bm=bm, fusion="wsum", w=0.3, gate_bm25=14)"""
import re, unicodedata
from collections import Counter
import numpy as np
import scipy.sparse as sp
import docs

STOP = set("""a al algo ante bien cada como con cual cuando de del desde donde el ella en entre era es esta este esto estoy hay la las le lo los me mi mis muy no nos o para pero por que se si sin sobre su sus te tu un una uno y ya
about after all also an and any are as at be been but by can could did do does for from get has have how i if in into is it its just like make may me more most my of on one or other our out so some such than that the their them then there these they this to up us use was we what when where which who will with would you your""".split())


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def tokens(text, stem=False):
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9+._-]*", fold(text)):
        t = t.strip("._-")
        if len(t) < 2 or t in STOP:
            continue
        if stem and len(t) > 4:
            t = t[:-2] if t.endswith("es") else t[:-1] if t.endswith("s") and not t.endswith("ss") else t
        out.append(t)
    return out


def build(idx, stem=False, k1=1.2, b=0.75):
    """BM25 weights as a CSC matrix (chunks x vocabulary); the title counts double (it is also embedded)."""
    vocab, rows, cols, tf = {}, [], [], []
    lens = np.zeros(len(idx["text"]), np.float32)
    for i, (ti, tx) in enumerate(zip(idx["title"], idx["text"])):
        toks = tokens(ti, stem) * 2 + tokens(tx, stem)
        lens[i] = len(toks)
        for w, c in Counter(toks).items():
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); tf.append(c)
    rows, cols, tf = np.array(rows), np.array(cols), np.array(tf, np.float32)
    n, df = len(lens), np.bincount(cols, minlength=len(vocab))
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[rows] / lens.mean())) * idf[cols]
    return dict(M=sp.csc_matrix((w, (rows, cols)), shape=(n, len(vocab))), vocab=vocab, stem=stem)


def bm25(bm, query):
    ids = sorted({bm["vocab"][t] for t in tokens(query, bm["stem"]) if t in bm["vocab"]})
    return np.asarray(bm["M"][:, ids].sum(axis=1)).ravel() if ids else np.zeros(bm["M"].shape[0], np.float32)


def rank(x):
    r = np.empty(len(x), np.float32); r[np.argsort(-x)] = np.arange(len(x)); return r


def retrieve(model, query, k=4, idx=None, bm=None, fusion="wsum", w=0.3, gate_bm25=None, gate=True):
    """Same output as `docs.retrieve`. fusion "wsum": base score + w * bm25/max(bm25); "rrf": 1/(60+rank) of base and bm25.
    The gate also lets a chunk through when its raw BM25 >= gate_bm25 (None = off)."""
    idx = idx or docs.load()
    q = model.encode([query])[0].astype(np.float32)
    cos = idx["emb"].astype(np.float32) @ q
    qw = {x for x in re.findall(r"[a-z0-9][a-z0-9+._-]{2,}", query.lower()) if x not in docs.STOP}
    title_hit = np.array([t.lower() in qw for t in idx["title"]], dtype=np.float32) if qw else np.zeros(len(cos), np.float32)
    base = cos + 0.30 * title_hit + 0.04 * np.array([x.startswith("tldr:") for x in idx["src"]], dtype=np.float32)
    b = bm25(bm, query)
    if fusion == "rrf":
        sc = 1 / (60 + rank(base)) + 1 / (60 + rank(b)) * (b.max() > 0)
    else:
        sc = base + w * b / max(float(b.max()), 1e-6)
    res, seen = [], set()
    for i in np.argsort(-sc):
        s = idx["src"][i]
        if s in seen:
            continue
        seen.add(s)
        if gate and not (title_hit[i] or cos[i] >= docs.MIN_COS or (gate_bm25 is not None and b[i] >= gate_bm25)):
            continue
        res.append({"score": float(sc[i]), "cos": float(cos[i]), "bm25": float(b[i]), "named": bool(title_hit[i]), "src": s, "title": idx["title"][i], "text": idx["text"][i]})
        if len(res) >= k:
            break
    return res
