"""Injection gate for the local docs retrieval: should ANY excerpt be injected for this question? Candidate, OFF by default: nothing imports it, `docs.retrieve` is untouched.
A logistic model over a few features of the retrieval itself; abstention = return no excerpts when the probability is below the calibrated threshold.
The weights and the threshold come from bench/gate_eval.py (fit on the tuning split only)."""
import re
import numpy as np
import docs, hybrid

CUES = set(hybrid.fold("""terminal comando comandos command commands shell bash fish zsh linux arch pacman systemd systemctl paquete paquetes package packages instalar servicio service
archivo archivos carpeta directorio folder directory proceso process puerto port permisos permissions disco disk particion partition kernel sudo root usuario user red network wifi ssh
flag opcion option linea script cron journal logs log mount montar usb""").split())
NAMES = ["cos_max", "cos_gap", "title_cmd_cue", "title_cmd_nocue", "log_bm25", "wiki_top", "n_content", "cue_count"]


_N = {}


def _names(idx):
    """set of lower-cased titles of man/tldr pages (cached per index)."""
    if id(idx) not in _N:
        _N[id(idx)] = {t.lower() for t, s in zip(idx["title"], idx["src"]) if not s.startswith("archwiki")}
    return _N[id(idx)]


def features(idx, bm, q_emb, query):
    cos = idx["emb"].astype(np.float32) @ q_emb.astype(np.float32)
    top = np.argsort(-cos)[:200]
    seen, best = set(), []
    for i in top:
        if idx["src"][i] not in seen:
            seen.add(idx["src"][i]); best.append(i)
        if len(best) == 5:
            break
    toks = hybrid.tokens(query, True)
    cue = sum(t in CUES or t.rstrip("s") in CUES for t in toks)
    qw = {w for w in re.findall(r"[a-z0-9][a-z0-9+._-]{2,}", query.lower()) if w not in docs.STOP}
    cmd = False
    if qw:                                                    # a man/tldr page whose name is a word of the question
        t = _names(idx)
        cmd = any(w in t for w in qw)
    b = hybrid.bm25(bm, query)
    return np.array([cos[best[0]], cos[best[0]] - cos[best[-1]], float(cmd and cue > 0), float(cmd and cue == 0), np.log1p(b.max()),
                     float(idx["src"][best[0]].startswith("archwiki")), len(toks), cue], dtype=np.float64)


def prob(model, f):
    z = (f - np.array(model["mu"])) / np.array(model["sd"])
    return 1 / (1 + np.exp(-(z @ np.array(model["w"]) + model["b"])))
