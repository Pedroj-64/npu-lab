"""Is the graphify answer the RIGHT one? Ground truth comes from the code itself: for sampled identifiers (unique names defined once), the files that really
mention them (word search). Question: "where is NAME defined and what uses it?". Metrics per project: the defining file appears in the answer; share of the
true user files that appear (recall); share of the files named in the answer that really mention NAME (precision).
usage: .venv/bin/python bench/graph_relevance.py DIR [--n 12] [--budget 1500]     (DIR/<project>/ copies already built with `graphify update .`)
Spanish: relevancia del mapa: la verdad sale del propio código (dónde se define y qué archivos mencionan cada nombre)."""
import argparse, json, os, random, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import graph
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--budget", type=int, default=1500); a = ap.parse_args()
graph.HOME = "/tmp"                                    # one-off measurement on copies outside the home
CODE = (".py", ".dart", ".kt", ".java", ".js", ".ts", ".rs", ".c", ".cc", ".cpp", ".h", ".swift", ".go")
tot = {"def": 0, "rec": 0, "prec": 0, "n": 0}
for proj in sorted(d for d in os.listdir(a.dir) if os.path.isdir(os.path.join(a.dir, d, "graphify-out"))):
    root = os.path.join(a.dir, proj); g = json.load(open(graph.graph_file(root)))
    files = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "graphify-out"]
        for f in fn:
            if f.endswith(CODE):
                p = os.path.relpath(os.path.join(dp, f), root); files[p] = open(os.path.join(dp, f), errors="ignore").read()
    labels = {}
    for n in g["nodes"]:
        nm = re.sub(r"\(\)$", "", n["label"]).lstrip(".")
        if n.get("source_file") and n.get("_callable") is not False and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{6,}", nm):
            labels.setdefault(nm, []).append(n["source_file"])
    cands = []
    for nm, srcs in labels.items():
        if len(srcs) != 1: continue
        users = [p for p, t in files.items() if p != srcs[0] and re.search(rf"\b{re.escape(nm)}\b", t)]
        if 1 <= len(users) <= 12: cands.append((nm, srcs[0], users))
    random.Random(0).shuffle(cands); sel = cands[:a.n]
    r_def = r_rec = r_prec = 0
    for nm, d, users in sel:
        text = graph.query(root, f"where is {nm} defined and what uses it", a.budget)["text"]
        named = {m for m in re.findall(r"src=([^\s\]]+)", text) if m} | set(re.findall(r"[\w./-]+\.(?:py|dart|kt|java|js|ts|rs|cc|cpp|c|h|swift|go)", text))
        r_def += d in named
        r_rec += len(set(users) & named) / len(users)
        real = {p for p, t in files.items() if re.search(rf"\b{re.escape(nm)}\b", t)}
        r_prec += (len(named & real) / len(named)) if named else 0
    n = len(sel); print(f"{proj:12} n={n:2}  defining file in answer {r_def}/{n} = {r_def/max(n,1):.2f}   user-file recall {r_rec/max(n,1):.2f}   precision of named files {r_prec/max(n,1):.2f}")
    for k, v in (("def", r_def), ("rec", r_rec), ("prec", r_prec), ("n", n)): tot[k] += v
print(f"{'ALL':12} n={tot['n']}  defining file {tot['def']/tot['n']:.2f}   recall {tot['rec']/tot['n']:.2f}   precision {tot['prec']/tot['n']:.2f}")
