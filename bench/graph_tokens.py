"""Token cost of a graphify answer vs the source files, on copies of real projects (estimates: 4 characters per token).
usage: .venv/bin/python bench/graph_tokens.py DIR   (DIR/<project>/ copies, each built with `graphify update .`; edit Q for your projects)
Spanish: coste en tokens de una respuesta del mapa frente a los archivos fuente."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import graph
M = sys.argv[1] if len(sys.argv) > 1 else "."      # folder holding copies of the projects, each already built with `graphify update .`
graph.HOME = "/tmp"                       # the home rule is for the server; this is a one-off measurement on copies
Q = {"inklus": ["how are notebooks saved and synced with Google Drive", "where is the text box formatting handled", "how does the app open a .inklus file"],
     "Gradify_App": ["how are grades calculated for a subject", "how do the widgets get updated", "where is the backup restored"],
     "npu-lab": ["how are chats saved", "what happens when a model is loaded", "how does the search index get built"]}
tot = []
for p, qs in Q.items():
    root = f"{M}/{p}"; st = graph.status(root)
    print(f"\n## {p}: {st['files']} files, {st['nodes']} nodes, {st['edges']} edges; all parsed source ~{st['source_tokens']:,} tokens")
    for q in qs:
        for b in (800, 1500, 3000):
            r = graph.query(root, q, b)
            files = {m for m in re.findall(r"src=([^\s\]]+)", r["text"]) if m}
            fsize = sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f)))
            ft = max(1, round(fsize / 4))
            print(f"  b={b:<5} answer ~{r['tokens']:>5,} tok | cited files {len(files):>2} ~{ft:>7,} tok | whole source ~{r['source_tokens']:>8,} | vs whole -{r['saved_pct']}% | vs cited files {'-' if r['tokens']<ft else '+'}{abs(round(100*(1-r['tokens']/ft)))}%  :: {q[:44]}")
            if b == 1500: tot.append((p, r["tokens"], ft, r["source_tokens"]))
import statistics as s
for p in Q:
    x = [t for t in tot if t[0] == p]
    print(f"\n{p}: median answer(b=1500) {int(s.median(t[1] for t in x)):,} tok; median cited-files {int(s.median(t[2] for t in x)):,}; whole {x[0][3]:,}")
