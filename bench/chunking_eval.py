"""Does heading-aware chunking (docs.chunk_headed) retrieve better than plain chunking (docs.chunk) on this repo's own Markdown documents?
Each question has a phrase that sits in the section that answers it; a hit = a retrieved chunk contains that phrase. CPU only.
usage: .venv/bin/python bench/chunking_eval.py          (small bank, n = len(Q): read the numbers as indicative, not as proof)
Spanish: ¿el chunking por encabezados recupera mejor que el simple? Banco pequeño sobre los documentos del propio repo."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, docs, search
FILES = ["README.md", "docs/PLAN.md", "docs/ARCHITECTURE.md", "docs/WORKING_RULES.md", "docs/RESEARCH_LOG.md", "docs/DECISIONS.md"]
Q = [("how do I use the local models from VS Code Copilot", "customOAIModels"), ("what happens to files the agent deletes", "recoverable trash"),
     ("which web search providers need no API key", "DuckDuckGo and Wikipedia"), ("is Windows supported", "Not supported yet"),
     ("where are cloud API keys stored", "providers.json"), ("when is the Tauri desktop shell planned", "Tauri shell"),
     ("how is the idle power of the sentinel checked", "idle power with sentinel"), ("what is the criterion to accept Whisper on the NPU", "3× real time"),
     ("which endpoints does the server expose", "/documents/upload"), ("is the server reachable from the network", "loopback only"),
     ("can two programs use the NPU at the same time", "one workload at a time"), ("who may make commits and pushes", "under their own identity"),
     ("which model uses per-layer embeddings", "per-layer embeddings"), ("what do I need for OCR of scanned PDFs", "tesseract"),
     ("why was Qwen3.5 removed", "needs the GPU for every token")]
model = search.make_model("cpu")
texts = {f: open(os.path.join(ROOT, f), encoding="utf-8").read() for f in FILES}
for name, fn in (("plain chunk()", docs.chunk), ("chunk_headed()", docs.chunk_headed)):
    items = [(f, c) for f, t in texts.items() for c in fn(t, 600)]
    E = np.concatenate([model.encode([f"{os.path.basename(f)}\n{c}" for f, c in items[i:i + 32]]) for i in range(0, len(items), 32)]).astype(np.float32)
    h1 = h3 = 0
    for q, phrase in Q:
        s = E @ model.encode([q])[0].astype(np.float32)
        top = np.argsort(-s)[:3]
        hit = [phrase.lower() in items[j][1].lower() for j in top]
        h1 += hit[0]; h3 += any(hit)
    print(f"{name:15} chunks {len(items):4}  hit@1 {h1}/{len(Q)}  hit@3 {h3}/{len(Q)}")
