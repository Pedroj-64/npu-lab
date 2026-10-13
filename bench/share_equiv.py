"""Does the GPU share change the generated tokens? Greedy decoding (engine stream(), repetition penalty included) on the 5 fixed prompts of bench/token_equiv.py, first N tokens.
usage: .venv/bin/python bench/share_equiv.py run SHARE [--model models/qwen35_2b] [--tokens 32]    -> bench/.share_equiv_<model>_<share>.json
       .venv/bin/python bench/share_equiv.py cmp BASE SHARE...                                        (token agreement against the BASE share)   (NPU + GPU: run alone)"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
model, n = arg("--model", "models/qwen35_2b"), arg("--tokens", 32)
f = lambda s: f"bench/.share_equiv_{os.path.basename(model)}_{s}.json"
doc = " ".join(f"Section {i}: the NPU streams int8 weights while the GPU keeps the cache of key and value vectors for attention." for i in range(30))
PROMPTS = ["What is the capital of France? Answer in one sentence.", "Escribe una función en Python que invierta una cadena y explica cómo funciona.",
           "Lista tres comandos de pacman para instalar, quitar y buscar paquetes.", "Explain in two sentences why the sky is blue.", f"Read this and summarise it in one sentence: {doc}"]
if sys.argv[1] == "run":
    share = sys.argv[2]
    import qwen35_npu as Q35
    m = Q35.load_engine(model, gpu=True, gpu_share=float(share), vram_gb=4.5, max_ctx=4096)
    out = []
    for p in PROMPTS:
        t = "".join(m.stream(p, n)); out.append(dict(text=t, tokens=m.tok.encode(t, add_special_tokens=False).ids[:n]))
    json.dump(out, open(f(share), "w"))
else:
    base = json.load(open(f(sys.argv[2])))
    for s in sys.argv[3:]:
        o = json.load(open(f(s))); eq = tot = 0; first = []
        for a, b in zip(base, o):
            k = min(len(a["tokens"]), len(b["tokens"])); eq += sum(x == y for x, y in zip(a["tokens"], b["tokens"])); tot += max(len(a["tokens"]), len(b["tokens"]))
            first.append(next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None))
        print(f"share {s} vs {sys.argv[2]}: {eq}/{tot} tokens equal = {eq / tot:.3f}; first different token per prompt: {first}; identical text {sum(a['text'] == b['text'] for a, b in zip(base, o))}/5")
