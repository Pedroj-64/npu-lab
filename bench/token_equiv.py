"""Same tokens before/after a change? Greedy decoding (the engine's own stream(): repetition penalty included, no sampling) on 5 fixed prompts,
one of them with a ~600-token document so the attention over a longer cache is exercised.
usage: .venv/bin/python bench/token_equiv.py save|check [--model models/qwen3s] [--tokens 40]   (file: bench/.equiv_<model>.json)"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["NPU_GPU"] = "0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import qwen_npu as Q

mode = sys.argv[1]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "models/qwen3s"
n = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 40
doc = " ".join(f"Section {i}: the NPU streams int8 weights while the GPU keeps the cache of key and value vectors for attention." for i in range(30))
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Escribe una función en Python que invierta una cadena y explica cómo funciona.",
           "Lista tres comandos de pacman para instalar, quitar y buscar paquetes.",
           "Explain in two sentences why the sky is blue.",
           f"Read this and summarise it in one sentence: {doc}"]
m = Q.Qwen(model, gpu=False)
out = []
for p in PROMPTS:
    text = "".join(m.stream(p, n))
    out.append({"prompt": p[:60], "text": text, "tokens": m.tok.encode(text, add_special_tokens=False).ids})
f = f"bench/.equiv_{os.path.basename(model)}.json"
if mode == "save":
    json.dump(out, open(f, "w"), ensure_ascii=False, indent=1); print("saved", f)
else:
    ref, bad = json.load(open(f)), 0
    for a, b in zip(ref, out):
        same = a["text"] == b["text"]; bad += not same
        print(("SAME " if same else "DIFF ") + a["prompt"])
        if not same:
            k = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), min(len(a["tokens"]), len(b["tokens"])))
            print(f"   first differing token: {k} of {len(a['tokens'])}\n   before: {a['text'][:160]!r}\n   after : {b['text'][:160]!r}")
    sys.exit(1 if bad else 0)
