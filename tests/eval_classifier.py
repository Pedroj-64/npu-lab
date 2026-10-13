"""Accuracy of the local-model router vs the rule layer on hand-labelled questions (needs the NPU + a model).
usage: .venv/bin/python tests/eval_classifier.py [models/qwen15]"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import router, qwen_npu

CASES = [
    ("how do I make a class in TypeScript", "NONE"), ("explain recursion", "NONE"), ("what is the capital of France", "NONE"),
    ("write a haiku about the sea", "NONE"), ("diferencia entre TCP y UDP", "NONE"), ("cómo funciona un hash table", "NONE"),
    ("how do I install an AUR package", "DOCS"), ("como listo los servicios activos con systemctl", "DOCS"), ("what does the -r flag of cp do", "DOCS"),
    ("cómo cambio los permisos de un archivo", "DOCS"), ("how to mount a usb drive from the terminal", "DOCS"), ("update all packages on arch", "DOCS"),
    ("who won the match yesterday", "WEB"), ("latest stable version of the linux kernel", "WEB"), ("precio del dólar hoy", "WEB"),
    ("what is the weather in Madrid", "WEB"), ("noticias de tecnología", "WEB"), ("what happened last week in the stock market", "WEB"),
    ("what did we talk about", "HISTORY"), ("repite lo último que dijiste", "HISTORY"), ("summarise the above", "HISTORY"),
    ("y en Python?", "HISTORY"), ("what was my first question", "HISTORY"), ("translate that to English", "HISTORY"),
]
q = qwen_npu.Qwen(sys.argv[1] if len(sys.argv) > 1 else "models/qwen3")


def of(d):
    return "HISTORY" if d["history"] else "WEB" if d["web"] else "DOCS" if d["docs"] else "NONE"


rules = model = hybrid = 0
t0 = time.time()
for text, want in CASES:
    r = of(router.decide(text))
    sure = router.decide(text)["sure"]
    got = raw = q.classify(text, router.CLASSIFY_SYSTEM, list(router.LABELS))
    rules += r == want; model += got == want; hybrid += (r if sure else got) == want
    print(f"{'ok ' if got == want else 'BAD'} want {want:7} model {got:7} ({raw.strip()!r}) rules {r:7} | {text}")
n = len(CASES)
print(f"\nrules {rules}/{n} = {100*rules/n:.0f}%   model alone {model}/{n} = {100*model/n:.0f}%   rules, then model where no rule is sure {hybrid}/{n} = {100*hybrid/n:.0f}%   ({(time.time()-t0)/n:.1f} s per question)")
