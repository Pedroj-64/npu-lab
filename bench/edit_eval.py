"""Does a repo map help a small model edit code? Five fixed edit tasks on a small project (plus --filler N unrelated modules to make it big),
each asked twice: with ALL files pasted in full ("full") and with the repo map + only the target file ("map"). The new file is applied and
checked by running code, so the score is "the edit works", not "the answer looks right".
usage: python bench/edit_eval.py --check                 verify the tasks themselves with the reference solutions (no model, ~1 s)
       python bench/edit_eval.py --model qwen35m|qwen35l [--filler 20] [--tasks 1,2,3,4,5] [--tokens 450] [--deadline 900]   (uses the NPU and GPU: run alone)"""
import http.client, json, os, re, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src"))
import repomap

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
BASE = {
    "models.py": 'from dataclasses import dataclass\n\n\n@dataclass\nclass Item:\n    name: str\n    qty: int\n    price: float\n',
    "util.py": 'def slug(text):\n    """Lowercase text with spaces turned into dashes."""\n    return text.lower().replace(" ", "-")\n\n\ndef clamp(x, lo, hi):\n    """Limit x to the range lo..hi."""\n    return max(lo, min(hi, x))\n',
    "pricing.py": 'def price_with_tax(price, rate=0.21):\n    """Price including tax."""\n    return round(price * (1 + rate), 2)\n\n\ndef apply_discount(price, pct):\n    """Price after a percentage discount."""\n    return round(price * (1 - pct / 100), 2)\n',
    "store.py": 'from models import Item\n\n\nclass Store:\n    """Keeps items by name."""\n\n    def __init__(self):\n        self._items = {}\n\n    def add(self, name, qty, price):\n        """Add an item, replacing one with the same name."""\n        self._items[name] = Item(name, qty, price)\n\n    def get(self, name):\n        """The item called name, or None."""\n        return self._items.get(name)\n\n    def items(self):\n        """All items as a list."""\n        return list(self._items.values())\n',
    "report.py": 'def format_line(name, qty, price):\n    """One text line for an item."""\n    return f"{name}: {qty} x {price:.2f}"\n',
    "cli.py": 'import sys\nfrom store import Store\nfrom report import format_line\n\n\ndef load(store):\n    """Fill the store with demo data."""\n    store.add("Pen", 10, 1.5)\n    store.add("book", 2, 12.0)\n    store.add("Ink", 1, 7.25)\n\n\ndef main(argv):\n    """Print every item, one per line."""\n    store = Store()\n    load(store)\n    for it in store.items():\n        print(format_line(it.name, it.qty, it.price))\n\n\nif __name__ == "__main__":\n    main(sys.argv[1:])\n',
}
# (target file, instruction, check code run in the project folder, reference new content of the target file)
TASKS = {
    1: ("report.py", "Add a function total_line(items) that takes a list of Item objects and returns the text 'TOTAL: ' followed by the sum of qty * price with two decimals.",
        "from models import Item\nfrom report import total_line\nassert total_line([Item('a', 2, 1.5), Item('b', 1, 0.25)]) == 'TOTAL: 3.25'",
        BASE["report.py"] + '\n\ndef total_line(items):\n    """Text line with the total of all items."""\n    return f"TOTAL: {sum(i.qty * i.price for i in items):.2f}"\n'),
    2: ("store.py", "Add a method names_sorted() to Store that returns the item names sorted alphabetically, ignoring upper and lower case.",
        "from store import Store\ns = Store(); s.add('Pen', 1, 1.0); s.add('book', 1, 1.0); s.add('Ink', 1, 1.0)\nassert s.names_sorted() == ['book', 'Ink', 'Pen']",
        BASE["store.py"] + '\n    def names_sorted(self):\n        """Item names, alphabetical, ignoring case."""\n        return sorted(self._items, key=str.lower)\n'),
    3: ("pricing.py", "Make apply_discount raise ValueError when pct is below 0 or above 100. Keep everything else the same.",
        "from pricing import apply_discount\nassert apply_discount(100, 10) == 90.0\nfor bad in (-1, 101):\n    try:\n        apply_discount(100, bad)\n    except ValueError:\n        continue\n    raise SystemExit('no ValueError for %r' % bad)",
        BASE["pricing.py"].replace('    """Price after a percentage discount."""\n', '    """Price after a percentage discount."""\n    if not 0 <= pct <= 100:\n        raise ValueError("pct must be between 0 and 100")\n')),
    4: ("report.py", "Change format_line so that the price it prints includes tax, using the existing function from the pricing module with its default rate. Keep printing exactly two decimals.",
        "from report import format_line\nassert format_line('a', 2, 10.0) == 'a: 2 x 12.10', format_line('a', 2, 10.0)",
        'from pricing import price_with_tax\n\n\ndef format_line(name, qty, price):\n    """One text line for an item, price with tax."""\n    return f"{name}: {qty} x {price_with_tax(price):.2f}"\n'),
    5: ("cli.py", "Make main accept an optional '--min-qty N' argument and print only the items whose qty is at least N.",
        "import io, contextlib\nimport cli\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main(['--min-qty', '2'])\nt = b.getvalue()\nassert 'Pen' in t and 'book' in t and 'Ink' not in t, t",
        BASE["cli.py"].replace('    for it in store.items():\n', '    n = int(argv[argv.index("--min-qty") + 1]) if "--min-qty" in argv else 0\n    for it in store.items():\n        if it.qty < n:\n            continue\n')),
}


def project(folder, filler):
    files = dict(BASE)
    for k in range(filler):                       # unrelated modules: the project is big, the task touches 1-2 files
        files[f"mod{k:02d}.py"] = "\n\n".join(f'def mod{k:02d}_op{j}(a, b=1):\n    """Operation {j} of module {k}: adds a and b, then scales by {k + j + 2}."""\n    return (a + b) * {k + j + 2}\n' for j in range(5))
    for n, t in files.items():
        open(os.path.join(folder, n), "w").write(t)
    return files


def ask_text(files, mode, tid, folder):
    target, instr = TASKS[tid][0], TASKS[tid][1]
    ctx = ("Project files:\n\n" + "\n\n".join(f"### {n}\n```python\n{t}```" for n, t in files.items())) if mode == "full" else \
          ("Map of the project (signatures only):\n" + repomap.build(folder) + f"\n\nFull file you will edit:\n### {target}\n```python\n{files[target]}```")
    return f"{ctx}\n\nTask: in `{target}`: {instr}\nReply with the complete new content of `{target}` in one ```python block and nothing else."


def apply_and_check(folder, tid, answer):
    m = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
    if not m:
        return False, "no code block"
    path = os.path.join(folder, TASKS[tid][0]); old = open(path).read()
    open(path, "w").write(m.group(1))
    try:
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.')\n" + TASKS[tid][2]], cwd=folder, capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:120]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        open(path, "w").write(old)


def check_tasks():
    with tempfile.TemporaryDirectory() as d:
        files = project(d, 0)
        for tid in TASKS:
            before, _ = apply_and_check(d, tid, "```python\n" + files[TASKS[tid][0]] + "```")       # the untouched file must FAIL the check
            ok, why = apply_and_check(d, tid, "```python\n" + TASKS[tid][3] + "```")                 # the reference must PASS it
            print(f"task {tid}: untouched fails={not before}  reference passes={ok} {why}")
            assert not before and ok
        big = project(d, 20)
        print(f"filler 20: {sum(len(t) for t in big.values())} chars in full, map {len(repomap.build(d))} chars")


def run():
    model, filler, tokens, deadline = arg("--model", "qwen35m"), arg("--filler", 20), arg("--tokens", 450), arg("--deadline", 900)
    tids = [int(x) for x in arg("--tasks", "1,2,3,4,5").split(",")]
    tmp = tempfile.mkdtemp(); T = "e" * 43; t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "NPU_LAB_TOKEN": T, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
    port = json.loads(p.stdout.readline())["port"]

    def call(path, body, timeout=300):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("POST", path, json.dumps(body), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"}); r = c.getresponse(); return r.status, json.loads(r.read())

    def chat(prompt):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        c.request("POST", "/chat", json.dumps({"prompt": prompt, "tokens": tokens, "history": [], "docs": "off", "web": "off", "mine": "off"}), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
        r = c.getresponse(); out, st = "", {}
        for line in r:
            m = json.loads(line); out += m.get("token", "")
            if m.get("error"): raise SystemExit("engine error: " + str(m["error"]))
            if m.get("done") is True: st = m.get("stats", {}); break          # stop at the end of the answer: no compaction wanted here
        return out.split("</think>")[-1], st

    score = {"full": 0, "map": 0}
    try:
        call("/profile", {"profile": "assisted"}); call("/rules", {"context_tokens": 8192, "llm_vram_gb": 5.0, "gpu_share": 0.0, "compact_at": 0})
        s, b = call("/mode", {"mode": "llm", "model": model}, 400)
        if s != 200: raise SystemExit(f"load failed: {b}")
        with tempfile.TemporaryDirectory() as d:
            files = project(d, filler)
            for tid in tids:
                for mode in ("full", "map"):
                    if time.time() - t0 > deadline: raise SystemExit(f"deadline of {deadline}s reached")
                    t = time.time(); ans, st = chat(ask_text(files, mode, tid, d)); ok, why = apply_and_check(d, tid, ans); score[mode] += ok
                    print(f"task {tid} {mode:4s}: {'PASS' if ok else 'FAIL'}  prompt {st.get('prefill_tokens', 0):5d} tokens  {time.time() - t:5.1f}s  {why}", flush=True)
    finally:
        try: call("/mode", {"mode": "none"}, 60)
        except Exception: pass
        p.terminate()
    print(f"{model}, filler {filler}: full {score['full']}/{len(tids)}   map {score['map']}/{len(tids)}")


if __name__ == "__main__":
    check_tasks() if "--check" in sys.argv else run()
