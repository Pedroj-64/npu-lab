"""Labelled data for the DONE guard, built from the agent task bank (bench/agent_tasks.py) with NO model: for every task the final project state of
(1) the reference solution, (2) an agent that did nothing, (3) partial solutions (a subset of the files), (4) single-line mutants of the reference
(a line dropped, an operator swapped, a number changed), (5) harmless edits (a comment, a blank line, a docstring). The label is whether the task's own
hidden check passes, run for real, so it is objective. Each state is rendered exactly as the guard sees it (src/decide.py done_state).
`objective_ok` = changed something, compiles, and the project's own tests pass (when it has some): the guard refuses the others without any model.
usage: .venv/bin/python bench/decide_data_done.py      -> bench/decide_data/done.jsonl   (deterministic: seed 0)
Splits are by TASK (no task appears in two splits); the 8 test tasks are also the ones to keep unseen in any later fine-tuning."""
import itertools, json, os, random, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import agent, agent_tasks as T, decide

SPLIT = {"train": [1, 3, 4, 6, 7, 10, 11, 14, 15, 18, 19, 21], "val": [2, 9, 13, 17, 22], "test": [5, 8, 12, 16, 20, 23, 24, 25]}
CLAIMS = ["All the changes are made.", "Task completed.", "Done, it should work now.", "Implemented as requested.", "Hecho: cambios aplicados.", "Tarea terminada.", "Listo, ya funciona.",
          "The requested change is in place."]
SWAPS = [("<=", "<"), (">=", ">"), ("<", "<="), (">", ">="), ("==", "!="), ("!=", "=="), (" + ", " - "), (" * ", " / "), (" and ", " or "), (" or ", " and "), ("True", "False"), ("False", "True"),
         (" not ", " "), (".lower()", ".upper()"), ("sorted(", "list("), ("round(", "float("), (" in ", " not in ")]


def added_lines(before, after):
    """[(file, line index)] of the lines of `after` that are not in `before` (non-blank, not a def/class/import/decorator line, so a mutant stays syntactically valid)."""
    out = []
    for n, t in after.items():
        if t is None:
            continue
        old = set((before.get(n) or "").splitlines())
        for i, l in enumerate(t.splitlines()):
            if l.strip() and l not in old and not re.match(r"\s*(def |class |import |from |@|\"\"\"|#|else|elif|try|except)", l):
                out.append((n, i))
    return out


def mutants(before, after, rng, limit):
    cands = []
    for n, i in added_lines(before, after):
        lines = after[n].splitlines()
        l = lines[i]
        if not l.rstrip().endswith(":") and not l.strip().startswith(("return", "if", "for", "while", "raise", "print", "assert")) or True:
            if not l.rstrip().endswith((":", "(", "[", "{", ",")):                       # dropping a block opener or a continuation would not compile
                cands.append((n, i, "drop line", None))
            for a, b in SWAPS:
                if a in l:
                    cands.append((n, i, f"swap {a.strip()!r}->{b.strip()!r}", (a, b)))
            for m in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", l):
                cands.append((n, i, f"number {m.group(1)}", (m.group(1), str(int(m.group(1)) + 1))))
    rng.shuffle(cands)
    out, seen = [], set()
    for n, i, how, sw in cands:
        lines = after[n].splitlines(); l = lines[i]
        if sw is None:
            lines.pop(i)
        else:
            lines[i] = l.replace(sw[0], sw[1], 1)
        t = "\n".join(lines) + ("\n" if after[n].endswith("\n") else "")
        if t == after[n] or (n, t) in seen:
            continue
        seen.add((n, t)); out.append((dict(after, **{n: t}), f"mutant {n}:{i + 1} {how}"))
        if len(out) >= limit:
            break
    return out


def harmless(before, after, rng):
    out = []
    pairs = added_lines(before, after)
    if pairs:
        n, i = rng.choice(pairs); lines = after[n].splitlines(); lines[i] += "  # ok"
        out.append((dict(after, **{n: "\n".join(lines) + "\n"}), "harmless: trailing comment"))
        n, i = rng.choice(pairs); lines = after[n].splitlines(); lines.insert(i, "")
        out.append((dict(after, **{n: "\n".join(lines) + "\n"}), "harmless: blank line"))
    for n, t in after.items():
        if t and '"""' in t and n.endswith(".py") and any(l.strip().startswith('"""') for l in t.splitlines()):
            out.append((dict(after, **{n: t.replace('"""', '"""Note: ', 1)}), "harmless: docstring wording")); break
    return out


def run_in(files, tid, tests_cmd):
    root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    try:
        for n, t in files.items():
            if t is not None:
                p = os.path.join(root, n); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)
        ok, why = T.run_check(root, tid)
        cerr = agent.compile_error(root)
        tests = None
        if tests_cmd:
            r = subprocess.run([sys.executable, *tests_cmd[1:]], cwd=root, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            tests = (r.stderr + r.stdout).strip(); tests_ok = r.returncode == 0
        else:
            tests_ok = None
        return ok, cerr, tests, tests_ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build():
    T.fill_original_refs()
    rng = random.Random(0)
    split_of = {t: s for s, ts in SPLIT.items() for t in ts}
    rows = []
    for tid, t in sorted(T.TASKS.items()):
        before = dict(T.FIXTURES[t["fixture"]])
        after = dict(before)
        for n, c in t["ref"].items():
            if c is None:
                after.pop(n, None)
            else:
                after[n] = c
        tests_cmd = T.TEST_COMMAND if t["fixture"] == "shop" else None
        variants = [(after, "reference"), (before, "nothing done")]
        files = [n for n in t["ref"]]
        if len(files) > 1:
            for k in range(1, len(files)):
                for sub in list(itertools.combinations(files, k))[:2]:
                    part = dict(before)
                    for n in sub:
                        part[n] = t["ref"][n]
                        if part[n] is None: part.pop(n)
                    variants.append((part, f"partial: only {', '.join(sub)}"))
        variants += mutants(before, after, rng, 24)
        variants += harmless(before, after, rng)
        for state_files, how in variants:
            ok, cerr, tests, tests_ok = run_in(state_files, tid, tests_cmd)
            lang = rng.choice(["en", "es"])
            claim = rng.choice(CLAIMS)
            state = decide.done_state(t[lang], claim, before, state_files, cerr, tests)
            changed = state_files != before
            rows.append(dict(id=f"{tid}-{len(rows)}", task=tid, split=split_of[tid], lang=lang, how=how, state=state, label="YES" if ok else "NO",
                             objective_ok=bool(changed and not cerr and tests_ok is not False)))
    return rows


if __name__ == "__main__":
    rows = build()
    out = os.path.join(ROOT, "bench", "decide_data"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "done.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in SPLIT:
        rs = [r for r in rows if r["split"] == s]
        print(f"{s}: {len(rs)} states, {sum(r['label'] == 'YES' for r in rs)} done; objective_ok {sum(r['objective_ok'] for r in rs)} of which done {sum(r['objective_ok'] and r['label'] == 'YES' for r in rs)}")
    print("total", len(rows))
