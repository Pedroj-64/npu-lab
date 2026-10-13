"""Closed decisions, calibrated and local: yes/no, pick one option, or a 1-5 score, answered as PROBABILITIES instead of free text.
    decide(state, questions, backend) -> [ {option: probability, ...}, ... ]      (one backend call per question; probabilities sum to 1)
A question is {"task": name} for a known task (TASKS below: "done", "route", "mixer"), or {"kind": "yesno"|"choice"|"score", "text": ..., "options": [...]}.
Backends (same interface, interchangeable):
  LogprobBackend(engine)  the model that is already loaded reads the state once and its next-word scores over the options are the probabilities
                          (no sampling, no extra memory; it overwrites the engine's KV cache, so the next chat turn re-reads its prompt).
  EmbedBackend(embedder)  MiniLM sentence vector of the state + a logistic regression trained on labelled examples (no model call at all).
Nothing here talks to a network. The guard for the agent's DONE is in done_guard(): objective checks first, the probability only as a secondary signal."""
import difflib, json, os, re
import numpy as np

SYSTEM_DONE = ("You check the work of a coding agent. Answer only from the evidence below. Do not trust the agent's own claim that it is finished: "
               "judge the files that changed and the output of the checks.")
SYSTEM_MIXER = ("You decide which model should answer a request. fast = short facts, translations, small talk; medium = explanations, short code, questions about "
                "documents; slow = multi-file code changes, refactors, long or multi-step reasoning. Do not answer the request.")
SYSTEM_ROUTE = ("You route questions to a source. Do not answer them.\n"
                "None: general knowledge, programming, science, writing, concepts. Example: 'explain what a closure is'.\n"
                "Docs: how to use a Linux command, package manager or installed tool. Example: 'how do I delete a folder with rm'.\n"
                "Web: recent events, news, prices, versions, anything that changes over time. Example: 'who is the current champion'.\n"
                "History: refers to what was said earlier in this conversation. Example: 'make that shorter'.")
# Flag (off by default): NPU_LAB_ROUTE_LABELS=letters shows the router options to the model as A-D instead of None/Docs/Web/History. The 2B has a strong prior for the
# word "None" and answers it for everything with the words (measured: bench/DECIDE.md); letters remove that prior. The options returned keep the data's names.
ROUTE_LETTERS = os.environ.get("NPU_LAB_ROUTE_LABELS", "words") == "letters"
SYSTEM_ROUTE_LETTERS = SYSTEM_ROUTE.replace("None:", "A:").replace("Docs:", "B:").replace("Web:", "C:").replace("History:", "D:")
TASKS = {
    "done": dict(system=SYSTEM_DONE, options=["YES", "NO"], text="Does the evidence show that the task is completely and correctly done?"),
    # the labels the model sees are ONE token each (NONE/DOCS/HISTORY are split into pieces by the tokenizer and score badly); the options returned keep the data's names
    "route": dict(system=SYSTEM_ROUTE_LETTERS if ROUTE_LETTERS else SYSTEM_ROUTE, options=["NONE", "DOCS", "WEB", "HISTORY"],
                  labels=["A", "B", "C", "D"] if ROUTE_LETTERS else ["None", "Docs", "Web", "History"], text=None),
    "mixer": dict(system=SYSTEM_MIXER, options=["fast", "medium", "slow"], text="Which model should answer this request?"),
}
for _t in TASKS.values():
    _t.setdefault("labels", list(_t["options"]))
KINDS = {"yesno": ["YES", "NO"], "score": ["1", "2", "3", "4", "5"]}


def resolve(q):
    """-> (task name or None, system prompt, options, question text, labels shown to the model)."""
    t = TASKS.get(q.get("task"))
    if t:
        return q["task"], t["system"], list(t["options"]), q.get("text") or t["text"] or "", list(t["labels"])
    opts = q.get("options") or KINDS.get(q.get("kind", ""))
    assert opts, f"question without options: {q}"
    return None, q.get("system", "Answer with one of the options. Do not explain."), list(opts), q.get("text", ""), list(opts)


class LogprobBackend:
    name = "logprob"

    def __init__(self, engine, limit=3000):
        self.engine, self.limit = engine, limit

    def probs(self, state, q):
        _, system, opts, text, labels = resolve(q)
        p = self.engine.option_probs((state + "\n" + text).strip(), system, labels, limit=self.limit, strict=True)
        return dict(zip(opts, (float(x) for x in p)))


class EmbedBackend:
    name = "embed"

    def __init__(self, embedder=None, models=None):
        self.emb, self.models = embedder, models or {}          # models: task -> (W (D,K), b (K,), options)

    def _vec(self, texts):
        return np.asarray(self.emb.encode(list(texts)), np.float32)

    def fit(self, task, texts, labels, C=1.0):
        """Multinomial logistic regression (L2, class-balanced) on the MiniLM vectors; labels are option names."""
        from scipy.optimize import minimize
        opts = list(TASKS[task]["options"]); X = self._vec(texts); y = np.array([opts.index(l) for l in labels]); n, d = X.shape; k = len(opts)
        w = np.array([n / (k * max(1, (y == j).sum())) for j in range(k)])[y]

        def loss(theta):
            W, b = theta[:d * k].reshape(d, k), theta[d * k:]
            z = X @ W + b; z -= z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            P = np.exp(lp); G = P.copy(); G[np.arange(n), y] -= 1; G *= w[:, None]
            return -(w * lp[np.arange(n), y]).sum() / n + 0.5 * (W * W).sum() / (C * n), np.concatenate([(X.T @ G / n + W / (C * n)).ravel(), G.mean(0)])

        theta = minimize(loss, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        self.models[task] = (theta[:d * k].reshape(d, k), theta[d * k:], opts)
        return self

    def probs(self, state, q):
        task = resolve(q)[0]
        W, b, mopts = self.models[task]
        z = self._vec([state])[0] @ W + b; z = np.exp(z - z.max()); z /= z.sum()
        return dict(zip(mopts, (float(x) for x in z)))

    def save(self, path):
        np.savez(path, **{f"{t}_W": m[0] for t, m in self.models.items()}, **{f"{t}_b": m[1] for t, m in self.models.items()})

    @classmethod
    def load(cls, path, embedder):
        z = np.load(path)
        return cls(embedder, {t: (z[f"{t}_W"], z[f"{t}_b"], list(TASKS[t]["options"])) for t in {k.rsplit("_", 1)[0] for k in z.files}})


def decide(state, questions, backend):
    """One backend call per question (a small model answers one closed question much better than five at once)."""
    return [backend.probs(state, q) for q in questions]


# ----- the evidence for the DONE guard: short, deterministic -----
def change_summary(before, after, max_lines=14):
    """before/after: {path: text}. -> ('Changed files: a.py (+3 -1), b.py (new)', 'Diff:\\n...' limited to max_lines +/- lines)."""
    names, diff = [], []
    for n in sorted(set(before) | set(after)):
        a, b = before.get(n), after.get(n)
        if a == b or n.endswith(".pyc") or "__pycache__" in n:
            continue
        al, bl = (a or "").splitlines(), (b or "").splitlines()
        d = [l for l in difflib.unified_diff(al, bl, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        tag = "new" if a is None else "deleted" if b is None else f"+{sum(l[0] == '+' for l in d)} -{sum(l[0] == '-' for l in d)}"
        names.append(f"{n} ({tag})"); diff += [f"{n}: {l}" for l in d]
    shown = diff[:max_lines] + ([f"... {len(diff) - max_lines} more changed lines"] if len(diff) > max_lines else [])
    return "Changed files: " + (", ".join(names) or "none"), "Diff:\n" + ("\n".join(shown) or "(no changes)")


def done_state(task, claim, before, after, compile_err="", tests=None):
    """The evidence text: task, the agent's claim (marked as unverified), what changed, compile status and the output of the user's tests (or 'none configured')."""
    files, diff = change_summary(before, after)
    tests_line = "Tests: none configured" if tests is None else "Tests: " + " | ".join(tests.strip().splitlines()[-3:])[:200]
    return f"Task: {task}\nThe agent claims (unverified): {claim}\n{files}\n{diff}\nCompile: {compile_err or 'ok'}\n{tests_line}"


def done_guard(before, after, compile_err, tests_passed, backend=None, state="", threshold=0.5):
    """Should the agent's DONE be accepted? -> (accepted, reason, p_yes).
    1) objective checks first (refuse on any failure): something changed, it compiles, the user's tests pass (when configured);
    2) then the backend's probability that the work is done must reach `threshold` (calibrated on held-out data, see bench/DECIDE.md);
    3) the model's verdict is never the only evidence: with no backend the objective checks alone decide."""
    if before == after:
        return False, "no file changed", 0.0
    if compile_err:
        return False, compile_err, 0.0
    if tests_passed is False:
        return False, "the project's tests fail", 0.0
    if backend is None:
        return True, "objective checks pass", None
    p = decide(state, [{"task": "done"}], backend)[0]["YES"]
    return (p >= threshold), (f"p(done)={p:.2f} {'>=' if p >= threshold else '<'} {threshold:.2f}"), p
