"""A compact map of a Python project: every file with its classes and functions (signature + first docstring line), so a model that sees only one
file in full still knows what the rest offers. stdlib only (ast); other languages are not mapped yet (ponytail: universal-ctags if installed).
  build(root, max_chars=4000) -> str"""
import ast, os

SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "site-packages"}


def _sig(f):
    a = f.args
    pos = a.posonlyargs + a.args
    d = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    parts = [x.arg + (f"={ast.unparse(v)}" if v is not None else "") for x, v in zip(pos, d)]
    if a.vararg: parts.append("*" + a.vararg.arg)
    parts += [x.arg + (f"={ast.unparse(v)}" if v is not None else "") for x, v in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg: parts.append("**" + a.kwarg.arg)
    return f"{f.name}({', '.join(parts)})"


def _doc(n):
    d = ast.get_docstring(n)
    return " - " + d.splitlines()[0][:80] if d else ""


def _file(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"  def {_sig(n)}{_doc(n)}")
        elif isinstance(n, ast.ClassDef):
            out.append(f"  class {n.name}{_doc(n)}")
            out += [f"    def {_sig(m)}{_doc(m)}" for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")]
    return out


def build(root, max_chars=4000):
    lines = []
    for d, subs, names in os.walk(root):
        subs[:] = sorted(s for s in subs if s not in SKIP and not s.startswith("."))
        for f in sorted(names):
            if f.endswith(".py"):
                body = _file(os.path.join(d, f))
                if body:
                    lines += [os.path.relpath(os.path.join(d, f), root) + ":", *body]
    text = "\n".join(lines)
    if len(text) > max_chars:                                 # over budget: drop the docstrings, then the tail
        text = "\n".join(l.split(" - ")[0] for l in lines)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n  ... (map cut: over budget)"
    return text
