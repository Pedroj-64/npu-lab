"""Continuous research: look for new public work on first-generation NPUs (XDNA1 / npu1) and log it in docs/RESEARCH_LOG.md.
usage: python tools/research_scan.py [--dry-run] [--classify] [--selftest]
--classify: the NPU model (Qwen3-0.6B, needs the NPU free) tags each NEW hit relevant / off-topic as a hint; the status stays `unverified`.
Privacy: only the fixed search keywords below leave the machine; no code, paths or user content. GitHub token is optional
(env GITHUB_TOKEN or ~/.config/npu-lab/github_token) and only raises the rate limit. New hits enter as `unverified`; the manual table is never edited.
Spanish: busca trabajo público nuevo sobre NPU de primera generación y lo anota en docs/RESEARCH_LOG.md (sin verificar).
Standard library only."""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "docs", "RESEARCH_LOG.md")
BEGIN, END = "<!-- research_scan:begin -->", "<!-- research_scan:end -->"
QUERIES = ["amdxdna", "xdna npu1", "ryzen ai npu phoenix linux", "mlir-aie npu1", "xdna whisper", "xdna llm linux"]
WATCH_REPOS = ["amd/iron", "Xilinx/mlir-aie", "ROCm/FastFlowLM", "lemonade-sdk/lemonade", "amd/xdna-driver", "drakosha/whisper-xdna", "hnws/xdna1-npu-lab", "FireBurn/ggml-xdna1"]


def token():
    t = os.environ.get("GITHUB_TOKEN")
    p = os.path.expanduser("~/.config/npu-lab/github_token")
    return t or (open(p).read().strip() if os.path.exists(p) else None)


def get(url):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "npu-lab-research-scan"}
    if token():
        h["Authorization"] = f"Bearer {token()}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
        return json.load(r)


def hits():
    out = {}
    for q in QUERIES:
        try:
            d = get("https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "sort": "updated", "per_page": 8}))
        except OSError as e:
            print(f"skip '{q}': {e}", file=sys.stderr); continue
        for r in d.get("items", []):
            out[r["html_url"]] = f"{r['full_name']} — {(r.get('description') or '').strip()[:110]} (updated {r['pushed_at'][:10]}, {r['stargazers_count']} stars)"
        time.sleep(2)                       # stay under the unauthenticated search limit
    for repo in WATCH_REPOS:
        try:
            for rel in get(f"https://api.github.com/repos/{repo}/releases?per_page=2"):
                out[rel["html_url"]] = f"{repo} release {rel['tag_name']} ({rel['published_at'][:10]})"
        except OSError as e:
            print(f"skip releases {repo}: {e}", file=sys.stderr)
    return out


ASK = ("Answer YES or NO. Question: can this item be used on a first-generation AMD XDNA1 NPU (Ryzen 7040/8040, Phoenix or Hawk Point, npu1) running Linux? "
       "Answer NO if it is only for Windows, only for Strix Point/Strix Halo/XDNA2, only for phones, or is unrelated to NPUs.")
XDNA1 = re.compile(r"xdna ?1|npu1|phoenix|hawk ?point|1022:1502|\b[78]040\b|aie2\b", re.I)
OTHER = re.compile(r"strix|xdna ?2|krackan|windows|npu2|npu4", re.I)


def hint(desc, engine=None):
    """'release' | 'relevant' | 'off-topic' | None. Plain rules first (a 0.6B-1.7B model alone tagged everything relevant, see docs/CHANGELOG.md);
    the NPU model only decides what the rules leave open, with a YES/NO question."""
    if re.search(r" release \S+ \(", desc):
        return "release"
    if XDNA1.search(desc):
        return "relevant"
    if OTHER.search(desc):
        return "off-topic"
    return None if engine is None else ("relevant" if engine.classify(desc[:400], ASK, ["YES", "NO"]) == "YES" else "off-topic")


def label_hits(found, text, engine=None):
    """{url: hint} for the hits NOT already in the log."""
    out = {u: hint(d, engine) for u, d in found.items() if u not in text}
    return {u: h for u, h in out.items() if h}


def merge(text, found, today, labels=None):
    """Add only URLs not already present anywhere in the log; return (new_text, added). `labels`: optional {url: hint} shown before the link."""
    new = {u: d for u, d in found.items() if u not in text}
    if not new:
        return text, 0
    rows = "".join(f"- {today} · unverified" + (f" · {labels[u]}?" if labels and u in labels else "") + f" · [{d}]({u})\n" for u, d in sorted(new.items()))
    return text.replace(END, rows + END), len(new)


def selftest():
    base = f"x\n{BEGIN}\n{END}\n"
    t1, n1 = merge(base, {"https://a": "A"}, "2026-01-01")
    t2, n2 = merge(t1, {"https://a": "A"}, "2026-01-02")
    assert (n1, n2) == (1, 0) and t1 == t2 and "unverified" in t1
    class Fake:
        def classify(s, p, system, labels): return "YES" if "usable" in p else "NO"
    f = {"https://a": "A", "https://b": "tool for XDNA1", "https://c": "Windows only app", "https://d": "x/y release v1 (2026-01-01)", "https://e": "usable thing", "https://f": "other thing"}
    lab = label_hits(f, t1, Fake())
    assert lab == {"https://b": "relevant", "https://c": "off-topic", "https://d": "release", "https://e": "relevant", "https://f": "off-topic"}, lab   # the logged one is skipped
    assert label_hits(f, t1) == {"https://b": "relevant", "https://c": "off-topic", "https://d": "release"}, "without a model only the rules speak"
    t3, _ = merge(t1, f, "2026-01-03", lab)
    assert "unverified · relevant? · [tool for XDNA1]" in t3 and "off-topic?" in t3 and "release?" in t3
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    text = open(LOG).read()
    if BEGIN not in text or END not in text:
        sys.exit("markers missing in docs/RESEARCH_LOG.md")
    found, engine = hits(), None
    if a.classify:
        import subprocess
        if subprocess.run("pgrep -af 'src/(serve|chat_cli|qwen)|bench/(perf_baseline|energy|watch)' | grep -v pgrep", shell=True, capture_output=True).stdout:
            print("NPU busy: not classifying", file=sys.stderr)
        else:
            sys.path.insert(0, os.path.join(ROOT, "src"))
            import qwen_npu
            engine = qwen_npu.Qwen(os.path.join(ROOT, "models/qwen3s"), gpu=False)
    new_text, added = merge(text, found, time.strftime("%Y-%m-%d"), label_hits(found, text, engine))
    print(f"{added} new")
    if added and not a.dry_run:
        open(LOG, "w").write(new_text)


if __name__ == "__main__":
    main()
