"""Continuous research: look for new public work on first-generation NPUs (XDNA1 / npu1) and log it in docs/RESEARCH_LOG.md.
usage: python tools/research_scan.py [--dry-run] [--selftest]
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


def merge(text, found, today):
    """Add only URLs not already present anywhere in the log; return (new_text, added)."""
    new = {u: d for u, d in found.items() if u not in text}
    if not new:
        return text, 0
    rows = "".join(f"- {today} · unverified · [{d}]({u})\n" for u, d in sorted(new.items()))
    return text.replace(END, rows + END), len(new)


def selftest():
    base = f"x\n{BEGIN}\n{END}\n"
    t1, n1 = merge(base, {"https://a": "A"}, "2026-01-01")
    t2, n2 = merge(t1, {"https://a": "A"}, "2026-01-02")
    assert (n1, n2) == (1, 0) and t1 == t2 and "unverified" in t1
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    text = open(LOG).read()
    if BEGIN not in text or END not in text:
        sys.exit("markers missing in docs/RESEARCH_LOG.md")
    new_text, added = merge(text, hits(), time.strftime("%Y-%m-%d"))
    print(f"{added} new")
    if added and not a.dry_run:
        open(LOG, "w").write(new_text)


main()
