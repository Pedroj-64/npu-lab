"""Your own documents (PDF, Markdown, text, HTML, DOCX, images) as a chat source. Everything stays on this PC.

  add(model, name, data)  -> meta     extract text (OCR for scans/images), chunk, embed, store in ~/.local/share/npu-lab/documents/
  listing() / delete(id)
  retrieve(model, query, k)           best chunks across all documents: [{src, title, text, score, cos, kind}]
PDF text comes from `pdftotext`; pages without a text layer and images go through `tesseract` (both optional tools; a
clear error says what to install). Files are only ever passed to those tools as argument lists inside a temp folder.
"""
import hashlib, html, json, os, pickle, re, shutil, subprocess, tempfile, time, zipfile
import numpy as np

DIR = os.environ.get("NPU_LAB_DOCS") or os.path.expanduser("~/.local/share/npu-lab/documents")
TEXT = (".md", ".markdown", ".txt", ".rst", ".log", ".csv", ".json", ".yaml", ".yml", ".toml", ".py", ".rs", ".js", ".ts", ".c", ".h", ".cpp", ".sh")
IMAGES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
EXTS = TEXT + IMAGES + (".pdf", ".html", ".htm", ".docx")
MAX_BYTES, MAX_CHUNKS, CHUNK = 30 * 1024 * 1024, 4000, 600
MIN_COS = 0.33      # a chunk is only used when its similarity to the question reaches this (relevant hits scored 0.37-0.71, unrelated <= 0.13; see tests/test_documents.py)
OCR_LANG = os.environ.get("NPU_LAB_OCR_LANG", "spa+eng")


class DocError(Exception):
    pass


def _run(cmd, **kw):
    if not shutil.which(cmd[0]):
        raise DocError(f"`{cmd[0]}` is not installed: " + {"pdftotext": "sudo pacman -S poppler", "pdftoppm": "sudo pacman -S poppler",
                       "tesseract": "sudo pacman -S tesseract tesseract-data-spa tesseract-data-eng (needed for scanned PDFs and images)"}.get(cmd[0], "install it"))
    try:
        return subprocess.run(cmd, capture_output=True, timeout=kw.get("timeout", 120), check=False)
    except subprocess.TimeoutExpired:
        raise DocError(f"`{cmd[0]}` took too long")


def ocr_image(path):
    r = _run(["tesseract", path, "stdout", "-l", OCR_LANG], timeout=180)
    if r.returncode:
        raise DocError("tesseract failed: " + r.stderr.decode("utf-8", "ignore")[:200])
    return r.stdout.decode("utf-8", "ignore")


def _pdf(path, progress):
    out = _run(["pdftotext", "-layout", path, "-"]).stdout.decode("utf-8", "ignore")
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    res = [[i + 1, t, False] for i, t in enumerate(pages)]
    scans = [r for r in res if len(re.sub(r"\s", "", r[1])) < 20]
    if scans:                                              # no text layer on these pages: recognise them
        d = os.path.dirname(path)
        for n, r in enumerate(scans):
            progress(f"OCR page {r[0]} ({n + 1}/{len(scans)})")
            _run(["pdftoppm", "-r", "200", "-png", "-f", str(r[0]), "-l", str(r[0]), path, os.path.join(d, "pg")], timeout=120)
            img = next((os.path.join(d, f) for f in sorted(os.listdir(d)) if f.startswith("pg") and f.endswith(".png")), None)
            if img:
                r[1], r[2] = ocr_image(img), True
                os.remove(img)
    return res


def extract(name, data, progress=lambda m: None):
    """-> [[page, text, ocr_used]]. Raises DocError for unsupported/empty/unreadable files."""
    ext = os.path.splitext(name.lower())[1]
    if ext not in EXTS:
        raise DocError(f"unsupported file type {ext or '(none)'}; supported: pdf, md, txt, html, docx, images")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "f" + ext)                 # never the user's file name
        open(path, "wb").write(data)
        if ext == ".pdf":
            pages = _pdf(path, progress)
        elif ext in IMAGES:
            progress("OCR")
            pages = [[1, ocr_image(path), True]]
        elif ext == ".docx":
            try:
                xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
            except (zipfile.BadZipFile, KeyError):
                raise DocError("not a valid .docx")
            pages = [[1, html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))), False]]
        else:
            t = data.decode("utf-8", "ignore")
            if ext in (".html", ".htm"):
                t = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)))
            pages = [[1, t, False]]
    pages = [[p, re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip(), o] for p, t, o in pages]
    pages = [p for p in pages if p[1]]
    if not pages:
        raise DocError("no text found in this file")
    return pages


def _chunks(pages):
    import docs
    out = []
    for p, t, ocr in pages:
        for c in docs.chunk(t, CHUNK):
            out.append({"page": p, "text": c, "ocr": ocr})
    return out[:MAX_CHUNKS]


def _paths(i):
    if not re.fullmatch(r"[0-9a-f]{16}", i or ""):
        raise DocError("bad document id")
    return os.path.join(DIR, i + ".json"), os.path.join(DIR, i + ".pkl")


def add(model, name, data, progress=lambda m: None):
    if len(data) > MAX_BYTES:
        raise DocError(f"file too large (max {MAX_BYTES // 2**20} MB)")
    name = os.path.basename(name)[:120] or "document"
    pages = extract(name, data, progress)
    chunks = _chunks(pages)
    progress(f"embedding {len(chunks)} chunks")
    emb = np.concatenate([model.encode([f"{name}\n{c['text']}" for c in chunks[i:i + 32]]) for i in range(0, len(chunks), 32)]).astype(np.float16)
    i = hashlib.sha256(data).hexdigest()[:16]
    meta = {"id": i, "name": name, "pages": len(pages), "chunks": len(chunks), "ocr": any(c["ocr"] for c in chunks), "added": int(time.time())}
    mp, dp = _paths(i)
    os.makedirs(DIR, exist_ok=True)
    pickle.dump({"chunks": chunks, "emb": emb}, open(dp + ".tmp", "wb"))
    os.replace(dp + ".tmp", dp)
    json.dump(meta, open(mp, "w"))
    return meta


def listing():
    out = []
    if os.path.isdir(DIR):
        for f in sorted(os.listdir(DIR)):
            if f.endswith(".json"):
                try:
                    out.append(json.load(open(os.path.join(DIR, f))))
                except (OSError, json.JSONDecodeError):
                    pass
    return sorted(out, key=lambda m: -m["added"])


def delete(i):
    for p in _paths(i):
        if os.path.exists(p):
            os.remove(p)


def retrieve(model, query, k=4, gate=True):
    metas = listing()
    if not metas:
        return []
    q = model.encode([query])[0].astype(np.float32)
    cand = []
    for m in metas:
        d = pickle.load(open(_paths(m["id"])[1], "rb"))
        cos = d["emb"].astype(np.float32) @ q
        for j in np.argsort(-cos)[:k]:
            cand.append((float(cos[j]), m, d["chunks"][j]))
    cand.sort(key=lambda x: -x[0])
    out = []
    for cos, m, c in cand:
        if gate and cos < MIN_COS:
            break
        out.append({"src": f"{m['name']} p.{c['page']}" + (" (OCR)" if c["ocr"] else ""), "title": m["name"], "text": c["text"],
                    "score": cos, "cos": cos, "kind": "doc"})
        if len(out) >= k:
            break
    return out
