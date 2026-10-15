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
TEXT = (".md", ".markdown", ".txt", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".tex", ".sql",
        ".py", ".rs", ".js", ".mjs", ".ts", ".jsx", ".tsx", ".c", ".h", ".cpp", ".hpp", ".go", ".java", ".kt", ".css", ".scss", ".sh", ".fish", ".lua", ".rb", ".php", ".nix")
IMAGES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
ZIPS = (".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp", ".epub")          # office / e-book formats: text lives in XML or XHTML inside a zip
BINARY = (".exe", ".dll", ".so", ".bin", ".o", ".a", ".dmg", ".iso", ".img", ".zip", ".gz", ".xz", ".bz2", ".zst", ".7z", ".rar", ".tar", ".apk", ".jar", ".class", ".msi", ".deb", ".rpm",
          ".mp3", ".mp4", ".mkv", ".avi", ".mov", ".wav", ".flac", ".ogg", ".sqlite", ".db", ".pyc")      # never read as text, whatever the content looks like
EXTS = TEXT + IMAGES + ZIPS + (".pdf", ".html", ".htm", ".rtf")
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


def _zipped(path, ext):
    """docx / pptx / xlsx / odt / ods / odp / epub -> [[page, text, False]] (one page per slide, sheet or chapter where the format has them)."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise DocError(f"not a valid {ext} file")
    names, rd = z.namelist(), lambda n: z.read(n).decode("utf-8", "ignore")
    strip = lambda x: html.unescape(re.sub(r"<[^>]+>", " ", x))
    key = lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]       # slide2 before slide10
    if ext == ".docx":
        return [[1, strip(re.sub(r"</w:p>", "\n", rd("word/document.xml"))), False]] if "word/document.xml" in names else []
    if ext == ".pptx":
        return [[i + 1, strip(re.sub(r"</a:p>", "\n", rd(n))), False] for i, n in enumerate(sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=key))]
    if ext == ".xlsx":
        sst = re.findall(r"(?s)<si>(.*?)</si>", rd("xl/sharedStrings.xml")) if "xl/sharedStrings.xml" in names else []
        sst = [strip(x).strip() for x in sst]
        out = []
        for i, n in enumerate(sorted((n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)), key=key)):
            rows = []
            for row in re.findall(r"(?s)<row[^>]*>(.*?)</row>", rd(n)):
                cells = []
                for attrs, body in re.findall(r"(?s)<c([^>]*)>(.*?)</c>", row):
                    v = re.search(r"<v>(.*?)</v>", body)
                    if v:
                        cells.append(sst[int(v[1])] if 't="s"' in attrs and v[1].isdigit() and int(v[1]) < len(sst) else html.unescape(v[1]))
                rows.append(" | ".join(cells))
            out.append([i + 1, "\n".join(rows), False])
        return out
    if ext == ".epub":
        return [[i + 1, strip(re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", rd(n))), False] for i, n in enumerate(sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))))]
    return [[1, strip(re.sub(r"</text:(p|h)>", "\n", rd("content.xml"))), False]] if "content.xml" in names else []      # odt / ods / odp


def extract(name, data, progress=lambda m: None):
    """-> [[page, text, ocr_used]]. Raises DocError for unsupported/empty/unreadable files."""
    ext = os.path.splitext(name.lower())[1]
    if ext not in EXTS and (ext in BINARY or data[:4] in (b"MZ\x90\0", b"\x7fELF") or data[:2] == b"MZ" or b"\0" in data[:8192] or not data.strip()):          # an unknown extension is fine when the content is plain text
        raise DocError(f"unsupported file type {ext or '(none)'}: not plain text. Supported: pdf, docx, pptx, xlsx, odt, epub, rtf, html, images (OCR), and any text/code file")
    if ext not in EXTS:
        ext = ".txt"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "f" + ext)                 # never the user's file name
        open(path, "wb").write(data)
        if ext == ".pdf":
            pages = _pdf(path, progress)
        elif ext in IMAGES:
            progress("OCR")
            pages = [[1, ocr_image(path), True]]
        elif ext in ZIPS:
            pages = _zipped(path, ext)
        elif ext == ".rtf":
            t = data.decode("latin-1")
            t = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m[1]).decode("cp1252", "ignore"), t)
            pages = [[1, re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", t), False]]
        else:
            t = data.decode("utf-8", "ignore")
            if ext in (".html", ".htm"):
                t = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)))
            pages = [[1, t, False]]
    pages = [[p, re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip(), o] for p, t, o in pages]
    pages = [p for p in pages if p[1]]
    if not pages:
        raise DocError("no readable text in this file" + (": for an image, the models here cannot look at pictures, they only get text found by OCR "
                       "(a photo without writing gives none)" if ext in IMAGES else ""))
    return pages


def _chunks(pages):
    import docs
    out = []
    for p, t, ocr in pages:
        for c in docs.chunk_headed(t, CHUNK):
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


POINTS_AT = re.compile(r"\b(this|that|the|my|attached|uploaded)\s+(file|document|doc|pdf|image|photo|picture|scan|screenshot|spreadsheet|slides?|sheet|book|paper|article)s?\b"
                       r"|\b(est[eaoé]s?|ese|esa|mi|el|la|adjunto|adjunta|subido|subida)\s+(archivo|documento|pdf|imagen|foto|captura|escaneo|hoja|libro|art[ií]culo|presentaci[oó]n|diapositivas?)s?\b"
                       r"|\b(archivo|documento|pdf|imagen|foto|adjunto)s?\b", re.I)


def refers(query, metas=()):
    """True when the question points at a file ("this PDF", "esta foto", or a stored file's name): the similarity gate is skipped then,
    because "what does this photo say?" shares no words with the text inside it."""
    q = query.lower()
    return bool(POINTS_AT.search(q)) or any(len(stem) >= 4 and stem in q for stem in (os.path.splitext(m["name"].lower())[0] for m in metas))


def retrieve(model, query, k=4, gate=True):
    metas = listing()
    if not metas:
        return []
    gate = gate and not refers(query, metas)
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
