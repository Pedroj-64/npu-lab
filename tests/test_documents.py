"""Document import without the NPU: formats, text fallback, empty-image message, and when a question points at a file.
usage: .venv/bin/python tests/test_documents.py"""
import io, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import documents as D
from PIL import Image, ImageDraw

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

def zipped(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, t in files.items(): z.writestr(n, t)
    return b.getvalue()

def png(text=None, noise=False):
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)) if noise else Image.new("RGB", (640, 200), "white")
    if text: ImageDraw.Draw(im).text((20, 20), text, fill="black")
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()

text = lambda n, d: D.extract(n, d)[0][1]
check("unknown extension with plain text is read", "hello world" in text("notes.xyz", b"hello world"))
check("no extension with plain text is read", "hello" in text("README", b"hello"))
try: D.extract("a.bin", b"\0\1\2"); check("binary with unknown extension is refused", False)
except D.DocError as e: check("binary with unknown extension is refused", "not plain text" in str(e))
check("code file (.go)", "func main" in text("m.go", b"package main\nfunc main() {}"))
check("pptx: one page per slide", [p[1] for p in D.extract("a.pptx", zipped({"ppt/slides/slide2.xml": "<a:p><a:t>second</a:t></a:p>", "ppt/slides/slide10.xml": "<a:p><a:t>tenth</a:t></a:p>", "ppt/slides/slide1.xml": "<a:p><a:t>first</a:t></a:p>"}))] == ["first", "second", "tenth"])
check("xlsx: shared strings and numbers", "name | 42" in text("a.xlsx", zipped({"xl/sharedStrings.xml": "<sst><si><t>name</t></si></sst>", "xl/worksheets/sheet1.xml": '<row><c t="s"><v>0</v></c><c><v>42</v></c></row>'})))
check("odt", "hola odt" in text("a.odt", zipped({"content.xml": "<text:p>hola odt</text:p>"})))
check("epub", "chapter one" in text("a.epub", zipped({"OEBPS/c1.xhtml": "<html><body><p>chapter one</p></body></html>"})))
check("docx still works", "plain docx" in text("a.docx", zipped({"word/document.xml": "<w:p><w:t>plain docx</w:t></w:p>"})))
check("rtf", "hola rtf" in text("a.rtf", br"{\rtf1\ansi {\b hola rtf}}"))
check("image with writing -> OCR text", "hola" in text("a.PNG", png("hola mundo")).lower())
try: D.extract("photo.jpg", png(noise=True)); check("photo without text is refused with a clear reason", False)
except D.DocError as e: check("photo without text is refused with a clear reason", "cannot look at pictures" in str(e))
metas = [{"name": "Informe_2024.pdf"}]
check("'this PDF' points at a file", D.refers("what does this PDF say?"))
check("'esta foto' points at a file", D.refers("¿qué dice esta foto?"))
check("a file name points at it", D.refers("resume informe_2024 por favor", metas))
check("an unrelated question does not", not D.refers("how do I install paru on Arch?", metas))
sys.exit(1 if fails else 0)
