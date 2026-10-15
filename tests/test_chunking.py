"""Offline test for docs.chunk_headed: sections are not mixed, each piece carries its heading path, plain text behaves like chunk().
usage: python tests/test_chunking.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import docs

md = "# Guide\nintro line\n\n## Install\nrun pacman -S foo\n\n### Arch\nuse paru for the AUR\n\n## Remove\nrun pacman -R foo\n"
c = docs.chunk_headed(md, 200)
assert any(x.startswith("[Guide > Install > Arch]") and "paru" in x for x in c), c
assert not any("paru" in x and "pacman -R" in x for x in c), "two sections in one piece"
assert any(x.startswith("[Guide > Remove]") for x in c), c
assert docs.chunk_headed("Title\n=====\nbody one\n\nSub\n---\nbody two\n", 200)[1].startswith("[Title > Sub]")
plain = "\n".join(f"line {i} " + "x" * 30 for i in range(40))
assert docs.chunk_headed(plain, 300) == docs.chunk(plain, 300), "text without headings changed"
long = "# A\n" + "\n".join("w" * 80 for _ in range(30))
assert all(x.startswith("[A]\n") and len(x) < 400 for x in docs.chunk_headed(long, 300)), "long sections are split and keep the heading"
print("chunking ok")
