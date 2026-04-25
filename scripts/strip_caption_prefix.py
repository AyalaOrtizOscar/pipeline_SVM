#!/usr/bin/env python3
"""Strip manual 'Figura X.Y.' / 'Tabla X.Y.' prefix from \caption{} commands.

This fixes the double-numbering issue in LoF/LoT where LaTeX autonumbers
and the caption also carries a manual numeric prefix from the Word draft.
"""
from pathlib import Path
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TEX = Path("D:/pipeline_SVM/informe_proyecto/informe_proyecto.tex")
txt = TEX.read_text(encoding="utf-8")

# Match \caption[...]{Figura N.N. ...} or \caption{Tabla N.N. ...}
# Handles optional [short] caption, letter-prefix (A.1), decimal (7.3a)
PREFIX = re.compile(
    r"(\\caption(?:\[[^\]]*\])?\{)(?:Figura|Tabla)\s+[A-Z0-9]+\.\d+[a-z]?\.?\s*"
)

matches = PREFIX.findall(txt)
print(f"Captions with manual prefix found: {len(matches)}")

new_txt = PREFIX.sub(r"\1", txt)

# Fix the \caption[short]{...} optional: the short form may also carry the
# prefix — rebuild entries where short != long by stripping inside [ ] too.
SHORT = re.compile(
    r"(\\caption)\[(?:Figura|Tabla)\s+[A-Z0-9]+\.\d+[a-z]?\.?\s*([^\]]*)\]"
)
new_txt = SHORT.sub(r"\1[\2]", new_txt)

# Clean up residual "Apéndice ." lone text lines that were headers
if new_txt != txt:
    TEX.write_text(new_txt, encoding="utf-8")
    print(f"✓ Saved {TEX}")
else:
    print("No changes.")
