#!/usr/bin/env python3
"""Scan v6.docx for paragraphs containing drawings (embedded figures)."""
from docx import Document
from pathlib import Path
from docx.oxml.ns import qn

SRC = Path('C:/Users/ayala/Downloads/files/Informe_del_proyecto_v6.docx')
doc = Document(str(SRC))

print("Paragraphs containing drawings/pictures:")
for idx, p in enumerate(doc.paragraphs):
    el = p._element
    has_drawing = el.find('.//' + qn('w:drawing')) is not None
    has_pict = el.find('.//' + qn('w:pict')) is not None
    if has_drawing or has_pict:
        txt = p.text.strip()[:80]
        kind = 'drawing' if has_drawing else 'pict'
        print(f"  [{idx:4d}] {kind:8s} | text: '{txt}'")
