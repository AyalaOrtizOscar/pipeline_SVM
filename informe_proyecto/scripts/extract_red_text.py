#!/usr/bin/env python3
"""Extract red-colored text from Word doc, grouped by heading context."""
import sys
from pathlib import Path
from docx import Document
from docx.shared import RGBColor

DOCX = Path(r"C:/Users/ayala/Downloads/Informe del proyecto_v2 (1).docx")

def is_red(run):
    color = run.font.color
    if color is None or color.rgb is None:
        return False
    r, g, b = color.rgb[0], color.rgb[1], color.rgb[2]
    return r > 150 and g < 80 and b < 80


def main():
    # Force UTF-8 output to avoid cp1252 errors
    sys.stdout.reconfigure(encoding="utf-8")
    doc = Document(DOCX)
    current_heading = "(inicio)"
    current_chapter = "(inicio)"
    hits = []

    for para in doc.paragraphs:
        style = (para.style.name or "").lower()
        text = para.text.strip()
        if "heading 1" in style or "título 1" in style or "titulo 1" in style:
            current_chapter = text
            current_heading = text
        elif "heading" in style or "título" in style or "titulo" in style:
            current_heading = text

        red_runs = [r.text for r in para.runs if is_red(r) and r.text.strip()]
        if red_runs:
            combined = "".join(red_runs)
            hits.append({
                "chapter": current_chapter,
                "heading": current_heading,
                "paragraph_preview": text[:200],
                "red_text": combined.strip(),
            })

    print(f"Total red fragments: {len(hits)}\n")
    for i, h in enumerate(hits, 1):
        print(f"--- [{i}] CHAPTER: {h['chapter']}")
        print(f"    HEADING: {h['heading']}")
        print(f"    PARA: {h['paragraph_preview']}")
        print(f"    RED : {h['red_text']}")
        print()


if __name__ == "__main__":
    main()
