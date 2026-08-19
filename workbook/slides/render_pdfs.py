#!/usr/bin/env python3
"""Render both HTML slide decks to PDF with WeasyPrint.

The LCG view ships an old tinycss2, so the user site (where weasyprint and
its upgraded deps live) must come first on PYTHONPATH:

    source scripts/setup_environment.sh ml
    PYTHONPATH="$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH" \
        python workbook/slides/render_pdfs.py
"""

import os
import sys
from pathlib import Path

USER_SITE = str(Path.home() / ".local/lib/python3.13/site-packages")
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

import weasyprint  # noqa: E402
from pypdf import PdfReader  # noqa: E402

HERE = Path(__file__).resolve().parent

for name in ("slides_en", "slides_cn"):
    html = HERE / f"{name}.html"
    pdf = HERE / f"{name}.pdf"
    weasyprint.HTML(str(html)).write_pdf(str(pdf))
    n = len(PdfReader(str(pdf)).pages)
    print(f"{pdf.name}: {n} pages")
