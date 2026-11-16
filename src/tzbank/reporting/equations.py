"""Report equations as Word equation objects (prompt 24b B; owner: no typed formulas).

A display equation is written in the report source as a paragraph of its own, ``$$ <LaTeX> $$``. The build converts it
LaTeX -> MathML (``latex2mathml``) -> OMML with Microsoft Office's own stylesheet (``MML2OMML.XSL``, read from the Office
installation the report build already needs for Word export; not copied into the repository), and inserts it as an
``m:oMathPara``: real fractions, subscripts and superscripts that Word treats as an equation.

``typed_formulas(text)`` finds formula-like text typed by hand (an underscore or caret between letters, such as
``k_e`` or ``x^2``) so the build can refuse it.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from lxml import etree

DISPLAY = re.compile(r"^\$\$(.+)\$\$$", re.S)
TYPED = re.compile(r"(?<![\w/\\.{])[A-Za-z][A-Za-z0-9]*(?:_\{?[A-Za-z0-9*]+\}?|\^\{?[A-Za-z0-9*]+\}?)(?![\w/.])")
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _xsl_path() -> Path:
    env = os.environ.get("TZBANK_MML2OMML")
    cands = [Path(env)] if env else []
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
        if base:
            cands += [Path(base) / "Microsoft Office" / "root" / f"Office{v}" / "MML2OMML.XSL" for v in (16, 15)]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("MML2OMML.XSL not found (Microsoft Office); set TZBANK_MML2OMML to its path")


@lru_cache(maxsize=1)
def _xslt() -> etree.XSLT:
    return etree.XSLT(etree.parse(str(_xsl_path())))


def omml(latex: str) -> etree._Element:
    """LaTeX -> an ``m:oMath`` element (Word equation)."""
    from latex2mathml.converter import convert

    mml = etree.fromstring(convert(latex.strip()).encode("utf-8"))
    out = _xslt()(mml).getroot()
    return out if etree.QName(out).localname == "oMath" else out.find(f".//{{{M}}}oMath")


def add_display(paragraph, latex: str) -> None:
    """Append a centred display equation (``m:oMathPara``) to a python-docx paragraph."""
    para = etree.SubElement(paragraph._p, f"{{{M}}}oMathPara")
    para.append(omml(latex))


def typed_formulas(text: str) -> list[str]:
    """Hand-typed formula fragments (``k_e``, ``y_TZ``, ``x^2``) in prose, outside $$...$$ equations and tokens."""
    t = re.sub(r"\$\$.+?\$\$|\{\{[^}]*\}\}|\[\[[^\]]*\]\]|<!--.*?-->|`[^`]*`|https?://\S+", " ", text, flags=re.S)
    return [m.group(0) for m in TYPED.finditer(t)]
