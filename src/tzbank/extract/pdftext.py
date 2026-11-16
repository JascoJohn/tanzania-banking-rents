"""Row reconstruction from PDF text layers (PyMuPDF). Words are grouped into visual rows by their y position so a
statement line reads 'Interest income 12,345 10,234'. Numbers in brackets are negative.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf

NUM = re.compile(r"^\(?-?[\d]{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?$|^\(?-?\d+(?:\.\d+)?\)?$|^-$")


@dataclass
class Row:
    page: int  # 1-based
    text: str
    label: str
    numbers: list[float | None]
    y: float = 0.0


def parse_number(tok: str) -> float | None:
    """'(1,234)' -> -1234.0; '-' -> 0.0 (nil); '12.5' -> 12.5."""
    t = tok.strip()
    if t in {"-", "–", "—"}:
        return 0.0
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(",", "").replace(" ", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


@lru_cache(maxsize=16)
def _doc(path: str) -> pymupdf.Document:
    return pymupdf.open(path)


def text_pages(path: Path) -> list[int]:
    """1-based pages with a usable text layer (> 200 characters)."""
    d = _doc(str(path))
    return [i + 1 for i in range(d.page_count) if len(d[i].get_text()) > 200]


def page_rows(path: Path, page: int, ytol: float = 3.0, half: str | None = None, split: float = 0.5) -> list[Row]:
    """Visual rows of one page (1-based). ``half`` = 'left' / 'right' reads one column of a two-column page."""
    pg = _doc(str(path))[page - 1]
    words = pg.get_text("words")
    if half:
        mid = pg.rect.width * split
        words = [w for w in words if ((w[0] + w[2]) / 2 < mid) == (half == "left")]
    words.sort(key=lambda w: (round(w[1] / ytol), w[0]))
    rows: list[list] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if rows and abs(rows[-1][0][1] - w[1]) <= ytol:
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        toks = [unicodedata.normalize("NFKC", w[4]) for w in r]  # ligatures (fi, fl) -> letters
        # merge thousands split by spaces: '12 345' -> '12345' only when following token has 3 digits
        # only when the two words nearly touch (a thin-space separator), not two adjacent columns
        merged: list[str] = []
        prev_x1 = None
        for t, w in zip(toks, r, strict=False):
            gap = (w[0] - prev_x1) if prev_x1 is not None else 99
            if (merged and gap < 3.0 and re.fullmatch(r"\d{3}\)?", t)
                    and re.fullmatch(r"\(?-?\d{1,3}(,\d{3})*", merged[-1])):
                merged[-1] = merged[-1] + "," + t
            else:
                merged.append(t)
            prev_x1 = w[2]
        # a dash followed by a word is a bullet, not a nil amount
        merged = [t for i, t in enumerate(merged)
                  if not (t in {"-", "–", "—", "•"} and i + 1 < len(merged) and re.match(r"[A-Za-z]", merged[i + 1]))]
        nums, label = [], []
        for t in merged:
            if NUM.match(t) and (nums or not re.fullmatch(r"\d{1,2}(\.\d+)?", t) or label == []):
                nums.append(parse_number(t))
            elif nums:
                nums.append(None) if False else label.append(t)
            else:
                label.append(t)
        out.append(Row(page, " ".join(merged), " ".join(label), [n for n in nums if n is not None],
                       sum(w[1] for w in r) / len(r)))
    return merge_orphans(out)


def _is_blank_label(label: str) -> bool:
    return not re.search(r"[A-Za-z]{2,}", label)


def merge_orphans(rows: list[Row], gap: float = 9.0) -> list[Row]:
    """Join a number-only row to the nearest label-only row within ``gap`` points.

    OCR text layers (prompt 02 OCR task) often place a statement line's amounts a few points above or below its
    label, so the two land on different visual rows. Rows with both a label and numbers are never changed.
    """
    used: set[int] = set()
    out = list(rows)
    for i, r in enumerate(rows):
        if not r.numbers or not _is_blank_label(r.label):
            continue
        best, best_gap = None, gap + 1
        for j in (i - 1, i + 1):
            if 0 <= j < len(rows) and j not in used:
                c = rows[j]
                if not c.numbers and not _is_blank_label(c.label) and abs(c.y - r.y) <= gap:
                    if abs(c.y - r.y) < best_gap:
                        best, best_gap = j, abs(c.y - r.y)
        if best is not None:
            c = rows[best]
            used.add(best)
            out[i] = Row(r.page, f"{c.label} {r.text}", c.label, r.numbers, r.y)
            out[best] = Row(c.page, "", "", [], c.y)
    return [r for r in out if r.text or r.numbers]


def search(path: Path, pattern: str, pages: list[int] | None = None) -> list[Row]:
    """Rows whose label matches ``pattern`` (case-insensitive)."""
    rx = re.compile(pattern, re.I)
    pages = pages or text_pages(path)
    hits = []
    for p in pages:
        for r in page_rows(path, p):
            if rx.search(r.label):
                hits.append(r)
    return hits


UNIT_RX = re.compile(r"(TZS|Tshs?|Shs?)\s*['’‘]?\s*(000|million|m\b|mn\b|millions|billion)|(?:in|amounts in)\s+(thousands|millions)",
                     re.I)


def page_unit(path: Path, page: int) -> str | None:
    """'000' / 'million' unit stated on a page."""
    t = _doc(str(path))[page - 1].get_text()
    m = UNIT_RX.search(t)
    if not m:
        return None
    g = (m.group(2) or m.group(3) or "").lower()
    return "000" if g in {"000", "thousands"} else "million" if g.startswith("m") else g
