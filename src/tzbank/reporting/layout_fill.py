"""Height estimates for the side-by-side fill rule (prompt 25, D96).

A side-by-side block is a text column beside an exhibit. The two columns should end at about the same height (within
``TOL`` twips, about 1.5 cm). The builder asks this module how tall each candidate element would render at the
column width, and takes the contiguous run of following elements whose total is closest to the exhibit's height:
text shorter than the exhibit pulls in the next paragraph, list, box or small table; text longer than the exhibit
leaves the excess to continue full width below. Heights are estimates from Calibri's glyph widths (the Windows font
file, via PIL) and the house spacing in ``house_docx``; the PDF layout check (``layout_check``) measures the result.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

TOL = 850  # twips: 1.5 cm
LINE = 1.15 * 1.22  # house line spacing x Calibri line height (1.22 em): 14.0 pt at 10 pt, measured in the PDF


@lru_cache(maxsize=8)
def _font(bold: bool = False, italic: bool = False):
    from PIL import ImageFont

    name = {(False, False): "calibri.ttf", (True, False): "calibrib.ttf", (False, True): "calibrii.ttf",
            (True, True): "calibriz.ttf"}[(bold, italic)]
    for d in (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/usr/share/fonts/truetype/msttcorefonts")):
        if (d / name).exists():
            return ImageFont.truetype(str(d / name), 100)  # 100 px = 100 pt units; widths scaled by size / 100
    return None


def _plain(text: str) -> str:
    return re.sub(r"\*\*|(?<!\\)\*|\{\{[^}]*\}\}|\[\[[^\]]*\]\]|<!--.*?-->", "", text).replace("\\*", "*")


def lines(text: str, width_twips: float, size: float, bold: bool = False) -> int:
    """Lines a text wraps to at ``width_twips`` and ``size`` pt: greedy word wrap with a simple model of Word's
    automatic hyphenation (a word of 7+ letters that does not fit breaks when at least ~40% of it fits on the line,
    as Word's hyphenation zone allows); checked against the PDF line counts (prompt 25)."""
    words = _plain(text).split()
    if not words:
        return 1
    f = _font(bold)
    width_pt = width_twips / 20
    if f is None:  # no Calibri: about 0.5 em per character
        return max(1, -(-int(len(" ".join(words)) * size * 0.5) // max(int(width_pt), 1)))
    space = f.getlength(" ") * size / 100
    hyph = f.getlength("-") * size / 100
    n, cur = 1, 0.0
    for w in words:
        lw = f.getlength(w) * size / 100
        if cur and cur + space + lw > width_pt:
            room = width_pt - cur - space - hyph
            if sum(c.isalpha() for c in w) >= 7 and room >= 0.4 * lw:  # hyphenated: part stays, rest wraps
                n, cur = n + 1, lw - room
            else:
                n, cur = n + 1, lw
        else:
            cur = cur + space + lw if cur else lw
    return n


def text_h(text: str, width: float, size: float = 10, after: float = 6, before: float = 0, bold: bool = False) -> float:
    return lines(text, width, size, bold) * size * LINE * 20 + (after + before) * 20


def node_h(n, width: float, size: float = 10) -> float:
    """Rendered height (twips) of one source node at ``width``: para, list, h3, block (house box) or md table."""
    if n.kind == "para":
        return text_h(n.text, width, size, 6)
    if n.kind == "h3":
        return text_h(n.text, width, 10.5, 4, 6, bold=True)
    if n.kind == "list":
        return sum(text_h(it, width - 300, size, 4) for it in n.items)
    if n.kind == "block":
        inner = width - 440
        h = 140 + 120 + (8.5 * LINE * 20 + 60)  # padding, title line
        size_in = 9 if n.label == "method" else 9.5
        for idx, k in enumerate(n.children):
            if idx == 0 and k.kind == "para" and k.text.startswith("**"):  # the bold lead is the title line
                rest = k.text.split("**", 2)[2].strip() if k.text.count("**") >= 2 else k.text
                h += text_h(rest, inner, size_in, 4) if rest else 0
                continue
            h += node_h(k, inner, size_in) - (2 * 20 if k.kind == "para" else 0)
        return h + 160  # spacer after the box
    if n.kind == "table":
        from tzbank.reporting import house_docx as H

        hdr, rows = n.items[0], n.items[1:]
        ws = H.col_widths(hdr, rows, int(width))
        h = 0.0
        for r in [hdr, *rows]:
            cells = zip(r, ws, strict=False)
            h += max(lines(c, w - 160, 8) for c, w in cells) * 8 * LINE * 20 + 70  # cell padding, rules (PDF)
        return h + 120
    return 0.0


def exhibit_h(title: str, source: str, png: Path, width: float, numbered: bool = True) -> float:
    """Height (twips) of an exhibit panel: label, title, chart at the panel's inner width, source line, padding."""
    from PIL import Image

    inner = width - 360
    with Image.open(png) as im:
        img = inner * im.size[1] / im.size[0]
    h = 140 + 120 + img + 60
    if numbered:
        h += 8 * LINE * 20 + 20
    h += text_h(title, inner, 9.5, 4, bold=True)
    h += lines(source, inner, 7) * 7 * LINE * 20
    return h  # the panel's own height: what the layout check measures


def trail(n) -> float:
    """Space below an element's last line of text (twips): not part of the column height the layout check measures
    (to the last word)."""
    return {"para": 120, "list": 80, "h3": 80, "block": 120 + 160 + 80, "table": 240}.get(n.kind, 0.0)


def fits_column(n, width: float) -> bool:
    """Can this node sit in a text column? Paragraphs, lists, owner boxes (house blocks) and tables of up to three
    columns. Equations, exhibits, headings above level 3 and page breaks cannot."""
    if n.kind in ("para", "list"):
        return not n.text.strip().startswith("$$")
    if n.kind == "block":
        return n.label in ("method", "case", "finding", "limitation")
    if n.kind == "table":
        return len(n.items[0]) <= 3 and width >= 4000
    return False


def choose(nodes: list, i: int, target: float, width: float) -> tuple[int, float]:
    """The end index j (exclusive) of the contiguous run nodes[i+1:j] to set beside an exhibit of height ``target``.
    Among the runs that end within TOL of the exhibit, the longest (so a following owner box is pulled in); if none
    does, the run closest to it, counting the first run that passes it (text longer than the exhibit leaves the rest
    to continue full width below). An optional leading h3 belongs to the run. Returns (j, h); a run with no body
    element (nothing, or only a heading) means nothing suitable, and the caller lays the exhibit out full width."""
    j, h = i + 1, 0.0
    if j < len(nodes) and nodes[j].kind == "h3":
        h += node_h(nodes[j], width)
        j += 1
    start = j
    runs = []
    while j < len(nodes) and fits_column(nodes[j], width):
        h, j = h + node_h(nodes[j], width), j + 1
        runs.append((j, h - trail(nodes[j - 1])))  # to the last line of text, as measured
        if h > target + TOL:  # the first run that passes the exhibit is the last candidate
            break
    if not runs:
        return start, 0.0 if start == i + 1 else h
    within = [r for r in runs if abs(target - r[1]) <= TOL]
    j, h = max(within) if within else min(runs, key=lambda r: abs(target - r[1]))
    if h - target > 2 * TOL:  # far longer than the exhibit (a paragraph cannot be split): full width instead
        return start, 0.0
    return j, h


def choose_lead(nodes: list, i: int, target: float, width: float) -> tuple[int, int, float]:
    """``choose`` with one more option: the opening paragraph (with its ### heading, if any) printed full width above
    the block, and the elements after it set beside the exhibit. Used when it balances better - typically an owner box
    that fits beside the exhibit on its own but not together with the paragraph before it. Returns (lead_end, j, h):
    nodes[i+1:lead_end] go full width above the block, nodes[lead_end:j] in the text column."""
    j0, h0 = choose(nodes, i, target, width)
    best = (i + 1, j0, h0, abs(target - h0) if j0 > i + 1 else float("inf"))
    k = i + 1
    if k < len(nodes) and nodes[k].kind == "h3":
        k += 1
    if k < len(nodes) and nodes[k].kind == "para" and k + 1 < len(nodes) and fits_column(nodes[k + 1], width):
        j1, h1 = choose(nodes, k, target, width)  # the run after the opening paragraph
        if j1 > k + 1 and abs(target - h1) < best[3] - 200:
            best = (k + 1, j1, h1, abs(target - h1))
    return best[0], best[1], best[2]


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
ABBREV = ("e.g.", "i.e.", "No.", "pp.", "p.", "para.", "Fig.", "cf.", "vs.", "al.", "etc.")


def sentences(text: str) -> list[str]:
    """A paragraph split into sentences, never after an abbreviation or an initial ('C. E.')."""
    out: list[str] = []
    for part in SENT.split(text):
        if out and (out[-1].endswith(ABBREV) or re.search(r"(^|\s)[A-Z]\.$", out[-1])):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _with(n, text: str):
    from dataclasses import replace

    return replace(n, text=text, raw=[text])


def plan(nodes: list, i: int, target: float, width: float, full: float) -> dict:
    """Side-block plan for the exhibit at nodes[i] (rule 1, prompt 25): what prints full width above the block,
    what sits in the text column, what continues full width below it, and where the main flow resumes.

    Candidates: an optional ### heading in the column; the opening paragraph whole in the column, whole above it, or
    split at a sentence (first sentences above, the rest beside); then following paragraphs, lists, boxes and small
    tables; the run's last paragraph may be split at a sentence (first sentences beside, the rest below: the excess
    text moves out of the block). Every split part keeps at least two lines. The closest column height to the
    exhibit's wins; a run containing an owner box wins a near tie (boxes fill the space beside exhibits). Returns
    {"above", "column", "below", "next", "h"}; an empty column means nothing suitable (lay the exhibit out full width).
    """
    def h_of(seq):
        h = sum(node_h(x, width) for x in seq)
        return h - trail(seq[-1]) if seq else 0.0

    def ok(text, w, size=10):
        return lines(text, w, size) >= 2

    head = [nodes[i + 1]] if i + 1 < len(nodes) and nodes[i + 1].kind == "h3" else []
    k = i + 1 + len(head)
    leads = [([], [], k)]  # (above, column prefix, index of the next node)
    if k < len(nodes) and nodes[k].kind == "para" and fits_column(nodes[k], width):
        first = nodes[k]
        leads.append((head + [first], [], k + 1))  # opening paragraph above, whole
        ss = sentences(first.text)
        for c in range(1, len(ss)):
            a, b = " ".join(ss[:c]), " ".join(ss[c:])
            if ok(a, full) and ok(b, width):
                leads.append((head + [_with(first, a)], [_with(first, b)], k + 1))
    best = None
    for above, prefix, start in leads:
        col_head = [] if above else head
        seq = col_head + prefix
        j = start
        cands = []
        if prefix:
            cands.append((list(seq), [], j))
        while j < len(nodes) and fits_column(nodes[j], width):
            n = nodes[j]
            if n.kind == "para":  # the run may end with this paragraph split: first sentences beside, rest below
                ss = sentences(n.text)
                for c in range(1, len(ss)):
                    a, b = " ".join(ss[:c]), " ".join(ss[c:])
                    if ok(a, width) and ok(b, full):
                        cands.append((seq + [_with(n, a)], [_with(n, b)], j + 1))
            seq = seq + [n]
            j += 1
            cands.append((list(seq), [], j))
            if h_of(seq) > target + 2 * TOL:
                break
        for col, below, nxt in cands:
            if not any(x.kind != "h3" for x in col):
                continue
            h = h_of(col)
            boxes = sum(x.kind == "block" for x in col)
            score = abs(target - h) - 300 * boxes - (100 if not above else 0)  # owner boxes beside; fewer moves
            if best is None or score < best[0]:
                best = (score, {"above": above, "column": col, "below": below, "next": nxt, "h": h})
    if best is None or abs(target - best[1]["h"]) > 2 * TOL:
        return {"above": [], "column": [], "below": [], "next": i + 1, "h": 0.0}
    return best[1]
