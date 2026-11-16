"""Layout check on the built PDFs (prompt 25, D96).

``python -m tzbank.reporting.layout_check [--report main|nmb|crdb|foreign|all]`` reads ``outputs/reports/<report>.pdf``
and its build log (the side-by-side blocks the build placed) and flags:

- ``side block``: a text column and its exhibit that end more than 1.5 cm apart, or a column that runs onto the next
  page;
- ``narrow panel``: a full-width exhibit panel whose chart is less than 85% of the panel's inner width;
- ``blank end``: a page ending with more than 6 cm of blank space although the next page's first element would have
  fitted in it (not before a new section, which starts on a fresh page by house style);
- ``empty page``: a body page with fewer than 3 lines of content;
- ``near-empty page``: a page less than about one-third full that is not the last page of a section.

Writes ``outputs/reports/<report>_layout_check.csv``. Page geometry: header and footer bands and the full-page
background art are ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from tzbank.reporting import house_docx as H

CM = 72 / 2.54  # points per cm
TOP = (H.MARGIN["top"] - 40) / 20  # body starts (points)
BOTTOM_MARGIN = H.MARGIN["bottom"] / 20
SECTION = re.compile(r"^(CHAPTER|PART|ANNEX|SECTION|ANNEXES|FINAL SYNTHESIS|PART [IV]+ SYNTHESIS|Disclosures|Contents)\b")


def _body_words(page) -> list[tuple]:
    h = page.rect.height
    return [w for w in page.get_text("words") if TOP <= w[1] <= h - BOTTOM_MARGIN]


def _content_bottom(page) -> float | None:
    h, wpage = page.rect.height, page.rect.width
    ys = [w[3] for w in _body_words(page)]
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        if (x1 - x0) > 0.9 * wpage and (y1 - y0) > 0.9 * h:  # full-page background art
            continue
        if TOP <= y0 <= h - BOTTOM_MARGIN:
            ys.append(min(y1, h - BOTTOM_MARGIN))
    for d in page.get_drawings():
        r = d["rect"]
        if TOP <= r.y0 <= h - BOTTOM_MARGIN and r.width > 30 and r.height > 10:
            ys.append(min(r.y1, h - BOTTOM_MARGIN))
    return max(ys) if ys else None


def _lines(page) -> int:
    return len({round(w[3]) for w in _body_words(page)})


def _starts_section(page) -> bool:
    words = sorted(_body_words(page), key=lambda w: (w[1], w[0]))
    head = " ".join(w[4] for w in words[:6])
    return bool(SECTION.match(head)) or any(  # a chapter / section banner: a tall full-width fill at the top
        d.get("fill") and min(d["fill"]) < 0.9 and d["rect"].width > 0.9 * (H.W_FULL / 20)  # coloured, not white
        and d["rect"].y0 < TOP + 60 and d["rect"].height > 35 for d in page.get_drawings())


def _first_element_h(page) -> float:
    """Height of the first element at the top of a page: the first drawn panel or the first text block."""
    cands = []
    rows = sorted((d["rect"] for d in page.get_drawings() if d["rect"].y0 >= TOP - 2 and d["rect"].width > 100
                   and d["rect"].height > 8), key=lambda r: r.y0)
    for r in rows:
        if r.height > 20:
            cands.append((r.y0, r.height))
    for b in page.get_text("blocks"):
        if b[1] >= TOP - 2 and b[1] <= page.rect.height - BOTTOM_MARGIN and b[4].strip():
            cands.append((b[1], b[3] - b[1]))
    first = min(cands) if cands else None
    if rows and (first is None or rows[0].y0 <= first[0] + first[1] + 12):
        # a table at the top (with its title line) is kept whole by house style: its height runs over its touching
        # row fills (prompt 26)
        y0, y1 = (first[0] if first else rows[0].y0), rows[0].y1
        for r in rows[1:]:
            if r.y0 > y1 + 3:
                break
            y1 = max(y1, r.y1)
        cands.append((y0 - 0.01, y1 - y0))
    return min(cands)[1] if cands else 0.0


def _find_seq(words: list[tuple], tokens: list[str]) -> tuple | None:
    """Last occurrence of a token sequence in reading order, joining hyphenated line ends; returns its last word."""
    norm = []
    k = 0
    while k < len(words):
        w = words[k]
        t = w[4]
        if t.endswith("-") and k + 1 < len(words):
            t, w = t[:-1] + words[k + 1][4], words[k + 1]
            k += 1
        norm.append((re.sub(r"[^\w%]", "", t).lower(), w))
        k += 1
    want = [re.sub(r"[^\w%]", "", t).lower() for t in tokens]
    want = [t for t in want if t]
    hit = None
    for i in range(len(norm) - len(want) + 1):
        if [x[0] for x in norm[i:i + len(want)]] == want:
            hit = norm[i + len(want) - 1][1]
    return hit


def check(pdf: Path, log: dict) -> pd.DataFrame:
    import pymupdf

    rows: list[dict] = []
    with pymupdf.open(pdf) as d:
        n = d.page_count
        # side blocks and full-width panels
        for sb in log.get("side_blocks", []):
            title = sb["title"][:60]
            pages = [i for i in range(n) if d[i].search_for(title[:40])]
            if not pages:
                rows.append({"page": None, "kind": "side block", "item": sb["exhibit"], "detail": "title not found"})
                continue
            pg = pages[0]
            page = d[pg]
            tb = page.search_for(title[:40])[0]
            panel = next((dr["rect"] for dr in sorted(page.get_drawings(), key=lambda x: x["rect"].height)
                          if dr["rect"].contains(tb) and dr["rect"].height > 40), None)
            if panel is None:
                continue
            if sb["layout"].startswith("full"):
                imgs = [im["bbox"] for im in page.get_image_info()
                        if panel.contains(pymupdf.Rect(im["bbox"]))]
                if imgs:
                    ratio = max(b[2] - b[0] for b in imgs) / (panel.width - 18)
                    if ratio < 0.85:
                        rows.append({"page": pg + 1, "kind": "narrow panel", "item": sb["exhibit"],
                                     "detail": f"chart {ratio:.0%} of the panel width"})
                continue
            split = panel.x0 - 4
            col = [w for w in sorted(_body_words(page), key=lambda w: (w[5], w[6], w[7]))
                   if w[2] <= split + 2 and w[1] >= panel.y0 - 30]
            mark = (sb["column"][-1] if sb["column"] else "").split()
            hit = _find_seq(col, mark[-3:]) if mark else None
            if hit is None:
                nxt = d[pg + 1] if pg + 1 < n else None
                cont = nxt is not None and _find_seq(_body_words(nxt), mark[-3:]) is not None
                rows.append({"page": pg + 1, "kind": "side block", "item": sb["exhibit"],
                             "detail": "text column runs onto the next page" if cont else "column end not found"})
                continue
            diff = (hit[3] - panel.y1) / CM
            sb["measured_diff_cm"] = round(diff, 2)  # for calibration (text column end minus exhibit end)
            sb["measured_text_cm"] = round((hit[3] - panel.y0) / CM, 2)
            sb["measured_exhibit_cm"] = round(panel.height / CM, 2)
            if abs(diff) > 1.5:
                rows.append({"page": pg + 1, "kind": "side block", "item": sb["exhibit"],
                             "detail": f"text column {'longer' if diff > 0 else 'shorter'} by {abs(diff):.1f} cm"})
        # pages
        for i in range(1, n):  # page 1 is the cover
            page = d[i]
            avail = page.rect.height - BOTTOM_MARGIN - TOP
            bottom = _content_bottom(page)
            nl = _lines(page)
            last = i == n - 1
            nxt_words = [] if last else sorted(_body_words(d[i + 1]), key=lambda w: (w[1], w[0]))
            nxt_head = " ".join(w[4] for w in nxt_words[:8])
            intended = any(m and m in nxt_head for m in log.get("page_breaks", []))  # an owner-requested break
            next_new = (not last) and (_starts_section(d[i + 1]) or intended)
            if nl < 3:
                rows.append({"page": i + 1, "kind": "empty page", "item": "", "detail": f"{nl} lines of content"})
                continue
            filled = ((bottom or TOP) - TOP) / avail
            free = page.rect.height - BOTTOM_MARGIN - (bottom or TOP)
            if filled < 1 / 3 and not next_new and not last:
                rows.append({"page": i + 1, "kind": "near-empty page", "item": "",
                             "detail": f"{filled:.0%} filled; the next page continues the section"})
            elif free > 6 * CM and not next_new and not last:
                fh = _first_element_h(d[i + 1])
                if fh and fh < free - 12:
                    rows.append({"page": i + 1, "kind": "blank end", "item": "",
                                 "detail": f"{free / CM:.1f} cm blank; the next page's first element "
                                           f"({fh / CM:.1f} cm) would fit"})
        for g in exhibit_gaps(d, log):  # prompt 26: a half-empty page before an exhibit the build could not fix
            rows.append({"page": g["page"], "kind": "half-empty page", "item": g["exhibit"],
                         "detail": f"{g['free_cm']:.1f} cm blank; the exhibit on the next page "
                                   f"({g['panel_cm']:.1f} cm) does not fit"})
    return pd.DataFrame(rows, columns=["page", "kind", "item", "detail"])


def exhibit_gaps(d, log: dict) -> list[dict]:
    """Pages that end with more than 6 cm blank because the next page opens with an exhibit too tall to fit (prompt 26,
    D97). Not before a new section (house style) or an intended page break. Returns, per page: the exhibit id (from
    the build log's exhibit numbers), the blank space and the exhibit panel's and chart's height and width (points),
    for the build's remedies: move the following text up, a side block, or a shorter chart."""
    import pymupdf

    numbers = {e.get("number"): e["id"] for e in log.get("exhibits", []) if e.get("number")}
    out = []
    n = d.page_count
    for i in range(1, n - 1):
        page, nxt = d[i], d[i + 1]
        bottom = _content_bottom(page)
        free = page.rect.height - BOTTOM_MARGIN - (bottom or TOP)
        if free <= 6 * CM or _starts_section(nxt) or nxt.rect.width > nxt.rect.height:  # landscape: own page
            continue
        head = " ".join(w[4] for w in sorted(_body_words(nxt), key=lambda w: (w[1], w[0]))[:8])
        if any(m and m in head for m in log.get("page_breaks", [])):
            continue
        labels = [w for w in _body_words(nxt) if w[4] == "Exhibit" and w[1] < TOP + 110]
        for w in sorted(labels, key=lambda w: w[1]):
            num = next((x[4] for x in _body_words(nxt) if abs(x[1] - w[1]) < 2 and x[0] > w[2] and x[0] - w[2] < 8),
                       None)
            if not (num and num.isdigit() and int(num) in numbers):
                continue
            lab = pymupdf.Rect(w[:4])
            panel = next((dr["rect"] for dr in sorted(nxt.get_drawings(), key=lambda x: x["rect"].height)
                          if dr["rect"].contains(lab) and dr["rect"].height > 40), None)
            if panel is None or panel.height <= free - 0.8 * CM:  # fits, with the spacing around a panel
                continue
            imgs = [pymupdf.Rect(im["bbox"]) for im in nxt.get_image_info() if panel.contains(pymupdf.Rect(im["bbox"]))]
            img = max(imgs, key=lambda r: r.height) if imgs else None
            out.append({"page": i + 1, "exhibit": numbers[int(num)], "free_cm": free / CM, "panel_cm": panel.height / CM,
                        "free_pt": free, "panel_pt": panel.height, "img_h_pt": img.height if img else 0.0,
                        "img_w_pt": img.width if img else 0.0})
            break
    return out


def main(argv: list[str] | None = None) -> int:
    from tzbank.config import find_root

    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.layout_check")
    ap.add_argument("--report", default="all", choices=["main", "nmb", "crdb", "foreign", "all"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = find_root() / "outputs" / "reports"
    total = 0
    for r in (["main", "nmb", "crdb", "foreign"] if a.report == "all" else [a.report]):
        pdf, log = out / f"{r}.pdf", out / f"{r}_build.json"
        if not pdf.exists():
            continue
        res = check(pdf, json.loads(log.read_text(encoding="utf-8")) if log.exists() else {})
        res.insert(0, "report", r)
        res.to_csv(out / f"{r}_layout_check.csv", index=False)
        total += len(res)
        print(r, len(res), res["kind"].value_counts().to_dict() if len(res) else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
