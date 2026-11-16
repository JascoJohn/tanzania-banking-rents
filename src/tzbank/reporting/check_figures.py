"""Figure checker (prompt 12a): every number in a draft (.md, .docx, .pptx) against the findings pack.

``python -m tzbank.reporting.check_figures <file> [--pack outputs/findings/findings_pack_v1.0.csv] [--out report.csv]``

Each number is read with its printed precision and unit: ``57.3%`` (ratio 0.573 +/- 0.0005), ``+2.6 pp`` (0.026),
``TZS 16,517.6bn`` (1.65176e13 +/- 5e7), ``TZS 1.591tn``, ``-2.4 pts`` / ``points`` (as pp), ``11,062`` (11,062 +/- 0.5), ``0.27``.
A unitless number may also match a pack ratio printed in percent (|x| > 1, or after a percent figure in its sentence,
as in ``-0.74 points [-1.53, -0.11]``). It matches a pack figure when the pack
value, rounded to the printed precision, equals it (the tolerance is half a unit of the last printed digit). Status:
``matched`` (with the pack ids), ``mismatch`` (no match, but a pack value within 5% exists: likely a stale or
mistyped figure; the nearest id is shown), ``unmatched`` (nothing close; a figure not in the pack), ``ignored``
(years, dates, year spans such as 2015-18, tenors such as 364-day, document references such as D59, H9.1, FY2025, p.44, issue 18, and
``Source:`` caption lines), ``quotation`` (after "source quotation:") or ``external`` (prompt 13b pre-flight A: a
figure from an outside source, never matched to the pack: an ``[ext]`` / ``[ext: <source>]`` tag or a cited
``(source: ...)`` in its sentence, or a context rule in ``config/settings.yaml`` ``checker.external_rules``). A
matched figure none of whose pack rows is shown on its dashboard page is flagged ``not_on_dashboard`` (report subset
of dashboard, owner 12a amendment).
Word and PowerPoint files are read as zipped XML (no extra dependency).
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root

NUM = re.compile(
    r"(?<![\w.#/,])(?<![A-Za-z]-)(?<![A-Za-z]–)(?P<pre>TZS\s*|USD\s*)?(?P<sign>[+\-−–])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:e[+\-]?\d+)?)"
    r"\s*(?P<unit>%|pp\b|points?\b|pts?\b|bp\b|tn\b|trillion\b|bn\b|billion\b|m\b|million\b|x\b)?")
REF = re.compile(r"(D|H|FY|p\.|pp\.|v|#|issue |Issue |Table |table |chapter |Chapter |step |§|MA-|WC-|GN |G\.N\. No\. |"
                 r"No\. |auction |prompt |Prompt |Figure |CH)\s*$")
SCALE = {"tn": 1e12, "trillion": 1e12, "bn": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6}
EXT_TAG = re.compile(r"\[ext(?::[^\]]*)?\]|\((?:source|Source|SOURCE):[^)]*\)")
SENT_END = re.compile(r"(?<=[.;!?])\s+(?=[A-Z\[(])")


def external_rules(root: Path | None = None) -> list[re.Pattern]:
    """Context rules for external figures (``checker.external_rules`` in config/settings.yaml)."""
    from tzbank.config import settings

    try:
        rules = (settings(root or find_root()).get("checker") or {}).get("external_rules") or []
    except Exception:  # noqa: BLE001 - checker works without a project config
        rules = []
    return [re.compile(r) for r in rules]


def _spans(line: str) -> list[tuple[int, int]]:
    """Sentence spans of a line."""
    cuts = [0] + [m.end() for m in SENT_END.finditer(line)] + [len(line)]
    return list(zip(cuts[:-1], cuts[1:], strict=True))


# Units printed in a table header (prompt 15b item 4): applied to bare numbers in that column (or row) before matching
HEADER_UNITS = [(re.compile(r"\bTZS\s*(?:tn|trillion)\b|\((?:tn|trillion)\)", re.I), " tn"),
                (re.compile(r"\bTZS\s*(?:bn|billion)\b|\((?:bn|billion)\)|\bbn\b", re.I), " bn"),
                (re.compile(r"\bTZS\s*(?:m|mn|million)\b|\((?:m|mn|million)\)", re.I), " m"),
                (re.compile(r"\bpp\b|percentage points", re.I), " pp"),
                (re.compile(r"(?<![\d.])%|per ?cent", re.I), "%"),  # a unit, not a figure such as 12.7%
                (re.compile(r"×|\(x\)|\btimes\b", re.I), "")]  # multiples are plain numbers in the pack
CONTACT = re.compile(r"https?://\S+|\+\d{1,3}(?:[ \u00a0]\d{2,4}){2,4}|ORCID\s*:?\s*\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
BARE = re.compile(r"\s*[+\-−–]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*")


def header_unit(text: str) -> str | None:
    """The unit a table header cell states ('TZS bn', '%', 'x'), or None."""
    for rx, u in HEADER_UNITS:
        if rx.search(text):
            return u
    return None


def _cell_text(xml: str) -> str:
    paras = re.findall(r"<w:p[ >].*?</w:p>|<w:p/>", xml, re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", q)).strip() for q in paras).strip()


def _table_lines(tbl: str) -> str:
    """A Word table as text lines, one per row ('a | b | c'). A bare number takes its column header's unit (first
    row), failing that its row label's unit, failing that the unit in the top-left header cell (a table-wide unit,
    as in 'TZS bn | FY2024A | ...'); a leading en dash in such a cell is read as a minus."""
    rows = [[_cell_text(c) for c in re.findall(r"<w:tc[ >].*?</w:tc>", r, re.S)]
            for r in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)]
    if not rows:
        return ""
    col_u = [header_unit(h) for h in rows[0]]
    stub_u = col_u[0] if col_u else None  # a unit in the top-left cell ("TZS bn") applies to the whole table
    out = [" | ".join(rows[0])]
    for r in rows[1:]:
        row_u = header_unit(r[0]) if r else None
        cells = []
        for j, c in enumerate(r):
            u = (col_u[j] if j < len(col_u) else None) or (row_u if j > 0 else None) or (stub_u if j > 0 else None)
            neg = re.fullmatch(r"\s*\(\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\)\s*", c)
            if j > 0 and neg:  # accounting brackets: (747) is -747
                c = f"−{neg.group(1)}"
            if j > 0 and u is not None and BARE.fullmatch(c):
                c = re.sub(r"^\s*–", "−", c).strip() + u
            cells.append(c)
        out.append(" | ".join(cells))
    return "\n".join(out)


TBL_TOK = re.compile(r"<w:tbl>|</w:tbl>|<w:tblPr>.*?</w:tblPr>|<w:tblGrid>.*?</w:tblGrid>|<w:trPr>.*?</w:trPr>|"
                     r"<w:tcPr>.*?</w:tcPr>|<w:tr(?: [^>]*)?>|</w:tr>|<w:tc>|</w:tc>", re.S)


def _unwrap_layout(xml: str) -> str:
    """Remove the table structure of layout tables (``tblCaption = layout``: banners, boxes, exhibit panels and
    columns of the designed reports, prompt 16) so their text reads as prose; data tables inside them are kept."""
    if 'w:tblCaption w:val="layout"' not in xml:
        return xml
    out, pos, stack = [], 0, []
    for m in TBL_TOK.finditer(xml):
        out.append(xml[pos:m.start()])
        pos = m.end()
        tok = m.group(0)
        if tok == "<w:tbl>":
            head = xml[m.end():xml.find("</w:tblPr>", m.end())]
            stack.append('w:tblCaption w:val="layout"' in head)
            if not stack[-1]:
                out.append(tok)
        elif tok == "</w:tbl>":
            if not (stack.pop() if stack else False):
                out.append(tok)
        elif not (stack and stack[-1]):
            out.append(tok)
    out.append(xml[pos:])
    return "".join(out)


def read_text(f: Path) -> str:
    """Plain text of a .md / .txt, .docx (word/document.xml; tables row by row with header units) or .pptx
    (ppt/slides/*.xml)."""
    if f.suffix.lower() in (".docx", ".pptx"):
        with zipfile.ZipFile(f) as z:
            names = (["word/document.xml"] if f.suffix.lower() == ".docx" else
                     sorted(n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)))
            xml = " ".join(z.read(n).decode("utf-8", "ignore") for n in names)
        tables: list[str] = []

        def keep(m: re.Match) -> str:
            tables.append(_table_lines(m.group(0)))
            return f"</w:p>@@TABLE{len(tables) - 1}@@</w:p>"

        xml = _unwrap_layout(xml)
        xml = re.sub(r"<w:tbl>.*?</w:tbl>", keep, xml, flags=re.S)
        xml = re.sub(r"</w:p>|</a:p>", "\n", xml)
        text = html.unescape(re.sub(r"<[^>]+>", "", xml))  # &amp; &lt; &gt; &apos; &quot; and numeric entities
        return re.sub(r"@@TABLE(\d+)@@", lambda m: tables[int(m.group(1))], text)
    return f.read_text(encoding="utf-8")


def numbers(text: str, rules: list[re.Pattern] | None = None) -> pd.DataFrame:
    rules = rules or []
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        q = line.lower().find("source quotation:")  # figures quoted from a source document (owner, 12a review C)
        urls = [(u.start(), u.end()) for u in CONTACT.finditer(line)]  # URLs, phone numbers, ORCID ids
        caption = bool(re.match(r"\s*Sources?:", line))  # "Source: ..." caption lines cite documents, not figures
        ext_sent = [(a, b) for a, b in _spans(line) if EXT_TAG.search(line[a:b])]
        pct_seen: set[int] = set()  # sentences (by start) that already printed a %, pp or points figure
        for m in NUM.finditer(line):
            raw = m.group("num")
            before = line[max(0, m.start() - 12):m.start()]
            after = line[m.end():m.end() + 1]
            val = float(raw.replace(",", ""))
            mant, _, ex = raw.partition("e")
            dec = (len(mant.split(".")[1]) if "." in mant else 0) - (int(ex) if ex else 0)
            unit = (m.group("unit") or "").lower()
            sign = -1.0 if m.group("sign") in ("-", "−", "–") and not re.search(r"\w$", before) else 1.0
            is_year = dec == 0 and "," not in raw and 1990 <= val <= 2035 and not unit
            # dates (2025-12-31, 31/12/2025) and year spans (2015-18); a range such as 8,038-14,433 is two figures
            four = dec == 0 and "," not in raw and len(raw) == 4
            nxt = line[m.end():m.end() + 2]
            span_end = dec == 0 and len(raw) <= 2 and bool(re.search(r"(19|20)\d\d[-–]$", before))  # 2015–18
            date_like = (bool(re.search(r"(^|[^\d,.])\d{4}-(\d{1,2}-)?$", before)) or bool(re.search(r"\d/$", before))
                         or bool(re.match(r"/\d", nxt)) or (four and bool(re.match(r"-\d", nxt))))
            tenor = bool(re.match(r"[-\u2011]\s?(day|week|month|year)s?\b", line[m.end():m.end() + 8]))  # 364-day
            ref = bool(REF.search(before)) or after in ("_",) or date_like or span_end or caption or tenor
            is_id = "," not in raw and "." not in raw and len(raw) >= 8  # file ids, seeds, undelimited dates
            in_url = any(a <= m.start() < b for a, b in urls)
            s0 = max([a for a, _ in _spans(line) if a <= m.start()] or [0])  # rules see the sentence up to the number
            ctx = line[s0:m.start()] + "{N}" + line[m.end():m.end() + 30]
            external = any(a <= m.start() < b for a, b in ext_sent) or any(r.search(ctx) for r in rules)
            status = ("quotation" if 0 <= q < m.start() else "ignored" if is_year or is_id or ref or in_url
                      else "external" if external else "")
            s_pct = s0 in pct_seen  # a unitless number after a percent figure in its sentence (intervals, lists)
            if unit in ("%", "pp", "point", "points", "pt", "pts"):
                pct_seen.add(s0)
            if unit in ("%", "pp", "point", "points", "pt", "pts"):
                v, tol, kind = sign * val / 100, 0.5 * 10 ** -dec / 100, "ratio"
            elif unit == "bp":
                v, tol, kind = sign * val / 1e4, 0.5 * 10 ** -dec / 1e4, "ratio"
            elif unit in SCALE:
                v, tol, kind = sign * val * SCALE[unit], 0.5 * 10 ** -dec * SCALE[unit], "level"
            else:
                v, tol, kind = sign * val, 0.5 * 10 ** -dec, "plain"
            rows.append({"line": ln, "text": m.group(0).strip(), "value": v, "tolerance": tol, "kind": kind,
                         "context": line[max(0, m.start() - 50):m.end() + 30].strip(), "status": status,
                         "pct_context": s_pct and kind == "plain"})
    return pd.DataFrame(rows, columns=["line", "text", "value", "tolerance", "kind", "context", "status", "pct_context"])


def check(doc: Path, pack: Path | None = None, root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    from tzbank.reporting.findings import latest_pack

    pack = pack or latest_pack(root)
    p = pd.read_csv(pack, low_memory=False)
    p = p[~p["unit"].isin(["text", "verdict"])].copy()
    p["v"] = pd.to_numeric(p["value"], errors="coerce")
    p = p.dropna(subset=["v"])
    vals, ids, shown = p["v"].to_numpy(), p["id"].to_numpy(), p["dashboard_shown"].astype(bool).to_numpy()
    units = p["unit"].to_numpy()
    # a value labelled TZS but no larger than 5 sits in a mixed-measure column (a return next to prices): a ratio
    ratio_like = (units == "ratio") | ((units == "TZS") & (np.abs(vals) <= 5))
    n = numbers(read_text(doc), external_rules(root))
    out = []
    for r in n.to_dict("records"):
        if r["status"] in ("ignored", "quotation", "external"):
            out.append({**r, "pack_ids": "", "not_on_dashboard": False})
            continue
        cand = vals if r["kind"] != "ratio" else np.where(ratio_like, vals, np.nan)
        hit = np.abs(cand - r["value"]) <= r["tolerance"] + 1e-12
        if r["kind"] == "plain" and not hit.any():  # a plain number may be a pack ratio printed in percent units
            pct_ok = abs(r["value"]) > 1 or r["pct_context"]  # e.g. -10.5 (points), or [-1.87, -0.50] after "points"
            hit = ratio_like & (np.abs(vals * 100 - r["value"]) <= r["tolerance"] + 1e-12) & pct_ok
        if hit.any():
            out.append({**r, "status": "matched", "pack_ids": ";".join(ids[hit][:5]) + (" ..." if hit.sum() > 5 else ""),
                        "not_on_dashboard": not shown[hit].any()})
            continue
        rel = np.abs(cand - r["value"]) / max(abs(r["value"]), 1e-12)
        j = int(np.nanargmin(rel)) if np.isfinite(rel).any() else -1
        close = j >= 0 and rel[j] <= 0.05
        out.append({**r, "status": "mismatch" if close else "unmatched",
                    "pack_ids": f"nearest {ids[j]} = {vals[j]:.6g}" if close else "", "not_on_dashboard": False})
    return pd.DataFrame(out)


def summary(res: pd.DataFrame) -> str:
    c = res["status"].value_counts()
    checked = len(res) - int(c.get("ignored", 0)) - int(c.get("quotation", 0)) - int(c.get("external", 0))
    return (f"{len(res)} numbers: {checked} checked, {int(c.get('matched', 0))} matched, "
            f"{int(c.get('mismatch', 0))} mismatch, {int(c.get('unmatched', 0))} unmatched, "
            f"{int(c.get('ignored', 0))} ignored (years, dates, references, URLs), "
            f"{int(c.get('quotation', 0))} labelled source quotations, {int(c.get('external', 0))} external; "
            f"{int(res['not_on_dashboard'].sum())} matched but not on the dashboard")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.check_figures",
                                 description="Check every number in a draft against the findings pack.")
    ap.add_argument("file")
    ap.add_argument("--pack")
    ap.add_argument("--out", help="write the per-number report to this CSV")
    a = ap.parse_args(argv)
    res = check(Path(a.file), Path(a.pack) if a.pack else None)
    if a.out:
        res.to_csv(a.out, index=False)
    sys.stdout.reconfigure(encoding="utf-8")
    print(summary(res))
    bad = res[res["status"].isin(["mismatch", "unmatched"]) | res["not_on_dashboard"]]
    if len(bad):
        print(bad[["line", "text", "status", "pack_ids", "context"]].to_string(index=False, max_colwidth=70))
    return 1 if (res["status"] == "mismatch").any() or res["not_on_dashboard"].any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
