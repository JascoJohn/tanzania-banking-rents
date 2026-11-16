"""Independent cross-check of the inclusion panel's table values against the PDFs (prompt 02c section 4).

Verification only: the owner's Panel_Long values are never changed. For each captured observation from a printed
table or appendix, the cited PDF page is re-read from its text layer (pymupdf words grouped into rows). The
reported value is searched among the printed numbers on that page, and the row holding it is compared with the
observation's label. Statuses:

- ``match``: the value is printed in a row whose label shares words with the observation's breakdown or indicator;
- ``value on page, row label differs``: the value is printed on the page, but in a row the label test cannot tie
  to the observation (often a wrapped label or a table read across rows; check by hand);
- ``value in unlabelled row``: the value is printed only in rows with no text (a transposed table, or a label in a
  column header), so the row cannot be tied to the label automatically;
- ``value on adjacent page``: the value is printed on the page before or after (a page-reference slip);
- ``NOT FOUND``: no printed number on the page or its neighbours equals the value (a mismatch to review);
- ``no text layer``: the page has no extractable text.

A printed number equals the value when it agrees to the printed precision (half a unit in the last printed digit),
directly or as a percentage of a ratio, ignoring sign (negatives print in brackets or with a dash).
Not part of ``run all``: the source PDFs are git-ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.extract.captions import INCLUSION_FILES, NPS_FILES
from tzbank.io import write_csv

NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])-?\d+(?:\.\d+)?")
STOP = {"and", "the", "of", "in", "for", "to", "by", "total", "number", "value", "all", "with", "tzs", "billion",
        "million", "per", "cent", "percent"}


def numbers(text: str) -> list[tuple[float, float]]:
    """(value, tolerance) for each printed number; tolerance is half a unit in the last printed digit."""
    out = []
    for tok in NUM.findall(text):
        dec = len(tok.split(".")[1]) if "." in tok else 0
        try:
            out.append((float(tok.replace(",", "")), 0.5 * 10 ** -dec + 1e-12))
        except ValueError:
            continue
    return out


def equals(value: float, printed: tuple[float, float]) -> bool:
    v, tol = abs(printed[0]), printed[1]  # negatives print as (2.92), with a minus sign or an en dash
    value = abs(value)
    return abs(v - value) <= tol or abs(v - 100 * value) <= tol


def page_rows(page: pymupdf.Page, gap: float = 3.0) -> list[str]:
    """Text rows of a page: words grouped by vertical position (within ``gap`` points)."""
    words = sorted(page.get_text("words"), key=lambda w: (round(w[1]), w[0]))
    rows: list[list] = []
    for w in words:
        if rows and abs(rows[-1][0][1] - w[1]) <= gap:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [" ".join(x[4] for x in sorted(r, key=lambda x: x[0])) for r in rows]


def _join_thousands(row: str) -> str:
    """Add space-separated thousands as extra candidates: '1 215 033' also yields '1,215,033'.

    The original tokens are kept, so a row of separate small numbers ('957 969 990') still yields each of them.
    """
    # '277. 62' (a decimal broken by the text layer) is added as a candidate; '2,006. 5.51' stays two numbers
    extra = re.findall(r"\d+\. \d+(?![\d.])", row)
    extra = [e.replace(". ", ".") for e in extra]
    toks = row.split()
    for i, t in enumerate(toks):
        if not re.fullmatch(r"\d{1,3}", t):
            continue
        j, joined = i + 1, t
        while j < len(toks) and re.fullmatch(r"\d{3}(?:\.\d+)?", toks[j]):
            joined += "," + toks[j]
            extra.append(joined)
            j += 1
    return " ".join([row, *extra])


def label_words(*labels: object) -> set[str]:
    """Word stems (first five letters) of the labels, without stop words."""
    words = set()
    for lab in labels:
        words |= {w[:5] for w in re.findall(r"[a-z]+", str(lab or "").lower()) if len(w) > 2 and w not in STOP}
    return words


def label_matches(row: str, labels: set[str], indicator: str) -> bool:
    """The row's label shares a word stem with the observation, or is a total row for a total indicator."""
    if labels & label_words(row):
        return True
    return bool(re.match(r"\s*total\b", row, re.I)) and "total" in str(indicator).lower()


def check_obs(rows_by_page: dict[int, list[str]], page: int, value: float, labels: set[str],
              indicator: str = "") -> tuple[str, str]:
    rows = rows_by_page.get(page)
    if rows is None or not any(r.strip() for r in rows):
        return "no text layer", ""
    hits = [(i, r) for i, r in enumerate(rows) if any(equals(value, n) for n in numbers(_join_thousands(r)))]
    for i, r in hits:
        # a row that starts with a number may carry its label on the line above (wrapped label)
        context = f"{rows[i - 1]} {r}" if i and re.match(r"\s*-?\d", r) else r
        if label_matches(r, labels, indicator) or label_matches(context, labels, indicator):
            return "match", r[:160]
    if hits and all(not re.search(r"[A-Za-z]", r) for _, r in hits):
        return "value in unlabelled row", hits[0][1][:160]
    if hits:
        return "value on page, row label differs", hits[0][1][:160]
    for p in (page - 1, page + 1):
        for r in rows_by_page.get(p, []):
            if any(equals(value, n) for n in numbers(_join_thousands(r))):
                return "value on adjacent page", f"p{p}: {r[:150]}"
    return "NOT FOUND", ""


def crosscheck(root: Path | None = None) -> pd.DataFrame:
    """Every Table/Appendix observation in Panel_Long against its cited PDF page."""
    root = root or find_root()
    long = pd.read_csv(path("interim", root) / "inclusion_long.csv")
    tab = long[long["Evidence_Type"].isin(["Table", "Appendix"]) & long["PDF_Page"].notna()].copy()
    docs = path("documents", root) / "inclusion"
    out = []
    for rep, grp in tab.groupby("Source_Report"):
        f = docs / INCLUSION_FILES.get(int(rep), "")
        if int(rep) not in INCLUSION_FILES or not f.exists():
            for _, r in grp.iterrows():
                out.append(r.to_dict() | {"check_status": "PDF not held", "pdf_row_text": ""})
            continue
        doc = pymupdf.open(f)
        pages = {i + 1: page_rows(doc[i]) for i in range(doc.page_count)}
        doc.close()
        for _, r in grp.iterrows():
            st, txt = check_obs(pages, int(r["PDF_Page"]), float(r["Reported_Value"]),
                                label_words(r["Breakdown"], r["Indicator"]), str(r["Indicator"]))
            out.append(r.to_dict() | {"check_status": st, "pdf_row_text": txt})
    cols = ["Obs_ID", "Series_ID", "Indicator", "Breakdown", "Year", "Reported_Value", "Reported_Unit",
            "Source_Report", "Source_Table_or_Section", "PDF_Page", "check_status", "pdf_row_text"]
    df = pd.DataFrame(out)[cols]
    df["evidence_class"] = "Derived"
    return df


def _report_numbers(pdf: Path) -> list[tuple[float, float]]:
    doc = pymupdf.open(pdf)
    out = [n for pg in doc for r in page_rows(pg) for n in numbers(_join_thousands(r))]
    doc.close()
    return out


def _sig_digits(raw: str) -> int:
    return len(re.sub(r"\D", "", str(raw)).lstrip("0"))


def nps_crosscheck(root: Path | None = None) -> pd.DataFrame:
    """Every NPS panel value searched among the printed numbers of its (inferred) source report, then the others.

    The NPS workbook gives no page references, so the check is report-wide: weaker than the inclusion check. A value
    with fewer than three significant digits is marked ``weak`` because it matches by chance too easily.
    """
    root = root or find_root()
    pan = pd.read_csv(path("processed", root) / "nps_panel.csv")
    docs = path("documents", root) / "nps"
    nums = {r: _report_numbers(docs / f) for r, f in NPS_FILES.items() if (docs / f).exists()}
    rows = []
    for _, r in pan.iterrows():
        v = r["value"]
        if pd.isna(v):
            st = "not numeric"
        else:
            src = int(r["source_report"]) if pd.notna(r["source_report"]) else None
            found = [rep for rep, ns in nums.items() if any(equals(float(v), n) for n in ns)]
            if src in found:
                st = "found in source report"
            elif found:
                st = "found in another NPS report: " + ", ".join(str(x) for x in found)
            elif src is not None and src not in nums:
                st = "source report not held"
            else:
                st = "NOT FOUND"
        rows.append({"series_id": r["series_id"], "year": r["year"], "value": v, "value_raw": r["value_raw"],
                     "unit": r["unit"], "source_report": r["source_report"], "check_status": st,
                     "weak_match": _sig_digits(r["value_raw"]) < 3})
    df = pd.DataFrame(rows)
    df["panel"] = "nps"
    df["evidence_class"] = "Derived"
    return df


def write_crosscheck(root: Path | None = None) -> list[Path]:
    """reports/validation/inclusion_nps_crosscheck.csv (every non-match, both panels) and the full logs."""
    root = root or find_root()
    df = crosscheck(root)
    df.insert(0, "panel", "inclusion")
    nps = nps_crosscheck(root)
    v = path("validation", root)
    issues = pd.concat([df[df["check_status"] != "match"],
                        nps[nps["check_status"] != "found in source report"].rename(
                            columns={"series_id": "Series_ID", "year": "Year", "value": "Reported_Value",
                                     "unit": "Reported_Unit", "source_report": "Source_Report"})],
                       ignore_index=True)
    return [write_csv(issues, v / "inclusion_nps_crosscheck.csv", require_evidence=True),
            write_csv(df, v / "inclusion_crosscheck_log.csv", require_evidence=True),
            write_csv(nps, v / "nps_crosscheck_log.csv", require_evidence=True)]


if __name__ == "__main__":
    for p in write_crosscheck():
        print(p)
