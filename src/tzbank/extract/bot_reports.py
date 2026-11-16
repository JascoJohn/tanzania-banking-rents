"""BoT Annual Report tables: inventory and code-extracted supplement (prompt 02f section 2; decisions D39, D40;
blueprint Â§6 data audit: rates, sector totals; chapters 1-3, 4.2, 6).

``inventory`` lists every captioned table in the held BoT Annual Reports (``data/raw/documents/reports/``), the
report's financial year and whether the owner's research database (D40) holds rows from that report and table.
``extract`` transcribes the priority chapter tables the database does not hold (selected interest rates, treasury
bond yields by tenor, licensed banks, selected financial soundness indicators) as a D39 supplement below the owner's
precedence. Gates:

- every value is found in the text of its page;
- totals tie (Licensed Banks: types sum to the total; domestic + foreign = listed + not listed = number);
- the lending-deposit margin row equals one-year lending minus one-year deposit rate (0.01 pp);
- a value the owner's database already holds for the same period (within 0.5%) is marked "already held" and used only
  as a cross-check.

Outputs: ``data/interim/bot_code_extracted.csv`` and ``reports/bot_report_inventory.csv`` (needs the PDFs; runs in the
``pdf`` stage, cached by input hash).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.pdftext import _doc, page_rows
from tzbank.io import write_csv

TARGETS = r"(Selected (Bank.s )?Interest Rates|Interest Rates? Structure|Treasury Bonds? Yields|Bond Performance by " \
          r"Tenure|Licensed Banks|Financial Soundness)"
PERIOD = r"(?:(?:19|20)\d\d/\d\d|(?:19|20)\d\d(?:p|\*)?|[A-Z][a-z]{2}-\d\d)"
NUM = r"-?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?"
CAPTION = re.compile(r"Table\s+(A?\d+\.\d+[a-z]?)\s*[:.\-]?\s*(.{0,90})")
COLUMNS = ["report_fy", "file", "table", "title", "pdf_page", "row_label", "period", "value", "unit", "gate",
           "gate_result", "db_match", "used", "extraction_method", "evidence_class"]


def report_dir(root: Path) -> Path:
    return path("documents", root) / "reports"


BOT_NAME = re.compile(r"^(V2)?AnnRep - \d{16}\.pdf$")


def bot_report_files(root: Path) -> list[Path]:
    """The BoT annual and supervision reports in documents/reports, by their BoT file names ('AnnRep - <id>.pdf',
    'V2AnnRep - <id>.pdf'). Other documents filed there (prompt 24 context-box sources, a duplicate copy) are not
    BoT tables and are skipped."""
    return sorted(f for f in report_dir(root).glob("*.pdf") if BOT_NAME.match(f.name))


def report_fy(root: Path) -> dict[str, str]:
    """File-name document ID -> BoT financial year, from the two workbooks' Source_Register sheets."""
    from tzbank.ingest.bot_database import V1, V2

    out = {}
    raw = path("raw", root)
    for wb, col, fy in [(V1, "Original_File_Name", "Publication_Title"), (V2, "Original file name",
                                                                           "Financial reporting period")]:
        if not (raw / wb).exists():
            continue
        s = pd.read_excel(raw / wb, sheet_name="Source_Register")
        for name, t in zip(s[col], s[fy], strict=False):
            m = re.search(r"(\d{16})", str(name))
            y = re.search(r"(\d{4}/\d{2})", str(t))
            if m and y:
                out[m.group(1)] = y.group(1)
    return out


def fy_of(file: Path, fys: dict[str, str]) -> str:
    m = re.search(r"(\d{16})", file.name)
    return fys.get(m.group(1), "unknown") if m else "unknown"


def inventory(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    fys = report_fy(root)
    db_f = path("interim", root) / "bot_research_database.csv"
    db = pd.read_csv(db_f, usecols=["source_document", "page_table"]) if db_f.exists() else pd.DataFrame(
        columns=["source_document", "page_table"])
    held = set()
    for sd, pt in zip(db["source_document"], db["page_table"], strict=False):
        m = re.search(r"(\d{16})", str(sd))
        t = re.search(r"table (\S+)", str(pt))
        if m and t:
            held.add((m.group(1), t.group(1).strip(",").lstrip("A")))
    rows = []
    for f in bot_report_files(root):
        d = _doc(str(f))
        fid = re.search(r"(\d{16})", f.name).group(1)
        seen = set()
        text_pages = 0
        for i in range(d.page_count):
            t = d[i].get_text()
            text_pages += len(t) > 200
            for line in t.split("\n"):
                m = CAPTION.match(line.strip())
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    num = m.group(1)
                    rows.append({"report_fy": fy_of(f, fys), "file": f.name, "table": num,
                                 "title": m.group(2).strip(), "pdf_page": i + 1,
                                 "appendix": num.startswith("A"),
                                 "in_owner_database": (fid, num.lstrip("A")) in held,
                                 "priority_target": bool(re.search(TARGETS, m.group(2), re.I)),
                                 "text_pages": text_pages})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Observed"
    return out


def _f(tok: str) -> float:
    t = tok.replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    return -float(t.strip("()")) if neg else float(t)


LB_HEADER = re.compile(r"Number\s+Domestic\s+Foreign\s+Listed\s+Not\s*Listed", re.I)


def _parse_rows(rows, start: int) -> tuple[list[str], list[tuple[str, list[str]]]]:
    periods: list[str] = []
    recs, pending = [], []
    for r in rows[start: start + 45]:
        if re.match(r"^(Source|Note|Table\s)", r.text):
            break
        if not periods and LB_HEADER.search(r.text):
            periods = ["Number", "Domestic", "Foreign", "Listed", "Not listed"]
            continue
        toks = re.findall(PERIOD, r.text)
        if not periods and len(toks) >= 2 and len(re.sub(PERIOD, "", r.text).strip()) < 40:
            periods = toks
            continue
        if not periods:
            continue
        m = re.match(rf"^(.*?)((?:\s*{NUM}){{{len(periods)}}})\s*$", r.text)
        if not m:
            if r.text and not re.search(r"\d", r.text):
                pending.append(r.text.strip())
            continue
        lab = m.group(1).strip()
        if pending and (not lab or lab[0].islower()) and len(pending[-1]) <= 60:
            lab = (pending[-1] + " " + lab).strip()
        pending = []
        if not lab or not re.match(r"^[A-Za-z(0-9]", lab):
            continue
        recs.append((lab, re.findall(NUM, m.group(2))))
    return periods, recs


def parse_table(file: Path, page: int, caption_num: str) -> tuple[list[str], list[tuple[str, list[str]]], str]:
    """(periods, [(label, value tokens)], reading) for the caption on ``page``: the reading (whole page or one column
    at several split points; the caption page or the next) with the most rows."""
    best: tuple[list[str], list, str] = ([], [], "")
    n = _doc(str(file)).page_count
    for pg in (page, page + 1):
        if pg > n:
            continue
        for half, split in [(None, 0.5)] + [(h, sp) for sp in (0.5, 0.42, 0.58) for h in ("left", "right")]:
            rows = page_rows(file, pg, half=half, split=split)
            idx = [i for i, r in enumerate(rows) if re.search(rf"Table\s+{re.escape(caption_num)}(?![\d.])", r.text)]
            start = idx[0] + 1 if idx else (0 if pg != page else None)
            if start is None:
                continue
            periods, recs = _parse_rows(rows, start)
            if len(recs) > len(best[1]):
                best = (periods, recs, f"p{pg} {half or 'full'} {split}")
        if best[1]:
            break
    return best


def gates(title: str, periods: list[str], recs: list[tuple[str, list[str]]]) -> dict[tuple[int, int], str]:
    """Gate result per (row, column)."""
    res = {}
    vals = {lab.lower(): [_f(v) for v in vs] for lab, vs in recs}
    if re.search("Licensed Banks", title, re.I) and "total" in vals:
        types = [v for k, v in vals.items() if k != "total"]
        for j in range(len(periods)):
            ok = abs(sum(t[j] for t in types) - vals["total"][j]) < 0.5
            for i in range(len(recs)):
                res[(i, j)] = "PASS (types sum to total)" if ok else "FAIL: types do not sum to the total"
        if len(periods) == 5:  # Number, Domestic, Foreign, Listed, Not listed
            for i, (_, vs) in enumerate(recs):
                v = [_f(x) for x in vs]
                if abs(v[1] + v[2] - v[0]) > 0.5 or abs(v[3] + v[4] - v[0]) > 0.5:
                    for j in range(5):
                        res[(i, j)] = "FAIL: ownership or listing split does not sum to the number"
    if re.search("Licensed Banks", title, re.I):
        for i, (lab, vs) in enumerate(recs):
            if not re.match(r"(commercial|community|development|microfinance|regional|financial leasing|mortgage|"
                            r"total|national development)", lab, re.I):
                for j in range(len(vs)):
                    res[(i, j)] = "FAIL: not a bank-type row (narrative text read into the table)"
    margin = next((i for i, (lab, _) in enumerate(recs) if "margin" in lab.lower()), None)
    lend = next((v for k, v in vals.items() if "one- year lending" in k or "one-year lending" in k), None)
    dep = next((v for k, v in vals.items() if "one- year deposit" in k or "one-year deposit" in k
                or "12 months deposit" in k), None)
    if margin is not None and lend and dep:
        for j in range(len(periods)):
            ok = abs((lend[j] - dep[j]) - _f(recs[margin][1][j])) <= 0.011
            res[(margin, j)] = "PASS (margin = lending - deposit)" if ok else "FAIL: margin != lending - deposit"
    return res


def extract(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    inv = inventory(root)
    db_f = path("interim", root) / "bot_research_database.csv"
    db = pd.read_csv(db_f, usecols=["period_end", "value", "unit"]) if db_f.exists() else pd.DataFrame(
        columns=["period_end", "value", "unit"])
    db["year"] = pd.to_datetime(db["period_end"], errors="coerce").dt.year
    out = []
    for _, t in inv[inv["priority_target"] & ~inv["appendix"]].iterrows():
        f = report_dir(root) / t["file"]
        periods, recs, _half = parse_table(f, int(t["pdf_page"]), t["table"])
        _half = _half or f"p{t['pdf_page']}"
        if not recs:
            out.append({"report_fy": t["report_fy"], "file": t["file"], "table": t["table"], "title": t["title"],
                        "pdf_page": t["pdf_page"], "gate_result": "NOT EXTRACTED: rows not parsed from the text "
                                                                    "layer (layout); owner check", "used": False})
            continue
        g = gates(t["title"], periods, recs)
        pg = int(re.search(r"p(\d+)", _half).group(1))
        text = re.sub(r"\s+", " ", _doc(str(f))[pg - 1].get_text())
        pct = not re.search("Licensed Banks", t["title"], re.I)
        for i, (lab, vs) in enumerate(recs):
            for j, (p, v) in enumerate(zip(periods, vs, strict=False)):
                val = _f(v)
                found = v.strip("()") in text
                readable = (len(lab) <= 80 and re.fullmatch(r"[A-Za-z0-9(][A-Za-z0-9 ,.()/%+&'’\-–:]*", lab)
                            and len(re.findall(r"[A-Za-z]", lab)) >= 3 and not re.search(PERIOD + r"\s+" + PERIOD, lab))
                if not readable:
                    res = "FAIL: row label not readable (narrative text or font-encoded layer)"
                else:
                    res = g.get((i, j), "PASS") if found else "FAIL: value not found in the page text"
                yr = re.match(r"(\d{4})", p)
                y = int(yr.group(1)) + (1 if "/" in p else 0) if yr else None
                same = db[(db["year"] == y) & ((db["value"] - (val / 100 if pct else val)).abs() < 1e-9 + 0.005 * abs(
                    val / 100 if pct else val))] if y else db.iloc[0:0]
                out.append({"report_fy": t["report_fy"], "file": t["file"], "table": t["table"], "title": t["title"],
                            "pdf_page": pg, "row_label": lab, "period": p, "value": val,
                            "unit": "percent" if pct else "count", "gate": "value on page; totals / margin identity",
                            "gate_result": res, "db_match": "already held in the owner database (same year and value)"
                            if len(same) else "new", "used": res.startswith("PASS")})
    df = pd.DataFrame(out).reindex(columns=COLUMNS)
    df["extraction_method"] = "automated"
    df["evidence_class"] = "Observed"
    return df


CBR_RX = re.compile(r"(?:mainta\w*|keeping|kept|set|raised|lowered|cut|reduced)\s+the CBR at (\d+(?:\.\d+)?) percent"
                    r"([^.]{0,80})", re.I)


def cbr_text(root: Path) -> pd.DataFrame:
    """D42: Central Bank Rate statements in the BoT Annual Reports' text ('...maintaining the CBR at 6.0 percent
    across all four quarters'), one row per statement with the report's financial year. Gate: the sentence is quoted
    from the page; a financial-year statement covering all four quarters dates the rate to the December inside it."""
    fys = report_fy(root)
    out = []
    for f in bot_report_files(root):
        d = _doc(str(f))
        fy = fy_of(f, fys)
        for i in range(d.page_count):
            t = re.sub(r"\s+", " ", d[i].get_text())
            for m in CBR_RX.finditer(t):
                whole = bool(re.search(r"all four quarters|throughout|the whole year|entire", m.group(2), re.I))
                yr = re.match(r"(\d{4})/(\d\d)", fy)
                out.append({"report_fy": fy, "file": f.name, "table": "text", "title": "Central Bank Rate (text)",
                            "pdf_page": i + 1, "row_label": "Central Bank Rate",
                            "period": f"{yr.group(1)}-12" if yr and whole else fy, "value": float(m.group(1)),
                            "unit": "percent", "gate": "sentence quoted from the page: " + m.group(0)[:150],
                            "gate_result": "PASS (rate in force all financial year; dated to its December)" if whole
                            else "FAIL: rate not stated for the whole year (date ambiguous)", "db_match": "new",
                            "used": whole, "extraction_method": "automated", "evidence_class": "Observed"})
    return pd.DataFrame(out, columns=COLUMNS)


def write_bot_reports(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    if not report_dir(root).exists() or not bot_report_files(root):
        return []
    inv = inventory(root)
    ex = pd.concat([extract(root), cbr_text(root)], ignore_index=True)
    return [write_csv(inv, path("reports", root) / "bot_report_inventory.csv", require_evidence=True),
            write_csv(ex, path("interim", root) / "bot_code_extracted.csv", require_evidence=True)]


if __name__ == "__main__":
    for p in write_bot_reports():
        print(p)
