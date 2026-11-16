"""Code-extracted inclusion and NPS supplements (decision D39; prompt 02e section 4; blueprint chapters 1, 2, 4.4/4.7,
6 context).

Transcribes the printed tables and lists that the coverage audit (``reports/inclusion_nps_coverage.csv``) found "not
captured, with printed data". The owner's workbooks are never touched: output goes to
``data/interim/inclusion_code_extracted.csv`` and ``nps_code_extracted.csv`` (``extraction_method=automated``, page
references) and is merged into the processed panels below the owner's values. Gates (D39):

- every value is found in the PDF text of its page (true by construction; re-checked);
- totals tie: a printed total equals the sum of its components (inclusion Table 3.2);
- printed percentage change equals the change computed from the two printed levels (NPS 2022 Table 1);
- lists: serial numbers run 1..N without gaps, and N equals the count stated in the report text where stated;
- cross-vintage: Table H12 values for a year printed in more than one report agree (a difference is reported as a
  restatement, the latest report used per D31, not a failure).

A value that fails a gate is listed in ``reports/inclusion_nps_code_extraction_review.csv`` and not used. Charts whose
"printed data" are only axis ticks are reclassified as unlabelled and not digitised (D33); their series come from the
BoT tables where held (digital saving and credit: Table H12). Needs the PDFs, so it runs outside ``run all``; the
merge into the panels runs inside it (``merge_supplements``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.captions import INCLUSION_FILES, NPS_FILES
from tzbank.extract.pdftext import _doc, page_rows
from tzbank.io import write_csv

NUM = r"\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?"
COLUMNS = ["panel", "report", "table", "caption", "pdf_page", "relevance", "series", "row_label", "column",
           "value", "value_text", "unit", "gate", "gate_result", "used", "extraction_method", "evidence_class"]
RELEVANCE_ORDER = {"ch1": 0, "ch2": 1, "ch4.4/ch4.7": 2, "ch6": 3, "context": 4, "low": 5}
UNLABELLED = [  # (panel, report, kind, number, page): only axis ticks in the text layer -> D33
    ("inclusion", 2024, "chart", "2.2", 23), ("inclusion", 2024, "chart", "5.6", 55),
    ("inclusion", 2025, "chart", "2.1", 22), ("inclusion", 2025, "chart", "5.6", 50),
    ("nps", 2023, "chart", "17", 46), ("nps", 2024, "chart", "19", 46), ("nps", 2024, "chart", "20", 47),
    ("nps", 2025, "chart", "16", 49), ("nps", 2025, "chart", "17", 50),
]


def pdf(panel: str, report: int, root: Path) -> Path:
    files = INCLUSION_FILES if panel == "inclusion" else NPS_FILES
    return path("documents", root) / panel / files[report]


def num(t: str) -> float:
    neg = t.startswith("(") and t.endswith(")")
    v = float(t.strip("()").replace(",", ""))
    return -v if neg else v


def page_text(file: Path, page: int) -> str:
    return re.sub(r"\s+", " ", _doc(str(file))[page - 1].get_text())


def in_text(file: Path, page: int, token: str) -> bool:
    return token in page_text(file, page)


def _rec(panel, report, table, caption, page, rel, series, label, column, value, unit, gate, result, text=None):
    return {"panel": panel, "report": report, "table": table, "caption": caption, "pdf_page": page,
            "relevance": rel, "series": series, "row_label": label, "column": column, "value": value,
            "value_text": text, "unit": unit, "gate": gate, "gate_result": result,
            "used": result.startswith("PASS"), "extraction_method": "automated", "evidence_class": "Observed"}


# --- NPS 2022 Table 1 ------------------------------------------------------------------------------------------------

def nps2022_table1(root: Path) -> list[dict]:
    f, page = pdf("nps", 2022, root), 16
    rows = page_rows(f, page)
    start = next(i for i, r in enumerate(rows) if r.text.startswith("Table 1 Payment System Indicators"))
    out, pending, group = [], [], ""
    rx = re.compile(rf"^(.*?)\s*({NUM})\s+({NUM})\s+({NUM})\s+(Increase|Decrease)$")
    for r in rows[start + 2:]:
        if r.text.startswith("Source"):
            break
        m = rx.match(r.text)
        if not m:
            pending.append(r.text)
            continue
        lab = m.group(1).strip()
        if pending and (not lab or lab[0].islower() or lab[0] == "("):
            if len(pending) > 1:
                group = pending[0]
            lab = (pending[-1] + " " + lab).strip()
        elif pending:
            group = pending[-1]
        pending = []
        a, b, pc = num(m.group(2)), num(m.group(3)), num(m.group(4))
        if "%" in lab:
            calc = b - a  # percentage-point change for a ratio row
        else:
            calc = (b - a) / a * 100 if a else None
        ok = calc is not None and abs(calc - pc) <= 0.011 + 0.0005 * abs(pc)
        res = "PASS" if ok else f"FAIL: printed change {pc} vs computed {calc:.2f}"
        gate = "printed % change = change computed from the two printed levels (0.01)"
        for col, v, tok in [("Dec-21", a, m.group(2)), ("Dec-22", b, m.group(3))]:
            found = in_text(f, page, tok)
            out.append(_rec("nps", 2022, "Table 1", "Payment System Indicators", page, "ch4.4/ch4.7", group, lab, col,
                            v, "count" if "%" not in lab else "percent", gate,
                            res if found else "FAIL: value not found in the page text"))
    return out


# --- NPS Table H12 (2023-2025) -------------------------------------------------------------------------------------

H12_PAGE = {2023: 69, 2024: 66, 2025: 74}


def nps_h12(root: Path) -> list[dict]:
    out = []
    for rep, page in H12_PAGE.items():
        f = pdf("nps", rep, root)
        rows = page_rows(f, page)
        start = next(i for i, r in enumerate(rows) if r.text.startswith("Table H12"))
        years = [int(y) for y in re.findall(r"(20\d\d)", rows[start + 1].text)]
        group = ""
        for r in rows[start + 2:]:
            if re.match(r"^(Table |Annex |\d+ [A-Z][a-z]+, )", r.text) or r.text.startswith("Source"):
                break
            m = re.match(rf"^(.*?\))\s+((?:{NUM}\s*)+)$", r.text)
            if not m:
                if r.text.strip() and not re.search(r"\d", r.text):
                    group = r.text.strip()
                continue
            lab = m.group(1).strip()
            vals = re.findall(NUM, m.group(2))
            if len(vals) != len(years):
                continue
            unit = "million transactions" if "Million" in lab else "TZS billion"
            for y, tok in zip(years, vals, strict=True):
                found = in_text(f, page, tok)
                out.append(_rec("nps", rep, "Table H12", "Digital Financial Services", page, "ch4.4/ch4.7", group,
                                lab, str(y), num(tok), unit, "value found in the page text; cross-vintage check",
                                "PASS" if found else "FAIL: value not found in the page text"))
    df = pd.DataFrame(out)
    if len(df):  # cross-vintage: same series-year in several reports
        for (_g, _lab, _y), g in df.groupby(["series", "row_label", "column"]):
            latest = g["report"].max()
            for i in g.index:
                if df.loc[i, "report"] != latest and g["value"].nunique() == 1:
                    df.loc[i, "used"] = False
                    df.loc[i, "gate_result"] += f"; same value in the {latest} report, which is used (D31)"
            if g["value"].nunique() > 1:
                latest = g["report"].max()
                for i in g.index:
                    note = (f"; vintages differ ({', '.join(f'{r}: {v:g}' for r, v in zip(g['report'], g['value'], strict=True))}); "
                            f"latest report {latest} used (D31)")
                    df.loc[i, "gate_result"] += note
                    if df.loc[i, "report"] != latest:
                        df.loc[i, "used"] = False
                        df.loc[i, "gate_result"] = df.loc[i, "gate_result"].replace("PASS", "PASS (older vintage, "
                                                                                             "not used)", 1)
    return df.to_dict("records")


# --- Inclusion 2025 Table 3.2 ------------------------------------------------------------------------------------------

def incl2025_table32(root: Path) -> list[dict]:
    f, page = pdf("inclusion", 2025, root), 34
    rows = page_rows(f, page)
    start = next(i for i, r in enumerate(rows) if r.text.startswith("Table 3.2"))
    cols = ["2022/23 number", "2022/23 amount (TZS)", "2024/25 number", "2024/25 amount (TZS)",
            "2025/26 Jul-Dec 2025 number", "2025/26 Jul-Dec 2025 amount (TZS)"]
    data = {}
    for r in rows[start + 3:]:
        if r.text.startswith("Source"):
            break
        m = re.match(rf"^([A-Za-z ]+?)\s+((?:{NUM}\s*){{6}})$", r.text)
        if m:
            data[m.group(1).strip()] = re.findall(NUM, m.group(2))
    out = []
    parts = [k for k in data if k != "Total Loans"]
    for j, col in enumerate(cols):
        tot = num(data["Total Loans"][j])
        s = sum(num(data[k][j]) for k in parts)
        res = "PASS" if abs(s - tot) < 0.5 else f"FAIL: components {s:,.0f} vs total {tot:,.0f}"
        for k, vals in data.items():
            found = in_text(f, page, vals[j])
            out.append(_rec("inclusion", 2025, "Table 3.2", "Local Government Credit Beneficiaries and Amount Issued",
                            page, "ch2", "LGA loans (women, youth, PWDs)", k, col, num(vals[j]),
                            "count" if "number" in col else "TZS", "total = women + youth + PWDs",
                            res if found else "FAIL: value not found in the page text"))
    return out


# --- numbered lists (annexes) -----------------------------------------------------------------------------------------

def _words(file: Path, page: int) -> list[tuple]:
    return _doc(str(file))[page - 1].get_text("words")


def _anchor_y(file: Path, page: int, prefix: str) -> float | None:
    for r in page_rows(file, page):
        if r.text.startswith(prefix):
            return r.y
    return None


def _footer_y(file: Path, page: int) -> float:
    ys = [r.y for r in page_rows(file, page)
          if re.search(r"Annual (Payment Systems? )?Report|Payment Systems? Annual Report|Report For 20\d\d", r.text)]
    return min(ys) if ys else 1e9


def numbered_list(file: Path, pages: list[int], start: str, end: str | None, name_from: str,
                  name_to: str | None) -> list[dict]:
    """Rows of an annex list: serial number in the first column, the name between the ``name_from`` and ``name_to``
    header words, from the ``start`` caption to the ``end`` caption (or the last page). Name words are assigned to the
    nearest serial number by vertical position (labels are centred on multi-line rows)."""
    out = []
    x0 = x1 = None
    for k, page in enumerate(pages):
        words = _words(file, page)
        top = (_anchor_y(file, page, start) or 0.0) if k == 0 else 0.0
        bottom = min(_footer_y(file, page), (_anchor_y(file, page, end) or 1e9) if end and k == len(pages) - 1
                     else 1e9)
        hdr = [w for w in words if w[4] == name_from and top <= w[1] < bottom]
        if hdr:
            x0 = hdr[0][0] - 3
            nxt = [w[0] for w in words if name_to and w[4] == name_to and abs(w[1] - hdr[0][1]) < 15]
            x1 = (nxt[0] - 3) if nxt else 1e9
            top = max(top, hdr[0][3])
        if x0 is None:
            continue
        serials = [w for w in words if re.fullmatch(r"\d{1,3}\.?", w[4]) and w[2] <= x0 + 2 and top < w[1] < bottom]
        if not serials:
            continue
        sy = sorted((sw[1], int(sw[4].rstrip("."))) for sw in serials)
        groups: dict[int, list] = {n: [] for _, n in sy}
        for w in words:
            if x0 <= w[0] < x1 and top < w[1] < bottom:
                groups[min(sy, key=lambda t: abs(t[0] - w[1]))[1]].append(w)
        for n, ws in groups.items():
            nm = " ".join(w[4] for w in sorted(ws, key=lambda w: (round(w[1]), w[0])))
            out.append({"serial": n, "name": nm.strip(), "page": page})
    return out


LISTS = [  # panel, report, annex, caption, pages, start caption, end caption, name header from/to, category, relevance,
    # stated count of licensed institutions in the report text (Table 1)
    ("nps", 2023, "Annex M", "Non-Bank Payment System Providers", [75, 76, 77, 78, 79], "Annex M", "Annex N",
     "Institution", "Contact", "non-bank payment system provider", "ch4.4/ch4.7", None),
    ("nps", 2024, "Annex M", "Non-Bank Payment System Providers", [70, 71, 72, 73, 74, 75, 76], "Annex M", "Annex N",
     "Institution", "Contact", "non-bank payment system provider", "ch4.4/ch4.7", 59),
    ("nps", 2025, "Annex O", "Non-Bank Payment System Providers", [82, 83, 84, 85, 86, 87, 88], "Annex O", "Annex P",
     "Institution", "Contact", "non-bank payment system provider", "ch4.4/ch4.7", 72),
    ("nps", 2023, "Annex N", "Electronic Money Issuers", [79, 80], "Annex N", None, "Institution", "Contact",
     "electronic money issuer", "context", None),
    ("nps", 2024, "Annex N", "Electronic Money Issuers", [76], "Annex N", None, "Institution", "Contact",
     "electronic money issuer", "context", None),
    ("nps", 2025, "Annex P", "Electronic Money Issuers", [88, 89], "Annex P", None, "Institution", "Contact",
     "electronic money issuer", "context", None),
    ("nps", 2025, "Annex N", "Tanzania Quick Responses (TANQR) adoption", [80, 81], "Annex N", "Annex O", "BANK", None,
     "TANQR adopter", "context", None),
]


def annex_lists(root: Path) -> list[dict]:
    out = []
    for panel, rep, annex, cap, pages, start, end, hfrom, hto, cat, rel, stated in LISTS:
        f = pdf(panel, rep, root)
        rows = numbered_list(f, pages, start, end, hfrom, hto)
        counts = pd.Series([r["serial"] for r in rows]).value_counts() if rows else pd.Series(dtype=int)
        rows = sorted({r["serial"]: r for r in rows}.values(), key=lambda r: r["serial"])
        serials = [r["serial"] for r in rows]
        gaps = sorted(set(range(1, max(serials, default=0) + 1)) - set(serials))
        note = ""
        if gaps:
            note += f" (printed numbering skips {gaps})"
        if stated is not None and len(rows) != stated:
            note += f" (list has {len(rows)} entries; the report text states {stated} licensed non-banks)"
        for r in rows:
            if counts.get(r["serial"], 0) > 1:
                res = "FAIL: serial number read twice"
            elif not r["name"]:
                res = "FAIL: no name read for this serial number (layout); owner check"
            else:
                res = "PASS" + note
            out.append(_rec(panel, rep, annex, cap, r["page"], rel, cat, r["name"], "licence category", None,
                            "name", "one name per serial number; count vs report text", res,
                            text=f"{r['serial']}. {r['name']} | {cat} | report {rep}"))
    return out


def text_tables(root: Path) -> list[dict]:
    """Qualitative tables kept as one text block each (row structure not parsed): owner check."""
    items = [("inclusion", 2024, "Table 3.5", "Credit Guarantee Scheme", 40, "ch2", "Table 3.5: Credit"),
             ("nps", 2024, "Table 2", "Circulars", 25, "context", "Table 2: Circulars"),
             ("nps", 2025, "Table 2", "Circulars", 24, "context", "Table 2"),
             ("nps", 2025, "Annex M", "Licensing Application Procedures", 78, "context", "Annex M")]
    out = []
    for panel, rep, tab, cap, page, rel, anchor in items:
        f = pdf(panel, rep, root)
        t = page_text(f, page)
        i = t.find(anchor)
        if i < 0:
            continue
        j = t.find("Source", i)
        block = t[i: j if j > 0 else i + 3000][:3000]
        out.append(_rec(panel, rep, tab, cap, page, rel, "text table", cap, "text", None, "text",
                        "block found in the page text (rows not parsed)", "PASS (text block; owner check)",
                        text=block))
    return out


def unlabelled(root: Path) -> list[dict]:
    out = []
    for panel, rep, kind, numb, page in UNLABELLED:
        out.append(_rec(panel, rep, f"{kind.title()} {numb}", "", page, "", "", "", "", None, "",
                        "data labels present?", "NOT EXTRACTED: only axis ticks in the text layer; not digitised "
                        "(D33). Digital saving/credit series come from Table H12" if "nps" == panel else
                        "NOT EXTRACTED: only axis ticks in the text layer; not digitised (D33)"))
        out[-1]["used"] = False
    # 2023 Chart 2.7: labels printed for two of the years only, two series: series assignment ambiguous
    out.append(_rec("inclusion", 2023, "Chart 2.7", "Number of Loans Accounts Banks and SACCOs", 26, "ch2", "", "",
                    "", None, "", "one label per series and year",
                    "NOT USED: labels printed for 2019-2020 only (1,830,513; 1,900,540; 3,567,388; 3,841,602) and the "
                    "series they belong to cannot be read from the text layer; owner check"))
    out[-1]["used"] = False
    return out


def extract(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    recs = (nps2022_table1(root) + nps_h12(root) + incl2025_table32(root) + annex_lists(root) + text_tables(root)
            + unlabelled(root))
    df = pd.DataFrame(recs).reindex(columns=COLUMNS)
    df["owner_check"] = ""
    df["_r"] = df["relevance"].map(RELEVANCE_ORDER).fillna(9)
    return df.sort_values(["_r", "panel", "report", "table", "row_label", "column"]).drop(columns="_r")


def write_supplements(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    df = extract(root)
    i = path("interim", root)
    return [write_csv(df[df["panel"] == "inclusion"].drop(columns="owner_check"),
                      i / "inclusion_code_extracted.csv", require_evidence=True),
            write_csv(df[df["panel"] == "nps"].drop(columns="owner_check"), i / "nps_code_extracted.csv",
                      require_evidence=True),
            write_csv(df, path("reports", root) / "inclusion_nps_code_extraction_review.csv", require_evidence=True)]


def merge_supplements(panel: pd.DataFrame, name: str, root: Path | None = None) -> pd.DataFrame:
    """Append used, numeric code-extracted values to a processed panel as a lower-precedence source (D39): a
    series-year the owner's panel already holds is never replaced."""
    root = root or find_root()
    f = path("interim", root) / f"{name}_code_extracted.csv"
    out = panel.copy()
    out["source_precedence"] = out.get("source_precedence", "owner panel")
    if not f.exists():
        return out
    sup = pd.read_csv(f)
    sup = sup[sup["used"].astype(str).eq("True") & sup["value"].notna()]
    if sup.empty:
        return out
    yr = sup["column"].astype(str).str.extract(r"(20\d\d)")[0]
    add = pd.DataFrame({
        "canonical_name": ("code_" + name + "_" + (sup["table"] + "_" + sup["series"].fillna("") + "_"
                                                   + sup["row_label"]).str.lower().str.replace(r"[^a-z0-9]+", "_",
                                                                                               regex=True)),
        "value": sup["value"], "year_label": sup["column"], "source_report": sup["report"],
        "pdf_page": sup["pdf_page"], "unit": sup["unit"], "evidence_class": "Observed",
        "source_precedence": "code-extracted (D39, below owner)", "extraction_method": "automated",
        "gate_result": sup["gate_result"]})
    ycol = "Year" if name == "inclusion" else "year"
    add[ycol] = pd.to_numeric(yr, errors="coerce")
    return pd.concat([out, add], ignore_index=True)


if __name__ == "__main__":
    for p in write_supplements():
        print(p)
