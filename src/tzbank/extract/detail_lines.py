"""NMB and CRDB detail lines for chapter 4 (prompt 03a section 2; blueprint §4: 4.1 revenue, 4.3 deployment, 4.4
transactions, 4.5 operating leverage).

Inspect first, then extract, from every held full annual report (current and comparative columns, Group):

- interest income by source (loans, government securities, placements/interbank, other) from the interest-income
  note; gate: the note lines sum to the note's printed total (0.1%), which is compared with the audited interest
  income held in the workbook;
- fee and commission income by type (transactions, cards, mobile/internet banking, loan-related fees, FX, other);
  gate: the lines sum to the printed fee total (0.1%);
- staff costs (employee benefits / personnel expenses); gate: printed on the page, between 15% and 75% of the held
  operating expenses;
- loans by economic sector (credit-risk concentration table); gate: the sector lines sum to the printed total (0.5%);
- operating KPIs (employees, branches, agents, ATMs, customers) from narrative or KPI pages: confidence MEDIUM,
  cross-checked against the adjacent year's report (ratio 0.7-1.6) and flagged otherwise.

Lines are mapped to categories by label; the label as printed is kept. Unmatched lines go to "other", so categories
always sum to the gated total. Outputs: ``data/extraction/detail_lines.csv`` (accepted) and
``data/extraction/detail_lines_review.csv`` (inspection: every note found, with the gate result).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.ocr import extraction_source
from tzbank.extract.pdftext import _doc, page_rows, page_unit, text_pages
from tzbank.extract.picker import SCALE

BANKS = {"B01": "CRDB", "B02": "NMB"}
YEAR_RX = re.compile(r"(20\d\d)")

NOTES = {
    "interest_income": {
        "heading": r"^(\d+\.?\s*)?(\(?[a-z]\)?\s*)?interest (and similar )?income\b(?!.*expense)",
        "categories": [
            ("Interest Income: Government Securities", r"treasury|government|bond|bills?\b|securit|investment"),
            ("Interest Income: Placements and Interbank", r"placement|interbank|inter-bank|due from|balances? with|"
                                                          r"other banks|banks and|nostro|repo|deposits? with"),
            ("Interest Income: Loans and Advances", r"loan|advance|overdraft|customer|mortgage|credit card|term|"
                                                     r"lease|microfinance|staff"),
        ],
        "other": "Interest Income: Other",
    },
    "fee_income": {
        "heading": r"^(\d+\.?\s*)?(\(?[a-z]\)?\s*)?(net )?fees? and commissions?( income)?\b",
        "categories": [
            ("Fee Income: Cards", r"card|visa|master|atm|pos\b|point of sale"),
            ("Fee Income: Mobile and Internet Banking", r"mobile|internet|digital|sim ?banking|wakala|agency|agent|"
                                                        r"e-?banking|ussd"),
            ("Fee Income: Loan-Related Fees", r"loan|credit|facility|application|commitment|arrangement|guarantee|"
                                              r"letters? of credit|\blc\b|bills? discount|bancassurance|insurance"),
            ("Fee Income: Foreign Exchange", r"foreign exchange|forex|\bfx\b|currency"),
            ("Fee Income: Transactions and Accounts", r"service charge|account|transfer|withdraw|teller|cheque|"
                                                      r"remittance|telegraph|swift|salary|payment|collection|"
                                                      r"commission|statement|ledger|draft|penalt|deposit"),
        ],
        "other": "Fee Income: Other",
    },
}
SECTOR_HEADING = r"(industr|economic sector|sector).*(concentration|analysis|distribution)|concentration of (credit )?" \
                 r"risk.*(industr|sector)|(loans|advances).*by (industry|economic sector|sector)"
STAFF_RX = r"^(staff costs?|employees? benefits?( expenses?)?|personnel (costs?|expenses?))"  # totals only (04 pre-flight)
KPI_RX = {
    "Number of Employees": r"(?:(\d{1,2},\d{3}|\d{3,5})\s+(?:employees|staff members|members of staff|staff)\b)",
    "Number of Branches": r"(?:(\d{2,3})\s+branches)",
    "Number of Banking Agents": r"(?:(\d{1,3},\d{3}|\d{4,6})\s+(?:bank(?:ing)? |wakala |crdb wakala |nmb wakala )?agents)",
    "Number of ATMs": r"(?:(\d{3,4})\s+ATMs)",
    "Number of Customers": r"(?:(\d{1,2}(?:\.\d+)?)\s*million\s+customers)",
}
DETAIL_COLUMNS = ["bank_id", "fiscal_year", "variable", "value", "file", "page", "label", "column", "gate",
                  "gate_passed", "confidence", "notes", "note_kind", "note_total", "audited_total", "evidence_class"]


def full_reports(bank_id: str, root: Path) -> dict[int, Path]:
    """Report year -> the held full annual report (largest page count when a year has several files)."""
    out: dict[int, tuple[int, Path, bool]] = {}
    for f in sorted((path("documents", root) / BANKS[bank_id]).glob("*.pdf")):
        y = YEAR_RX.findall(f.name)
        if not y or "Q3" in f.name:
            continue
        src = extraction_source(f, root)
        n = _doc(str(src)).page_count
        if n <= 6:
            continue
        yr = report_year(src) or int(y[0])  # file names can mislabel the year (CRDB/2022.pdf is the 2021 report)
        named = int(y[0]) == yr
        if yr not in out or (named, n) > (out[yr][2], out[yr][0]):
            out[yr] = (n, f, named)
    return {y: f for y, (_, f, _n) in sorted(out.items())}


def report_year(pdf: Path) -> int | None:
    """Financial year of a full report: the most frequent year in '31 December YYYY' across its pages."""
    d = _doc(str(pdf))
    counts: dict[int, int] = {}
    for i in range(d.page_count):
        for y in re.findall(r"31(?:st)?\s+December,?\s+(20\d\d)", d[i].get_text()):
            counts[int(y)] = counts.get(int(y), 0) + 1
    return max(counts, key=counts.get) if counts else None


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z/&\- ]", " ", label)).strip().lower()


def _amounts(r) -> list[float]:
    return [n for n in r.numbers if n is not None and not (1990 <= abs(n) <= 2035 and float(n).is_integer())]


def note_lines(pdf: Path, page: int, heading: str, half: str | None, split: float) -> list[tuple[str, list[float], float]]:
    """Rows after the heading on one page reading: (label, amounts, y), stopping at the next note heading."""
    rows = page_rows(pdf, page, half=half, split=split)
    start = next((i for i, r in enumerate(rows) if re.search(heading, _clean(r.label)) and len(r.label) < 70), None)
    if start is None:
        return []
    out, pending = [], ""
    for r in rows[start + 1:]:
        lab = _clean(r.label)
        if out and re.match(r"^(\d+\.?\s*)?(interest|fees?|net fee|foreign exchange|other (operating )?income|"
                            r"net trading|operating expenses|dividend income)", lab) and not _amounts(r):
            break
        a = _amounts(r)
        if not a:
            pending = lab if lab else pending
            continue
        out.append(((pending + " " + lab).strip() if lab == "" else lab, a, r.y))
        pending = ""
    return out


def reconcile(lines: list[tuple[str, list[float], float]], col: int) -> tuple[list[tuple[str, float]], float] | None:
    """(component lines, printed total) where the lines before a row sum to that row's value (0.1%); subtotal rows
    (equal to the sum of the immediately preceding lines since the last subtotal) are skipped."""
    vals = [(lab, a[col]) for lab, a, _ in lines if len(a) > col]
    comps: list[tuple[str, float]] = []
    block: list[float] = []
    for lab, v in vals:
        s = sum(x for _, x in comps)
        if len(comps) >= 2 and abs(abs(v) - abs(s)) <= 0.001 * max(abs(v), 1) and (
                "total" in lab or lab == "" or lab.startswith(("fee", "interest", "net", "gross"))
                or len(block) != len(comps)):
            return comps, v
        if len(block) >= 2 and abs(abs(v) - abs(sum(block))) <= 0.001 * max(abs(v), 1):
            block = []  # a subtotal of the preceding lines: not a component
            continue
        comps.append((lab, v))
        block.append(v)
    return None


def categorise(lines: list[tuple[str, float]], spec: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab, v in lines:
        cat = next((c for c, rx in spec["categories"] if re.search(rx, lab)), spec["other"])
        out[cat] = out.get(cat, 0.0) + abs(v)
    return out


def find_note(pdf: Path, kind: str, col: int, valid=None) -> dict | None:
    """First note reading whose lines sum to a printed total, all lines of one sign, and whose total passes
    ``valid(total_in_full_TZS)``."""
    spec = NOTES[kind]
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if kind == "interest_income" and not re.search(r"interest (and similar )?income", t):
            continue
        if kind == "fee_income" and not re.search(r"fees? and commission", t):
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        for half, split in [(None, 0.5)] + [(h, s) for s in (0.5, 0.42, 0.58) for h in ("left", "right")]:
            lines = note_lines(pdf, p, spec["heading"], half, split)
            if len(lines) < 3:
                continue
            hit = reconcile(lines, col)
            if hit and len(hit[0]) >= 2:
                comps, total = hit
                if abs(sum(abs(v) for _, v in comps) - abs(total)) > 0.001 * abs(total):
                    continue  # mixed signs (e.g. a net fee line): not a gross composition
                if valid is not None and not valid(abs(total) * SCALE[unit]):
                    continue
                return {"page": p, "unit": unit, "scale": SCALE[unit], "lines": comps, "total": abs(total),
                        "categories": categorise(comps, spec)}
    return None


def staff_costs(pdf: Path, col: int, opex: float | None) -> dict | None:
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if not re.search(r"staff cost|employee benefit|personnel (cost|expense)|salaries", t):
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        for half in (None, "left", "right"):
            for r in page_rows(pdf, p, half=half):
                if re.match(STAFF_RX, _clean(r.label)) and "%" not in r.text:
                    a = _amounts(r)
                    if len(a) > col:
                        v = abs(a[col]) * SCALE[unit]
                        if opex and 0.15 * opex <= v <= 0.75 * opex:
                            return {"page": p, "value": v, "label": r.label.strip()[:60]}
    return None


def sector_loans(pdf: Path, col: int, gross: float | None) -> dict | None:
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if not re.search(SECTOR_HEADING, t):
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        for half, split in [(None, 0.5), ("left", 0.5), ("right", 0.5)]:
            rows = page_rows(pdf, p, half=half, split=split)
            start = next((i for i, r in enumerate(rows) if re.search(SECTOR_HEADING, r.text.lower())), None)
            if start is None:
                continue
            lines = []
            for r in rows[start + 1: start + 40]:
                a = _amounts(r)
                lab = _clean(r.label)
                if a and lab and not re.match(r"^\d", lab):
                    lines.append((lab, a, r.y))
            hit = reconcile(lines, col)
            if hit and len(hit[0]) >= 4:
                comps, total = hit
                tot = abs(total) * SCALE[unit]
                if gross and abs(tot - gross) <= 0.15 * gross:
                    return {"page": p, "unit": unit, "scale": SCALE[unit], "lines": comps, "total": tot}
    return None


def kpis(pdf: Path) -> list[dict]:
    out = []
    for p in text_pages(pdf)[:120]:
        t = re.sub(r"\s+", " ", _doc(str(pdf))[p - 1].get_text())
        for var, rx in KPI_RX.items():
            for m in re.finditer(rx, t, re.I):
                v = float(m.group(1).replace(",", ""))
                if var == "Number of Customers":
                    v *= 1e6
                out.append({"variable": var, "value": v, "page": p, "quote": t[max(0, m.start() - 60): m.end() + 30]})
    return out


def held(root: Path) -> pd.DataFrame:
    from tzbank.ingest.inputs_workbook import load_interim

    hf = load_interim("Historical Financials", root)
    hf = hf[hf["Bank ID"].isin(list(BANKS)) & (hf["Entity"] == "Group") & hf["value_numeric"].notna()
            & ~hf["Notes"].astype(str).str.startswith("RESTATED")]
    hf = hf.assign(fy=hf["Period End"].dt.year)
    return hf.drop_duplicates(["Bank ID", "fy", "Variable"]).set_index(["Bank ID", "fy", "Variable"])["value_numeric"]


def extract(root: Path | None = None, years: range = range(2008, 2026)) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = root or find_root()
    h = held(root)
    acc, review = [], []

    def g(b, y, v):
        return h.get((b, y, v))

    for b in BANKS:
        reports = full_reports(b, root)
        kpi_by_year: dict[int, list[dict]] = {}
        for ry, f in reports.items():
            src = extraction_source(f, root)
            rel = f"{f.parent.name}/{f.name}"
            for col, fy in [(0, ry), (1, ry - 1)]:
                if fy not in years:
                    continue
                conf = "HIGH" if col == 0 else "MEDIUM"
                cname = "current" if col == 0 else "prior"
                for kind in NOTES:
                    audited = g(b, fy, "Interest Income") if kind == "interest_income" else g(
                        b, fy, "Non-Interest Income")
                    if kind == "interest_income":
                        def valid(x, a=audited):
                            return bool(a) and abs(x / a - 1) <= 0.02  # Islamic / other interest lines aside
                    else:
                        def valid(x, a=audited):
                            # gross fees sit in non-interest income net of fee expense, so they can exceed it a little
                            return bool(a) and 0.2 * a <= x <= 1.5 * a
                    hit = find_note(src, kind, col, valid)
                    review.append({"bank_id": b, "fiscal_year": fy, "note_kind": kind, "file": rel, "column": cname,
                                   "page": hit["page"] if hit else None,
                                   "note_total": hit["total"] * hit["scale"] if hit else None,
                                   "audited_total": audited,
                                   "status": "lines sum to the printed note total" if hit else "note not reconciled"})
                    if not hit:
                        continue
                    tot = hit["total"] * hit["scale"]
                    note = (f"note lines sum to the printed total {hit['total']:,.0f} ({hit['unit']}); lines: "
                            + "; ".join(f"{lab} {v:,.0f}" for lab, v in hit["lines"]))[:900]
                    if audited:
                        ref = "audited interest income" if kind == "interest_income" else "held non-interest income"
                        note += f" | {ref} {audited:,.0f}; note total / held = {tot / audited:.4f}"
                    for var, v in hit["categories"].items():
                        acc.append({"bank_id": b, "fiscal_year": fy, "variable": var, "value": v * hit["scale"],
                                    "file": rel, "page": hit["page"], "label": "sum of category lines",
                                    "column": cname, "gate": "note lines sum to the printed note total (0.1%)",
                                    "gate_passed": True, "confidence": conf, "notes": note, "note_kind": kind,
                                    "note_total": tot, "audited_total": audited})
                    total_var = "Interest Income: Note Total" if kind == "interest_income" else "Fee and Commission Income"
                    acc.append({"bank_id": b, "fiscal_year": fy, "variable": total_var, "value": tot, "file": rel,
                                "page": hit["page"], "label": "printed note total", "column": cname,
                                "gate": "note lines sum to the printed note total (0.1%)", "gate_passed": True,
                                "confidence": conf, "notes": note, "note_kind": kind, "note_total": tot,
                                "audited_total": audited})
                st = staff_costs(src, col, g(b, fy, "Operating Expenses"))
                review.append({"bank_id": b, "fiscal_year": fy, "note_kind": "staff_costs", "file": rel,
                               "column": cname, "page": st["page"] if st else None,
                               "note_total": st["value"] if st else None, "audited_total": g(b, fy, "Operating Expenses"),
                               "status": "found within 15-75% of operating expenses" if st else "not found"})
                if st:
                    acc.append({"bank_id": b, "fiscal_year": fy, "variable": "Staff Costs", "value": st["value"],
                                "file": rel, "page": st["page"], "label": st["label"], "column": cname,
                                "gate": "printed staff-cost line within 15-75% of held operating expenses",
                                "gate_passed": True, "confidence": conf, "notes": "", "note_kind": "staff_costs",
                                "note_total": st["value"], "audited_total": g(b, fy, "Operating Expenses")})
                sec = sector_loans(src, col, g(b, fy, "Gross Loans and Advances to Customers"))
                review.append({"bank_id": b, "fiscal_year": fy, "note_kind": "loans_by_sector", "file": rel,
                               "column": cname, "page": sec["page"] if sec else None,
                               "note_total": sec["total"] if sec else None,
                               "audited_total": g(b, fy, "Gross Loans and Advances to Customers"),
                               "status": "sector lines sum to the printed total" if sec else "not reconciled"})
                if sec:
                    for lab, v in sec["lines"]:
                        acc.append({"bank_id": b, "fiscal_year": fy, "variable": "Loans by Sector: " + lab.title()[:50],
                                    "value": abs(v) * sec["scale"], "file": rel, "page": sec["page"], "label": lab,
                                    "column": cname, "gate": "sector lines sum to the printed total (0.1%); total "
                                    "within 15% of held gross loans", "gate_passed": True, "confidence": conf,
                                    "notes": f"printed total {sec['total']:,.0f}", "note_kind": "loans_by_sector",
                                    "note_total": sec["total"],
                                    "audited_total": g(b, fy, "Gross Loans and Advances to Customers")})
            kpi_by_year[ry] = kpis(src)
        # KPIs: first mention per variable per report; cross-check against the adjacent year's report
        first = {}
        for ry, ks in kpi_by_year.items():
            for k in ks:
                first.setdefault((ry, k["variable"]), k)
        for (ry, var), k in first.items():
            nb = [first.get((ry + d, var)) for d in (-1, 1)]
            ratios = [k["value"] / n["value"] for n in nb if n and n["value"]]
            ok = bool(ratios) and all(0.7 <= r <= 1.6 for r in ratios)
            row = {"bank_id": b, "fiscal_year": ry, "variable": var, "value": k["value"],
                   "file": f"{BANKS[b]}/{reports[ry].name}", "page": k["page"], "label": k["quote"][:120],
                   "column": "current", "gate": "narrative/KPI page; adjacent-year report ratio within 0.7-1.6",
                   "gate_passed": ok, "confidence": "MEDIUM",
                   "notes": "KPI from narrative text (MEDIUM); adjacent-year ratios "
                            + ", ".join(f"{r:.2f}" for r in ratios), "note_kind": "kpi", "note_total": None,
                   "audited_total": None}
            (acc if ok else review).append(row | ({} if ok else {"status": "KPI failed the adjacent-year check"}))
    a = pd.DataFrame(acc).reindex(columns=DETAIL_COLUMNS)
    a["evidence_class"] = "Observed"
    # one source per bank-year and note (never mix lines of two reports): own-year report first
    a["_r"] = (a["column"] != "current").astype(int)
    a["_k"] = a["note_kind"].where(a["note_kind"].isin(list(NOTES) + ["loans_by_sector"]), a["variable"])
    pick = a.sort_values(["_r"]).drop_duplicates(["bank_id", "fiscal_year", "_k"])[["bank_id", "fiscal_year", "_k",
                                                                                       "file", "column"]]
    a = a.merge(pick, on=["bank_id", "fiscal_year", "_k", "file", "column"]).drop_duplicates(
        ["bank_id", "fiscal_year", "variable"]).drop(columns=["_r", "_k"])
    r = pd.DataFrame(review)
    r["evidence_class"] = "Observed"
    return a.reset_index(drop=True), r


def write_detail_lines(root: Path | None = None) -> list[Path]:
    from tzbank.io import write_csv

    root = root or find_root()
    a, r = extract(root)
    d = root / "data" / "extraction"
    return [write_csv(a, d / "detail_lines.csv", require_evidence=True),
            write_csv(r, d / "detail_lines_review.csv", require_evidence=True)]


if __name__ == "__main__":
    for p in write_detail_lines():
        print(p)
