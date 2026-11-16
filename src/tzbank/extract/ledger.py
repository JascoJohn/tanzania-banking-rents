"""Extraction ledger (prompt 02 sections 1.2-1.3): gated candidates -> one reviewed set of rows for the working
inputs workbook. data/extraction/ledger.csv is the committed audit trail (file, page, label, gate) of every value
written to the workbook; data/extraction/manual_review.csv lists what no gate accepted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.clean.normalise import load_canonical
from tzbank.config import find_root, path
from tzbank.extract.gated import apply_capital_gate, apply_phase1_gate, candidates_for
from tzbank.extract.pdftext import _doc, page_rows, page_unit, text_pages
from tzbank.extract.picker import SCALE, _amounts
from tzbank.io import write_csv

FOLDERS = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "Maendeleo", "B05": "MKCB", "B06": "MCB",
           "B07": "MUCOBA"}
FOCUS = {"B01", "B02"}
# Variables the ledger writes. For NMB/CRDB only lines the workbook does not hold yet.
SMALL_BANK_VARS = {"Interest Income", "Interest Expense", "Net Interest Income", "Interest Expense on Customer Deposits",
                   "Interest Expense on Time Deposits", "Interest Expense on Savings Deposits",
                   "Interest Expense on Current Accounts", "Interest Expense on Call Deposits",
                   "Interest Expense on Borrowings", "Gross Loans and Advances to Customers",
                   "Loss Allowance on Loans (ECL)", "Total RWA", "Tier 1 Capital", "Income Tax Expense",
                   "Customer Deposits", "Non-Interest Income", "Operating Expenses", "Impairment Charges", "Net Profit",
                   "Profit Before Tax", "Total Equity", "Total Assets", "Loans and Advances to Customers (Net)",
                   "Customer Deposits: Current Accounts", "Customer Deposits: Savings", "Customer Deposits: Time",
                   "Customer Deposits: Other"}
FOCUS_VARS = {"Interest Expense on Customer Deposits", "Interest Expense on Time Deposits",
              "Interest Expense on Savings Deposits", "Interest Expense on Current Accounts",
              "Interest Expense on Call Deposits", "Interest Expense on Borrowings",
              "Customer Deposits: Current Accounts", "Customer Deposits: Savings", "Customer Deposits: Time",
              "Customer Deposits: Other"}
CAPITAL_VARS = {"Total RWA", "Tier 1 Capital"}


def wanted(bank_id: str, year: int, variable: str) -> bool:
    """Which (bank, year, variable) the ledger may write (prompt 02 sections 1.2-1.3)."""
    if bank_id not in FOCUS:
        return variable in SMALL_BANK_VARS
    if variable in FOCUS_VARS:
        return True
    if bank_id == "B01" and year in (2009, 2010) and variable in SMALL_BANK_VARS:
        return True  # CRDB FY2009 gap fill and first-reported FY2009-10 lines
    return bank_id == "B01" and 2009 <= year <= 2014 and variable in CAPITAL_VARS  # D17, Bank basis
MIX = {"Customer Deposits: Current Accounts": r"^(current|demand)( and demand)?( accounts?| deposits?)?$",
       # NMB reports its savings products as 'personal accounts'; the as-reported label is kept in the ledger
       "Customer Deposits: Savings": r"^(savings?|personal)( accounts?| deposits?)?$",
       "Customer Deposits: Time": r"^(time|fixed|term)( deposits?)?( accounts?)?$",
       "Customer Deposits: Other": r"^(call|other|margin)( deposits?| accounts?)?$"}


def deposit_mix(pdf: Path, total: float, col: int) -> list[tuple[str, float, int, str]] | None:
    """Deposit-mix lines on one page whose sum equals balance-sheet customer deposits (0.5%)."""
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if not re.search(r"deposits? (from|due to) customers|customer deposits", t):
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        found: dict[str, tuple[float, str]] = {}
        for r in page_rows(pdf, p):
            lab = _clean(r.label)
            a = _amounts(r)
            if len(a) <= col:
                continue
            for var, pat in MIX.items():
                if var not in found and re.match(pat, lab):
                    found[var] = (abs(a[col]) * SCALE[unit], r.label[:60])
        if len(found) >= 2 and abs(sum(v for v, _ in found.values()) - total) <= 0.005 * total:
            return [(k, v, p, lab) for k, (v, lab) in found.items()]
    return None


def mix_candidates(c: pd.DataFrame, bank_id: str, root: Path) -> list[dict]:
    """Deposit-mix candidates for each document/column where balance-sheet deposits were found."""
    from tzbank.extract.ocr import extraction_source

    out = []
    dep = c[(c["variable"] == "Customer Deposits") & c["value"].notna()]
    for _, r in dep.drop_duplicates(["file", "column"]).iterrows():
        src = extraction_source(path("documents", root) / r["file"], root)
        col = 0 if r["column"] == "current" else 1
        mix = deposit_mix(src, float(r["value"]), col)
        if not mix:
            continue
        for var, v, p, lab in mix:
            out.append({"bank_id": bank_id, "fiscal_year": r["fiscal_year"], "variable": var, "value": v,
                        "file": r["file"], "page": p, "label": lab, "column": r["column"],
                        "gate": "mix lines sum to BS customer deposits (0.5%)", "gate_passed": True,
                        "confidence": r["confidence"], "notes": "", "extra": {}, "ocr": src != path(
                            "documents", root) / r["file"]})
    return out


IE_PARTS = {
    "Interest Expense on Current Accounts": r"^(current|demand)( accounts?| deposits?)?$",
    "Interest Expense on Savings Deposits": r"^savings?( accounts?| deposits?)?$",
    "Interest Expense on Time Deposits": r"^(time|fixed|term)( deposits?| accounts?)?$",
    "Interest Expense on Call Deposits": r"^call( deposits?| accounts?)?$",
    "Interest Expense on Customer Deposits": r"^deposits? (from|due to|by) customers$|^customer deposits$",
    "Interest Expense on Bank Deposits": r"^deposits?.*(due to|from|by) (other )?banks|^inter ?-?bank|^balances due to (other )?banks",
    "Interest Expense on Borrowings": r".*borrow|.*bonds?$|.*bond issued|^subordinated|^sukuk|^debt securities",
    "Interest Expense on Leases": r".*lease liabilit|.*leases?$",
    "Interest Expense Other": r"^other( interest)?( expense)?$|^repurchase",
}
DEP_TYPES = ["Interest Expense on Current Accounts", "Interest Expense on Savings Deposits",
             "Interest Expense on Time Deposits", "Interest Expense on Call Deposits"]


def _clean(label: str) -> str:
    """Lower-case letters only, without a trailing '(Note nn)' reference."""
    lab = re.sub(r"^[^a-z]+", "", re.sub(r"[^a-z ]", " ", label.lower())).strip()
    return re.sub(r"\s*\bnotes?\b[\s\w]*$", "", lab).strip()


def ie_note_components(pdf: Path, target: float, col: int) -> dict | None:
    """Interest-expense note lines on one page whose sum equals the audited total interest expense (0.5%)."""
    for p in text_pages(pdf):
        t = _doc(str(pdf))[p - 1].get_text().lower()
        if not re.search(r"interest (and similar )?expense", t) or "deposit" not in t:
            continue
        unit = page_unit(pdf, p)
        if unit not in SCALE:
            continue
        for strict in (True, False):
            for half, split in [(None, 0.5)] + [(h, f) for f in (0.5, 0.42, 0.58, 0.35, 0.65) for h in ("left", "right")]:
                hit = _ie_rows(pdf, p, unit, target, col, half, strict, split)
                if hit:
                    return hit
    return None


def _ie_rows(pdf: Path, p: int, unit: str, target: float, col: int, half: str | None,
             strict: bool = True, split: float = 0.5) -> dict | None:
    """One pass of the note reconciliation over a page or one column of it. ``strict`` collects only below an
    interest-expense heading; the relaxed pass reads the whole page (the 0.1% sum test still applies)."""
    found: dict[str, tuple[float, str]] = {}
    rows = page_rows(pdf, p, half=half, split=split)
    widths = [len([n for n in r.numbers if not (1990 <= abs(n) <= 2035)]) for r in rows]
    ncol = max(set(w for w in widths if w >= 2), key=widths.count, default=2)
    pending = ""  # a label-only row whose text wraps onto the next (numeric) row
    started = not strict  # strict: collect only below the interest-expense heading
    for r in rows:
        lab = re.sub(r"\s+", " ", _clean(r.label))
        if not started:
            started = bool(re.match(r"(\d+ )?interest (and similar )?expenses?", lab)) and "income" not in lab
            continue
        if found and re.match(r"(\d+ )?(net )?(fees?|commission|foreign exchange|interest (and similar )?income)", lab):
            break  # next note
        if found and re.fullmatch(r"(b )?(bank|company)", lab):
            break  # a separate Bank section repeats the Group lines
        if not r.numbers and lab:
            pending = lab
            continue
        a = [n for n in r.numbers if not (1990 <= abs(n) <= 2035 and float(n).is_integer())]
        if len(a) == ncol + 1 and abs(a[0]) < 100 and float(a[0]).is_integer():
            a = a[1:]  # a note reference read as the first amount
        tail = re.search(r"\s(\d{1,2})$", r.label.strip())
        if tail and len(a) == ncol - 1:
            a = [float(tail.group(1)), *a]  # a small amount read as a note reference
        if len(a) <= col:
            continue
        matched = False
        for cand in (lab, f"{pending} {lab}".strip()):
            for var, pat in IE_PARTS.items():
                if re.match(pat, cand):
                    v0, l0 = found.get(var, (0.0, ""))
                    found[var] = (v0 + abs(a[col]) * SCALE[unit], (l0 + " + " if l0 else "") + cand[:40])
                    matched = True
                    break
            if matched:
                break
        pending = ""
    if len(found) >= 2 and abs(sum(v for v, _ in found.values()) - target) <= 0.001 * target:
        return {"page": p, "parts": found, "half": half}
    return None


def focus_ie_candidates(bank_id: str, root: Path, cands: pd.DataFrame | None = None) -> list[dict]:
    """Deposit interest from the interest-expense note, gated on the audited total interest expense: the
    inputs workbook's value for NMB/CRDB, the gated income-statement value for B03-B07."""
    from tzbank.extract.ocr import extraction_source
    from tzbank.ingest.inputs_workbook import load_interim

    if bank_id in FOCUS:
        hf = load_interim("Historical Financials", root)
        ie = hf[(hf["Bank ID"] == bank_id) & (hf["Variable"] == "Interest Expense") & hf["value_numeric"].notna()
                & (hf["Entity"] == "Group")]
        target = {int(d.year): v for d, v in zip(ie["Period End"], ie["value_numeric"], strict=False)}
    else:
        ie = cands[(cands["variable"] == "Interest Expense") & cands["gate_passed"]] if cands is not None else []
        target = {int(y): v for y, v in zip(ie["fiscal_year"], ie["value"], strict=False)} if len(ie) else {}
    out = []
    for pdf in sorted((path("documents", root) / FOLDERS[bank_id]).glob("*.pdf")):
        src = extraction_source(pdf, root)
        if not text_pages(src) or _doc(str(src)).page_count <= 6:
            continue
        rel = f"{FOLDERS[bank_id]}/{pdf.name}"
        years = re.findall(r"(20\d\d)", pdf.name)
        if not years:
            continue
        fy = int(years[0])
        for col, y in [(0, fy), (1, fy - 1)]:
            if y not in target:
                continue
            hit = ie_note_components(src, target[y], col)
            if not hit:
                continue
            parts = hit["parts"]
            for var, (v, lab) in parts.items():
                if var in {"Interest Expense on Bank Deposits", "Interest Expense on Leases", "Interest Expense Other"}:
                    continue
                out.append({"bank_id": bank_id, "fiscal_year": y, "variable": var, "value": v, "file": rel,
                            "page": hit["page"], "label": lab, "column": "current" if col == 0 else "prior",
                            "gate": "note lines sum to audited total interest expense (0.1%)", "gate_passed": True,
                            "confidence": "HIGH" if col == 0 else "MEDIUM", "notes": "", "extra": {},
                            "ocr": src != pdf})
            types = [parts[k][0] for k in DEP_TYPES if k in parts]
            if types and "Interest Expense on Customer Deposits" not in parts:
                out.append({"bank_id": bank_id, "fiscal_year": y, "variable": "Interest Expense on Customer Deposits",
                            "value": sum(types), "file": rel, "page": hit["page"],
                            "label": "sum of " + ", ".join(parts[k][1] for k in DEP_TYPES if k in parts),
                            "column": "current" if col == 0 else "prior",
                            "gate": "note lines sum to audited total interest expense (0.1%)", "gate_passed": True,
                            "confidence": "HIGH" if col == 0 else "MEDIUM",
                            "notes": "COMPUTED = sum of deposit-type lines", "extra": {}, "ocr": src != pdf})
    return out


def all_candidates(root: Path | None = None, banks: list[str] | None = None) -> pd.DataFrame:
    """Gated candidates for the given banks (all seven by default)."""
    root = root or find_root()
    canon = load_canonical(root)
    parts = []
    for b in banks or list(FOLDERS):
        c = candidates_for(b, path("documents", root) / FOLDERS[b], root)
        if c.empty:
            continue
        extra = mix_candidates(c, b, root) + focus_ie_candidates(b, root, c)
        c = pd.concat([c, pd.DataFrame(extra)], ignore_index=True)
        parts.append(c)
    c = pd.concat(parts, ignore_index=True)
    c = c[c["variable"] != "ERROR"]
    return apply_capital_gate(apply_phase1_gate(c, canon), canon)


def select(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ledger rows, manual-review rows). Own-year report first; a differing later comparative is kept as a
    restatement row (D21)."""
    want = c.apply(lambda r: wanted(r["bank_id"], int(r["fiscal_year"]), r["variable"]), axis=1)
    c = c[want & c["value"].notna()].copy()
    # prompt 02d 1.3: for NMB/CRDB the weak '<= total' check never admits deposit interest
    weak = c["bank_id"].isin(FOCUS) & c["gate"].astype(str).str.contains("deposit interest <= total")
    c.loc[weak, "gate_passed"] = False
    c["doc_year"] = c["file"].str.extract(r"(\d{4})")[0].astype(float)
    acc = c[c["gate_passed"]].copy()
    acc["rank"] = (acc["column"] != "current").astype(int)
    # stronger gates first: note lines reconciling to the audited total beat the weak '<= total' check
    acc["gate_rank"] = (~acc["gate"].astype(str).str.contains("sum to audited total|sum to BS|II-IE=NII|"
                                                              "gross - allowance|Phase 1|Tier1/RWA")).astype(int)
    acc = acc.sort_values(["bank_id", "fiscal_year", "variable", "gate_rank", "rank", "doc_year"])
    rows = []
    for _, g in acc.groupby(["bank_id", "fiscal_year", "variable"], sort=False):
        first = g.iloc[0].to_dict()
        first["row_type"] = "original"
        rows.append(first)
        for _, r in g.iloc[1:].iterrows():
            if r["file"] != first["file"] and abs(r["value"] - first["value"]) > 0.005 * abs(first["value"] or 1):
                rows.append(r.to_dict() | {"row_type": "restated", "confidence": "MEDIUM",
                                           "notes": f"RESTATED in {r['file']} comparative "
                                                    f"(original {first['value']:,.0f} from {first['file']})"})
                break
    ledger = pd.concat([pd.DataFrame(rows), manual_rows()], ignore_index=True)
    keys = set(zip(ledger["bank_id"], ledger["fiscal_year"], ledger["variable"], strict=False)) if len(ledger) else set()
    review = c[~c["gate_passed"]]
    review = review[[(b, y, v) not in keys for b, y, v in zip(review["bank_id"], review["fiscal_year"],
                                                                review["variable"], strict=False)]]
    review = review.drop_duplicates(["bank_id", "fiscal_year", "variable", "file"])
    return ledger, review


def manual_rows() -> pd.DataFrame:
    """Values read by a person (D29): data/raw/manual_entries/manual_values.csv, validated by tzbank.ingest.manual."""
    from tzbank.ingest.manual import read_manual

    valid, _ = read_manual()
    if valid.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "bank_id": valid["bank_id"], "fiscal_year": valid["fiscal_year"].astype(int), "variable": valid["variable"],
        "value": valid["value_full"], "file": valid["file"], "page": valid["page"].astype(int),
        "label": valid["as_reported_label"], "column": "current", "confidence": valid["confidence"].fillna("HIGH"),
        "notes": valid["notes"].fillna("") + " | read by " + valid["checked_by"], "row_type": "original",
        "gate": "manual entry (D29)", "gate_passed": True, "ocr": False, "extraction_method": "manual"})


LEDGER_COLUMNS = ["bank_id", "fiscal_year", "variable", "value", "row_type", "file", "page", "label", "column",
                  "gate", "confidence", "ocr", "extraction_method", "notes", "phase1_value", "phase1_rel_diff"]


def write_ledger(root: Path | None = None) -> list[Path]:
    """Write data/extraction/ledger.csv and data/extraction/manual_review.csv."""
    root = root or find_root()
    c = all_candidates(root)
    ledger, review = select(c)
    d = root / "data" / "extraction"
    ledger = ledger.reindex(columns=LEDGER_COLUMNS).sort_values(["bank_id", "fiscal_year", "variable", "row_type"])
    ledger["extraction_method"] = ledger["extraction_method"].fillna("gated")
    review = review.reindex(columns=[x for x in LEDGER_COLUMNS if x != "row_type"])
    ledger["evidence_class"] = "Observed"
    review["evidence_class"] = "Observed"
    return [write_csv(ledger, d / "ledger.csv"), write_csv(review, d / "manual_review.csv"),
            write_csv(c.drop(columns=["extra"]), d / "candidates_all.csv")]


def rebuild_from_candidates(root: Path | None = None) -> list[Path]:
    """Re-run selection (and manual entries) from data/extraction/candidates_all.csv without re-reading PDFs."""
    root = root or find_root()
    d = root / "data" / "extraction"
    c = pd.read_csv(d / "candidates_all.csv", low_memory=False)
    c["gate_passed"] = c["gate_passed"].astype(str).eq("True")
    c["ocr"] = c["ocr"].astype(str).eq("True")
    ledger, review = select(c)
    ledger = ledger.reindex(columns=LEDGER_COLUMNS).sort_values(["bank_id", "fiscal_year", "variable", "row_type"])
    ledger["extraction_method"] = ledger["extraction_method"].fillna("gated")
    review = review.reindex(columns=[x for x in LEDGER_COLUMNS if x != "row_type"])
    ledger["evidence_class"] = "Observed"
    review["evidence_class"] = "Observed"
    return [write_csv(ledger, d / "ledger.csv"), write_csv(review, d / "manual_review.csv")]
