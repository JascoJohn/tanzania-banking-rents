"""Quarterly BoT-format statements of NMB and CRDB from the DSE library documents (prompt 15 item 4a; D73).

The banks publish a condensed statement every quarter under the BFI (Disclosures) Regulations 2014 (before 2014: the
Report of Condition under section 32(3) of the BFI Act 2006). The layout is fixed by the regulator: a balance sheet
(current and previous quarter), an income statement (current and comparative quarter, current and comparative year
cumulative) and selected performance indicators, either for one entity or for Group and Bank side by side.

Method (deterministic, page-recorded):

* Text comes from the PDF text layer, or from the OCR copy under ``data/ocr/dse/`` for image-only files.
* Each text line is either a label or a run of numbers (enumerators such as ``3.``, ``6.1`` and ``(iv)`` are dropped);
  a run belongs to the label lines just above it (or to the label on the same line).
* Column choice follows the published layout: balance-sheet lines take the current quarter (Group when Group and
  Bank are both shown); income-statement lines take the current-year cumulative column (Group first), i.e. year to
  date; the first quarter's quarter and cumulative columns coincide. Ambiguous layouts are rejected, not guessed.
* A value is accepted only if its statement passes an arithmetic gate in the chosen column:
  interest income - interest expense = net interest income (0.5%); net interest income - bad debts - impairment +
  non-interest income - non-interest expense = operating income (1%); operating income - tax = net income (0.5%);
  total liabilities + shareholders' funds = total assets (0.5%). Year-to-date columns must exceed the quarter column
  after the first quarter. Failed or ambiguous documents are listed for manual review, never written.
* Units come from the page ("amounts in million shillings"); when absent the scale is inferred from the bank's
  audited total assets for the nearest year (factor between 0.6 and 1.6) and noted.

Accepted rows feed the inputs workbook "Quarterly Results" sheet (D18 session) only for periods not already held
(Interim Results holds H1 2025-26 and 9M 2024-26; fourth-quarter statements are not written because the audited
annual statements are held; they are used here only to cross-check the parser).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.extract.pdftext import parse_number

NUM = re.compile(r"^\(?-?\d[\d,]*(\.\d+)?\)?%?$|^[-–—.]$")
ENUM = re.compile(r"^(\d{1,2}\.?|\d{1,2}\.\d{1,2}|\(?[ivx]{1,4}\)|[a-f]\.)$", re.I)
UNIT = re.compile(r"amounts?\s+in\s+(tzs\s+|tshs?\s+)?(million|'?000|thousand)|in\s+(million|thousands)\s+(of\s+)?"
                  r"(tanzania\s+)?shillings|(tzs|tshs?)\s*['’]?\s*(000|million|m)\b", re.I)
TICKER_BANK = {"CRDB": "B01", "NMB": "B02"}

# item -> (statement, regex on the normalised label, full match)
ITEMS: list[tuple[str, str, str]] = [
    ("nim_reported", "ratio", r"net interest (income to average earning assets|margin)"),
    ("net_interest_income", "pl", r"net interest income( 1 ?minus ?2)?"),
    ("interest_income", "pl", r"interest income"),
    ("interest_expense", "pl", r"interest expenses?"),
    ("bad_debts_written_off", "pl", r"bad debts written off"),
    ("impairment_charges", "pl", r"impairment loss(es)? on loans( and advances)?"),
    ("non_interest_income", "pl", r"non ?-?interest income"),
    ("non_interest_expense", "pl", r"non ?-?interest expenses?"),
    ("operating_income", "pl", r"operating (income|profit)( loss)?"),
    ("income_tax", "pl", r"income tax (provision|expense)"),
    ("net_income", "pl", r"net (income|profit)( loss)? ?after ?(income )?tax"),
    ("net_loans", "bs", r"loans,? advances and overdrafts( net of allowances? for probable losses)?|"
                        r"net of allowances? for probable losses"),
    ("total_assets", "bs", r"total assets"),
    ("total_liabilities", "bs", r"total liabilities"),
    ("customer_deposits", "bs", r"customer deposits|deposits from customers"),
    ("total_equity", "bs", r"total shareholders? funds"),
    ("npl_gross", "bs", r"non ?-?performing loans (& |and )?advances"),
    ("allowances", "bs", r"allowances? for probable losses"),
]
# accepted items -> inputs-workbook variable (Quarterly Results sheet)
WORKBOOK_VARS = {
    "interest_income": "Interest Income", "interest_expense": "Interest Expense",
    "net_interest_income": "Net Interest Income", "impairment_charges": "Impairment Charges",
    "non_interest_income": "Non-Interest Income", "non_interest_expense": "Operating Expenses",
    "operating_income": "Profit Before Tax", "income_tax": "Income Tax Expense", "net_income": "Net Profit",
    "total_assets": "Total Assets", "net_loans": "Loans and Advances to Customers (Net)",
    "customer_deposits": "Customer Deposits", "total_equity": "Total Equity",
    "npl_gross": "Non-Performing Loans (Gross)", "nim_reported": "Net Interest Margin (reported)",
}
POSITIVE = {"interest_expense", "impairment_charges", "non_interest_expense", "income_tax", "bad_debts_written_off"}


def _norm(s: str) -> str:
    s = re.sub(r"[()/:'’.’]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def quarter_end(listing_date: str) -> pd.Timestamp | None:
    """The quarter end a DSE listing date refers to: the quarter end q with q - 10 days <= date <= q + 45 days."""
    d = pd.Timestamp(listing_date)
    for q in pd.date_range(d - pd.Timedelta(days=50), d + pd.Timedelta(days=12), freq="QE"):
        if q - pd.Timedelta(days=10) <= d <= q + pd.Timedelta(days=45):
            return q
    return None


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], start=1)}


def text_period_end(text: str) -> pd.Timestamp | None:
    """The statement's own period end: the latest quarter end printed in it (comparative columns are earlier).
    Reads 31.03.2024, 31/03/2024, 31.03.24, 31-Mar-2024 and '31st March, 2024' forms."""
    t = text.lower()
    found = []
    for d, m, y in re.findall(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b", t):
        found.append((int(d), int(m), int(y) + (2000 if len(y) == 2 else 0)))
    for d, mn, y in re.findall(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s.-]*(" + "|".join(MONTHS) + r"|mar|jun|sep|sept|dec)"
                               r"[a-z]*[\s,.-]*(\d{4})\b", t):
        m = MONTHS.get(mn) or {"mar": 3, "jun": 6, "sep": 9, "sept": 9, "dec": 12}[mn]
        found.append((int(d), m, int(y)))
    qe = []
    for d, m, y in found:
        if m in (3, 6, 9, 12) and d >= 28 and 2005 <= y <= 2030:
            qe.append(pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))
    return max(qe) if qe else None


def source_file(local_file: str, root: Path) -> tuple[Path, str]:
    """The PDF to read (OCR copy for image-only files) and the text source label."""
    from tzbank.extract.ocr import needs_ocr, ocr_target

    p = root / local_file
    if needs_ocr(p):
        t = ocr_target(p, root)
        return (t, "OCR copy") if t.exists() else (p, "image-only (no OCR copy)")
    return p, "text layer"


def runs(pdf: Path) -> tuple[list[dict], str]:
    """Label / number runs (page-recorded) and the document text."""
    doc = pymupdf.open(pdf)
    out, text = [], []
    for pi, page in enumerate(doc, start=1):
        t = page.get_text()
        text.append(t)
        unit = UNIT.search(t)
        labels: list[str] = []
        cur: dict | None = None
        for raw in t.splitlines():
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) == 1 and ENUM.match(toks[0]) and not toks[0].startswith("("):
                continue
            k = len(toks)
            while k > 0 and NUM.match(toks[k - 1]):
                k -= 1
            lab, nums = " ".join(t_ for t_ in toks[:k] if not ENUM.match(t_)), toks[k:]
            if lab:
                if cur is not None:
                    out.append(cur)
                    cur = None
                    labels = []
                labels.append(lab)
            if nums:
                vals = [0.0 if n in {"-", "–", "—", "."} else parse_number(n.rstrip("%")) for n in nums]
                pct = any(n.endswith("%") for n in nums)
                if cur is None:
                    cur = {"page": pi, "labels": labels[-3:], "values": [], "pct": False,
                           "unit": unit.group(0) if unit else ""}
                cur["values"] += vals
                cur["pct"] = cur["pct"] or pct
        if cur is not None:
            out.append(cur)
    return out, "\n".join(text)


def classify(r: dict) -> tuple[str, str, str] | None:
    """(item, statement, matched label) for a run, trying the last one, two and three label lines."""
    labs = [_norm(x) for x in r["labels"]]
    for k in (1, 2, 3):
        if len(labs) < k:
            break
        cand = " ".join(labs[-k:])
        for item, st, pat in ITEMS:
            if re.fullmatch(pat, cand):
                return item, st, cand
    return None


def layout(text: str) -> tuple[bool, bool]:
    """(two entities shown, Group column first)."""
    lines = [_norm(x) for x in text.splitlines()]
    g = next((i for i, x in enumerate(lines) if x in {"group", "consolidated"}), None)
    b = next((i for i, x in enumerate(lines) if x in {"bank", "solo", "company"}), None)
    return g is not None, g is not None and (b is None or g < b)


def pick(item: str, st: str, n: int, q: int, two: bool, group_first: bool) -> tuple[int | None, int | None, str]:
    """(chosen column, quarter column for the YTD check, column description) or (None, None, reason)."""
    if st in {"bs", "ratio"}:
        if two and n >= 4:
            return (0, None, "Group, current quarter") if group_first else (2, None, "Group, current quarter")
        if n >= 2:
            return 0, None, "current quarter" + (" (first column)" if two else "")
        return None, None, f"{n} values"
    if n >= 8 and two:
        return (4, 0, "Group, current year cumulative") if group_first else (6, 2, "Group, current year cumulative")
    if n == 4 and two:
        return ((0, 0, "Group, current quarter = year to date (Q1)") if q == 1 else
                (None, None, "4 values for two entities after Q1 (ambiguous)"))
    if n == 4:
        return 2, 0, "current year cumulative"
    if n == 2 and q == 1:
        return 0, 0, "current quarter = year to date (Q1)"
    return None, None, f"{n} values (ambiguous layout)"


def extract_document(row: dict, root: Path, fy_assets: dict) -> list[dict]:
    pdf, src = source_file(row["local_file"], root)
    pe = quarter_end(row["date"])
    base = {"ticker": row["ticker"], "bank_id": TICKER_BANK[row["ticker"]], "doc_date": row["date"],
            "title": row["title"], "local_file": row["local_file"], "sha256": row.get("sha256", ""),
            "text_source": src, "period_end": str(pe.date()) if pe is not None else ""}
    if pe is None or not pdf.exists() or "image-only" in src:
        return [base | {"item": "", "status": "not extracted", "reason": src if pe is not None else "no quarter"}]
    rs, text = runs(pdf)
    te = text_period_end(text)
    base["period_end_listing"] = base["period_end"]
    if te is not None:
        base["period_end"] = str(te.date())
        base["period_basis"] = "dates printed in the statement (latest quarter end)" + (
            "" if te == pe else f"; DSE listing date {row['date']} points to {pe.date()} (listing mislabelled)")
        pe = te
    else:
        base["period_basis"] = "DSE listing date (no quarter-end date recognised in the text)"
    two, gf = layout(text)
    q = (pe.month - 1) // 3 + 1
    found: dict[str, dict] = {}
    for r in rs:
        c = classify(r)
        if c is None or c[0] in found:
            continue
        item, st, lab = c
        vals = [v for v in r["values"] if v is not None]
        if st == "bs" and not any(vals):
            continue  # a dash-only line (the values sit on the label's continuation line)
        i, iq, desc = pick(item, st, len(vals), q, two, gf)
        found[item] = {"item": item, "statement": st, "label": lab, "page": r["page"], "n_values": len(vals),
                       "raw_values": ";".join(f"{v:g}" for v in vals), "column": desc,
                       "value_raw": vals[i] if i is not None else np.nan,
                       "quarter_value_raw": vals[iq] if iq is not None else np.nan, "unit_text": r["unit"]}
    unit_txt = next((f["unit_text"] for f in found.values() if f["unit_text"]), "")
    scale, unit_note = (1e6, "page: million") if re.search(r"million|\bm\b", unit_txt, re.I) else \
        (1e3, "page: thousands") if unit_txt else (np.nan, "")
    ta = found.get("total_assets", {}).get("value_raw", np.nan)
    if np.isnan(scale) and pd.notna(ta):
        ref = fy_assets.get((base["bank_id"], pe.year)) or fy_assets.get((base["bank_id"], pe.year - 1))
        for s in (1e6, 1e3):
            if ref and 0.6 <= ta * s / ref <= 1.6:
                scale, unit_note = s, f"inferred from audited total assets ({ta * s / ref:.2f}x)"
    v = {k: f["value_raw"] for k, f in found.items()}

    def close(a, b, tol):
        return pd.notna(a) and pd.notna(b) and b != 0 and abs(a - b) <= tol * abs(b)

    g_nii = close(v.get("interest_income", np.nan) - abs(v.get("interest_expense", np.nan)),
                  v.get("net_interest_income", np.nan), 0.005)
    g_pbt = close(v.get("net_interest_income", np.nan) - abs(v.get("bad_debts_written_off", 0.0))
                  - abs(v.get("impairment_charges", np.nan)) + v.get("non_interest_income", np.nan)
                  - abs(v.get("non_interest_expense", np.nan)), v.get("operating_income", np.nan), 0.01)
    g_np = close(v.get("operating_income", np.nan) - abs(v.get("income_tax", np.nan)), v.get("net_income", np.nan),
                 0.005)
    g_bs = close(v.get("total_liabilities", np.nan) + v.get("total_equity", np.nan), v.get("total_assets", np.nan),
                 0.005)
    ytd_ok = True
    if q > 1 and "interest_income" in found:
        f = found["interest_income"]
        ytd_ok = pd.notna(f["quarter_value_raw"]) and f["value_raw"] > f["quarter_value_raw"]
    out = []
    for k, f in found.items():
        st = f["statement"]
        if st == "pl":
            gate = {"interest_income": g_nii, "interest_expense": g_nii, "net_interest_income": g_nii,
                    "operating_income": g_pbt and g_np, "income_tax": g_np, "net_income": g_np}.get(k, g_nii and g_pbt)
            gate = gate and ytd_ok
            gname = "II - IE = NII; PBT build; PBT - tax = NI; YTD > quarter"
        elif st == "bs":
            gate = g_bs and (k not in {"customer_deposits", "net_loans", "npl_gross"}
                             or (pd.notna(f["value_raw"]) and 0 <= f["value_raw"] < v.get("total_assets", np.inf)))
            gname = "TL + equity = TA"
        else:
            val = f["value_raw"] / 100 if pd.notna(f["value_raw"]) else np.nan
            gate = pd.notna(val) and 0.01 < val < 0.3 and g_bs
            gname = "0.01 < NIM < 0.30 on a balanced statement"
        val = f["value_raw"]
        if st == "ratio":
            val = val / 100 if pd.notna(val) else np.nan
        else:
            val = abs(val) * scale if k in POSITIVE else val * scale
        ok = bool(gate) and pd.notna(val)
        out.append(base | f | {"two_entities": two, "group_first": gf, "quarter": q, "scale": scale,
                               "unit_note": unit_note, "value": val, "gate": gname, "gate_passed": ok,
                               "status": "accepted" if ok else "review",
                               "reason": "" if ok else ("ambiguous column layout" if pd.isna(f["value_raw"])
                                                        else "arithmetic gate failed")})
    if not out:
        out.append(base | {"item": "", "status": "not extracted", "reason": "no BoT-format lines recognised"})
    return out


def extract_all(root: Path | None = None) -> pd.DataFrame:
    """Every NMB / CRDB quarterly statement in the DSE documents index."""
    root = root or find_root()
    d = pd.read_csv(root / "data" / "library" / "dse" / "documents_index.csv")
    d = d[d["ticker"].isin(list(TICKER_BANK)) & d["kind"].eq("quarterly") & d["download_status"].eq("OK")]
    d = d[~d["title"].str.contains("swahili|mwananchi|audited", case=False)]  # duplicates of English statements
    hf = pd.read_csv(path("interim", root) / "inputs_historical_financials.csv")
    hf = hf[hf["Variable"].eq("Total Assets")]
    fy_assets = {(b, pd.Timestamp(p).year): float(x) for b, p, x in
                 zip(hf["Bank ID"], hf["Period End"], hf["value_numeric"], strict=False)
                 if pd.notna(x)}
    rows = [r for doc in d.to_dict("records") for r in extract_document(doc, root, fy_assets)]
    return pd.DataFrame(rows)


def crosscheck(ext: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Accepted values against the held workbook values for the same bank and period (Interim Results H1 / 9M and
    Historical Financials FY for fourth-quarter statements)."""
    root = root or find_root()
    held = []
    for sheet, ptypes in (("interim_results", None), ("historical_financials", {"FY"})):
        h = pd.read_csv(path("interim", root) / f"inputs_{sheet}.csv")
        if ptypes:
            h = h[h["Period Type"].isin(ptypes)]
        held.append(h.assign(sheet=sheet))
    h = pd.concat(held, ignore_index=True)
    h = h[h["Basis"].astype(str).str.upper().eq("GROUP") | ~h["Bank ID"].duplicated()]
    h["period_end"] = pd.to_datetime(h["Period End"]).dt.date.astype(str)
    h["value_held"] = h["value_numeric"]
    inv = {v: k for k, v in WORKBOOK_VARS.items()}
    h["item"] = h["Variable"].map(inv)
    a = ext[ext["status"].eq("accepted")]
    m = a.merge(h[["Bank ID", "period_end", "item", "value_held", "sheet", "Basis"]].rename(columns={"Bank ID":
                                                                                                    "bank_id"}),
                on=["bank_id", "period_end", "item"], how="inner")
    m = m.sort_values("Basis").drop_duplicates(["bank_id", "period_end", "item", "local_file"])
    m["rel_diff"] = (m["value"] - m["value_held"]) / m["value_held"].abs()
    m["match_1pct"] = m["rel_diff"].abs() <= 0.01
    return m[["ticker", "period_end", "item", "value", "value_held", "rel_diff", "match_1pct", "sheet", "Basis",
              "page", "local_file"]]
