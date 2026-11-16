"""D18 workbook session for prompt 02d (v0.11): owner data-review fixes.

Usage (after ``workbook.backup()`` and a commit):
    python -m tzbank.extract.session_02d
Then: ``python -m tzbank manifest --note NMB_CRDB_Valuation_Inputs.xlsx "<change>"``, ``run all``, commit.

1. Deposit interest, NMB/CRDB (section 1): every row corrected by ``reports/validation/deposit_interest_reconciliation
   .csv`` takes the reconciled value (the old value is kept in Notes); every accepted row records how it reconciles.
2. CRDB FY2009-FY2010 total equity and operating expenses (section 4): read from the FY2010 annual report (original)
   and the FY2011 annual report's restated comparatives (D21), each re-checked against a printed identity on its page.
3. Macroeconomic Data (section 5.4): D34 primary long-bond yield for both valuation dates; the 15- and 20-year
   single-auction yields for 2025-12-31 fill their TO COLLECT rows.
4. Sources (section 6, D35): the NPS reports' PDF modification dates move to 'Lower-Bound Publication Date'.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import openpyxl
import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.pdftext import _doc
from tzbank.extract.workbook import _append, _header, _last_row, _next_id

VERSION = "v0.11"
TODAY = "2026-11-06"


def _printed(file: Path, page: int, values: list[int]) -> bool:
    text = _doc(str(file))[page - 1].get_text()
    nums = {int(t.replace(",", "")) for t in re.findall(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d,.])\d+(?![\d,.])",
                                                          text)}
    return all(abs(v) in nums for v in values)


# CRDB FY2009-FY2010 (Group). Identity on the page: total assets - total liabilities = total shareholders' funds;
# income statement lines sum to profit before tax. Operating expenses = general and administrative expenses + staff
# costs (the FY2011 workbook definition, impairment excluded).
CRDB_ROWS = [
    # (fiscal year, variable, doc, file, page, unit, value parts, identity parts, identity total, restated)
    (2010, "Total Equity", "CRDB-AR-FY2010", "CRDB/2010.pdf", 33, 1_000, [240_921_695],
     [2_305_224_620, -2_064_302_925], 240_921_695, False),
    (2009, "Total Equity", "CRDB-AR-FY2010", "CRDB/2010.pdf", 33, 1_000, [207_774_139],
     [1_854_867_005, -1_647_092_866], 207_774_139, False),
    (2010, "Operating Expenses", "CRDB-AR-FY2010", "CRDB/2010.pdf", 32, 1_000, [63_360_635, 46_180_800],
     [104_648_147, 46_590_928, 22_080_586, 2_060_367, -63_360_635, -46_180_800, -201_833], 65_636_760, False),
    (2009, "Operating Expenses", "CRDB-AR-FY2010", "CRDB/2010.pdf", 32, 1_000, [44_931_239, 40_409_415],
     [91_794_190, 39_110_882, 15_243_855, 1_113_262, -44_931_239, -40_409_415], 61_921_535, False),
    (2010, "Total Equity", "CRDB-AR-FY2011", "CRDB/2011.pdf", 40, 1_000_000, [233_511], [2_305_402, -2_071_891],
     233_511, True),
    (2009, "Total Equity", "CRDB-AR-FY2011", "CRDB/2011.pdf", 40, 1_000_000, [201_501], [1_854_679, -1_653_178],
     201_501, True),
    (2010, "Operating Expenses", "CRDB-AR-FY2011", "CRDB/2011.pdf", 39, 1_000_000, [63_361, 47_683],
     [104_648, 46_591, 22_081, 2_060, -63_361, -47_683, -202], 64_134, True),
]


def crdb_rows(root: Path) -> list[dict]:
    """Checked values for the CRDB FY2009-FY2010 rows (raises if a printed identity no longer holds)."""
    out = []
    for fy, var, doc, file, page, unit, parts, ident, total, restated in CRDB_ROWS:
        f = path("documents", root) / file
        if not _printed(f, page, [*parts, *ident, total]):
            raise ValueError(f"{file} p.{page}: a value is not printed on the page")
        if abs(sum(ident) - total) > 1:
            raise ValueError(f"{file} p.{page}: identity does not hold ({sum(ident)} vs {total})")
        u = "TZS '000" if unit == 1_000 else "TZS million"
        gate = ("total assets - total liabilities = total shareholders' funds" if var == "Total Equity" else
                "income statement lines sum to profit before tax")
        label = ("TOTAL SHAREHOLDERS' FUNDS" if var == "Total Equity" and not restated else "TOTAL CAPITAL"
                 if var == "Total Equity" else "General and administrative expenses + Staff costs")
        note = (f"Extracted by tzbank.extract.session_02d (prompt 02d section 4; gate: {gate} on the page); label "
                f"'{label}'; Group column, {u}; extraction_method=gated")
        if restated:
            note = (f"RESTATED in {doc} comparative (column marked '(Restated)'; original from CRDB-AR-FY2010) | "
                    + note)
        out.append({"fiscal_year": fy, "Variable": var, "Doc ID": doc, "Page / Note": f"PDF p.{page}",
                    "Value": float(sum(parts) * unit), "Status": "COMPUTED" if len(parts) > 1 else "REPORTED",
                    "Confidence": "MEDIUM" if restated or doc[-4:] != str(fy) else "HIGH", "Notes": note,
                    "restated": restated})
    return out


def _dt(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def run() -> dict[str, int]:
    root = find_root()
    f = path("inputs_workbook", root)
    rec = pd.read_csv(path("validation", root) / "deposit_interest_reconciliation.csv")
    coe = pd.read_csv(path("processed", root) / "coe_inputs_annual.csv")
    used = pd.read_csv(path("processed", root) / "coe_long_bond_auctions_used.csv")
    wb = openpyxl.load_workbook(f)
    hf, ma, src, me = wb["Historical Financials"], wb["Macroeconomic Data"], wb["Sources"], wb["Read Me"]
    counts = {"deposit_corrected": 0, "deposit_reconciled_note": 0, "crdb_filled": 0, "crdb_restated_added": 0,
              "macro_rows": 0, "sources_added": 0, "sources_lower_bound": 0}

    # 1. deposit interest
    h = _header(hf)
    row_of = {hf.cell(r, 1).value: r for r in range(2, _last_row(hf) + 1)}
    for _, x in rec[rec["variable"].str.startswith("Interest Expense on")].iterrows():
        r = row_of[x["row_id"]]
        old_note = hf.cell(r, h["Notes"]).value or ""
        if x["status"] == "corrected" and x["acts_on_workbook"]:
            old = hf.cell(r, h["Value"]).value
            hf.cell(r, h["Value"]).value = float(x["reconciled_value"])
            hf.cell(r, h["Page / Note"]).value = f"PDF p.{int(x['page'])}"
            hf.cell(r, h["Status"]).value = "COMPUTED" if x["variable"].endswith("Customer Deposits") else "REPORTED"
            hf.cell(r, h["Notes"]).value = (
                f"CORRECTED (prompt 02d section 1): was {old:,.0f}, accepted by the weak '<= total' check only; "
                f"{x['detail']}; {x['file']} p.{int(x['page'])}, note lines sum to the audited total interest expense "
                f"{x['audited_total']:,.0f} (0.1%) | previous note: {old_note}")
            counts["deposit_corrected"] += 1
        elif str(x["status"]).startswith("reconciled") and "deposit interest <= total" in old_note:
            hf.cell(r, h["Notes"]).value = (f"Reconciled (prompt 02d section 1): {x['status']}; {x['detail']} | "
                                            f"{old_note}")
            counts["deposit_reconciled_note"] += 1

    # 2. CRDB FY2009-FY2010
    hold = {}
    for r in range(2, _last_row(hf) + 1):
        pe = hf.cell(r, h["Period End"]).value
        if hf.cell(r, h["Bank ID"]).value == "B01" and hasattr(pe, "year") and hf.cell(r, h["Period Type"]).value \
                == "FY" and hf.cell(r, h["Entity"]).value == "Group":
            hold.setdefault((pe.year, hf.cell(r, h["Variable"]).value), r)
    for x in crdb_rows(root):
        vals = {"Bank ID": "B01", "Bank": "CRDB Bank", "Entity": "Group", "Basis": "GROUP", "Period Type": "FY",
                "Period End": dt.datetime(x["fiscal_year"], 12, 31), "Variable": x["Variable"], "Value": x["Value"],
                "Unit": "TZS", "Status": x["Status"], "Doc ID": x["Doc ID"], "Page / Note": x["Page / Note"],
                "Confidence": x["Confidence"], "Notes": x["Notes"]}
        r = hold.get((x["fiscal_year"], x["Variable"]))
        if not x["restated"] and r is not None and hf.cell(r, h["Value"]).value in (None, ""):
            prev = hf.cell(r, h["Notes"]).value
            for k, v in vals.items():
                hf.cell(r, h[k]).value = v
            hf.cell(r, h["Notes"]).value = f"{x['Notes']} | previous note: {prev}" if prev else x["Notes"]
            counts["crdb_filled"] += 1
        elif x["restated"]:
            vals["Row ID"] = f"HF-{_next_id(hf, 'HF'):04d}"
            _append(hf, vals)
            counts["crdb_restated_added"] += 1
        else:
            raise ValueError(f"CRDB FY{x['fiscal_year']} {x['Variable']}: no empty placeholder to fill")

    # 3. Macroeconomic Data and their Sources
    hs = _header(src)
    if "Lower-Bound Publication Date" not in hs:
        c = src.max_column + 1
        src.cell(1, c).value = "Lower-Bound Publication Date"
        hs["Lower-Bound Publication Date"] = c
    have_docs = {src.cell(r, 1).value: r for r in range(2, _last_row(src) + 1)}

    def add_source(doc: str, title: str, date: str, note: str) -> None:
        if doc in have_docs:
            return
        r = _last_row(src) + 1
        for k, v in {"Doc ID": doc, "Owner": "BoT", "Document Title": title, "Document Type": "Auction results",
                     "Period Covered": date, "Primary / Secondary": "PRIMARY",
                     "URL": "https://www.bot.go.tz/TBonds", "Local File Path": "data/raw/bot/auctions/",
                     "Notes": note, "Period End": _dt(date), "Actual Publication Date": _dt(date),
                     "Date Evidence": "auction result published by BoT on the auction date (data/raw/bot/auctions)"
                     }.items():
            src.cell(r, hs[k]).value = v
        have_docs[doc] = r
        counts["sources_added"] += 1

    hm = _header(ma)
    ma_row = {ma.cell(r, 1).value: r for r in range(2, _last_row(ma) + 1)}
    single = {("FY2025", "15"): ("MA-0001", 688), ("FY2025", "20"): ("MA-0002", 693)}
    for (label, tenor), (rid, bond) in single.items():
        u = used[(used["date_label"] == label) & (used["series"] == "d34_primary") & (used["bond_number"] == bond)]
        if u.empty:
            raise ValueError(f"{rid}: bond {bond} not among the D34 auctions for {label}")
        u = u.iloc[0]
        r = ma_row[rid]
        if ma.cell(r, hm["Value"]).value not in (None, ""):
            continue
        doc = f"BOT-TBOND-{tenor}Y-{pd.Timestamp(u['auction_date']).strftime('%b%y').upper()}"
        add_source(doc, f"BoT Treasury bond auction result, {tenor}-year bond No. {bond}, {u['auction_date']}",
                   u["auction_date"], "Auction result (WAYTM) from the BoT TBonds endpoint; prompt 02d section 5.")
        prev = ma.cell(r, hm["Notes"]).value
        for k, v in {"Value": float(u["waytm"]), "Observation / Auction Date": _dt(u["auction_date"]),
                     "Status": "REPORTED", "Doc ID": doc, "Confidence": "HIGH",
                     "Notes": f"WAYTM {u['waytm']:.4%}, bond No. {bond} (issue {int(u['issue_number'])}), auction "
                              f"{u['auction_date']}; enters the D34 primary for 2025-12-31 | previous note: {prev}"
                     }.items():
            ma.cell(r, hm[k]).value = v
        counts["macro_rows"] += 1
    for vlabel, asat in (("valuation_historical", "2025-12-31"), ("valuation_current", "2026-10-31")):
        p = coe[(coe["date_label"] == vlabel) & (coe["series"] == "d34_primary")].iloc[0]
        alts = coe[(coe["date_label"] == vlabel) & (coe["series"] != "d34_primary")]
        u = used[(used["date_label"] == vlabel) & (used["series"] == "d34_primary")]
        last = u["auction_date"].max()
        doc = f"BOT-TBOND-D34-{'HIST' if vlabel.endswith('historical') else 'CURR'}"
        add_source(doc, f"BoT Treasury bond auction results, tenor >= 15 years, final quarter to {asat} (D34)",
                   last, "Constituent auctions listed in data/processed/coe_long_bond_auctions_used.csv; dated by "
                         "the last constituent auction.")
        singles = "; ".join(f"bond {int(a['bond_number'])} ({int(a['tenor_years'])}y, issue {int(a['issue_number'])}"
                            f") {a['auction_date']} WAYTM {a['waytm']:.4%}" for _, a in u.iterrows())
        alt = "; ".join(f"{a['series']} {a['waytm']:.4%}" for _, a in alts.iterrows())
        note = (f"D34 primary: simple average of the WAYTM of all auctions with tenor >= 15 years in the final quarter "
                f"to {asat} ({p['n_auctions']:.0f} auctions). Single-auction values: {singles}. Sensitivity series: "
                f"{alt}.")
        if vlabel == "valuation_current":
            note += " DSE secondary-market yield on bond No. 722 (25y) at 30 Oct 2026: 10.5262% (MA-0046)."
        exists = [r for r in range(2, _last_row(ma) + 1) if ma.cell(r, hm["Doc ID"]).value == doc]
        if exists:
            continue
        _append(ma, {"Row ID": f"MA-{_next_id(ma, 'MA'):04d}", "Category": "Rates",
                     "Variable": "TZS Long-Bond Yield (D34: tenor >= 15y, final-quarter average)",
                     "As At (Valuation Date)": _dt(asat), "Value": float(p["waytm"]), "Unit": "ratio",
                     "Observation / Auction Date": _dt(last), "Status": "COMPUTED", "Doc ID": doc,
                     "Confidence": "HIGH", "Notes": note})
        counts["macro_rows"] += 1

    # 4. Sources: NPS lower bounds (D35)
    for r in range(2, _last_row(src) + 1):
        did = str(src.cell(r, 1).value)
        if did.startswith("BOT-NPSAR-") and src.cell(r, hs["Estimated Publication Date"]).value is not None:
            src.cell(r, hs["Lower-Bound Publication Date"]).value = src.cell(r, hs["Estimated Publication Date"]).value
            src.cell(r, hs["Estimated Publication Date"]).value = None
            basis = src.cell(r, hs["Estimate Basis"]).value or ""
            src.cell(r, hs["Estimate Basis"]).value = ("D35 lower bound only (not a posting date; ineligible at "
                                                       "valuation dates until confirmed): " + basis)
            counts["sources_lower_bound"] += 1

    lines = [f"{VERSION} ({TODAY}, prompt 02d, D18 working workbook)",
             f"Historical Financials: {counts['deposit_corrected']} NMB/CRDB deposit-interest rows corrected to the "
             f"reconciled note values (old values kept in Notes; reports/validation/deposit_interest_reconciliation"
             f".csv); {counts['deposit_reconciled_note']} accepted rows annotated with their reconciliation; CRDB "
             f"FY2009-FY2010 total equity and operating expenses: {counts['crdb_filled']} placeholders filled from the "
             f"FY2010 report, {counts['crdb_restated_added']} restated rows from the FY2011 report (D21).",
             f"Macroeconomic Data: {counts['macro_rows']} rows (D34 primary long-bond yield for both valuation dates; "
             "15- and 20-year auction yields for 2025-12-31). Sources: "
             f"{counts['sources_added']} BoT auction documents; {counts['sources_lower_bound']} NPS reports moved to "
             "'Lower-Bound Publication Date' (D35)."]
    r = me.max_row + 2
    for line in lines:
        me.cell(r, 1).value = line
        r += 1
    wb.save(f)
    return counts


if __name__ == "__main__":
    print(run())
