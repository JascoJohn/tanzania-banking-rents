"""Plausibility flags and the outlier verification list (prompt 02d sections 1.2, 2.3 and 3; blueprint §6 data audit,
§5 persistence: chapter 5 estimates are sensitive to these bank-years).

Three families of checks on the processed bank-year panel, each flagging a bank-year for verification:

- deposit interest (1.2): deposit interest / total interest expense outside the bank's own median +/- 30 points; cost
  of deposits changing by more than 2x year on year (up, or down to less than half);
- unit consistency (2.3): net profit / total assets outside +/- 15%; total equity <= 0; interest income / total assets
  above 30%; any line changing more than 100x year on year;
- outliers (3): |ROE| > 50%; equity <= 0; NII / average assets > 15%; cost-to-income > 2; cost of deposits outside
  0-15%.

A flag is open until ``config/flag_verifications.csv`` records it as verified or corrected, with evidence; status
"verified, exclude" records a genuine value that is still unfit for estimation (e.g. a part-year period). Open and
excluding flags block acceptance: ``data/processed/bank_year_flags.csv`` sets ``accepted_for_analysis`` False for the bank-year (an
analysis filter for chapter 5; the values themselves stay in the panel, CLAUDE.md 3.7). Verification rows can also
name known cases that no threshold trips; they are listed in the outlier file with their status.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

LINES = ["interest_income", "interest_expense", "net_interest_income", "non_interest_income", "operating_expenses",
         "net_profit", "profit_before_tax", "total_assets", "total_equity", "customer_deposits", "net_loans",
         "interest_expense_deposits"]
FLAG_COLUMNS = ["family", "check", "bank_id", "fiscal_year", "value", "threshold", "inputs", "status", "evidence",
                "blocking", "evidence_class"]


def _lag(w: pd.DataFrame, col: str) -> pd.Series:
    s = w.sort_values(["bank_id", "fiscal_year"])
    prev = s.groupby("bank_id")[col].shift(1)
    gap = s["fiscal_year"] - s.groupby("bank_id")["fiscal_year"].shift(1)
    return prev.where(gap == 1).reindex(w.index)


def _get(w: pd.DataFrame, c: str) -> pd.Series:
    return w[c] if c in w.columns else pd.Series(np.nan, index=w.index)


def cost_of_deposits(w: pd.DataFrame) -> pd.Series:
    """Blueprint §2: interest expense on deposits / average customer deposits (both year-ends needed)."""
    dep = _get(w, "customer_deposits")
    avg = (dep + _lag(w.assign(customer_deposits=dep), "customer_deposits")) / 2
    return _get(w, "interest_expense_deposits") / avg.where(avg != 0)


def deposit_checks(w: pd.DataFrame) -> list[dict]:
    """Prompt 02d 1.2."""
    w = w.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    share = _get(w, "interest_expense_deposits") / _get(w, "interest_expense").where(_get(w, "interest_expense") != 0)
    med = share.groupby(w["bank_id"]).transform("median")
    cod = cost_of_deposits(w)
    ratio = cod / _lag(w.assign(cod=cod), "cod")
    out = []
    for i, r in w.iterrows():
        if pd.notna(share[i]) and abs(share[i] - med[i]) > 0.30:
            out.append({"family": "deposit interest", "check": "deposit share of interest expense vs own median",
                        "bank_id": r["bank_id"], "fiscal_year": r["fiscal_year"], "value": share[i],
                        "threshold": f"median {med[i]:.1%} +/- 30 points",
                        "_vars": ["interest_expense_deposits", "interest_expense"]})
        if pd.notna(ratio[i]) and (ratio[i] > 2 or ratio[i] < 0.5):
            out.append({"family": "deposit interest", "check": "cost of deposits changes more than 2x y/y",
                        "bank_id": r["bank_id"], "fiscal_year": r["fiscal_year"], "value": ratio[i],
                        "threshold": "ratio to prior year in [0.5, 2]",
                        "_vars": ["interest_expense_deposits", "customer_deposits"]})
    return out


def unit_checks(w: pd.DataFrame) -> list[dict]:
    """Prompt 02d 2.3."""
    w = w.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    ta = _get(w, "total_assets").where(_get(w, "total_assets") != 0)
    tests = [("net profit / total assets outside +/-15%", _get(w, "net_profit") / ta, lambda v: abs(v) > 0.15,
              "|ratio| <= 15%", ["net_profit", "total_assets"]),
             ("total equity <= 0", _get(w, "total_equity"), lambda v: v <= 0, "> 0 (or a note)", ["total_equity"]),
             ("interest income / total assets above 30%", _get(w, "interest_income") / ta, lambda v: v > 0.30,
              "<= 30%", ["interest_income", "total_assets"])]
    out = []
    for name, s, bad, thr, vars_ in tests:
        for i in s.index[s.notna()]:
            if bad(s[i]):
                out.append({"family": "unit consistency", "check": name, "bank_id": w.at[i, "bank_id"],
                            "fiscal_year": w.at[i, "fiscal_year"], "value": s[i], "threshold": thr, "_vars": vars_})
    for c in [x for x in LINES if x in w.columns]:
        prev = _lag(w, c)
        r = (w[c].abs() / prev.abs().where(prev != 0)).where(w[c] != 0)
        for i in r.index[r.notna() & ((r > 100) | (r < 0.01))]:
            out.append({"family": "unit consistency", "check": f"{c} changes more than 100x y/y",
                        "bank_id": w.at[i, "bank_id"], "fiscal_year": w.at[i, "fiscal_year"], "value": r[i],
                        "threshold": "ratio to prior year in [0.01, 100]", "_vars": [c]})
    return out


def outlier_checks(w: pd.DataFrame, panel: pd.DataFrame) -> list[dict]:
    """Prompt 02d 3."""
    p = panel.set_index(["bank_id", "fiscal_year"])
    w = w.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    cod = cost_of_deposits(w)
    out = []

    def add(check, i_or_key, value, thr, vars_):
        b, y = i_or_key
        out.append({"family": "outlier", "check": check, "bank_id": b, "fiscal_year": y, "value": value,
                    "threshold": thr, "_vars": vars_})

    for (b, y), r in p.iterrows():
        if pd.notna(r["roe"]) and abs(r["roe"]) > 0.5:
            add("|ROE| > 50%", (b, y), r["roe"], "|ROE| <= 50%",
                ["profit_attributable", "net_profit", "equity_attributable", "total_equity"])
        if pd.notna(r["nii_to_avg_assets"]) and r["nii_to_avg_assets"] > 0.15:
            add("NII / average assets > 15%", (b, y), r["nii_to_avg_assets"], "<= 15%",
                ["net_interest_income", "total_assets"])
        if pd.notna(r["cost_to_income"]) and r["cost_to_income"] > 2:
            add("cost-to-income > 2", (b, y), r["cost_to_income"], "<= 2",
                ["operating_expenses", "net_interest_income", "non_interest_income"])
    for i, r in w.iterrows():
        if pd.notna(r.get("total_equity")) and r["total_equity"] <= 0:
            add("equity <= 0", (r["bank_id"], r["fiscal_year"]), r["total_equity"], "> 0", ["total_equity"])
        if pd.notna(cod[i]) and not 0 <= cod[i] <= 0.15:
            add("cost of deposits outside 0-15%", (r["bank_id"], r["fiscal_year"]), cod[i], "0-15%",
                ["interest_expense_deposits", "customer_deposits"])
    return out


def source_refs(canon: pd.DataFrame, hf: pd.DataFrame) -> dict[tuple, str]:
    """(bank, year, variable) -> 'DOC p.N' for the FY value of the bank's own entity (original row first)."""
    page = dict(zip(hf["Row ID"], hf["Page / Note"], strict=False)) if len(hf) else {}
    c = canon[(canon["period_type"] == "FY") & canon["value"].notna()].sort_values(["is_restated", "provenance"])
    out: dict[tuple, str] = {}
    for _, r in c.iterrows():
        k = (r["bank_id"], int(r["fiscal_year"]), r["canonical_name"])
        if k in out:
            continue
        if r["provenance"] == "inputs_workbook":
            pg = page.get(r["source_row_id"])
            out[k] = f"{r['doc_id']} {pg if isinstance(pg, str) else ''} ({r['source_row_id']})".strip()
        else:
            m = re.search(r"\bp\.?\s?(\d+)", str(r.get("notes")))
            src = r.get("source_document") if isinstance(r.get("source_document"), str) else "Phase 1"
            out[k] = f"{src}{' p.' + m.group(1) if m else ''} ({r['source_row_id']})"
    return out


def reconciled_deposit_years(root: Path, w: pd.DataFrame) -> set[tuple[str, int]]:
    """Bank-years whose deposit interest in the panel is reconciled: NMB/CRDB rows in
    ``reports/validation/deposit_interest_reconciliation.csv`` (reconciled, or corrected to the reconciled value), and
    B03-B07 values admitted by the note reconciliation in ``data/extraction/ledger.csv``."""
    have = (w.set_index(["bank_id", "fiscal_year"])["interest_expense_deposits"].dropna()
            if "interest_expense_deposits" in w.columns else pd.Series(dtype=float))
    refs: list[tuple] = []
    f = path("validation", root) / "deposit_interest_reconciliation.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d["variable"] == "Interest Expense on Customer Deposits") & d["reconciled_value"].notna()
              & (d["status"].str.startswith("reconciled") | (d["status"] == "corrected"))]
        refs += list(zip(d["bank_id"], d["fiscal_year"], d["reconciled_value"], strict=False))
    led = root / "data" / "extraction" / "ledger.csv"
    if led.exists():
        lg = pd.read_csv(led)
        lg = lg[(lg["variable"] == "Interest Expense on Customer Deposits")
                & lg["gate"].astype(str).str.contains("sum to audited total")]
        refs += list(zip(lg["bank_id"], lg["fiscal_year"], lg["value"], strict=False))
    out = set()
    for b, y, v in refs:
        cur = have.get((b, int(y)))
        if cur is not None and abs(cur - v) <= 0.001 * abs(v):
            out.add((b, int(y)))
    return out


def read_verifications(root: Path) -> pd.DataFrame:
    f = path("config", root) / "flag_verifications.csv"
    cols = ["check", "bank_id", "fiscal_year", "status", "evidence", "verified_on"]
    return pd.read_csv(f, dtype={"fiscal_year": int}).reindex(columns=cols) if f.exists() else pd.DataFrame(
        columns=cols)


def attach(rows: list[dict], w: pd.DataFrame, refs: dict, ver: pd.DataFrame, family_known: str | None = None,
           reconciled: set | None = None) -> pd.DataFrame:
    """Inputs with values and source pages, verification status, blocking flag."""
    wi = w.set_index(["bank_id", "fiscal_year"])
    vmap = {(r["check"], r["bank_id"], int(r["fiscal_year"])): r for _, r in ver.iterrows()}
    out = []
    for r in rows:
        b, y = r["bank_id"], int(r["fiscal_year"])
        parts = []
        for v in r.pop("_vars"):
            for yy in (y, y - 1) if v in {"total_assets", "total_equity", "customer_deposits",
                                         "equity_attributable"} and r["family"] != "unit consistency" else (y,):
                val = wi[v].get((b, yy)) if v in wi.columns else None
                if val is not None and pd.notna(val):
                    parts.append(f"{v}[{yy}]={val:,.0f} <{refs.get((b, yy, v), 'source not mapped')}>")
        hit = vmap.get((r["check"], b, y))
        status = hit["status"] if hit is not None else "to verify"
        evidence = hit["evidence"] if hit is not None else ""
        years = (y, y - 1) if "cost of deposits" in r["check"] and "changes" in r["check"] else (y,)
        if (hit is None and r["family"] == "deposit interest" and reconciled is not None
                and all((b, yy) in reconciled for yy in years)):
            status, evidence = "verified", ("deposit interest reconciled: note lines sum to the audited total interest "
                                            "expense (0.1%) for FY" + "/FY".join(str(x) for x in years))
        out.append(r | {"fiscal_year": y, "inputs": "; ".join(parts), "status": status, "evidence": evidence,
                        "blocking": status not in {"verified", "corrected"}, "evidence_class": "Derived"})
    df = pd.DataFrame(out, columns=FLAG_COLUMNS)
    if family_known is not None:  # known cases named in the verification file but tripping no threshold
        have = set(zip(df["check"], df["bank_id"], df["fiscal_year"], strict=False))
        extra = [{"family": family_known, "check": k[0], "bank_id": k[1], "fiscal_year": k[2], "value": np.nan,
                  "threshold": "known case (prompt 02d 3)", "inputs": "", "status": v["status"],
                  "evidence": v["evidence"], "blocking": v["status"] not in {"verified", "corrected"},
                  "evidence_class": "Derived"}
                 for k, v in vmap.items() if k[0].startswith("known case") and k not in have]
        df = pd.concat([df, pd.DataFrame(extra, columns=FLAG_COLUMNS)], ignore_index=True)
    return df.sort_values(["bank_id", "fiscal_year", "family", "check"]).reset_index(drop=True)


def flags(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(plausibility flags, outliers, bank-year acceptance table)."""
    from tzbank.chapters.ch04.profitability import dupont_panel
    from tzbank.clean.normalise import load_canonical
    from tzbank.ingest.inputs_workbook import load_interim

    root = root or find_root()
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", low_memory=False)
    refs = source_refs(load_canonical(root), load_interim("Historical Financials", root))
    ver = read_verifications(root)
    plaus = attach(deposit_checks(w) + unit_checks(w), w, refs, ver, reconciled=reconciled_deposit_years(root, w))
    outl = attach(outlier_checks(w, dupont_panel(w)), w, refs, ver, family_known="outlier")
    allf = pd.concat([plaus, outl], ignore_index=True)
    openf = allf[allf["blocking"]]
    acc = w[["bank_id", "fiscal_year"]].copy()
    g = openf.groupby(["bank_id", "fiscal_year"])["check"].agg(lambda s: "; ".join(sorted(set(s))))
    acc["open_flags"] = [g.get((b, y), "") for b, y in zip(acc["bank_id"], acc["fiscal_year"], strict=False)]
    acc["n_open_flags"] = acc["open_flags"].str.count(";") + acc["open_flags"].ne("").astype(int)
    acc["accepted_for_analysis"] = acc["n_open_flags"] == 0
    acc["evidence_class"] = "Derived"
    return plaus, outl, acc


def write_flags(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    plaus, outl, acc = flags(root)
    v = path("validation", root)
    return [write_csv(plaus, v / "plausibility_flags.csv", require_evidence=True),
            write_csv(outl, v / "outliers.csv", require_evidence=True),
            write_csv(acc, path("processed", root) / "bank_year_flags.csv", require_evidence=True)]
