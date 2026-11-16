"""Quarterly BoT-format statements, Q4 2025 - Q3 2026 (prompt 20 item 2; D89).

Input (raw, MANIFEST-hashed): ``data/raw/manual_entries/quarterly/quarterly_transcribed.csv``, every printed line of the
eight statements in ``data/raw/quarterly/`` (git-ignored PDFs; NMB Q3 2026 and CRDB Q2 2026 scanned, OCR verified
against the page image), with page references. Fields map by the BoT template's fixed line numbers.

Outputs: ``data/processed/quarterly/quarterly_panel.csv`` (long, with check_status and vintage: the Q4 2025 statements
are "first published", the audited FY2025 annual figures are added as "audited"; the model uses audited), and tables
``ch13_qtr_checks``, ``ch13_qtr_ttm`` (trailing twelve months at 30 Sep 2026), ``ch13_qtr_monitor`` (r0-r9 current
values refreshed to the latest quarter), ``ch13_qtr_five`` (five-quarter exhibit data), ``ch13_qtr_ldr``,
``ch13_qtr_eps_split``, ``ch13_qtr_roll`` (the 31 Oct 2026 valuation with year 1 re-based on the quarter: owner
decision, not adopted); chart ``ch13_qtr_five.png``; manifest ``ch13_qtr.json``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import write_csv, write_json, write_text

ENTRY = Path("data/raw/manual_entries/quarterly/quarterly_transcribed.csv")
SEED = Path("data/raw/quarterly/quarterly_register_seed.csv")
BID = {"NMB": "B02", "CRDB": "B01"}
BS = {"1": "cash", "2": "bot_balances", "3": "govt_securities", "4": "interbank_balances", "9": "interbank_loans",
      "11": "net_loans", "16": "total_assets", "18": "customer_deposits", "28": "borrowings", "29": "total_liabilities",
      "30": "net_assets", "36": "minority_interest", "37": "total_shareholders_funds", "39": "npl", "40": "allowances"}
IS = {"1": "interest_income", "2": "interest_expense", "3": "nii", "4": "bad_debts", "5": "impairment",
      "6": "non_interest_income", "6.1": "fx_dealings", "6.2": "fees_commissions", "6.3": "dividend_income",
      "6.4": "other_operating_income", "7": "non_interest_expense", "7.1": "staff", "7.2": "fee_expense",
      "7.3": "other_opex", "8": "pbt", "9": "tax", "10": "pat"}
LABEL = [("income_statement", r"basic earnings per share", "eps_basic"),
         ("income_statement", r"diluted earnings per share", "eps_diluted"),
         ("income_statement", r"number of employees", "employees"),
         ("income_statement", r"number of branches", "branches"),
         ("income_statement", r"other comprehensive income", "oci"),
         ("indicators", r"shareholders funds to total assets", "sf_to_assets"),
         ("indicators", r"non performing loans to total gross loans", "npl_ratio"),
         ("indicators", r"gross loans and advances to total deposits", "gross_ldr"),
         ("indicators", r"loans and advances to total assets", "loans_to_assets"),
         ("indicators", r"earnings assets to total assets", "earning_assets_share"),
         ("capital", r"^national minimum.*cet1", "min_cet1_ratio"),
         ("capital", r"^national minimum.*tier-?1", "min_tier1_ratio"),
         ("capital", r"^national minimum.*total capital", "min_total_capital_ratio"),
         ("capital", r"^national minimum.*conservation", "min_conservation_buffer"),
         ("capital", r"^capital conservation buffer", "conservation_buffer"),
         ("capital", r"^common equity tier 1 ratio|^cet1 to total rwa", "cet1_ratio"),
         ("capital", r"^tier 1 ratio|^tier-1 capital to total rwa", "tier1_ratio"),
         ("capital", r"^total capital ratio|^total capital to total rwa", "total_capital_ratio"),
         ("capital", r"^total risk.weighted assets", "rwa")]
SHARES = {"NMB": (500e6, 5_000e6, pd.Timestamp("2026-08-24")), "CRDB": (2_611.7e6, 2_611.7e6, None)}


def load(root: Path) -> pd.DataFrame:
    q = pd.read_csv(root / ENTRY, dtype=str)
    q["value"] = pd.to_numeric(q["value"], errors="coerce")
    q["period_end"] = pd.to_datetime(q["period_end"])
    q["value_date"] = pd.to_datetime(q["value_date"], errors="coerce").fillna(q["period_end"])
    ln = q["line_no"].fillna("").str.strip().str.rstrip(".")
    lab = q["label"].fillna("").str.strip().str.lower()
    f = np.where(q["statement"] == "balance_sheet", ln.map(BS), np.where(q["statement"] == "income_statement",
                                                                           ln.map(IS), None))
    q["field"] = pd.Series(f, index=q.index).astype(object)
    for st, pat, name in LABEL:
        m = (q["statement"] == st) & q["field"].isna() & lab.str.contains(pat, regex=True)
        q.loc[m, "field"] = name
    q["vintage"] = np.where(q["period_end"].dt.month == 12, "first published (unaudited)", "as published")
    return q


def get(q, bank, field, pe, col, basis="group"):
    x = q[(q["bank"] == bank) & (q["field"] == field) & (q["period_end"] == pd.Timestamp(pe))
          & (q["column_type"] == col) & (q["basis"].isin([basis, "single"]))]
    return float(x["value"].iloc[0]) if len(x) else np.nan


def _row(rows, bank, basis, pe, check, computed, ref, tol, note=""):
    dev = computed / ref - 1 if pd.notna(computed) and pd.notna(ref) and ref else np.nan
    st = "not computable" if pd.isna(dev) else ("pass" if abs(dev) <= tol else "fail")
    rows.append({"bank": bank, "basis": basis, "period_end": pd.Timestamp(pe).date().isoformat(), "check": check,
                 "computed": computed, "reference": ref, "deviation": dev, "tolerance": tol, "status": st,
                 "note": note, "evidence_class": "Derived"})


def checks(q: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    pes = sorted(q["period_end"].unique())
    for bank in BID:
        for basis in ("group", "bank"):
            for pe in pes:
                for col in ("current_quarter", "previous_quarter"):
                    g = lambda f, c=col, b=bank, d=pe, s=basis: get(q, b, f, d, c, s)  # noqa: E731
                    _row(rows, bank, basis, pe, f"assets = liabilities + equity ({col})",
                         g("total_liabilities") + g("total_shareholders_funds"), g("total_assets"), 0.001)
                for col in ("current_quarter", "ytd_current"):
                    g = lambda f, c=col, b=bank, d=pe, s=basis: get(q, b, f, d, c, s)  # noqa: E731
                    _row(rows, bank, basis, pe, f"NII = interest income + interest expense ({col})",
                         g("interest_income") + g("interest_expense"), g("nii"), 0.001)
                    _row(rows, bank, basis, pe, f"PAT = PBT + tax ({col})", g("pbt") + g("tax"), g("pat"), 0.001)
            for i in range(1, len(pes)):
                a, z = pes[i - 1], pes[i]
                if pd.Timestamp(z).month == 3:
                    continue
                for f in ("nii", "pat", "fees_commissions", "impairment"):
                    _row(rows, bank, basis, z, f"YTD {f}: previous YTD + this quarter",
                         get(q, bank, f, a, "ytd_current", basis) + get(q, bank, f, z, "current_quarter", basis),
                         get(q, bank, f, z, "ytd_current", basis), 0.002)
            for i in range(1, len(pes)):
                a, z = pes[i - 1], pes[i]
                e0, e1 = get(q, bank, "total_shareholders_funds", a, "current_quarter", basis), \
                    get(q, bank, "total_shareholders_funds", z, "current_quarter", basis)
                pat = get(q, bank, "pat", z, "current_quarter", basis)
                oci = get(q, bank, "oci", z, "current_quarter", basis)
                resid = e1 - e0 - pat - (0.0 if pd.isna(oci) else oci)
                rows.append({"bank": bank, "basis": basis, "period_end": pd.Timestamp(z).date().isoformat(),
                             "check": "equity roll-forward: closing - opening - PAT - OCI (= -dividends + other)",
                             "computed": resid, "reference": e0, "deviation": resid / e0 if e0 else np.nan,
                             "tolerance": 0.01, "status": "pass" if abs(resid / e0) <= 0.01 else
                             ("dividend in quarter" if resid < 0 and pd.Timestamp(z).month == 6 else "fail"),
                             "note": "a negative residual in Q2 is the final dividend paid", "evidence_class": "Derived"})
        for pe in pes:
            eps = get(q, bank, "eps_basic", pe, "ytd_current")
            pat = get(q, bank, "pat", pe, "ytd_current")
            pre, post, sd = SHARES[bank]
            sh = post if sd is not None and pd.Timestamp(pe) >= sd else pre
            m = pd.Timestamp(pe).month
            ann = abs(eps * sh / 1e6 / (pat * 12 / m) - 1) < abs(eps * sh / 1e6 / pat - 1)
            _row(rows, bank, "group", pe, "EPS x shares = PAT (YTD; annualised where the bank prints it so)",
                 eps * sh / 1e6, pat * 12 / m if ann else pat, 0.03,
                 f"shares {sh / 1e6:,.1f}m; EPS printed {'annualised (x 12 / months)' if ann else 'year to date'}")
    seed = pd.read_csv(root / SEED)
    for r in seed.itertuples():
        x = q[(q["source_file"] == r.suggested_name) & (q["column_type"] == "current_quarter")
              & (q["value"] == float(r.spot_check_value_tzs_m))]
        rows.append({"bank": r.bank, "basis": "first column", "period_end": r.period_end,
                     "check": f"seed spot check: {r.spot_check_label}", "computed": float(r.spot_check_value_tzs_m),
                     "reference": float(r.spot_check_value_tzs_m) if len(x) else np.nan, "deviation": 0.0 if len(x)
                     else np.nan, "tolerance": 0.0, "status": "pass" if len(x) else "fail", "note": "",
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def panel(q: pd.DataFrame, chk: pd.DataFrame, root: Path) -> pd.DataFrame:
    fails = chk[chk["status"] == "fail"].groupby(["bank", "period_end"])["check"].apply("; ".join)
    p = q.copy()
    p["check_status"] = [("fail: " + fails[(b, d.date().isoformat())]) if (b, d.date().isoformat()) in fails.index
                         else "ok" for b, d in zip(p["bank"], p["period_end"], strict=True)]
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    aud = []
    for b, bid in BID.items():
        r = w[(w["bank_id"] == bid) & (w["fiscal_year"] == 2025)].iloc[0]
        for f, col in (("total_assets", "total_assets"), ("customer_deposits", "customer_deposits"),
                       ("net_loans", "net_loans"), ("pat", "net_profit"), ("pbt", "profit_before_tax")):
            aud.append({"bank": b, "basis": "group", "period_end": pd.Timestamp("2025-12-31"),
                        "column_type": "annual_audited", "field": f, "value": float(r[col]) / 1e6,
                        "unit": "TZS_millions", "source_file": "inputs workbook (audited FY2025)",
                        "vintage": "audited", "check_status": "ok"})
    p = pd.concat([p, pd.DataFrame(aud)], ignore_index=True)
    cols = ["bank", "basis", "period_end", "column_type", "value_date", "statement", "line_no", "label", "field",
            "value", "unit", "source_file", "page", "extraction", "verified", "vintage", "check_status"]
    return p[[c for c in cols if c in p.columns]].assign(evidence_class="Observed")


def ttm(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing twelve months to 30 Sep 2026 (group): FY2025 (Q4 statement YTD) + 9M 2026 - 9M 2025."""
    rows = []
    for b in BID:
        def t(f, b=b):
            return (get(q, b, f, "2025-12-31", "ytd_current") + get(q, b, f, "2026-09-30", "ytd_current")
                    - get(q, b, f, "2026-09-30", "ytd_prior"))
        e = [get(q, b, "total_shareholders_funds", "2025-12-31", "previous_quarter"),
             get(q, b, "total_shareholders_funds", "2026-09-30", "current_quarter")]
        mi = [get(q, b, "minority_interest", "2025-12-31", "previous_quarter"),
              get(q, b, "minority_interest", "2026-09-30", "current_quarter")]
        eq = np.nanmean([x - (0 if pd.isna(m) else m) for x, m in zip(e, mi, strict=True)])
        loans = np.nanmean([get(q, b, "net_loans", "2025-12-31", "previous_quarter"),
                            get(q, b, "net_loans", "2026-09-30", "current_quarter")])
        assets = [get(q, b, "total_assets", "2025-12-31", "previous_quarter"),
                  get(q, b, "total_assets", "2026-09-30", "current_quarter")]
        ea = [get(q, b, "earning_assets_share", "2025-12-31", "current_quarter"),
              get(q, b, "earning_assets_share", "2026-09-30", "current_quarter")]
        earning = np.nanmean([a * s for a, s in zip(assets, ea, strict=True)])
        pat, nii, imp = t("pat"), t("nii"), -t("impairment")
        nonii, opex = t("non_interest_income"), -t("non_interest_expense")
        rows.append({"bank": b, "date": "2026-09-30", "pat_ttm_tzs_m": pat, "roe_ttm": pat / eq,
                     "cost_of_risk_ttm": imp / loans, "nim_ttm": nii / earning, "cost_to_income_ttm": opex / (nii + nonii),
                     "avg_equity_basis": "average of 30 Sep 2025 and 30 Sep 2026 shareholders' funds less minority",
                     "nim_basis": "NII / average earning assets (total assets x printed earning-assets share)",
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def five(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pes = sorted(q["period_end"].unique())
    for b in BID:
        for pe in pes:
            e1 = get(q, b, "total_shareholders_funds", pe, "current_quarter")
            e0 = get(q, b, "total_shareholders_funds", pe, "previous_quarter")
            pat = get(q, b, "pat", pe, "current_quarter")
            nii, nonii = get(q, b, "nii", pe, "current_quarter"), get(q, b, "non_interest_income", pe, "current_quarter")
            opex = -get(q, b, "non_interest_expense", pe, "current_quarter")
            cap = q[(q["bank"] == b) & (q["field"] == "cet1_ratio") & (q["value_date"] == pe)
                    & (q["statement"] == "capital")].sort_values("period_end").tail(1)
            loans, dep = get(q, b, "net_loans", pe, "current_quarter"), get(q, b, "customer_deposits", pe,
                                                                            "current_quarter")
            rows.append({"bank": b, "period_end": pd.Timestamp(pe).date().isoformat(),
                         "roe_quarter_annualised": 4 * pat / np.mean([e0, e1]),
                         "npl_ratio": get(q, b, "npl_ratio", pe, "current_quarter"),
                         "cost_to_income_quarter": opex / (nii + nonii),
                         "cet1_or_core_ratio": float(cap["value"].iloc[0]) if len(cap) else np.nan,
                         "customer_deposits_tzs_m": dep, "net_loans_tzs_m": loans, "net_ldr": loans / dep,
                         "gross_ldr_printed": get(q, b, "gross_ldr", pe, "current_quarter"),
                         "loans_to_assets_printed": get(q, b, "loans_to_assets", pe, "current_quarter"),
                         "evidence_class": "Derived"})
    f = pd.DataFrame(rows)
    f["deposits_qoq"] = f.groupby("bank")["customer_deposits_tzs_m"].pct_change()
    f["loans_qoq"] = f.groupby("bank")["net_loans_tzs_m"].pct_change()
    return f


def monitor(q: pd.DataFrame, tt: pd.DataFrame, root: Path) -> pd.DataFrame:
    tb = pd.read_csv(path("tables", root) / "ch13_text_thesis_breaks.csv")
    rows = []
    for r in tb.itertuples():
        b = r.bank
        new, date, how = r.current, "as in the register / annual data", "unchanged (not a quarterly series)"
        if "cost of risk" in r.indicator:
            new = float(tt.set_index("bank").loc[b, "cost_of_risk_ttm"])
            date, how = "TTM to 30 Sep 2026", "impairment TTM / average net loans (quarterly statements)"
        elif "capital headroom" in r.indicator:
            cp = q[(q["bank"] == b) & (q["statement"] == "capital") & q["field"].notna()]
            if cp.empty:
                rows.append({"row": r.row, "bank": b, "indicator": r.indicator, "current_annual": r.current,
                             "current_latest": r.current, "latest_date": "FY2025 (annual report)",
                             "method": "no capital section in the 2026 quarterly statements", "threshold": r.threshold,
                             "evidence_class": "Derived"})
                continue
            vd = cp["value_date"].max()
            cp = cp[cp["value_date"] == vd].sort_values("period_end").groupby("field")["value"].last()
            c = lambda f, cp=cp: float(cp[f]) if f in cp.index else np.nan  # noqa: E731
            pe = vd
            buf = c("min_conservation_buffer")
            buf = 0.025 if pd.isna(buf) else buf
            head = [c("cet1_ratio") - (c("min_cet1_ratio") + buf), c("total_capital_ratio")
                    - (c("min_total_capital_ratio") + buf)]
            new = float(np.nanmin(head)) if not all(pd.isna(head)) else np.nan
            date = pd.Timestamp(pe).date().isoformat()
            how = "min(CET1 - (min + buffer), total - (min + buffer)), minima and buffer as printed"
        rows.append({"row": r.row, "bank": b, "indicator": r.indicator, "current_annual": r.current,
                     "current_latest": new, "latest_date": date, "method": how, "threshold": r.threshold,
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def eps_split(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pe in sorted(q["period_end"].unique()):
        eps = get(q, "NMB", "eps_basic", pe, "ytd_current")
        pat = get(q, "NMB", "pat", pe, "ytd_current")
        pre, post, sd = SHARES["NMB"]
        rows.append({"bank": "NMB", "period_end": pd.Timestamp(pe).date().isoformat(), "eps_printed_ytd": eps,
                     "pat_ytd_tzs_m": pat, "eps_pre_split_basis": pat * 1e6 / pre,
                     "eps_post_split_basis": pat * 1e6 / post,
                     "printed_basis": "post-split" if abs(eps - pat * 1e6 / post) < abs(eps - pat * 1e6 / pre)
                     else "pre-split", "split_effective": sd.date().isoformat(), "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def roll(q: pd.DataFrame, ctx) -> pd.DataFrame:
    """Owner decision (not adopted): the 31 Oct 2026 valuation with year 1 re-based on a quarter-end balance sheet.

    The base case already uses the 9M 2026 profit (nowcast) and annualised interim loan growth. Re-basing replaces the
    year-1 loan and deposit growth with the actual growth to the quarter-end followed by the base path for the rest of
    the year ((L_q / L_0) (1 + g_base)^((12 - m) / 12) - 1) and adds
    the difference between actual quarter-end equity and the model's (opening equity ex the declared FY2025 dividend +
    YTD profit) to
    opening equity (OCI and other movements)."""
    from tzbank.chapters.ch13_report_support import quote

    rows = []
    for b, bid in BID.items():
        c = ctx.cases[("CURR", bid, "base")]
        base = quote(ctx, "CURR", bid)
        rows.append({"bank": b, "variant": "base (FY2025 audited book; 9M 2026 profit nowcast)", "roll_date": "",
                     "value_per_share": base["value_per_share"], "target_12m": base["target_12m"],
                     "expected_return": base["expected_return"], "rating": base["rating"], "change_vs_base": 0.0,
                     "evidence_class": "Scenario"})
        for rd in settings(ctx.root)["quarterly"]["roll_dates"]:
            m = pd.Timestamp(rd).month
            L = get(q, b, "net_loans", rd, "current_quarter") * 1e6
            D = get(q, b, "customer_deposits", rd, "current_quarter") * 1e6
            E = (get(q, b, "total_shareholders_funds", rd, "current_quarter")
                 - np.nan_to_num(get(q, b, "minority_interest", rd, "current_quarter"))) * 1e6
            pat = get(q, b, "pat", rd, "ytd_current") * 1e6
            # actual growth to the quarter-end, then the base path for the remaining months of year 1
            gl = (L / c.op.loans) * (1 + c.dr.g_L[0]) ** ((12 - m) / 12) - 1
            gd = (D / c.op.deposits) * (1 + c.dr.g_D[0]) ** ((12 - m) / 12) - 1
            e_model = c.op.equity_attr + pat  # opening equity is already ex the declared FY2025 dividend (e_ex)
            de = E - e_model
            op = replace(c.op, equity_attr=c.op.equity_attr + de)
            dr = replace(c.dr, g_L=[gl] + list(c.dr.g_L[1:]), g_D=[gd] + list(c.dr.g_D[1:]),
                         nowcast={"np_attr_ytd": pat, "months": m})
            c2 = replace(c, op=op)
            ctx.cases[("CURR", bid, "base")] = c2
            try:
                qv = quote(ctx, "CURR", bid, dr)
            finally:
                ctx.cases[("CURR", bid, "base")] = c
            rows.append({"bank": b, "variant": f"year 1 re-based on {rd}", "roll_date": rd,
                         "year1_loan_growth": gl, "year1_deposit_growth": gd, "equity_true_up_tzs": de,
                         "value_per_share": qv["value_per_share"], "target_12m": qv["target_12m"],
                         "expected_return": qv["expected_return"], "rating": qv["rating"],
                         "change_vs_base": qv["value_per_share"] / base["value_per_share"] - 1,
                         "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def chart(f: pd.DataFrame, figd: Path) -> Path:
    fig, axs = plt_subplots(1, 5, figsize=(13, 3.0))
    spec = [("roe_quarter_annualised", "ROE (quarter, annualised), %", 100),
            ("npl_ratio", "NPL ratio, %", 100), ("cost_to_income_quarter", "cost-to-income (quarter), %", 100),
            ("cet1_or_core_ratio", "CET1 / core capital ratio, %", 100),
            ("customer_deposits_tzs_m", "customer deposits, TZS tn", 1e-6)]
    for ax, (col, lab, k) in zip(axs, spec, strict=True):
        for b in BID:
            g = f[f["bank"] == b]
            ax.plot(pd.to_datetime(g["period_end"]), g[col] * k, color=COLORS[BID[b]], marker="o", ms=3, label=b)
        style(ax, lab, "")
        ax.tick_params(axis="x", rotation=40, labelsize=6)
    axs[0].legend(fontsize=6.5, frameon=False)
    return save(fig, figd / "ch13_qtr_five.png", "Source: BoT-format quarterly statements, Q4 2025 - Q3 2026 (group); "
                "Q4 2025 first published (unaudited). Evidence class: Derived.")


NOTE = """# Quarterly statements (prompt 20, D89)

Generated by `tzbank.chapters.ch13_quarterly` (stage `market_ext`).

## Research question
What do the four quarters after FY2025 show for ROE, credit, capital and funding, and does the latest book value change
the 31 October 2026 valuation?

## Variable definitions
BoT template line numbers (balance sheet 1-41, income statement 1-15); TTM = FY2025 + 9M 2026 - 9M 2025; ROE (quarter,
annualised) = 4 x quarterly PAT / average shareholders' funds; net LDR = net loans / customer deposits.

## Sources
data/raw/manual_entries/quarterly/quarterly_transcribed.csv (page-referenced); data/raw/quarterly (PDFs); audited
FY2025 from the inputs workbook.

## Validation test
`tests/test_quarterly.py`.

## Results
{results}

## Evidence class
Observed (statements), Derived (ratios, checks), Scenario (re-based valuation).

## Limitations
- Q4 2025 statements are unaudited and differ from the audited FY2025 figures; the model uses the audited ones.
- Two statements were scanned; their numbers were read off the page image. The re-based valuation is a sensitivity for
  the owner, not the base case.
"""


def write_quarterly(ctx) -> list[Path]:
    root = ctx.root
    if not (root / ENTRY).exists():
        return []
    q = load(root)
    chk = checks(q, root)
    out = path("processed", root) / "quarterly"
    out.mkdir(parents=True, exist_ok=True)
    files = [write_csv(panel(q, chk, root), out / "quarterly_panel.csv", require_evidence=True)]
    tt, fv = ttm(q), five(q)
    tables = [(chk, "ch13_qtr_checks.csv", "Checks per statement", "Derived"),
              (tt, "ch13_qtr_ttm.csv", "TTM metrics at 30 Sep 2026", "Derived"),
              (monitor(q, tt, root), "ch13_qtr_monitor.csv", "r0-r9 refreshed to the latest quarter", "Derived"),
              (fv, "ch13_qtr_five.csv", "Five-quarter exhibit data", "Derived"),
              (fv[["bank", "period_end", "net_loans_tzs_m", "customer_deposits_tzs_m", "net_ldr", "gross_ldr_printed",
                   "loans_to_assets_printed", "deposits_qoq", "loans_qoq", "evidence_class"]], "ch13_qtr_ldr.csv",
               "Loan-to-deposit path", "Derived"),
              (eps_split(q), "ch13_qtr_eps_split.csv", "NMB EPS around the split", "Derived"),
              (roll(q, ctx), "ch13_qtr_roll.csv", "CURR valuation re-based on the latest quarter (owner decision)",
               "Scenario")]
    tab, figd = path("tables", root), path("figures", root)
    nf = path("notes", root) / "ch13_qtr.md"
    man_t = []
    for df, name, desc, ev in tables:
        f = write_csv(ctx.stamp(df), tab / name, require_evidence=True)
        files.append(f)
        man_t.append({"path": f.relative_to(root).as_posix(), "description": desc, "evidence_class": ev,
                      "note": nf.relative_to(root).as_posix()})
    p = chart(fv, figd)
    dd = write_csv(fv, p.with_name(f"{p.stem}_data.csv"), require_evidence=True)
    files += [p, dd]
    n_fail = int((chk["status"] == "fail").sum())
    res = [f"- Checks: {len(chk)} run, {n_fail} failed (ch13_qtr_checks)."]
    res += [f"- {r.bank} TTM to 30 Sep 2026: ROE {r.roe_ttm:.1%}, cost of risk {r.cost_of_risk_ttm:.2%}, NIM "
            f"{r.nim_ttm:.1%}, cost-to-income {r.cost_to_income_ttm:.1%}" for r in tt.itertuples()]
    files.append(write_text(NOTE.format(results="\n".join(res)), nf))
    m = {"chapter": "13_qtr", "title": "Quarterly statements (prompt 20)", "status": "complete", "hypotheses": [],
         "tables": man_t, "figures": [{"path": p.relative_to(root).as_posix(), "data": dd.relative_to(root).as_posix(),
                                       "description": "Five quarters: ROE, NPL, CIR, CET1, deposits",
                                       "evidence_class": "Derived", "note": nf.relative_to(root).as_posix()}],
         "notes": [nf.relative_to(root).as_posix()], "inputs": [str(ENTRY)], "module": "tzbank.chapters.ch13_quarterly",
         "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_qtr.json"))
    return files

