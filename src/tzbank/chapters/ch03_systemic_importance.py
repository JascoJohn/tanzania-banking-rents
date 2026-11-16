"""Chapter 3: systemic importance (blueprint §4 chapter 3; H3.1, H3.2; open issue 1).

1. D-SIB framework and designations: verified by text search of the BoT Annual Reports and both banks' reports
   (``data/interim/bot_policy_text.csv`` kinds ``dsib_statement`` / ``dsib_search``); what is not found is a collection
   task (open issue 1).
2. Capital buffers against requirements, every panel bank: headroom = ratio - requirement, with the requirement the BoT
   minimum (10% core / 12% total) plus the 2.5% conservation buffer from `capital_rules.ccb_start` (D52, D55); NMB and
   CRDB use their chapter 7 requirement (stated levels, reading per config).
3. Cost of deposits of NMB / CRDB against the smaller panel banks (chapter 6).
4. P/B against ROE across the panel: pooled OLS P/B = a + b ROE + c SIB (SIB = 1 for NMB and CRDB), 90% t interval for
   c; peers are TO COLLECT (open issue 22).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tzbank.chapters.ch04.common import COLORS, FOCUS, pct, plt_subplots, save, style, tickers, wide
from tzbank.config import path, settings
from tzbank.io import rel, write_csv, write_json, write_text

LEVEL = 0.90


def dsib(root: Path) -> pd.DataFrame:
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    d = p[p["kind"].isin(["dsib_statement", "dsib_search"])].copy()
    rows = []
    sc = {"BoT": "BoT Annual Reports", "B01": "CRDB reports", "B02": "NMB reports"}
    for r in d[d["kind"] == "dsib_search"].itertuples():
        rows.append({"item": f"search record: {sc.get(r.scope, r.scope)}", "finding": f"{int(r.value)} passage(s) "
                     f"in {r.file}", "date": "", "source": r.file})
    hits = d[d["kind"] == "dsib_statement"]
    fw = hits[(hits["scope"] == "BoT") & hits["quote"].str.contains("Framework for Identification", case=False, na=False)]
    rows.append({"item": "BoT D-SIB framework", "finding": "exists: 'Framework for Identification of Domestic "
                 "Systemically Important Banks, 2021'" if len(fw) else "NOT FOUND in the BoT reports held",
                 "date": "2021 (issued; reported in the BoT Annual Report 2021/22)" if len(fw) else "",
                 "source": f"{fw['file'].iloc[0]} p.{int(fw['pdf_page'].iloc[0])}" if len(fw) else "collection task"})
    nm = hits[(hits["scope"] == "B02") & hits["quote"].str.contains("designat", case=False, na=False)]
    rows.append({"item": "NMB designation", "finding": "designated a D-SIB by BoT ('In December 2025, BOT designated "
                 "NMB Bank Plc as a D-SIB'); surcharge rate not stated" if len(nm) else "not found",
                 "date": "December 2025 (as reported in NMB's FY2025 annual report, published 2026)" if len(nm) else "",
                 "source": "; ".join(sorted({f"{r.file} p.{int(r.pdf_page)}" for r in nm.itertuples()}))})
    cr = hits[hits["scope"] == "B01"]
    rows.append({"item": "CRDB designation", "finding": "NOT FOUND in any CRDB report" if cr.empty else
                 "passages found (see bot_policy_text.csv)", "date": "",
                 "source": "collection task: BoT announcement / D-SIB list (open issue 1)"})
    rows.append({"item": "BoT announcement of designations", "finding": "NOT FOUND in the BoT reports held (to FY2024/25)",
                 "date": "", "source": "collection task: BoT announcement and date (open issue 1; affects D1 visibility "
                                       "at 31 Dec 2025, `capital_rules.dsib_visible_at`)"})
    return pd.DataFrame(rows).assign(evidence_class="Observed")


def buffers(root: Path) -> pd.DataFrame:
    w = wide(root)
    ccb = pd.Timestamp(settings(root)["capital_rules"]["ccb_start"])
    c7 = pd.read_csv(path("tables", root) / "ch07_capital.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    t = tickers(root)
    rows = []
    for r in w.itertuples():
        if pd.isna(r.tier1_ratio) and pd.isna(r.total_capital_ratio):
            continue
        y = int(r.fiscal_year)
        if r.bank_id in FOCUS and (r.bank_id, y) in c7.index:
            k = c7.loc[(r.bank_id, y)]
            t1r, totr, t1, tot, basis = k["tier1_requirement"], k["total_requirement"], k["tier1_ratio"], \
                k["total_capital_ratio"], f"chapter 7 ({k['capital_basis']})"
        else:
            buf = 0.025 if pd.Timestamp(y, 12, 31) >= ccb else 0.0
            t1r, totr, t1, tot, basis = 0.10 + buf, 0.12 + buf, r.tier1_ratio, r.total_capital_ratio, \
                "BoT minimum + conservation buffer; ratio as reported (Bank = Group)"
        rows.append({"bank_id": r.bank_id, "bank": t.get(r.bank_id, r.bank_id), "fiscal_year": y,
                     "systemic": r.bank_id in FOCUS, "tier1_ratio": t1, "tier1_requirement": t1r,
                     "tier1_headroom": t1 - t1r if pd.notna(t1) else np.nan, "total_capital_ratio": tot,
                     "total_requirement": totr, "total_headroom": tot - totr if pd.notna(tot) else np.nan,
                     "binding_headroom": pd.Series([t1 - t1r, tot - totr], dtype=float).min(),
                     "basis": basis, "evidence_class": "Derived"})
    return pd.DataFrame(rows).dropna(subset=["tier1_ratio", "total_capital_ratio"], how="all").reset_index(drop=True)


def funding(root: Path) -> pd.DataFrame:
    d = pd.read_csv(path("tables", root) / "ch06_deposits.csv", dtype={"bank_id": str})
    d = d.dropna(subset=["cost_of_deposits"])
    rows = []
    for y, g in d.groupby("fiscal_year"):
        s, o = g[g["bank_id"].isin(FOCUS)], g[~g["bank_id"].isin(FOCUS)]
        if s.empty or o.empty:
            continue
        rows.append({"fiscal_year": int(y), "systemic_mean": s["cost_of_deposits"].mean(),
                     "smaller_median": o["cost_of_deposits"].median(), "n_smaller": len(o),
                     "advantage": o["cost_of_deposits"].median() - s["cost_of_deposits"].mean(),
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def pb_roe(root: Path) -> tuple[pd.DataFrame, dict]:
    w = wide(root)
    w["market_cap_used"] = w["market_cap"].fillna(w["share_price_close"] * w["shares_outstanding"])
    eq = w["equity_attributable"].fillna(w["total_equity"])
    d = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv", dtype={"bank_id": str})
    x = w.assign(pb=w["market_cap_used"] / eq)[["bank_id", "fiscal_year", "pb"]].merge(
        d[["bank_id", "fiscal_year", "roe"]], on=["bank_id", "fiscal_year"]).dropna()
    x = x[(x["pb"] > 0) & np.isfinite(x["pb"])]
    x["systemic"] = x["bank_id"].isin(FOCUS).astype(int)
    fit = _ols(x)
    pos = _ols(x[x["roe"] > 0])
    fit.update({f"pos_{k}": v for k, v in pos.items()})
    x["fitted_no_premium"] = fit["a"] + fit["b_roe"] * x["roe"]
    x["residual_vs_roe_line"] = x["pb"] - x["fitted_no_premium"]
    x["evidence_class"] = "Derived"
    return x, fit


def _ols(x: pd.DataFrame) -> dict:
    """Pooled OLS P/B = a + b ROE + c SIB with a 90% t interval for c (Blueprint chapter 3, H3.2)."""
    X = np.column_stack([np.ones(len(x)), x["roe"], x["systemic"]])
    y = x["pb"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    dof = len(x) - 3
    se = np.sqrt(np.diag((e @ e / dof) * np.linalg.inv(X.T @ X)))
    tc = stats.t.ppf(1 - (1 - LEVEL) / 2, dof)
    return {"a": b[0], "b_roe": b[1], "c_sib": b[2], "c_lo": b[2] - tc * se[2], "c_hi": b[2] + tc * se[2],
            "n": len(x), "n_banks": int(x["bank_id"].nunique())}


def hypotheses(fu: pd.DataFrame, bf: pd.DataFrame, fit: dict, ds: pd.DataFrame) -> pd.DataFrame:
    adv = fu["advantage"].mean() if len(fu) else np.nan
    req = bf.groupby("systemic")["tier1_requirement"].mean()
    higher_req = bool(req.get(True, 0) > req.get(False, 0) + 1e-9)
    v1 = "supported" if adv > 0.01 and higher_req else "rejected" if adv <= 0 else "inconclusive"
    hb = bf.groupby("systemic")["binding_headroom"].mean()
    v2 = ("supported (discount)" if fit["c_hi"] < 0 else "not rejected (low power)" if fit["c_lo"] <= 0 <= fit["c_hi"]
          else "rejected")
    return pd.DataFrame([
        {"hypothesis": "H3.1", "verdict": v1,
         "evidence": f"cost of deposits: smaller banks' median exceeds the NMB / CRDB mean by {pct(adv, 2)} on average "
                     f"({len(fu)} years); stated Tier 1 requirement: systemic banks {pct(req.get(True), 2)} vs others "
                     f"{pct(req.get(False), 2)} on average (no higher requirement before NMB's December 2025 D-SIB "
                     f"designation, surcharge not stated); held headroom: systemic {pct(hb.get(True))}, others "
                     f"{pct(hb.get(False))}",
         "rule": "supported when the deposit-cost advantage exceeds 100 bp AND systemic banks face a higher stated "
                 "requirement; rejected when there is no advantage; inconclusive otherwise"},
        {"hypothesis": "H3.2", "verdict": v2,
         "evidence": f"pooled P/B = {fit['a']:.2f} + {fit['b_roe']:.2f} ROE + c SIB: c = {fit['c_sib']:+.2f} "
                     f"[{fit['c_lo']:+.2f}, {fit['c_hi']:+.2f}] (90%), n = {fit['n']} bank-years, {fit['n_banks']} "
                     f"banks; profitable bank-years only (ROE > 0): c = {fit['pos_c_sib']:+.2f} [{fit['pos_c_lo']:+.2f}, "
                     f"{fit['pos_c_hi']:+.2f}], n = {fit['pos_n']}; East African peers TO COLLECT",
         "rule": "H3.2 (no systemic premium) is not rejected when the 90% interval of the systemic-bank coefficient "
                 "includes zero (low power: 56 bank-years, 7 banks, a wide interval; D59 amended), supported when it "
                 "lies below zero, rejected when it lies above zero; the ROE > 0 specification "
                 "is reported as a robustness check"}]).assign(evidence_class="Interpretation")


NOTE = """# Chapter 3: systemic importance

Generated by `tzbank.chapters.ch03_systemic_importance` (stage `part1`). Blueprint §4 chapter 3; H3.1, H3.2; open
issue 1 (D-SIB).

## Research question

Does systemic importance give NMB and CRDB a funding advantage, is it offset by higher capital requirements, and do
their valuation multiples carry a premium for it?

## Variable definitions

Headroom = capital ratio - requirement (BoT minimum + conservation buffer; NMB and CRDB as in chapter 7); deposit-cost
advantage = smaller banks' median cost of deposits - NMB / CRDB mean; P/B = market capitalisation / attributable equity
(year-end); systemic-bank premium = coefficient c in P/B = a + b ROE + c SIB.

## Sources and transformation trail

`data/interim/bot_policy_text.csv` (D-SIB search), `outputs/tables/ch07_capital.csv`, `data/processed/bank_year_wide.csv`,
`outputs/tables/ch06_deposits.csv`, `outputs/tables/ch04_dupont_panel.csv`; config `capital_rules`. Functions: `dsib`,
`buffers`, `funding`, `pb_roe`, `hypotheses`.

## Validation test

`tests/test_part1.py`: headroom = ratio - requirement; P/B regression reproduces a hand OLS; D-SIB table lists the
search record for every source.

## Results

### D-SIB framework and designations (Observed)

{dsib}

### Capital buffers against requirements (Derived)

{buf}

### Cost of deposits (Derived)

{fund}

### P/B against ROE (Derived)

{pb}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

D-SIB findings: **Observed**; headroom, cost gaps and the regression: **Derived**; verdicts: **Interpretation**.

## Limitations

- The BoT's announcement and list of designated banks are not held; NMB's designation is known from its own report
  (open issue 1). CRDB's status is unknown.
- Smaller banks' capital ratios are as reported (Bank = Group); their requirement is the BoT minimum plus the buffer.
- DCB, MKCB and MUCOBA report ratios below the requirement in some years (as published; no adjustment).
- The P/B panel is small and unbalanced (listing dates, thin trading); peers are not held.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

A deposit-cost advantage without a matching capital requirement is consistent with a **regulatory advantage** or a
perceived too-big-to-fail **funding subsidy** as well as with **low-cost deposits** from the franchise; chapter 3
cannot separate them. Chapter 6 measures the franchise net of costs.
"""


def write_ch03(root: Path) -> list[Path]:
    t = tickers(root)
    ds = dsib(root)
    bf = buffers(root)
    fu = funding(root)
    pb, fit = pb_roe(root)
    hyp = hypotheses(fu, bf, fit, ds)
    tab, figd = path("tables", root), path("figures", root)
    tables = [(ds, "ch03_dsib.csv", "D-SIB framework and designations", "Observed"),
              (bf, "ch03_capital_buffers.csv", "Capital headroom against requirements, panel banks", "Derived"),
              (fu, "ch03_deposit_cost.csv", "Cost of deposits: NMB / CRDB vs smaller banks", "Derived"),
              (pb, "ch03_pb_roe.csv", "P/B against ROE across the panel", "Derived"),
              (hyp, "ch03_hypotheses.csv", "H3.1-H3.2", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    fig, ax = plt_subplots(figsize=(7.5, 4.4))
    for bid, g in pb.groupby("bank_id"):
        ax.scatter(g["roe"] * 100, g["pb"], s=18, color=COLORS.get(bid, "#9a9a94"), label=t.get(bid, bid))
    xs = np.linspace(pb["roe"].min(), pb["roe"].max(), 20)
    ax.plot(xs * 100, fit["a"] + fit["b_roe"] * xs, color="#1f1f1d", lw=1, label="P/B on ROE (no premium)")
    style(ax, "Price to book against ROE, panel banks", "P/B")
    ax.set_xlabel("ROE, percent", fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    png = save(fig, figd / "ch03_pb_roe.png", "Evidence class: Derived. Year-end market cap / attributable equity; "
               "line: pooled OLS without the systemic-bank term.")
    fd = write_csv(pb[["bank_id", "fiscal_year", "roe", "pb", "evidence_class"]], figd / "ch03_pb_roe_data.csv",
                   require_evidence=True)
    files += [png, fd]
    ds_txt = "\n".join(f"- **{r.item}**: {r.finding}" + (f" ({r.date})" if r.date else "") + f". Source: {r.source}."
                       for r in ds.itertuples())
    bb = bf[bf["fiscal_year"] >= 2018].groupby(["bank"])["binding_headroom"].agg(["mean", "count"])
    buf_txt = "\n".join(f"- {k}: mean binding headroom FY2018-25 {pct(r['mean'])} ({int(r['count'])} years)"
                        for k, r in bb.sort_values("mean", ascending=False).iterrows())
    fu_txt = (f"Smaller banks' median cost of deposits exceeds the NMB / CRDB mean by {pct(fu['advantage'].mean(), 2)} "
              f"on average over {len(fu)} years (latest FY{int(fu['fiscal_year'].max())}: "
              f"{pct(fu['advantage'].iloc[-1], 2)}).") if len(fu) else "No overlapping years."
    pb_txt = (f"P/B = {fit['a']:.2f} + {fit['b_roe']:.2f} x ROE + {fit['c_sib']:+.2f} x SIB; SIB coefficient 90% "
              f"interval [{fit['c_lo']:+.2f}, {fit['c_hi']:+.2f}]; {fit['n']} bank-years, {fit['n_banks']} banks. "
              f"Alternative specification, ROE > 0 only: P/B = {fit['pos_a']:.2f} + {fit['pos_b_roe']:.2f} x ROE + "
              f"{fit['pos_c_sib']:+.2f} x SIB [{fit['pos_c_lo']:+.2f}, {fit['pos_c_hi']:+.2f}], {fit['pos_n']} bank-years "
              "(the full-sample ROE slope is negative because loss-making, thinly traded small banks trade above "
              "book). Mean residual from the ROE line: " + "; ".join(
                  f"{t.get(k, k)} {v:+.2f}" for k, v in pb.groupby("bank_id")["residual_vs_roe_line"].mean().items()))
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Rule: {r.rule}." for r in hyp.itertuples())
    interp = ("**Interpretation.** NMB and CRDB fund far more cheaply than the smaller listed "
              "banks, while the stated capital requirement was the same for all banks until NMB's December 2025 "
              "designation; any offset so far is NMB's own choice to hold more capital, not a requirement.")
    nf = write_text(NOTE.format(dsib=ds_txt, buf=buf_txt, fund=fu_txt, pb=pb_txt, hyp=hyp_txt, interp=interp),
                    path("notes", root) / "ch03.md")
    files.append(nf)
    m = {"chapter": "03", "title": "Systemic importance", "hypotheses": ["H3.1", "H3.2"],
         "status": "built",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for _, n, d, c in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "P/B against ROE",
                      "evidence_class": "Derived", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "module": "tzbank.chapters.ch03_systemic_importance", "run_log_stage": "part1"}
    files.append(write_json(m, path("manifests", root) / "ch03.json"))
    return files
