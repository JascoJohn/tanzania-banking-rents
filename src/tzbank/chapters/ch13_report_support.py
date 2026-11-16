"""Report support (prompt 13): exhibits the owner's drafts need, computed in the pipeline (D66) and shown on the
dashboard page "13. Report support" with findings-pack ids (CH13.*). No report text is written here.

Exhibits (one function each; evidence classes on every row):
 1 ``roe_reconcile``      year-1 base ROE vs FY reported: engine ROE = NP / opening equity ex the declared dividend
                          (clean-surplus year-end convention); restated on the reported basis (NP / average equity cum
                          dividend) it matches the reported figure.
 2 ``cr2_profit_direct``  direct 2025 CR2 profit share from the BoT Banking Supervision Report 2025 (all banking
                          institutions, PBT and PAT), against the asset share; ``implied_credit_shares``: bank loan
                          amounts / BoT credit by activity, by category.
 3 ``earnings_bridge``    NP_attr FY2025A -> FY2027E by driver, per bank and scenario; ``roe_waterfall`` (FY2025A ->
                          FY2027E base); ``roe_sustainable_bridge`` reported FY2025 ROE -> ROE* (credit, rate, capital,
                          residual).
 4 ``rating_flips``       the shift in each key assumption at which the D4 rating changes; ``stress_mixture``
                          P(value > price) against the weight on a stress regime (bear-case Monte Carlo);
                          ``rate_consistent`` rate path and COE moved together; ``named_scenarios`` (dominance without
                          excess return: ROE* = COE, everything else base).
 5 ``deposit_scenarios``  x% of CASA repriced to the T-bill; falling-rate deposit beta; CASA cost floor; ROE per 1 pp
                          T-bill fall; 2018 bank-closure event test (deposit growth).
 6 ``valuation_bands``    own-history P/B and P/E bands; ``kpi_table`` (7 banks, FY2025); P/B-ROE chart with the
                          justified P/B line as the anchor.
 7 ``capacity_windows``   pre-rally (to 31 Dec 2025) vs 2026 windows; ``block_board`` turnover by quarter.
 8 ``franchise_breakeven``break-even T-bill for the net deposit franchise and its value at the v1.3 path;
                          ``surplus_drag`` capital surplus earning the T-bill instead of the COE, against (a) the
                          regulatory minimum and (b) the model target k* (prompt 13b item 3).
10 ``financial_table``    FY2024A-FY2028E per bank; ``price_export`` 1-year price, 52-week range, rating price lines.
13 ``usd_exports``        USD return by depreciation (0, 2.3, 3.8, 24.5%), break-even depreciation, entry price meeting
                          the USD hurdle, joint 2015-style currency and credit stress, worst calendar-year USD return.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.banks.engine import project
from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import models as vm
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un

T = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB", "B07": "MUCOBA"}
FOCUS = ("B02", "B01")
DEPRECIATION = (0.0, 0.023, 0.038, 0.245)


def _wide(root: Path) -> pd.DataFrame:
    return pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)


def _split(root: Path, bank: str) -> float:
    ca = pd.read_csv(root / "config" / "corporate_actions.csv")
    return {r.bank_id: float(r.ratio) for r in ca.itertuples() if r.action == "split"}.get(bank, 1.0)


# --- valuation helper (chapter 9 rule) --------------------------------------------------------------------------------
def quote(ctx, val: str, bank: str, dr=None, r: float | None = None) -> dict:
    """Value, target, expected return and D4 rating for a base case with replaced drivers / COE."""
    from tzbank.chapters.ch09_valuation import _divs_in_window, rating

    c = ctx.cases[(val, bank, "base")]
    d = dr or c.dr
    rr = c.r if r is None else r
    fc = project(c.op, d)
    v0 = vm.blend(fc, rr, c.vd, c.w, lag_months=c.lag_months)["blend"] / c.shares
    v1 = vm.blend(fc, rr, c.vd + pd.DateOffset(years=1), c.w, lag_months=c.lag_months)["blend"] / c.shares
    dps = _divs_in_window(fc, c.vd, c.lag_months) / c.shares
    exp = (v1 + dps - c.price) / c.price
    bear = un.value(ctx.cases[(val, bank, "bear")]) / c.shares
    rt, _ = rating(exp, rr, bear / c.price - 1)
    return {"value_per_share": v0, "target_12m": v1, "dps_12m": dps, "expected_return": exp, "rating": rt,
            "coe": rr, "price": c.price}


# --- 1 ---------------------------------------------------------------------------------------------------------------
def roe_reconcile(ctx) -> pd.DataFrame:
    d = pd.read_csv(path("tables", ctx.root) / "ch04_dupont_panel.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            r = project(c.op, c.dr).rows.set_index("t")
            e0, e1, np1 = r.loc[0, "E"], r.loc[1, "E"], r.loc[1, "NP_attr"]
            div0 = ctx.inputs[(val, b)].final_dps_anchor * ctx.inputs[(val, b)].fy["shares_outstanding"]
            div0 = 0.0 if not np.isfinite(div0) else div0
            div1 = r.loc[1, "Div"]
            rep_basis = np1 / ((e0 + div0 + e1 + div1) / 2)
            fy = int(r.loc[1, "year"])
            anchor = int(r.loc[0, "year"])
            rows.append({"valuation": val, "bank": T[b], "forecast_year": fy, "np_attr_y1": np1,
                         "opening_equity_ex_dividend": e0, "declared_dividend_anchor": div0,
                         "roe_engine_opening_ex_div": np1 / e0, "roe_avg_ex_div": np1 / ((e0 + e1) / 2),
                         "roe_reported_basis": rep_basis, "reported_roe_anchor_fy": anchor,
                         "reported_roe_anchor": float(d.loc[(b, anchor), "roe"]) if (b, anchor) in d.index else np.nan,
                         "explanation": "engine ROE divides by opening equity ex the declared anchor-year dividend "
                                        "(year-end clean surplus); the reported basis divides by average equity cum "
                                        "the declared dividend (as the banks report)",
                         "evidence_class": "Derived"})
    out = pd.DataFrame(rows)
    out["definition_effect"] = out["roe_engine_opening_ex_div"] - out["roe_reported_basis"]
    out["like_for_like_change"] = out["roe_reported_basis"] - out["reported_roe_anchor"]
    return out


# --- 2 ---------------------------------------------------------------------------------------------------------------
SECTOR = {"ta": "sector_total_assets_all_banks", "pbt": "sector_profit_before_tax_all_banks",
          "pat": "sector_profit_after_tax_all_banks"}


def cr2_profit_direct(root: Path) -> pd.DataFrame:
    """CR2 profit share against asset share (prompt 13b item 2). Two bases, never mixed in one row:

    * ``all banking institutions`` (primary, one consistent series 2011-2025): sector totals from the BoT supervision
      report appendices (consolidated balance sheet and income statement of the banking sector, calendar year, TZS
      million; latest vintage held for each year, via ``data/raw/manual_entries/web_collected.csv``). Numerator: NMB +
      CRDB Group profit before / after tax and Group total assets (CLAUDE.md 3.5: Group; CRDB Group includes its
      Burundi and DRC subsidiaries, which the Tanzanian sector total excludes, so the share is overstated by their
      contribution). Year-end asset shares.
    * ``commercial banks (Estimated)``: chapter 1's profit share via the BoT sector ROA (end-June fiscal year; pre- or
      post-tax not stated), times the CR2 share of commercial-bank assets: CR2 ROA (pre-tax, average assets) / sector
      ROA x asset share. Shown where the BoT ROA is held.
    """
    from tzbank.ingest.web_collected import load

    wc = load(root)
    w = _wide(root).set_index(["bank_id", "fiscal_year"])
    rows = []
    for y in range(2011, 2026):
        g = wc[wc["period"].astype(str) == str(y)].set_index("canonical_name")
        if SECTOR["ta"] not in g.index:
            continue
        sec = {k: float(g.loc[v, "value"]) * 1e6 if v in g.index else np.nan for k, v in SECTOR.items()}
        vint = "; ".join(sorted({str(g.loc[v, "vintage"]) for v in SECTOR.values() if v in g.index}))
        pub = max(str(g.loc[v, "publication_date"]) for v in SECTOR.values() if v in g.index)
        num = {k: sum(float(w.loc[(b, y), c]) for b in FOCUS) if all((b, y) in w.index for b in FOCUS) else np.nan
               for k, c in (("ta", "total_assets"), ("pbt", "profit_before_tax"), ("pat", "net_profit"))}
        sh = {k: num[k] / sec[k] if np.isfinite(sec[k]) else np.nan for k in SECTOR}
        rows.append({"year": y, "sector_basis": "all banking institutions (BoT supervision report appendix)",
                     "numerator_basis": "NMB + CRDB Group (consolidated)", "period_basis": "calendar year; year-end assets",
                     "sector_source": vint, "sector_publication_date": pub, "sector_total_assets": sec["ta"],
                     "sector_pbt": sec["pbt"], "sector_pat": sec["pat"], "cr2_total_assets": num["ta"],
                     "cr2_pbt": num["pbt"], "cr2_pat": num["pat"], "cr2_asset_share": sh["ta"],
                     "cr2_pbt_share": sh["pbt"], "cr2_pat_share": sh["pat"],
                     "pbt_minus_asset_share": sh["pbt"] - sh["ta"], "pat_minus_asset_share": sh["pat"] - sh["ta"],
                     "status": "observed sector totals" if np.isfinite(sec["pbt"]) else
                     "sector profit NOT DISCLOSED (BoT 2023 income-statement column not usable in any vintage)",
                     "evidence_class": "Derived"})
        # CRDB bank-solo numerator where the parent-bank lines are held (edit pack A B3, D82): CRDB Group includes its
        # Burundi and DRC subsidiaries, which the Tanzanian sector total excludes; NMB Group = Bank within 0.5%
        solo = {"ta": "bank_only_total_assets", "pbt": "bank_only_pbt", "pat": "bank_only_pat"}
        if ("B01", y) in w.index and ("B02", y) in w.index and all(
                c in w.columns and pd.notna(w.loc[("B01", y), c]) for c in solo.values()):
            ns = {k: float(w.loc[("B02", y), grp]) + float(w.loc[("B01", y), solo[k]])
                  for k, grp in (("ta", "total_assets"), ("pbt", "profit_before_tax"), ("pat", "net_profit"))}
            ss = {k: ns[k] / sec[k] if np.isfinite(sec[k]) else np.nan for k in SECTOR}
            rows.append({"year": y, "sector_basis": "all banking institutions (BoT supervision report appendix)",
                         "numerator_basis": "NMB Group + CRDB Bank-solo", "period_basis":
                         "calendar year; year-end assets", "sector_source": vint, "sector_publication_date": pub,
                         "sector_total_assets": sec["ta"], "sector_pbt": sec["pbt"], "sector_pat": sec["pat"],
                         "cr2_total_assets": ns["ta"], "cr2_pbt": ns["pbt"], "cr2_pat": ns["pat"],
                         "cr2_asset_share": ss["ta"], "cr2_pbt_share": ss["pbt"], "cr2_pat_share": ss["pat"],
                         "pbt_minus_asset_share": ss["pbt"] - ss["ta"], "pat_minus_asset_share": ss["pat"] - ss["ta"],
                         "status": "observed sector totals; CRDB parent bank" if np.isfinite(sec["pbt"]) else
                         "sector profit NOT DISCLOSED (BoT 2023 income-statement column not usable in any vintage)",
                         "evidence_class": "Derived"})
    pg = pd.read_csv(path("tables", root) / "ch01_profit_vs_assets.csv")
    for r in pg.itertuples():
        rows.append({"year": int(r.year), "sector_basis": "commercial banks (Estimated via BoT sector ROA, ch01)",
                     "numerator_basis": "NMB + CRDB Group ROA (pre-tax, average assets)",
                     "period_basis": "BoT ROA end-June fiscal year; banks calendar year", "sector_source":
                     str(r.sector_roa_source), "cr2_asset_share": r.asset_share, "cr2_pbt_share": r.profit_share_pretax,
                     "pbt_minus_asset_share": r.profit_minus_asset_share_pretax,
                     "pat_minus_asset_share": r.profit_minus_asset_share_posttax, "status": "estimated",
                     "evidence_class": "Estimated"})
    return pd.DataFrame(rows).sort_values(["sector_basis", "year"]).reset_index(drop=True)


def profit_2023_sensitivity(cr: pd.DataFrame) -> pd.DataFrame:
    """Profit-share conclusions with FY2023 excluded (the default: sector profit not usable in any BoT vintage) and
    with FY2023 sector profit interpolated linearly between 2022 and 2024 (an Estimated series, never mixed into the
    observed rows; prompt 24, 3c.9; Blueprint ch. 1). Both numerator bases (NMB + CRDB Group; NMB Group + CRDB
    Bank-solo); premium = profit-before-tax share less asset share."""
    rows = []
    obs = cr[cr["sector_basis"].str.startswith("all banking")]
    for basis, g in obs.groupby("numerator_basis"):
        g = g.set_index("year").sort_index()
        prem = g["pbt_minus_asset_share"]
        interp = g.copy()
        if 2023 in interp.index and {2022, 2024} <= set(interp.index):
            spbt = (interp.loc[2022, "sector_pbt"] + interp.loc[2024, "sector_pbt"]) / 2
            interp.loc[2023, "cr2_pbt_share"] = interp.loc[2023, "cr2_pbt"] / spbt
            interp.loc[2023, "pbt_minus_asset_share"] = interp.loc[2023, "cr2_pbt_share"] - interp.loc[2023,
                                                                                                    "cr2_asset_share"]
        pi = interp["pbt_minus_asset_share"]
        for case, x in (("FY2023 excluded (default)", prem), ("FY2023 interpolated (Estimated)", pi)):
            x = x.dropna()
            rows.append({"numerator_basis": basis, "case": case,
                         "share_2023": float(interp.loc[2023, "cr2_pbt_share"]) if case.startswith("FY2023 interp")
                         and 2023 in interp.index else np.nan,
                         "premium_2023": float(pi.get(2023, np.nan)) if case.startswith("FY2023 interp") else np.nan,
                         "years": len(x), "mean_premium_all": float(x.mean()),
                         "mean_premium_2019_25": float(x[x.index >= 2019].mean()),
                         "years_premium_positive": int((x > 0).sum()), "latest_premium_2025": float(x.get(2025, np.nan)),
                         "evidence_class": "Derived" if case.startswith("FY2023 excl") else "Estimated"})
    return pd.DataFrame(rows)


def implied_credit_shares(root: Path) -> pd.DataFrame:
    m = pd.read_csv(path("tables", root) / "ch02_loan_mix.csv")
    rows = []
    for y, g in m.groupby("fiscal_year"):
        sysv = g[g["book"] == "BoT commercial banks"].set_index("category")["value"]
        for bk in ("NMB", "CRDB", "NMB + CRDB"):
            x = g[g["book"] == bk].set_index("category")["value"]
            for cat in x.index.intersection(sysv.index):
                if sysv[cat] > 0:
                    rows.append({"fiscal_year": int(y), "book": bk, "category": cat, "bank_amount": x[cat],
                                 "system_amount": sysv[cat], "implied_share": x[cat] / sysv[cat],
                                 "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- 3 ---------------------------------------------------------------------------------------------------------------
LINES = [("NII", "net_interest_income", 1), ("NonII", "non_interest_income", 1), ("OpEx", "operating_expenses", -1),
         ("Imp", "impairment_charges", -1)]


def earnings_bridge(ctx) -> pd.DataFrame:
    w = _wide(ctx.root).set_index(["bank_id", "fiscal_year"])
    rows = []
    for b in FOCUS:
        a = w.loc[(b, 2025)]
        for sc in ("bear", "base", "bull"):
            c = ctx.cases[("CURR", b, sc)]
            f = project(c.op, c.dr).rows.set_index("year")
            if 2027 not in f.index:
                continue
            y = f.loc[2027]
            prev = float(a["profit_attributable"])
            steps = [("FY2025A attributable profit", prev)]
            for lab, col, sgn in LINES:
                steps.append((f"change in {lab}", sgn * (abs(float(y[lab])) - abs(float(a[col])))))
            pbt27 = float(y["PBT"])
            pbt25 = float(a["profit_before_tax"])
            other = (pbt27 - pbt25) - sum(s for _, s in steps[1:])
            steps.append(("other operating items", other))
            tax = -((pbt27 - float(y["NP"])) - float(a["income_tax_expense"]))
            steps.append(("change in tax", tax))
            nci = float(y["NP_attr"]) - prev - sum(s for _, s in steps[1:])
            steps.append(("change in NCI and rounding", nci))
            steps.append(("FY2027E attributable profit", float(y["NP_attr"])))
            for i, (lab, v) in enumerate(steps):
                rows.append({"bank": T[b], "scenario": sc, "step_no": i, "step": lab, "tzs": v,
                             "evidence_class": "Observed" if i == 0 else "Scenario"})
    return pd.DataFrame(rows)


def roe_waterfall(ctx) -> pd.DataFrame:
    w = _wide(ctx.root).set_index(["bank_id", "fiscal_year"])
    d = pd.read_csv(path("tables", ctx.root) / "ch04_dupont_panel.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    rows = []
    for b in FOCUS:
        a, a0 = w.loc[(b, 2025)], w.loc[(b, 2024)]
        e25 = (float(a["equity_attributable"]) + float(a0["equity_attributable"])) / 2
        c = ctx.cases[("CURR", b, "base")]
        f = project(c.op, c.dr).rows.set_index("year")
        e27 = (float(f.loc[2027, "E"]) + float(f.loc[2026, "E"])) / 2
        comp25 = {lab: sgn * abs(float(a[col])) for lab, col, sgn in LINES}
        comp27 = {lab: sgn * abs(float(f.loc[2027, lab])) for lab, _, sgn in LINES}
        roe25 = float(d.loc[(b, 2025), "roe"])
        roe27 = float(f.loc[2027, "NP_attr"]) / e27
        steps = [("FY2025A ROE (average equity)", roe25)]
        for lab in comp25:
            steps.append((f"{lab} / equity", (comp27[lab] - comp25[lab]) / e27))
        steps.append(("equity base (denominator)", sum(comp25.values()) * (1 / e27 - 1 / e25)))
        steps.append(("tax, other and NCI", roe27 - roe25 - sum(s for _, s in steps[1:])))
        steps.append(("FY2027E ROE (average equity, base)", roe27))
        rows += [{"bank": T[b], "step_no": i, "step": s, "roe_pts": v, "evidence_class": "Scenario" if i else
                  "Derived"} for i, (s, v) in enumerate(steps)]
    return pd.DataFrame(rows)


def roe_sustainable_bridge(ctx) -> pd.DataFrame:
    w = _wide(ctx.root).set_index(["bank_id", "fiscal_year"])
    d = pd.read_csv(path("tables", ctx.root) / "ch04_dupont_panel.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    rows = []
    for b in FOCUS:
        c = ctx.cases[("CURR", b, "base")]
        a, a0 = w.loc[(b, 2025)], w.loc[(b, 2024)]
        e = (float(a["equity_attributable"]) + float(a0["equity_attributable"])) / 2
        ta = (float(a["total_assets"]) + float(a0["total_assets"])) / 2
        gl = (float(a["gross_loans"]) + float(a0["gross_loans"])) / 2
        iea = (float(a["interest_earning_assets"]) + float(a0["interest_earning_assets"])) / 2
        tax = c.dr.tax
        roe = float(d.loc[(b, 2025), "roe"])
        cor25 = abs(float(a["impairment_charges"])) / gl
        cor_ttc = float(c.dr.CoR[-1])
        f = project(c.op, c.dr).rows
        last = f[f["phase"] == "explicit"].iloc[-1] if "phase" in f else f.iloc[-1]
        credit = -(cor_ttc - cor25) * gl / e * (1 - tax)
        rate = (c.dr.asset_beta - c.dr.deposit_beta * float(a["customer_deposits"]) / iea) * \
            (float(c.dr.rate[-1]) - float(c.dr.rate0)) * iea / e * (1 - tax)
        lev_end = float(last["TA"]) / float(last["E"])
        capital = (roe / (ta / e)) * (lev_end - ta / e)
        roe_star = c.dr.roe_star
        resid = roe_star - roe - credit - rate - capital
        for i, (s, v) in enumerate([("FY2025A reported ROE", roe), ("credit: cost of risk to through-the-cycle",
                                                                     credit),
                                    ("rate: T-bill path (v1.3)", rate), ("capital: leverage to the k* path", capital),
                                    ("residual: fade to ROE*, mix, efficiency", resid),
                                    ("sustainable ROE* (register v1.3 base)", roe_star)]):
            rows.append({"bank": T[b], "step_no": i, "step": s, "roe_pts": v,
                         "evidence_class": "Derived" if i == 0 else "Scenario"})
    return pd.DataFrame(rows)


# --- 4 ---------------------------------------------------------------------------------------------------------------
def _shift(dr, key: str, x: float):
    if key == "COE":
        return dr
    if key == "ROE*":
        return un._with_roe(dr, dr.roe_star + x)
    if key == "terminal g":
        return un._with_g(dr, dr.g + x)
    if key == "lambda":
        return replace(dr, lam=float(np.clip(dr.lam + x, 0.0, 0.99)))
    if key == "rate path":
        return replace(dr, rate=[v + x for v in dr.rate])
    if key == "NIM":
        return replace(dr, NIM=[v + x for v in dr.NIM])
    if key == "cost of risk":
        return replace(dr, CoR=[max(0.0, v + x) for v in dr.CoR])
    if key == "loan and deposit growth":
        return replace(dr, g_L=[dr.g_L[0]] + [v + x for v in dr.g_L[1:]], g_D=[dr.g_D[0]] + [v + x for v in dr.g_D[1:]])
    raise KeyError(key)


FLIP_STEPS = {"COE": 0.0025, "ROE*": 0.0025, "terminal g": 0.0025, "lambda": 0.02, "rate path": 0.0025,
              "NIM": 0.001, "cost of risk": 0.001, "loan and deposit growth": 0.005}


def rating_flips(ctx, max_steps: int = 40) -> pd.DataFrame:
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            base = quote(ctx, val, b)
            for key, step in FLIP_STEPS.items():
                for sign in (-1, 1):
                    hit = None
                    for k in range(1, max_steps + 1):
                        x = sign * step * k
                        if key == "terminal g" and c.r - (c.dr.g + x) < 0.005:
                            break
                        try:
                            q = quote(ctx, val, b, _shift(c.dr, key, x), c.r + x if key == "COE" else None)
                        except Exception:  # noqa: BLE001 - an infeasible driver set ends the scan
                            break
                        if q["rating"] != base["rating"]:
                            hit = (x, q)
                            break
                    rows.append({"valuation": val, "bank": T[b], "assumption": key, "direction": "down" if sign < 0
                                 else "up", "base_rating": base["rating"], "base_expected_return": base[
                                     "expected_return"], "shift_to_flip": hit[0] if hit else np.nan,
                                 "flipped_rating": hit[1]["rating"] if hit else "no flip in range",
                                 "value_at_flip": hit[1]["value_per_share"] if hit else np.nan,
                                 "scan_range": sign * step * max_steps, "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def stress_mixture(ctx, draws: int = 4000) -> pd.DataFrame:
    st = settings(ctx.root)
    vc = st["valuation"]
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            cs = {s: ctx.cases[(val, b, s)] for s in ("bear", "base", "bull")}
            coe_sh = (cs["bull"].r - cs["base"].r + 0.015, cs["bear"].r - cs["base"].r - 0.015)
            bounds = un.mc_bounds(cs["base"].dr, cs["bear"].dr, cs["bull"].dr, (min(coe_sh), max(coe_sh)),
                                  (-0.015, 0.015))
            stress = un.monte_carlo(cs["bear"], bounds, vc["mc_correlation"], draws,
                                    [int(st["random_seed"]), 31, list(rg.DATES).index(val), FOCUS.index(b)])
            base = ctx.monte_carlo(val, b)
            price, sh = cs["base"].price, cs["base"].shares
            pb = float((base["value_per_share"].dropna() > price).mean())
            ps = float((stress["value_per_share"].dropna() > price).mean())
            for wgt in np.round(np.arange(0, 1.0001, 0.1), 1):
                rows.append({"valuation": val, "bank": T[b], "stress_weight": wgt,
                             "prob_value_above_price": (1 - wgt) * pb + wgt * ps, "p_base_regime": pb,
                             "p_stress_regime": ps, "stress_regime": "bear-case Monte Carlo (2015-18 stress growth, "
                             "bear ER*, bear NIM/CoR)", "evidence_class": "Scenario"})
            del sh
    return pd.DataFrame(rows)


def rate_consistent(ctx) -> pd.DataFrame:
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            base = quote(ctx, val, b)
            for dx in (-0.02, -0.01, 0.01, 0.02):
                for mode in ("rate path only", "COE only", "rate path and COE together"):
                    dr = _shift(c.dr, "rate path", dx) if mode != "COE only" else c.dr
                    r = c.r + dx if mode != "rate path only" else c.r
                    q = quote(ctx, val, b, dr, r)
                    tag = {"rate path only": "RATE", "COE only": "COE", "rate path and COE together": "JOINT"}[mode]
                    rows.append({"valuation": val, "bank": T[b], "shift": dx, "mode": mode,
                                 "scenario_id": f"RC-{val}-{T[b]}-{tag}-{'+' if dx > 0 else '-'}{abs(dx) * 100:.0f}PP",
                                 "scenario_label": f"{T[b]}, {rg.DATES[val]}: {mode}, "
                                                   f"{'+' if dx > 0 else '-'}{abs(dx) * 100:.0f} pp"
                                                   + (" (T-bill path)" if mode == "rate path only" else
                                                      " (COE)" if mode == "COE only" else
                                                      " (T-bill path and COE by the same amount)"),
                                 "value_per_share": q["value_per_share"], "base_value": base["value_per_share"],
                                 "value_change": q["value_per_share"] / base["value_per_share"] - 1,
                                 "rating": q["rating"], "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def named_scenarios(ctx) -> pd.DataFrame:
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            q = quote(ctx, val, b, un._with_roe(c.dr, c.r + 1e-6))
            rows.append({"valuation": val, "bank": T[b], "scenario": "dominance without excess return",
                         "definition": "ROE* = COE (ER* = 0); market shares, growth and every other driver base",
                         "value_per_share": q["value_per_share"], "price": q["price"],
                         "value_to_price": q["value_per_share"] / q["price"], "rating": q["rating"],
                         "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


# --- 5 ---------------------------------------------------------------------------------------------------------------
MIX = {"Customer Deposits: Current Accounts": "deposits_current", "Customer Deposits: Savings": "deposits_savings",
       "Customer Deposits: Time": "deposits_time"}


def _wide_with_manual_mix(root: Path) -> tuple[pd.DataFrame, set]:
    """bank_year_wide indexed by (bank, FY), with deposit-mix cells the workbook lacks filled from validated D29 manual
    entries (``data/interim/manual_values.csv``; fill-review pre-flight B: NMB FY2025). Returns the frame and the
    filled (bank, year, variable) cells, so every use is labelled."""
    w = _wide(root).set_index(["bank_id", "fiscal_year"])
    f = path("interim", root) / "manual_values.csv"
    filled = set()
    if f.exists():
        m = pd.read_csv(f, dtype={"bank_id": str})
        for r in m[m["variable"].isin(list(MIX))].itertuples():
            k, col = (r.bank_id, int(r.fiscal_year)), MIX[r.variable]
            if k in w.index and pd.isna(w.loc[k, col]):
                w.loc[k, col] = float(r.value_full)
                filled.add((r.bank_id, int(r.fiscal_year), col))
    return w, filled


def deposit_scenarios(ctx) -> dict[str, pd.DataFrame]:
    w, filled = _wide_with_manual_mix(ctx.root)
    beta = pd.read_csv(path("tables", ctx.root) / "ch06_deposit_beta.csv", dtype={"bank_id": str})
    rows, sens, floor = [], [], []
    for b in FOCUS:
        a, a0 = w.loc[(b, 2025)], w.loc[(b, 2024)]
        casa = float(a["deposits_current"]) + float(a["deposits_savings"])
        casa_src = ("D29 manual entry (annual report note; not yet in the inputs workbook)"
                    if (b, 2025, "deposits_current") in filled else "inputs workbook")
        e = (float(a["equity_attributable"]) + float(a0["equity_attributable"])) / 2
        c = ctx.cases[("CURR", b, "base")]
        tb = float(c.dr.rate[0])
        hist = []
        for y in range(2012, 2026):
            if (b, y) in w.index and (b, y - 1) in w.index:
                x, x0 = w.loc[(b, y)], w.loc[(b, y - 1)]
                ca = (float(x["deposits_current"]) + float(x["deposits_savings"]) + float(x0["deposits_current"])
                      + float(x0["deposits_savings"])) / 2
                ie = float(x.get("interest_expense_deposits_current", np.nan)) + \
                    float(x.get("interest_expense_deposits_savings", np.nan))
                if np.isfinite(ie) and ca > 0:
                    hist.append((y, abs(ie) / ca))
        cost = hist[-1][1] if hist else np.nan
        iea = float(a["interest_earning_assets"])
        base = quote(ctx, "CURR", b)
        for x in (0.10, 0.20, 0.30, 0.50):
            dcost = x * casa * (tb - cost)
            dnim = -dcost / iea
            q = quote(ctx, "CURR", b, _shift(c.dr, "NIM", dnim))
            rows.append({"bank": T[b], "casa_repriced": x, "casa_tzs": casa, "casa_source": casa_src,
                         "casa_cost": cost, "tbill_v13": tb,
                         "added_interest_cost": dcost, "nim_change": dnim,
                         "roe_change_pts": -dcost * (1 - c.dr.tax) / e, "value_per_share": q["value_per_share"],
                         "base_value": base["value_per_share"], "rating": q["rating"], "evidence_class": "Scenario"})
        bt = beta[(beta["bank_id"] == b) & (beta["deposits"] == "all deposits")].iloc[0]
        for lab, bd in (("estimated deposit beta (all)", float(bt["beta"])),
                        ("falling-rate deposit beta", float(bt["beta_down"])),
                        ("CASA cost at its floor (beta 0)", 0.0)):
            d_roe = -(c.dr.asset_beta * iea - bd * float(a["customer_deposits"])) * 0.01 * (1 - c.dr.tax) / e
            sens.append({"bank": T[b], "case": lab, "deposit_beta": bd, "asset_beta": c.dr.asset_beta,
                         "roe_change_per_1pp_tbill_fall": -d_roe, "beta_lo": float(bt["beta_down_lo"]) if "falling"
                         in lab else (float(bt["beta_lo"]) if "all" in lab else np.nan),
                         "beta_hi": float(bt["beta_down_hi"]) if "falling" in lab else (float(bt["beta_hi"]) if "all"
                                                                                         in lab else np.nan),
                         "evidence_class": "Estimated"})
        if hist:
            hh = pd.DataFrame(hist, columns=["year", "casa_cost"])
            floor.append({"bank": T[b], "casa_cost_latest": cost, "casa_cost_floor": float(hh["casa_cost"].min()),
                          "floor_year": int(hh.loc[hh["casa_cost"].idxmin(), "year"]),
                          "headroom_to_floor": cost - float(hh["casa_cost"].min()), "years": f"FY{hh['year'].min()}-"
                          f"{hh['year'].max()}", "evidence_class": "Derived"})
    # 2018 event test: deposit growth around the January 2018 closures (five banks) and Bank M (August 2018)
    s = pd.read_csv(path("processed", ctx.root) / "sector_totals.csv")
    sysd = s[s["series_id"] == "BOTDB_BANK_DEPOSITS__TOTAL_DEPOSIT"].set_index("year")["value"] * 1e6
    ev = []
    for b in ("B02", "B01", "B03", "B04", "B05", "B06"):
        for y in (2017, 2018, 2019):
            if (b, y) in w.index and (b, y - 1) in w.index:
                g = float(w.loc[(b, y), "customer_deposits"]) / float(w.loc[(b, y - 1), "customer_deposits"]) - 1
                gs = float(sysd[y] / sysd[y - 1] - 1) if y in sysd.index and y - 1 in sysd.index else np.nan
                ev.append({"bank": T[b], "fiscal_year": y, "deposit_growth": g, "system_deposit_growth": gs,
                           "excess_over_system": g - gs, "event": "2018 closures (5 banks, Jan 2018) and Bank M "
                           "(Aug 2018)" if y == 2018 else "", "evidence_class": "Derived"})
    return {"migration": pd.DataFrame(rows), "rate_sensitivity": pd.DataFrame(sens), "casa_floor": pd.DataFrame(floor),
            "event_2018": pd.DataFrame(ev)}


# --- 5b (prompt 13b item 1): asset betas by component, lagged pass-through, repricing-based sensitivity -----------------
ENGINE_NOTE = ("The engine's NIM path does not use these component or lagged betas: it applies one contemporaneous asset "
               "beta (register `asset_beta`: OLS of the annual change in the IEA yield on the annual change in the "
               "T-bill, FY <= anchor) to all interest-earning assets, with no lag: NIM_t = NIM path_t + (asset_beta - "
               "deposit_beta x D / IEA) x (rate_t - rate_0).")


def _consecutive_diff(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    d = s.diff()
    return d[s.index.to_series().diff() == 1]


def _dl_betas(y: pd.Series, r: pd.Series, lags: int) -> list[dict]:
    """Distributed-lag OLS of the annual change in a yield on the T-bill change at t ... t-lags; cumulative pass-through
    after h = 1 ... lags+1 years with a 90% t interval (variance of the coefficient sum)."""
    from scipy import stats

    dy, dr = _consecutive_diff(y), _consecutive_diff(r)
    rows = []
    for t in dy.index:
        xs = [dr.get(t - j, np.nan) for j in range(lags + 1)]
        if np.all(np.isfinite(xs)):
            rows.append((t, dy[t], *xs))
    n = len(rows)
    k = lags + 2
    if n < k + 3:
        return [{"horizon_years": h + 1, "n": n, "beta": np.nan, "lo": np.nan, "hi": np.nan,
                 "years": f"FY{rows[0][0]}-{rows[-1][0]}" if rows else "",
                 "status": f"not estimated: n = {n} < {k + 3}"} for h in range(lags + 1)]
    a = np.array(rows, dtype=float)
    X = np.column_stack([np.ones(n), a[:, 2:]])
    b, *_ = np.linalg.lstsq(X, a[:, 1], rcond=None)
    e = a[:, 1] - X @ b
    cov = (e @ e / (n - k)) * np.linalg.inv(X.T @ X)
    tc = stats.t.ppf(0.95, n - k)
    out = []
    for h in range(lags + 1):
        wv = np.zeros(k)
        wv[1:h + 2] = 1
        c, se = float(wv @ b), float(np.sqrt(wv @ cov @ wv))
        out.append({"horizon_years": h + 1, "n": n, "beta": c, "lo": c - tc * se, "hi": c + tc * se,
                    "years": f"FY{int(a[0, 0])}-{int(a[-1, 0])}", "status": "estimated"})
    return out


def _bond_index(root: Path) -> pd.Series:
    """Tenor-adjusted annual Treasury bond auction yield (OLS of WAYTM on year and tenor effects; 5-year tenor)."""
    b = pd.read_csv(path("interim", root) / "bond_auction_results.csv")
    b = b[b["status"] == "SUCCESS"].copy()
    b["y"] = pd.to_datetime(b["auction_date"]).dt.year
    b["ten"] = b["bond_title"].str.extract(r"(\d+)-YEAR")[0].astype(float)
    b = b.dropna(subset=["waytm", "ten"])
    ys, ts = sorted(b["y"].unique()), sorted(b["ten"].unique())
    ref = 5.0 if 5.0 in ts else ts[0]
    X = np.column_stack([(b["y"] == y).to_numpy(float) for y in ys]
                        + [(b["ten"] == t).to_numpy(float) for t in ts if t != ref])
    coef, *_ = np.linalg.lstsq(X, b["waytm"].to_numpy(float), rcond=None)
    return pd.Series(coef[:len(ys)], index=[int(y) for y in ys])


def _repricing(root: Path) -> pd.DataFrame:
    """IFRS 7 repricing profiles (raw manual entries), each value checked on its annual-report page."""
    from tzbank.ingest.web_collected import _in_pdf

    d = pd.read_csv(root / "data" / "raw" / "manual_entries" / "repricing_profile.csv", dtype={"bank_id": str})
    d["page_check"] = [("zero (printed '-')" if v == 0 else "value found on the cited page"
                        if _in_pdf(root, doc, str(pg), float(v), "level") else "NOT FOUND on the cited page")
                       for v, doc, pg in zip(d["value_tzs_mn"], d["source_doc"], d["page"], strict=True)]
    key = ["bank_id", "line_as_published"]
    tot = d[d["bucket"] == "total"].set_index(key)["value_tzs_mn"]
    sm = d[d["bucket"] != "total"].groupby(key)["value_tzs_mn"].sum()
    chk = {k: "buckets add to the printed total" if abs(sm[k] - tot[k]) <= 2 else
           f"source inconsistency: printed buckets sum to {sm[k]:,.0f} vs printed total {tot[k]:,.0f} (kept as printed)"
           for k in tot.index}
    d["bucket_sum_check"] = [chk.get((b, ln), "") for b, ln in zip(d["bank_id"], d["line_as_published"], strict=True)]
    return d


def _frac(window: tuple[float, float], h: int, grid: int = 2000) -> float:
    """Average share of a bucket repriced during forecast year h (uniform repricing within the window)."""
    a, b = window
    s = h - 1 + (np.arange(grid) + 0.5) / grid
    f = np.where(s >= b, 1.0, 0.0) if b <= a else np.clip((s - a) / (b - a), 0.0, 1.0)
    return float(f.mean())


def _beta_rows(bank: str, comp: str, y: pd.Series, rate: pd.Series, rate_name: str, used: str, lags: int) -> list:
    from tzbank.chapters.ch06_funding import beta_ols

    dy, dr = _consecutive_diff(y), _consecutive_diff(rate)
    yrs = dy.index.intersection(dr.index)
    o = beta_ols(dy[yrs].to_numpy(), dr[yrs].to_numpy()) if len(yrs) else {"n": 0, "status": "not estimated: no "
                                                                                              "overlapping years"}
    out = [{"bank": bank, "component": comp, "rate": rate_name, "method": "OLS on annual changes, contemporaneous",
            "frequency": "annual", "horizon_years": 1, "years": f"FY{min(yrs)}-{max(yrs)}" if len(yrs) else "",
            "n": int(o.get("n", 0)), "beta": o.get("beta", np.nan), "lo": o.get("lo", np.nan),
            "hi": o.get("hi", np.nan), "status": o.get("status", ""), "used_in_engine": used,
            "evidence_class": "Estimated"}]
    for r in _dl_betas(y, rate, lags):
        out.append({"bank": bank, "component": comp, "rate": rate_name, "method": f"distributed lag (T-bill change "
                    f"at t to t-{lags}), cumulative", "frequency": "annual", **r, "used_in_engine": "no",
                    "evidence_class": "Estimated"})
    return out


def asset_beta_diagnostics(ctx) -> dict[str, pd.DataFrame]:
    """Prompt 13b item 1. (i) Asset betas by component (loans; government securities; interbank placements; non-loan
    earning assets as the long-history proxy for securities + interbank; all IEA as the register uses): sample, annual
    frequency, n, OLS on annual changes and 90% interval; (ii) lagged pass-through: cumulative betas after 1-3 years
    (distributed lag on the T-bill), and bond auction yields on the T-bill; (iii) ROE change per 1 pp sustained T-bill
    fall in years 1-3 with the securities and interbank books repricing at contractual repricing / maturity (FY2025
    IFRS 7 profiles), loans at the estimated cumulative loan beta, deposits at the chapter 6 deposit beta and
    wholesale borrowings repricing on their profile; against the engine's single contemporaneous asset beta."""
    from tzbank.valuation.register import _tbill

    root = ctx.root
    cfg = settings(root)["support"]
    lags = int(cfg["beta_lags"])
    pt = float(cfg["repricing_passthrough"])
    win = {k: tuple(v) for k, v in cfg["repricing_windows"].items()}
    w = _wide(root)
    wi = w.set_index(["bank_id", "fiscal_year"])
    dpl = pd.read_csv(path("tables", root) / "ch04_4_3_deployment.csv", dtype={"bank_id": str})
    dbeta = pd.read_csv(path("tables", root) / "ch06_deposit_beta.csv", dtype={"bank_id": str})
    tb = _tbill(root, 2025)
    tb.index = tb.index.astype(int)
    rate_name = "T-bill (BoT overall, annual; register series)"
    brows = []
    for b in FOCUS:
        x = w[w["bank_id"] == b].set_index("fiscal_year").sort_index()
        d = dpl[dpl["bank_id"] == b].set_index("fiscal_year").sort_index()
        nl = x["interest_earning_assets"] - x["net_loans"]
        cons = x.index.to_series().diff() == 1
        avg_nl = ((nl + nl.shift(1)) / 2).where(cons)
        proxy = ((x["interest_income"] - x["interest_income_loans"]) / avg_nl).where(avg_nl > 0)
        comps = {"loans": d["loan_yield"], "government securities (incl. bonds)": d["securities_yield"],
                 "interbank placements": d["placements_yield"],
                 "non-loan earning assets (securities + interbank; proxy)": proxy, "all IEA": d["iea_yield"]}
        for comp, y in comps.items():
            brows += _beta_rows(T[b], comp, y[y.index <= 2025], tb, rate_name,
                                "yes (register asset_beta)" if comp == "all IEA" else "no", lags)
    bond = _bond_index(root)
    brows += _beta_rows("market", "Treasury bond auction yield (5-year tenor-adjusted)", bond[bond.index <= 2025], tb,
                        rate_name, "no", lags)
    beta = pd.DataFrame(brows)

    rep = _repricing(root)
    sens = []
    for b in FOCUS:
        a, a0 = wi.loc[(b, 2025)], wi.loc[(b, 2024)]
        e = (float(a["equity_attributable"]) + float(a0["equity_attributable"])) / 2
        c = ctx.cases[("CURR", b, "base")]
        tax = c.dr.tax
        bt = dbeta[(dbeta["bank_id"] == b) & (dbeta["deposits"] == "all deposits")].iloc[0]
        dep_beta, deps, loans = float(bt["beta"]), float(a["customer_deposits"]), float(a["gross_loans"])
        lb = beta[(beta["bank"] == T[b]) & (beta["component"] == "loans")]
        lb_c = float(lb[lb["method"].str.startswith("OLS")]["beta"].iloc[0])
        r = rep[rep["bank_id"] == b]
        for h in (1, 2, 3):
            parts = {}
            for cls, sign in (("government_securities", -1), ("placements", -1), ("borrowings", 1)):
                g = r[(r["asset_class"] == cls) & r["bucket"].isin(list(win))]
                parts[cls] = sign * 0.01 * pt * sum(float(v) * 1e6 * _frac(win[k], h)
                                                    for k, v in zip(g["bucket"], g["value_tzs_mn"], strict=True))
            ld = lb[lb["method"].str.startswith("distributed") & (lb["horizon_years"] == h)]
            lbeta = float(ld["beta"].iloc[0]) if len(ld) and np.isfinite(ld["beta"].iloc[0]) else lb_c
            parts["loans"] = -0.01 * lbeta * loans
            parts["customer_deposits"] = 0.01 * dep_beta * deps
            nii = sum(parts.values())
            sens.append({"bank": T[b], "case": "repricing at maturity (securities, interbank, borrowings) + lagged "
                         "loan beta", "year": h, "securities_tzs": parts["government_securities"],
                         "interbank_tzs": parts["placements"], "loans_tzs": parts["loans"],
                         "deposits_tzs": parts["customer_deposits"], "borrowings_tzs": parts["borrowings"],
                         "loan_beta_used": lbeta, "deposit_beta_used": dep_beta, "asset_beta_engine": np.nan,
                         "nii_change_tzs": nii, "roe_change_pts": nii * (1 - tax) / e, "evidence_class": "Estimated"})
        iea = float(a["interest_earning_assets"])
        nii = -0.01 * (c.dr.asset_beta * iea - dep_beta * deps)
        sens.append({"bank": T[b], "case": "engine: one contemporaneous asset beta on all IEA (register)", "year": 1,
                     "loan_beta_used": np.nan, "deposit_beta_used": dep_beta, "asset_beta_engine": c.dr.asset_beta,
                     "nii_change_tzs": nii, "roe_change_pts": nii * (1 - tax) / e, "evidence_class": "Estimated"})
    sens = pd.DataFrame(sens)
    sens["engine_uses_these_betas"] = "no: " + ENGINE_NOTE
    return {"asset_beta": beta, "rate_sensitivity_repricing": sens, "repricing_profile": rep.assign(
        evidence_class="Observed")}


# --- 6 ---------------------------------------------------------------------------------------------------------------
def valuation_bands(root: Path) -> pd.DataFrame:
    w = _wide(root)
    rows = []
    for b in FOCUS:
        x = w[w["bank_id"] == b].set_index("fiscal_year").sort_index()
        mc = x["share_price_close"] * x["shares_outstanding"]
        pb = mc / x["equity_attributable"]
        pe = mc / x["profit_attributable"]
        for y in x.index:
            if np.isfinite(pb.get(y, np.nan)):
                rows.append({"bank": T[b], "fiscal_year": int(y), "pb": pb[y], "pe": pe[y], "evidence_class": "Derived"})
    d = pd.DataFrame(rows)
    out = []
    for bank, g in d.groupby("bank"):
        for m in ("pb", "pe"):
            s = g[m].dropna()
            out.append({"bank": bank, "multiple": m.upper().replace("PB", "P/B").replace("PE", "P/E"),
                        "years": f"FY{g['fiscal_year'].min()}-{g['fiscal_year'].max()}", "mean": s.mean(),
                        "minus_1sd": s.mean() - s.std(), "plus_1sd": s.mean() + s.std(), "min": s.min(), "max": s.max(),
                        "latest_fy": float(s.iloc[-1]), "latest_percentile": float((s < s.iloc[-1]).mean()),
                        "evidence_class": "Derived"})
    return pd.concat([d.assign(kind="history"), pd.DataFrame(out).assign(kind="band")], ignore_index=True)


def kpi_table(root: Path) -> pd.DataFrame:
    w = _wide(root)
    d = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv", dtype={"bank_id": str})
    dep = pd.read_csv(path("tables", root) / "ch06_deposits.csv", dtype={"bank_id": str})
    rows = []
    for b in T:
        x = w[w["bank_id"] == b].set_index("fiscal_year")
        yrs = [y for y in x.index if pd.notna(x.loc[y, "total_assets"])]
        if not yrs:
            continue
        y = max(yrs)
        r = x.loc[y]
        dd = d[(d["bank_id"] == b) & (d["fiscal_year"] == y)]
        cd = dep[(dep["bank_id"] == b) & (dep["fiscal_year"] == y)]
        mc = r["share_price_close"] * r["shares_outstanding"]
        mc_basis = "year-end price x shares"
        if pd.isna(mc) and pd.notna(r.get("market_cap")):  # smaller banks: the reported year-end market cap (Phase 1)
            mc, mc_basis = r["market_cap"], "reported market capitalisation"
        eq, eq_basis = r["equity_attributable"], "equity attributable"
        if pd.isna(eq) and pd.notna(r.get("total_equity")):
            eq, eq_basis = r["total_equity"], "total equity (no NCI split held)"
        rows.append({"bank": T[b], "fiscal_year": int(y), "roe": float(dd["roe"].iloc[0]) if len(dd) else np.nan,
                     "roa": float(dd["roa"].iloc[0]) if len(dd) and "roa" in dd else np.nan,
                     "cost_to_income": r.get("cost_to_income_reported"), "npl_ratio": r.get("npl_ratio"),
                     "npl_coverage": r.get("npl_coverage_ratio"), "tier1_ratio": r.get("tier1_ratio"),
                     "loan_to_deposit": r.get("loan_to_deposit_ratio_reported"),
                     "cost_of_deposits": float(cd["cost_of_deposits"].iloc[0]) if len(cd) else np.nan,
                     "pb": mc / eq if pd.notna(mc) and pd.notna(eq) and eq > 0 else np.nan,
                     "pb_basis": f"{mc_basis} / {eq_basis}",
                     "pe": mc / r["profit_attributable"] if pd.notna(mc) and r["profit_attributable"] > 0 else np.nan,
                     "dividend_yield": r.get("dps") / r["share_price_close"] if pd.notna(r.get("dps")) and pd.notna(
                         r["share_price_close"]) else np.nan, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- 7 ---------------------------------------------------------------------------------------------------------------
def capacity_windows(root: Path) -> pd.DataFrame:
    from tzbank.chapters.ch12_usd_implementation import _cfg, fee_per_side, macro_at, vdates, window

    vd = vdates(root)
    w = window(root, vd["CURR"])
    cfg = _cfg(root)
    fx, _ = macro_at(root, vd["CURR"], "macro_tzs_per_usd")
    rows = []
    for lab, lo, hi in (("pre-rally (to 31 Dec 2025)", None, vd["HIST"]),
                        ("2026 (1 Jan - 31 Oct 2026)", pd.Timestamp("2026-01-01"), vd["CURR"])):
        x = w[(w["date"] <= hi) & ((w["date"] >= lo) if lo is not None else True)]
        for bid, g in x.groupby("bank_id"):
            nb = g[~g["block"]]
            adv = float(nb["turnover"].mean())
            for usd in cfg["positions_usd"]:
                q = 0.20 * adv
                rows.append({"window": lab, "bank": T[bid], "start": g["date"].min().date(), "end": g["date"].max()
                             .date(), "exchange_days": len(g), "adv_tzs_ex_blocks": adv,
                             "block_share_of_turnover": float(g.loc[g["block"], "turnover"].sum() / g["turnover"].sum()),
                             "position_usd": usd, "days_to_build_at_20pct": int(np.ceil(usd * fx / q)),
                             "fee_per_side": fee_per_side(q, cfg["fees"]), "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def block_board(root: Path) -> pd.DataFrame:
    from tzbank.chapters.ch12_usd_implementation import vdates, window

    w = window(root, vdates(root)["CURR"])
    w["quarter"] = w["date"].dt.to_period("Q").astype(str)
    rows = []
    for (bid, q), g in w.groupby(["bank_id", "quarter"]):
        bl, tot = float(g.loc[g["block"], "turnover"].sum()), float(g["turnover"].sum())
        rows.append({"bank": T[bid], "quarter": q, "block_turnover_tzs": bl, "board_turnover_tzs": tot - bl,
                     "block_share": bl / tot if tot else np.nan, "exchange_days": len(g),
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- 8 ---------------------------------------------------------------------------------------------------------------
def franchise_breakeven(ctx) -> pd.DataFrame:
    fr = pd.read_csv(path("tables", ctx.root) / "ch06_franchise.csv", dtype={"bank_id": str})
    rows = []
    for b in FOCUS:
        g = fr[fr["bank_id"] == b].sort_values("fiscal_year")
        r = g.iloc[-1]
        dep = r["avg_deposits"]
        opex_alloc = r["gdfa"] - r["reserve_cost"] - r["deposit_insurance"] - r["net_franchise_central"]
        be = (r["cost_of_deposits"] * dep + r["deposit_insurance"] + opex_alloc) / (dep * (1 - r["smr_avg"]))
        roe_pts_per_nf = r["net_franchise_central_roe_pts"] / r["net_franchise_central"] if r[
            "net_franchise_central"] else np.nan
        for lab, tb in (("31 Dec 2025 register path", float(ctx.cases[("HIST", b, "base")].dr.rate[0])),
                        ("v1.3 path (31 Oct 2026)", float(ctx.cases[("CURR", b, "base")].dr.rate[0]))):
            nf = (tb - r["cost_of_deposits"]) * dep - r["smr_avg"] * dep * tb - r["deposit_insurance"] - opex_alloc
            rows.append({"bank": T[b], "fiscal_year": int(r["fiscal_year"]), "tbill_case": lab, "tbill": tb,
                         "breakeven_tbill": be, "net_franchise_tzs": nf, "net_franchise_roe_pts": nf * roe_pts_per_nf,
                         "history_mean_roe_pts": float(g["net_franchise_central_roe_pts"].mean()),
                         "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def surplus_drag(ctx) -> pd.DataFrame:
    """Capital surplus earning the T-bill after tax instead of the COE (prompt 13b item 3), FY anchor at 31 Oct 2026.

    Surplus = max(0, Tier 1 ratio - hurdle) x RWA on the register's binding basis (the tighter of Tier 1 and total
    capital with Tier 2 held; NMB Bank-solo, D2), against three hurdles, each labelled: (a) the regulatory minimum ex
    the conservation buffer (requirement - ``capital_rules.ccb_rate``); the requirement incl. the buffer (reference);
    (b) the model's target k* (requirement + bank buffer + D-SIB, register base). Drag = surplus x (COE - T-bill x
    (1 - tax)); share of FY attributable profit.
    """
    cap = pd.read_csv(path("tables", ctx.root) / "ch07_capital.csv", dtype={"bank_id": str})
    reg = pd.read_csv(ctx.root / "config" / "assumption_register.csv")
    ccb = float(settings(ctx.root)["capital_rules"]["ccb_rate"])
    w = _wide(ctx.root).set_index(["bank_id", "fiscal_year"])
    rows = []
    for b in FOCUS:
        g = cap[(cap["bank_id"] == b) & cap["tier1_ratio"].notna()].sort_values("fiscal_year").iloc[-1]
        y = int(g["fiscal_year"])
        rwa, t1 = float(g["rwa_total"]), float(g["tier1_ratio"])
        c = ctx.cases[("CURR", b, "base")]
        k = reg[(reg["valuation"] == "CURR") & (reg["bank"] == T[b]) & (reg["symbol"] == "k_star")].iloc[0]
        req = float(k["range_low"])  # requirement on the binding constraint, Tier 1 terms (register, D52 / D55)
        np_ = float(w.loc[(b, y), "profit_attributable"])
        tb = float(c.dr.rate[0])
        for lab, hurdle in (("(a) regulatory minimum ex conservation buffer", req - ccb),
                            ("requirement incl. conservation buffer (reference)", req),
                            ("(b) model target k* (requirement + bank buffer + D-SIB, register base)",
                             float(c.dr.k_star))):
            surplus = max(0.0, t1 - hurdle) * rwa
            drag = surplus * (c.r - tb * (1 - c.dr.tax))
            rows.append({"bank": T[b], "fiscal_year": y, "hurdle": lab, "hurdle_ratio": hurdle, "tier1_ratio": t1,
                         "capital_basis": g["capital_basis"], "headroom": t1 - hurdle, "rwa": rwa,
                         "surplus_capital_tzs": surplus, "coe": c.r, "tbill_v13": tb, "tax": c.dr.tax,
                         "drag_tzs": drag, "profit_attributable": np_, "drag_share_of_profit": drag / np_,
                         "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


FIN_DEFS = ("NPL = stage-3 (IFRS 9) gross loans / gross loans and NPL_coverage = loss allowance / stage-3 loans "
            "(chapter 8, ch08_credit_history); CIR = operating expenses / (NII + non-interest income) (chapter 4.5, as "
            "the engine); LDR = gross loans / customer deposits (chapter 6 liquidity); LDR_net = net loans / customer "
            "deposits (the engine's L / D, comparable across A and E years); *_reported = as printed by the bank")


def financial_table(ctx) -> pd.DataFrame:
    w = _wide(ctx.root).set_index(["bank_id", "fiscal_year"])
    h8 = pd.read_csv(path("tables", ctx.root) / "ch08_credit_history.csv", dtype={"bank_id": str})
    ch8 = {(r.bank_id, int(r.fiscal_year)): {"npl_ratio": r.npl_ratio, "coverage": r.coverage} for r in h8.itertuples()}
    rows = []
    for b in FOCUS:
        c = ctx.cases[("CURR", b, "base")]
        f = project(c.op, c.dr).rows.set_index("year")
        split = _split(ctx.root, b)
        sh_post = c.shares
        for y in (2024, 2025, 2026, 2027, 2028):
            if y <= 2025:
                a = w.loc[(b, y)]
                a0 = w.loc[(b, y - 1)]
                sh = float(a["shares_outstanding"]) * split
                e_avg = (float(a["equity_attributable"]) + float(a0["equity_attributable"])) / 2
                iea = (float(a["interest_earning_assets"]) + float(a0["interest_earning_assets"])) / 2
                gl = (float(a["gross_loans"]) + float(a0["gross_loans"])) / 2
                price = float(a["share_price_close"]) / split
                rec = {"NII": a["net_interest_income"], "NonII": a["non_interest_income"],
                       "OpEx": -abs(a["operating_expenses"]), "Imp": -abs(a["impairment_charges"]),
                       "PBT": a["profit_before_tax"], "NP_attr": a["profit_attributable"], "net_loans": a["net_loans"],
                       "deposits": a["customer_deposits"], "total_assets": a["total_assets"],
                       "equity": a["equity_attributable"], "NIM": a["net_interest_income"] / iea,
                       "CIR": abs(a["operating_expenses"]) / (a["net_interest_income"] + a["non_interest_income"]),
                       "CIR_reported": a.get("cost_to_income_reported"), "CoR": abs(a["impairment_charges"]) / gl,
                       "NPL": ch8.get((b, y), {}).get("npl_ratio", np.nan), "NPL_reported": a.get("npl_ratio"),
                       "NPL_coverage": ch8.get((b, y), {}).get("coverage", np.nan),
                       "NPL_coverage_reported": a.get("npl_coverage_ratio"), "Tier1": a.get("tier1_ratio"),
                       "LDR": a["gross_loans"] / a["customer_deposits"],
                       "LDR_net": a["net_loans"] / a["customer_deposits"],
                       "EPS": a["profit_attributable"] / sh, "DPS": (a.get("dps_total") if pd.notna(
                           a.get("dps_total")) else a.get("dps")) / split, "BVPS": a["equity_attributable"] / sh,
                       "ROE": a["profit_attributable"] / e_avg, "price_basis": f"FY{y} year-end price", "price": price,
                       "kind": "A"}
            else:
                r, r0 = f.loc[y], f.loc[y - 1]
                sh = sh_post
                e_avg = (float(r["E"]) + float(r0["E"])) / 2
                rec = {"NII": r["NII"], "NonII": r["NonII"], "OpEx": -abs(r["OpEx"]), "Imp": -abs(r["Imp"]),
                       "PBT": r["PBT"], "NP_attr": r["NP_attr"], "net_loans": r["L"], "deposits": r["D"],
                       "total_assets": r["TA"], "equity": r["E"], "NIM": r["NIM_eff"],
                       "CIR": abs(r["OpEx"]) / (r["NII"] + r["NonII"]), "CoR": float(c.dr.CoR[min(y - 2026,
                                                                                                 len(c.dr.CoR) - 1)]),
                       "CIR_reported": np.nan, "NPL": np.nan, "NPL_reported": np.nan, "NPL_coverage": np.nan,
                       "NPL_coverage_reported": np.nan, "Tier1": r["T1"] / r["RWA"], "LDR": np.nan,
                       "LDR_net": r["L"] / r["D"],
                       "EPS": r["NP_attr"] / sh, "DPS": r["Div"] / sh, "BVPS": r["E"] / sh,
                       "ROE": r["NP_attr"] / e_avg, "price_basis": "price at 31 Oct 2026", "price": c.price,
                       "kind": "E"}
            rec["payout"] = rec["DPS"] / rec["EPS"] if rec["EPS"] else np.nan
            rec["P/E"] = rec["price"] / rec["EPS"] if rec["EPS"] else np.nan
            rec["P/B"] = rec["price"] / rec["BVPS"]
            rec["dividend_yield"] = rec["DPS"] / rec["price"]
            rows.append({"bank": T[b], "fiscal_year": f"FY{y}{rec.pop('kind')}", **{k: (float(v) if not isinstance(
                v, str) and v is not None else v) for k, v in rec.items()},
                "per_share_basis": "post-split NMB (1:10, 24 Aug 2026)" if split != 1 else "ordinary shares",
                "ratio_definitions": FIN_DEFS,
                "evidence_class": "Observed" if y <= 2025 else "Scenario"})
    return pd.DataFrame(rows)


def price_export(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tzbank.chapters.ch09_valuation import BUY, DOWN

    m = pd.read_csv(path("processed", ctx.root) / "market_daily.csv", dtype={"bank_id": str})
    m["date"] = pd.to_datetime(m["date"])
    end = pd.Timestamp("2026-10-31")
    x = m[(m["date"] > end - pd.DateOffset(years=1)) & (m["date"] <= end) & m["bank_id"].isin(FOCUS)].copy()
    x = x[["date", "bank_id", "adj_close", "turnover", "traded_day"]].assign(bank=x["bank_id"].map(T),
                                                                            evidence_class="Observed")
    lines = []
    for b in FOCUS:
        g = x[x["bank_id"] == b]
        q = quote(ctx, "CURR", b)
        num = q["target_12m"] + q["dps_12m"]
        lines += [{"bank": T[b], "line": "52-week high", "price": float(g["adj_close"].max()), "evidence_class":
                   "Observed"},
                  {"bank": T[b], "line": "52-week low", "price": float(g["adj_close"].min()), "evidence_class":
                   "Observed"},
                  {"bank": T[b], "line": "Buy below (expected return > COE + 5 pp)", "price": num / (1 + q["coe"] + BUY),
                   "evidence_class": "Scenario"},
                  {"bank": T[b], "line": "Buy below (bear-case downside within 25%, D4 screen)",
                   "price": un.value(ctx.cases[("CURR", b, "bear")]) / ctx.cases[("CURR", b, "bear")].shares
                   / (1 + DOWN), "evidence_class": "Scenario"},
                  {"bank": T[b], "line": "Sell above (expected return < COE - 5 pp)",
                   "price": num / (1 + q["coe"] - BUY), "evidence_class": "Scenario"},
                  {"bank": T[b], "line": "value per share (v1.3, 31 Oct 2026)", "price": q["value_per_share"],
                   "evidence_class": "Estimated"},
                  {"bank": T[b], "line": "price at 31 Oct 2026", "price": q["price"], "evidence_class": "Observed"}]
    return x.drop(columns=["bank_id"]), pd.DataFrame(lines)


def price_monthly(px: pd.DataFrame) -> pd.DataFrame:
    """Monthly view of the 1-year daily series (report annex 'price history'; prompt 16): month-end close, high and
    low close, turnover (TZS) and days traded. Grouping only (Observed)."""
    d = px.assign(date=pd.to_datetime(px["date"])).sort_values("date")
    g = d.groupby(["bank", d["date"].dt.to_period("M")], sort=False)
    out = g.agg(close=("adj_close", "last"), high=("adj_close", "max"), low=("adj_close", "min"),
                turnover_tzs=("turnover", "sum"), days_traded=("traded_day", "sum")).reset_index()
    out["month"] = out["date"].dt.strftime("%b %Y")
    return out.drop(columns=["date"])[["bank", "month", "close", "high", "low", "turnover_tzs", "days_traded"]].assign(
        evidence_class="Observed")


# --- 13 --------------------------------------------------------------------------------------------------------------
def usd_exports(ctx) -> pd.DataFrame:
    """Foreign-investor figures on the chapter 12 USD return (prompt 24 rebuild: target price and each dividend on its
    payment date translated at the scenario's rate, withholding on dividends only, no trading cost; the round-trip
    cost is reported separately in ``ch12_expected_usd``)."""
    from tzbank.chapters.ch12_usd_implementation import _cfg, hurdle, usd_total_return

    cfg = _cfg(ctx.root)
    w = float(cfg["div_wht_foreign"])
    ann = pd.read_csv(path("tables", ctx.root) / "ch12_returns_annual.csv")
    dv = pd.read_csv(path("tables", ctx.root) / "ch09_dividends_12m.csv")
    rows = []
    for val in rg.DATES:
        for b in FOCUS:
            q = quote(ctx, val, b)
            h, _ = hurdle(ctx.root, val, T[b])
            dd = dv[(dv["valuation"] == val) & (dv["bank"] == T[b])]
            p0, p1 = q["price"], q["target_12m"]
            divs = [(float(r.dps) / p0, float(r.years_from_valuation)) for r in dd.itertuples()]
            pr = p1 / p0 - 1
            for d in DEPRECIATION:
                rows.append({"valuation": val, "bank": T[b], "measure": "expected USD return", "depreciation": d,
                             "value": usd_total_return(pr, divs, w, d), "usd_hurdle": h, "evidence_class": "Scenario"})
                # entry price P at which the USD total return equals the hurdle: [P1 / (1 + d) + sum D_i (1 - w) /
                # (1 + d)^t_i] / (1 + h)
                pv = p1 / (1 + d) + sum(y * p0 * (1 - w) / (1 + d) ** t for y, t in divs)
                rows.append({"valuation": val, "bank": T[b], "measure": "entry price meeting the USD hurdle",
                             "depreciation": d, "value": pv / (1 + h), "usd_hurdle": h, "evidence_class": "Scenario"})
            from scipy.optimize import brentq

            be = float(brentq(lambda d, pr=pr, divs=divs, h=h: usd_total_return(pr, divs, w, d) - h, -0.95, 3.0))
            rows.append({"valuation": val, "bank": T[b], "measure": "break-even depreciation", "depreciation": np.nan,
                         "value": be, "usd_hurdle": h, "evidence_class": "Scenario"})
            bear = un.value(ctx.cases[(val, b, "bear")]) / ctx.cases[(val, b, "base")].shares
            joint = usd_total_return(bear / p0 - 1, divs, w, 0.245)
            rows.append({"valuation": val, "bank": T[b], "measure": "joint 2015-style stress (bear value, +24.5% "
                         "depreciation)", "depreciation": 0.245, "value": joint, "usd_hurdle": h,
                         "evidence_class": "Scenario"})
    for bank, g in ann.dropna(subset=["usd_return"]).groupby("bank"):
        r = g.sort_values("usd_return").iloc[0]
        yr = int(str(r["period"])[:4])
        rows.append({"valuation": "history", "bank": bank, "measure": "worst calendar-year USD return", "year": yr,
                     "depreciation": r["fx_change"], "value": r["usd_return"], "usd_hurdle": np.nan,
                     "evidence_class": "Derived"})
        rows.append({"valuation": "history", "bank": bank, "measure": "year of the worst calendar-year USD return",
                     "year": yr, "depreciation": np.nan, "value": yr, "usd_hurdle": np.nan, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- write -----------------------------------------------------------------------------------------------------------
NOTE = """# Chapter 13 (report support): exhibits for the owner's drafts

Generated by `tzbank.chapters.ch13_report_support` (stage `support`; prompt 13). Every exhibit is a pipeline output
(D66), shown on the dashboard page "13. Report support" and in the findings pack under `CH13.*`. Register {version},
input hash `{hash}`. No report text is written; the drafts in `drafts/` are checked against the pack
(`outputs/findings/fill_list.md`).

## Research question

Support for the owner's reports: reconciliations, bridges, sensitivities, exports and benchmarks that the drafts mark
`[to rerun]`, `[refresh: pack]` or `[export]`.

## Variable definitions

See the module docstring of `tzbank.chapters.ch13_report_support` for each exhibit's definition.

## Sources and transformation trail

Inputs are the processed tables, chapter outputs (4-12) and the valuation context (register {version}); sector
totals from `data/raw/manual_entries/web_collected.csv` (WC-041 to WC-083, BoT supervision report appendices 2013-2025);
repricing profiles from `data/raw/manual_entries/repricing_profile.csv` (NMB and CRDB FY2025 annual reports, each value
page-checked); Damodaran country risk from `data/raw/documents/damodaran`; peers and benchmarks from
`data/raw/manual_entries/peers_collected.csv` and `benchmarks_collected.csv` (documents in
`data/raw/documents/benchmarks`), each row with source URL, publication and access dates.

## Validation test

`tests/test_ch13.py`: ROE reconciliation identity, bridge sums, rating-flip consistency, mixture linearity,
financial-table identities, entry-price identity.

## Results

{results}

## Evidence class

Reported figures: **Observed**; ratios and bridges from them: **Derived**; forecasts, sensitivities and scenarios:
**Scenario** / **Estimated**.

## Limitations

- Sector totals (exhibit 2) cover all banking institutions (BoT supervision report appendices, latest vintage per
  year), not commercial banks only; 2023 sector profit is not usable in any vintage held (stays missing); the NMB +
  CRDB numerator is Group (CRDB includes Burundi and DRC subsidiaries), so profit shares are upper bounds.
- Asset betas (exhibit 5b): government-securities and interbank balances are held from FY2020 only, so their own
  betas are not estimable (n = 4); the non-loan earning-asset yield is the long-history proxy. The T-bill series has no
  2016 value, so lagged estimates use n = 9-12 annual changes. The repricing sensitivity uses one FY2025 profile per
  bank and a pass-through of `support.repricing_passthrough` on repricing.
- The DSE all-share index is not held: the 1-year price export has no index series (collection task).
- Peer and benchmark tables hold only what could be collected with dated sources; shortfalls are listed in the
  tables.
- Exhibit charts for the drafts are written to `outputs/exhibits/<report>/<exhibit_no>.png` with their plotted data
  (`outputs/exhibits/index.csv`); table exhibits are filled from the findings pack.
- Bridges between reported and forecast figures mix reported group accounts with the engine's definitions; the
  residual step absorbs the difference.
"""


def constant_coe_sensitivity(root: Path) -> pd.DataFrame:
    """Chapter 5 excess returns measured against one constant cost of equity, the valuation's central COE at 31 Oct
    2026 (``ch09_coe_ab.coe_used``), instead of each year's own COE (prompt 25, 7.1; previously a side calculation,
    D66; Blueprint ch. 5). Per focus bank, on the chapter 5 estimation sample: mean excess return on each basis, the
    constant-basis mean at +/- 2 points, the AR(1) slope (OLS) and the Chow 2015-18 p-value on each basis. A constant
    shift moves the mean one for one and leaves the AR(1) slope and the Chow test of the ROE series unchanged."""
    from tzbank.chapters.ch05 import models

    e = pd.read_csv(path("tables", root) / "ch05_excess_returns.csv")
    ab = pd.read_csv(path("tables", root) / "ch09_coe_ab.csv")
    c = float(ab.loc[ab["valuation"] == "CURR", "coe_used"].iloc[0])
    rows = []
    for b, name in (("B02", "NMB"), ("B01", "CRDB")):
        x = e[(e["bank_id"] == b) & e["in_estimation"].astype(bool)].set_index("fiscal_year").sort_index()
        out = {"bank": name, "years": f"{int(x.index.min())}-{int(x.index.max())}", "coe_constant": c}
        for basis, ser in (("own", x["er"].astype(float)), ("constant", x["roe"].astype(float) - c)):
            d = models.pairs(ser)
            yrs = d["year"].to_numpy()
            out[f"mean_er_{basis}"] = float(ser.mean())
            out[f"ar1_{basis}"] = float(models.ar1(d)["lambda"])
            out[f"p_chow_episode_{basis}"] = float(models.chow(d, (yrs >= 2015) & (yrs <= 2018))["p_value"])
        out["mean_er_constant_minus_2pp"] = out["mean_er_constant"] + 0.02
        out["mean_er_constant_plus_2pp"] = out["mean_er_constant"] - 0.02
        rows.append(out | {"evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def write_ch13(ctx) -> list[Path]:
    root = ctx.root
    tab, figd = path("tables", root), path("figures", root)
    dep = deposit_scenarios(ctx)
    ab = asset_beta_diagnostics(ctx)
    px, lines = price_export(ctx)
    tables = [
        (roe_reconcile(ctx), "ch13_roe_reconcile.csv", "1. Year-1 base ROE vs reported (definitions)", "Derived"),
        (crp := cr2_profit_direct(root), "ch13_cr2_profit_direct.csv", "2. CR2 profit share vs asset share, "
         "2011-2025, all banking institutions (BoT appendices) and commercial banks (Estimated)", "Derived"),
        (profit_2023_sensitivity(crp), "ch13_cr2_profit_2023_sensitivity.csv", "2. Profit-share conclusions with "
         "FY2023 excluded and interpolated (prompt 24)", "Estimated"),
        (constant_coe_sensitivity(root), "ch13_constant_coe_sensitivity.csv", "5. Excess return against a constant "
         "COE (the central valuation COE) vs each year's COE (prompt 25)", "Estimated"),
        (implied_credit_shares(root), "ch13_implied_credit_shares.csv", "2. Bank share of system credit by "
                                                                        "category (amounts)", "Derived"),
        (earnings_bridge(ctx), "ch13_earnings_bridge.csv", "3. Attributable profit bridge FY2025A -> FY2027E",
         "Scenario"),
        (roe_waterfall(ctx), "ch13_roe_waterfall.csv", "3. Forward ROE waterfall FY2025A -> FY2027E (base)",
         "Scenario"),
        (roe_sustainable_bridge(ctx), "ch13_roe_sustainable_bridge.csv", "3. Reported FY2025 ROE -> sustainable "
                                                                         "ROE* (credit, rate, capital)", "Scenario"),
        (rating_flips(ctx), "ch13_rating_flips.csv", "4. Rating flip per key assumption", "Scenario"),
        (stress_mixture(ctx), "ch13_stress_mixture.csv", "4. P(value > price) vs the weight on a stress regime",
         "Scenario"),
        (rate_consistent(ctx), "ch13_rate_consistent.csv", "4. Rate-consistent valuation (rate path and COE "
                                                           "together)", "Scenario"),
        (named_scenarios(ctx), "ch13_named_scenarios.csv", "4. Named scenario: dominance without excess return",
         "Scenario"),
        (dep["migration"], "ch13_deposit_migration.csv", "5. Deposit migration: CASA repriced to the T-bill",
         "Scenario"),
        (dep["rate_sensitivity"], "ch13_rate_sensitivity.csv", "5. ROE per 1 pp T-bill fall (deposit beta, "
                                                               "falling-rate beta, CASA floor)", "Estimated"),
        (ab["asset_beta"], "ch13_asset_beta.csv", "5b. Asset betas by component: sample, n, method, 90% interval; "
         "lagged pass-through (13b item 1)", "Estimated"),
        (ab["rate_sensitivity_repricing"], "ch13_rate_sensitivity_repricing.csv", "5b. ROE per 1 pp T-bill fall, "
         "years 1-3, bond book repricing at maturity vs the engine (13b item 1)", "Estimated"),
        (ab["repricing_profile"], "ch13_repricing_profile.csv", "5b. IFRS 7 repricing profiles FY2025 (page-checked)",
         "Observed"),
        (dep["casa_floor"], "ch13_casa_floor.csv", "5. CASA cost and its floor", "Derived"),
        (dep["event_2018"], "ch13_event_2018.csv", "5. 2018 bank-closure event test (deposit growth)", "Derived"),
        (valuation_bands(root), "ch13_valuation_bands.csv", "6. Own-history P/B and P/E bands", "Derived"),
        (kpi_table(root), "ch13_kpi_table.csv", "6. Seven-bank KPI table (latest FY)", "Derived"),
        (capacity_windows(root), "ch13_capacity_windows.csv", "7. Capacity: pre-rally vs 2026 windows", "Derived"),
        (block_board(root), "ch13_block_board.csv", "7. Block vs board turnover by quarter", "Derived"),
        (franchise_breakeven(ctx), "ch13_franchise_breakeven.csv", "8. Net deposit franchise break-even T-bill",
         "Estimated"),
        (surplus_drag(ctx), "ch13_surplus_drag.csv", "8. Capital surplus drag at the v1.3 rate path: vs the regulatory "
         "minimum and vs the model target k*",
         "Estimated"),
        (financial_table(ctx), "ch13_financial_table.csv", "10. Financial table FY2024A-FY2028E", "Scenario"),
        (px, "ch13_price_series.csv", "10. 1-year price series (adjusted, post-split)", "Observed"),
        (lines, "ch13_price_lines.csv", "10. 52-week range and rating price lines (v1.3)", "Scenario"),
        (price_monthly(px), "ch13_price_monthly.csv", "10. Monthly close, range, turnover and days traded (report "
         "annex)", "Observed"),
        (usd_exports(ctx), "ch13_usd_exports.csv", "13. Foreign-investor exports", "Scenario"),
    ]
    for f, name, desc, ev in _external(root):
        tables.append((f, name, desc, ev))
    tables += peer_analysis(root, ctx)
    files = [write_csv(ctx.stamp(df), tab / n, require_evidence=True) for df, n, _, _ in tables]
    figs = _figures(root, ctx, tables, figd)
    files += [p for p, _, _ in figs]
    from tzbank.chapters.ch13_exhibits import write_exhibits

    files += write_exhibits(root)  # fill-review pre-flight E: outputs/exhibits/<report>/<exhibit_no>.png
    res = _results(tables)
    nf = write_text(NOTE.format(version=ctx.version, hash=ctx.hash, results=res), path("notes", root) / "ch13.md")
    files.append(nf)
    m = {"chapter": "13", "title": "Report support", "hypotheses": [], "status": "built - report support (prompt 13)",
         "register_version": ctx.version, "input_hash": ctx.hash,
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for _, n, d, c in tables],
         "figures": [{"path": rel(p, root), "data": rel(dd, root), "description": desc, "evidence_class": "Derived",
                      "note": rel(nf, root)} for p, dd, desc in figs],
         "notes": [rel(nf, root)], "exhibits": "outputs/exhibits/index.csv",
         "module": "tzbank.chapters.ch13_report_support", "run_log_stage": "support"}
    files.append(write_json(m, path("manifests", root) / "ch13.json"))
    return files


PEERS = ["KCB Group", "Equity Group Holdings", "Co-operative Bank of Kenya", "Stanbic Uganda Holdings", "BK Group",
         "GTCO", "GCB Bank"]
KPIS = ["roe", "roa", "equity_assets", "npl_ratio", "npl_coverage", "cost_to_income", "net_loan_growth"]
PEER_DATES = {"HIST": ("2025-12-31", 2024), "CURR": ("2026-10-30", 2025)}  # price date, FY published by then (PIT)
DAMODARAN_HIST = "ctryprem24.xlsx"  # January 2025 update: latest held on or before 31 Dec 2025


def _damodaran_xlsx(root: Path, name: str, countries: list[str]) -> dict:
    """Default spread, CRP and the mature-market ERP from a held Damodaran ctryprem workbook ('ERPs by country')."""
    import openpyxl

    wb = openpyxl.load_workbook(path("documents", root) / "damodaran" / name, read_only=True, data_only=True)
    rows = list(wb["ERPs by country"].iter_rows(values_only=True))
    erp = next(r[4] for r in rows[:6] if r and str(r[0]).startswith("Enter the current risk premium for a mature"))
    out = {"erp_mature": float(erp)}
    for r in rows:
        if r and r[0] in countries:
            out[r[0]] = {"default_spread": float(r[3]), "crp": float(r[5])}
    return out


def _peer_coe(root: Path, p: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """Country COE by approach A (VALUATION_SPEC 5): (10-year local yield - default spread) + beta (ERP + CRP) + LP, with
    the Tanzanian banks' register beta and LP at each date (common for comparability) and the latest Damodaran
    vintage held on or before the date."""
    from tzbank.valuation.coe import approach_a

    ctry = ["Kenya", "Uganda", "Rwanda", "Nigeria", "Ghana"]
    rows = []
    for val, (d, _) in PEER_DATES.items():
        r = reg[reg["valuation"] == val]
        beta = float(r[r["symbol"] == "beta"]["base"].astype(float).mean())
        lp = float(r[r["symbol"] == "LP"]["base"].astype(float).mean())
        if val == "HIST":
            dm = _damodaran_xlsx(root, DAMODARAN_HIST, ctry)
            src = {c: f"Damodaran ctryprem January 2025 ({DAMODARAN_HIST}, held)" for c in ctry}
            erp = dm["erp_mature"]
        else:
            dm, src = {}, {}
            for c in ctry:
                g = p[(p["peer"] == c) & p["variable"].isin(["default_spread", "crp"])].copy()
                g["d"] = pd.to_datetime(g["period"].astype(str).str[:10], errors="coerce")
                g = g[g["d"].notna() & (g["d"] <= pd.Timestamp(d))].sort_values("d")
                if g["variable"].nunique() == 2:
                    last = g.groupby("variable").tail(1).set_index("variable")
                    dm[c] = {"default_spread": float(last.loc["default_spread", "value"]),
                             "crp": float(last.loc["crp", "value"])}
                    src[c] = f"Damodaran update dated {last['d'].max().date()} (peers_collected.csv)"
            e = p[p["variable"] == "erp_mature"].copy()
            e["d"] = pd.to_datetime(e["period"].astype(str).str[:10], errors="coerce")
            e = e[e["d"].notna() & (e["d"] <= pd.Timestamp(d))].sort_values("d")
            erp = float(e["value"].iloc[-1]) if len(e) else np.nan
        for c in ctry:
            y = p[(p["peer"] == c) & (p["variable"] == "bond_10y") & (p["period"].astype(str) == d)]
            yv = float(y["value"].iloc[0]) if len(y) else np.nan
            ds, crp = (dm[c]["default_spread"], dm[c]["crp"]) if c in dm else (np.nan, np.nan)
            coe = approach_a(yv, ds, beta, erp, crp, lp) if np.all(np.isfinite([yv, ds, crp, erp])) else np.nan
            rows.append({"country": c, "valuation": val, "date": d, "bond_10y": yv, "default_spread": ds, "crp": crp,
                         "erp_mature": erp, "beta": beta, "lp": lp, "coe_approach_a": coe,
                         "damodaran_source": src.get(c, "not held"),
                         "status": "computed" if np.isfinite(coe) else
                         "not computable: 10-year local bond yield at the date not held" if not np.isfinite(yv)
                         else "not computable: Damodaran inputs not held", "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy import stats

    n = len(x)
    if n < 4:
        return {"n": n, "status": f"not run: n = {n} < 4"}
    r = stats.linregress(x, y)
    t = stats.t.ppf(0.95, n - 2)
    return {"n": n, "intercept": r.intercept, "slope": r.slope, "slope_lo": r.slope - t * r.stderr,
            "slope_hi": r.slope + t * r.stderr, "r2": r.rvalue ** 2, "status": "estimated"}


def peer_analysis(root: Path, ctx=None) -> list:
    """Item 11 / 13b item 4 on the collected peer panel: latest-FY KPI table; ROE persistence (AR(1) on ROE by the
    chapter 5 method, >= 6 pairs; equals the excess-return lambda when the COE is constant); market multiples at both
    valuation dates on point-in-time per-share data (price at the date; EPS, DPS and ROE of the last FY published by
    then: FY2024 at 31 Dec 2025, FY2025 at 30 Oct 2026; BVPS = EPS / ROE on average equity, Derived); country COE by
    approach A and excess ROE; the P/B-ROE regression with NMB and CRDB at each date; shortfalls per bank."""
    f = path("raw", root) / "manual_entries" / "peers_collected.csv"
    if not f.exists():
        return []
    p = pd.read_csv(f)
    reg = pd.read_csv(root / "config" / "assumption_register.csv")
    fy = p[p["period"].astype(str).str.match(r"FY\d{4}$")].copy()
    fy["fiscal_year"] = fy["period"].str[2:].astype(int)
    kpi_rows, pers = [], []
    for peer in PEERS:
        g = fy[fy["peer"] == peer]
        if len(g):
            y = int(g["fiscal_year"].max())
            last = g[g["fiscal_year"] == y].drop_duplicates("variable").set_index("variable")["value"]
            kpi_rows.append({"peer": peer, "country": g["country"].iloc[0], "fiscal_year": y,
                             **{k: last.get(k, np.nan) for k in KPIS}, "evidence_class": "Observed"})
        roe = g[g["variable"] == "roe"].drop_duplicates("fiscal_year").set_index("fiscal_year")["value"].sort_index()
        pairs = [(roe[t - 1], roe[t]) for t in roe.index if t - 1 in roe.index]
        if len(pairs) >= 6:
            x, yv = np.array(pairs).T
            lam, a = np.polyfit(x, yv, 1)
            pers.append({"peer": peer, "years": f"FY{roe.index.min()}-{roe.index.max()}", "pairs": len(pairs),
                         "lambda_roe": lam, "long_run_roe": a / (1 - lam) if lam < 1 else np.nan,
                         "status": "estimated (OLS AR(1) on ROE; ER lambda if COE constant)",
                         "evidence_class": "Estimated"})
        else:
            pers.append({"peer": peer, "years": f"FY{roe.index.min()}-{roe.index.max()}" if len(roe) else "none",
                         "pairs": len(pairs), "lambda_roe": np.nan, "long_run_roe": np.nan,
                         "status": f"not estimated: {len(pairs)} pairs < 6 (chapter 5 minimum)",
                         "evidence_class": "Estimated"})
    coe = _peer_coe(root, p, reg)
    # market multiples at both dates
    mk = []
    for val, (d, fyr) in PEER_DATES.items():
        for peer in PEERS:
            g = fy[(fy["peer"] == peer) & (fy["fiscal_year"] == fyr)].drop_duplicates("variable").set_index(
                "variable")["value"]
            pr = p[(p["peer"] == peer) & (p["variable"] == "price") & (p["period"].astype(str) == d)]
            price = float(pr["value"].iloc[0]) if len(pr) else np.nan
            eu = fy[(fy["peer"] == peer) & (fy["variable"] == "eps")]["unit"]
            pcy, ecy = (str(pr["unit"].iloc[0]).split()[0] if len(pr) else ""), (str(eu.iloc[0]).split()[0] if len(eu)
                                                                                  else "")
            ccy_ok = bool(pcy) and pcy == ecy
            if not ccy_ok:  # e.g. BK Group: NSE secondary-listing price (KES) vs per-share data in RWF; not converted
                price = np.nan
            eps, dps, roe = (float(g.get(k, np.nan)) for k in ("eps", "dps", "roe"))
            bvps = eps / roe if np.isfinite(eps) and np.isfinite(roe) and roe > 0 else np.nan
            ctry = fy[fy["peer"] == peer]["country"].iloc[0]
            c = coe[(coe["country"] == ctry) & (coe["valuation"] == val)]
            k = float(c["coe_approach_a"].iloc[0]) if len(c) else np.nan
            mk.append({"bank": peer, "country": ctry, "valuation": val, "price_date": d, "fy_basis": f"FY{fyr}",
                       "price": price, "eps": eps, "dps": dps, "roe": roe, "bvps": bvps,
                       "pe": price / eps if eps > 0 else np.nan, "pb": price / bvps if bvps > 0 else np.nan,
                       "dividend_yield": dps / price if np.isfinite(price) and price > 0 else np.nan,
                       "coe_approach_a": k, "excess_roe": roe - k,
                       "pb_basis": "price / (EPS / ROE): average-equity book (Derived)" if ccy_ok else
                       f"not computed: price held in {pcy or 'n/a'} (secondary listing), per-share data in {ecy or 'n/a'}",
                       "evidence_class": "Derived"})
        if ctx is not None:
            w = _wide(root).set_index(["bank_id", "fiscal_year"])
            d4 = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv", dtype={"bank_id": str}).set_index(
                ["bank_id", "fiscal_year"])
            for b in FOCUS:
                cs = ctx.cases[(val, b, "base")]
                mc = cs.price * cs.shares
                a = w.loc[(b, fyr)]
                roe = float(d4.loc[(b, fyr), "roe"])
                mk.append({"bank": T[b], "country": "Tanzania", "valuation": val, "price_date": str(cs.vd.date()),
                           "fy_basis": f"FY{fyr}", "price": cs.price, "eps": float(a["profit_attributable"]) / cs.shares,
                           "dps": np.nan, "roe": roe, "bvps": float(a["equity_attributable"]) / cs.shares,
                           "pe": mc / float(a["profit_attributable"]), "pb": mc / float(a["equity_attributable"]),
                           "dividend_yield": np.nan, "coe_approach_a": cs.r, "excess_roe": roe - cs.r,
                           "pb_basis": "market cap / year-end attributable equity; COE = register (31 Dec 2025 / "
                                       "31 Oct 2026)", "evidence_class": "Derived"})
    mk = pd.DataFrame(mk)
    # P/B-ROE regression at each date (peers with P/B, plus NMB and CRDB)
    rg = []
    for val, g in mk.groupby("valuation", sort=False):
        for lab, xcol in (("P/B on ROE", "roe"), ("P/B on excess ROE (ROE - COE, approach A)", "excess_roe")):
            h = g.dropna(subset=["pb", xcol])
            o = _ols(h[xcol].to_numpy(float), h["pb"].to_numpy(float))
            rg.append({"valuation": val, "model": lab, "banks": ", ".join(h["bank"]), **o, "evidence_class": "Estimated"})
            if o.get("status") == "estimated":
                for r in h[h["country"] == "Tanzania"].itertuples():
                    fit = o["intercept"] + o["slope"] * getattr(r, xcol)
                    rg.append({"valuation": val, "model": lab, "banks": r.bank, "n": o["n"], "fitted_pb": fit,
                               "actual_pb": r.pb, "residual_pb": r.pb - fit, "status": "residual",
                               "evidence_class": "Estimated"})
    # shortfalls per bank
    short = []
    for peer in PEERS:
        g = fy[fy["peer"] == peer]
        held = {(v, int(t)) for v, t in zip(g["variable"], g["fiscal_year"], strict=True)}
        miss_kpi = sum((k, t) not in held for k in KPIS for t in range(2012, 2026))
        m = mk[mk["bank"] == peer]
        miss_mkt = [f"{c} {r.valuation}" for r in m.itertuples() for c in ("price", "pb", "pe", "dividend_yield")
                    if not np.isfinite(getattr(r, c))]
        ctry = g["country"].iloc[0] if len(g) else ""
        cs = coe[coe["country"] == ctry]
        short.append({"peer": peer, "kpi_cells_requested": len(KPIS) * 14, "kpi_cells_held": len(KPIS) * 14 - miss_kpi,
                      "years_held": f"FY{g['fiscal_year'].min()}-{g['fiscal_year'].max()}" if len(g) else "none",
                      "market_items_missing": "; ".join(miss_mkt) or "none",
                      "country_coe_approach_a": "; ".join(f"{r.valuation}: {r.status}" for r in cs.itertuples()),
                      "pb_basis": "Derived (EPS / ROE book)", "evidence_class": "Observed"})
    return [(pd.DataFrame(kpi_rows), "ch13_peer_kpi.csv", "11. African peer KPIs (latest FY held)", "Observed"),
            (pd.DataFrame(pers), "ch13_peer_persistence.csv", "11. Peer ROE persistence (chapter 5 method)",
             "Estimated"),
            (mk, "ch13_peer_market.csv", "11. Peer and NMB / CRDB multiples at both dates (PIT), COE approach A, "
                                         "excess ROE", "Derived"),
            (coe, "ch13_peer_coe.csv", "11. Country COE by approach A at both dates", "Estimated"),
            (pd.DataFrame(short), "ch13_peer_shortfalls.csv", "11. Peer data shortfalls per bank", "Observed"),
            (pd.DataFrame(rg), "ch13_peer_pb_roe_regression.csv", "11. P/B-ROE regression with NMB and CRDB, both "
                                                                  "dates", "Estimated")]


def _external(root: Path) -> list:
    """Peer panel and benchmark tables (items 11-12) when their collected files exist."""
    out = []
    for name, desc, ev in (("peers_collected.csv", "11. African peer panel (collected, dated sources)", "Observed"),
                           ("benchmarks_collected.csv", "12. Benchmark figures (collected, dated sources)",
                            "Observed")):
        f = path("raw", root) / "manual_entries" / name
        if f.exists():
            d = pd.read_csv(f)
            if "evidence_class" not in d:
                d["evidence_class"] = ev
            ver = path("raw", root) / "manual_entries" / "benchmarks_verified.csv"
            if name == "benchmarks_collected.csv" and ver.exists():  # prompt 24: figures confirmed on the page
                for v in pd.read_csv(ver).itertuples():
                    m = ((d["document"] == v.document) & (d["metric"] == v.metric) & (d["scope"] == v.scope)
                         & (d["period"].astype(str) == str(v.period)))
                    d.loc[m, "confidence"] = v.confidence
                    d.loc[m, "source_doc"] = v.source_doc
                    d.loc[m, "page_or_section"] = f"p. {v.printed_page} (PDF p. {v.pdf_page})"
                    d.loc[m, "evidence_class"] = v.evidence_class
            out.append((d, f"ch13_{name.replace('_collected', '')}", desc, ev))
    return out


def _dsei(root: Path) -> pd.DataFrame:
    """DSE All Share Index daily levels from the DSE data library (one year as published; Observed)."""
    f = root / "data" / "library" / "dse" / "index_history.csv"
    if not f.exists():
        return pd.DataFrame()
    h = pd.read_csv(f, parse_dates=["trading_date"])
    return h[h["index"] == "DSEI"].sort_values("trading_date")[["trading_date", "level"]]


def _figures(root: Path, ctx, tables, figd: Path) -> list:
    figs = []
    t = {n: df for df, n, _, _ in tables}
    # P/B-ROE with the justified P/B line (anchor)
    k = t["ch13_kpi_table.csv"].dropna(subset=["pb", "roe"]).assign(point="latest fiscal year-end")
    w = _wide(root)
    cur = []
    for b in FOCUS:  # 31 Oct 2026 price on the same (FY2025) book and ROE: isolates the price move since year end
        c = ctx.cases[("CURR", b, "base")]
        x = k[k["bank"] == T[b]].iloc[0]
        eq = w[(w["bank_id"] == b) & (w["fiscal_year"] == x["fiscal_year"])]["equity_attributable"].iloc[0]
        cur.append({"bank": T[b], "fiscal_year": x["fiscal_year"], "roe": x["roe"], "pb": c.price * c.shares / eq,
                    "point": "price at 31 Oct 2026 on the FY2025 book", "evidence_class": "Derived"})
    k = pd.concat([k, pd.DataFrame(cur)], ignore_index=True)
    fig, ax = plt_subplots(figsize=(7.5, 4.4))
    inv = {v: kk for kk, v in T.items()}
    for r in k.itertuples():
        col = COLORS.get(inv[r.bank], "#9a9a94")
        hollow = r.point.startswith("price at")
        ax.scatter(r.roe * 100, r.pb, s=34, facecolors="none" if hollow else col, edgecolors=col, zorder=3)
        left = r.bank == "NMB"  # NMB labels to the left so they do not run into CRDB's (close ROE)
        ax.annotate(f"{r.bank} FY{int(r.fiscal_year)}" if not hollow else f"{r.bank} 31 Oct 2026", (r.roe * 100, r.pb),
                    fontsize=6.5, xytext=(-5 if left else 5, 0), textcoords="offset points", va="center",
                    ha="right" if left else "left")
    for b in FOCUS:
        g = k[k["bank"] == T[b]]
        ax.annotate("", xy=(g["roe"].iloc[1] * 100, g["pb"].iloc[1]), xytext=(g["roe"].iloc[0] * 100, g["pb"].iloc[0]),
                    arrowprops={"arrowstyle": "->", "color": COLORS[b], "lw": 0.8})
    c = ctx.cases[("CURR", "B02", "base")]
    xs = np.linspace(max(c.dr.g + 0.005, 0.0), 0.40, 50)
    ax.plot(xs * 100, (xs - c.dr.g) / (c.r - c.dr.g), color="#1f1f1d", lw=1,
            label=f"justified P/B = (ROE - g) / (COE - g); COE {c.r:.1%}, g {c.dr.g:.1%} (31 Oct 2026)")
    ax.axvline(0, color="#8a8a85", lw=0.6)
    n = k.loc[k["point"] == "latest fiscal year-end", "bank"].nunique()
    style(ax, f"P/B against ROE, {n} listed Tanzanian banks (latest year-end; NMB, CRDB also 31 Oct 2026)", "P/B")
    ax.title.set_fontsize(9)
    ax.set_xlabel("ROE, percent (latest fiscal year)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    p = save(fig, figd / "ch13_pb_roe_anchor.png", "Evidence class: Derived. Filled: year-end market cap (price x "
             "shares, or the reported market cap for the smaller banks) / equity; hollow: 31 Oct 2026 price on the "
             "FY2025 book. Line: justified P/B at the v1.3 COE and g.")
    dd = write_csv(k[["bank", "fiscal_year", "point", "roe", "pb", "evidence_class"]],
                   figd / "ch13_pb_roe_anchor_data.csv", require_evidence=True)
    figs.append((p, dd, "P/B against ROE with the justified P/B anchor"))
    # stress mixture
    s = t["ch13_stress_mixture.csv"]
    fig, ax = plt_subplots(figsize=(7.5, 4.0))
    for (val, bank), g in s.groupby(["valuation", "bank"]):
        bid = {v: kk for kk, v in T.items()}[bank]
        ax.plot(g["stress_weight"], g["prob_value_above_price"] * 100, color=COLORS[bid],
                ls="-" if val == "CURR" else "--",
                label=f"{bank}, valuation at {pd.Timestamp(rg.DATES[val]):%d %b %Y}")
    style(ax, "P(value > price) against the weight on a 2015-18 stress regime", "percent")
    ax.set_xlabel("weight on the stress regime", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    p = save(fig, figd / "ch13_stress_mixture.png", "Evidence class: Scenario. Mixture of the base-centred Monte Carlo "
             "and a bear-case Monte Carlo.")
    dd = write_csv(s[["valuation", "bank", "stress_weight", "prob_value_above_price", "evidence_class"]],
                   figd / "ch13_stress_mixture_data.csv", require_evidence=True)
    figs.append((p, dd, "Stress-regime mixture"))
    # price chart with lines
    px, ln = t["ch13_price_series.csv"], t["ch13_price_lines.csv"]
    idx = _dsei(root)
    style_of = {"52-week high": ("--", "#9a9a94"), "52-week low": ("--", "#9a9a94"),
                "Buy below (expected return": ("-", "#1baf7a"), "Buy below (bear-case": ("-.", "#1baf7a"),
                "Sell above": ("-", "#d1453b"), "value per share": (":", "#1f1f1d")}
    fig, axes = plt_subplots(1, 2, figsize=(11, 4.4))
    for ax, b in zip(axes, FOCUS, strict=True):
        g = px[px["bank"] == T[b]].sort_values("date")
        dts = pd.to_datetime(g["date"])
        ax.plot(dts, g["adj_close"], color=COLORS[b], lw=1.2, label=f"{T[b]} close")
        if len(idx):
            i0 = idx[(idx["trading_date"] >= dts.min()) & (idx["trading_date"] <= dts.max())]
            if len(i0):
                first = g.set_index(dts)["adj_close"].asof(i0["trading_date"].iloc[0])
                ax.plot(i0["trading_date"], i0["level"] / i0["level"].iloc[0] * first, color="#5a5a55", lw=0.8,
                        alpha=0.8, label=f"DSEI, rebased to {T[b]} on {i0['trading_date'].iloc[0]:%d %b %Y}")
        lines = ln[(ln["bank"] == T[b]) & ~ln["line"].str.startswith("price at")].sort_values("price")
        ymin = min(lines["price"].min(), g["adj_close"].min())
        ymax = max(lines["price"].max(), g["adj_close"].max())
        gap = 0.045 * (ymax - ymin)
        last = -np.inf
        for r in lines.itertuples():
            ls, c = next((v for k_, v in style_of.items() if r.line.startswith(k_)), (":", "#1f1f1d"))
            ax.axhline(r.price, lw=0.8, ls=ls, color=c)
            ylab = max(r.price, last + gap)  # labels never overlap: pushed up when two lines are close
            last = ylab
            tag = " (bear screen)" if "bear" in r.line else " (return test)" if r.line.startswith("Buy") else ""
            ax.annotate(f"{r.line.split(' (')[0]}{tag}: {r.price:,.0f}",
                        xy=(1.0, r.price), xycoords=("axes fraction", "data"), xytext=(4, (ylab - r.price) / gap * 9),
                        textcoords="offset points", fontsize=6, va="center", color=c,
                        arrowprops={"arrowstyle": "-", "color": c, "lw": 0.4} if ylab != r.price else None)
        ax.set_ylim(ymin - 2 * gap, ymax + 3 * gap)
        style(ax, f"{T[b]}: 1-year price and rating lines (TZS, post-split basis)", "TZS")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    p = save(fig, figd / "ch13_price_lines.png", "Evidence class: Observed prices and DSEI (DSE indices page, data/"
             "library/dse); Scenario lines (v1.3 value path, D4 thresholds). Two Buy lines: the return test (expected "
             "return > COE + 5 pp) and the bear screen (bear-case downside within 25%); a Buy needs both.")
    dd = write_csv(px.assign(evidence_class="Observed"), figd / "ch13_price_lines_data.csv", require_evidence=True)
    figs.append((p, dd, "1-year price with the 52-week range and rating price lines"))
    del root
    return figs


def _results(tables) -> str:
    out = []
    t = {n: df for df, n, _, _ in tables}
    r = t["ch13_roe_reconcile.csv"]
    out.append("**1. Year-1 ROE reconciliation.** " + "; ".join(
        f"{x.bank} {x.valuation} FY{x.forecast_year}: engine {x.roe_engine_opening_ex_div:.1%} (opening equity ex "
        f"dividend) vs reported FY{x.reported_roe_anchor_fy} {x.reported_roe_anchor:.1%}; on the reported basis "
        f"(average equity cum dividend) {x.roe_reported_basis:.1%}: definition effect "
        f"{x.definition_effect:+.1%}, like-for-like change {x.like_for_like_change:+.1%}" for x in r.itertuples())
        + ". The engine's year-1 ROE is not comparable with the reported ROE until restated; the remaining step is a "
          "forecast fall in ROE as equity grows faster than profit.")
    c = t["ch13_cr2_profit_direct.csv"]
    d = c[c["sector_basis"].str.startswith("all banking") & c["cr2_pbt_share"].notna()
          & c["numerator_basis"].str.startswith("NMB + CRDB Group")]
    k = c[c["sector_basis"].str.startswith("commercial")]
    k24 = k[k["year"] == 2024]
    out.append("**2. CR2 profit share (one basis: all banking institutions, BoT supervision report appendices, calendar "
               "year, latest vintage per year; numerator NMB + CRDB Group).** " + "; ".join(
                   f"{int(x.year)} assets {x.cr2_asset_share:.1%}, PBT {x.cr2_pbt_share:.1%} (gap "
                   f"{x.pbt_minus_asset_share:+.1%})" for x in d.itertuples())
               + ". 2023: sector profit not disclosed in usable form (the BoT appendix prints 2023 profit before tax "
                 "of TZS 9,272m in the 2023, 2024 and 2025 reports, inconsistent with its own 2023 ROA of 4.4%). The "
                 "2025 sector total is published (BoT Banking Supervision Annual Report 2025, 2 July 2026, full-year "
                 "audited appendix), not partial. The two 2024 rows in the v1.1 table differ in denominator and "
                 "method: the BSR row divides NMB + CRDB Group profit before tax by the all-institution sector total "
                 "(TZS 3,013,327m)" + (f" -> {d[d['year'] == 2024]['cr2_pbt_share'].iloc[0]:.1%}" if (d["year"] == 2024)
                                       .any() else "") + "; the ch01 row is commercial banks only, Estimated as the CR2 "
                 "pre-tax ROA (average assets) over the BoT sector ROA (end-June fiscal year, tax basis not stated) "
                 "times the commercial-bank asset share" + (f" -> {k24['cr2_pbt_share'].iloc[0]:.1%}" if len(k24) else "")
               + ". The Group numerator includes CRDB's Burundi and DRC subsidiaries (outside the Tanzanian sector "
                 "total), so both shares are upper bounds; a Bank-solo numerator is held for too few years.")
    f = t["ch13_rating_flips.csv"]
    ff = f[f["shift_to_flip"].notna()].copy()
    ff["abs"] = ff["shift_to_flip"].abs()
    ff = ff.sort_values("abs").groupby(["valuation", "bank"]).head(1)
    out.append("**4. Smallest rating flip.** " + "; ".join(
        f"{x.bank} {x.valuation}: {x.assumption} {x.direction} by {x.shift_to_flip:+.4f} -> {x.flipped_rating}"
        for x in ff.itertuples()))
    n = t["ch13_named_scenarios.csv"]
    out.append("**4. Dominance without excess return** (ROE* = COE): " + "; ".join(
        f"{x.bank} {x.valuation} {x.value_per_share:,.0f} vs price {x.price:,.0f} ({x.rating})" for x in n.itertuples()))
    s = t["ch13_rate_sensitivity.csv"]
    out.append("**5. ROE per 1 pp T-bill fall.** " + "; ".join(
        f"{x.bank} {x.case}: {x.roe_change_per_1pp_tbill_fall:.1%}" for x in s.itertuples()))
    ab = t["ch13_asset_beta.csv"]
    ok = ab[(ab["status"] == "estimated") & ab["method"].str.startswith("OLS")]
    na = ab[(ab["status"] != "estimated") & ab["method"].str.startswith("OLS")]
    dl = ab[(ab["status"] == "estimated") & ab["method"].str.startswith("distributed")]
    out.append("**5b. Asset betas (annual changes on the T-bill, FY <= 2025, 90% intervals).** " + "; ".join(
        f"{x.bank} {x.component}: {x.beta:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}], {x.years}, n = {x.n}"
        for x in ok.itertuples()) + ". Not estimated (balances held FY2020+ only): " + "; ".join(
        f"{x.bank} {x.component} (n = {x.n})" for x in na.itertuples()) + ". Lagged pass-through, cumulative after "
        "1 / 2 / 3 years: " + "; ".join(
        f"{bk} {cp}: " + " / ".join(f"{v:+.2f}" for v in g.sort_values("horizon_years")["beta"])
        for (bk, cp), g in dl.groupby(["bank", "component"], sort=False)) + ".")
    rs = t["ch13_rate_sensitivity_repricing.csv"]
    out.append("**5b. ROE per 1 pp sustained T-bill fall, securities and interbank repricing at contractual "
               "repricing / maturity (FY2025 IFRS 7 profiles), loans at the lagged loan beta.** " + "; ".join(
                   f"{bk}: " + ", ".join(f"year {int(x.year)} {x.roe_change_pts * 100:+.2f} pts" for x in
                                         g[g["case"].str.startswith("repricing")].itertuples())
                   + "; engine " + ", ".join(f"{x.roe_change_pts * 100:+.2f} pts" for x in
                                             g[g["case"].str.startswith("engine")].itertuples())
                   for bk, g in rs.groupby("bank", sort=False)) + ". " + ENGINE_NOTE)
    b = t["ch13_franchise_breakeven.csv"]
    out.append("**8. Franchise break-even T-bill.** " + "; ".join(
        f"{x.bank} {x.tbill_case}: break-even {x.breakeven_tbill:.2%}, T-bill {x.tbill:.2%}, net franchise "
        f"{x.net_franchise_roe_pts:.1%} of ROE" for x in b.itertuples()))
    sd = t["ch13_surplus_drag.csv"]
    out.append("**8. Capital surplus drag** (surplus Tier 1 earning the after-tax T-bill instead of the COE; binding "
               "basis, NMB Bank-solo). " + "; ".join(
                   f"{x.bank} FY{x.fiscal_year} vs {x.hurdle} ({x.hurdle_ratio:.2%}; Tier 1 {x.tier1_ratio:.2%}): "
                   f"{x.drag_share_of_profit:.1%} of profit" for x in sd.itertuples()))
    if "ch13_peer_market.csv" in t:
        pm, pc, pr = t["ch13_peer_market.csv"], t["ch13_peer_coe.csv"], t["ch13_peer_pb_roe_regression.csv"]
        out.append("**11. Country COE, approach A** ((10-year local yield - default spread) + beta (ERP + CRP) + LP, "
                   "register beta and LP): " + "; ".join(
                       f"{x.country} {x.valuation} {x.coe_approach_a:.1%}" if np.isfinite(x.coe_approach_a) else
                       f"{x.country} {x.valuation} not computable" for x in pc.itertuples()) + ".")
        out.append("**11. Multiples at the dates (PIT: FY2024 at 31 Dec 2025, FY2025 at 30 Oct 2026; peer P/B on "
                   "EPS / ROE book, Derived).** " + "; ".join(
                       f"{x.bank} {x.valuation}: P/B {x.pb:.2f}, P/E {x.pe:.1f}, ROE {x.roe:.1%}"
                       + (f", excess ROE {x.excess_roe:+.1%}" if np.isfinite(x.excess_roe) else "")
                       for x in pm.itertuples() if np.isfinite(x.pb)) + ".")
        fits = pr[pr["status"] == "estimated"]
        res = pr[pr["status"] == "residual"]
        out.append("**11. P/B-ROE regression (peers with P/B, plus NMB and CRDB).** " + "; ".join(
            f"{x.valuation} {x.model}: slope {x.slope:+.2f} [{x.slope_lo:+.2f}, {x.slope_hi:+.2f}], R2 {x.r2:.2f}, "
            f"n = {int(x.n)}" for x in fits.itertuples()) + ". Residual P/B (actual - fitted): " + "; ".join(
            f"{x.banks} {x.valuation} {x.model.split(' (')[0]}: {x.residual_pb:+.2f}" for x in res.itertuples())
            + ". The slopes' intervals include zero at both dates: across these markets P/B is not explained by ROE "
              "alone (n = 7-8, mixed currencies and COEs).")
    return "\n\n".join(out)
