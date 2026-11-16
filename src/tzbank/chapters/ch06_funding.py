"""Chapter 6: funding architecture (blueprint §2 deposit franchise, §4 chapter 6; H6.1, H6.2; H5.3 revisited; D12).

- Cost of deposits = deposit interest / average deposits (by type where the split is held); deposit mix.
- Deposit beta: OLS of the annual change in the cost of deposits on the change in (a) the annual-average overall
  T-bill rate and (b) the December policy rate (D42), per bank, 90% t interval; asymmetric betas (rising vs falling
  rate years) where each side has >= 4 years. Interims hold no deposit interest, so the data are annual. Rates are
  dated inside the fiscal year (calendar year for every bank).
- Gross deposit funding advantage (GDFA, a **proxy**, D12) = (T-bill - cost of deposits) x average deposits.
- Net deposit franchise contribution = GDFA - reserve cost (SMR x average deposits x T-bill; reserves are
  unremunerated) - deposit-gathering OpEx (share s of OpEx excluding deposit insurance; central s = 50%, band
  25-75%, Scenario) - deposit insurance (printed expense line where held, else the mean printed rate x average
  deposits, Estimated, band +/-50%). Shown as % of average equity (pre-tax) and as ROE points (after tax, over
  average attributable equity), each with its band. Reconciled with the 4.0 funding term.
- Liquidity: loan-to-deposit ratio, wholesale (borrowings) share, funding gap and its cost against the 4.0 funding
  contribution.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tzbank.chapters.ch04.common import COLORS, FOCUS, avg, col, div, lag, pct, plt_subplots, save, style, tickers, wide
from tzbank.chapters.ch04.funding import TBILL
from tzbank.config import find_root, path
from tzbank.io import rel, write_csv, write_json, write_text

OPEX_SHARES = {"low": 0.75, "central": 0.50, "high": 0.25}  # deduction share; "high" net = smallest deduction
DI_BAND = 0.5
LEVEL = 0.90
MIN_SIDE = 4


# --- rates ---------------------------------------------------------------------------------------------------------


def tbill_annual(root: Path) -> pd.DataFrame:
    """Annual mean of the quarter-end overall T-bill rate, with the dates used (all inside the calendar year)."""
    q = pd.read_csv(path("processed", root) / "rates_quarterly.csv")
    q = q[q["series_id"] == TBILL].copy()
    q["period_end"] = pd.to_datetime(q["period_end"])
    # D64 (owner issue 18): years the BoT database does not hold take the collected auction results: the 364-day
    # yield (proxy for the overall rate, within ~0.1 pp 2019-2025) at the last auction on or before each quarter-end.
    from tzbank.ingest.web_collected import load

    w = load(root)
    w = w[w["canonical_name"] == "tbill_364d_way"].copy()
    if len(w):
        w["date"] = pd.to_datetime(w["as_at"])
        held = set(q["period_end"].dt.year)
        add = []
        for qe in pd.date_range(f"{w['date'].dt.year.min()}-03-31", w["date"].max(), freq="QE"):
            if qe.year in held:
                continue
            x = w[(w["date"] <= qe) & (w["date"] > qe - pd.DateOffset(months=3))]
            if len(x):
                add.append({"period_end": qe, "value": float(x.sort_values("date")["value"].iloc[-1])})
        if add:
            q = pd.concat([q, pd.DataFrame(add)], ignore_index=True)
    g = q.groupby(q["period_end"].dt.year.rename("year"))
    return pd.DataFrame({"tbill": g["value"].mean(), "tbill_n_quarters": g["value"].count(),
                         "tbill_first_date": g["period_end"].min(), "tbill_last_date": g["period_end"].max()})


def policy_annual(root: Path) -> pd.DataFrame:
    """December policy rate (D42: discount rate to 2023, CBR from 2024), dated inside the year."""
    r = pd.read_csv(path("processed", root) / "rates_annual.csv")
    r = r[r["series_id"] == "POLICY_RATE_D42"].copy()
    r["period_end"] = pd.to_datetime(r["period_end"])
    r = r[r["period_end"].dt.month == 12].sort_values("period_end").drop_duplicates("year", keep="last")
    return r.set_index("year")[["value", "period_end"]].rename(columns={"value": "policy_rate",
                                                                        "period_end": "policy_date"})


def smr_events(root: Path) -> pd.DataFrame:
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    e = p[(p["kind"] == "SMR") & (p["scope"] == "private deposit liabilities")].copy()
    e["event_date"] = pd.to_datetime(e["event_date"])
    return e.sort_values("event_date")


def smr_annual(events: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Time-weighted calendar-year average and year-end SMR on private deposit liabilities."""
    rows = []
    for y in years:
        start, end = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
        before = events[events["event_date"] < start]
        if before.empty:
            rows.append({"year": y, "smr_avg": np.nan, "smr_year_end": np.nan, "smr_events_in_year": 0})
            continue
        level, t, acc = before["value"].iloc[-1], start, 0.0
        inside = events[(events["event_date"] >= start) & (events["event_date"] < end)]
        for _, ev in inside.iterrows():
            acc += level * (ev["event_date"] - t).days
            level, t = ev["value"], ev["event_date"]
        acc += level * (end - t).days
        rows.append({"year": y, "smr_avg": acc / (end - start).days, "smr_year_end": level,
                     "smr_events_in_year": len(inside)})
    return pd.DataFrame(rows).set_index("year")


# --- deposits, beta -------------------------------------------------------------------------------------------------


def deposits(w: pd.DataFrame) -> pd.DataFrame:
    dep = avg(w, "customer_deposits")
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"], "avg_deposits": dep,
                        "cost_of_deposits": div(col(w, "interest_expense_deposits").abs(), dep)})
    for k in ("current", "savings", "time"):
        out[f"cost_{k}"] = div(col(w, f"interest_expense_deposits_{k}").abs(), avg(w, f"deposits_{k}"))
        out[f"share_{k}"] = div(col(w, f"deposits_{k}"), col(w, "customer_deposits"))
    out["casa_share"] = out["share_current"] + out["share_savings"]
    out["deposit_growth"] = div(col(w, "customer_deposits"), lag(w, col(w, "customer_deposits"))) - 1
    out["evidence_class"] = "Derived"
    return out


def beta_ols(dc: np.ndarray, dr: np.ndarray) -> dict:
    n = len(dc)
    if n < 5:
        return {"n": n, "beta": np.nan, "lo": np.nan, "hi": np.nan, "status": f"not estimated: {n} years < 5"}
    r = stats.linregress(dr, dc)
    t = stats.t.ppf(1 - (1 - LEVEL) / 2, n - 2)
    return {"n": n, "beta": r.slope, "lo": r.slope - t * r.stderr, "hi": r.slope + t * r.stderr,
            "r2": r.rvalue ** 2, "status": "estimated"}


def asym_beta(dc: np.ndarray, dr: np.ndarray) -> dict:
    up, dn = dr > 0, dr < 0
    if up.sum() < MIN_SIDE or dn.sum() < MIN_SIDE:
        return {"status": f"not estimated: {int(up.sum())} rising and {int(dn.sum())} falling years (need "
                          f">= {MIN_SIDE} each)"}
    X = np.column_stack([np.ones(len(dr)), np.where(up, dr, 0), np.where(dn, dr, 0)])
    b, *_ = np.linalg.lstsq(X, dc, rcond=None)
    e = dc - X @ b
    df = len(dc) - 3
    cov = (e @ e / df) * np.linalg.inv(X.T @ X)
    t = stats.t.ppf(1 - (1 - LEVEL) / 2, df)
    se = np.sqrt(np.diag(cov))
    return {"beta_up": b[1], "beta_up_lo": b[1] - t * se[1], "beta_up_hi": b[1] + t * se[1], "beta_down": b[2],
            "beta_down_lo": b[2] - t * se[2], "beta_down_hi": b[2] + t * se[2], "n_up": int(up.sum()),
            "n_down": int(dn.sum()), "status": "estimated"}


def deposit_beta(dep: pd.DataFrame, tb: pd.DataFrame, pol: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rates = {"T-bill, overall rate (annual mean of quarter-ends)": tb["tbill"], "policy rate (December, D42)":
             pol["policy_rate"]}
    for b, g in dep.groupby("bank_id"):
        g = g.set_index("fiscal_year").sort_index()
        for series, cost in (("all deposits", "cost_of_deposits"), ("time deposits", "cost_time"),
                             ("savings deposits", "cost_savings")):
            c = g[cost]
            for rname, r in rates.items():
                full = pd.DataFrame({"c": c, "r": r.reindex(c.index)}).dropna()
                consec = full.index.to_series().diff() == 1
                dc = full["c"].diff()[consec].to_numpy()
                dr = full["r"].diff()[consec].to_numpy()
                yrs = full.index[consec]
                o = beta_ols(dc, dr)
                a = asym_beta(dc, dr) if o["status"] == "estimated" else {"status": "not estimated"}
                rows.append({"bank_id": b, "deposits": series, "rate": rname, "frequency": "annual (interims hold no "
                             "deposit interest)", "years": f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "",
                             **{f"beta_{k}" if k in ("lo", "hi") else k: v for k, v in o.items()},
                             **{f"asym_{k}" if k == "status" else k: v for k, v in a.items()},
                             "interval": "90% t interval"})
    out = pd.DataFrame(rows).rename(columns={"beta_lo": "beta_lo", "beta_hi": "beta_hi"})
    out["evidence_class"] = "Estimated"
    return out


# --- franchise, reconciliation, liquidity ----------------------------------------------------------------------------


def di_observed(root: Path) -> pd.DataFrame:
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    d = p[p["kind"] == "deposit_insurance_expense"].copy()
    d["fiscal_year"] = d["report_fy"].astype(int)
    return d.rename(columns={"scope": "bank_id"})[["bank_id", "fiscal_year", "value", "file", "pdf_page"]]


def franchise(root: Path, w: pd.DataFrame, dep: pd.DataFrame, tb: pd.DataFrame, smr: pd.DataFrame) -> pd.DataFrame:
    comp = pd.read_csv(path("tables", root) / "ch04_4_0_components.csv", dtype={"bank_id": str})
    comp = comp.set_index(["bank_id", "fiscal_year"])
    f = dep[["bank_id", "fiscal_year", "avg_deposits", "cost_of_deposits"]].copy()
    f["tbill"] = f["fiscal_year"].map(tb["tbill"])
    f["smr_avg"] = f["fiscal_year"].map(smr["smr_avg"])
    f["gdfa"] = (f["tbill"] - f["cost_of_deposits"]) * f["avg_deposits"]
    f["reserve_cost"] = f["smr_avg"] * f["avg_deposits"] * f["tbill"]
    opex = col(w, "operating_expenses").abs().to_numpy()
    di = di_observed(root).groupby(["bank_id", "fiscal_year"])["value"].first()
    key = list(zip(f["bank_id"], f["fiscal_year"], strict=True))
    f["di_observed"] = [di.get(k, np.nan) for k in key]
    rate = (f["di_observed"] / f["avg_deposits"]).dropna()
    di_rate = float(rate.mean()) if len(rate) else np.nan
    f["di_rate_used"] = di_rate
    f["deposit_insurance"] = f["di_observed"].fillna(di_rate * f["avg_deposits"])
    f["di_basis"] = np.where(f["di_observed"].notna(), "printed expense line (Observed)",
                             f"estimated: mean printed rate {di_rate:.4%} of average deposits x average deposits")
    f["opex_ex_di"] = opex - f["deposit_insurance"].fillna(0).to_numpy()
    avg_eq = avg(w, "equity_attributable").fillna(avg(w, "total_equity")).to_numpy()
    avg_te = avg(w, "total_equity").to_numpy()
    avg_ta = avg(w, "total_assets").to_numpy()
    tax = np.array([comp["tax_rate"].get(k, np.nan) for k in key])
    for band, s in OPEX_SHARES.items():
        dib = 1 + DI_BAND if band == "low" else 1 - DI_BAND if band == "high" else 1
        net = f["gdfa"] - f["reserve_cost"] - s * f["opex_ex_di"] - dib * f["deposit_insurance"]
        f[f"net_franchise_{band}"] = net
        f[f"net_franchise_{band}_pct_equity"] = net / avg_te
        f[f"net_franchise_{band}_roe_pts"] = net * (1 - tax) / avg_eq
        f[f"net_franchise_{band}_pct_assets"] = net / avg_ta
    f["gdfa_pct_equity"] = f["gdfa"] / avg_te
    f["gdfa_roe_pts"] = f["gdfa"] * (1 - tax) / avg_eq
    f["gdfa_pct_assets"] = f["gdfa"] / avg_ta
    f["deposit_opex_share_rule"] = "deposit-gathering share of OpEx (ex deposit insurance): 50% central; 25-75% band"
    # reconciliation with the 4.0 funding term (ROE units): funding term = -(interest expense / avg TA) x k
    k = np.array([(1 - comp["tax_rate"].get(x, np.nan)) * comp["ta_e"].get(x, np.nan) * comp["attr_share"].get(x, np.nan)
                  for x in key])
    c_fund = np.array([comp["c_fund"].get(x, np.nan) for x in key])
    ibl_ta = np.array([comp["ibl_ta"].get(x, np.nan) for x in key])
    f["funding_term_40_roe"] = -c_fund * ibl_ta * k
    f["gdfa_roe_40_basis"] = f["gdfa"] / avg_ta * k
    f["bill_yield_on_deposits_roe"] = f["tbill"] * f["avg_deposits"] / avg_ta * k
    f["non_deposit_interest_roe"] = (col(w, "interest_expense").abs().to_numpy()
                                     - col(w, "interest_expense_deposits").abs().to_numpy()) / avg_ta * k
    f["reconciliation_residual"] = (f["gdfa_roe_40_basis"] - (f["funding_term_40_roe"] + f["bill_yield_on_deposits_roe"]
                                                            + f["non_deposit_interest_roe"]))
    f["measure_label"] = ("GDFA: gross deposit funding advantage (proxy, D12); net deposit franchise contribution is "
                          "the measure discussed as rent")
    f["evidence_class"] = "Estimated"
    return f[f["gdfa"].notna()].reset_index(drop=True)


def liquidity(w: pd.DataFrame, dep: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"]})
    loans = col(w, "gross_loans").fillna(col(w, "net_loans"))
    d = col(w, "customer_deposits")
    b = col(w, "borrowings").fillna(0)
    out["ldr"] = div(loans, d)
    out["wholesale_share"] = div(b, d + b)
    out["loan_growth"] = div(loans, lag(w, loans)) - 1
    out["deposit_growth"] = dep["deposit_growth"].to_numpy()
    out["growth_gap"] = out["loan_growth"] - out["deposit_growth"]
    out["nim_assets"] = div(col(w, "net_interest_income"), avg(w, "total_assets"))
    gap = (avg(w.assign(_l=loans), "_l") - avg(w, "customer_deposits"))
    non_dep_int = col(w, "interest_expense").abs() - col(w, "interest_expense_deposits").abs()
    out["cost_of_borrowings"] = div(non_dep_int, avg(w.assign(_b=b), "_b"))
    out["funding_gap_avg"] = gap
    cod = dep["cost_of_deposits"].to_numpy()
    out["funding_gap_cost"] = gap.clip(lower=0) * (out["cost_of_borrowings"] - cod)
    m = fr.set_index(["bank_id", "fiscal_year"])
    k = [(r.bank_id, r.fiscal_year) for r in out.itertuples()]
    ft = np.array([m["funding_term_40_roe"].get(x, np.nan) for x in k])
    scale = np.array([(m["gdfa_roe_40_basis"].get(x, np.nan) / m["gdfa"].get(x, np.nan)
                       if m["gdfa"].get(x, np.nan) else np.nan) for x in k])
    out["funding_gap_cost_roe"] = out["funding_gap_cost"] * scale
    out["funding_term_40_roe"] = ft
    out["gap_cost_share_of_funding_term"] = div(out["funding_gap_cost_roe"], -pd.Series(ft, index=out.index))
    out["evidence_class"] = "Derived"
    return out


# --- hypotheses ------------------------------------------------------------------------------------------------------


def _perm_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rho = stats.spearmanr(a, b).statistic
    perms = [stats.spearmanr(a, np.array(q)).statistic for q in itertools.permutations(b)]
    return float(rho), float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def h61_parts(dep: pd.DataFrame, beta: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """H6.1 parts as numbers (prompt 13b pre-flight B): NMB, CRDB and the smaller banks' median per metric, with the same
    filters as ``hypotheses`` (blueprint section 6, H6.1)."""
    cod = dep.groupby("bank_id")["cost_of_deposits"].agg(["mean", "count"])
    bt = beta[(beta["deposits"] == "all deposits") & beta["rate"].str.startswith("T-bill")].set_index("bank_id")
    nf = fr.groupby("bank_id")["net_franchise_central_roe_pts"].agg(["mean", "count"])
    small = [b for b in cod.index if b not in FOCUS]
    rows = []
    for lab, m, lower in (("cost of deposits (mean)", cod["mean"].where(cod["count"] >= 3), True),
                          ("deposit beta on the T-bill (all deposits)", bt["beta"].where(bt["status"] == "estimated"),
                           True),
                          ("net deposit franchise, ROE points, central (mean)", nf["mean"].where(nf["count"] >= 3),
                           False)):
        m = m.dropna()
        sm = m.reindex(small).dropna()
        rows.append({"hypothesis": "H6.1", "metric": lab, "better": "lower" if lower else "higher",
                     "nmb": m.get("B02", np.nan), "crdb": m.get("B01", np.nan),
                     "smaller_banks_median": float(sm.median()) if len(sm) >= 2 else np.nan,
                     "smaller_banks_n": len(sm), "smaller_banks": ", ".join(sm.index), "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def hypotheses(root: Path, dep: pd.DataFrame, beta: pd.DataFrame, fr: pd.DataFrame, liq: pd.DataFrame,
               t: dict[str, str]) -> pd.DataFrame:
    rows = []
    cod = dep.groupby("bank_id")["cost_of_deposits"].agg(["mean", "count"])
    bt = beta[(beta["deposits"] == "all deposits") & beta["rate"].str.startswith("T-bill")].set_index("bank_id")
    nf = fr.groupby("bank_id")["net_franchise_central_roe_pts"].agg(["mean", "count"])
    small = [b for b in cod.index if b not in FOCUS]

    def better(metric: pd.Series, lower: bool) -> tuple[str, str]:
        m = metric.dropna()
        sm = m.reindex(small).dropna()
        if any(b not in m.index for b in FOCUS) or len(sm) < 2:
            return "inconclusive", (f"fewer than 2 smaller banks with data (n={len(sm)}); NMB "
                                    f"{m.get('B02', float('nan')):.4f}, CRDB {m.get('B01', float('nan')):.4f}")
        med = sm.median()
        ok = [(m[b] < med) if lower else (m[b] > med) for b in FOCUS]
        return ("supported" if all(ok) else "rejected" if not any(ok) else "inconclusive"), \
            f"NMB {m['B02']:.4f}, CRDB {m['B01']:.4f}; smaller banks' median {med:.4f} (n={len(sm)})"

    v1, e1 = better(cod["mean"].where(cod["count"] >= 3), lower=True)
    v2, e2 = better(bt["beta"].where(bt["status"] == "estimated"), lower=True)
    v3, e3 = better(nf["mean"].where(nf["count"] >= 3), lower=False)
    parts = [v1, v2, v3]
    verdict = "supported" if all(p == "supported" for p in parts) else "rejected" if "rejected" in parts and \
        parts.count("supported") == 0 else "inconclusive"
    rows.append({"hypothesis": "H6.1", "verdict": verdict,
                 "evidence": f"cost of deposits (mean, lower is better): {v1}: {e1}; deposit beta on the T-bill "
                             f"(lower is better): {v2}: {e2}; net deposit franchise, ROE points, central (higher is "
                             f"better): {v3}: {e3}",
                 "rule": "each part: supported when both NMB and CRDB beat the smaller banks' median, rejected when "
                         "neither does; overall supported when all three parts are supported"})
    # H6.2: growth gap -> change in wholesale share (+) and change in NIM (-), bank fixed effects
    d = liq.sort_values(["bank_id", "fiscal_year"]).copy()
    prev = d.groupby("bank_id")["fiscal_year"].diff() == 1
    d["d_wholesale"] = d.groupby("bank_id")["wholesale_share"].diff().where(prev)
    d["d_nim"] = d.groupby("bank_id")["nim_assets"].diff().where(prev)
    res = {}
    for y in ("d_wholesale", "d_nim"):
        x = d[["bank_id", "growth_gap", y]].dropna()
        x = x[x["growth_gap"].abs() < 1]  # drop extreme growth years (mergers, recapitalisations)
        xd = x["growth_gap"] - x.groupby("bank_id")["growth_gap"].transform("mean")
        yd = x[y] - x.groupby("bank_id")[y].transform("mean")
        bcoef = float((xd * yd).sum() / (xd * xd).sum())
        e = yd - bcoef * xd
        dfree = len(x) - x["bank_id"].nunique() - 1
        se = float(np.sqrt((e @ e) / dfree / (xd @ xd)))
        tc = stats.t.ppf(1 - (1 - LEVEL) / 2, dfree)
        res[y] = (bcoef, bcoef - tc * se, bcoef + tc * se, len(x))
    w_, n_ = res["d_wholesale"], res["d_nim"]
    sup = w_[1] > 0 and n_[2] < 0
    rej = w_[2] < 0 or n_[1] > 0
    rows.append({"hypothesis": "H6.2", "verdict": "supported" if sup else "rejected" if rej else "inconclusive",
                 "evidence": f"within-bank slope of the change in wholesale share on (loan growth - deposit growth): "
                             f"{w_[0]:+.3f} [{w_[1]:+.3f}, {w_[2]:+.3f}], n = {w_[3]}; of the change in NII / assets: "
                             f"{n_[0]:+.4f} [{n_[1]:+.4f}, {n_[2]:+.4f}], n = {n_[3]}",
                 "rule": "supported when the wholesale slope is positive and the margin slope negative (90% intervals "
                         "exclude zero); rejected when either has the opposite sign significantly"})
    # H5.3 revisited with net franchise ranks
    from tzbank.chapters.ch05.persistence import verdict_specs

    p = pd.read_csv(path("tables", root) / "ch05_persistence.csv", dtype={"bank_id": str})
    br = pd.read_csv(path("tables", root) / "ch05_break_tests.csv", dtype={"bank_id": str})
    v = verdict_specs(p, br)
    j = v[["lambda"]].join(nf[nf["count"] >= 3]["mean"].rename("nf"), how="inner").dropna()
    if len(j) >= 4:
        rho, pv = _perm_spearman(j["lambda"].to_numpy(float), j["nf"].to_numpy(float))
        v53 = "supported" if rho >= 0.5 and pv < 0.10 else "rejected" if rho <= -0.5 and pv < 0.10 else "inconclusive"
        ev = (f"Spearman(lambda, mean net deposit franchise in ROE points) = {rho:+.2f}, exact permutation p = "
              f"{pv:.2f}, {len(j)} banks: " + "; ".join(f"{t.get(b, b)} lambda {r['lambda']:.2f}, net franchise "
                                                         f"{pct(r['nf'], 1)}" for b, r in j.iterrows()))
    else:
        v53, ev = "inconclusive", f"only {len(j)} banks with a lambda and >= 3 years of net franchise"
    rows.append({"hypothesis": "H5.3 (revisited)", "verdict": v53, "evidence": ev,
                 "rule": "persistence (chapter 5 verdict-specification lambda) rises with the net deposit franchise: "
                         "supported if rho >= 0.5 with p < 0.10"})
    h = pd.DataFrame(rows)
    h["evidence_class"] = "Interpretation"
    return h


NOTE = """# Chapter 6: funding architecture

Generated by `tzbank.chapters.ch06_funding` (stage `funding`). Blueprint §2 (deposit franchise), §4 chapter 6;
hypotheses H6.1, H6.2; H5.3 revisited; guardrail D12 (CLAUDE.md §8): GDFA is a proxy; only the net deposit franchise
contribution is discussed as rent.

## Research question

What makes the deposit franchise valuable: how cheap and how sticky are deposits, and what is left of the funding
advantage after reserves, the cost of gathering deposits and deposit insurance?

## Variable definitions

- Cost of deposits = deposit interest / average deposits (by type where split); deposit mix shares, CASA share.
- Deposit beta = OLS slope of the annual change in the cost of deposits on the annual change in the rate (overall
  T-bill rate, annual mean of quarter-ends; December policy rate, D42), 90% t interval; asymmetric betas where each side
  has >= {min_side} years.
- GDFA (proxy) = (T-bill - cost of deposits) x average deposits.
- Net deposit franchise contribution = GDFA - SMR x average deposits x T-bill - s x OpEx (ex deposit insurance) -
  deposit insurance; s = 50% central, band 25-75% (Scenario); deposit insurance +/-50% where estimated. Shown as %
  of average total assets (pre-tax; comparable across banks), % of average total equity (pre-tax) and ROE points
  (after tax, over average attributable equity). **ROE points are scaled by leverage**: the same funding advantage per
  unit of assets is worth more ROE points at a higher assets-to-equity ratio, so compare banks on the % of assets
  figures and read ROE points as the contribution to each bank's own ROE.
- SMR (annual) = time-weighted calendar-year average of the statutory minimum reserve ratio on private deposit
  liabilities (`bot_long.csv`, category `reserve_requirements`).
- LDR = gross loans / deposits; wholesale share = borrowings / (deposits + borrowings); funding gap = average loans -
  average deposits; gap cost = max(gap, 0) x (cost of borrowings - cost of deposits).

## Sources and transformation trail

`data/processed/bank_year_wide.csv`; `rates_quarterly.csv` (T-bill), `rates_annual.csv` (POLICY_RATE_D42);
`data/interim/bot_policy_text.csv` and `bot_long.csv` (SMR events, deposit insurance lines; `tzbank.extract.bot_policy`);
`outputs/tables/ch04_4_0_components.csv` (tax, leverage, funding term); `ch05_persistence.csv` (H5.3). Functions:
`smr_annual`, `deposits`, `deposit_beta`, `franchise`, `liquidity`, `hypotheses`.

## Validation test

`tests/test_ch06.py`: net franchise below GDFA in every year; band ordering on every net figure; deposit-beta rates
dated inside the fiscal year; SMR time-weighting hand case; reconciliation identity with the 4.0 funding term.

## Results

### Statutory minimum reserve (Observed, text-extracted)

{smr}

### Cost of deposits and mix (Derived)

{dep}

### Deposit beta (Estimated)

{beta}

### Gross and net deposit franchise (Estimated; band = 25-75% OpEx allocation, +/-50% deposit insurance)

% of average total assets is the cross-bank comparison; ROE points are scaled by each bank's leverage.

{fr}

### Reconciliation with the 4.0 funding term (Derived)

GDFA in ROE units = 4.0 funding term + T-bill yield on average deposits + non-deposit interest (all x (1 - t) x TA/E
x attributable share). Largest absolute residual: {resid:.2e}.

{recon}

### Liquidity (Derived)

{liq}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

Cost of deposits, mix and liquidity: **Derived**. Deposit betas, GDFA and the net franchise: **Estimated** (the
OpEx allocation band is a **Scenario**). Hypothesis verdicts: **Interpretation**.

## Limitations

- Deposit interest is annual only (interims do not disclose it), so betas rest on about 15-17 annual changes per bank;
  intervals are wide and annual averaging dampens betas.
- The SMR applies to private deposit liabilities; government deposits carried higher requirements (20-40% in BoT
  Table A3.15, 2009-14; 20% to 30% in November 2011 per the text) and their share in each bank's deposits is not held,
  so the reserve cost is a lower bound. The text-extracted private-deposit SMR agrees with Table A3.15 (10%, 2009-14).
  No SMR change after June 2020 was found in the 2021/22-2024/25 reports: unchanged at 6% is an inference
  (collection task: confirm with BoT monetary policy statements). FX-deposit requirements are not modelled.
- Deposit insurance: printed lines only for CRDB FY2017-18; other bank-years use the mean printed rate (Estimated).
  The Deposit Insurance Board premium schedule is a collection task.
- The deposit-gathering share of OpEx is an allocation rule, not an observation (branch and agent cost shares are
  not disclosed); the 25-75% band brackets it.
- Smaller banks: deposit interest is sparse, so H6.1 comparisons rest on few bank-years.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

A large and positive **net** franchise with a low deposit beta supports **low-cost deposits** as a source of rent;
if the net figure collapses once the OpEx of the branch and agent network is charged, the advantage is better read
as **scale economies** in distribution (the network pays for itself) than as rent. Reserve cost moves with BoT policy
(SMR 10% to 6% since 2015), a **regulatory** driver of the franchise value.
"""


def write_funding(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    w = wide(root)
    tb, pol = tbill_annual(root), policy_annual(root)
    ev = smr_events(root)
    smr = smr_annual(ev, list(range(2005, 2027)))
    dep = deposits(w)
    beta = deposit_beta(dep, tb, pol)
    fr = franchise(root, w, dep, tb, smr)
    liq = liquidity(w, dep, fr)
    hyp = hypotheses(root, dep, beta, fr, liq, t)
    tab, figd = path("tables", root), path("figures", root)
    smr_t = smr.reset_index().assign(evidence_class="Derived",
                                     basis="time-weighted average of BoT SMR events (private deposit liabilities)")
    rates_t = tb.join(pol, how="outer").reset_index().rename(columns={"index": "year"}).assign(evidence_class="Observed")
    files = [write_csv(smr_t, tab / "ch06_smr_annual.csv", require_evidence=True),
             write_csv(rates_t, tab / "ch06_rates_used.csv", require_evidence=True),
             write_csv(dep, tab / "ch06_deposits.csv", require_evidence=True),
             write_csv(beta, tab / "ch06_deposit_beta.csv", require_evidence=True),
             write_csv(fr, tab / "ch06_franchise.csv", require_evidence=True),
             write_csv(liq, tab / "ch06_liquidity.csv", require_evidence=True),
             write_csv(hyp, tab / "ch06_hypotheses.csv", require_evidence=True),
             write_csv(h61_parts(dep, beta, fr), tab / "ch06_h61_parts.csv", require_evidence=True)]
    # figures
    figs = []
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in FOCUS:
        d = dep[dep["bank_id"] == b].dropna(subset=["cost_of_deposits"])
        ax.plot(d["fiscal_year"], d["cost_of_deposits"] * 100, color=COLORS[b], lw=2, marker="o", ms=3,
                label=f"{t[b]} cost of deposits")
    y0 = int(dep.loc[dep["bank_id"].isin(FOCUS)].dropna(subset=["cost_of_deposits"])["fiscal_year"].min())
    y1 = int(dep["fiscal_year"].max())
    tbv, polv = tb.loc[y0:y1, "tbill"] * 100, pol.loc[y0:y1, "policy_rate"] * 100  # rates over the bank years only
    ax.plot(tbv.index, tbv, color="#5a5a55", lw=1.2, ls="--",
            label="T-bill, overall rate (annual mean of quarter-ends)")
    ax.plot(polv.index, polv, color="#1f1f1d", lw=1, ls=":", label="policy rate (December, D42)")
    ax.set_xlim(y0 - 0.5, y1 + 0.5)
    ax.set_ylim(0, float(np.nanmax([tbv.max(), polv.max()])) * 1.08)
    style(ax, "Cost of deposits against the T-bill and policy rates", "percent")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch06_cost_of_deposits.png", "Evidence class: Derived. Deposit interest / average deposits; "
               "T-bill: BoT overall rate, mean of quarter-ends (364-day auction yield where the BoT series is not held, D64).")
    data = write_csv(dep[dep["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "cost_of_deposits", "evidence_class"]],
                     figd / "ch06_cost_of_deposits_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Cost of deposits and rates", "evidence_class": "Derived"})
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in FOCUS:
        d = fr[fr["bank_id"] == b]
        ax.plot(d["fiscal_year"], d["gdfa_roe_pts"] * 100, color=COLORS[b], lw=1, ls="--", label=f"{t[b]} GDFA (proxy)")
        ax.plot(d["fiscal_year"], d["net_franchise_central_roe_pts"] * 100, color=COLORS[b], lw=2,
                label=f"{t[b]} net franchise (central)")
        ax.fill_between(d["fiscal_year"], d["net_franchise_low_roe_pts"] * 100, d["net_franchise_high_roe_pts"] * 100,
                        color=COLORS[b], alpha=0.15, lw=0)
    ax.axhline(0, color="#5a5a55", lw=0.8)
    style(ax, "Gross (proxy) and net deposit franchise, ROE points", "ROE percentage points")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch06_franchise.png", "Evidence class: Estimated. Band: 25-75% OpEx allocation, +/-50% "
               "deposit insurance. GDFA is a proxy (D12).")
    data = write_csv(fr[fr["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "gdfa_roe_pts", "net_franchise_low_roe_pts",
                                                     "net_franchise_central_roe_pts", "net_franchise_high_roe_pts",
                                                     "evidence_class"]], figd / "ch06_franchise_data.csv",
                     require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Gross and net deposit franchise (ROE points, band)",
                 "evidence_class": "Estimated"})
    files += [f["path"] for f in figs] + [f["data"] for f in figs]
    # note text
    smr_txt = "\n".join(f"- {r.event_date:%Y-%m-%d} ({r.date_precision}): {r.value:.0%}"
                        + (f" (from {r.from_value:.0%})" if pd.notna(r.from_value) else "")
                        + f"; {r.sources}; {r.gate_result}" for r in ev.itertuples())
    smr_txt += ("\n- Annual averages: " + ", ".join(f"{y}: {v:.1%}" for y, v in smr["smr_avg"].loc[2008:2025].items()))
    dep_lines = []
    for b in FOCUS:
        d = dep[(dep["bank_id"] == b)].dropna(subset=["cost_of_deposits"])
        last = d.iloc[-1]
        mix = dep[(dep["bank_id"] == b)].dropna(subset=["casa_share"])
        tm = dep[(dep["bank_id"] == b)].dropna(subset=["cost_time"])
        dep_lines.append(f"- {t[b]}: mean cost of deposits {pct(d['cost_of_deposits'].mean(), 2)} "
                         f"({int(d['fiscal_year'].min())}-{int(d['fiscal_year'].max())}); FY{int(last['fiscal_year'])} "
                         f"{pct(last['cost_of_deposits'], 2)}"
                         + (f"; CASA share FY{int(mix['fiscal_year'].iloc[-1])} {pct(mix['casa_share'].iloc[-1])} "
                            f"(FY{int(mix['fiscal_year'].iloc[0])} {pct(mix['casa_share'].iloc[0])})" if len(mix) else "")
                         + (f"; time-deposit cost FY{int(tm['fiscal_year'].iloc[-1])} {pct(tm['cost_time'].iloc[-1], 2)}"
                            if len(tm) else "") + ".")
    bl = []
    for _, r in beta[(beta["deposits"] == "all deposits")].iterrows():
        if r["status"] == "estimated":
            a = (f"; rising {r['beta_up']:+.2f} [{r['beta_up_lo']:+.2f}, {r['beta_up_hi']:+.2f}], falling "
                 f"{r['beta_down']:+.2f} [{r['beta_down_lo']:+.2f}, {r['beta_down_hi']:+.2f}]"
                 if r.get("asym_status") == "estimated" else f"; asymmetric: {r.get('asym_status')}")
            bl.append(f"- {t.get(r['bank_id'], r['bank_id'])}, {r['rate']} ({r['years']}, n={int(r['n'])}): beta "
                      f"{r['beta']:+.2f} [{r['beta_lo']:+.2f}, {r['beta_hi']:+.2f}]{a}.")
        elif r["bank_id"] in FOCUS:
            bl.append(f"- {t[r['bank_id']]}, {r['rate']}: {r['status']}")
    frl = []
    for b in FOCUS:
        d = fr[fr["bank_id"] == b]
        last = d.iloc[-1]
        frl.append(f"- {t[b]} FY{int(last['fiscal_year'])}: GDFA (proxy) {pct(last['gdfa_pct_assets'], 2)} of average "
                   f"assets, net franchise {pct(last['net_franchise_central_pct_assets'], 2)} (band "
                   f"{pct(last['net_franchise_low_pct_assets'], 2)} to {pct(last['net_franchise_high_pct_assets'], 2)});"
                   f" in ROE points (scaled by leverage): GDFA {pct(last['gdfa_roe_pts'])}; net "
                   f"franchise {pct(last['net_franchise_central_roe_pts'])} (band "
                   f"{pct(last['net_franchise_low_roe_pts'])} to {pct(last['net_franchise_high_roe_pts'])}); "
                   f"{pct(last['net_franchise_central_pct_equity'])} of average equity pre-tax (band "
                   f"{pct(last['net_franchise_low_pct_equity'])} to {pct(last['net_franchise_high_pct_equity'])}). "
                   f"Mean 2010-25: net {pct(d.loc[d['fiscal_year'] >= 2010, 'net_franchise_central_roe_pts'].mean())}"
                   f" (band {pct(d.loc[d['fiscal_year'] >= 2010, 'net_franchise_low_roe_pts'].mean())} to "
                   f"{pct(d.loc[d['fiscal_year'] >= 2010, 'net_franchise_high_roe_pts'].mean())}).")
    rec = []
    for b in FOCUS:
        d = fr[(fr["bank_id"] == b)].dropna(subset=["funding_term_40_roe"])
        last = d.iloc[-1]
        rec.append(f"- {t[b]} FY{int(last['fiscal_year'])}: 4.0 funding term {pct(last['funding_term_40_roe'])}; + bill "
                   f"yield on deposits {pct(last['bill_yield_on_deposits_roe'])}; + non-deposit interest "
                   f"{pct(last['non_deposit_interest_roe'])}; = GDFA {pct(last['gdfa_roe_40_basis'])} (ROE units).")
    ll = []
    for b in FOCUS:
        d = liq[liq["bank_id"] == b].dropna(subset=["ldr"])
        last = d.iloc[-1]
        ll.append(f"- {t[b]} FY{int(last['fiscal_year'])}: LDR {pct(last['ldr'])}, wholesale share "
                  f"{pct(last['wholesale_share'])}, funding gap cost {pct(last['funding_gap_cost_roe'], 2)} of ROE "
                  f"against a 4.0 funding term of {pct(last['funding_term_40_roe'])}.")
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Rule: {r.rule}."
                        for r in hyp.itertuples())
    interp = ("**Interpretation.** Read the net franchise, not GDFA: the gap between the two is "
              "the cost of running the deposit network and holding reserves. A net figure that stays positive across "
              "the whole OpEx band is evidence of a funding rent; one that changes sign inside the band is not.")
    text = NOTE.format(min_side=MIN_SIDE, smr=smr_txt, dep="\n".join(dep_lines), beta="\n".join(bl),
                       fr="\n".join(frl), resid=float(fr["reconciliation_residual"].abs().max()),
                       recon="\n".join(rec), liq="\n".join(ll), hyp=hyp_txt, interp=interp)
    nf = write_text(text, path("notes", root) / "ch06.md")
    files.append(nf)
    desc = ["SMR, annual time-weighted average", "Rates used (T-bill, policy rate) with their dates",
            "Cost of deposits and deposit mix", "Deposit betas (T-bill, policy rate; asymmetric)",
            "GDFA (proxy) and net deposit franchise with bands; reconciliation with 4.0",
            "Liquidity: LDR, wholesale share, funding gap cost", "H6.1, H6.2 and H5.3 revisited",
            "H6.1 parts: NMB, CRDB and the smaller banks' median per metric"]
    cls = ["Derived", "Observed", "Derived", "Estimated", "Estimated", "Derived", "Interpretation", "Derived"]
    m = {"chapter": "06", "title": "Funding architecture", "hypotheses": ["H6.1", "H6.2", "H5.3"],
         "status": "built",
         "tables": [{"path": rel(f, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for f, d, c in zip(files[:8], desc, cls, strict=True)],
         "figures": [{"path": rel(f["path"], root), "data": rel(f["data"], root), "description": f["description"],
                      "evidence_class": f["evidence_class"], "note": rel(nf, root)} for f in figs],
         "notes": [rel(nf, root)],
         "inputs": ["data/processed/bank_year_wide.csv", "data/processed/rates_quarterly.csv",
                    "data/processed/rates_annual.csv", "data/interim/bot_policy_text.csv",
                    "outputs/tables/ch04_4_0_components.csv", "outputs/tables/ch05_persistence.csv"],
         "module": "tzbank.chapters.ch06_funding", "run_log_stage": "funding"}
    files.append(write_json(m, path("manifests", root) / "ch06.json"))
    return files
