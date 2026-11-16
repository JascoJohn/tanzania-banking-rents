"""Chapter 14 (prompts 14-15): market behaviour and valuation extensions. No register changes (register v1.3).

Exhibits (one function each; evidence class on every row; dashboard page "14. Market behaviour and extensions";
findings pack ids CH14.*). Daily analyses read the DSE data library (D72) through ``ch14_market_data``:
 1 ``return_attribution``  log total return = EPS growth + P/E change + dividends (exact, additive), and = BVPS growth
                           + P/B change + dividends, over 1, 3, 5, 10 years and since listing to FY2025 (year-end
                           data). Dividends: total-return index with the dividend declared for FY t-1 received in year t.
 2 ``event_study``         ARs / CARs around dated events against the market proxy (all DSE equities ex NMB and CRDB;
                           market model on [-250,-30]) with the other bank as a second benchmark (prompt 15 item 5a).
 3 split study             NMB 1:10 split: DiD against CRDB and the market, +/-5, 20, 60 days, liquidity and spread
                           measures (stale-filtered), stated objectives vs outcomes, earlier DSE splits (item 5b).
 4 ``seasonality``         monthly returns and turnover 2015-2026, January / December / ex-dividend-month dummies with
                           HAC errors, block share by month (item 5c). Co-movement and rolling correlation (item 5d).
   ``microstructure``      autocorrelation and variance ratios unfiltered (labelled as affected by stale prices) and
                           stale-filtered; block vs board days; clustering; 52-week high; turnover spikes (item 5e).
 5 ``valuation_extensions`` price-implied terminal growth (v1.3 ROE path held); bear / base / bull price lines; terminal
                           ROE x persistence grid; real ROE vs real COE (IMF CPI); dividend yield minus the T-bill.
 6 ``annual_report_items`` page-checked annual-report extraction FY2023-25 (raw manual entries).
 7 ``catalysts`` / ``thesis_breaks`` dated catalysts and thesis-break indicators per bank.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters import ch14_market_data as M
from tzbank.chapters.ch04.common import plt_subplots, save, style
from tzbank.chapters.ch09_valuation import DOWN
from tzbank.chapters.ch13_report_support import FOCUS, T, _shift, _wide
from tzbank.config import path, settings
from tzbank.io import rel, write_csv, write_json, write_text
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un

BENCH = ("market proxy: capitalisation-weighted index of all DSE equities except NMB and CRDB (data/library/dse, "
         "Derived); second benchmark: the other focus bank")
LISTED = {"B02": 2008, "B01": 2009}
HOUSE = {"NMB": "#eb6834", "CRDB": "#2a78d6", "market proxy": "#9a9a94"}  # house colours (NMB orange, CRDB blue)


def _raw(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("raw", root) / "manual_entries" / name, dtype={"bank_id": str})


# --- 1 ---------------------------------------------------------------------------------------------------------------
def _per_share(root: Path) -> pd.DataFrame:
    w = _wide(root)
    rows = []
    for b in FOCUS:
        x = w[w["bank_id"] == b].set_index("fiscal_year").sort_index()
        for y, r in x.iterrows():
            sh = r["shares_outstanding"]
            dps = r["dps_total"] if pd.notna(r.get("dps_total")) else r.get("dps")
            rows.append({"bank_id": b, "fiscal_year": int(y), "price": r["share_price_close"],
                         "eps": r["profit_attributable"] / sh, "bvps": r["equity_attributable"] / sh, "dps": dps})
    return pd.DataFrame(rows)


def return_attribution(root: Path) -> pd.DataFrame:
    ps = _per_share(root)
    out = []
    for b in FOCUS:
        g = ps[ps["bank_id"] == b].set_index("fiscal_year").sort_index()
        g = g[g.index >= LISTED[b]]
        end = 2025
        run = [y for y in range(end, LISTED[b] - 1, -1) if y in g.index and pd.notna(g.loc[y, "price"])]
        first = min(y for i, y in enumerate(run) if run[:i + 1] == list(range(end, y - 1, -1)))  # continuous run
        g["tr_index"] = np.nan
        tr = 1.0
        for y in range(first, end + 1):
            if y > first:
                d = g.loc[y - 1, "dps"] if pd.notna(g.loc[y - 1, "dps"]) else 0.0  # FY t-1 dividend paid in year t
                tr *= (g.loc[y, "price"] + d) / g.loc[y - 1, "price"]
            g.loc[y, "tr_index"] = tr
        listing = (("since listing", int(g.index.min())) if first == int(g.index.min()) else
                   (f"since FY{first} (continuous prices; FY{int(g.index.min())} listing, gap before FY{first})", first))
        horizons = [("1 year", end - 1), ("3 years", end - 3), ("5 years", end - 5), ("10 years", end - 10), listing]
        ok_ps = [y for y in range(first, end) if g.loc[y, ["eps", "bvps"]].notna().all() and g.loc[y, "eps"] > 0]
        if ok_ps and ok_ps[0] > listing[1]:  # listing-year per-share data not held: add the first complete start
            horizons.append((f"since FY{ok_ps[0]} (first year with EPS and BVPS; listed FY{int(g.index.min())})",
                             ok_ps[0]))
        for lab, start in horizons:
            if start < first or start not in g.index:
                continue
            s, e = g.loc[start], g.loc[end]
            lp = np.log(e["price"] / s["price"])
            ltr = np.log(e["tr_index"] / s["tr_index"])
            ok = s["eps"] > 0 and e["eps"] > 0
            n = end - start
            out.append({"bank": T[b], "horizon": lab, "start_fy": start, "end_fy": end, "years": n,
                        "log_total_return": ltr, "annualised_total_return": np.exp(ltr / n) - 1,
                        "log_price_change": lp, "eps_growth_log": np.log(e["eps"] / s["eps"]) if ok else np.nan,
                        "pe_change_log": np.log((e["price"] / e["eps"]) / (s["price"] / s["eps"])) if ok else np.nan,
                        "bvps_growth_log": np.log(e["bvps"] / s["bvps"]),
                        "pb_change_log": np.log((e["price"] / e["bvps"]) / (s["price"] / s["bvps"])),
                        "dividends_log": ltr - lp,
                        "split_flag": "per-share data as reported; no split inside FY2008-25 (NMB 1:10 split on 24 Aug "
                                      "2026 falls after FY2025); EPS / BVPS on year-end shares, so rights or bonus "
                                      "issues change per-share growth" if b == "B02" else
                                      "per-share data as reported; no split inside the window; EPS / BVPS on year-end "
                                      "shares (rights issues change per-share growth)",
                        "evidence_class": "Derived"})
    d = pd.DataFrame(out)
    d["check_eps_route"] = d["eps_growth_log"] + d["pe_change_log"] + d["dividends_log"] - d["log_total_return"]
    d["check_bvps_route"] = d["bvps_growth_log"] + d["pb_change_log"] + d["dividends_log"] - d["log_total_return"]
    return d


# --- 2 ---------------------------------------------------------------------------------------------------------------
def _events(root: Path) -> pd.DataFrame:
    return _raw(root, "events_collected.csv")


def event_study(root: Path) -> pd.DataFrame:
    """Prompt 15 item 5a: ARs and CARs against the market proxy (all equities ex NMB and CRDB), market model on
    [-250,-30], with the other bank as a second benchmark (``ch14_market_data.event_study``)."""
    return M.event_study(root, _events(root))


# --- 3 ---------------------------------------------------------------------------------------------------------------
def microstructure(root: Path) -> pd.DataFrame:
    """Prompt 15 item 5e / pre-flight C: both versions (unfiltered, labelled as affected by stale prices; and
    stale-filtered) from the DSE library, 2015 to the cutoff."""
    return M.microstructure(root)


# --- 4 ---------------------------------------------------------------------------------------------------------------
def dividend_months(root: Path) -> dict[str, set]:
    ev = _events(root)
    x = ev[ev["event"].str.contains("ex-dividend", case=False)]
    return {b: set(pd.to_datetime(g["event_date"]).dt.month) for b, g in x.groupby("bank")}


def seasonality(root: Path) -> pd.DataFrame:
    """Prompt 15 item 5c: monthly returns and turnover 2015-2026, January / December (and ex-dividend month) dummies
    with HAC errors; block share by month."""
    return M.seasonality(root, dividend_months(root))


# --- 5 ---------------------------------------------------------------------------------------------------------------
def _v(ctx, val, b, scen="base", dr=None) -> float:
    c = ctx.cases[(val, b, scen)]
    return un.value(c, dr or c.dr) / c.shares


def valuation_extensions(ctx) -> dict[str, pd.DataFrame]:
    root = ctx.root
    guard = float(settings(root)["valuation"]["coe_g_min_gap"])
    imp, lines, grid = [], [], []
    for val in rg.DATES:
        for b in FOCUS:
            c = ctx.cases[(val, b, "base")]
            f = lambda g, c=c: un.value(c, _shift(c.dr, "terminal g", g - c.dr.g)) / c.shares - c.price  # noqa: E731
            lo, hi = -0.10, c.r - guard
            flo, fhi = f(lo), f(hi)
            if flo * fhi > 0:
                g_imp, status = np.nan, (f"price above value for every g from -10% to the COE - g guard "
                                         f"({hi:.1%}): no implied growth" if fhi < 0 else
                                         f"price below value for every g from -10% to {hi:.1%}: no implied growth")
            else:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if f(mid) * flo > 0:
                        lo, flo = mid, f(mid)
                    else:
                        hi = mid
                g_imp, status = (lo + hi) / 2, "solved (bisection)"
            imp.append({"valuation": val, "bank": T[b], "price": c.price, "base_g": c.dr.g, "coe": c.r,
                        "g_upper_bound_guard": c.r - guard,
                        "value_at_upper_bound": fhi + c.price if not np.isfinite(g_imp) else np.nan,
                        "value_at_g_minus_10pct": f(-0.10) + c.price, "implied_g": g_imp,
                        "implied_minus_base": g_imp - c.dr.g if np.isfinite(g_imp) else np.nan,
                        "status": status, "held": "v1.3 ROE path, ROE*, lambda, payout", "evidence_class": "Scenario"})
            for scen in ("bear", "base", "bull"):
                lines.append({"valuation": val, "bank": T[b], "line": f"{scen}-case value per share",
                              "price_line": _v(ctx, val, b, scen), "price": c.price,
                              "value_to_price": _v(ctx, val, b, scen) / c.price, "evidence_class": "Scenario"})
            # prompt 15b item 3: the price at which the bear-case downside equals the D4 screen (bear / P - 1 = -25%):
            # at or below this price the bear screen no longer blocks a Buy
            bear = _v(ctx, val, b, "bear")
            lines.append({"valuation": val, "bank": T[b], "line": "bear-screen Buy line (bear downside = 25%, D4)",
                          "price_line": bear / (1 + DOWN), "price": c.price,
                          "value_to_price": bear / (1 + DOWN) / c.price, "definition": "P such that bear / P - 1 = "
                          f"{DOWN:.0%}; the Buy also needs expected return > COE + 5 pp", "evidence_class": "Scenario"})
            if val == "CURR":
                for dro in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08):
                    for lam in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                        dr = _shift(_shift(c.dr, "ROE*", dro), "lambda", lam - c.dr.lam)
                        v = un.value(c, dr) / c.shares
                        grid.append({"valuation": val, "bank": T[b], "roe_star": c.dr.roe_star + dro,
                                     "roe_star_shift": dro, "lambda": lam, "value_per_share": v, "price": c.price,
                                     "value_to_price": v / c.price, "base_roe_star": c.dr.roe_star,
                                     "base_lambda": c.dr.lam, "evidence_class": "Scenario"})
    # real ROE vs real COE, and dividend yield minus the T-bill (FY2010-25)
    cpi = _raw(root, "imf_cpi_tza.csv").set_index("year")["inflation_avg_cpi"]
    ca = pd.read_csv(path("tables", root) / "ch05_coe_annual.csv")
    ca = ca[ca["date_label"].astype(str).str.startswith("FY")] if ca["date_label"].astype(str).str.startswith("FY").any()         else ca
    coe = ca.drop_duplicates("year", keep="first").set_index("year")["coe"]  # fiscal year-end COE (chapter 5)
    d4 = pd.read_csv(path("tables", root) / "ch04_dupont_panel.csv", dtype={"bank_id": str})
    tb = pd.read_csv(path("tables", root) / "ch06_rates_used.csv").set_index("year")["tbill"]
    ps = _per_share(root).set_index(["bank_id", "fiscal_year"])
    real, dy = [], []
    for b in FOCUS:
        for y in range(2010, 2026):
            rr = d4[(d4["bank_id"] == b) & (d4["fiscal_year"] == y)]["roe"]
            if len(rr) and y in cpi.index and y in coe.index:
                pi = float(cpi[y])
                real.append({"bank": T[b], "fiscal_year": y, "roe": float(rr.iloc[0]), "coe": float(coe[y]),
                             "inflation_imf": pi, "real_roe": (1 + float(rr.iloc[0])) / (1 + pi) - 1,
                             "real_coe": (1 + float(coe[y])) / (1 + pi) - 1,
                             "real_excess": (1 + float(rr.iloc[0])) / (1 + pi) - (1 + float(coe[y])) / (1 + pi),
                             "evidence_class": "Derived"})
            if (b, y) in ps.index:
                p = ps.loc[(b, y)]
                yl = p["dps"] / p["price"] if pd.notna(p["dps"]) and p["price"] else np.nan
                t = float(tb[y]) if y in tb.index and pd.notna(tb.get(y)) else np.nan
                dy.append({"bank": T[b], "fiscal_year": y, "dividend_yield": yl, "tbill": t,
                           "yield_minus_tbill": yl - t, "tbill_series": "BoT overall T-bill rate, annual (ch06; the "
                           "364-day yield is not held as a series)", "evidence_class": "Derived"})
    return {"implied_growth": pd.DataFrame(imp), "price_lines_scenarios": pd.DataFrame(lines),
            "roe_lambda_grid": pd.DataFrame(grid), "real_roe_coe": pd.DataFrame(real),
            "yield_minus_tbill": pd.DataFrame(dy)}


# --- 6 ---------------------------------------------------------------------------------------------------------------
def annual_report_items(root: Path) -> pd.DataFrame:
    import pymupdf

    d = _raw(root, "annual_report_items.csv")
    docs = {}
    chk = []
    for r in d.itertuples():
        if r.status != "observed":
            chk.append("")
            continue
        f = path("documents", root) / r.source_doc
        if not f.exists():
            chk.append("document not held")
            continue
        if f not in docs:
            docs[f] = pymupdf.open(f)
        t = re.sub(r"\s+", " ", docs[f][int(r.page) - 1].get_text())
        chk.append("value found on the cited page" if str(r.value_as_printed) in t else "NOT FOUND on the cited page")
    d["page_check"] = chk
    d["bank"] = d["bank_id"].map(T)
    sal = d[d["variable"] == "salary_backed_loans"].set_index(["bank_id", "fiscal_year"])["value"].astype(float)
    gl = d[d["variable"] == "gross_loans_note"].set_index(["bank_id", "fiscal_year"])["value"].astype(float)
    extra = [{"bank_id": k[0], "bank": T[k[0]], "fiscal_year": k[1], "variable": "salary_backed_share_of_loans",
              "value": sal[k] / gl[k], "unit": "ratio", "status": "derived", "note": "salary_backed_loans / "
              "gross_loans_note (CRDB: Personal segment proxy)", "evidence_class": "Derived"}
             for k in sal.index if k in gl.index]
    return pd.concat([d, pd.DataFrame(extra)], ignore_index=True)


# --- 7 ---------------------------------------------------------------------------------------------------------------
def catalysts(root: Path) -> pd.DataFrame:
    ev = _raw(root, "events_collected.csv")
    rows = [{"bank": r.bank, "date": r.event_date, "catalyst": r.event, "status": "past (dated)",
             "date_basis": f"dated source ({r.confidence})", "evidence_class": "Observed"} for r in ev.itertuples()]
    fut = [("NMB", "2027-01-31", "FY2026 unaudited results", "pattern: FY2025 unaudited results on 28 Jan 2026"),
           ("NMB", "2027-06-10", "AGM and FY2026 dividend approval", "pattern: AGM on 10 Jun 2026"),
           ("CRDB", "2027-04-15", "FY2026 audited results and dividend declaration", "pattern: 13-14 Apr 2026"),
           ("CRDB", "2027-05-05", "ex-dividend (FY2026 dividend)", "pattern: ex-date 5 May 2026"),
           ("NMB", "2027-07-31", "H1 2027 results", "pattern: 24 Jul 2026"),
           ("CRDB", "2027-07-31", "H1 2027 results", "pattern: 31 Jul 2026"),
           ("both", "", "BoT D-SIB designation and any capital surcharge", "date not known (open issue 1)"),
           ("both", "", "BoT Monetary Policy Committee decisions on the CBR (T-bill path driver)",
            "quarterly; dates not collected")]
    rows += [{"bank": b, "date": d, "catalyst": c, "status": "expected", "date_basis": basis,
              "evidence_class": "Interpretation"} for b, d, c, basis in fut]
    return pd.DataFrame(rows)


def thesis_breaks(root: Path) -> pd.DataFrame:
    tab = path("tables", root)
    fb = pd.read_csv(tab / "ch13_franchise_breakeven.csv")
    fb = fb[fb["tbill_case"].str.startswith("v1.3")].set_index("bank")
    ttc = pd.read_csv(tab / "ch08_ttc_cost_of_risk.csv", dtype={"bank_id": str})
    ttc = ttc[ttc["definition_type"].astype(str).str.contains("primary", case=False)].set_index("bank_id")["ttc_mean"]
    h8 = pd.read_csv(tab / "ch08_credit_history.csv", dtype={"bank_id": str}).set_index(["bank_id", "fiscal_year"])
    beta = pd.read_csv(tab / "ch06_deposit_beta.csv", dtype={"bank_id": str})
    cap = pd.read_csv(tab / "ch07_capital.csv", dtype={"bank_id": str})
    val = pd.read_csv(tab / "ch09_valuations.csv")
    val = val[val["valuation"] == "CURR"].set_index("bank")
    rows = []

    def add(bank, ind, cur, thr, brk, src, note=""):
        rows.append({"bank": bank, "indicator": ind, "current": cur, "threshold": thr, "status":
                     "BREACHED" if brk else "not breached", "source": src, "note": note, "evidence_class": "Derived"})

    for b in FOCUS:
        n = T[b]
        add(n, "T-bill rate vs the net deposit franchise break-even", fb.loc[n, "tbill"], fb.loc[n, "breakeven_tbill"],
            fb.loc[n, "tbill"] < fb.loc[n, "breakeven_tbill"], "ch13_franchise_breakeven")
        cr = float(h8.loc[(b, 2025), "cost_of_risk"]) if (b, 2025) in h8.index else np.nan
        add(n, "cost of risk FY2025 vs through-the-cycle mean", cr, float(ttc.get(b, np.nan)),
            cr > float(ttc.get(b, np.nan)), "ch08_credit_history; ch08_ttc_cost_of_risk",
            "break when credit cost normalises above the cycle mean")
        bt = beta[(beta["bank_id"] == b) & (beta["deposits"] == "all deposits")].iloc[0]
        add(n, "deposit beta (upper 90% bound) vs 0.10", float(bt["beta_hi"]), 0.10, float(bt["beta_hi"]) > 0.10,
            "ch06_deposit_beta", "break when depositors start to reprice")
        c = cap[(cap["bank_id"] == b) & cap["binding_headroom"].notna()].sort_values("fiscal_year").iloc[-1]
        add(n, "binding capital headroom vs 2 pp", float(c["binding_headroom"]), 0.02,
            float(c["binding_headroom"]) < 0.02, "ch07_capital")
        v = val.loc[n]
        add(n, "price vs base value per share (31 Oct 2026)", float(v["price"]), float(v["value_per_share"]),
            float(v["price"]) < float(v["value_per_share"]), "ch09_valuations",
            "break for a Sell call when the price falls below the base value (value / price above 1)")
    return pd.DataFrame(rows)


# --- note, figures, manifest -------------------------------------------------------------------------------------------
NOTE = """# Chapter 14: market behaviour and valuation extensions

Generated by `tzbank.chapters.ch14_market_extensions` with `tzbank.chapters.ch14_market_data` (stage `market_ext`;
prompts 14 and 15). Register {version} (no register changes), input hash `{hash}`.

## Research question

How did the two banks' shares behave (long-horizon return sources, around corporate events and the NMB split, by
season, together, in daily trading), what does the price imply under the v1.3 path, and which dated catalysts and
thresholds would change the view?

## Variable definitions

Return attribution: log total return = ln(EPS_T / EPS_0) + ln(PE_T / PE_0) + dividends (total-return index; the FY t-1
dividend received in year t); the same with BVPS and P/B. Daily data: the DSE data library (`data/library/dse/`,
D72), split-adjusted, cut off at {cutoff}. Stale observation: a record with zero volume (close carried forward).
Stale-filtered return: log change between consecutive non-stale closes. Market proxy: capitalisation-weighted index of
all DSE equities except NMB and CRDB (shares in issue from the DSE overview pages, split-adjusted backwards).
AR = return - (alpha + beta x proxy return), market model on [-250,-30] trading days (market-adjusted when fewer than
100 estimation days); CAR = sum of ARs; t = CAR / (residual sd x sqrt(days)). Split study: windows of 5, 20 and 60
trading days before and from 24 Aug 2026; DiD = (NMB post - pre) - (control post - pre); HAC regressions (Newey-West,
5 lags) of daily outcomes on treat, post and treat x post. Amihud = mean(|r| / TZS bn turnover) on traded non-stale
days; Roll = 2 sqrt(-cov(r_t, r_t-1)); Corwin-Schultz two-day high-low estimator. Seasonality: monthly returns on
January, December and ex-dividend-month dummies, Newey-West (3 lags). Variance ratio VR(q) = Var(q-day sum) /
(q Var(1-day)). Quarterly cost of risk = YTD impairment x 12 / months / period-end net loans.

## Sources and transformation trail

`data/library/dse/` (daily_prices, index_proxy, index_history, corporate_actions; built by `tzbank.ingest.dse_library`
from raw DSE snapshots), `data/processed/bank_year_wide.csv`, `quarterly_long.csv` and `interim_long.csv` (inputs
workbook Quarterly Results / Interim Results, D73), `market_daily.csv` (reconciliation),
`data/raw/manual_entries/events_collected.csv`, `annual_report_items.csv`, `imf_cpi_tza.csv`; chapter tables ch04-ch09,
ch13; the valuation context.

## Validation test

`tests/test_ch14.py` and `tests/test_dse_library.py`: attribution identities, CAR arithmetic, the stale filter, Roll and
Corwin-Schultz on known inputs, DiD arithmetic, split adjustment, the index proxy on a two-stock example, the library
schema, the quarterly extractor's gates and its match with the held interim values.

## Results

{results}

## Evidence class

Attribution, event studies, split study, seasonality means, co-movement, microstructure, quarterly ratios: **Derived**
(descriptive); regression tests: **Estimated**. Valuation extensions: **Scenario**. Annual-report items, index history,
events: **Observed**. Catalysts: **Observed** (dated) / **Interpretation** (expected).

## Mechanisms (blueprint section 0)

The stale filter matters: the unfiltered autocorrelation and variance ratios mix carried-forward closes (zero returns)
with real price changes and so misstate predictability; the filtered versions are the ones to read. Results and
dividend events move the event bank relative to the market proxy, and the split study separates the mechanical
affordability effect from trading-activity changes against CRDB and the market. These are market-behaviour findings;
they neither support nor contradict the rent explanations of chapters 4-8 directly, but they bound how fast the market
prices news about them.

## Limitations

- Event studies are case studies (a handful of dated events per bank); t statistics assume independent residuals.
- The market proxy holds share counts constant except for the NMB split (rights and bonus issues not modelled); over
  the year the DSE publishes, it tracks the DSEI with the correlation shown in the proxy check.
- No earlier DSE split is documented in the DSE filings, so the split benchmark set is empty; candidate price breaks are
  stale-price or listing effects.
- Average trade size and retail participation cannot be measured: the public history has no deal counts or investor
  data.
- Quarterly statements are in the BoT regulatory format, which differs from the IFRS annual presentation (fourth-quarter
  values match the audited annual figures for {q4_match:.0%} of lines within 1%; the H1 / 9M values match the held
  interim statements exactly).
- Expected catalyst dates are pattern-based estimates from the 2026 calendar.
"""


def _results(t: dict) -> str:
    out = []
    a = t["ch14_return_attribution.csv"]
    out.append("**1. Return attribution (log points).** " + "; ".join(
        f"{x.bank} {x.horizon}: total {x.log_total_return:+.2f} = EPS {x.eps_growth_log:+.2f} + P/E "
        f"{x.pe_change_log:+.2f} + dividends {x.dividends_log:+.2f}" for x in a.itertuples()) + ".")
    e = t["ch14_event_study.csv"]
    k = e[(e["role"] == "event bank") & e["window"].isin(["[-1,+1]", "[0,+5]"])]
    out.append("**2. Event study (event bank, CAR vs the market proxy, t in brackets).** " + "; ".join(
        f"{x.event_bank} {x.event_date} {x.event[:38]} {x.window}: {x.car_market:+.3f} ({x.car_t_stat:+.1f})"
        for x in k.itertuples()) + ". Benchmark: " + BENCH + ".")
    re_ = t["ch14_results_events.csv"]
    k = re_[re_["include"] & (re_["bank"] == re_["event_bank"])]
    ev1 = re_.drop_duplicates(["event_bank", "event_date"])
    n_in, n_out = int(ev1["include"].sum()), int((~ev1["include"].astype(bool)).sum())
    out.append(f"**2a. Results releases (confirmed dates only; {n_in} kept, {n_out} dropped as estimated or "
               "superseded).** " + "; ".join(
                   f"{x.event_bank} {x.event_date} {x.window}: {x.car_market:+.3f} (t {x.car_t_stat:+.1f})"
                   for x in k.itertuples()) + ".")
    pl = t["ch14_price_lines_scenarios.csv"]
    pl = pl[pl["line"].str.startswith("bear-screen")]
    out.append("**8a. Bear-screen Buy lines (price at which the bear downside is 25%).** " + "; ".join(
        f"{x.bank} {x.valuation}: TZS {x.price_line:,.0f} vs price {x.price:,.0f}" for x in pl.itertuples()) + ".")
    sp = t["ch14_split_study.csv"]
    r = sp[sp["metric"].str.startswith("DiD regression", na=False)]
    d = sp[(sp["metric"] == "mean_turnover_tzs")]
    out.append("**3. NMB split (24 Aug 2026).** Turnover DiD vs CRDB (TZS m/day): " + "; ".join(
        f"+/-{int(x.window_days)}d {x.did_vs_crdb / 1e6:+,.0f}" for x in d.itertuples()) + "; log turnover DiD vs the "
        "market: " + "; ".join(f"+/-{int(x.window_days)}d {x.did_log_vs_market:+.2f}" for x in d.itertuples())
        + ". HAC tests (60 days): " + "; ".join(f"{x.metric[16:]} {x.did_coefficient:+.3f} (t {x.t_stat:+.2f})"
                                                 for x in r.itertuples()) + ".")
    se = t["ch14_seasonality.csv"]
    se = se[se["month"].astype(str).isin(["jan", "dec", "dividend_month"])]
    out.append("**4. Seasonality (monthly, HAC).** " + "; ".join(
        f"{x.series} {x.month} {x.coefficient:+.3f} (t {x.t_stat:+.2f}, n {int(x.n_months)})" for x in se.itertuples())
        + ".")
    c = t["ch14_comovement.csv"]
    out.append("**5. Co-movement.** " + "; ".join(f"{x.pair} {x.frequency}: {x.correlation:.2f}"
                                                  for x in c.head(5).itertuples()) + ".")
    cv = t["ch14_daily_coverage.csv"]
    thin = cv[cv["coverage"] != "adequate"]
    out.append("**5a. Daily-data coverage.** Years with under "
               f"{cv['threshold'].iloc[0]:.0%} non-stale days: " + ("; ".join(
                   f"{b} {_years(list(g['year']))} (non-stale {g['nonstale_share'].min():.0%}-"
                   f"{g['nonstale_share'].max():.0%}, traded {g['traded_share'].min():.0%}-{g['traded_share'].max():.0%})"
                   for b, g in thin.groupby("bank")) or "none")
               + ". Rolling correlations whose window falls below the threshold for either bank are not drawn; daily "
                 "statistics over these years mostly reflect carried-forward prices.")
    m = t["ch14_microstructure.csv"]
    q = m[m["statistic"].isin(["lag 1", "VR(5)"])]
    out.append("**6. Microstructure (descriptive).** " + "; ".join(
        f"{x.bank} {x.statistic} {x.value:.2f} [{x.version.split(' ')[0]}]" for x in q.itertuples()) + ".")
    pc = t["ch14_index_proxy_check.csv"]
    out.append("**7. Proxy check.** " + "; ".join(
        f"{x.proxy[:22]} vs {x.dse_index}: correlation {x.daily_return_correlation:.2f}, cumulative "
        f"{x.cum_log_return_proxy:+.2f} vs {x.cum_log_return_dse:+.2f}" for x in pc.itertuples()) + ".")
    g = t["ch14_implied_growth.csv"]
    out.append("**8. Price-implied terminal growth (v1.3 ROE path held).** " + "; ".join(
        f"{x.bank} {x.valuation}: {x.implied_g:.2%} vs base {x.base_g:.2%}" if np.isfinite(x.implied_g) else
        f"{x.bank} {x.valuation}: {x.status} (value {x.value_at_upper_bound:,.0f} at g {x.g_upper_bound_guard:.1%}; "
        f"{x.value_at_g_minus_10pct:,.0f} at g -10%; price {x.price:,.0f})" for x in g.itertuples()) + ".")
    ai = t["ch14_annual_report_items.csv"]
    out.append(f"**9. Annual-report items.** {int((ai['status'] == 'observed').sum())} values, "
               f"{int((ai.get('page_check', pd.Series(dtype=str)) == 'value found on the cited page').sum())} found on "
               "their cited pages; not disclosed: " + ", ".join(sorted(set(ai.loc[ai['status'].str.startswith('NOT'),
                                                                                   'variable']))) + ".")
    tb = t["ch14_thesis_breaks.csv"]
    out.append("**10. Thesis-break indicators breached.** " + ("; ".join(
        f"{x.bank}: {x.indicator}" for x in tb[tb["status"] == "BREACHED"].itertuples()) or "none") + ".")
    return "\n\n".join(out)


def _horizon_label(h: str) -> str:
    """Tick label for a return-attribution horizon: the start, with the listing year on a second line if it differs."""
    if "listed FY" in h:
        return h.split(" (")[0] + "\n(listed FY" + h.split("listed FY")[1][:4] + ")"
    return h.split(" (")[0]


def _years(ys: list) -> str:
    """Compact year list: 2017-2021, 2023."""
    ys = sorted(int(y) for y in ys)
    out, a = [], ys[0]
    for x, nxt in zip(ys, ys[1:] + [None], strict=True):
        if nxt != x + 1:
            out.append(f"{a}" if a == x else f"{a}-{x}")
            a = nxt
    return ", ".join(out)


def _event_label(event: str, date: str, complete) -> str:
    """Two-line tick label: reporting period and release date; * when the window is cut by the data cutoff."""
    m = re.search(r"-INT-(H1|9M|Q1)(\d{4})", event)
    fy = re.search(r"FY\d{4}", event)
    period = f"{m.group(1)} {m.group(2)}" if m else (fy.group(0) if fy else event[:10])
    star = "*" if complete is not None and not pd.isna(complete) and not bool(complete) else ""
    return f"{period}{star}\n{pd.Timestamp(date):%d %b %y}"


def _figures(t: dict, figd: Path) -> list:
    figs = []
    a = t["ch14_return_attribution.csv"]
    fig, axs = plt_subplots(1, 2, figsize=(10, 3.8))
    for ax, (b, g) in zip(axs, a.dropna(subset=["eps_growth_log"]).groupby("bank", sort=False), strict=False):
        x = np.arange(len(g))
        bottom_p = np.zeros(len(g))
        bottom_n = np.zeros(len(g))
        for col, lab, c in (("eps_growth_log", "EPS growth", "#1baf7a"), ("pe_change_log", "P/E change", "#6250d6"),
                            ("dividends_log", "dividends", "#eda100")):
            v = g[col].fillna(0).to_numpy()
            ax.bar(x, v, bottom=np.where(v >= 0, bottom_p, bottom_n), color=c, label=lab)
            bottom_p += np.clip(v, 0, None)
            bottom_n += np.clip(v, None, 0)
        ax.scatter(x, g["log_total_return"], color=HOUSE[b], edgecolors="#1f1f1d", zorder=3, label="total")
        ax.set_xticks(x, [_horizon_label(h) for h in g["horizon"]], fontsize=7)
        style(ax, f"{b}: sources of total return to FY2025", "log points")
        ax.legend(fontsize=6, frameon=False)
    p = save(fig, figd / "ch14_return_attribution.png", "Evidence class: Derived. Year-end data; dividends via a "
             "total-return index. Horizons without EPS at the start (CRDB since listing, FY2009) are in the table only.")
    d = write_csv(a, figd / "ch14_return_attribution_data.csv", require_evidence=True)
    figs.append((p, d, "Return attribution: EPS growth, P/E change, dividends"))
    gr = t["ch14_roe_lambda_grid.csv"]
    fig, axs = plt_subplots(1, 2, figsize=(10, 3.8))
    for ax, (b, g) in zip(axs, gr.groupby("bank", sort=False), strict=False):
        pv = g.pivot(index="lambda", columns="roe_star", values="value_to_price")
        im = ax.imshow(pv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.7, origin="lower")
        ax.set_xticks(range(pv.shape[1]), [f"{c:.0%}" for c in pv.columns], fontsize=6)
        ax.set_yticks(range(pv.shape[0]), [f"{r:.2f}" for r in pv.index], fontsize=6)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5)
        style(ax, f"{b}: value / price, ROE* x lambda (31 Oct 2026)", "lambda")
        ax.set_xlabel("terminal ROE*", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04)
    p = save(fig, figd / "ch14_roe_lambda_grid.png", "Evidence class: Scenario. Register v1.3, other drivers held.")
    d = write_csv(gr, figd / "ch14_roe_lambda_grid_data.csv", require_evidence=True)
    figs.append((p, d, "Terminal ROE x persistence grid (value / price)"))
    e = t["ch14_results_events.csv"]
    k = e[e["include"].astype(bool) & (e["role"] == "event bank")].copy()
    k["label"] = [_event_label(ev, d, wc) for ev, d, wc in zip(k["event"], k["event_date"], k["window_complete"],
                                                                strict=True)]
    fig, axs = plt_subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, b in zip(axs, ("NMB", "CRDB"), strict=True):
        g = k[k["event_bank"] == b].sort_values("event_date")
        labs = list(dict.fromkeys(g["label"]))
        x = np.arange(len(labs))
        for i, (wdw, alpha) in enumerate((("[0,+5]", 1.0), ("[0,+20]", 0.45))):
            v = g[g["window"] == wdw].set_index("label")["car_market"].reindex(labs).to_numpy() * 100
            ax.bar(x + (i - 0.5) * 0.4, v, width=0.4, color=HOUSE[b], alpha=alpha, label=f"CAR {wdw}")
        ax.set_xticks(x, labs, rotation=0, fontsize=5.6)
        ax.axhline(0, color="#1f1f1d", lw=0.6)
        style(ax, f"{b}: results releases (confirmed dates)", "CAR vs market proxy, percent")
        ax.legend(fontsize=7, frameon=False)
    p = save(fig, figd / "ch14_event_windows.png", "Evidence class: Derived. Confirmed results-release dates only "
             "(filing, posting or press); market model on [-250,-30] against the proxy ex NMB and CRDB. "
             "* window cut by the 30 Oct 2026 data cutoff.")
    d = write_csv(k, figd / "ch14_event_windows_data.csv", require_evidence=True)
    figs.append((p, d, "Results releases: CAR vs the market proxy, by bank"))
    sd = t["_split_daily"]
    fig, axs = plt_subplots(1, 2, figsize=(10, 3.6))
    for b, c in (("NMB", HOUSE["NMB"]), ("CRDB", HOUSE["CRDB"])):
        g = sd[sd["ticker"] == b]
        axs[0].plot(g["trading_date"], g["turnover"].rolling(5, min_periods=1).mean() / 1e9, color=c, lw=1, label=b)
        axs[1].plot(g["trading_date"], g["adj_volume"].rolling(5, min_periods=1).mean() / 1e6, color=c, lw=1, label=b)
    for ax, ttl, yl in ((axs[0], "TZS turnover (5-day mean)", "TZS bn"),
                        (axs[1], "Split-adjusted shares traded (5-day mean)", "million shares")):
        ax.axvline(pd.Timestamp("2026-08-24"), color="#1f1f1d", lw=0.7, ls="--")
        style(ax, ttl, yl)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
    p = save(fig, figd / "ch14_split_study.png", "Evidence class: Derived. NMB 1:10 split effective 24 Aug 2026 "
                                                 "(dashed); +/-60 trading days; CRDB as control.")
    d = write_csv(sd, figd / "ch14_split_study_data.csv", require_evidence=True)
    figs.append((p, d, "NMB split: turnover and shares traded, NMB vs CRDB"))
    se = t["ch14_seasonality.csv"]
    mm = se[se["mean_return"].notna()].copy()
    fig, ax = plt_subplots(figsize=(8, 3.4))
    for i, sname in enumerate(("NMB", "CRDB", "market proxy")):
        c = HOUSE[sname]
        g = mm[mm["series"] == sname]
        ax.bar(g["month"].astype(int) + (i - 1) * 0.27, g["mean_return"], width=0.27, color=c, label=sname)
    ax.axhline(0, color="#1f1f1d", lw=0.6)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    style(ax, "Mean monthly return by calendar month, 2015-2026", "return")
    ax.legend(fontsize=7, frameon=False)
    p = save(fig, figd / "ch14_seasonality.png", "Evidence class: Derived (means); tests in the table (HAC).")
    d = write_csv(mm, figd / "ch14_seasonality_data.csv", require_evidence=True)
    figs.append((p, d, "Seasonality: mean monthly returns"))
    rc = t["ch14_rolling_correlation.csv"]
    cov = t["ch14_daily_coverage.csv"]
    fig, ax = plt_subplots(figsize=(8, 3.4))
    for wdw, c, lw in (("60 trading days (daily returns)", "#9a9a94", 0.6), ("12 months (monthly returns)",
                                                                          "#1f1f1d", 1.4)):
        g = rc[rc["window"] == wdw]
        ax.plot(pd.to_datetime(g["trading_date"]), g["correlation"].where(g["coverage_ok"].astype(bool)), color=c,
                lw=lw, label=wdw)
    low = cov[cov["nonstale_share"] < cov["threshold"]]
    for b, yy in low.groupby("bank")["year"]:
        for y in yy:
            ax.axvspan(pd.Timestamp(f"{int(y)}-01-01"), pd.Timestamp(f"{int(y)}-12-31"), ymin=0 if b == "NMB" else 0.5,
                       ymax=0.5 if b == "NMB" else 1, color=HOUSE[b], alpha=0.12, lw=0)
    gaps = "; ".join(f"{b} {_years(list(yy))}" for b, yy in low.groupby("bank")["year"])
    ax.axhline(0, color="#1f1f1d", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style(ax, "NMB-CRDB rolling return correlation (breaks where daily data are thin)", "correlation")
    if gaps:
        ax.text(0.01, 0.99, f"Shaded: under {low['threshold'].iloc[0]:.0%} non-stale days ({gaps}); NMB lower half, "
                "CRDB upper half", transform=ax.transAxes, fontsize=6, va="top", color="#5a5a55")
    p = save(fig, figd / "ch14_rolling_correlation.png", "Evidence class: Derived. Log returns, all days; monthly "
             "correlation over the trailing 12 months; a window is drawn only when both banks have at least the "
             "threshold share of non-stale days in it (dse.library.min_nonstale_share).")
    d = write_csv(rc, figd / "ch14_rolling_correlation_data.csv", require_evidence=True)
    figs.append((p, d, "NMB-CRDB rolling correlation"))
    return figs


def _split_daily(root: Path) -> pd.DataFrame:
    p = M.panel(root)
    d0 = pd.Timestamp("2026-08-24")
    out = []
    for _, g in p.groupby("ticker"):
        g = g.reset_index(drop=True)
        e0 = int(g.index[g["trading_date"] >= d0][0])
        out.append(g.iloc[max(0, e0 - 60):e0 + 60][["ticker", "trading_date", "turnover", "adj_volume", "traded",
                                                     "stale", "block_proxy"]])
    return pd.concat(out, ignore_index=True).assign(evidence_class="Derived")


def _q4_match(root: Path) -> tuple[float, Path]:
    """Share of fourth-quarter BoT-format lines within 1% of the audited annual values (parser cross-check)."""
    from tzbank.extract import quarterly as Q

    e = Q.extract_all(root)
    rv = e[e["status"] != "accepted"].assign(evidence_class="Observed")
    write_csv(rv, path("validation", root) / "quarterly_extraction_review.csv", require_evidence=True)
    c = Q.crosscheck(e, root).assign(evidence_class="Derived")
    f = write_csv(c, path("validation", root) / "quarterly_extraction_crosscheck.csv", require_evidence=True)
    h = c[c["sheet"] == "historical_financials"]
    return (float(h["match_1pct"].mean()) if len(h) else float("nan")), f


def write_ch14(ctx) -> list[Path]:
    root = ctx.root
    tab, figd = path("tables", root), path("figures", root)
    ve = valuation_extensions(ctx)
    split = M.split_study(root)
    co, roll = M.comovement(root)
    qual = M.quality(root)
    tables = [
        (return_attribution(root), "ch14_return_attribution.csv", "1. Long-horizon return attribution", "Derived"),
        (event_study(root), "ch14_event_study.csv", "2. Event study: CARs vs the market proxy", "Derived"),
        (split, "ch14_split_study.csv", "3. NMB split: DiD vs CRDB and the market (+/-5, 20, 60 days)", "Derived"),
        (M.split_objectives(root, split), "ch14_split_objectives.csv", "3. NMB split: stated objectives vs outcomes",
         "Derived"),
        (M.split_benchmarks(root), "ch14_split_benchmarks.csv", "3. Earlier DSE splits (benchmark set)", "Observed"),
        (seasonality(root), "ch14_seasonality.csv", "4. Seasonality 2015-2026 (monthly; HAC tests)", "Derived"),
        (M.results_events(root), "ch14_results_events.csv",
         "2. Results releases with confirmed dates: CARs vs the market proxy, [0,+5] and [0,+20]", "Derived"),
        (co, "ch14_comovement.csv", "5. NMB-CRDB co-movement (daily, weekly, monthly, Dimson)", "Derived"),
        (roll, "ch14_rolling_correlation.csv", "5. Rolling correlation (60 days daily; 12 months monthly)",
         "Derived"),
        (M.daily_coverage(root), "ch14_daily_coverage.csv", "5. Daily-data coverage by bank and year (non-stale days)",
         "Derived"),
        (microstructure(root), "ch14_microstructure.csv", "6. Microstructure (unfiltered and stale-filtered)",
         "Derived"),
        (M.proxy_check(root), "ch14_index_proxy_check.csv", "7. Market proxy vs DSE indices", "Derived"),
        (qual, "ch14_data_quality.csv", "7. DSE library vs the owner's daily file; block matching", "Derived"),
        (M.quarterly_fundamentals(root), "ch14_quarterly_fundamentals.csv",
         "7. Quarterly NIM, cost of risk, profit, deposits, loans", "Derived"),
        (ve["implied_growth"], "ch14_implied_growth.csv", "8. Price-implied terminal growth", "Scenario"),
        (ve["price_lines_scenarios"], "ch14_price_lines_scenarios.csv", "8. Bear / base / bull price lines",
         "Scenario"),
        (ve["roe_lambda_grid"], "ch14_roe_lambda_grid.csv", "8. Terminal ROE x persistence grid", "Scenario"),
        (ve["real_roe_coe"], "ch14_real_roe_coe.csv", "8. Real ROE vs real COE (IMF CPI)", "Derived"),
        (ve["yield_minus_tbill"], "ch14_yield_minus_tbill.csv", "8. Dividend yield minus the T-bill", "Derived"),
        (annual_report_items(root), "ch14_annual_report_items.csv", "9. Annual-report items FY2023-25 (page-checked)",
         "Observed"),
        (catalysts(root), "ch14_catalysts.csv", "10. Catalyst calendar", "Observed"),
        (thesis_breaks(root), "ch14_thesis_breaks.csv", "10. Thesis-break indicators", "Derived"),
    ]
    files = [write_csv(ctx.stamp(df), tab / n, require_evidence=True) for df, n, _, _ in tables]
    t = {n: df for df, n, _, _ in tables}
    t["_split_daily"] = _split_daily(root)
    figs = _figures(t, figd)
    files += [p for p, _, _ in figs]
    q4, qf = _q4_match(root)
    files.append(qf)
    files.append(write_csv(qual, path("validation", root) / "dse_library_reconciliation.csv", require_evidence=True))
    nf = write_text(NOTE.format(version=ctx.version, hash=ctx.hash, results=_results(t), q4_match=q4,
                                cutoff=settings(root)["dse"]["library"]["cutoff"]), path("notes", root) / "ch14.md")
    files.append(nf)
    m = {"chapter": "14", "title": "Market behaviour and valuation extensions", "hypotheses": [],
         "status": "built - prompts 14-15 (no register changes)", "register_version": ctx.version,
         "input_hash": ctx.hash,
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for _, n, d, c in tables],
         "figures": [{"path": rel(p, root), "data": rel(dd, root), "description": desc, "evidence_class": "Derived",
                      "note": rel(nf, root)} for p, dd, desc in figs],
         "notes": [rel(nf, root)], "module": "tzbank.chapters.ch14_market_extensions", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch14.json"))
    return files
