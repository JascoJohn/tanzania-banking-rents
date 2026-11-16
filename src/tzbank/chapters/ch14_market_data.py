"""Chapter 14 market analyses on the DSE data library (prompt 15 item 5; D72). Data: ``data/library/dse/`` with the
``dse.library.cutoff`` (30 Oct 2026). Every row carries an evidence class and a ``reading`` (descriptive or test).

* ``panel``             daily panel per ticker: split-adjusted close, traded / stale flags, returns on all days and on
                        non-stale observations only (consecutive non-stale closes; multi-day gaps allowed).
* ``microstructure``    autocorrelation and variance ratios, unfiltered (labelled as affected by stale prices) and
                        stale-filtered; block vs board days; clustering; 52-week high; turnover spikes.
* ``event_study``       abnormal returns and CARs against the market proxy ex NMB and CRDB (market model on [-250,-30]
                        or market-adjusted when < 100 estimation days) and against the other bank.
* ``split_study``       NMB 1:10 split: pre / post windows +/-5, 20, 60 days; DiD against CRDB and the market proxy
                        (HAC regression for the 60-day window); liquidity measures incl. Roll and Corwin-Schultz spreads.
* ``seasonality``       monthly returns and turnover 2015-2026; January / December dummies with Newey-West (HAC)
                        errors; block share by month; dividend-month dummy.
* ``comovement``        NMB-CRDB daily and weekly return correlation, rolling 60-day correlation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import settings

FOCUS = ("NMB", "CRDB")
PROXY = "all equities ex NMB and CRDB (event-study market proxy)"


def _lib(root: Path) -> Path:
    return root / "data" / "library" / "dse"


def cutoff(root: Path) -> pd.Timestamp:
    return pd.Timestamp(settings(root)["dse"]["library"]["cutoff"])


def panel(root: Path, tickers=FOCUS) -> pd.DataFrame:
    d = pd.read_csv(_lib(root) / "daily_prices.csv", parse_dates=["trading_date"], low_memory=False)
    d = d[d["ticker"].isin(tickers) & (d["trading_date"] <= cutoff(root))].sort_values(["ticker", "trading_date"])
    d["traded"] = d["volume"].fillna(0) > 0
    d["stale"] = d["stale"].astype(str).str.lower().eq("true")
    d["block_proxy"] = d["block_proxy"].astype(str).str.lower().eq("true")
    d["lr_all"] = np.log(d["adj_close"]).groupby(d["ticker"]).diff()
    ok = d[~d["stale"]].copy()
    ok["lr_nonstale"] = np.log(ok["adj_close"]).groupby(ok["ticker"]).diff()
    d = d.merge(ok[["ticker", "trading_date", "lr_nonstale"]], on=["ticker", "trading_date"], how="left")
    return d.reset_index(drop=True)


def market(root: Path, name: str = PROXY) -> pd.Series:
    i = pd.read_csv(_lib(root) / "index_proxy.csv", parse_dates=["trading_date"])
    i = i[(i["index"] == name) & (i["trading_date"] <= cutoff(root))].set_index("trading_date")
    return np.log1p(i["daily_return"])


# --- data quality (item 3) ------------------------------------------------------------------------------------------
def quality(root: Path) -> pd.DataFrame:
    """Reconcile the library's NMB / CRDB rows with the owner's daily file (data/processed/market_daily.csv) and match
    off-range VWAP rows (turnover / volume outside low-high) against the held block records (its data_flag)."""
    lib = pd.read_csv(_lib(root) / "daily_prices.csv", low_memory=False)
    lib = lib[lib["ticker"].isin(FOCUS)].rename(columns={"trading_date": "date"})
    own = pd.read_csv(root / "data" / "processed" / "market_daily.csv")
    m = own.merge(lib, on=["ticker", "date"], how="outer", suffixes=("_own", "_lib"), indicator=True)
    rows = []

    def add(bank, check, value, n, note=""):
        rows.append({"bank": bank, "check": check, "value": value, "n": n, "note": note, "reading": "descriptive",
                     "evidence_class": "Derived"})

    for b in FOCUS:
        g = m[m["ticker"] == b]
        rng = g.loc[g["_merge"] != "right_only", "date"]
        g = g[(g["date"] >= rng.min()) & (g["date"] <= rng.max())]
        both = g[g["_merge"] == "both"]
        add(b, "dates in the owner's daily file", float(len(own[own["ticker"] == b])), len(g),
            f"{rng.min()} to {rng.max()}")
        add(b, "dates in both", float(len(both)), len(g))
        add(b, "dates only in the owner's file", float((g["_merge"] == "left_only").sum()), len(g))
        add(b, "dates only in the DSE library", float((g["_merge"] == "right_only").sum()), len(g))
        for col in ("close", "volume", "turnover", "high", "low"):
            eq = np.isclose(both[f"{col}_own"].astype(float), both[f"{col}_lib"].astype(float), rtol=1e-6)
            add(b, f"{col}: share of common dates equal", float(eq.mean()) if len(both) else np.nan, len(both),
                "; ".join(both.loc[~eq, "date"].head(5)))
        flag = both["validation_flags"].fillna("").str.contains("vwap_outside")
        held = both["data_flag"].fillna("").str.contains("block|negotiated|VWAP", case=False)
        add(b, "off-range VWAP rows (DSE library)", float(flag.sum()), len(both))
        add(b, "held block records (owner's data_flag)", float(held.sum()), len(both))
        add(b, "match rate: off-range VWAP rows that are held block records", float((flag & held).sum() / flag.sum())
            if flag.sum() else np.nan, int(flag.sum()), "; ".join(both.loc[flag & ~held, "date"].head(8)))
        add(b, "held block records detected by the off-range VWAP rule", float((flag & held).sum() / held.sum())
            if held.sum() else np.nan, int(held.sum()), "; ".join(both.loc[held & ~flag, "date"].head(8)))
        vw = both["turnover_lib"].astype(float) / both["volume_lib"].astype(float)
        far = (vw < 0.95 * both["low_lib"].astype(float)) | (vw > 1.05 * both["high_lib"].astype(float))
        add(b, "off-range VWAP rows, more than 5% outside low-high", float(far.sum()), len(both))
        add(b, "match rate at the 5% tolerance (the owner's flag rule)", float((far & held).sum() / far.sum())
            if far.sum() else np.nan, int(far.sum()), "; ".join(both.loc[far & ~held, "date"].head(8)))
    # year-end prices in the panel (inputs workbook / Phase 1) against the last DSE close of each year
    w = pd.read_csv(root / "data" / "processed" / "bank_year_wide.csv")
    for b, t in (("B01", "CRDB"), ("B02", "NMB")):
        g = lib[lib["ticker"] == t].sort_values("date")
        last = g.groupby(g["date"].str[:4])["close"].last()
        for _, r in w[(w["bank_id"] == b) & w["share_price_close"].notna()].iterrows():
            y = str(int(r["fiscal_year"]))
            if y in last.index and int(y) <= 2025:
                add(t, f"year-end price FY{y}: panel vs last DSE close", float(r["share_price_close"]), 1,
                    f"DSE {last[y]:,.0f}; " + ("equal" if np.isclose(r["share_price_close"], last[y]) else
                                               "DIFFERENT (panel value as reported; see the source)"))
    n = lib[lib["ticker"] == "NMB"].sort_values("date").set_index("date")["close"]
    i = int(np.searchsorted(n.index.to_numpy(), "2026-08-24"))
    if 0 < i < len(n):
        add("NMB", "split: close before / close on the effective date", float(n.iloc[i - 1] / n.iloc[i]), 1,
            f"{n.index[i - 1]} {n.iloc[i - 1]:,.0f} -> {n.index[i]} {n.iloc[i]:,.0f}; a 1:10 split implies about 10")
    return pd.DataFrame(rows)


def proxy_check(root: Path) -> pd.DataFrame:
    """The capitalisation-weighted proxies against the DSE-published indices over the year the indices page embeds:
    daily return correlation, tracking error and cumulative log return (DSEI for all equities, TSI for domestic)."""
    h = pd.read_csv(_lib(root) / "index_history.csv", parse_dates=["trading_date"])
    p = pd.read_csv(_lib(root) / "index_proxy.csv", parse_dates=["trading_date"])
    rows = []
    for proxy, off in (("all equities (DSEI-like)", "DSEI"), ("domestic equities (TSI-like)", "TSI"),
                       (PROXY, "DSEI"), (PROXY, "TSI")):
        a = p[p["index"] == proxy].set_index("trading_date")["level"]
        b = h[h["index"] == off].set_index("trading_date")["level"]
        j = pd.concat([a, b], axis=1, keys=["proxy", "dse"]).dropna()
        r = np.log(j).diff().dropna()
        rows.append({"proxy": proxy, "dse_index": off, "first": str(j.index.min().date()),
                     "last": str(j.index.max().date()), "days": len(r),
                     "daily_return_correlation": float(r.corr().iloc[0, 1]),
                     "tracking_error_daily": float((r["proxy"] - r["dse"]).std()),
                     "cum_log_return_proxy": float(r["proxy"].sum()), "cum_log_return_dse": float(r["dse"].sum()),
                     "reading": "descriptive check of a Derived proxy against Observed DSE index levels",
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def quarterly_fundamentals(root: Path) -> pd.DataFrame:
    """Quarterly NIM (as published), cost of risk, profit, deposits and loans from the BoT-format quarterly statements
    (inputs workbook Quarterly Results and Interim Results; cumulative for the period)."""
    parts = []
    for f in ("quarterly_long.csv", "interim_long.csv"):
        x = pd.read_csv(root / "data" / "processed" / f, parse_dates=["period_end"], low_memory=False)
        parts.append(x.assign(sheet=f.split("_")[0]))
    x = pd.concat(parts, ignore_index=True)
    x = x[x["bank_id"].isin(["B01", "B02"]) & x["value"].notna()]
    keep = {"nim_reported_bot": "nim_reported", "net_profit": "net_profit_ytd", "impairment_charges":
            "impairment_ytd", "net_loans": "net_loans", "customer_deposits": "customer_deposits",
            "net_interest_income": "nii_ytd", "total_assets": "total_assets"}
    x = x[x["canonical_name"].isin(keep)]
    x = x.sort_values("basis").drop_duplicates(["bank_id", "period_end", "canonical_name"])  # GROUP first
    w = x.pivot_table(index=["bank_id", "period_end", "period_type"], columns="canonical_name", values="value",
                      aggfunc="first").rename(columns=keep).reset_index()
    src = x.groupby(["bank_id", "period_end"])["sheet"].first().rename("source_sheet")
    w = w.merge(src, on=["bank_id", "period_end"], how="left")
    q = w["period_end"].dt.month // 3
    w["months"] = q * 3
    loans = w["net_loans"].where(w["net_loans"] > 0)
    w["cost_of_risk_annualised"] = w["impairment_ytd"] * (12 / w["months"]) / loans
    w["roa_annualised"] = w["net_profit_ytd"] * (12 / w["months"]) / w["total_assets"]
    w["bank"] = w["bank_id"].map({"B01": "CRDB", "B02": "NMB"})
    w["definition"] = ("cost of risk = YTD impairment x 12 / months / period-end net loans (approximate: period-end, "
                       "not average, loans); NIM as published (net interest income to average earning assets)")
    w["evidence_class"] = "Derived"
    return w.sort_values(["bank", "period_end"]).reset_index(drop=True)


def split_objectives(root: Path, split: pd.DataFrame) -> pd.DataFrame:
    """NMB's stated split objectives (DSE notice, 27 Jul 2026) against the measured outcomes (60-day windows)."""
    s60 = split[(split["window_days"] == 60) & split["metric"].isin(["trading_day_share", "mean_turnover_tzs",
                                                                     "mean_shares_traded_adj", "amihud_per_tzs_bn",
                                                                     "roll_spread", "corwin_schultz_spread",
                                                                     "block_share_of_turnover"])].set_index("metric")
    p = panel(root, ("NMB",)).set_index("trading_date")
    pre_px = float(p.loc[p.index < "2026-08-24", "close"].iloc[-1])
    post_px = float(p.loc[p.index >= "2026-08-24", "close"].iloc[0])

    def did(m):
        return float(s60.loc[m, "did_vs_crdb"]) if m in s60.index else np.nan

    rows = [
        ("improve affordability", "price per share (TZS)", pre_px, post_px, np.nan,
         "mechanical: one share now costs a tenth (DSE minimum lot unchanged)"),
        ("improve tradability", "trading-day share", s60.loc["trading_day_share", "nmb_pre"],
         s60.loc["trading_day_share", "nmb_post"], did("trading_day_share"), "DiD vs CRDB, 60 trading days"),
        ("improve tradability", "mean daily TZS turnover", s60.loc["mean_turnover_tzs", "nmb_pre"],
         s60.loc["mean_turnover_tzs", "nmb_post"], did("mean_turnover_tzs"), "DiD vs CRDB, 60 trading days"),
        ("improve tradability", "Amihud illiquidity (|r| per TZS bn)", s60.loc["amihud_per_tzs_bn", "nmb_pre"],
         s60.loc["amihud_per_tzs_bn", "nmb_post"], did("amihud_per_tzs_bn"), "lower = more liquid; stale-filtered"),
        ("improve tradability", "Corwin-Schultz spread", s60.loc["corwin_schultz_spread", "nmb_pre"],
         s60.loc["corwin_schultz_spread", "nmb_post"], did("corwin_schultz_spread"), "stale-filtered"),
        ("broaden retail participation", "split-adjusted shares traded per day",
         s60.loc["mean_shares_traded_adj", "nmb_pre"], s60.loc["mean_shares_traded_adj", "nmb_post"],
         did("mean_shares_traded_adj"), "proxy only: investor-level and deal-count data are not public"),
        ("broaden retail participation", "block share of turnover", s60.loc["block_share_of_turnover", "nmb_pre"],
         s60.loc["block_share_of_turnover", "nmb_post"], did("block_share_of_turnover"),
         "a falling block share is consistent with more board (smaller) trades"),
        ("broaden retail participation", "average trade size", np.nan, np.nan, np.nan,
         "NOT AVAILABLE: the DSE daily history carries no deal counts"),
    ]
    return pd.DataFrame(rows, columns=["stated_objective", "measure", "pre_60d", "post_60d", "did_vs_crdb", "note"]
                        ).assign(source="DSE filing: NMB public notice, share split (27 Jul 2026)",
                                 reading="descriptive (window means)", evidence_class="Derived")


def split_benchmarks(root: Path) -> pd.DataFrame:
    """Earlier DSE splits as a benchmark set: confirmed actions in the library and the candidate price breaks."""
    ca = pd.read_csv(_lib(root) / "corporate_actions.csv")
    ca = ca[ca["ticker"] != "NMB"].copy()
    ca["benchmark_use"] = np.where(ca["status"].str.startswith("confirmed"), "usable",
                                   "not usable: no split notice among the DSE documents (stale-price or listing break)")
    ca["reading"] = "descriptive"
    return ca


# --- microstructure -----------------------------------------------------------------------------------------------
def _vr(r: pd.Series, q: int) -> float:
    r = r.dropna()
    return float(r.rolling(q).sum().dropna().var() / (q * r.var())) if len(r) > q + 10 else np.nan


def microstructure(root: Path) -> pd.DataFrame:
    p = panel(root)
    rows = []

    def add(bank, test, stat, value, n, version, note=""):
        rows.append({"bank": bank, "test": test, "statistic": stat, "version": version, "value": value, "n": n,
                     "note": note, "reading": "descriptive", "evidence_class": "Derived"})

    for b, g in p.groupby("ticker", sort=False):
        for col, ver in (("lr_all", "unfiltered (affected by stale prices: carried-forward closes enter as zero "
                                    "returns)"), ("lr_nonstale", "stale-filtered (consecutive non-stale closes)")):
            r = g[col].dropna()
            for k in range(1, 6):
                add(b, "autocorrelation", f"lag {k}", float(r.autocorr(k)), len(r), ver)
            for q in (2, 5, 10, 20):
                add(b, "variance ratio", f"VR({q})", _vr(r, q), len(r), ver)
        t = g[g["traded"] & ~g["stale"]].copy()
        t["next"] = t["lr_nonstale"].shift(-1)
        for lab, m in (("block days", t["block_proxy"]), ("board-only days", ~t["block_proxy"])):
            add(b, "block vs board", f"mean return, {lab}", float(t.loc[m, "lr_nonstale"].mean()), int(m.sum()),
                "stale-filtered")
            add(b, "block vs board", f"mean next-observation return, {lab}", float(t.loc[m, "next"].mean()),
                int(m.sum()), "stale-filtered")
        c = g.loc[g["traded"], "close"].astype(float)
        for lab, mod in (("close ends in 0", 10), ("close ends in 00", 100), ("close ends in 50 or 00", 50)):
            add(b, "price clustering", f"share of traded days: {lab}", float((c % mod == 0).mean()), len(c), "raw close")
        px = g.set_index("trading_date")["adj_close"]
        hi = px.rolling(250, min_periods=120).max()
        near = (px >= 0.98 * hi) & hi.notna()
        f5 = np.log(px.shift(-5) / px)
        add(b, "52-week high", "share of days within 2% of the running high", float(near[hi.notna()].mean()),
            int(hi.notna().sum()), "all days")
        add(b, "52-week high", "mean next-5-day log return, near the high", float(f5[near].mean()), int(near.sum()),
            "all days")
        add(b, "52-week high", "mean next-5-day log return, not near", float(f5[~near & hi.notna()].mean()),
            int((~near & hi.notna()).sum()), "all days")
        tr = g[g["traded"]].copy()
        med = tr["turnover"].rolling(20, min_periods=10).median().shift(1)
        spike = tr["turnover"] > 3 * med
        add(b, "turnover spikes", "mean |return|, spike days (> 3x trailing 20-day median)",
            float(tr.loc[spike, "lr_nonstale"].abs().mean()), int(spike.sum()), "stale-filtered")
        add(b, "turnover spikes", "mean |return|, other traded days", float(tr.loc[~spike, "lr_nonstale"].abs().mean()),
            int((~spike).sum()), "stale-filtered")
    return pd.DataFrame(rows)


# --- spreads and liquidity ------------------------------------------------------------------------------------------
def roll_spread(px: pd.Series) -> float:
    """Roll (1984): 2 sqrt(-cov(dp_t, dp_t-1)) as a share of price; NaN when the covariance is positive."""
    dp = np.log(px).diff().dropna()
    if len(dp) < 10:
        return np.nan
    cv = np.cov(dp.iloc[1:], dp.iloc[:-1])[0, 1]
    return float(2 * np.sqrt(-cv)) if cv < 0 else np.nan


def corwin_schultz(high: pd.Series, low: pd.Series) -> float:
    """Corwin and Schultz (2012) high-low spread, two-day estimator averaged; negative estimates set to zero."""
    h, lo = np.log(high.to_numpy(float)), np.log(low.to_numpy(float))
    beta = (h[1:] - lo[1:]) ** 2 + (h[:-1] - lo[:-1]) ** 2
    gamma = (np.maximum(h[1:], h[:-1]) - np.minimum(lo[1:], lo[:-1])) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0), np.nan)
    return float(np.nanmean(s)) if len(s) else np.nan


def liquidity(g: pd.DataFrame) -> dict:
    t = g[g["traded"] & ~g["stale"]]
    r = t["lr_nonstale"].abs()
    tot = g["turnover"].sum()
    return {"days": len(g), "trading_day_share": float(g["traded"].mean()),
            "mean_turnover_tzs": float(g["turnover"].mean()), "mean_shares_traded_adj": float(g["adj_volume"].mean()),
            "block_share_of_turnover": float(g.loc[g["block_proxy"], "turnover"].sum() / tot) if tot else np.nan,
            "amihud_per_tzs_bn": float((r / (t["turnover"] / 1e9)).replace(np.inf, np.nan).mean()),
            "roll_spread": roll_spread(t["adj_close"]), "corwin_schultz_spread":
                corwin_schultz(t["adj_high"], t["adj_low"]) if len(t) > 2 else np.nan}


# --- event study ----------------------------------------------------------------------------------------------------
def _event_index(dates: pd.DatetimeIndex, d: pd.Timestamp) -> int | None:
    idx = np.where(dates >= d)[0]
    return int(idx[0]) if len(idx) else None


def event_study(root: Path, events: pd.DataFrame) -> pd.DataFrame:
    p = panel(root)
    m = market(root)
    ret = p.pivot_table(index="trading_date", columns="ticker", values="lr_all").sort_index()
    ret["MKT"] = m.reindex(ret.index)
    dates = ret.index
    rows = []
    for e in events.itertuples():
        e0 = _event_index(dates, pd.Timestamp(e.event_date))
        if e0 is None:
            continue
        for b in FOCUS:
            other = [x for x in FOCUS if x != b][0]
            est = ret.iloc[max(0, e0 - 250):max(0, e0 - 30)][[b, "MKT"]].dropna()
            if len(est) >= 100:
                beta, alpha = np.polyfit(est["MKT"], est[b], 1)
                resid_sd = float(np.std(est[b] - (alpha + beta * est["MKT"]), ddof=2))
                model = "market model on [-250,-30]"
            else:
                alpha, beta, resid_sd = 0.0, 1.0, float(np.std(est[b] - est["MKT"], ddof=1)) if len(est) > 5 else np.nan
                model = f"market-adjusted (only {len(est)} estimation days)"
            for lab, lo, hi in (("[-20,-1]", -20, -1), ("[0,+20]", 0, 20), ("[-5,-1]", -5, -1), ("[0,+5]", 0, 5),
                                ("[-1,+1]", -1, 1)):
                w = ret.iloc[max(0, e0 + lo):min(len(ret), e0 + hi + 1)]
                ar = w[b] - (alpha + beta * w["MKT"])
                car = float(ar.sum())
                n = int(ar.notna().sum())
                rows.append({"event_bank": e.bank, "event": e.event, "event_date": e.event_date, "kind": e.kind,
                             "bank": b, "role": "event bank" if b == e.bank else "other bank",
                             "event_day": str(dates[e0].date()), "window": lab, "days": n,
                             "raw_log_return": float(w[b].sum()), "market_log_return": float(w["MKT"].sum()),
                             "car_market": car, "car_t_stat": car / (resid_sd * np.sqrt(n)) if resid_sd and n else np.nan,
                             "car_vs_other_bank": float((w[b] - w[other]).sum()), "model": model,
                             "alpha": alpha, "beta": beta, "benchmark": PROXY, "reading":
                                 "case study; t = CAR / (estimation-residual sd x sqrt(days)), reported, not a "
                                 "sample-wide test", "source_confidence": getattr(e, "confidence", ""),
                             "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- split study ----------------------------------------------------------------------------------------------------
def split_study(root: Path, split_date: str = "2026-08-24", bank: str = "NMB", controls=("CRDB",)) -> pd.DataFrame:
    import statsmodels.api as sm

    p = panel(root)
    mkt_turn = pd.read_csv(_lib(root) / "daily_prices.csv", parse_dates=["trading_date"], low_memory=False)
    mkt_turn = mkt_turn[(~mkt_turn["ticker"].isin(FOCUS)) & (mkt_turn["trading_date"] <= cutoff(root))]
    mkt = mkt_turn.groupby("trading_date").agg(turnover=("turnover", "sum"), volume=("volume", "sum")).sort_index()
    d0 = pd.Timestamp(split_date)
    rows = []
    groups = {b: g.set_index("trading_date") for b, g in p.groupby("ticker")}
    dates = groups[bank].index
    e0 = _event_index(dates, d0)
    for k in (5, 20, 60):
        pre_d, post_d = dates[max(0, e0 - k):e0], dates[e0:e0 + k]
        res = {}
        for name in (bank, *controls):
            g = groups[name]
            res[name] = {"pre": liquidity(g.loc[g.index.isin(pre_d)].reset_index()),
                         "post": liquidity(g.loc[g.index.isin(post_d)].reset_index())}
        mp, mq = mkt.loc[mkt.index.isin(pre_d)], mkt.loc[mkt.index.isin(post_d)]
        for metric in res[bank]["pre"]:
            pre, post = res[bank]["pre"][metric], res[bank]["post"][metric]
            row = {"window_days": k, "metric": metric, "nmb_pre": pre, "nmb_post": post, "nmb_change": post - pre}
            for c in controls:
                cp, cq = res[c]["pre"][metric], res[c]["post"][metric]
                row |= {f"{c.lower()}_pre": cp, f"{c.lower()}_post": cq, f"did_vs_{c.lower()}": (post - pre) - (cq - cp)}
            if metric == "mean_turnover_tzs":
                row |= {"market_pre": float(mp["turnover"].mean()), "market_post": float(mq["turnover"].mean()),
                        "did_log_vs_market": float(np.log(post / pre) - np.log(mq["turnover"].mean() /
                                                                              mp["turnover"].mean()))}
            rows.append(row | {"reading": "descriptive (difference-in-differences of window means)",
                               "evidence_class": "Derived"})
        if k == 60:  # HAC regression: log turnover and trading-day share, treat x post
            stack = []
            for name in (bank, *controls):
                g = groups[name]
                g = g.loc[g.index.isin(pre_d.union(post_d))].copy()
                g["treat"] = int(name == bank)
                g["post"] = g.index.isin(post_d).astype(int)
                stack.append(g)
            s = pd.concat(stack)
            for y, lab in ((np.log1p(s["turnover"]), "log(1 + TZS turnover)"),
                           (s["traded"].astype(float), "trading-day indicator"),
                           (np.log1p(s["adj_volume"]), "log(1 + split-adjusted shares traded)")):
                X = sm.add_constant(pd.DataFrame({"treat": s["treat"], "post": s["post"],
                                                  "treat_x_post": s["treat"] * s["post"]}).astype(float))
                f = sm.OLS(y.astype(float).to_numpy(), X.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
                rows.append({"window_days": 60, "metric": f"DiD regression: {lab}", "did_coefficient": float(f.params[3]),
                             "hac_se": float(f.bse[3]), "t_stat": float(f.tvalues[3]), "p_value": float(f.pvalues[3]),
                             "n": int(f.nobs), "reading": "test: OLS with Newey-West (5 lags) errors, NMB vs CRDB",
                             "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- seasonality ----------------------------------------------------------------------------------------------------
def seasonality(root: Path, dividend_months: dict[str, set] | None = None) -> pd.DataFrame:
    import statsmodels.api as sm

    p = panel(root)
    m = market(root)
    mm = np.exp(m.groupby(m.index.to_period("M")).sum()) - 1
    rows = []
    series = {b: g.set_index("trading_date") for b, g in p.groupby("ticker")}
    for name in (*FOCUS, "market proxy"):
        if name == "market proxy":
            r = mm
            turn = None
            block = None
        else:
            g = series[name]
            px = g["adj_close"].groupby(g.index.to_period("M")).last()
            r = px.pct_change()
            turn = g["turnover"].groupby(g.index.to_period("M")).sum()
            block = (g.loc[g["block_proxy"], "turnover"].groupby(g.loc[g["block_proxy"]].index.to_period("M")).sum()
                     / turn).reindex(turn.index).fillna(0)
        r = r.dropna()
        r = r[(r.index >= pd.Period("2015-02", "M"))]
        yrs = r.index.year.nunique()
        for mo in range(1, 13):
            sel = r[r.index.month == mo]
            row = {"series": name, "month": mo, "mean_return": float(sel.mean()), "n_months": len(sel),
                   "reading": "descriptive", "evidence_class": "Derived"}
            if turn is not None:
                t = turn[turn.index.month == mo]
                row |= {"mean_turnover_tzs": float(t.mean()), "mean_block_share": float(block[block.index.month == mo]
                                                                                       .mean())}
            rows.append(row)
        X = pd.DataFrame({"jan": (r.index.month == 1).astype(float), "dec": (r.index.month == 12).astype(float)},
                         index=r.index)
        if dividend_months and name in dividend_months:
            X["dividend_month"] = [float(mo in dividend_months[name]) for mo in r.index.month]
        f = sm.OLS(r.to_numpy(float), sm.add_constant(X).to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        for i, c in enumerate(["const", *X.columns]):
            rows.append({"series": name, "month": c, "coefficient": float(f.params[i]), "hac_se": float(f.bse[i]),
                         "t_stat": float(f.tvalues[i]), "p_value": float(f.pvalues[i]), "n_months": int(f.nobs),
                         "years": yrs, "reading": "test: OLS of monthly returns on dummies, Newey-West (3 lags)",
                         "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


# --- co-movement ----------------------------------------------------------------------------------------------------
def dimson(y: pd.Series, x: pd.Series, k: int = 1) -> dict:
    """Dimson (1979) lead-lag regression y_t = a + sum_{j=-k..k} b_j x_{t+j}: the summed beta (HAC t) and the summed
    cross-correlations (non-synchronous trading adjustment)."""
    import statsmodels.api as sm

    X = pd.concat({f"x{j:+d}": x.shift(-j) for j in range(-k, k + 1)}, axis=1)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    f = sm.OLS(d["y"], sm.add_constant(d[list(X.columns)])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    b = f.params[list(X.columns)]
    cov = f.cov_params().loc[list(X.columns), list(X.columns)].to_numpy()
    se = float(np.sqrt(np.ones(len(b)) @ cov @ np.ones(len(b))))
    cc = {j: float(d["y"].corr(d[f"x{j:+d}"])) for j in range(-k, k + 1)}
    return {"beta_sum": float(b.sum()), "beta_sum_se": se, "beta_sum_t": float(b.sum()) / se, "n": int(f.nobs),
            "cross_corr": cc, "corr_sum": float(sum(cc.values()))}


def comovement(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NMB-CRDB co-movement (prompt 15 item 5d; 15b item 1): correlations at daily, weekly and monthly frequency, the
    Dimson lead-lag version at daily frequency, by year; rolling 60-day (daily) and 12-month (monthly) correlations."""
    p = panel(root)
    m = market(root)
    r = p.pivot_table(index="trading_date", columns="ticker", values="lr_nonstale").sort_index()
    ra = p.pivot_table(index="trading_date", columns="ticker", values="lr_all").sort_index()
    r["MKT"] = m.reindex(r.index)
    rows = []

    def add(pair, freq, corr, n, note="", **kw):
        rows.append({"pair": pair, "frequency": freq, "correlation": corr, "n": n, "note": note} | kw)

    both = r[["NMB", "CRDB"]].dropna()
    add("NMB-CRDB", "daily, both non-stale", float(both.corr().iloc[0, 1]), len(both))
    w = ra[["NMB", "CRDB"]].resample("W-FRI").sum()
    w = w[(w != 0).any(axis=1)]
    add("NMB-CRDB", "weekly (all days)", float(w.corr().iloc[0, 1]), len(w),
        "Friday weeks; weeks with no price change for either bank dropped")
    mo = ra[["NMB", "CRDB"]].resample("ME").sum()
    add("NMB-CRDB", "monthly (all days)", float(mo.corr().iloc[0, 1]), len(mo), "calendar months")
    dd = ra[["NMB", "CRDB"]].dropna()
    dm = dimson(dd["NMB"], dd["CRDB"])
    add("NMB-CRDB", "daily, Dimson (lags -1, 0, +1)", dm["corr_sum"], dm["n"],
        "sum of cross-correlations corr(NMB_t, CRDB_t+j), j = -1, 0, +1, all days (stale days enter as zero returns, "
        "which the lead-lag terms adjust for); Dimson beta = sum of the three slopes, HAC t",
        cross_corr_lag_minus1=dm["cross_corr"][-1], cross_corr_lag0=dm["cross_corr"][0],
        cross_corr_lag_plus1=dm["cross_corr"][1], dimson_beta=dm["beta_sum"], dimson_beta_t=dm["beta_sum_t"])
    dm2 = dimson(dd["CRDB"], dd["NMB"])
    add("CRDB-NMB", "daily, Dimson (lags -1, 0, +1)", dm2["corr_sum"], dm2["n"], "CRDB regressed on NMB",
        cross_corr_lag_minus1=dm2["cross_corr"][-1], cross_corr_lag0=dm2["cross_corr"][0],
        cross_corr_lag_plus1=dm2["cross_corr"][1], dimson_beta=dm2["beta_sum"], dimson_beta_t=dm2["beta_sum_t"])
    for b in FOCUS:
        x = r[[b, "MKT"]].dropna()
        add(f"{b}-market proxy ex banks", "daily, non-stale", float(x.corr().iloc[0, 1]), len(x))
        xm = pd.concat([ra[b], ra[b].reindex(ra.index) * 0 + r["MKT"]], axis=1, keys=[b, "MKT"]).resample("ME").sum()
        add(f"{b}-market proxy ex banks", "monthly", float(xm.corr().iloc[0, 1]), len(xm))
    for yr, g in both.groupby(both.index.year):
        if len(g) > 30:
            add("NMB-CRDB", f"daily, {yr}", float(g.corr().iloc[0, 1]), len(g))
    d = pd.DataFrame(rows).assign(reading="descriptive (Dimson beta t: HAC, Newey-West 5 lags)",
                                  evidence_class="Derived")
    roll = ra[["NMB", "CRDB"]].rolling(60, min_periods=40).corr().xs("NMB", level=1)["CRDB"].dropna()
    rm = mo.rolling(12, min_periods=12).corr().xs("NMB", level=1)["CRDB"].dropna()
    # coverage (figure fix, prompt 15c): share of non-stale days in each window, the lower of the two banks
    ns = (~p.pivot_table(index="trading_date", columns="ticker", values="stale").astype(bool)).astype(float)
    ns = ns.reindex(ra.index)[["NMB", "CRDB"]]
    ns_d = ns.rolling(60, min_periods=40).mean().min(axis=1).reindex(roll.index)
    nm = ns.resample("ME").agg(["sum", "count"])
    ns_m = pd.concat({b: nm[(b, "sum")].rolling(12).sum() / nm[(b, "count")].rolling(12).sum() for b in ("NMB", "CRDB")},
                     axis=1).min(axis=1).reindex(rm.index)
    thr = min_nonstale_share(root)
    rl = pd.concat([pd.DataFrame({"trading_date": roll.index, "window": "60 trading days (daily returns)",
                                  "correlation": roll.values, "window_nonstale_share_min": ns_d.values}),
                    pd.DataFrame({"trading_date": rm.index, "window": "12 months (monthly returns)",
                                  "correlation": rm.values, "window_nonstale_share_min": ns_m.values})],
                   ignore_index=True)
    rl["coverage_ok"] = rl["window_nonstale_share_min"] >= thr
    return d, rl.assign(threshold=thr, evidence_class="Derived")


def min_nonstale_share(root: Path) -> float:
    """dse.library.min_nonstale_share: windows with fewer non-stale days (either bank) are not drawn (prompt 15c)."""
    return float(settings(root)["dse"]["library"].get("min_nonstale_share", 0.5))


def daily_coverage(root: Path) -> pd.DataFrame:
    """Daily-data coverage per bank and year (prompt 15c): trading days in the library, days with a trade, non-stale
    days (a price change or a close inside the day's range), and whether the year passes the coverage threshold."""
    p = panel(root)
    thr = min_nonstale_share(root)
    rows = []
    for (b, y), g in p.groupby(["ticker", p["trading_date"].dt.year]):
        n = len(g)
        rows.append({"bank": b, "year": int(y), "trading_days": n, "traded_days": int(g["traded"].astype(bool).sum()),
                     "nonstale_days": int((~g["stale"].astype(bool)).sum()),
                     "traded_share": float(g["traded"].astype(bool).mean()),
                     "nonstale_share": float((~g["stale"].astype(bool)).mean()), "threshold": thr,
                     "coverage": "thin (below threshold)" if (~g["stale"].astype(bool)).mean() < thr else "adequate",
                     "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def results_events(root: Path) -> pd.DataFrame:
    """Results-release events with confirmed dates only (prompt 15b item 2). Candidates: interim statements whose
    Sources publication date is the bank's posting or DSE filing date; annual reports in Sources (dates are
    earliest-possible estimates: dropped and labelled); events_collected results rows (filing or press) and the
    results-announcement dates printed in the NMB AGM notices. An events_collected date contradicted by a confirmed
    posting within 10 days is dropped as superseded. Per event and bank: CAR vs the market proxy at [0,+5], [0,+20]."""
    ev = pd.read_csv(root / "data" / "raw" / "manual_entries" / "events_collected.csv")
    s = pd.read_csv(root / "data" / "processed" / "sources.csv", parse_dates=["Publication Date"])
    s = s[s["Owner"].isin(FOCUS) & s["Doc ID"].astype(str).str.contains("-INT-|-AR-FY")
          & (s["Publication Date"] >= "2015-06-30")]
    rows = []
    for _, r in s.iterrows():
        how = str(r["How Publication Date Established"])
        doc = str(r["Doc ID"])
        ok = "-INT-" in doc and any(k in how.lower() for k in ("posting", "dse filing date", "file timestamp"))
        rows.append({"bank": r["Owner"], "event_date": str(r["Publication Date"].date()),
                     "event": f"{doc}: results statement published",
                     "date_basis": "confirmed: bank posting / DSE filing date (Sources)" if ok else
                     "estimated: earliest-possible date (board approval / auditor report) or report posting after "
                     "the results release", "source": f"Sources {doc}: {how[:120]}", "include": ok})
    x = ev[(ev["kind"] == "results") | ev["event"].str.contains("results announced", case=False)]
    for r in x.itertuples():
        filing = str(r.source_title).startswith("DSE filing")
        rows.append({"bank": r.bank, "event_date": r.event_date, "event": r.event,
                     "date_basis": "confirmed: " + ("filing" if filing else "press report"),
                     "source": f"events_collected: {r.source_title}", "include": True})
    d = pd.DataFrame(rows).sort_values(["bank", "event_date"]).reset_index(drop=True)
    conf = d[d["include"] & d["source"].str.startswith("Sources")]
    for i, r in d[d["include"] & d["source"].str.startswith("events_collected")].iterrows():
        gap = (pd.to_datetime(conf["event_date"]) - pd.Timestamp(r["event_date"])).dt.days.abs()
        near = conf[(conf["bank"] == r["bank"]) & gap.between(1, 10)]
        if len(near):
            d.loc[i, "include"] = False
            d.loc[i, "date_basis"] = (f"superseded: the confirmed posting on {near['event_date'].iloc[0]} (Sources) "
                                      "contradicts this date")
    d = d[~(d["include"] & d.duplicated(["bank", "event_date"]))]
    late = pd.to_datetime(d["event_date"]) > cutoff(root)
    d.loc[late, "include"] = False
    d.loc[late, "date_basis"] = d.loc[late, "date_basis"] + "; after the data cutoff"
    inc = d[d["include"]].assign(confidence=lambda z: z["date_basis"], kind="results")
    es = event_study(root, inc)
    es = es[es["window"].isin(["[0,+5]", "[0,+20]"])].drop(columns=["kind", "source_confidence", "reading",
                                                                    "evidence_class"])
    out = d.rename(columns={"bank": "event_bank"}).merge(es, on=["event_bank", "event_date", "event"], how="left")
    out["reading"] = np.where(out["include"], "case study; t = CAR / (estimation-residual sd x sqrt(days))",
                              "dropped: date not confirmed")
    full = out["window"].map({"[0,+5]": 6, "[0,+20]": 21})
    out["window_complete"] = np.where(out["include"], out["days"] >= full, np.nan)
    out.loc[out["include"] & (out["days"] < full), "reading"] = ("case study; window truncated by the data cutoff "
                                                                 "(fewer trading days than the window)")
    out["evidence_class"] = "Derived"
    return out.reset_index(drop=True)


