"""Chapter 5: persistence and economic rents (blueprint §2 core definitions, §4 chapter 5; H5.1-H5.3; D8, D12).

ER_{i,t} = ROE_{i,t} - COE_t;  RI_{i,t} = NI_{i,t} - COE_t * B_{i,t-1}.

ROE is attributable profit over average attributable equity (Group) for NMB and CRDB; for banks where Bank = Group
(B03-B07) it is net profit over average total equity (``dupont_panel``). Estimation excludes D-flagged bank-years
(``bank_year_flags.accepted_for_analysis``: MBP FY2013), MUCOBA FY2020 (near-zero average equity, shown but not
estimated) and years where average equity is not positive (ROE undefined). Calibration candidates for chapters 8-9
are listed under every specification; none is chosen here (D8).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tzbank.chapters.ch04.common import COLORS, FOCUS, pct, plt_subplots, save, style, tickers, wide
from tzbank.chapters.ch04.profitability import dupont_panel
from tzbank.chapters.ch05 import models
from tzbank.chapters.ch05.coe import coe_annual
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text

EXCLUDE_SHOWN = {("B07", 2020): "MUCOBA FY2020: near-zero average equity (recapitalisation year); shown, not estimated"}
SPECS = ["ar1_benchmark", "ar1_break_ifrs9", "ar1_break_episode", "mean_reversion_peer_mean",
         "mean_reversion_bot_sector", "robust_coe_minus_200bp", "robust_coe_plus_200bp", "robust_excl_fy2018",
         "robust_d41_adjusted_2008_12", "robust_coe_2025_vintage", "robust_beta_2012_replaced",
         "robust_original_vintage", "robust_bank_basis",
         "subperiod_pre_break",
         "subperiod_post_break", "ar1_break_episode_excl_fy2018"]  # appended last: earlier specs keep their seeds
SPEC_TEXT = {
    "ar1_benchmark": "AR(1) benchmark, all eligible years (descriptive, D12); ER* is THROUGH-THE-CYCLE (stress years "
                     "2015-18 included)",
    "ar1_break_ifrs9": "AR(1) with an intercept dummy for FY2018 onward (IFRS 9); ER* in the current (2018+) regime",
    "ar1_break_episode": "AR(1) with an intercept dummy for the 2015-18 credit episode; BREAK-ADJUSTED ER*, which "
                         "EXCLUDES the stress period",
    "mean_reversion_peer_mean": "AR(1) of ER minus the mean ER of the other panel banks that year; ER* is the long-run "
                                "gap to peers",
    "mean_reversion_bot_sector": "AR(1) of ER minus the BoT sector ER (FSI ROE - COE, 2019-25 only); ER* is the "
                                 "long-run gap to the sector",
    "robust_coe_minus_200bp": "AR(1) with COE - 200 bp (lambda unchanged by a parallel shift; ER* + 2 pp)",
    "robust_coe_plus_200bp": "AR(1) with COE + 200 bp (lambda unchanged by a parallel shift; ER* - 2 pp)",
    "robust_excl_fy2018": "AR(1) dropping every pair that contains FY2018",
    "ar1_break_episode_excl_fy2018": "AR(1) with the 2015-18 episode dummy, every pair containing FY2018 (the IFRS 9 "
                                     "transition year) dropped (prompt 24)",
    "robust_d41_adjusted_2008_12": "AR(1) with the D41 term-spread-adjusted COE for FY2008-12",
    "robust_coe_2025_vintage": "AR(1) with the previous COE (2025 Damodaran vintage in every year, prompt 04)",
    "robust_beta_2012_replaced": "D51 sensitivity (not the primary series): AR(1) with the January 2012 EM bank beta "
                                 "(0.31, outlier) replaced by the January 2013 value in FY2008-12",
    "through_cycle_mean": "Mean ER over all eligible years (stress years included), no persistence model: "
                          "THROUGH-THE-CYCLE alternative when the AR(1) ER* is undefined (lambda at 1)",
    "robust_original_vintage": "AR(1) on originally published figures (no restatements)",
    "robust_bank_basis": "AR(1) on Bank-solo ROE (NMB, CRDB)",
    "subperiod_pre_break": "AR(1) before the sup-F break year (only where the sup-F p-value < 0.10)",
    "subperiod_post_break": "AR(1) from the sup-F break year (only where the sup-F p-value < 0.10)",
}


# --- panel ----------------------------------------------------------------------------------------------------------


def _vintage_wide(root: Path, original: bool = False, basis_override: str | None = None) -> pd.DataFrame:
    """Bank-year frame of profit and equity from the long table: original (unrestated) rows or a given basis."""
    lg = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False, dtype={"bank_id": str})
    lg = lg[(lg["period_type"] == "FY") & lg["canonical_name"].isin(
        ["profit_attributable", "net_profit", "equity_attributable", "total_equity"]) & lg["value"].notna()]
    if basis_override:
        lg = lg[lg["basis"] == basis_override]
    else:
        own = lg["bank_id"].map(lambda b: "GROUP" if b in FOCUS else "BANK")
        lg = lg[lg["basis"] == own]
    if original:
        lg = lg[~lg["is_restated"].astype(bool)]
    lg = lg.sort_values("publication_date")
    v = lg.drop_duplicates(["bank_id", "fiscal_year", "canonical_name"], keep="first" if original else "last")
    return v.pivot_table(index=["bank_id", "fiscal_year"], columns="canonical_name", values="value",
                         aggfunc="first").reset_index().sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)


def _roe_frame(w: pd.DataFrame) -> pd.DataFrame:
    p = dupont_panel(w)
    w = w.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    col = lambda c: w[c] if c in w.columns else pd.Series(np.nan, index=w.index)  # noqa: E731
    attr = col("profit_attributable").notna() & col("equity_attributable").notna()
    ni = col("profit_attributable").where(attr, col("net_profit"))
    eq = col("equity_attributable").where(attr, col("total_equity"))
    prev_ok = w.groupby("bank_id")["fiscal_year"].diff() == 1
    b_open = eq.groupby(w["bank_id"]).shift(1).where(prev_ok)
    avg_eq = (eq + b_open) / 2
    out = w[["bank_id", "fiscal_year"]].assign(ni=ni, b_open=b_open, avg_equity=avg_eq)
    return out.merge(p[["bank_id", "fiscal_year", "roe", "roe_basis"]], on=["bank_id", "fiscal_year"], how="left")


def er_panel(root: Path, coe: pd.DataFrame) -> pd.DataFrame:
    """ER and RI per bank-year (all seven banks) with the estimation-sample flag and the reason for exclusion."""
    w = wide(root)
    f = _roe_frame(w)
    c = coe[coe["date_label"].str.startswith("FY")].set_index("year")
    f = f[f["fiscal_year"].isin(c.index)].copy()
    for k in ("coe", "coe_minus_200bp", "coe_plus_200bp", "coe_d41_adjusted", "coe_2025_vintage",
              "coe_beta2012_replaced"):
        f[k] = f["fiscal_year"].map(c[k])
    f["er"] = f["roe"] - f["coe"]
    f["ri"] = f["ni"] - f["coe"] * f["b_open"]
    flags = pd.read_csv(path("processed", root) / "bank_year_flags.csv", dtype={"bank_id": str})
    acc = flags.set_index(["bank_id", "fiscal_year"])["accepted_for_analysis"]
    reason = []
    for _, r in f.iterrows():
        k = (r["bank_id"], int(r["fiscal_year"]))
        if pd.isna(r["roe"]):
            reason.append("ROE not computable (average equity or profit not held)")
        elif k in EXCLUDE_SHOWN:
            reason.append(EXCLUDE_SHOWN[k])
        elif not (r["avg_equity"] > 0):
            reason.append("average equity not positive: ROE undefined")
        elif k in acc.index and not bool(acc[k]):
            reason.append("D-flagged bank-year (bank_year_flags: not accepted for analysis)")
        else:
            reason.append("")
    f["excluded_reason"] = reason
    f["in_estimation"] = f["excluded_reason"] == ""
    f["evidence_class"] = "Estimated"  # ROE is Derived; COE is Estimated, so ER and RI are Estimated
    return f.reset_index(drop=True)


def _alt_er(root: Path, coe: pd.DataFrame, base: pd.DataFrame, **kw) -> pd.DataFrame:
    """ER under an alternative vintage or basis, keeping the primary exclusion list."""
    f = _roe_frame(_vintage_wide(root, **kw))
    c = coe[coe["date_label"].str.startswith("FY")].set_index("year")["coe"]
    f["er"] = f["roe"] - f["fiscal_year"].map(c)
    bad = base.loc[~base["in_estimation"], ["bank_id", "fiscal_year"]].assign(x=1)
    f = f.merge(bad, on=["bank_id", "fiscal_year"], how="left")
    return f[f["x"].isna() & (f["avg_equity"] > 0)].drop(columns="x")


# --- estimation -----------------------------------------------------------------------------------------------------


def _drop18(d: pd.DataFrame) -> pd.DataFrame:
    """Pairs without FY2018 (as the dependent or the lagged year): the IFRS 9 transition year (prompt 24)."""
    return d[(d["year"] != 2018) & (d["year"] != 2019)]


def ifrs9_sensitivity(p: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Does FY2018 (IFRS 9 transition) drive the 2015-18 results? (prompt 24, 3c.7; Blueprint ch. 5). For each bank:
    benchmark and break-adjusted lambda and ER* with FY2018 in and out, and the episode break test on both samples."""
    rows = []
    for b in sorted(set(p["bank_id"]) - {"pooled"}):
        x = p[p["bank_id"] == b].set_index("specification")
        t = bt[bt["bank_id"] == b].set_index("test")
        def g(spec, col, x=x):
            return float(x.loc[spec, col]) if spec in x.index and col in x else float("nan")
        def pv(test, t=t):
            return float(t.loc[test, "p_value"]) if test in t.index else float("nan")
        rows.append({"bank_id": b,
                     "lambda_benchmark": g("ar1_benchmark", "lambda"), "lambda_excl_fy2018": g("robust_excl_fy2018", "lambda"),
                     "er_star_benchmark": g("ar1_benchmark", "er_star"),
                     "er_star_excl_fy2018": g("robust_excl_fy2018", "er_star"),
                     "lambda_episode": g("ar1_break_episode", "lambda"),
                     "lambda_episode_excl_fy2018": g("ar1_break_episode_excl_fy2018", "lambda"),
                     "er_star_episode": g("ar1_break_episode", "er_star"),
                     "er_star_episode_excl_fy2018": g("ar1_break_episode_excl_fy2018", "er_star"),
                     "p_chow_episode": pv("Chow 2015-18 episode"),
                     "p_chow_episode_excl_fy2018": pv("Chow 2015-18 episode, FY2018 excluded"),
                     "p_chow_ifrs9": pv("Chow FY2018 (IFRS 9)"), "evidence_class": "Estimated"})
    return pd.DataFrame(rows)

def _series(df: pd.DataFrame, bank: str, col: str = "er") -> pd.Series:
    d = df[df["bank_id"] == bank]
    return d.set_index("fiscal_year")[col].astype(float)


def _estimate_all(d: pd.DataFrame, dmy, at, cfg: dict, seed: list[int], primary: str, mp: int) -> dict:
    """Standard moving-block, Kilian (1998) and Andrews (1993) estimates; the primary method fills lambda, its interval
    and ER* (ER* interval: the primary method's, or Kilian's when the primary method defines none)."""
    reps, block, level = cfg["bootstrap_reps"], cfg["block_length"], cfg["interval"]
    std = models.estimate(d, reps, block, level, seed, dmy, mp, at)
    if std["status"] != "estimated":
        return std
    kil = models.kilian(d, reps, reps, level, [*seed, 2], dmy, at)
    andr = models.andrews(d, level, cfg["andrews_sims"], [seed[0], 903], dmy, at)
    out = {"n": std["n"], "years": std["years"], "status": "estimated", "lambda_ols": std["lambda"],
           "lambda_std_lo": std["lambda_lo"], "lambda_std_hi": std["lambda_hi"], "er_star_std": std["er_star"],
           "er_star_std_lo": std["er_star_lo"], "er_star_std_hi": std["er_star_hi"],
           "lambda_kilian": kil["lambda"], "lambda_kilian_lo": kil["lambda_lo"], "lambda_kilian_hi": kil["lambda_hi"],
           "kilian_bias": kil["bias"], "er_star_kilian": kil["er_star"], "er_star_kilian_lo": kil["er_star_lo"],
           "er_star_kilian_hi": kil["er_star_hi"], "lambda_andrews": andr["lambda"],
           "lambda_andrews_lo": andr["lambda_lo"], "lambda_andrews_hi": andr["lambda_hi"],
           "er_star_andrews": andr["er_star"]}
    src = {models.METHODS[0]: std, models.METHODS[1]: kil, models.METHODS[2]: andr}[primary]
    out |= {"primary_method": primary, "lambda": src["lambda"], "lambda_lo": src["lambda_lo"],
            "lambda_hi": src["lambda_hi"], "er_star": src["er_star"]}
    if np.isfinite(src["er_star_lo"]) or primary != models.METHODS[2]:
        out |= {"er_star_lo": src["er_star_lo"], "er_star_hi": src["er_star_hi"], "er_star_interval_method": primary}
    else:
        out |= {"er_star_lo": kil["er_star_lo"], "er_star_hi": kil["er_star_hi"],
                "er_star_interval_method": "Kilian (1998) (the Andrews method defines no ER* interval)"}
    for k in ("", "_std", "_kilian"):
        lo, hi = out.get(f"er_star{k}_lo"), out.get(f"er_star{k}_hi")
        if lo is not None and hi is not None and (not np.isfinite(lo) or not np.isfinite(hi) or lo < -1 or hi > 1):
            out[f"er_star{k}_lo"] = out[f"er_star{k}_hi"] = float("nan")
            if k == "":
                out["er_star_interval_method"] += "; interval unbounded (lambda near 1 in the bootstrap)"
    return out


def persistence_table(root: Path, panel: pd.DataFrame, coe: pd.DataFrame,
                      primary: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, seed = settings(root)["persistence"], int(settings(root)["random_seed"])
    reps, block, level, mp = cfg["bootstrap_reps"], cfg["block_length"], cfg["interval"], cfg["min_pairs"]
    est = panel[panel["in_estimation"]]
    band = cfg["coe_band_bp"] / 10_000
    orig = _alt_er(root, coe, panel, original=True)
    bank_basis = _alt_er(root, coe, panel, basis_override="BANK")
    sec = pd.read_csv(path("processed", root) / "sector_totals.csv")
    sec = sec[sec["series_id"] == "BOTDB_FINANCIAL_SOUNDNESS__RETURN_ON_EQUITY"].drop_duplicates("year")
    c = coe[coe["date_label"].str.startswith("FY")].set_index("year")["coe"]
    sector_er = (sec.set_index("year")["value"] - c).dropna()
    rows, breaks = [], []
    for bi, b in enumerate(sorted(est["bank_id"].unique()), start=1):
        er = _series(est, b)
        base = models.pairs(er)
        dm = pd.DataFrame({"d_ifrs9": (base["year"] >= 2018).astype(int).to_numpy(),
                           "d_episode": base["year"].between(2015, 2018).astype(int).to_numpy()},
                          index=base["year"].to_numpy())
        peers = est[est["bank_id"] != b].groupby("fiscal_year")["er"].agg(["mean", "count"])
        peer_mean = peers["mean"].where(peers["count"] >= 3)
        er_d41 = er.copy()
        alt = _series(est, b, "coe_d41_adjusted")
        er_d41.loc[alt.notna()] = (_series(est, b, "roe") - alt)[alt.notna()]
        er_v25 = _series(est, b, "roe") - _series(est, b, "coe_2025_vintage")
        er_b12 = _series(est, b, "roe") - _series(est, b, "coe_beta2012_replaced")
        data = {
            "ar1_benchmark": (base, None),
            "ar1_break_ifrs9": (models.pairs(er, dm[["d_ifrs9"]]), ["d_ifrs9"], [1.0]),
            "ar1_break_episode": (models.pairs(er, dm[["d_episode"]]), ["d_episode"], [0.0]),
            "mean_reversion_peer_mean": (models.pairs((er - peer_mean).dropna()), None),
            "mean_reversion_bot_sector": (models.pairs((er - sector_er).dropna()), None),
            "robust_coe_minus_200bp": (models.pairs(er + band), None),
            "robust_coe_plus_200bp": (models.pairs(er - band), None),
            "robust_excl_fy2018": (base[(base["year"] != 2018) & (base["year"] != 2019)], None),
            "ar1_break_episode_excl_fy2018": (_drop18(models.pairs(er, dm[["d_episode"]])), ["d_episode"], [0.0]),
            "robust_d41_adjusted_2008_12": (models.pairs(er_d41), None),
            "robust_coe_2025_vintage": (models.pairs(er_v25), None),
            "robust_beta_2012_replaced": (models.pairs(er_b12), None),
            "robust_original_vintage": (models.pairs(_series(orig, b)), None),
            "robust_bank_basis": (models.pairs(_series(bank_basis, b)) if b in FOCUS else base.iloc[0:0], None),
        }
        # break tests on the benchmark sample
        yrs = base["year"].to_numpy()
        b18 = _drop18(base)
        y18 = b18["year"].to_numpy()
        for name, regime, smp in (("Chow FY2018 (IFRS 9)", yrs >= 2018, base),
                                  ("Chow 2015-18 episode", (yrs >= 2015) & (yrs <= 2018), base),
                                  ("Chow 2015-18 episode, FY2018 excluded", (y18 >= 2015) & (y18 <= 2018), b18)):
            r = models.chow(smp, regime)
            breaks.append({"bank_id": b, "test": name, "break_year": 2018 if "IFRS" in name else 2015, "n": len(smp),
                           "F": r["F"], "df1": r["df1"], "df2": r["df2"], "p_value": r["p_value"],
                           "method": "F test of intercept and slope shift (known date)"
                           + ("" if r["status"] == "computed" else f"; {r['status']}")})
        if len(base) >= cfg["supf_min_pairs"]:
            s = models.sup_f(base, cfg["supf_reps"], cfg["supf_min_regime"], [seed, bi, 99])
            breaks.append({"bank_id": b, "test": "sup-F (unknown date)", "break_year": s["break_year"],
                           "n": len(base), "F": s["F"], "df1": 2, "df2": len(base) - 4, "p_value": s["p_value"],
                           "method": f"max Chow F with >= {cfg['supf_min_regime']} pairs per side; p-value from "
                                     f"{cfg['supf_reps']} parametric-bootstrap draws of the fitted AR(1) (no break)"})
            if s["p_value"] < 0.10 and s["break_year"]:
                data["subperiod_pre_break"] = (base[base["year"] < s["break_year"]], None)
                data["subperiod_post_break"] = (base[base["year"] >= s["break_year"]], None)
        else:
            breaks.append({"bank_id": b, "test": "sup-F (unknown date)", "break_year": None, "n": len(base),
                           "F": float("nan"), "df1": 2, "df2": len(base) - 4, "p_value": float("nan"),
                           "method": f"not run: {len(base)} pairs < {cfg['supf_min_pairs']}"})
        for si, spec in enumerate(SPECS, start=1):
            if spec not in data:
                continue
            d, dmy, *at = data[spec]
            # the parallel COE shifts reuse the benchmark seed, so their ER* intervals move exactly by the band
            sx = 1 if spec.startswith("robust_coe_") else si
            r = _estimate_all(d, dmy, at[0] if at else None, cfg, [seed, bi, sx], primary, mp)
            rows.append({"bank_id": b, "specification": spec, "description": SPEC_TEXT[spec], **r,
                         "block_length": block, "bootstrap_reps": reps, "seed": f"[{seed}, {bi}, {sx}]",
                         "interval": f"{level:.0%}; primary method {primary} (chosen by the coverage study)"})
        # through-the-cycle mean ER (no persistence model): a base-case alternative when the AR(1) ER* is undefined
        m, lo, hi = models.mean_interval(er, cfg["bootstrap_reps"], block, level, [seed, bi, 50])
        rows.append({"bank_id": b, "specification": "through_cycle_mean", "description": SPEC_TEXT["through_cycle_mean"],
                     "n": int(er.notna().sum()), "years": f"{int(er.index.min())}-{int(er.index.max())}",
                     "status": "estimated", "er_star": m, "er_star_lo": lo, "er_star_hi": hi,
                     "primary_method": "sample mean; moving-block bootstrap interval",
                     "er_star_interval_method": "moving-block bootstrap of the mean", "seed": f"[{seed}, {bi}, 50]",
                     "interval": f"{level:.0%} percentile"})
    # pooled fixed effects (secondary)
    pp = pd.concat([models.pairs(_series(est, b)).assign(bank_id=b) for b in est["bank_id"].unique()])
    fe = models.pooled_fe(pp, level)
    rows.append({"bank_id": "pooled", "specification": "pooled_fixed_effects",
                 "description": "Within (FE) AR(1), all banks; SECONDARY: Nickell bias towards zero, approx "
                                f"{fe['nickell_bias_approx']:+.3f}", "n": fe["n"], "lambda": fe["lambda"],
                 "lambda_lo": fe["lambda_lo"], "lambda_hi": fe["lambda_hi"], "status": "estimated (secondary)",
                 "primary_method": "within estimator (classical t interval)",
                 "interval": f"{level:.0%} t interval (classical)",
                 "half_life_status": models.half_life(fe["lambda"], fe["lambda_hi"])[1]})
    p = pd.DataFrame(rows)
    p["half_life"] = [models.half_life(lam, hi)[0] if pd.notna(lam) else np.nan
                      for lam, hi in zip(p["lambda"], p["lambda_hi"], strict=True)]
    p["half_life_status"] = [models.half_life(lam, hi)[1] if pd.notna(lam) else
                             ("n/a (no persistence model)" if st == "estimated" else st)
                             for lam, hi, st in zip(p["lambda"], p["lambda_hi"], p["status"], strict=True)]
    p["evidence_class"] = "Estimated"
    bt = pd.DataFrame(breaks)
    bt["evidence_class"] = "Estimated"
    return p, bt


def sign_and_rank(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    est = panel[panel["in_estimation"]]
    s = pd.DataFrame([{"bank_id": b, **models.sign_persistence(_series(est, b)),
                       "cumulative_ri_tzs": float(est.loc[est["bank_id"] == b, "ri"].sum())}  # shown in the note (12a)
                      for b in sorted(est["bank_id"].unique())])
    s["evidence_class"] = "Estimated"
    rk = pd.concat([models.rank_persistence(est, lag) for lag in (1, 3, 5)], ignore_index=True)
    rk["evidence_class"] = "Estimated"
    return s, rk


# --- hypotheses and calibration ------------------------------------------------------------------------------------


def _perm_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rho = stats.spearmanr(a, b).statistic
    perms = [stats.spearmanr(a, np.array(q)).statistic for q in itertools.permutations(b)]
    return float(rho), float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def verdict_specs(p: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Per bank, the specification the verdicts read: the benchmark AR(1), or, where a known-date Chow test rejects
    at 10%, the AR(1) with that break dummy (blueprint §2: re-estimate where a break is found; lowest p wins)."""
    rows = []
    for b, g in p[p["bank_id"] != "pooled"].groupby("bank_id"):
        spec, why = "ar1_benchmark", "no known-date break at 10%"
        ch = bt[(bt["bank_id"] == b) & bt["test"].str.startswith("Chow") & (bt["p_value"] < 0.10)]
        if len(ch):
            c = ch.sort_values("p_value").iloc[0]
            cand = "ar1_break_episode" if "episode" in c["test"] else "ar1_break_ifrs9"
            if (g.loc[g["specification"] == cand, "status"] == "estimated").any():
                spec, why = cand, f"{c['test']} p = {c['p_value']:.2f}"
        r = g[g["specification"] == spec].iloc[0].to_dict()
        rows.append(r | {"verdict_spec": spec, "verdict_spec_reason": why})
    return pd.DataFrame(rows).set_index("bank_id")


def hypotheses(p: pd.DataFrame, bt: pd.DataFrame, t: dict[str, str], root: Path, band: float) -> pd.DataFrame:
    v = verdict_specs(p, bt)
    est = v[v["status"] == "estimated"]
    bench = p[p["specification"] == "ar1_benchmark"].set_index("bank_id")
    rows = []

    def name(b: str) -> str:
        return t.get(b, b)

    def lab(b: str) -> str:
        return f"{name(b)} [{v.loc[b, 'verdict_spec']}; {v.loc[b, 'verdict_spec_reason']}]"

    def v51(r) -> str:
        if r["lambda_lo"] > 0 and r["lambda_hi"] < 1:
            return "supported"
        if r["lambda_hi"] <= 0 or r["lambda_lo"] >= 1:
            return "rejected"
        return "inconclusive"

    per = {b: v51(r) for b, r in est.iterrows()}
    focus = [per.get(b) for b in FOCUS]
    overall = ("supported" if all(x == "supported" for x in focus) and "rejected" not in per.values() else
               "rejected" if sum(x == "rejected" for x in per.values()) > len(per) / 2 else "inconclusive")
    # owner (prompt 21, D90): when the focus verdicts rest on the break-adjusted estimates while the full-cycle
    # benchmark cannot exclude lambda = 1, the verdict is conditional, worded as below
    full_unit = [b for b in FOCUS if b in bench.index and bench.loc[b, "status"] == "estimated"
                 and bench.loc[b, "lambda_hi"] >= 1 - 1e-9]
    if overall == "supported" and full_unit and all(v.loc[b, "verdict_spec"] != "ar1_benchmark" for b in full_unit):
        overall = ("conditionally supported: mean reversion appears in the break-adjusted estimates but is not "
                   "established in the full-cycle sample")
    rows.append({"hypothesis": "H5.1", "scope": "all banks", "verdict": overall,
                 "evidence": "; ".join(f"{lab(b)} lambda {r['lambda']:.2f} [{r['lambda_lo']:.2f}, "
                                       f"{r['lambda_hi']:.3f}]: {per[b]}" for b, r in est.iterrows())
                 + "; not estimated: " + (", ".join(name(b) for b in v.index if b not in est.index) or "none"),
                 "against": "benchmark AR(1) without break dummies: " + "; ".join(
                     f"{name(b)} lambda {r['lambda']:.2f} [{r['lambda_lo']:.2f}, {r['lambda_hi']:.3f}]"
                     for b, r in bench.iterrows() if r["status"] == "estimated"),
                 "rule": "per bank: supported when the 90% interval lies inside (0, 1), rejected when it lies at or "
                         "below 0 or at or above 1; overall supported when NMB and CRDB are supported and no bank is "
                         "rejected. Verdicts read the break-dummy AR(1) where a known-date Chow test rejects at 10%"})
    def v52(tab: pd.DataFrame) -> tuple[str, str]:
        tab = tab[tab["status"] == "estimated"]
        small = [b for b in tab.index if b not in FOCUS]
        pos = {b: bool(pd.notna(tab.loc[b, "er_star_lo"]) and tab.loc[b, "er_star_lo"] > 0)
               for b in FOCUS if b in tab.index}
        band_ok = {b: bool(pos[b] and tab.loc[b, "er_star_lo"] - band > 0) for b in pos}
        small_ok = {b: bool(pd.isna(tab.loc[b, "er_star_lo"]) or tab.loc[b, "er_star_lo"] <= 0) for b in small}
        if any(pd.notna(tab.loc[b, "er_star_hi"]) and tab.loc[b, "er_star_hi"] < 0 for b in pos):
            verdict = "rejected"
        elif all(pos.values()) and all(small_ok.values()):
            verdict = "supported" if all(band_ok.values()) else "inconclusive (not robust to COE + 200 bp)"
        else:
            verdict = "inconclusive"
        txt = "; ".join(f"{name(b)} ER* {pct(r['er_star'], reason='undefined')} "
                        f"[{pct(r['er_star_lo'], reason='undefined')}, {pct(r['er_star_hi'], reason='undefined')}]"
                        for b, r in tab.iterrows())
        txt += "; NMB/CRDB lower bounds at COE + 200 bp: " + ", ".join(
            f"{name(b)} {pct(tab.loc[b, 'er_star_lo'] - band, reason='undefined')}" for b in pos)
        return verdict, txt

    ttc_v, ttc_t = v52(bench)
    brk_v, brk_t = v52(v)
    rows.append({"hypothesis": "H5.2", "scope": "NMB, CRDB vs smaller banks", "verdict": ttc_v,
                 "evidence": f"THROUGH-THE-CYCLE ER* (AR(1) benchmark, stress years included; decides the verdict, "
                             f"base case): {ttc_v}. {ttc_t}",
                 "against": f"BREAK-ADJUSTED ER* (excludes the 2015-18 stress period where the Chow test rejects; "
                            f"bull case): {brk_v}. " + "; ".join(lab(b) for b in v.index if b in FOCUS) + ". "
                            + brk_t,
                 "rule": "supported when NMB and CRDB ER* intervals lie above zero, also at COE + 200 bp (a conclusion "
                         "that flips inside the band is not robust, §2), and each smaller bank's interval reaches zero "
                         "or below; rejected if NMB or CRDB ER* lies below zero. Through-the-cycle ER* decides "
                         "(D46 as amended); the break-adjusted reading is reported alongside"})
    f = pd.read_csv(path("tables", root) / "ch04_4_2_funding.csv", dtype={"bank_id": str})
    cod = f.groupby("bank_id")["cost_of_deposits"].agg(["mean", "count"])
    j = est[["lambda"]].join(cod, how="inner").dropna()
    j = j[j["count"] >= 3]
    if len(j) >= 5:
        rho, pval = _perm_spearman(j["lambda"].to_numpy(float), j["mean"].to_numpy(float))
        v53 = ("supported" if rho <= -0.5 and pval < 0.10 else "rejected" if rho >= 0.5 and pval < 0.10
               else "inconclusive")
        ev = (f"Spearman(lambda, mean cost of deposits) = {rho:+.2f}, exact permutation p = {pval:.2f}, "
              f"{len(j)} banks with >= 3 years of cost of deposits: " + "; ".join(
                  f"{name(b)} lambda {r['lambda']:.2f}, cost {pct(r['mean'], 2)} (n={int(r['count'])})"
                  for b, r in j.iterrows()))
    else:
        v53, ev = "inconclusive", f"only {len(j)} banks with >= 3 years of cost of deposits"
    rows.append({"hypothesis": "H5.3", "scope": "all banks (provisional)", "verdict": v53, "evidence": ev,
                 "against": "cost of deposits is a gross proxy with few years for the smaller banks; lambda "
                            "intervals overlap widely",
                 "rule": "PROVISIONAL: deposit-franchise strength proxied by the chapter 4.2 cost-of-deposits rank "
                         "(lower cost = stronger); supported if rho <= -0.5 with p < 0.10; revise with the chapter 6 "
                         "net deposit franchise"})
    h = pd.DataFrame(rows)
    h["evidence_class"] = "Interpretation"
    return h


def calibration_candidates(p: pd.DataFrame, t: dict[str, str]) -> pd.DataFrame:
    c = p[p["bank_id"].isin(FOCUS) & (p["status"] == "estimated")].copy()
    c.insert(0, "bank", c["bank_id"].map(t))
    c["er_star_definition"] = np.where(c["specification"].str.startswith("mean_reversion"),
                                       "long-run gap to the benchmark (not a level)", "long-run ER level")
    role = {"through_cycle_mean": "BASE CASE ER* (D50, confirm at 08/09a): through-the-cycle mean excess return, stress "
                                  "years included",
            "ar1_break_episode": "BULL CASE ER* (D50): break-adjusted ER*, EXCLUDES the 2015-18 stress period; its "
                                 "Andrews lambda is the FADE SPEED for all three cases (D50)",
            "ar1_benchmark": "alternative: through-the-cycle AR(1) (ER* undefined where lambda = 1)"}
    c["case_role"] = c["specification"].map(role).fillna("alternative / robustness")
    # bear case (D50): ER* = 0 (ROE fades to COE) at the break-adjusted (Andrews) fade speed
    bear = c[c["specification"] == "ar1_break_episode"].copy()
    if len(bear):
        bear["specification"] = "bear_zero_rent"
        bear["description"] = "Bear case (D50): long-run excess return zero (ROE -> COE); fade at the break-adjusted lambda"
        bear[["er_star", "er_star_lo", "er_star_hi", "er_star_std", "er_star_kilian", "er_star_andrews"]] = 0.0
        bear["er_star_interval_method"] = "set by assumption (Scenario)"
        bear["case_role"] = "BEAR CASE ER* (D50): zero rent; fade speed = break-adjusted Andrews lambda"
        bear["evidence_class"] = "Scenario"
        c = pd.concat([c, bear], ignore_index=True)
    c["coe_series"] = np.where(c["specification"].str.startswith("robust_"),
                               "robustness specification: " + c["specification"].str.replace("robust_", "", regex=False),
                               "primary COE / ER series")
    c.loc[c["specification"] == "robust_beta_2012_replaced", "coe_series"] = "D51 sensitivity: 2012 beta replaced"
    c["selection"] = "candidate only: values are chosen in prompts 08/09a, not here (D8)"
    cols = ["bank", "bank_id", "specification", "case_role", "description", "years", "n", "primary_method",
            "lambda", "lambda_lo", "lambda_hi", "er_star", "er_star_lo", "er_star_hi", "er_star_interval_method",
            "er_star_definition", "half_life", "half_life_status", "lambda_ols", "lambda_std_lo", "lambda_std_hi",
            "lambda_kilian", "lambda_kilian_lo", "lambda_kilian_hi", "lambda_andrews", "lambda_andrews_lo",
            "lambda_andrews_hi", "er_star_std", "er_star_kilian", "er_star_andrews", "interval", "seed", "coe_series",
            "selection", "evidence_class"]
    return c[cols].reset_index(drop=True)


# --- outputs --------------------------------------------------------------------------------------------------------


NOTE = """# Chapter 5: persistence and economic rents

Generated by `tzbank.chapters.ch05.persistence` (stage `persistence`). Blueprint §2 (core definitions, persistence
model) and §4 chapter 5; hypotheses H5.1-H5.3; guardrails D12 (CLAUDE.md §8); D8 (no calibration choice here).

## Research question

Do the banks earn returns above the cost of equity, how persistent are those excess returns, and what long-run rent
level and fade speed do the data support?

## Variable definitions

- COE_t (approach A) = (long-bond yield - sovereign default spread) + beta x (mature ERP + CRP) + illiquidity premium;
  `ch05_coe_annual.csv`, with COE +/- 200 bp and the D41 term-spread-adjusted alternative for FY2008-12.
- ROE = attributable profit / average attributable equity (Group) for NMB and CRDB; net profit / average total equity
  for B03-B07 (Bank = Group).
- ER = ROE - COE; RI = attributable profit - COE x opening equity (TZS).
- AR(1): ER_t = alpha + lambda ER_t-1 + e; ER* = alpha / (1 - lambda); half-life = ln 0.5 / ln lambda (quoted only
  when 0 < lambda < 1 and the interval excludes 1).

## Sources and transformation trail

`data/processed/coe_inputs_annual.csv` (D34, D41), `data/interim/inputs_macroeconomic_data.csv` (Damodaran rows
MA-0016 to MA-0021, MA-0038 to MA-0043), `data/processed/assumptions.csv` (LP), `data/processed/bank_year_wide.csv`,
`bank_period_long.csv` (original vintage, Bank basis), `bank_year_flags.csv`, `sector_totals.csv` (BoT FSI ROE),
`outputs/tables/ch04_4_2_funding.csv` (H5.3). Functions: `coe.coe_annual`, `persistence.er_panel`,
`persistence.persistence_table`, `models.estimate`, `models.chow`, `models.sup_f`, `models.pooled_fe`.

## Validation test

`tests/test_ch05.py`: COE and ER/RI hand cases; the AR(1) estimator recovers a known lambda from simulated data within
its interval; bootstrap intervals reproduce with the seed; every printed half-life passes the validity rule; MBP
FY2013 and MUCOBA FY2020 are outside the estimation sample.

## Results

### Cost of equity (Estimated)

{coe}

### Excess returns (Estimated)

{er}

### Small-sample inference (Estimated)

Coverage of the nominal 90% interval for lambda, {sims} simulated AR(1) samples of 16 pairs per true lambda:

| true lambda | method | coverage | mean width |
| --- | --- | --- | --- |
{coverage}

**Primary method for lambda: {primary}** (coverage closest to 90% on average). The other two are kept as
alternatives in `ch05_persistence.csv` (`lambda_ols`, `lambda_std_*`, `lambda_kilian*`, `lambda_andrews*`). The
half-life rule is applied to the primary method's point estimate and interval.

### Persistence, AR(1) benchmark (Estimated)

Primary method {primary}; standard moving-block bootstrap (block length {block}, {reps} replicates, seed {seed})
kept as an alternative.

{ar1}

### Long-run excess return: through-the-cycle and break-adjusted (Estimated)

Cases (D50, to be confirmed at 08/09a): **base** ER* = through-the-cycle mean excess return (stress years
included); **bull** ER* = break-adjusted ER*, which **excludes the 2015-18 stress period**; **bear** ER* = 0 (ROE fades
to COE). The fade speed in all three cases is the break-adjusted lambda (Andrews). Lambda verdicts use the
break-adjusted model where a known-date break is found (D46 as amended).

{ttc}

### Sensitivity: 2012 EM bank beta (D51) (Estimated)

D51 is a sensitivity, not a change to the primary series: the primary COE, ER, persistence estimates and every
case-role row in `ch05_calibration_candidates.csv` keep the January 2012 beta (flagged). In the sensitivity the January
2012 EM bank beta (0.31) is replaced by the January 2013 value in FY2008-12 (`ch05_beta_sensitivity.csv`):

{beta_sens}

### Break tests (Estimated)

{breaks}

### Alternative specifications and robustness (Estimated)

{alts}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

COE, ER, RI and all persistence estimates are **Estimated** (COE rests on assumed beta, ERP, CRP and LP). Hypothesis
verdicts are **Interpretation**. Calibration candidates are **Estimated** and are not chosen here (D8).

## Limitations

- About 15-17 annual pairs per bank for NMB and CRDB and fewer for the smaller banks: intervals are wide and the
  bootstrap quantifies uncertainty without creating independent observations (blueprint §2).
- Small-sample inference: the moving-block bootstrap of the OLS AR(1) undercovers badly with 16 pairs (see the
  coverage table); the primary method ({primary}) is chosen by coverage and is still only approximately exact
  (Andrews with an intercept dummy uses a plug-in shift; Kilian assumes i.i.d. residuals).
- Damodaran inputs (D48): FY2016-24 default spread and CRP, FY2012-24 bank beta and FY2008-24 implied ERP are the
  archived January versions public at each year-end; the July mid-year updates are not archived. Tanzania has no
  Damodaran estimate before the January 2016 release (unrated), so FY2008-15 use that earliest estimate, and FY2008-11
  use the January 2012 beta (0.31, an outlier against 0.61-0.85 in other versions): all flagged; the D51 sensitivity
  replaces it with the January 2013 value and leaves the primary series unchanged. The previous
  constant-vintage COE is kept as a robustness specification.
- A parallel COE shift leaves lambda unchanged and moves ER* one-for-one; COE errors that vary by year (the
  Damodaran vintage, the D41 10-year fallback) do affect lambda (see the D41 row).
- FY2008-12 COE uses the 10-year rate (D41, flagged).
- Bank-basis figures for NMB and CRDB are held for few years, so the Bank-vs-Group check is mostly not estimable.
- BoT sector ROE is held for 2019-25 only; the sector-benchmark specification is short.
- H5.3 is provisional: the chapter 4.2 cost of deposits is a gross proxy; the net deposit franchise comes in chapter 6.

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

Persistent positive ER* for the two large banks with mean reversion is consistent with **low-cost deposits**,
**scale economies** and **limited competition** sustaining rents that erode only slowly; a break around the 2015-18
episode points to **temporary credit-cycle conditions** in the level of returns. These estimates do not separate the
explanations; chapters 6-8 do.
"""


def _fmt_rows(df: pd.DataFrame, t: dict[str, str]) -> str:
    lines = []
    for _, r in df.iterrows():
        if r.get("status") != "estimated":
            lines.append(f"- {t.get(r['bank_id'], r['bank_id'])}: {r.get('status', 'not estimated')}")
            continue
        hl = f"half-life {r['half_life']:.1f} years" if pd.notna(r["half_life"]) else r["half_life_status"]
        lines.append(f"- {t.get(r['bank_id'], r['bank_id'])} ({r['years']}, n={int(r['n'])}): lambda "
                     f"{r['lambda']:.2f} [{r['lambda_lo']:.2f}, {r['lambda_hi']:.3f}]; ER* "
                     f"{pct(r['er_star'], reason='undefined: lambda >= 1')} "
                     f"[{pct(r['er_star_lo'], reason='undefined')}, {pct(r['er_star_hi'], reason='undefined')}]; {hl}.")
    return "\n".join(lines)


def _figures(root: Path, panel: pd.DataFrame, coe: pd.DataFrame, t: dict[str, str]) -> list[dict]:
    figd = path("figures", root)
    figs = []
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in sorted(panel["bank_id"].unique()):
        d = panel[(panel["bank_id"] == b) & panel["in_estimation"]]
        ax.plot(d["fiscal_year"], d["er"] * 100, color=COLORS.get(b), lw=2.2 if b in FOCUS else 0.9, marker="o",
                ms=3 if b in FOCUS else 2, label=t.get(b, b))
    ax.axhline(0, color="#5a5a55", lw=0.8)
    style(ax, "Excess return on equity (ROE - COE), estimation sample", "percentage points")
    ax.legend(ncol=7, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch05_excess_returns.png", "Evidence class: Estimated. COE approach A (D34/D41; Damodaran "
               "2025 vintage before FY2025, flagged). Excludes MBP FY2013 and MUCOBA FY2020.")
    data = write_csv(panel[panel["in_estimation"]][["bank_id", "fiscal_year", "roe", "coe", "er", "evidence_class"]],
                     figd / "ch05_excess_returns_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Excess returns by bank", "evidence_class": "Estimated"})
    c = coe[coe["date_label"].str.startswith("FY")]
    fig, ax = plt_subplots(figsize=(8, 4.4))
    ax.plot(c["year"], c["coe"] * 100, color="#1f1f1d", lw=2.2, label="COE (approach A)")
    ax.fill_between(c["year"], c["coe_minus_200bp"] * 100, c["coe_plus_200bp"] * 100, color="#e4e3dd",
                    label="COE +/- 200 bp")
    ax.plot(c["year"], c["y_long"] * 100, color="#2a78d6", lw=1.2, label="long-bond yield")
    ax.plot(c["year"], (c["y_long"] - c["default_spread"]) * 100, color="#2a78d6", lw=1, ls="--",
            label="yield net of default spread")
    ax.plot(c["year"], c["coe_d41_adjusted"] * 100, color="#eb6834", lw=1.2, ls=":", label="D41 adjusted (2008-12)")
    style(ax, "Annual cost of equity, approach A", "percent")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch05_coe_annual.png", "Evidence class: Estimated. (y - DS) + beta (ERP + CRP) + LP; "
               "FY2008-12 10-year rate (D41, flagged).")
    data = write_csv(c[["year", "coe", "coe_minus_200bp", "coe_plus_200bp", "y_long", "default_spread",
                        "coe_d41_adjusted", "evidence_class"]], figd / "ch05_coe_annual_data.csv",
                     require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Annual cost of equity", "evidence_class": "Estimated"})
    return figs


def beta_sensitivity(root: Path, panel: pd.DataFrame, coe: pd.DataFrame, p: pd.DataFrame, bt: pd.DataFrame,
                     primary: str, t: dict[str, str], band: float) -> pd.DataFrame:
    """D51: re-estimate with the January 2012 beta replaced by the January 2013 value (FY2008-12) and compare the
    lambda / ER* of the case specifications and the H5.1-H5.3 verdicts with the primary run."""
    alt = panel.copy()
    alt["coe"] = alt["coe_beta2012_replaced"]
    alt["er"] = alt["roe"] - alt["coe"]
    pa, bta = persistence_table(root, alt, coe.assign(coe=coe["coe_beta2012_replaced"]), primary)
    rows = []
    for b in FOCUS:
        for spec in ("ar1_benchmark", "ar1_break_episode", "through_cycle_mean"):
            for run, tab in (("primary", p), ("beta 2012 replaced", pa)):
                r = tab[(tab["bank_id"] == b) & (tab["specification"] == spec)]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"item": f"{t[b]} {spec}", "run": run, "lambda": r.get("lambda"),
                                 "lambda_lo": r.get("lambda_lo"), "lambda_hi": r.get("lambda_hi"),
                                 "er_star": r.get("er_star"), "er_star_lo": r.get("er_star_lo"),
                                 "er_star_hi": r.get("er_star_hi"), "half_life": r.get("half_life")})
    for run, (tab, br) in (("primary", (p, bt)), ("beta 2012 replaced", (pa, bta))):
        h = hypotheses(tab, br, t, root, band)
        for r in h.itertuples():
            rows.append({"item": r.hypothesis, "run": run, "verdict": r.verdict})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Estimated"
    return out


def _sens_text(sens: pd.DataFrame) -> str:
    out = []
    for item, g in sens.groupby("item", sort=False):
        g = g.set_index("run")
        if "verdict" in g.columns and g["verdict"].notna().all():
            out.append(f"- {item}: {g.loc['primary', 'verdict']} -> {g.loc['beta 2012 replaced', 'verdict']}")
        else:
            a, b = g.loc["primary"], g.loc["beta 2012 replaced"]
            lam = (f"lambda {a['lambda']:.2f} [{a['lambda_lo']:.2f}, {a['lambda_hi']:.2f}] -> {b['lambda']:.2f} "
                   f"[{b['lambda_lo']:.2f}, {b['lambda_hi']:.2f}]; " if pd.notna(a["lambda"]) else "")
            out.append(f"- {item}: {lam}ER* {pct(a['er_star'], reason='undefined')} -> "
                       f"{pct(b['er_star'], reason='undefined')}")
    return "\n".join(out)


def _er_both(p: pd.DataFrame, t: dict[str, str]) -> list[str]:
    out = []
    for b in FOCUS:
        for spec, label in (("ar1_benchmark", "through-the-cycle AR(1) (stress years included)"),
                            ("through_cycle_mean", "through-the-cycle mean ER, no persistence model"),
                            ("ar1_break_episode", "break-adjusted (excludes the 2015-18 stress period)")):
            r = p[(p["bank_id"] == b) & (p["specification"] == spec)]
            if len(r) and r["status"].iloc[0] == "estimated":
                r = r.iloc[0]
                lam = (f"; lambda {r['lambda']:.2f} [{r['lambda_lo']:.2f}, {r['lambda_hi']:.3f}]"
                       if pd.notna(r.get("lambda")) else "")
                alt = (f" (alternatives: OLS ER* {pct(r['er_star_std'], reason='undefined')}, Kilian ER* "
                       f"{pct(r['er_star_kilian'], reason='undefined')})" if pd.notna(r.get("lambda")) else "")
                out.append(f"- {t[b]}, {label}: ER* {pct(r['er_star'], reason='undefined: lambda at 1')} "
                           f"[{pct(r['er_star_lo'], reason='unbounded')}, {pct(r['er_star_hi'], reason='unbounded')}]"
                           f"{lam}{alt}.")
    return out


def write_persistence(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    cfg, seed = settings(root)["persistence"], settings(root)["random_seed"]
    coe = coe_annual(root)
    panel = er_panel(root, coe)
    cov = models.coverage_study(tuple(cfg["coverage_lambdas"]), cfg["coverage_pairs"], cfg["coverage_sims"],
                                cfg["coverage_reps"], cfg["block_length"], cfg["interval"], int(seed),
                                cfg["andrews_sims"])
    cov["evidence_class"] = "Estimated"
    primary = models.primary_method(cov)
    cov["primary"] = cov["method"] == primary
    p, bt = persistence_table(root, panel, coe, primary)
    sign, rank = sign_and_rank(panel)
    sens = beta_sensitivity(root, panel, coe, p, bt, primary, t, cfg["coe_band_bp"] / 10_000)
    hyp = hypotheses(p, bt, t, root, cfg["coe_band_bp"] / 10_000)
    cal = calibration_candidates(p, t)
    tab = path("tables", root)
    files = [write_csv(coe, tab / "ch05_coe_annual.csv", require_evidence=True),
             write_csv(panel, tab / "ch05_excess_returns.csv", require_evidence=True),
             write_csv(p, tab / "ch05_persistence.csv", require_evidence=True),
             write_csv(bt, tab / "ch05_break_tests.csv", require_evidence=True),
             write_csv(sign, tab / "ch05_sign_persistence.csv", require_evidence=True),
             write_csv(rank, tab / "ch05_rank_persistence.csv", require_evidence=True),
             write_csv(hyp, tab / "ch05_hypotheses.csv", require_evidence=True),
             write_csv(cal, tab / "ch05_calibration_candidates.csv", require_evidence=True),
             write_csv(cov, tab / "ch05_interval_coverage.csv", require_evidence=True),
             write_csv(sens, tab / "ch05_beta_sensitivity.csv", require_evidence=True),
             write_csv(ifrs9_sensitivity(p, bt), tab / "ch05_ifrs9_sensitivity.csv", require_evidence=True)]
    figs = _figures(root, panel, coe, t)
    files += [f["path"] for f in figs] + [f["data"] for f in figs]
    c = coe[coe["date_label"].str.startswith("FY")]
    coe_txt = (f"- COE (approach A) ranges {pct(c['coe'].min())} to {pct(c['coe'].max())} over FY2008-25 (previous "
               f"constant 2025-vintage series: {pct(c['coe_2025_vintage'].min())} to "
               f"{pct(c['coe_2025_vintage'].max())}); FY2025 {pct(c.loc[c['year'] == 2025, 'coe'].iloc[0])}. "
               "FY2025 inputs (inputs workbook): default spread "
               f"{pct(c['default_spread'].iloc[-1], 2)}, CRP {pct(c['crp'].iloc[-1], 2)}, mature ERP "
               f"{pct(c['erp_mature'].iloc[-1], 2)}, beta {c['beta_levered'].iloc[-1]:.2f}, LP "
               f"{pct(c['illiquidity_premium'].iloc[-1], 1)}.\n- {int((c['flags'] != 'none').sum())} of {len(c)} years "
               "carry a flag (Damodaran nearest archived version; D41 10-year rate; beta outlier).")
    est = panel[panel["in_estimation"]]
    er_lines = []
    for b in sorted(est["bank_id"].unique()):
        d = est[est["bank_id"] == b]
        s = sign[sign["bank_id"] == b].iloc[0]
        er_lines.append(f"- {t.get(b, b)} ({int(d['fiscal_year'].min())}-{int(d['fiscal_year'].max())}): mean ER "
                        f"{pct(d['er'].mean())}, ER > 0 in {s['share_positive']:.0%} of years; P(ER > 0 | ER_t-1 > 0) "
                        f"{pct(s['p_positive_after_positive'], 0, reason='no positive years')}; cumulative RI TZS "
                        f"{d['ri'].sum() / 1e9:,.0f}bn.")
    excl = panel[~panel["in_estimation"] & panel["roe"].notna()]
    er_lines.append("- Outside the estimation sample: " + "; ".join(
        f"{t.get(r.bank_id, r.bank_id)} FY{int(r.fiscal_year)} ({r.excluded_reason})" for r in excl.itertuples()))
    bench = p[p["specification"] == "ar1_benchmark"]
    br = []
    for b in sorted(bt["bank_id"].unique()):
        x = bt[bt["bank_id"] == b]
        br.append(f"- {t.get(b, b)}: " + "; ".join(
            f"{r.test}{' at FY' + str(int(r.break_year)) if pd.notna(r.break_year) and 'sup' in r.test else ''}: "
            + (f"F = {r.F:.2f}, p = {r.p_value:.2f}" if pd.notna(r.F) else r.method.split("; ")[-1])
            for r in x.itertuples()))
    alts = []
    for b in FOCUS:
        alts.append(f"**{t[b]}**")
        for _, r in p[(p["bank_id"] == b) & (p["specification"] != "ar1_benchmark")].iterrows():
            if r["status"] == "estimated":
                alts.append(f"- {r['specification']} ({r['years']}, n={int(r['n'])}): lambda {r['lambda']:.2f} "
                            f"[{r['lambda_lo']:.2f}, {r['lambda_hi']:.3f}]; ER* {pct(r['er_star'], reason='undefined')} "
                            f"[{pct(r['er_star_lo'], reason='undefined')}, "
                            f"{pct(r['er_star_hi'], reason='undefined')}]; {r['half_life_status']}")
            else:
                alts.append(f"- {r['specification']}: {r['status']}")
    fe = p[p["specification"] == "pooled_fixed_effects"].iloc[0]
    alts.append(f"\n**Pooled fixed effects (secondary, Nickell bias):** lambda {fe['lambda']:.2f} "
                f"[{fe['lambda_lo']:.2f}, {fe['lambda_hi']:.2f}], n = {int(fe['n'])}.")
    r1 = rank[rank["lag"] == 1]
    alts.append(f"\n**Rank persistence:** mean Spearman correlation of bank ER ranks year to year "
                f"{r1['spearman'].mean():+.2f} ({len(r1)} year pairs); 3 years apart "
                f"{rank.loc[rank['lag'] == 3, 'spearman'].mean():+.2f}; 5 years apart "
                f"{rank.loc[rank['lag'] == 5, 'spearman'].mean():+.2f}.")
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Against: {r.against}. "
                        f"Rule: {r.rule}." for r in hyp.itertuples())
    interp = ("**Interpretation.** Read lambda and ER* together with their intervals and the "
              "break tests: where a break is found, the full-sample AR(1) mixes regimes and the sub-period rows are "
              "the better description. The calibration candidates file lists every specification for NMB and CRDB; "
              "the choice of sustainable ROE and fade speed is made in prompts 08/09a (D8), against the chapter 10 "
              "iso-value surface.")
    text = NOTE.format(coe=coe_txt, er="\n".join(er_lines), block=cfg["block_length"], reps=cfg["bootstrap_reps"],
                       seed=seed, ar1=_fmt_rows(bench, t), breaks="\n".join(br), alts="\n".join(alts),
                       hyp=hyp_txt, interp=interp, primary=primary,
                       coverage="\n".join(f"| {r.true_lambda:.1f} | {r.method} | {r.coverage:.0%} | {r.mean_width:.2f} |"
                                          for r in cov.itertuples()), sims=cfg["coverage_sims"],
                       ttc="\n".join(_er_both(p, t)), beta_sens=_sens_text(sens))
    nf = write_text(text, path("notes", root) / "ch05.md")
    files.append(nf)
    desc = {"ch05_coe_annual.csv": "Annual cost of equity (approach A), inputs and sources, +/- 200 bp",
            "ch05_excess_returns.csv": "ER and RI per bank-year, estimation flag and exclusion reason",
            "ch05_persistence.csv": "Persistence estimates by bank and specification (lambda, ER*, half-life)",
            "ch05_break_tests.csv": "Chow (FY2018, 2015-18) and sup-F break tests",
            "ch05_sign_persistence.csv": "Sign persistence of ER by bank",
            "ch05_rank_persistence.csv": "Rank persistence across banks (Spearman, lags 1, 3, 5)",
            "ch05_hypotheses.csv": "H5.1-H5.3 verdicts",
            "ch05_calibration_candidates.csv": "Calibration candidates for NMB and CRDB (not chosen, D8)",
            "ch05_interval_coverage.csv": "Coverage of the 90% interval for lambda: standard, Kilian, Andrews",
            "ch05_beta_sensitivity.csv": "Sensitivity: 2012 EM bank beta replaced by the 2013 value (D51)",
            "ch05_ifrs9_sensitivity.csv": "FY2018 (IFRS 9 transition) in and out: lambda, ER* and the episode break "
                                          "(prompt 24)"}
    m = {"chapter": "05", "title": "Persistence and economic rents", "hypotheses": ["H5.1", "H5.2", "H5.3"],
         "status": "built - calibration not chosen (D8)",
         "tables": [{"path": rel(f, root), "description": desc[f.name], "evidence_class":
                     "Interpretation" if "hypotheses" in f.name else "Estimated", "note": rel(nf, root)}
                    for f in files[:10]],
         "figures": [{"path": rel(f["path"], root), "data": rel(f["data"], root), "description": f["description"],
                      "evidence_class": f["evidence_class"], "note": rel(nf, root)} for f in figs],
         "notes": [rel(nf, root)],
         "inputs": ["data/processed/coe_inputs_annual.csv", "data/interim/inputs_macroeconomic_data.csv",
                    "data/processed/assumptions.csv", "data/processed/bank_year_wide.csv",
                    "data/processed/bank_period_long.csv", "data/processed/bank_year_flags.csv",
                    "data/processed/sector_totals.csv", "outputs/tables/ch04_4_2_funding.csv"],
         "module": "tzbank.chapters.ch05.persistence", "run_log_stage": "persistence"}
    files.append(write_json(m, path("manifests", root) / "ch05.json"))
    return files


__all__ = ["er_panel", "persistence_table", "hypotheses", "calibration_candidates", "write_persistence"]
