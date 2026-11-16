"""Prompt 04: chapter 5 (COE, ER/RI, persistence) and the pre-flight fixes from the 03b review."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch05 import models
from tzbank.chapters.ch05.coe import build_coe
from tzbank.config import path

# --- pre-flight fixes (03b review) --------------------------------------------------------------------------------


def test_40_capital_structure_split(root):
    r = pd.read_csv(path("tables", root) / "ch04_4_0_subsection_ranking.csv")
    subs = set(r["subsection"])
    assert {"4.7 leverage (TA/E)", "4.7 tax (1-t)", "4.7 NCI share"} <= subs
    assert "4.7" not in subs  # leverage is never lumped with tax and NCI again
    m = pd.read_csv(path("tables", root) / "ch04_business_model_map.csv")
    q6 = m[m["dimension"] == "question 6: capital structure"].iloc[0]
    assert q6["attribution_subsection"] == "4.7 leverage (TA/E)"


def test_staff_costs_corrected(root):
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])["staff_costs"]
    assert w[("B01", 2023)] == pytest.approx(338_243e6)  # note 23 Group total, PDF p.301
    assert w[("B01", 2016)] == pytest.approx(184_370e6)
    assert w[("B02", 2025)] == pytest.approx(379_674e6)  # income statement, employee benefits expense
    assert w[("B02", 2014)] == pytest.approx(116_693e6)  # was the income tax expense


def test_fee_type_shares_low_confidence(root):
    t = pd.read_csv(path("tables", root) / "ch04_4_4_transactions.csv")
    assert t["fee_type_share_confidence"].str.startswith("LOW").all()
    m = pd.read_csv(path("tables", root) / "ch04_business_model_map.csv")
    verdicts = m[m["winner"].isin(["NMB", "CRDB"])]
    assert not verdicts["evidence_for"].str.contains("digital", case=False).any()


# --- COE and ER/RI ------------------------------------------------------------------------------------------------


def test_coe_hand_case():
    # (12% - 4%) + 0.6 x (4% + 6%) + 1.5% = 15.5%
    assert build_coe(0.12, 0.04, 0.06, 0.04, 0.6, 0.015) == pytest.approx(0.155)


def test_coe_table(root):
    c = pd.read_csv(path("tables", root) / "ch05_coe_annual.csv")
    fy = c[c["date_label"].str.startswith("FY")]
    assert set(fy["year"]) == set(range(2008, 2026))
    rebuilt = (fy["y_long"] - fy["default_spread"]) + fy["beta_levered"] * (fy["erp_mature"] + fy["crp"]) \
        + fy["illiquidity_premium"]
    assert np.allclose(rebuilt, fy["coe"], atol=1e-12)
    assert np.allclose(fy["coe_plus_200bp"] - fy["coe"], 0.02) and np.allclose(fy["coe"] - fy["coe_minus_200bp"], 0.02)
    # D48: version public at each year-end; Tanzania absent before the January 2016 release -> flagged
    assert fy.loc[fy["year"] <= 2015, "flags"].str.contains("ds: nearest archived").all()
    assert not fy.loc[fy["year"].between(2016, 2024), "flags"].str.contains("ds: nearest|crp: nearest").any()
    a = pd.read_csv(root / "data" / "interim" / "damodaran_archive.csv")
    for k, col in (("ds", "default_spread"), ("crp", "crp"), ("erp", "erp_mature"), ("beta", "beta_levered")):
        x = a[a["input"] == k].set_index("fiscal_year")["value"]
        yrs = [y for y in range(2016, 2025) if y in x.index]
        assert np.allclose(fy.set_index("year").loc[yrs, col], x.loc[yrs]), k
    assert fy["coe_2025_vintage"].notna().all()
    assert fy.loc[fy["year"] <= 2012, "flags"].str.contains("D41").all()
    assert fy.loc[fy["year"] <= 2012, "coe_d41_adjusted"].notna().all()
    assert (c["evidence_class"] == "Estimated").all()


def test_er_ri_hand_and_exclusions(root):
    e = pd.read_csv(path("tables", root) / "ch05_excess_returns.csv", dtype={"bank_id": str})
    ok = e["roe"].notna()
    assert np.allclose(e.loc[ok, "er"], e.loc[ok, "roe"] - e.loc[ok, "coe"])
    r = e[ok & e["b_open"].notna()]
    assert np.allclose(r["ri"], r["ni"] - r["coe"] * r["b_open"])
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    nmb = e.set_index(["bank_id", "fiscal_year"]).loc[("B02", 2025)]
    avg = (w.loc[("B02", 2025), "equity_attributable"] + w.loc[("B02", 2024), "equity_attributable"]) / 2
    assert nmb["roe"] == pytest.approx(w.loc[("B02", 2025), "profit_attributable"] / avg)
    k = e.set_index(["bank_id", "fiscal_year"])
    assert not k.loc[("B04", 2013), "in_estimation"]  # MBP FY2013 D-flag
    assert not k.loc[("B07", 2020), "in_estimation"] and "MUCOBA FY2020" in k.loc[("B07", 2020), "excluded_reason"]
    assert pd.notna(k.loc[("B07", 2020), "er"])  # shown, not estimated


def test_preflight_no_open_outliers(root):
    o = pd.read_csv(root / "reports" / "validation" / "outliers.csv")
    assert o["status"].astype(str).str.startswith("verified").all()


# --- estimators: quality gates --------------------------------------------------------------------------------------


def _sim(lam: float, alpha: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = [alpha / (1 - lam)]
    for _ in range(n + 50):
        y.append(alpha + lam * y[-1] + rng.normal(0, 0.03))
    return pd.Series(y[-n:], index=range(1960, 1960 + n))


def test_ar1_recovers_known_lambda():
    er = _sim(0.6, 0.02, 80, 7)
    r = models.estimate(models.pairs(er), 2000, 3, 0.90, [1, 2, 3])
    assert r["lambda_lo"] < 0.6 < r["lambda_hi"]
    assert r["lambda"] == pytest.approx(0.6, abs=0.15)
    assert r["er_star"] == pytest.approx(0.05, abs=0.03)


def test_ar1_coverage_small_samples():
    hits = 0
    for s in range(40):
        r = models.estimate(models.pairs(_sim(0.5, 0.01, 17, 100 + s)), 500, 3, 0.90, [5, s])
        hits += r["lambda_lo"] < 0.5 < r["lambda_hi"]
    # measured about 63% at nominal 90% with 16 pairs: the bootstrap undercovers in small samples. The guard
    # catches further degradation; the note must disclose the measured coverage.
    assert hits >= 22


def test_coverage_study_and_primary_method(root):
    c = pd.read_csv(path("tables", root) / "ch05_interval_coverage.csv")
    assert set(c["true_lambda"]) == {0.3, 0.6, 0.9} and (c["n_pairs"] == 16).all()
    dev = (c["coverage"] - 0.90).abs().groupby(c["method"]).mean()
    prim = c.loc[c["primary"], "method"].unique()
    assert len(prim) == 1 and prim[0] == dev.idxmin()
    p = pd.read_csv(path("tables", root) / "ch05_persistence.csv")
    est = p[p["lambda_ols"].notna()]
    assert (est["primary_method"] == prim[0]).all()
    note = (path("notes", root) / "ch05.md").read_text(encoding="utf-8")
    assert f"Primary method for lambda: {prim[0]}" in note


def test_kilian_reduces_small_sample_bias():
    ols_b, kil_b = [], []
    for s in range(30):
        d = models.pairs(_sim(0.6, 0.01, 17, 300 + s))
        ols_b.append(models.ar1(d)["lambda"] - 0.6)
        kil_b.append(models.kilian(d, 300, 300, 0.90, [s])["lambda"] - 0.6)
    assert np.mean(ols_b) < -0.05  # OLS is biased towards zero in short samples
    assert abs(np.mean(kil_b)) < abs(np.mean(ols_b))


def test_andrews_median_unbiased():
    mu = [models.andrews(models.pairs(_sim(0.6, 0.01, 17, 500 + s)), 0.90, 1000, [7])["lambda"] for s in range(40)]
    ols = [models.ar1(models.pairs(_sim(0.6, 0.01, 17, 500 + s)))["lambda"] for s in range(40)]
    assert abs(np.median(mu) - 0.6) < abs(np.median(ols) - 0.6)
    r = models.andrews(models.pairs(_sim(0.6, 0.01, 17, 1)), 0.90, 1000, [7])
    assert r["lambda_lo"] <= r["lambda"] <= r["lambda_hi"] <= 1.0


def test_bootstrap_reproducible_with_seed():
    d = models.pairs(_sim(0.5, 0.01, 17, 3))
    a = models.bootstrap(d, 300, 3, [20261106, 1, 1])
    b = models.bootstrap(d, 300, 3, [20261106, 1, 1])
    c = models.bootstrap(d, 300, 3, [20261106, 1, 2])
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_half_life_rule():
    assert models.half_life(0.5, 0.8)[0] == pytest.approx(1.0)
    assert math.isnan(models.half_life(0.5, 1.0)[0])  # interval includes 1
    assert math.isnan(models.half_life(-0.2, 0.3)[0])
    assert math.isnan(models.half_life(1.02, 1.3)[0])


def test_every_printed_half_life_is_valid(root):
    for f in ("ch05_persistence.csv", "ch05_calibration_candidates.csv"):
        p = pd.read_csv(path("tables", root) / f)
        h = p[p["half_life"].notna()]
        assert ((h["lambda"] > 0) & (h["lambda"] < 1) & (h["lambda_hi"] < 1)).all(), f
        assert np.allclose(h["half_life"], np.log(0.5) / np.log(h["lambda"]))
    note = (path("notes", root) / "ch05.md").read_text(encoding="utf-8")
    p = pd.read_csv(path("tables", root) / "ch05_persistence.csv")
    valid = set(np.round(p.loc[p["half_life"].notna(), "half_life"], 1))
    for x in re.findall(r"half-life (\d+\.\d) years", note):
        assert float(x) in valid
    assert "nan%" not in note


def test_chow_detects_break():
    rng = np.random.default_rng(0)
    y = [0.0]
    for t in range(1, 30):
        y.append((0.10 if t >= 15 else -0.05) + 0.3 * y[-1] + rng.normal(0, 0.005))
    d = models.pairs(pd.Series(y, index=range(1990, 2020)))
    assert models.chow(d, d["year"].to_numpy() >= 2005)["p_value"] < 0.01
    s = models.sup_f(d, 99, 4, [1])
    assert s["break_year"] in (2004, 2005, 2006) and s["p_value"] < 0.05


def test_calibration_candidates_not_chosen(root):
    c = pd.read_csv(path("tables", root) / "ch05_calibration_candidates.csv")
    assert set(c["bank"]) == {"NMB", "CRDB"}
    assert c["selection"].str.contains("D8").all()
    assert not any(k in c.columns for k in ("chosen", "selected", "central"))
    assert {"ar1_benchmark", "ar1_break_episode", "robust_coe_plus_200bp"} <= set(c["specification"])
    role = c.set_index(["bank", "specification"])["case_role"]
    # D50: base = through-the-cycle mean, bull = break-adjusted ER*, bear = zero rent; fade = break-adjusted lambda
    assert role[("NMB", "through_cycle_mean")].startswith("BASE CASE")
    assert role[("NMB", "ar1_break_episode")].startswith("BULL CASE") and "EXCLUDES" in role[("NMB", "ar1_break_episode")]
    assert role[("NMB", "bear_zero_rent")].startswith("BEAR CASE")
    cc = c.set_index(["bank", "specification"])
    for b in ("NMB", "CRDB"):
        assert cc.loc[(b, "bear_zero_rent"), "er_star"] == 0
        assert cc.loc[(b, "bear_zero_rent"), "lambda"] == cc.loc[(b, "ar1_break_episode"), "lambda"]
    # prompt 07b task 6: every case-role row uses the primary series (D51 is a sensitivity only)
    cases = c[c["case_role"].str.contains("CASE")]
    assert (cases["coe_series"] == "primary COE / ER series").all()
    assert (c.loc[c["specification"] == "robust_beta_2012_replaced", "coe_series"].str.startswith("D51")).all()
    p = pd.read_csv(path("tables", root) / "ch05_persistence.csv").set_index(["bank_id", "specification"])
    for r in cases[cases["specification"] != "bear_zero_rent"].itertuples():
        assert r.er_star == pytest.approx(p.loc[(r.bank_id, r.specification), "er_star"], nan_ok=True)
    note = (path("notes", root) / "ch05.md").read_text(encoding="utf-8")
    assert "is a sensitivity" in note  # the public copy drops the decision id (prompt 26)


def test_ch05_manifest_and_hypotheses(root):
    m = json.loads((path("manifests", root) / "ch05.json").read_text(encoding="utf-8"))
    assert m["chapter"] == "05" and len(m["tables"]) == 10
    for t in m["tables"] + m["figures"]:
        assert (root / t["path"]).exists()
    h = pd.read_csv(path("tables", root) / "ch05_hypotheses.csv")
    assert list(h["hypothesis"]) == ["H5.1", "H5.2", "H5.3"]
    assert h["verdict"].str.match(r"^(conditionally supported|supported|rejected|inconclusive)").all()  # D90
    note = (path("notes", root) / "ch05.md").read_text(encoding="utf-8")
    for head in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                 "## Evidence class", "## Limitations", "## Cross-cutting explanations"):
        assert head in note
