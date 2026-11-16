"""Prompt 06: chapter 7 (capital allocation and incremental returns)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch07_capital import payout, raises, rore
from tzbank.config import find_root, path


def _w():
    return pd.DataFrame({
        "bank_id": "B99", "fiscal_year": [2018, 2019, 2020, 2021], "profit_attributable": [100.0, 110.0, 120.0, 150.0],
        "equity_attributable": [800.0, 860.0, 925.0, 1000.0], "shares_outstanding": [10.0] * 4,
        "dps": [5.0, 5.0, 6.0, 7.0], "eps_basic": [10.0, 11.0, 12.0, 15.0], "total_assets": [5000.0] * 4})


def test_rore_hand_case_and_raise_exclusion():
    w = _w()
    pay = payout(w)
    assert pay["payout_total"].iloc[0] == pytest.approx(50 / 100)
    rz = pd.DataFrame(columns=["bank_id", "fiscal_year", "amount", "kind"])
    r = rore(w, pay, rz, {}).set_index(["k", "fiscal_year"]).loc[(3, 2021)]
    # (150 - 100) / ((100 - 50) + (110 - 50) + (120 - 60)) = 50 / 170
    assert r["rore_primary"] == pytest.approx(50 / 170)
    assert r["d_roe"] == pytest.approx(50 / 200)
    assert r["rore_eps_supplementary"] == pytest.approx(5 / ((10 - 5) + (11 - 5) + (12 - 6)))
    rz = pd.DataFrame([{"bank_id": "B99", "fiscal_year": 2020, "amount": 30.0, "kind": "rights issue"}])
    r = rore(w, pay, rz, {}).set_index(["k", "fiscal_year"]).loc[(3, 2021)]
    assert np.isnan(r["rore_primary"]) and np.isnan(r["d_roe"])  # unadjusted raise in the window: excluded
    assert r["rore_adjusted"] == pytest.approx(50 / 200)


def test_primary_rore_excludes_every_raise_window(root):
    r = pd.read_csv(path("tables", root) / "ch07_rore.csv", dtype={"bank_id": str})
    rz = raises()
    for x in rz.itertuples():
        hit = r[(r["bank_id"] == x.bank_id) & (r["fiscal_year"] - r["k"] < x.fiscal_year)
                & (r["fiscal_year"] >= x.fiscal_year)]
        assert hit["rore_primary"].isna().all() and (hit["capital_raised_in_window"] > 0).all()
        assert hit["d_roe"].isna().all()
    assert len(r[(r["bank_id"] == "B01") & (r["capital_raised_in_window"] > 0)]) >= 5  # the FY2015 windows exist
    clean = r[(r["capital_raised_in_window"] == 0) & r["bank_id"].isin(["B01", "B02"]) & r["d_ni_attr"].notna()]
    assert clean["rore_primary"].notna().mean() > 0.8


def test_headroom_uses_d2_d17_basis(root):
    c = pd.read_csv(path("tables", root) / "ch07_capital.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    assert c.loc[("B02", 2025), "capital_basis"].startswith("Bank-solo (D2)")
    assert c.loc[("B01", 2014), "capital_basis"].startswith("Bank-solo (D17)")
    assert c.loc[("B01", 2015), "capital_basis"] == "Group"
    cl = pd.read_csv(path("processed", root) / "capital_long.csv", low_memory=False, dtype={"bank_id": str})
    nmb = cl[(cl["bank_id"] == "B02") & (cl["canonical_name"] == "tier1_ratio") & (cl["basis"] == "BANK")
             & (cl["fiscal_year"] == 2025) & (cl["period_type"] == "FY")]
    assert c.loc[("B02", 2025), "tier1_ratio"] == pytest.approx(nmb["value"].iloc[-1])
    x = c.dropna(subset=["tier1_ratio", "tier1_requirement"])
    assert np.allclose(x["tier1_headroom"], x["tier1_ratio"] - x["tier1_requirement"])
    assert (x.loc[[i for i in x.index if i[1] >= 2018], "tier1_requirement"] >= 0.125 - 1e-12).all()


def test_leverage_decomposition_adds_up(root):
    lv = pd.read_csv(path("tables", root) / "ch07_leverage.csv")
    r = lv["residual"].dropna()
    assert len(r) > 8 and (r.abs() < 1e-9).all()


def test_ch07_manifest_and_hypotheses(root):
    m = json.loads((path("manifests", root) / "ch07.json").read_text(encoding="utf-8"))
    assert m["chapter"] == "07"
    for t in m["tables"] + m["figures"]:
        assert (root / t["path"]).exists()
    h = pd.read_csv(path("tables", root) / "ch07_hypotheses.csv")
    assert list(h["hypothesis"]) == ["H7.1", "H7.2", "H7.3"]
    note = (path("notes", root) / "ch07.md").read_text(encoding="utf-8")
    for head in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                 "## Evidence class", "## Limitations", "## Cross-cutting explanations"):
        assert head in note
    assert "nan%" not in note
    p = pd.read_csv(path("tables", root) / "ch07_payout.csv", dtype={"bank_id": str}).set_index(["bank_id", "fiscal_year"])
    assert p.loc[("B02", 2025), "payout_special"] > 0  # NMB FY2025 special dividend shown separately


# --- prompt 07b -------------------------------------------------------------------------------------------------------


def test_rore_window_list_reconciles(root):
    w = pd.read_csv(path("tables", root) / "ch07_rore_windows.csv", dtype={"bank_id": str})
    r = pd.read_csv(path("tables", root) / "ch07_rore.csv", dtype={"bank_id": str})
    r = r[r["bank_id"].isin(["B01", "B02"])]
    assert len(w) == len(r)  # every NMB and CRDB window listed
    m = w.merge(r[["bank_id", "k", "window", "rore_primary"]], on=["bank_id", "k", "window"], suffixes=("", "_r"))
    assert (m["primary_included"] == m["rore_primary_r"].notna()).all()
    c5 = w[(w["bank_id"] == "B01") & (w["k"] == 5)]
    by_rule = c5[~c5["primary_reason"].str.contains("(D52)", regex=False)]
    assert len(by_rule) == 7 and c5["primary_included"].sum() == 6  # 7 pass the (t-k, t] rule, 6 computed
    gap = by_rule[~by_rule["primary_included"]]
    assert list(gap["window"]) == ["2009-2014"] and "FY2009" in gap["primary_reason"].iloc[0]
    assert (w["stress_start_flag"] == w["start_year"].between(2015, 2018)).all()
    s = pd.read_csv(path("tables", root) / "ch07_rore_summary.csv", dtype={"bank_id": str})
    assert {"rore_primary", "rore_adjusted", "d_roe", "d_roe_adjusted", "rore_eps_supplementary"} <= set(s["series"])
    assert set(s["k"]) == {3, 5}
    x = s[(s["bank_id"] == "B01") & (s["k"] == 5) & (s["series"] == "rore_adjusted")].iloc[0]
    assert x["median"] == pytest.approx(c5["rore_adjusted"].median())
    assert x["median_excl_stress_start"] == pytest.approx(c5.loc[~c5["stress_start_flag"], "rore_adjusted"].median())


def test_requirements_earlier_ccb_date_and_readings(root):
    q = pd.read_csv(path("tables", root) / "ch07_capital_requirements.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    assert q.loc[("B02", 2017), "tier1_requirement"] == pytest.approx(0.125)  # Aug 2017 start (primary)
    assert q.loc[("B02", 2017), "tier1_requirement_later_ccb"] == pytest.approx(0.10)  # Aug 2018 start
    assert q.loc[("B02", 2016), "tier1_requirement"] == pytest.approx(0.10)
    assert q.loc[("B02", 2025), "total_requirement"] == pytest.approx(0.145)  # 12% stated + 2.5% CCB
    for y in range(2015, 2026):
        assert q.loc[("B01", y), "tier1_stated"] == pytest.approx(0.125)
        assert q.loc[("B01", y), "total_stated"] == pytest.approx(0.145)
        assert q.loc[("B01", y), "tier1_requirement"] == pytest.approx(0.125)  # reading A
        exp_b = 0.15 if y >= 2017 else 0.125
        assert q.loc[("B01", y), "tier1_requirement_reading_b"] == pytest.approx(exp_b)
        assert q.loc[("B01", y), "total_requirement_reading_b"] == pytest.approx(exp_b + 0.02)
        assert "reading A primary" in q.loc[("B01", y), "requirement_flag"]
    for y in range(2015, 2026):  # NMB: readings coincide
        assert q.loc[("B02", y), "tier1_requirement_reading_b"] == pytest.approx(q.loc[("B02", y), "tier1_requirement"])


def test_total_headroom_and_binding_constraint(root):
    c = pd.read_csv(path("tables", root) / "ch07_capital.csv", dtype={"bank_id": str})
    x = c.dropna(subset=["total_capital_ratio", "total_requirement"])
    assert np.allclose(x["total_headroom"], x["total_capital_ratio"] - x["total_requirement"])
    both = x.dropna(subset=["tier1_headroom"])
    exp = np.where(both["total_headroom"] < both["tier1_headroom"], "total capital", "Tier 1")
    assert (both["binding_constraint"] == exp).all()
    assert np.allclose(both["binding_headroom"], both[["tier1_headroom", "total_headroom"]].min(axis=1))
    binds = c[c["bank_id"].isin(["B01", "B02"]) & c["fiscal_year"].between(2019, 2025)
              & (c["binding_constraint"] == "total capital")]
    d = pd.read_csv(path("tables", root) / "ch07_distributable_capacity_total.csv", dtype={"bank_id": str})
    if len(binds):  # prompt 07b task 3: the total-capital version exists when total capital binds
        assert set(d["measure"]) == {"total"} and set(d["bank_id"]) == {"B01", "B02"}
    t1 = pd.read_csv(path("tables", root) / "ch07_distributable_capacity.csv", dtype={"bank_id": str})
    g1 = t1.groupby("bank_id")["rwa_growth_2019_25_mean"].first()
    g2 = d.groupby("bank_id")["rwa_growth_2019_25_mean"].first()
    assert np.allclose(g1, g2)  # same growth assumption for both measures
    r = d[(d["bank_id"] == "B02") & (d["management_buffer"] == 0)].iloc[0]
    p = pd.read_csv(path("tables", root) / "ch07_payout.csv", dtype={"bank_id": str}).set_index(
        ["bank_id", "fiscal_year"])
    ni = p.loc[("B02", int(r["fiscal_year"])), "ni_attr"]
    # payout_max = 1 - (req_total + buffer - T2/RWA) x RWA x g / NI (Tier 2 held at its share of RWA)
    assert r["payout_max_steady_state"] == pytest.approx(
        1 - (r["target_ratio"] - r["tier2_share_of_rwa_held"]) * r["rwa"] * r["rwa_growth_2019_25_mean"] / ni)
    assert r["excess_capital_tzs"] == pytest.approx((r["ratio"] - r["target_ratio"]) * r["rwa"])


def test_requirement_statements_and_dsib_search(root):
    s = pd.read_csv(path("tables", root) / "ch07_requirement_statements.csv", dtype={"bank_id": str})
    cr = s[(s["bank_id"] == "B01") & (s["statement"] == "requirement as stated")]
    assert sorted(cr["fiscal_year"]) == list(range(2015, 2026))
    assert cr["quote"].str.contains("12.5%").all() and cr["pdf_page"].notna().all()
    ctx = s[s["statement"] == "context of the requirement"]
    assert ctx["quote"].str.contains("increased to 12.5%").any()
    d = pd.read_csv(path("tables", root) / "ch07_dsib_search.csv", dtype={"scope": str})
    rec = d[d["kind"] == "dsib_search"]
    assert set(rec["scope"]) == {"B01", "B02", "BoT"}  # every scope searched and recorded, hits or not
    hits = d[d["kind"] == "dsib_statement"]
    assert hits["quote"].str.contains("designat", case=False).any()  # NMB FY2025 D-SIB designation


def test_h72_block_bootstrap_reproducible():
    from tzbank.chapters.ch07_capital import slope_bootstrap

    rng = np.random.default_rng(1)
    x = pd.DataFrame({"bank_id": ["A"] * 12 + ["B"] * 12, "fiscal_year": list(range(2010, 2022)) * 2,
                      "ls": np.r_[np.linspace(1, 2, 12), np.linspace(2, 3, 12)]})
    x["rore_adjusted"] = -0.3 * x["ls"] + rng.normal(0, 0.01, 24)
    a = slope_bootstrap(x, 300, 3, [7, 72], 0.90)
    b = slope_bootstrap(x, 300, 3, [7, 72], 0.90)
    assert a == b
    assert a["pooled"][1] < 0  # a clear negative slope is detected
    h = pd.read_csv(path("tables", find_root()) / "ch07_hypotheses.csv")
    assert "block bootstrap" in h.loc[h["hypothesis"] == "H7.2", "evidence"].iloc[0]
    assert "NMB supported" in h.loc[h["hypothesis"] == "H7.1", "verdict_by_bank"].iloc[0] or \
        "NMB" in h.loc[h["hypothesis"] == "H7.1", "verdict_by_bank"].iloc[0]
