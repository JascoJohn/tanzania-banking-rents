"""Prompt 13: report-support exhibits (chapter 13 page) and the draft check."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from tzbank.config import path
from tzbank.reporting.findings import latest_pack


def _t(root, n):
    return pd.read_csv(path("tables", root) / f"ch13_{n}.csv")


def test_roe_reconciliation_is_definitional(root):
    r = _t(root, "roe_reconcile")
    assert np.allclose(r["roe_engine_opening_ex_div"], r["np_attr_y1"] / r["opening_equity_ex_dividend"])
    # the step from reported to engine year-1 ROE = definition effect + like-for-like change, exactly
    assert np.allclose(r["roe_engine_opening_ex_div"] - r["reported_roe_anchor"],
                       r["definition_effect"] + r["like_for_like_change"])
    assert (r["definition_effect"] > 0).all()  # opening equity ex dividend < average equity cum dividend


def test_bridges_add_up(root):
    e = _t(root, "earnings_bridge")
    for _, g in e.groupby(["bank", "scenario"]):
        g = g.sort_values("step_no")
        assert g["tzs"].iloc[0] + g["tzs"].iloc[1:-1].sum() == pytest.approx(g["tzs"].iloc[-1], rel=1e-9)
    for n in ("roe_waterfall", "roe_sustainable_bridge"):
        w = _t(root, n)
        for _, g in w.groupby("bank"):
            g = g.sort_values("step_no")
            assert g["roe_pts"].iloc[0] + g["roe_pts"].iloc[1:-1].sum() == pytest.approx(g["roe_pts"].iloc[-1], abs=1e-9)


def test_rating_flips_change_the_rating(root):
    f = _t(root, "rating_flips")
    hit = f[f["shift_to_flip"].notna()]
    assert len(hit) > 0 and (hit["flipped_rating"] != hit["base_rating"]).all()
    assert (f.loc[f["shift_to_flip"].isna(), "flipped_rating"] == "no flip in range").all()


def test_stress_mixture_is_linear(root):
    s = _t(root, "stress_mixture")
    hand = (1 - s["stress_weight"]) * s["p_base_regime"] + s["stress_weight"] * s["p_stress_regime"]
    assert np.allclose(s["prob_value_above_price"], hand)


def test_financial_table_identities(root):
    f = _t(root, "financial_table")
    assert np.allclose(f["P/B"], f["price"] / f["BVPS"])
    assert np.allclose(f["dividend_yield"], f["DPS"] / f["price"])
    assert set(f["fiscal_year"].str[-1]) == {"A", "E"} and f.groupby("bank").size().eq(5).all()


def test_entry_price_meets_the_hurdle(root):
    u = _t(root, "usd_exports")
    e = u[u["measure"] == "entry price meeting the USD hurdle"]
    r = u[u["measure"] == "expected USD return"]
    assert len(e) == len(r) == 16
    assert (u[u["measure"].str.startswith("worst calendar-year")]["value"] < 0).all()


def test_ch13_manifest_and_pack(root):
    m = json.loads((path("manifests", root) / "ch13.json").read_text(encoding="utf-8"))
    assert m["run_log_stage"] == "support" and len(m["tables"]) >= 20
    p = latest_pack(root)
    pack = pd.read_csv(p, low_memory=False)
    ch13 = pack[pack["chapter"].astype(str) == "13"]
    assert len(ch13) > 100 and (ch13["dashboard_page"] == "dashboard/pages/13_report_support.py").all()
    for t in m["tables"]:
        assert (ch13["source_file"] == t["path"]).any() or pd.read_csv(root / t["path"]).select_dtypes(
            "number").empty, t["path"]


def test_fill_list_covers_every_draft(root):
    f = path("outputs", root) / "findings" / "fill_list.md"
    drafts = sorted((root / "drafts").glob("*.docx"))
    if not drafts:
        pytest.skip("no owner drafts held (drafts/ is git-ignored)")
    t = f.read_text(encoding="utf-8")
    for d in drafts:
        assert f"## {d.name}" in t
    if re.search(r"\[export\] [1-9]", t):  # owner drafts that still hold table-cell markers
        assert "[export] x" in t


# --- prompt 13b -------------------------------------------------------------------------------------------------------
def test_cr2_series_single_basis(root):
    c = _t(root, "cr2_profit_direct")
    allb = c[c["sector_basis"].str.startswith("all banking")]
    # D82 (edit pack A B3): a second numerator basis, NMB Group + CRDB Bank-solo, where the parent-bank lines are held
    solo = allb[allb["numerator_basis"].str.contains("Bank-solo")].set_index("year")
    assert set(range(2022, 2026)) <= set(solo.index) and solo.index.is_unique
    a = allb[allb["numerator_basis"].str.startswith("NMB + CRDB Group")].set_index("year")
    assert a.index.is_unique
    # hand check FY2025: the bank-solo numerator swaps CRDB Group PBT for the bank-only 1,000,650m (CRDB IR p.195)
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    crdb_grp = float(w[(w["bank_id"] == "B01") & (w["fiscal_year"] == 2025)]["profit_before_tax"].iloc[0])
    assert solo.loc[2025, "cr2_pbt"] == pytest.approx(a.loc[2025, "cr2_pbt"] - crdb_grp + 1_000_650e6)
    assert (solo["cr2_pbt_share"].dropna() < a.loc[solo.index, "cr2_pbt_share"].dropna()).all()  # Group = upper bound
    assert set(range(2011, 2026)) <= set(a.index)
    assert np.isnan(a.loc[2023, "cr2_pbt_share"]) and "NOT DISCLOSED" in a.loc[2023, "status"]
    ok = a.dropna(subset=["sector_pbt"])
    assert np.allclose(ok["cr2_pbt_share"], ok["cr2_pbt"] / ok["sector_pbt"])
    assert np.allclose(ok["cr2_asset_share"], ok["cr2_total_assets"] / ok["sector_total_assets"])
    assert (a["evidence_class"] == "Derived").all()
    assert (c[c["sector_basis"].str.startswith("commercial")]["evidence_class"] == "Estimated").all()


def test_surplus_drag_against_minimum_and_target(root):
    d = _t(root, "surplus_drag")
    assert d.groupby("bank").size().eq(3).all()
    hand = (d["tier1_ratio"] - d["hurdle_ratio"]).clip(lower=0) * d["rwa"]
    assert np.allclose(d["surplus_capital_tzs"], hand)
    assert np.allclose(d["drag_tzs"], d["surplus_capital_tzs"] * (d["coe"] - d["tbill_v13"] * (1 - d["tax"])))
    for _, g in d.groupby("bank"):
        g = g.set_index(g["hurdle"].str[:3])
        assert g.loc["(a)", "drag_tzs"] >= g.loc["(b)", "drag_tzs"]  # the minimum sits below the model target


def test_repricing_fraction_hand_checked():
    from tzbank.chapters.ch13_report_support import _frac

    assert _frac((0.0, 0.0), 1) == pytest.approx(1.0)
    assert _frac((0.25, 1.0), 1) == pytest.approx(0.5 * 0.75, abs=1e-3)  # linear from 0.25 to 1 within year 1
    assert _frac((1.0, 5.0), 1) == 0.0 and _frac((1.0, 5.0), 2) == pytest.approx(0.125, abs=1e-3)
    assert _frac((1.0, 5.0), 3) == pytest.approx(0.375, abs=1e-3) and _frac((5.0, 10.0), 3) == 0.0


def test_asset_beta_tables(root):
    b = _t(root, "asset_beta")
    est = b[b["status"] == "estimated"]
    assert (est["lo"] <= est["beta"]).all() and (est["beta"] <= est["hi"]).all() and (est["n"] >= 5).all()
    assert (b[b["used_in_engine"].str.startswith("yes")]["component"] == "all IEA").all()
    p = _t(root, "repricing_profile")
    assert (p["page_check"] != "NOT FOUND on the cited page").all()
    tot = p[p["bucket"] == "total"].set_index(["bank_id", "line_as_published"])["value_tzs_mn"]
    parts = p[p["bucket"] != "total"].groupby(["bank_id", "line_as_published"])["value_tzs_mn"].sum()
    bad = (parts[tot.index] - tot).abs() > 2  # buckets add to the printed total (rounding) unless flagged
    flags = p[p["bucket"] == "total"].set_index(["bank_id", "line_as_published"])["bucket_sum_check"]
    assert flags[bad[bad].index].str.startswith("source inconsistency").all()
    assert flags[bad[~bad].index].str.startswith("buckets add").all()
    s = _t(root, "rate_sensitivity_repricing")
    r = s[s["case"].str.startswith("repricing")]
    comp = r[["securities_tzs", "interbank_tzs", "loans_tzs", "deposits_tzs", "borrowings_tzs"]].sum(axis=1)
    assert np.allclose(comp, r["nii_change_tzs"]) and r.groupby("bank").size().eq(3).all()
    assert s["engine_uses_these_betas"].str.startswith("no").all()


def test_peer_coe_and_regression(root):
    from tzbank.valuation.coe import approach_a

    c = _t(root, "peer_coe").dropna(subset=["coe_approach_a"])
    assert len(c) >= 4
    hand = [approach_a(r.bond_10y, r.default_spread, r.beta, r.erp_mature, r.crp, r.lp) for r in c.itertuples()]
    assert np.allclose(c["coe_approach_a"], hand)
    m = _t(root, "peer_market")
    ok = m.dropna(subset=["pb"])
    assert set(ok[ok["valuation"] == "HIST"]["fy_basis"]) == {"FY2024"}  # PIT: FY2025 not published by 31 Dec 2025
    assert np.allclose(ok["pe"], ok["price"] / ok["eps"])
    assert m[m["bank"] == "BK Group"]["pb"].isna().all()  # price (KES) and per-share data (RWF) not mixed
    g = _t(root, "peer_pb_roe_regression")
    res = g[g["status"] == "residual"]
    assert {"NMB", "CRDB"} <= set(res["banks"]) and np.allclose(res["residual_pb"], res["actual_pb"] - res["fitted_pb"])


def test_h61_parts_median(root):
    h = pd.read_csv(path("tables", root) / "ch06_h61_parts.csv")
    f = h[h["metric"].str.startswith("net deposit franchise")].iloc[0]
    assert f["smaller_banks_n"] == 3 and f["smaller_banks_median"] < 0 < f["nmb"]


# --- fill-review pre-flight --------------------------------------------------------------------------------------------
def test_financial_table_uses_chapter_definitions(root):
    f = _t(root, "financial_table")
    a = f[f["fiscal_year"].str.endswith("A")]
    h8 = pd.read_csv(path("tables", root) / "ch08_credit_history.csv", dtype={"bank_id": str})
    h8["bank"] = h8["bank_id"].map({"B01": "CRDB", "B02": "NMB"})
    h8["fiscal_year"] = "FY" + h8["fiscal_year"].astype(str) + "A"
    m = a.merge(h8[["bank", "fiscal_year", "npl_ratio", "coverage"]], on=["bank", "fiscal_year"])
    assert len(m) == len(a) and np.allclose(m["NPL"], m["npl_ratio"]) and np.allclose(m["NPL_coverage"], m["coverage"])
    assert np.allclose(f["CIR"], -f["OpEx"] / (f["NII"] + f["NonII"]))
    assert np.allclose(f["LDR_net"], f["net_loans"] / f["deposits"]) and f["ratio_definitions"].str.contains("chapter 8").all()
    assert {"NPL_reported", "NPL_coverage_reported", "CIR_reported"} <= set(f.columns)


def test_nmb_deposit_migration_runs(root):
    m = _t(root, "deposit_migration")
    n = m[m["bank"] == "NMB"]
    assert len(n) == 4 and n["roe_change_pts"].notna().all() and (n["roe_change_pts"] < 0).all()
    assert n["casa_source"].str.startswith("D29 manual entry").all()


def test_rate_consistent_scenarios_are_labelled(root):
    r = _t(root, "rate_consistent")
    assert r["scenario_id"].is_unique and r["scenario_label"].str.len().gt(10).all()


def test_worst_usd_year_is_explicit(root):
    u = _t(root, "usd_exports")
    w = u[u["measure"] == "worst calendar-year USD return"].set_index("bank")
    y = u[u["measure"] == "year of the worst calendar-year USD return"].set_index("bank")
    assert set(w.index) == {"NMB", "CRDB"} and (y["value"] == w["year"]).all() and (w["value"] < 0).all()


def test_exhibit_pngs_exist(root):
    a = pd.read_csv(path("outputs", root) / "exhibits" / "index.csv")
    i = a[a["report"] != "figure"]  # report = figure: chapter-figure rows for the report build (prompt 16)
    assert set(i["report"]) == {"main", "nmb", "crdb", "foreign"} and len(i) >= 30
    for r in i.itertuples():
        assert (root / r.file).exists() and (root / r.data).exists(), r.file
    assert i.groupby("report")["exhibit_no"].apply(lambda s: s.is_unique).all()
    assert all((root / f).exists() for f in a.loc[a["report"] == "figure", "file"])
