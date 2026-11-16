"""Prompt 10: Part I, chapters 1-3, and the BoT label crosswalk (blueprint chapters 1-3; D59)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.ch02_credit_allocation import CATEGORIES, dissimilarity
from tzbank.chapters.ch03_systemic_importance import _ols
from tzbank.config import path


def _t(root, name):
    return pd.read_csv(path("tables", root) / name, dtype={"bank_id": str})


# --- task 1: BoT label harmonisation -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["bot_long.csv", "bot_research_database.csv"])
def test_no_canonical_bot_series_has_duplicate_periods(root, name):
    d = pd.read_csv(path("interim", root) / name, usecols=["series_id", "period_end"], low_memory=False)
    d = d[d["series_id"].astype(str).str.startswith("BOTDB_")]
    assert not d.duplicated(["series_id", "period_end"]).any()


def test_crosswalk_variants_are_merged_into_canonical_series(root):
    x = pd.read_csv(root / "config" / "bot_series_crosswalk.csv")
    ids = set(pd.read_csv(path("interim", root) / "bot_research_database.csv", usecols=["series_id"])["series_id"])
    live = x[~x["status"].str.upper().str.startswith("REJECTED")]
    assert len(live) > 0
    def stem(s: str) -> str:  # series IDs carry a frequency / basis suffix (_ANNUAL, _MONTHLY, _FY_JUNE, _PCT ...)
        return re.sub(r"(_(ANNUAL|MONTHLY|QUARTERLY|FY_JUNE|PCT))+$", "", s)

    stems = {stem(i) for i in ids}
    for r in live.itertuples():
        variant = f"BOTDB_{r.topic.upper()}__{r.label_variant.upper()}"
        canon = f"BOTDB_{r.topic.upper()}__{r.canonical_label.upper()}"
        assert variant not in stems, variant
        assert canon in stems, canon
    assert pd.read_csv(path("validation", root) / "bot_crosswalk_collisions.csv").empty


def test_v1_v2_recheck_covers_the_48_conflicts(root):
    r = pd.read_csv(path("validation", root) / "bot_v1_v2_recheck.csv")
    assert len(r) == 48 and r["classification"].notna().all()
    assert r["classification"].str.startswith(("revision", "label swap")).all()


def test_sector_totals_assets_series_is_continuous(root):
    s = pd.read_csv(path("processed", root) / "sector_totals.csv", low_memory=False)
    a = s[s["series_id"] == "BOTDB_BANK_ASSETS__TOTAL_ASSET"]
    yrs = sorted(pd.to_datetime(a["period_end"]).dt.year.unique())
    assert yrs[0] <= 2009 and yrs[-1] >= 2024 and yrs == list(range(yrs[0], yrs[-1] + 1))


# --- chapter 1 --------------------------------------------------------------------------------------------------------
def test_cr2_shares_between_0_and_1_and_add_up(root):
    s = _t(root, "ch01_shares.csv")
    for c in ("nmb_share", "crdb_share", "cr2"):
        assert s[c].between(0, 1).all(), c
    assert np.allclose(s["cr2"], s["nmb_share"] + s["crdb_share"])
    assert np.allclose(s["nmb_share"] * s["sector_tzs"], s["nmb_tzs"], rtol=1e-9)
    assert (s["nmb_tzs"] + s["crdb_tzs"] < s["sector_tzs"]).all()


def test_cr2_consistent_with_bank_totals(root):
    s = _t(root, "ch01_shares.csv")
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    w = w.set_index(["bank_id", "fiscal_year"])
    for measure, var in (("assets", "total_assets"), ("deposits", "customer_deposits"), ("loans", "gross_loans")):
        for r in s[s["measure"] == measure].itertuples():
            assert r.nmb_tzs == pytest.approx(w.loc[("B02", r.year), var], rel=1e-9)
            assert r.crdb_tzs == pytest.approx(w.loc[("B01", r.year), var], rel=1e-9)


def test_profit_gap_formula(root):
    p = _t(root, "ch01_profit_vs_assets.csv")
    hand = p["asset_share"] * (p["cr2_roa_pretax"] / p["sector_roa"] - 1)
    assert np.allclose(p["profit_minus_asset_share_pretax"], hand)
    assert np.allclose(p["profit_share_pretax"] - p["asset_share"], p["profit_minus_asset_share_pretax"])
    assert (p["evidence_class"] == "Estimated").all()


def test_dse_shares_between_0_and_1(root):
    d = _t(root, "ch01_dse_shares.csv")
    assert d["share"].between(0, 1).all()
    assert (d["cr2_value_tzs"] <= d["market_value_tzs"]).all()


# --- chapter 2 --------------------------------------------------------------------------------------------------------
def test_dissimilarity_hand_cases():
    a = pd.Series({"agriculture": 1.0})
    b = pd.Series({"trade": 1.0})
    assert dissimilarity(a, b) == pytest.approx(1.0)
    assert dissimilarity(a, a) == pytest.approx(0.0)
    c = pd.Series({"agriculture": 0.5, "trade": 0.5})
    d = pd.Series({"agriculture": 0.25, "trade": 0.75})
    assert dissimilarity(c, d) == pytest.approx(0.25)
    assert dissimilarity(pd.Series({"agriculture": 2.0, "trade": 2.0}), c) == pytest.approx(0.0)  # scale-free


def test_loan_mix_shares_sum_to_one_and_index_in_range(root):
    m = _t(root, "ch02_loan_mix.csv")
    assert set(m["category"]) <= set(CATEGORIES)
    assert m["share"].between(0, 1).all()
    assert np.allclose(m.groupby(["fiscal_year", "book"])["share"].sum(), 1.0)
    assert _t(root, "ch02_dissimilarity.csv")["dissimilarity"].between(0, 1).all()


def test_sector_loan_extraction_reconciles(root):
    d = pd.read_csv(path("interim", root) / "loans_by_sector.csv", dtype={"bank_id": str})
    g = d.groupby(["bank_id", "fiscal_year"])
    assert np.allclose(g["value"].sum(), g["total_tzs"].first(), rtol=0.005)
    assert np.allclose(g["share"].sum(), 1.0, atol=0.005)
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    net = w.set_index(["bank_id", "fiscal_year"])["net_loans"]
    for (b, y), tot in g["total_tzs"].first().items():
        assert abs(tot - net[(b, y)]) <= 0.05 * net[(b, y)]


# --- chapter 3 --------------------------------------------------------------------------------------------------------
def test_capital_headroom_is_ratio_minus_requirement(root):
    b = _t(root, "ch03_capital_buffers.csv")
    assert np.allclose(b["tier1_headroom"], b["tier1_ratio"] - b["tier1_requirement"], equal_nan=True)
    assert np.allclose(b["total_headroom"], b["total_capital_ratio"] - b["total_requirement"], equal_nan=True)
    assert np.allclose(b["binding_headroom"], b[["tier1_headroom", "total_headroom"]].min(axis=1), equal_nan=True)


def test_pb_regression_reproduces_hand_ols():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"bank_id": list("ABCDEFGHIJ") * 3, "roe": rng.uniform(0, 0.3, 30)})
    x["systemic"] = x["bank_id"].isin(["A", "B"]).astype(int)
    x["pb"] = 0.5 + 2.0 * x["roe"] + 0.3 * x["systemic"]
    f = _ols(x)
    assert (f["a"], f["b_roe"], f["c_sib"]) == pytest.approx((0.5, 2.0, 0.3))
    assert f["c_lo"] == pytest.approx(0.3) and f["n"] == 30 and f["n_banks"] == 10


def test_dsib_table_records_every_search(root):
    items = " ".join(_t(root, "ch03_dsib.csv")["item"])
    for s in ("BoT Annual Reports", "CRDB reports", "NMB reports", "BoT D-SIB framework", "NMB designation"):
        assert s in items


# --- quality gates ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ch", ["01", "02", "03"])
def test_notes_carry_no_causal_claims(root, ch):
    t = (path("notes", root) / f"ch{ch}.md").read_text(encoding="utf-8")
    assert not re.search(r"\bcaus(e|es|ed|ing)\b|\bled to\b|\bdrives?\b|\bresult(s|ed)? in\b", t, re.I)


@pytest.mark.parametrize("ch", ["01", "02", "03"])
def test_part1_tables_carry_evidence_class(root, ch):
    m = json.loads((path("manifests", root) / f"ch{ch}.json").read_text(encoding="utf-8"))
    assert m["run_log_stage"] == "part1"
    for t in m["tables"]:
        df = pd.read_csv(root / t["path"])
        assert "evidence_class" in df and df["evidence_class"].notna().all()
