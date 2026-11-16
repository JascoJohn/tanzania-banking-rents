"""Prompt 08: Part II synthesis and calibration inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters.part2_synthesis import EXPLANATIONS
from tzbank.config import path

CLASSES = {"Observed", "Derived", "Estimated", "Scenario", "Interpretation"}


def test_no_unresolved_contradiction(root):
    c = pd.read_csv(path("reports", root) / "part2_consistency.csv")
    assert not c["status"].str.startswith("CONTRADICTION").any()
    assert {"funding (4.2 vs 6)", "credit (4.6 vs 8)", "leverage (4.0 vs 7)"} <= set(c["area"])
    ex = c[c["status"].str.startswith("explained")]
    assert ex["explanation"].astype(str).str.len().gt(20).all()  # every difference carries its explanation
    md = (path("reports", root) / "part2_consistency.md").read_text(encoding="utf-8")
    assert "Unresolved contradictions: none" in md


def test_ch04_and_ch08_return_on_rwa_agree(root):
    a = pd.read_csv(path("tables", root) / "ch04_4_6_risk.csv", dtype={"bank_id": str})
    b = pd.read_csv(path("tables", root) / "ch08_capital_efficiency.csv", dtype={"bank_id": str})
    m = a.merge(b, on=["bank_id", "fiscal_year"], suffixes=("_46", "_8"))
    m = m[m["bank_id"].isin(["B01", "B02"])].dropna(subset=["return_on_rwa_46", "return_on_rwa_8"])
    assert len(m) > 20 and np.allclose(m["return_on_rwa_46"], m["return_on_rwa_8"])


def test_rent_sources_cover_every_explanation(root):
    r = pd.read_csv(path("tables", root) / "part2_rent_sources.csv", dtype={"bank_id": str})
    assert set(r["evidence_class"]) <= CLASSES and r["evidence_class"].notna().all()
    assert set(r["direction"]) <= {"supports", "contradicts"}
    assert set(r["strength"]) <= {"strong", "moderate", "weak"}
    for b in ("B01", "B02"):
        x = r[r["bank_id"] == b]
        assert set(EXPLANATIONS) <= set(x["explanation"])
        ov = x[x["explanation"].str.startswith("OVERALL")]
        assert len(ov) == 1 and ov["evidence"].iloc[0].split(":")[0] in (
            "GENUINE ADVANTAGE with a cyclical component", "STRUCTURAL FEATURES AND CYCLE")
    assert r["table"].astype(str).str.len().gt(0).all()


def test_hypothesis_register_complete(root):
    h = pd.read_csv(path("tables", root) / "hypotheses_status.csv")
    need = {"H4.1", "H4.2", "H5.1", "H5.2", "H5.3", "H6.1", "H6.2", "H7.1", "H7.2", "H7.3", "H8.1", "H8.2"}
    assert need <= set(h["hypothesis"])
    assert h["statement"].notna().all() and h["evidence"].notna().all()
    assert h["verdict"].str.match(r"^(conditionally supported|supported|not supported|rejected|inconclusive)").all()  # D90
    assert "Open issue 4" in h.loc[h["hypothesis"] == "H7.1", "review_note"].iloc[0]


def test_calibration_inputs_are_ranges(root):
    c = pd.read_csv(path("tables", root) / "calibration_inputs.csv", dtype={"bank_id": str})
    params = {"long-run excess return ER*", "sustainable ROE (ER* + COE)", "fade speed lambda",
              "through-the-cycle cost of risk", "NIM (NII / avg interest-earning assets)", "cost-to-income path",
              "payout ratio", "capital headroom (FY2025)", "deposit beta (all deposits)"}
    for b in ("B01", "B02"):
        assert params <= set(c.loc[c["bank_id"] == b, "parameter"])
    assert (c["range_low"] <= c["range_high"] + 1e-12).all()
    assert (c["interval_low"] <= c["interval_high"] + 1e-12).all()
    assert c["selection"].str.contains("D8").all()
    assert not any(k in c.columns for k in ("chosen", "selected", "central", "value"))
    assert c["source_table"].str.contains("ch0").all()
    # sustainable ROE = COE + ER* bounds (hand check against chapter 5 tables)
    cal = pd.read_csv(path("tables", root) / "ch05_calibration_candidates.csv").set_index(["bank", "specification"])
    coe = pd.read_csv(path("tables", root) / "ch05_coe_annual.csv").set_index("date_label")["coe"]
    s = c[(c["bank_id"] == "B02") & (c["parameter"] == "sustainable ROE (ER* + COE)")].iloc[0]
    assert s["range_high"] == pytest.approx(max(coe["FY2025"], coe["valuation_current"])
                                            + cal.loc[("NMB", "ar1_break_episode"), "er_star"])
    assert s["range_low"] == pytest.approx(min(coe["FY2025"], coe["valuation_current"]))  # bear: ER* = 0
    lam = c[(c["bank_id"] == "B01") & (c["parameter"] == "fade speed lambda")].iloc[0]
    assert lam["range_low"] == pytest.approx(cal.loc[("CRDB", "ar1_break_episode"), "lambda_lo"])


def test_summary_and_manifest(root):
    s = (path("reports", root) / "part2_executive_summary.md").read_text(encoding="utf-8")
    assert "DRAFT for owner review" in s and "nan" not in s.lower().replace("financ", "")
    m = json.loads((path("manifests", root) / "part2.json").read_text(encoding="utf-8"))
    for t in m["tables"]:
        assert (root / t["path"]).exists()
    note = (path("notes", root) / "part2.md").read_text(encoding="utf-8")
    for head in ("## Research question", "## Variable definitions", "## Sources", "## Validation test",
                 "## Evidence class", "## Limitations", "## Cross-cutting explanations"):
        assert head in note
