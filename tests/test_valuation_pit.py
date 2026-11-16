"""Prompt 09a task 2 and 5: point-in-time integrity of the valuation runs and the proposed register."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.valuation.inputs import base_inputs, eligible_long, pick, pit_violations


@pytest.mark.parametrize("bank", ["B01", "B02"])
def test_fy2025_audited_row_cannot_enter_31_dec_2025_run(root, bank):
    lg = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False, dtype={"bank_id": str})
    fy25 = lg[(lg["bank_id"] == bank) & (lg["period_type"] == "FY") & (lg["fiscal_year"] == 2025)
              & (lg["canonical_name"] == "net_profit") & lg["value"].notna()]
    assert len(fy25) and (pd.to_datetime(fy25["publication_date"]) > "2025-12-31").all()  # audited FY2025: 2026
    e = eligible_long(root, "2025-12-31")
    assert e[(e["bank_id"] == bank) & (e["period_type"] == "FY") & (e["fiscal_year"] == 2025)
             & (e["canonical_name"] == "net_profit")].empty
    v, _ = pick(e, bank, "net_profit", "FY", 2025)
    assert pd.isna(v)
    b = base_inputs(bank, "2025-12-31", root)
    assert b.anchor_fy == 2024 and not pit_violations(b)
    assert all(not k.startswith("FY2025") for k in b.sources)
    # the same row does enter the 31 Oct 2026 run
    b2 = base_inputs(bank, "2026-10-31", root)
    assert b2.anchor_fy == 2025 and b2.fy["net_profit"] == pytest.approx(fy25.sort_values("publication_date")
                                                                         ["value"].iloc[-1])


def test_injected_late_row_is_excluded(root):
    e = eligible_long(root, "2025-12-31")
    fake = e.iloc[[0]].copy()
    fake["publication_date"] = "2026-01-02"
    from tzbank.pit import eligible

    assert eligible(fake, "2025-12-31", warn=False).empty


def test_historical_shares_not_taken_from_a_2026_publication(root):
    b = base_inputs("B02", "2025-12-31", root)
    assert b.shares == pytest.approx(5e8)  # pre-split; the AS AT row is published in April 2026
    assert b.sources["shares_note"].startswith("AS AT shares not eligible")
    c = base_inputs("B02", "2026-10-31", root)
    assert c.shares == pytest.approx(5e9)  # after the 1:10 split (24 Aug 2026)


def test_proposed_register(root):
    r = pd.read_csv(root / "config" / "assumption_register_proposed.csv")
    need = {"g_L", "g_D", "NIM", "phi", "CIR", "CoR", "rho", "rate", "omega", "theta", "l_min", "s_W", "t", "nu",
            "delta", "k_star", "pi_max", "ROE_star", "g", "payout_star", "lambda", "LP", "beta", "deposit_beta",
            "asset_beta", "ER_star", "CoR_ttc"}
    for (v, b), x in r.groupby(["valuation", "bank"]):
        assert need <= set(x["symbol"]), (v, b)
        assert x[["base", "bear", "bull"]].notna().all().all()
        assert (x["inside_range"] != "NO").all()
    assert {"s_sub", "u", "fx"} <= set(r.loc[r["bank"] == "CRDB", "symbol"])
    h = r[r["valuation"] == "HIST"]
    assert h["information_cutoff"].str.contains("FY<=2024").all()
    assert h.loc[h["symbol"] == "ER_star", "evidence"].str.contains("2024").all()  # PIT re-estimation
    assert not h["range_source"].str.contains("calibration_inputs").any()  # CURR-only source
    t = r[r["symbol"] == "payout_star"].set_index(["valuation", "bank"])
    g = r[r["symbol"] == "g"].set_index(["valuation", "bank"])
    roe = r[r["symbol"] == "ROE_star"].set_index(["valuation", "bank"])
    for s in ("base", "bear", "bull"):
        assert ((g[s] - roe[s] * (1 - t[s])).abs() < 1e-12).all()  # terminal growth internally funded
    assert r["status"].str.startswith("PROPOSED").all() and r["version"].eq("v0.1-proposed").all()


def test_dry_run_checks_logged_without_values(root):
    c = pd.read_csv(path("outputs", root) / "valuation" / "dryrun_engine_checks.csv")
    assert (c["status"] == "PASS").all()
    assert not any(k in c.columns for k in ("value", "value_per_share", "price", "target_price"))
    for name in ("balance sheet closes", "Tier 1 >= k*", "liquid assets >= floor", "LDR within bounds",
                 "clean surplus", "RIM and DDM within 10%", "target-price identity", "information cut-off",
                 "five-step bridge", "Monte Carlo reproducible"):
        assert c["check"].str.contains(name, regex=False).any(), name
    rep = (path("reports", root) / "valuation_09a_report.md").read_text(encoding="utf-8")
    assert "No valuation results are published" in rep and "break-adjusted lambda" in rep
