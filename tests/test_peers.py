"""Peer-bank checks (tzbank.chapters.ch13_peers; prompt 19, D87): hand-checked formulas on a synthetic bank, and the
generated tables when present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters import ch13_peers as cp
from tzbank.config import find_root, path

ROOT = find_root()


def _fin(rows):
    d = pd.DataFrame(rows, columns=["bank", "fy", "field", "value", "unit", "column"])
    d["value_reported"] = d["value"]
    d["value"] = d["value"] * d["unit"].map(cp.SCALE).fillna(1.0)
    return d


SYN = _fin([
    ("XB", 2024, "equity_attributable", 100, "millions", "current"),
    ("XB", 2024, "total_assets", 1000, "millions", "current"),
    ("XB", 2025, "equity_attributable", 120, "millions", "current"),
    ("XB", 2025, "total_assets", 1200, "millions", "current"),
    ("XB", 2025, "profit_attributable", 33, "millions", "current"),
    ("XB", 2025, "equity_opening", 100, "millions", "current"),
    ("XB", 2025, "dividends_paid", 10, "millions", "current"),
    ("XB", 2025, "oci_attributable", -3, "millions", "current"),
    ("XB", 2025, "other_equity_movements", 0, "millions", "current"),
    ("XB", 2025, "shares_outstanding_ye", 10_000_000, "shares", "current"),
    ("XB", 2025, "shares_weighted", 10_000_000, "shares", "current"),
    ("XB", 2025, "eps_basic", 3.3, "per_share", "current"),
    ("XB", 2025, "dps_declared", 1.1, "per_share", "current"),
    ("XB", 2025, "bvps_reported", 12.0, "per_share", "current"),
    ("XB", 2024, "total_assets", 1010, "millions", "comparative"),
])


def _c(chk, prefix):
    return chk[chk["check"].str.startswith(prefix) & (chk["fy"] == 2025)].iloc[0]


def test_hand_checked_ratios_and_identities():
    chk = cp.checks(SYN, ROOT)
    assert _c(chk, "ROE")["computed"] == pytest.approx(33 / 110)        # average of 100 and 120
    assert _c(chk, "ROA")["computed"] == pytest.approx(33 / 1100)
    assert _c(chk, "payout")["computed"] == pytest.approx(1.1 * 10e6 / 33e6)
    bridge = _c(chk, "equity bridge")
    assert bridge["computed"] == pytest.approx(120e6) and bridge["status"] == "pass"  # 100 + 33 - 10 - 3
    assert _c(chk, "BVPS")["status"] == "pass"
    assert _c(chk, "EPS")["status"] == "pass"
    rest = chk[chk["check"].str.startswith("restatement")].iloc[0]
    assert rest["status"] == "fail" and rest["deviation"] == pytest.approx(1010 / 1000 - 1)  # 1% > 0.5% tolerance


def test_bridge_failure_is_reported_not_hidden():
    bad = SYN.copy()
    bad.loc[bad["field"] == "dividends_paid", "value"] = 20e6
    assert _c(cp.checks(bad, ROOT), "equity bridge")["status"] == "fail"


def test_justified_pb_formula():
    roe, coe, g = 0.25, 0.16, 0.06
    assert (roe - g) / (coe - g) == pytest.approx(1 + (roe - coe) / (coe - g))


def test_generated_tables_carry_evidence_class():
    files = sorted(path("tables", ROOT).glob("ch13_peers_*.csv"))
    if not files:
        pytest.skip("peer tables not generated")
    for f in files:
        d = pd.read_csv(f)
        assert d["evidence_class"].notna().all(), f.name
    prem = pd.read_csv(path("tables", ROOT) / "ch13_peers_premium.csv")
    tz = prem[prem["country"] == "Tanzania"]
    peers = prem[(prem["country"] != "Tanzania") & (prem["valuation"] == "HIST")]
    for r in tz[tz["valuation"] == "HIST"].itertuples():
        assert bool(r.larger_than_every_peer) == bool(r.pb_minus_justified > np.nanmax(peers["pb_minus_justified"]))
