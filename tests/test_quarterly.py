"""Quarterly BoT-format statements (tzbank.chapters.ch13_quarterly; prompt 20, D89)."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.chapters import ch13_quarterly as cq
from tzbank.config import find_root, path

ROOT = find_root()


@pytest.fixture(scope="module")
def q():
    return cq.load(ROOT)


def test_identities_hold(q):
    chk = cq.checks(q, ROOT)
    ident = chk[chk["check"].str.startswith(("assets =", "NII =", "PAT =", "YTD"))]
    assert (ident["status"] != "fail").all(), ident[ident["status"] == "fail"]


def test_crdb_q3_deposit_fall_and_ldr(q):
    d0 = cq.get(q, "CRDB", "customer_deposits", "2026-06-30", "current_quarter")
    d1 = cq.get(q, "CRDB", "customer_deposits", "2026-09-30", "current_quarter")
    l1 = cq.get(q, "CRDB", "net_loans", "2026-09-30", "current_quarter")
    assert (d0, d1, l1) == (18_544_695, 17_409_616, 18_437_223)          # owner's figures (prompt 20)
    assert d1 / d0 - 1 == pytest.approx(-0.0612, abs=5e-4)
    f = cq.five(q).set_index(["bank", "period_end"])
    assert f.loc[("CRDB", "2026-09-30"), "net_ldr"] == pytest.approx(l1 / d1)


def test_ttm_hand_check(q):
    t = cq.ttm(q).set_index("bank")
    pat = (cq.get(q, "NMB", "pat", "2025-12-31", "ytd_current") + cq.get(q, "NMB", "pat", "2026-09-30", "ytd_current")
           - cq.get(q, "NMB", "pat", "2026-09-30", "ytd_prior"))
    assert t.loc["NMB", "pat_ttm_tzs_m"] == pytest.approx(pat)


def test_nmb_eps_printed_annualised_and_post_split(q):
    e = cq.eps_split(q).set_index("period_end")
    assert e.loc["2026-09-30", "printed_basis"] == "post-split"
    assert e.loc["2026-06-30", "printed_basis"] == "pre-split"


def test_roll_is_a_sensitivity_not_the_base():
    f = path("tables", ROOT) / "ch13_qtr_roll.csv"
    if not f.exists():
        pytest.skip("not generated")
    r = pd.read_csv(f)
    v = pd.read_csv(path("tables", ROOT) / "ch09_valuations.csv").query("valuation == 'CURR'").set_index("bank")
    base = r[r["variant"].str.startswith("base")].set_index("bank")
    for b in ("NMB", "CRDB"):
        assert base.loc[b, "value_per_share"] == pytest.approx(v.loc[b, "value_per_share"], rel=1e-9)
