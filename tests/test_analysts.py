"""Local analyst register (tzbank.chapters.ch13_analysts; prompts 19-20, D88)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzbank.chapters import ch13_analysts as ca
from tzbank.config import find_root, path

ROOT = find_root()


def test_register_merged_and_split_adjusted():
    r = ca.register(ROOT)
    assert len(r) == 17 and r["id"].is_unique
    pre = r[r["per_share_basis_reported"].fillna("").str.startswith("pre-split")]
    assert np.allclose(pre["fair_value_post_split"], pre["fair_value_reported"] / 10)
    assert np.allclose(pre["price_post_split"].dropna(), pre["price_reported"].dropna() / 10)
    assert set(r["bank"]) <= {"NMB", "CRDB", "other", "market"}
    # owner-confirmed Solomon views (prompt 20)
    s = r.set_index("id")
    assert s.loc[12, "fair_value_reported"] == pytest.approx(16347.02) and s.loc[12, "bank"] == "NMB"
    assert s.loc[11, "coe"] == pytest.approx(0.1497) and s.loc[11, "data_as_of"] == "2026-03-31"


def test_price_lookup_uses_last_close_on_or_before():
    px = ca.dse_prices(ROOT)
    p, d = ca.price_on(px, "NMB", pd.Timestamp("2026-06-25"))
    assert d <= "2026-06-25" and p == pytest.approx(1609, rel=0.01)  # the note prints 16,090 pre-split


def test_revision_hand_check():
    f = path("tables", ROOT) / "ch13_analyst_revisions.csv"
    if not f.exists():
        pytest.skip("not generated")
    v = pd.read_csv(f)
    x = v[(v["broker"] == "Solomon") & (v["bank"] == "NMB")].iloc[0]
    assert x["target_change"] == pytest.approx(13544.03 / 16347.02 - 1)        # -17.15%
    assert x["target_to_price_after"] == pytest.approx(1354.403 / x["price_to"] - 1)
    c = v[(v["broker"] == "Solomon") & (v["bank"] == "CRDB")].iloc[0]
    assert c["target_change"] == pytest.approx(2756.40 / 2934.63 - 1)          # -6.07%
