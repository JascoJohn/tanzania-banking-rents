"""D64: web-collected values (IMF forecasts, 2026 T-bill auctions) enter through a gated path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzbank.config import path
from tzbank.ingest.web_collected import NOMINAL, fill_macro, raw_file, validate


def test_every_row_is_cited_dated_and_point_in_time(root):
    raw = pd.read_csv(raw_file(root), dtype=str)
    assert raw["source_url"].str.startswith("http").all()
    assert pd.to_datetime(raw["access_date"]).notna().all()
    assert (pd.to_datetime(raw["publication_date"]) <= pd.to_datetime(raw["as_at"])).all()
    good, issues = validate(root)
    assert issues.empty and len(good) >= len(raw)


def test_reo_values_are_printed_on_the_cited_page(root):
    if not (root / "data" / "raw" / "documents").exists():
        pytest.skip("needs the source PDFs in data/raw/documents/ (third-party, not in the public repository; "
                    "fetch them as docs/DATA_REQUIREMENTS.md describes)")
    good, _ = validate(root)
    held = good[good["source_doc"].astype(str).str.endswith(".pdf") & (good["evidence_class"] == "Observed")]
    assert len(held) >= 4 and (held["pdf_check"] == "value found on the cited page").all()


def test_nominal_growth_is_derived_by_formula(root):
    good, _ = validate(root)
    for asat, g in good.groupby("as_at"):
        g = g.set_index("canonical_name")
        if NOMINAL in g.index:
            real = float(g.loc["macro_tanzania_real_gdp_growth_forecast", "value"])
            cpi = float(g.loc["macro_tanzania_inflation_forecast_medium_term", "value"])
            assert float(g.loc[NOMINAL, "value"]) == pytest.approx((1 + real) * (1 + cpi) - 1), asat
            assert g.loc[NOMINAL, "evidence_class"] == "Derived"


def test_workbook_keeps_precedence(root):
    m = pd.DataFrame({"canonical_name": ["macro_tanzania_inflation_forecast_medium_term"],
                      "As At (Valuation Date)": [pd.Timestamp("2025-12-31")], "value_numeric": [0.123],
                      "Status": ["REPORTED"]})
    out = fill_macro(m, root)
    assert out.loc[0, "value_numeric"] == 0.123  # a workbook value is never overwritten


def test_2026_tbill_uses_the_quarter_end_auction(root):
    r = pd.read_csv(path("tables", root) / "ch06_rates_used.csv").set_index("year")
    w = pd.read_csv(path("interim", root) / "web_collected.csv")
    w = w[(w["canonical_name"] == "tbill_364d_way") & (w["as_at"] <= "2026-09-30")].sort_values("as_at")
    assert r.loc[2026, "tbill"] == pytest.approx(float(w["value"].iloc[-1]))
    assert int(r.loc[2026, "tbill_n_quarters"]) == 1
    assert np.isclose(r.loc[2025, "tbill"], 0.09495, atol=1e-6)  # BoT database years unchanged
