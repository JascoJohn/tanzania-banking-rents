"""Prompt 15c: figure fixes (ch04, ch06, ch13, ch14), the CRDB fee-note session and the coverage table."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tzbank.config import path  # noqa: E402


def _tab(root, n):
    return pd.read_csv(path("tables", root) / n)


def _fig(root, n):
    return pd.read_csv(path("figures", root) / n)


# --- helpers ---------------------------------------------------------------------------------------------------------
def test_year_ticks_integer_on_year_axes_only():
    from tzbank.chapters.ch04.common import year_ticks

    fig, (a, b) = plt.subplots(1, 2)
    a.plot([2015, 2016, 2017], [1, 2, 3])
    b.plot(pd.to_datetime(["2025-01-01", "2025-06-01"]), [1, 2])
    a.set_xlim(2014.6, 2017.4)
    year_ticks(fig)
    ticks = [t for t in a.get_xticks() if 2014.6 <= t <= 2017.4]
    assert ticks and all(float(t).is_integer() for t in ticks)
    assert a.xaxis.get_major_formatter()(2016.0, 0) == "2016"
    assert type(b.xaxis.get_major_formatter()).__name__ != "FuncFormatter"  # date axis untouched
    plt.close(fig)


def test_waterfall_full_range_and_plain_labels():
    from tzbank.chapters.ch04.dupont_attribution import plain_label, waterfall

    fig, ax = plt.subplots()
    vals = [-0.03, 0.05, 0.04, -0.10]  # running path: 0, -3, +2, +6, -4 (pp)
    waterfall(ax, ["a", "b", "c", "d"], vals, "total")
    lo, hi = ax.get_ylim()
    assert lo < -4 and hi > 6
    plt.close(fig)
    assert plain_label("c_fund", "4.2 Funding economics (cost of funds)") == "cost of funds (4.2)"
    assert plain_label("ta_e") == "leverage"


def test_event_label_and_years():
    from tzbank.chapters.ch14_market_extensions import _event_label, _horizon_label, _years

    assert _event_label("NMB-INT-H12026: results statement published", "2026-07-30", True) == "H1 2026\n30 Jul 26"
    assert _event_label("FY2022 audited results published", "2023-04-11", np.nan) == "FY2022\n11 Apr 23"
    assert _event_label("NMB-INT-9M2026: x", "2026-10-30", False).startswith("9M 2026*")
    assert _years([2017, 2018, 2019, 2021]) == "2017-2019, 2021"
    assert _horizon_label("since FY2011 (first year with EPS and BVPS; listed FY2009)") == "since FY2011\n(listed FY2009)"


# --- data and tables ---------------------------------------------------------------------------------------------------
def test_crdb_fee_notes_fy2015_fy2025(root):
    from tzbank.extract.session_15c import NOTES_15C, rows

    for _, _, _, total, lines in NOTES_15C:
        assert sum(v for _, v in lines) == total  # the printed lines add to the printed gross total
    r = rows(root)
    for fy in (2015, 2025):
        x = r[r["fiscal_year"] == fy]
        tot = x[x["label"] == "printed note total"]["value"].iloc[0]
        assert x[x["label"] != "printed note total"]["value"].sum() == pytest.approx(tot)
    f = _fig(root, "ch04_4_1_fee_mix_data.csv")
    c = f[f["bank_id"] == "B01"].set_index("fiscal_year")
    assert {2015, 2025} <= set(c.index)
    shares = c.filter(like="share_").fillna(0).sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_nmb_other_fee_composition(root):
    o = _tab(root, "ch04_4_1_nmb_other_fees.csv")
    assert (o["evidence_class"] == "Observed").all()
    s = o.groupby("fiscal_year").agg(v=("value", "sum"), t=("other_total", "first"))
    assert np.allclose(s["v"], s["t"], rtol=1e-6)
    assert "mno collaboration fees" in set(o.loc[o["fiscal_year"] == 2025, "printed_line"])


def test_crdb_since_listing_components(root):
    a = _tab(root, "ch14_return_attribution.csv")
    c = a[(a["bank"] == "CRDB") & a["horizon"].str.startswith("since FY")]
    assert len(c) == 1 and c[["eps_growth_log", "pe_change_log", "bvps_growth_log"]].notna().all().all()
    assert c["start_fy"].iloc[0] == 2011


def test_daily_coverage_and_rolling_breaks(root):
    cov = _tab(root, "ch14_daily_coverage.csv")
    thin = cov[cov["coverage"] != "adequate"]
    assert set(thin.loc[thin["bank"] == "NMB", "year"]) >= {2017, 2018, 2019, 2020, 2021}
    assert (thin["nonstale_share"] < thin["threshold"]).all()
    r = _tab(root, "ch14_rolling_correlation.csv")
    assert (r["coverage_ok"] == (r["window_nonstale_share_min"] >= r["threshold"])).all()
    nmb2019 = r[pd.to_datetime(r["trading_date"]).dt.year.eq(2019) & r["window"].str.startswith("60")]
    assert not nmb2019["coverage_ok"].any()


def test_event_windows_confirmed_only(root):
    k = _fig(root, "ch14_event_windows_data.csv")
    assert k["include"].astype(bool).all() and k["date_basis"].str.startswith("confirmed").all()
    assert set(k["role"]) == {"event bank"} and set(k["window"]) == {"[0,+5]", "[0,+20]"}


def test_pb_roe_all_banks_and_oct_points(root):
    k = _tab(root, "ch13_kpi_table.csv")
    assert k["pb"].notna().sum() == 7
    d = _fig(root, "ch13_pb_roe_anchor_data.csv")
    cur = d[d["point"].str.startswith("price at")]
    assert set(cur["bank"]) == {"NMB", "CRDB"} and (cur["pb"] > 0).all()


def test_price_lines_two_buy_lines(root):
    ln = _tab(root, "ch13_price_lines.csv")
    pl = _tab(root, "ch14_price_lines_scenarios.csv")
    for b in ("NMB", "CRDB"):
        x = ln[ln["bank"] == b]
        assert x["line"].str.startswith("Buy below").sum() == 2
        bear = x[x["line"].str.contains("bear-case")]["price"].iloc[0]
        ref = pl[(pl["bank"] == b) & (pl["valuation"] == "CURR") & pl["line"].str.startswith("bear-screen")]
        assert bear == pytest.approx(ref["price_line"].iloc[0])


def test_roe_figure_flags_mucoba_2020(root):
    d = _fig(root, "ch04_roe_by_bank_data.csv")
    f = d[(d["bank_id"] == "B07") & (d["fiscal_year"] == 2020)]
    assert f["flag"].str.contains("near-zero average equity").all()
