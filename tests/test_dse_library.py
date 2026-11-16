"""Prompt 15: DSE data library (D72), quarterly statements (D73) and the stand-alone ingest fix (pre-flight B)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from tzbank.extract import quarterly as Q
from tzbank.ingest import dse_library as L


def _lib(root, name, **kw):
    return pd.read_csv(root / "data" / "library" / "dse" / f"{name}.csv", low_memory=False, **kw)


def test_ingest_stage_preserves_pdf_stage_rows(root):
    """Pre-flight B: the ingest stage re-appends the SMR / deposit-insurance rows the pdf stage wrote."""
    from tzbank import pipeline

    src = inspect.getsource(pipeline.stage_ingest)
    assert "merge_into_bot_long" in src
    bl = pd.read_csv(root / "data" / "interim" / "bot_long.csv", low_memory=False)
    assert bl["series_id"].astype(str).str.startswith("BOT_SMR_").any()


def test_library_schema_and_provenance(root):
    d = pd.read_parquet(root / "data" / "library" / "dse" / "daily_prices.parquet")
    need = {"ticker", "trading_date", "open", "low", "high", "close", "volume", "turnover", "split_factor",
            "adj_close", "adj_volume", "stale", "stale_reason", "block_proxy", "source_url", "snapshot_file",
            "collected_at_utc", "evidence_class"}
    assert need <= set(d.columns)
    assert not d.duplicated(["ticker", "trading_date"]).any()
    assert d["ticker"].nunique() >= 25 and d["snapshot_file"].notna().all()
    sec = _lib(root, "securities")
    assert len(sec) >= 30 and sec["evidence_class"].eq("Observed").all()
    docs = _lib(root, "documents_index")
    ok = docs[docs["download_status"] == "OK"]
    assert len(ok) > 1000 and ok["sha256"].str.len().eq(64).all()


def test_stale_flag(root):
    d = pd.read_parquet(root / "data" / "library" / "dse" / "daily_prices.parquet",
                        columns=["volume", "stale", "stale_reason"])
    st = d["stale"].astype(str).str.lower().eq("true")
    assert st[d["volume"].fillna(0) == 0].all()
    assert d.loc[st, "stale_reason"].str.len().gt(0).all()


def test_split_adjustment(root):
    d = pd.read_parquet(root / "data" / "library" / "dse" / "daily_prices.parquet")
    n = d[d["ticker"] == "NMB"].sort_values("trading_date")
    pre, post = n[n["trading_date"].astype(str) < "2026-08-24"], n[n["trading_date"].astype(str) >= "2026-08-24"]
    assert (pre["split_factor"] == 10).all() and (post["split_factor"] == 1).all()
    assert np.allclose(pre["adj_close"], pre["close"] / 10) and np.allclose(pre["adj_volume"], pre["volume"] * 10)
    # the effective-date jump disappears after adjustment
    jump = post["adj_close"].iloc[0] / pre["adj_close"].iloc[-1]
    assert 0.8 < jump < 1.2
    ca = _lib(root, "corporate_actions")
    assert (ca.loc[(ca["ticker"] == "NMB") & (ca["action"] == "price break"), "status"]
            == "explained by a confirmed action").all()
    others = d[(d["ticker"] != "NMB")]
    assert (others["split_factor"] == 1).all()  # candidate breaks are never applied


def test_index_proxy_two_stock_example():
    """Return = sum(shares x close_t) / sum(shares x close_t-1) over stocks with both closes."""
    prices = pd.DataFrame({"ticker": ["A", "A", "B", "B"], "trading_date": pd.to_datetime(["2026-01-01",
                           "2026-01-02"] * 2), "close": [10.0, 11.0, 20.0, 20.0], "split_factor": 1.0})
    shares = pd.DataFrame({"ticker": ["A", "B"], "shares_in_issue_current": [100.0, 50.0], "listing": "Domestic"})
    sec = pd.DataFrame({"symbol": ["A", "B"], "market": "main"})
    ix = L.index_proxy(prices, shares, sec)
    a = ix[ix["index"] == "all equities (DSEI-like)"].set_index("trading_date")
    assert a["daily_return"].iloc[1] == pytest.approx((100 * 11 + 50 * 20) / (100 * 10 + 50 * 20) - 1)
    assert a["level"].iloc[0] == 1000 and (ix["evidence_class"] == "Derived").all()


def test_index_history_and_missing_days(root):
    h = _lib(root, "index_history")
    assert {"DSEI", "TSI"} <= set(h["index"]) and h.groupby("index").size().min() > 200
    last = h.sort_values("trading_date").groupby("index").tail(1)
    assert np.allclose(last["level"], last["close_on_page"])
    m = _lib(root, "missing_weekdays")
    assert m["classification"].notna().all()


def test_quarter_end_and_text_period():
    assert str(Q.quarter_end("2023-11-03").date()) == "2023-09-30"
    assert str(Q.quarter_end("2012-09-28").date()) == "2012-09-30"
    assert str(Q.quarter_end("2017-10-31").date()) == "2017-09-30"
    t = "STATEMENT AS AT 31.03.2021 previous quarter 31.12.2020 comparative 31/03/2020"
    assert str(Q.text_period_end(t).date()) == "2021-03-31"
    assert str(Q.text_period_end("for the period ended 30th June, 2015").date()) == "2015-06-30"


def test_quarterly_column_choice():
    # 8 P&L values, Group first: current-year cumulative Group = index 4
    assert Q.pick("net_income", "pl", 8, 2, True, True)[0] == 4
    # one entity, 4 values after Q1: CQ, comparative Q, YTD, comparative YTD
    assert Q.pick("net_income", "pl", 4, 3, False, False)[0] == 2
    # two entities with only 4 values after Q1 is ambiguous: rejected, not guessed
    assert Q.pick("net_income", "pl", 4, 2, True, True)[0] is None
    assert Q.pick("total_assets", "bs", 4, 2, True, True)[0] == 0


def test_quarterly_extraction_matches_held_interims(root):
    c = pd.read_csv(root / "reports" / "validation" / "quarterly_extraction_crosscheck.csv")
    ir = c[c["sheet"] == "interim_results"]
    assert len(ir) >= 60 and ir["match_1pct"].all()
    q = pd.read_csv(root / "data" / "processed" / "quarterly_long.csv")
    assert len(q) > 800 and set(q["period_type"]) <= {"Q1", "H1", "9M"}
    assert q["notes"].str.contains("BoT regulatory format").all()


def test_quarterly_balances_positive(root):
    """A dash-only label line never yields a zero balance (CRDB 2022 layout puts values on the first label line)."""
    q = pd.read_csv(root / "data" / "processed" / "quarterly_long.csv")
    bs = q[q["canonical_name"].isin(["net_loans", "customer_deposits", "total_assets", "total_equity"])]
    assert (bs["value"] > 0).all()
    assert q["publication_date"].notna().all() and not q["doc_id"].str.endswith(".0").any()
