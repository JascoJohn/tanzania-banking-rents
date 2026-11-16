"""DSE trading-history collector (D71): parsing against the saved live snapshot, validation rules, de-duplication."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from tzbank.config import find_root
from tzbank.ingest import dse_trading_history as dse

ROOT = find_root()
SNAPS = dse.snapshots("CRDB", ROOT)


def _snapshot() -> tuple[bytes, dict, str]:
    ok = [(f, m) for f, m in SNAPS if m["status"] == "OK" and f.exists()]
    if not ok:
        pytest.skip("no saved CRDB snapshot under data/raw/dse/trading_history")
    f, m = ok[0]
    return gzip.decompress(f.read_bytes()), m, f"dse/trading_history/{f.name}"


@pytest.mark.parametrize(("text", "expected"), [("2,790", 2790.0), ("725,583,460", 725583460.0), ("0", 0.0),
                                                ("", None), ("-", None), ("abc", None), (None, None), (430, 430.0)])
def test_number(text: object, expected: float | None) -> None:
    assert dse.number(text) == expected


def test_snapshot_payload_matches_rendered_page() -> None:
    body, meta, rel = _snapshot()
    df, s = dse.records_from_snapshot(body, meta, rel)
    assert s["html_header_ok"] and s["html_rows_checked"] == s["page_size"] > 0
    assert s["html_rows_mismatched"] == 0
    assert s["n_pages_implied"] == s["n_pages_displayed"]  # every client page is in the payload
    assert df["trading_date"].notna().all() and not df["trading_date"].duplicated().any()
    assert df["trading_date"].is_monotonic_decreasing  # newest first, as the site pages it
    assert meta["records_in_payload"] == len(df)
    assert df.at[0, "company_name"] == "CRDB BANK PUBLIC LIMITED COMPANY"


def test_pagination_reference() -> None:
    body, meta, rel = _snapshot()
    df, s = dse.records_from_snapshot(body, meta, rel)
    size = s["page_size"]
    assert df.at[0, "pagination_ref"].startswith("page 1 of ")
    assert df.at[size, "pagination_ref"].startswith("page 2 of ") and "row 1 (" in df.at[size, "pagination_ref"]
    assert df.iloc[-1]["pagination_ref"].startswith(f"page {s['n_pages_displayed']} of ")


def test_payload_column_order_against_display() -> None:
    """Payload is [date, open, high, low, close, volume, turnover]; the display order differs. Low <= high on the
    first rendered page confirms the mapping is not swapped."""
    body, meta, rel = _snapshot()
    df, _ = dse.records_from_snapshot(body, meta, rel)
    first = df[df["html_display_check"] == "match"]
    assert len(first) and (first["low"] <= first["high"]).all()


def _frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["ticker", "trading_date", "open", "low", "high", "close", "turnover", "volume"]
    df = pd.DataFrame(rows, columns=cols)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


def test_validation_rules() -> None:
    df = _frame([
        ("X", "2026-01-05", 100, 95, 105, 100, 1000, 10),       # clean, vwap 100
        ("X", "2026-01-06", 100, 95, 105, 100, 0, 0),           # no trades
        ("X", "2026-01-07", 110, 105, 110, 100, 1080, 10),      # close = previous close outside range: source rule
        ("X", "2026-01-08", 120, 125, 130, 128, 1270, 10),      # open below low: failure
        ("X", "2026-01-09", 100, 95, 105, 107, 1000, 10),       # close above high, not the previous close: failure
        ("X", "2026-01-10", 100, 95, 105, 100, 1000, 10),       # Saturday
        ("X", "2026-01-12", 100, 95, 105, 100, 2000, 10),       # vwap 200 outside range: source rule
        ("X", "2026-01-13", None, 95, 105, 100, 1000, 10.5),    # missing open, fractional volume: failure
    ])
    out = dse.validate(df, pd.Series(["2026-02-01T00:00:00+00:00"] * len(df)), tol=0.005)
    st = out["validation_status"].tolist()
    assert st == ["PASS", "PASS_NO_TRADES", "FLAGGED_SOURCE_RULE", "FAILED", "FAILED", "FLAGGED_SOURCE_RULE",
                  "FLAGGED_SOURCE_RULE", "FAILED"]
    f = out["validation_flags"].tolist()
    assert f[2] == "close_carried_forward_outside_low_high"
    assert "open_outside_low_high" in f[3]
    assert f[4] == "close_outside_low_high"
    assert f[5] == "weekend_date" and f[6] == "vwap_outside_low_high"
    assert "open_missing_or_non_numeric" in f[7] and "volume_not_integer" in f[7]
    assert out.at[0, "implied_vwap"] == pytest.approx(100.0)
    assert (out["extraction_status"] == "EXTRACTED").all()


def test_future_date_and_bad_date_fail() -> None:
    df = _frame([("X", "2026-03-02", 100, 95, 105, 100, 1000, 10)])
    out = dse.validate(df, pd.Series(["2026-02-01T00:00:00+00:00"]), tol=0.005)
    assert out.at[0, "validation_status"] == "FAILED" and "date_after_collection" in out.at[0, "validation_flags"]
    df.loc[0, "trading_date"] = pd.NaT
    out = dse.validate(df, pd.Series(["2026-02-01T00:00:00+00:00"]), tol=0.005)
    assert "invalid_date" in out.at[0, "validation_flags"]


def test_recollection_does_not_duplicate_and_reports_revisions() -> None:
    body, meta, rel = _snapshot()
    a, _ = dse.records_from_snapshot(body, meta, rel)
    b = a.copy()
    b["collected_at_utc"], b["snapshot_file"] = "2099-01-01T00:00:00+00:00", "later.html.gz"
    out, dups, rev = dse.combine([a, b])
    assert len(out) == len(a) and not out.duplicated(dse.KEY).any()
    assert (out["n_snapshots_seen"] == 2).all() and len(dups) == 0 and len(rev) == 0
    b.loc[0, "close_raw"] = "999999"
    out, _, rev = dse.combine([a, b])
    assert len(rev) == 1 and out.loc[out["trading_date"] == a.at[0, "trading_date"], "revised_later"].item()
    assert out.loc[out["trading_date"] == a.at[0, "trading_date"], "close_raw"].item() == a.at[0, "close_raw"]


def test_duplicate_inside_snapshot_reported() -> None:
    body, meta, rel = _snapshot()
    a, _ = dse.records_from_snapshot(body, meta, rel)
    a2 = pd.concat([a, a.iloc[[3]]], ignore_index=True)
    out, dups, _ = dse.combine([a2])
    assert len(out) == len(a) and len(dups) == 2


def test_processed_dataset_unique_and_classed() -> None:
    f = ROOT / "data" / "processed" / "dse_daily_prices.csv"
    if not f.exists():
        pytest.skip("processed DSE dataset not built")
    d = pd.read_csv(f)
    assert not d.duplicated(["ticker", "trading_date"]).any()
    assert d["evidence_class"].eq("Observed").all()
    assert d["validation_status"].isin(["PASS", "PASS_NO_TRADES", "FLAGGED_SOURCE_RULE", "FAILED"]).all()
