"""DSE daily trading history per security (data layer; blueprint chapters 1 and 12, market data; D71).

How the page delivers its data (inspected 2026-11-13 on the CRDB page): ``/en/main-market/company-profile/<T>/
trading-history`` is a server-rendered Next.js page. The HTML table holds the first client page (20 rows) and the
pagination bar (``Prev 1 2 3 ... 147 Next``). Clicking a page number sends no request: the complete history is
already embedded in the page's React flight payload (``self.__next_f.push``) as the ``rows`` prop of the table
component, one array per trading day ``[date, open, high, low, close, volume, turnover]``. The column order was
checked against the rendered table (first and last client page). One GET therefore returns every record the site
exposes, and pagination is reconstructed deterministically from the payload order and the rendered page size.

Two steps, as in ``auction_results``:

* ``fetch`` (network, run by hand, not part of ``run all``) saves each download unchanged (gzip container) to
  ``data/raw/dse/trading_history/<T>_<UTC stamp>.html.gz`` with a metadata JSON (URL, collection time, HTTP status,
  sha256 of the body) and registers both in MANIFEST. Snapshots are never overwritten. Nothing bypasses any access
  control; requests are paced and identify the project.
* ``write_dataset`` (offline, in the ingest stage) rebuilds the dataset from every snapshot: one row per (ticker,
  trading date), first-collected values kept, later differing values written to a revisions report, duplicates
  reported, every row validated. Outputs: ``data/processed/dse_daily_prices.csv``,
  ``outputs/market/DSE_Daily_Prices_<T>.xlsx`` (+ extraction report) and ``reports/validation/dse_*.csv``.

Modes: ``initial`` downloads the full accessible history for a ticker; ``update`` downloads again and reports the
dates added since the previous build. Both rebuild from all snapshots, so re-collecting never duplicates a record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from tzbank.config import find_root, path, settings
from tzbank.io import write_csv, write_json, write_text

PAYLOAD_FIELDS = ["trading_date", "open", "high", "low", "close", "volume", "turnover"]  # order in the payload
HTML_HEADER = ["Date", "Open (TZS)", "Low (TZS)", "High (TZS)", "Close (TZS)", "Turnover (TZS)", "Volume"]
HTML_TO_FIELD = dict(zip(HTML_HEADER, ["trading_date", "open", "low", "high", "close", "turnover", "volume"],
                         strict=True))
VALUES = ["open", "low", "high", "close", "turnover", "volume"]
PRICES = ["open", "low", "high", "close"]
KEY = ["ticker", "trading_date"]


# ----------------------------------------------------------------------------------------------- configuration
def cfg(root: Path | None = None) -> dict[str, Any]:
    """The ``dse`` block of config/settings.yaml."""
    return settings(root or find_root())["dse"]


def raw_dir(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / "dse" / "trading_history"


def page_url(ticker: str, root: Path | None = None) -> str:
    import urllib.parse

    return cfg(root)["page_url"].format(ticker=urllib.parse.quote(ticker.upper()))  # e.g. "ZAN TIMIZA ETF"


# ----------------------------------------------------------------------------------------------- parsing
class _PageParser(HTMLParser):
    """Collects the company heading (h2 ticker, h4 name), the trading table and the pagination buttons."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[tuple[str, str]] = []
        self.header: list[str] = []
        self.rows: list[list[str]] = []
        self.pages: list[str] = []
        self._tag: str | None = None
        self._buf: list[str] = []
        self._in_table = self._in_nav = False
        self._row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self._in_table = True
        elif tag == "nav" and a.get("aria-label") == "Pagination":
            self._in_nav = True
        elif tag == "tr" and self._in_table:
            self._row = []
        if tag in {"h2", "h4", "th", "td", "button"}:
            self._tag, self._buf = tag, []

    def handle_data(self, data: str) -> None:
        if self._tag:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._tag:
            text = " ".join("".join(self._buf).split())
            if tag in {"h2", "h4"}:
                self.headings.append((tag, text))
            elif tag == "th" and self._in_table:
                self.header.append(text)
            elif tag == "td" and self._row is not None:
                self._row.append(text)
            elif tag == "button" and self._in_nav:
                self.pages.append(text)
            self._tag = None
        if tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._in_table = False
        elif tag == "nav":
            self._in_nav = False


def flight_text(html: str) -> str:
    """The concatenated React flight payload (``self.__next_f.push([1, "..."])`` string chunks, JSON-decoded)."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html, flags=re.S)
    return "".join(json.loads(c) for c in chunks)


def _is_trading_row(x: object) -> bool:
    return (isinstance(x, list) and len(x) == len(PAYLOAD_FIELDS) and isinstance(x[0], str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x[0]) is not None
            and all(v is None or isinstance(v, int | float) for v in x[1:]))


def payload_rows(html: str) -> list[list[Any]]:
    """The table component's ``rows`` array. Exactly one candidate must exist; otherwise the page has changed."""
    text, dec, found = flight_text(html), json.JSONDecoder(), []
    for m in re.finditer(r'"rows":\[', text):
        try:
            arr, _ = dec.raw_decode(text, m.end() - 1)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and arr and all(_is_trading_row(r) for r in arr):
            found.append(arr)
    if len(found) != 1:
        raise ValueError(f"expected one trading 'rows' array in the flight payload, found {len(found)}")
    return found[0]


def number(text: object) -> float | None:
    """Standardise a displayed or payload value: thousands commas removed; blank, dash or non-numeric -> None."""
    if text is None:
        return None
    if isinstance(text, int | float):
        return float(text) if math.isfinite(text) else None
    t = str(text).strip().replace(",", "").replace(" ", "")
    if t in {"", "-", "--", "N/A", "n/a"}:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_page(html: str) -> dict[str, Any]:
    """Company name, rendered table (first client page), page count and the full payload history."""
    p = _PageParser()
    p.feed(html)
    h2 = [i for i, (tag, _) in enumerate(p.headings) if tag == "h2"]
    name = next((p.headings[i + 1][1] for i in h2 if i + 1 < len(p.headings) and p.headings[i + 1][0] == "h4"), None)
    numbered = [int(b) for b in p.pages if b.isdigit()]
    return {"company_name": name, "html_header": p.header, "html_rows": p.rows,
            "n_pages": max(numbered) if numbered else None, "page_size": len(p.rows),
            "rows": payload_rows(html)}


# ----------------------------------------------------------------------------------------------- fetch (network)
def _download(url: str, user_agent: str) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (configured https host)
            return r.status, r.read(), ""
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b"", f"HTTP {e.code}"
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, b"", f"network error: {e}"


def snapshots(ticker: str | None = None, root: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Saved snapshots (metadata JSON + body), oldest first."""
    pat = f"{ticker.upper()}_*.json" if ticker else "*.json"
    out = []
    for m in sorted(raw_dir(root).glob(pat)):
        out.append((m.with_suffix(".html.gz"), json.loads(m.read_text(encoding="utf-8"))))
    return sorted(out, key=lambda x: x[1]["collected_at_utc"])


def fetch(tickers: list[str] | None = None, root: Path | None = None, force: bool = False) -> list[dict[str, Any]]:
    """Download each ticker's page once (skipped within ``min_refetch_hours`` unless ``force``). Returns the
    metadata of each attempt; a failed attempt is recorded (metadata with an error), never silently dropped."""
    root = root or find_root()
    c = cfg(root)
    d = raw_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    results = []
    for i, t in enumerate([x.upper() for x in (tickers or c["tickers"])]):
        prev = snapshots(t, root)
        now = dt.datetime.now(dt.UTC)
        if prev and not force:
            age = now - dt.datetime.fromisoformat(prev[-1][1]["collected_at_utc"])
            if age < dt.timedelta(hours=float(c["min_refetch_hours"])) and prev[-1][1]["status"] == "OK":
                results.append(prev[-1][1] | {"skipped": f"collected {age} ago (< min_refetch_hours)"})
                continue
        if i:
            time.sleep(float(c["pause_seconds"]))
        url = page_url(t, root)
        code, body, err = _download(url, c["user_agent"])
        meta: dict[str, Any] = {"ticker": t, "source_url": url, "collected_at_utc": now.isoformat(timespec="seconds"),
                                "http_status": code, "bytes": len(body), "error": err}
        if not err:
            try:
                info = parse_page(body.decode("utf-8"))
                meta |= {"records_in_payload": len(info["rows"]), "page_size": info["page_size"],
                         "n_pages": info["n_pages"], "company_name": info["company_name"]}
            except (ValueError, UnicodeDecodeError) as e:
                meta["error"] = f"parse error: {e}"
        meta["status"] = "OK" if not meta["error"] else "FAILED"
        stem = f"{t}_{now.strftime('%Y%m%dT%H%M%SZ')}"
        if body:
            body_file = d / f"{stem}.html.gz"
            body_file.write_bytes(gzip.compress(body, mtime=0))
            meta |= {"body_file": body_file.name, "body_sha256": _sha_bytes(body)}
        write_json(meta, d / f"{stem}.json")
        results.append(meta)
    from tzbank.ingest.manifest import write_manifest

    write_manifest(root)
    return results


def _sha_bytes(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------------------------- records
def records_from_snapshot(body: bytes, meta: dict[str, Any], rel_file: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per payload record, with original values, standardised values and the client-page reference.
    The rendered first page is parsed independently and compared value by value with the payload."""
    html = body.decode("utf-8")
    info = parse_page(html)
    size = info["page_size"] or 20
    rows = []
    for i, r in enumerate(info["rows"]):
        raw = dict(zip(PAYLOAD_FIELDS, r, strict=True))
        rec: dict[str, Any] = {"ticker": meta["ticker"], "company_name": info["company_name"],
                               "trading_date_raw": raw["trading_date"]}
        for f in VALUES:
            rec[f"{f}_raw"] = json.dumps(raw[f])
            rec[f] = number(raw[f])
        rec |= {"source_url": meta["source_url"], "collected_at_utc": meta["collected_at_utc"],
                "snapshot_file": rel_file, "payload_index": i,
                "pagination_ref": f"page {i // size + 1} of {info['n_pages']}, row {i % size + 1} "
                                  f"(client-side pagination, page size {size}; payload index {i})"}
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["trading_date"] = pd.to_datetime(df["trading_date_raw"], format="%Y-%m-%d", errors="coerce")
    check = _html_check(info, df)
    df["html_display_check"] = df["payload_index"].map(check["by_index"]).fillna("not rendered server-side")
    summary = {"snapshot_file": rel_file, "ticker": meta["ticker"], "collected_at_utc": meta["collected_at_utc"],
               "records_in_payload": len(df), "page_size": size, "n_pages_displayed": info["n_pages"],
               "n_pages_implied": math.ceil(len(df) / size) if size else None,
               "html_header_ok": info["html_header"] == HTML_HEADER, "html_rows_checked": check["n"],
               "html_rows_mismatched": check["bad"]}
    return df, summary


def _html_check(info: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Compare the server-rendered first page (display text) with the payload rows at the same positions."""
    by_index, bad = {}, 0
    if info["html_header"] != HTML_HEADER:
        return {"by_index": {}, "n": 0, "bad": 0}
    for i, cells in enumerate(info["html_rows"]):
        if i >= len(df) or len(cells) != len(HTML_HEADER):
            continue
        shown = dict(zip([HTML_TO_FIELD[h] for h in HTML_HEADER], cells, strict=True))
        ok = pd.to_datetime(shown["trading_date"], format="%d %b %Y", errors="coerce") == df.at[i, "trading_date"]
        ok = ok and all(number(shown[f]) == df.at[i, f] for f in VALUES)
        by_index[i] = "match" if ok else "MISMATCH"
        bad += 0 if ok else 1
    return {"by_index": by_index, "n": len(by_index), "bad": bad}


# ----------------------------------------------------------------------------------------------- validation
def validate(df: pd.DataFrame, collected: pd.Series, tol: float) -> pd.DataFrame:
    """Row checks (flags; nothing is changed or dropped). Blueprint data layer; CLAUDE.md section 3.7."""
    flags: list[list[str]] = [[] for _ in range(len(df))]

    def add(mask: pd.Series, flag: str) -> None:
        for k in mask[mask.fillna(False)].index:
            flags[df.index.get_loc(k)].append(flag)

    add(df["trading_date"].isna(), "invalid_date")
    add(df["trading_date"] > pd.to_datetime(collected.str[:10]), "date_after_collection")
    add(df["trading_date"].dt.dayofweek >= 5, "weekend_date")
    for f in VALUES:
        add(df[f].isna(), f"{f}_missing_or_non_numeric")
        add(df[f] < 0, f"{f}_negative")
    for f in PRICES:
        add(df[f] == 0, f"{f}_zero")
    for f in ["volume"]:
        add(df[f].notna() & (df[f] != df[f].round()), "volume_not_integer")
    add(df[["low", "open", "high"]].notna().all(axis=1) & ~((df["low"] <= df["open"]) & (df["open"] <= df["high"])),
        "open_outside_low_high")
    # The DSE close can be the previous close carried forward (closing-price rule: a day's trades do not always
    # set a new close), so a close outside [low, high] that equals the previous close is a source rule, not an error.
    prev_close = df.sort_values(KEY).groupby("ticker")["close"].shift().reindex(df.index)
    outside = df[["low", "close", "high"]].notna().all(axis=1) & ~((df["low"] <= df["close"])
                                                                   & (df["close"] <= df["high"]))
    add(outside & (df["close"] == prev_close), "close_carried_forward_outside_low_high")
    add(outside & (df["close"] != prev_close), "close_outside_low_high")
    add((df["volume"] > 0) & (df["turnover"] == 0), "turnover_zero_with_volume")
    add((df["volume"] == 0) & (df["turnover"] > 0), "volume_zero_with_turnover")
    vwap = df["turnover"] / df["volume"].where(df["volume"] > 0)
    add(vwap.notna() & ((vwap < df["low"] * (1 - tol)) | (vwap > df["high"] * (1 + tol))), "vwap_outside_low_high")
    out = df.copy()
    out["implied_vwap"] = vwap
    out["validation_flags"] = [";".join(f) for f in flags]
    hard = [";".join(x for x in f if x not in SOFT_FLAGS) != "" for f in flags]
    out["extraction_status"] = "EXTRACTED"
    out["validation_status"] = "PASS"
    out.loc[out["volume"] == 0, "validation_status"] = "PASS_NO_TRADES"
    out.loc[out["validation_flags"] != "", "validation_status"] = "FLAGGED_SOURCE_RULE"
    out.loc[hard, "validation_status"] = "FAILED"
    return out


# Flags that describe how DSE publishes the series (kept as published, labelled), not extraction or data errors.
SOFT_FLAGS = {"close_carried_forward_outside_low_high", "vwap_outside_low_high", "weekend_date"}


def combine(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge snapshots on (ticker, trading date): the first-collected record is kept; repeats with identical values
    are the same observation (counted as re-collections); differing values are revisions (reported, not applied);
    two records for one key inside one snapshot are duplicates in the source (reported, first kept)."""
    all_ = pd.concat(frames, ignore_index=True).sort_values(["collected_at_utc", "payload_index"], kind="stable")
    in_snap = all_.duplicated(["snapshot_file", *KEY], keep=False)
    dups = all_[in_snap].copy()
    base = all_.drop_duplicates(["snapshot_file", *KEY], keep="first")
    first = base.drop_duplicates(KEY, keep="first").set_index(KEY)
    later = base[base.duplicated(KEY, keep="first")]
    sig = [f"{f}_raw" for f in VALUES]
    joined = later.join(first[sig], on=KEY, rsuffix="_first")
    differs = pd.concat([joined[c] != joined[f"{c}_first"] for c in sig], axis=1).any(axis=1)
    revisions = joined[differs]
    seen = base.groupby(KEY).agg(n_snapshots_seen=("snapshot_file", "nunique"), last_seen_utc=("collected_at_utc",
                                                                                               "max"))
    out = first.join(seen).reset_index()
    out["first_collected_utc"] = out["collected_at_utc"]
    out["revised_later"] = out.set_index(KEY).index.isin(revisions.set_index(KEY).index)
    return out, dups, revisions


def missing_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    """Weekdays between the first and last record with no record. Expected to include public holidays and market
    closures; listed for review, never imputed (CLAUDE.md section 3.7)."""
    out = []
    for t, g in df.dropna(subset=["trading_date"]).groupby("ticker"):
        days = pd.bdate_range(g["trading_date"].min(), g["trading_date"].max())
        miss = days.difference(pd.DatetimeIndex(g["trading_date"]))
        out.append(pd.DataFrame({"ticker": t, "missing_weekday": miss}))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ticker", "missing_weekday"])
    res["note"] = "no DSE record; may be a public holiday or market closure (not imputed)"
    res["evidence_class"] = "Observed"
    return res


def crosscheck_original(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Compare with the owner's original daily file (data/interim/daily_trading_original.csv), same dates only."""
    f = path("interim", root) / "daily_trading_original.csv"
    cols = ["ticker", "trading_date", "field", "dse_page", "original_file", "status", "evidence_class"]
    if not f.exists():
        return pd.DataFrame(columns=cols)
    o = pd.read_csv(f, parse_dates=["Date"]).rename(columns={"Date": "trading_date", **{
        h: HTML_TO_FIELD[h] for h in HTML_HEADER[1:]}})
    m = df.merge(o[["ticker", "trading_date", *VALUES]], on=KEY, suffixes=("", "_orig"))
    rows = []
    for f_ in VALUES:
        a, b = m[f_], m[f"{f_}_orig"]
        same = (a == b) | (a.isna() & b.isna())
        for _, r in m[~same].iterrows():
            rows.append({"ticker": r["ticker"], "trading_date": r["trading_date"], "field": f_, "dse_page": r[f_],
                         "original_file": r[f"{f_}_orig"], "status": "CONFLICT", "evidence_class": "Observed"})
    summary = [{"ticker": t, "trading_date": pd.NaT, "field": "ALL", "dse_page": len(g), "original_file": None,
                "status": f"overlapping dates compared: {len(g)}", "evidence_class": "Observed"}
               for t, g in m.groupby("ticker")]
    return pd.DataFrame(summary + rows, columns=cols)


# ----------------------------------------------------------------------------------------------- build (offline)
OUT_COLS = ["ticker", "company_name", "trading_date", "open", "low", "high", "close", "turnover", "volume",
            "trading_date_raw", "open_raw", "low_raw", "high_raw", "close_raw", "turnover_raw", "volume_raw",
            "implied_vwap", "source_url", "collected_at_utc", "first_collected_utc", "last_seen_utc",
            "n_snapshots_seen", "snapshot_file", "pagination_ref", "payload_index", "html_display_check",
            "extraction_status", "validation_status", "validation_flags", "revised_later", "evidence_class"]


def build(root: Path | None = None) -> dict[str, Any]:
    """Dataset, issue tables and per-snapshot summaries from every saved snapshot."""
    root = root or find_root()
    tol = float(cfg(root)["vwap_tolerance"])
    frames, summaries, failed = [], [], []
    for body_file, meta in snapshots(root=root):
        rel_file = f"dse/trading_history/{body_file.name}"
        if meta.get("status") != "OK" or not body_file.exists():
            failed.append(meta | {"snapshot_file": rel_file})
            continue
        try:
            df, s = records_from_snapshot(gzip.decompress(body_file.read_bytes()), meta, rel_file)
        except (ValueError, UnicodeDecodeError) as e:
            failed.append(meta | {"snapshot_file": rel_file, "error": f"parse error: {e}", "status": "FAILED"})
            continue
        frames.append(df)
        summaries.append(s)
    if not frames:
        return {"data": pd.DataFrame(columns=OUT_COLS), "duplicates": pd.DataFrame(), "revisions": pd.DataFrame(),
                "missing": missing_weekdays(pd.DataFrame(columns=["ticker", "trading_date"])),
                "snapshots": pd.DataFrame(summaries), "failed": pd.DataFrame(failed),
                "crosscheck": pd.DataFrame()}
    data, dups, revisions = combine(frames)
    data = validate(data, data["collected_at_utc"], tol)
    data["evidence_class"] = "Observed"
    data = data.sort_values(KEY).reset_index(drop=True)[OUT_COLS]
    return {"data": data, "duplicates": dups, "revisions": revisions, "missing": missing_weekdays(data),
            "snapshots": pd.DataFrame(summaries), "failed": pd.DataFrame(failed),
            "crosscheck": crosscheck_original(data, root)}


def report(ticker: str, b: dict[str, Any], previous: pd.DataFrame | None = None) -> str:
    """Markdown extraction report for one ticker."""
    d = b["data"][b["data"]["ticker"] == ticker]
    s = b["snapshots"][b["snapshots"]["ticker"] == ticker] if len(b["snapshots"]) else b["snapshots"]
    f = b["failed"][b["failed"]["ticker"] == ticker] if len(b["failed"]) else b["failed"]
    last = s.iloc[-1] if len(s) else None
    flags = d.loc[d["validation_flags"] != "", "validation_flags"].str.split(";").explode().value_counts()
    nulls = {c: int(d[c].isna().sum()) for c in VALUES if d[c].isna().any()}
    cc = b["crosscheck"]
    cc = cc[cc["ticker"] == ticker] if len(cc) else cc
    lines = [f"# DSE daily trading history: {ticker} (extraction report)", "",
             f"Built {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} from {len(s)} snapshot(s) "
             f"under `data/raw/dse/trading_history/`. Evidence class: Observed.", "",
             "## Delivery mechanism",
             "Server-rendered Next.js page. The HTML table shows the first client page only; the full history is "
             "embedded in the page's React flight payload (`rows` prop: `[date, open, high, low, close, volume, "
             "turnover]`) and paginated in the browser without further requests. Extraction is deterministic "
             "(JSON decode of the payload; HTML parser for the rendered page as a cross-check).", "",
             "## Summary", "",
             f"- Records retrieved (unique ticker-date): **{len(d)}**",
             f"- Earliest / latest trading date: **{d['trading_date'].min():%Y-%m-%d}** / "
             f"**{d['trading_date'].max():%Y-%m-%d}**" if len(d) else "- No records",
             f"- Client pages reconstructed (latest snapshot): {last['n_pages_implied']} implied by "
             f"{last['records_in_payload']} records at page size {last['page_size']}; pagination bar shows "
             f"{last['n_pages_displayed']}" if last is not None else "- No successful snapshot",
             f"- Pages failed: {0 if last is not None and last['n_pages_implied'] == last['n_pages_displayed'] else 'see note'}"
             f" (all pages come from one payload; a failure is a whole-download failure)",
             f"- Failed downloads / unparsable snapshots: {len(f)}",
             f"- Rendered first page vs payload: {int(last['html_rows_checked'])} rows checked, "
             f"{int(last['html_rows_mismatched'])} mismatched" if last is not None else "",
             f"- Duplicate ticker-date records inside a snapshot: {len(b['duplicates'])}",
             f"- Revisions across snapshots (differing values, first kept): {len(b['revisions'])}",
             "- Extraction status: " + ", ".join(f"{k} {v}" for k, v in d["extraction_status"].value_counts().items()),
             "- Validation status: " + ", ".join(f"{k} {v}" for k, v in d["validation_status"].value_counts().items()),
             f"- Missing values: {nulls or 'none'}",
             f"- Weekdays with no record between first and last date: "
             f"{int((b['missing']['ticker'] == ticker).sum())} (holidays / closures expected; not imputed)"]
    if previous is not None:
        prev = previous[previous["ticker"] == ticker]
        new = set(d["trading_date"].dt.strftime("%Y-%m-%d")) - set(prev["trading_date"].astype(str))
        lines.append(f"- New dates since the previous build: {len(new)}"
                     + (f" ({min(new)} to {max(new)})" if new else ""))
    lines += ["", "## Validation flags", ""]
    lines += [f"- `{k}`: {v}" for k, v in flags.items()] or ["- none"]
    if len(cc):
        n = cc[cc["field"] == "ALL"]["dse_page"].sum()
        lines += ["", "## Cross-check with the owner's original daily file", "",
                  f"- Overlapping dates compared: {int(n)}; field conflicts: {int((cc['status'] == 'CONFLICT').sum())}"
                  " (listed in `reports/validation/dse_crosscheck_original.csv`)"]
    lines += ["", "## Coverage limitations", "",
              "- The page describes itself as 'the last 12 months', but the payload served on the collection date "
              "runs further back (see earliest date). The depth the site exposes may change; earlier snapshots are "
              "kept so coverage never shrinks in the dataset.",
              "- `close_carried_forward_outside_low_high`: the close equals the previous close although the day's "
              "trades ranged elsewhere (DSE closing-price rule); for every such row checked the close is the prior "
              "close. Kept as published; use with care in return calculations.",
              "- `vwap_outside_low_high`: turnover / volume lies outside the day's low-high (likely trades outside "
              "the board range, e.g. block or negotiated deals); kept as published.",
              "- Rows with zero volume and zero turnover (`PASS_NO_TRADES`) carry a price on a day without trades "
              "(e.g. 1 Jan); treat as no-trade days, not as transactions.",
              "- Prices are as reported by DSE (not adjusted for splits or dividends; adjustments live in "
              "`config/corporate_actions.csv`)."]
    return "\n".join(lines) + "\n"


def write_dataset(root: Path | None = None) -> list[Path]:
    """Ingest-stage entry point: processed CSV, per-ticker Excel and report, validation reports."""
    root = root or find_root()
    proc = path("processed", root) / "dse_daily_prices.csv"
    previous = pd.read_csv(proc, dtype=str) if proc.exists() else None
    b = build(root)
    val = path("validation", root)
    files = [write_csv(b["data"], proc, require_evidence=True)]
    issues = b["data"][b["data"]["validation_status"].isin(["FAILED", "FLAGGED_SOURCE_RULE"])]
    for name, df in [("dse_validation_issues", issues), ("dse_duplicates", b["duplicates"]),
                     ("dse_revisions", b["revisions"]), ("dse_missing_weekdays", b["missing"]),
                     ("dse_failed_snapshots", b["failed"]), ("dse_crosscheck_original", b["crosscheck"]),
                     ("dse_snapshots", b["snapshots"])]:
        files.append(write_csv(df, val / f"{name}.csv"))
    out = path("outputs", root) / "market"
    for t in sorted(b["data"]["ticker"].unique()):
        text = report(t, b, previous)
        files.append(write_text(text, out / f"dse_extraction_report_{t}.md"))
        files.append(_excel(t, b, text, out / f"DSE_Daily_Prices_{t}.xlsx"))
    return files


def _excel(ticker: str, b: dict[str, Any], text: str, file: Path) -> Path:
    def sel(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["ticker"] == ticker] if len(df) and "ticker" in df.columns else df

    file.parent.mkdir(parents=True, exist_ok=True)
    data = sel(b["data"]).copy()
    data["trading_date"] = data["trading_date"].dt.date
    with pd.ExcelWriter(file, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="DSE_Daily_Prices", index=False)
        sel(b["data"][b["data"]["validation_status"].isin(["FAILED", "FLAGGED_SOURCE_RULE"])]).to_excel(
            w, sheet_name="Validation_Issues", index=False)
        for name, key in [("Duplicates", "duplicates"), ("Revisions", "revisions"), ("Missing_Weekdays", "missing"),
                          ("Crosscheck_Original", "crosscheck"), ("Snapshots", "snapshots"),
                          ("Failed_Snapshots", "failed")]:
            sel(b[key]).to_excel(w, sheet_name=name, index=False)
        pd.DataFrame({"report": text.splitlines()}).to_excel(w, sheet_name="Extraction_Report", index=False)
        ws = w.sheets["DSE_Daily_Prices"]
        ws.freeze_panes = "D2"
        for col, width in zip("ABCDEFGHI", [8, 36, 12, 10, 10, 10, 10, 16, 12], strict=True):
            ws.column_dimensions[col].width = width
    return file


# ----------------------------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    """``python -m tzbank.ingest.dse_trading_history {initial,update,build} [--ticker CRDB ...] [--force]``."""
    ap = argparse.ArgumentParser(prog="dse_trading_history")
    ap.add_argument("mode", choices=["initial", "update", "build"],
                    help="initial: full history download; update: re-download and report new dates; build: offline")
    ap.add_argument("--ticker", action="append", help="DSE ticker (repeatable); default: config dse.tickers")
    ap.add_argument("--force", action="store_true", help="ignore min_refetch_hours")
    a = ap.parse_args(argv)
    root = find_root()
    if a.mode != "build":
        for m in fetch(a.ticker, root, force=a.force or a.mode == "initial"):
            print(f"{m['ticker']}: {m['status']} {m.get('skipped', '')} {m.get('error', '')} "
                  f"records={m.get('records_in_payload')} pages={m.get('n_pages')}".rstrip())
    for f in write_dataset(root):
        print(f.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
