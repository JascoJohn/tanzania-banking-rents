"""DSE data library: every listed security, its documents, shares in issue and an index proxy (prompt 15; D72).

Extends the D71 collector (``dse_trading_history``) with the same method: one GET per public page, the page's
embedded React flight payload decoded deterministically, every download saved unchanged as an immutable,
MANIFEST-registered snapshot, and an offline rebuild in the ingest stage. Nothing bypasses authentication, CAPTCHA,
paywalls or access restrictions: only pages and PDFs that the public site serves to any browser are fetched.

Pages (``dse.library`` in config/settings.yaml):

* ``/en/all-equities``: the securities array (symbol, name, market, last price, market cap).
* ``/en/main-market/company-profile/<T>``: overview facts (shares in issue, sector, listing domestic / cross-listed,
  market segment) and the "Recent Trades" panel (daily summaries as published).
* ``.../trading-history``: full daily history (handled by ``dse_trading_history.fetch``).
* ``.../financial-reports`` and ``.../disclosures``: document rows ``{date, title, type, href}``; every PDF is
  downloaded once to ``data/raw/dse/documents/<T>/`` (git-ignored; sha256 in the index and MANIFEST).
* ``.../dividends``: dividend rows (as published; often empty).
* ``/en/investors/products/indices``: checked for an embedded index history (none found on 2026-11-13: the page
  shows only the latest close, previous close, 52-week range and the level one year ago).

The offline build writes ``data/library/dse/`` (market data only; README there). Chapter datasets derive from it with
the 30 Oct 2026 cutoff (``tzbank.chapters.ch14_market_extensions``).
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tzbank.config import find_root, path, settings
from tzbank.ingest import dse_trading_history as th
from tzbank.io import write_csv, write_json, write_text

BASE = "https://dse.co.tz"
PAGES = {"overview": "/en/main-market/company-profile/{t}", "financial-reports":
         "/en/main-market/company-profile/{t}/financial-reports", "disclosures":
         "/en/main-market/company-profile/{t}/disclosures", "dividends": "/en/main-market/company-profile/{t}/dividends"}
SITE_PAGES = {"all-equities": "/en/all-equities", "indices": "/en/investors/products/indices",
              "big-blocks": "/en/investors/big-blocks"}


def lcfg(root: Path | None = None) -> dict[str, Any]:
    return settings(root or find_root())["dse"]


def raw(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / "dse"


def lib(root: Path | None = None) -> Path:
    return (root or find_root()) / "data" / "library" / "dse"


def _q(t: str) -> str:
    return urllib.parse.quote(t)


# ------------------------------------------------------------------------------------------------- network (by hand)
def _get(url: str, ua: str) -> tuple[int, bytes, str]:
    return th._download(url, ua)


def _save(sub: str, stem: str, url: str, code: int, body: bytes, err: str, extra: dict | None = None,
          root: Path | None = None) -> dict[str, Any]:
    d = raw(root) / sub
    d.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC)
    meta = {"source_url": url, "collected_at_utc": now.isoformat(timespec="seconds"), "http_status": code,
            "bytes": len(body), "error": err, "status": "OK" if not err and code == 200 else "FAILED", **(extra or {})}
    name = f"{stem}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    if body:
        (d / f"{name}.html.gz").write_bytes(gzip.compress(body, mtime=0))
        meta |= {"body_file": f"{name}.html.gz", "body_sha256": hashlib.sha256(body).hexdigest()}
    write_json(meta, d / f"{name}.json")
    return meta


def _latest(sub: str, stem: str, root: Path | None = None) -> tuple[bytes | None, dict | None]:
    ms = sorted((raw(root) / sub).glob(f"{stem}_*.json"))
    for m in reversed(ms):
        meta = json.loads(m.read_text(encoding="utf-8"))
        f = m.with_name(meta.get("body_file", ""))
        if meta.get("status") == "OK" and f.exists():
            return gzip.decompress(f.read_bytes()), meta
    return None, None


def fetch(root: Path | None = None, tickers: list[str] | None = None, documents: bool = True) -> dict[str, int]:
    """Download the site pages, every security's pages and (``documents``) every listed PDF not yet held."""
    root = root or find_root()
    c = lcfg(root)
    ua, pause = c["user_agent"], float(c["pause_seconds"])
    n = {"pages": 0, "failed": 0, "pdfs": 0, "pdf_failed": 0, "pdf_held": 0}
    for name, p in SITE_PAGES.items():
        code, body, err = _get(BASE + p, ua)
        m = _save("site", name, BASE + p, code, body, err, root=root)
        n["pages"] += 1
        n["failed"] += m["status"] != "OK"
        time.sleep(pause)
    sec = securities_from_snapshot(root)
    tick = tickers or sec["symbol"].tolist()
    th.fetch(tick, root)  # D71 trading history, one GET per ticker (skips tickers collected within min_refetch_hours)
    for t in tick:
        for name, p in PAGES.items():
            prev = sorted((raw(root) / "profile").glob(f"{t.replace(' ', '_')}__{name}_*.json"))
            if prev and json.loads(prev[-1].read_text(encoding="utf-8")).get("status") == "OK":
                age = dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(
                    json.loads(prev[-1].read_text(encoding="utf-8"))["collected_at_utc"])
                if age < dt.timedelta(hours=float(c["min_refetch_hours"])):
                    continue
            url = BASE + p.format(t=_q(t))
            code, body, err = _get(url, ua)
            m = _save("profile", f"{t.replace(' ', '_')}__{name}", url, code, body, err, {"ticker": t, "page": name},
                      root=root)
            n["pages"] += 1
            n["failed"] += m["status"] != "OK"
            time.sleep(pause)
        if documents:
            for r in document_rows(t, root).to_dict("records"):
                got = _download_pdf(r, ua, root)
                n[got] += 1
                if got != "pdf_held":
                    time.sleep(pause / 2)
    from tzbank.ingest.manifest import write_manifest

    write_manifest(root)
    return n


def _doc_file(r: dict, root: Path | None = None) -> Path:
    t = str(r["ticker"]).replace(" ", "_")
    base = Path(urllib.parse.urlparse(r["url"]).path).name or "document.pdf"
    return raw(root) / "documents" / t / base


def _download_pdf(r: dict, ua: str, root: Path | None = None) -> str:
    f = _doc_file(r, root)
    meta_f = f.with_suffix(f.suffix + ".json")
    if f.exists() and meta_f.exists():
        return "pdf_held"
    code, body, err = _get(r["url"], ua)
    ok = not err and code == 200 and body[:4] == b"%PDF"
    f.parent.mkdir(parents=True, exist_ok=True)
    meta = {"ticker": r["ticker"], "source_url": r["url"], "title": r["title"], "date": r["date"], "kind": r["kind"],
            "listing_page": r["listing_page"], "collected_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "http_status": code, "bytes": len(body), "status": "OK" if ok else "FAILED",
            "error": err or ("" if ok else "not a PDF")}
    if ok:
        f.write_bytes(body)
        meta["sha256"] = hashlib.sha256(body).hexdigest()
    write_json(meta, meta_f)
    return "pdfs" if ok else "pdf_failed"


# ------------------------------------------------------------------------------------------------- parsing (offline)
def _arrays(text: str, start_pat: str) -> list[list]:
    dec, out = json.JSONDecoder(), []
    for m in re.finditer(start_pat, text):
        try:
            arr, _ = dec.raw_decode(text, m.start() if start_pat.startswith(r"\[") else m.end() - 1)
        except json.JSONDecodeError:
            continue
        out.append(arr)
    return out


def securities_from_snapshot(root: Path | None = None) -> pd.DataFrame:
    body, meta = _latest("site", "all-equities", root)
    if body is None:
        return pd.DataFrame(columns=["symbol", "name", "market"])
    t = th.flight_text(body.decode("utf-8"))
    arrs = [a for a in _arrays(t, r'\[\{"symbol":') if a and isinstance(a[0], dict)]
    best = max(arrs, key=len)
    d = pd.DataFrame(best)
    d["snapshot_collected_utc"] = meta["collected_at_utc"]
    d["source_url"] = meta["source_url"]
    return d


def document_rows(ticker: str, root: Path | None = None) -> pd.DataFrame:
    rows = []
    for page in ("financial-reports", "disclosures"):
        body, meta = _latest("profile", f"{ticker.replace(' ', '_')}__{page}", root)
        if body is None:
            continue
        t = th.flight_text(body.decode("utf-8"))
        for arr in _arrays(t, r'"rows":\['):
            if arr and isinstance(arr[0], dict) and "href" in arr[0] and "date" in arr[0]:
                for r in arr:
                    href = r["href"]
                    url = href if href.startswith("http") else BASE + href
                    rows.append({"ticker": ticker, "date": r.get("date"), "title": r.get("title"),
                                 "kind": r.get("type") or ("disclosure" if page == "disclosures" else "report"),
                                 "url": url, "listing_page": page, "listing_snapshot": meta["body_file"]})
    d = pd.DataFrame(rows, columns=["ticker", "date", "title", "kind", "url", "listing_page", "listing_snapshot"])
    return d.drop_duplicates(["ticker", "url"])


FACTS = ["Shares in Issue", "Market Cap.", "Sector", "Listing", "Market Segment", "Listing Date", "Share Capital",
         "Company Type", "Incorporation"]


def overview_facts(ticker: str, root: Path | None = None) -> list[dict]:
    """Every overview snapshot's facts (one row per snapshot), as displayed."""
    out = []
    for m in sorted((raw(root) / "profile").glob(f"{ticker.replace(' ', '_')}__overview_*.json")):
        meta = json.loads(m.read_text(encoding="utf-8"))
        f = m.with_name(meta.get("body_file", ""))
        if meta.get("status") != "OK" or not f.exists():
            continue
        html = gzip.decompress(f.read_bytes()).decode("utf-8")
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", html))
        row = {"ticker": ticker, "collected_at_utc": meta["collected_at_utc"], "snapshot_file": meta["body_file"]}
        for k in FACTS:
            mm = re.search(re.escape(k) + r"\s*\|\s*(?:\|\s*)*([^|]{1,80}?)\s*\|", txt)
            row[k] = mm.group(1).strip() if mm else None
        out.append(row)
    return out


def recent_trades(ticker: str, root: Path | None = None) -> pd.DataFrame:
    body, meta = _latest("profile", f"{ticker.replace(' ', '_')}__overview", root)
    if body is None:
        return pd.DataFrame()
    t = th.flight_text(body.decode("utf-8"))
    start = t.find('"children":"Recent Trades"')
    if start < 0:
        return pd.DataFrame()
    seg = t[start:start + 60000]
    cells = re.findall(r'\["\$","td","(\d)",\{"className":"[^"]*","children":"([^"]*)"\}\]', seg)
    cols = ["date_raw", "open", "low", "high", "close", "turnover", "volume"]
    rows, cur = [], []
    for idx, val in cells:
        if int(idx) != len(cur):
            break
        cur.append(val)
        if len(cur) == len(cols):
            rows.append(dict(zip(cols, cur, strict=True)))
            cur = []
    d = pd.DataFrame(rows)
    if len(d):
        d.insert(0, "ticker", ticker)
        d["trading_date"] = pd.to_datetime(d["date_raw"], format="%d %b %Y", errors="coerce").dt.date.astype(str)
        for c in cols[1:]:
            d[c] = d[c].map(_num)
        d["record_type"] = "daily summary (date, open, low, high, close, turnover, volume as published): not trade-level"
        d["snapshot_file"] = meta["body_file"]
        d["evidence_class"] = "Observed"
    return d


# ------------------------------------------------------------------------------------------------- library build
def _num(x) -> float:
    v = th.number(x)
    return np.nan if v is None else v


def corporate_actions(root: Path, prices: pd.DataFrame) -> pd.DataFrame:
    """Confirmed actions (config/corporate_actions.csv) plus candidate breaks: close-to-close ratio jumps above the
    configured threshold on consecutive records (flagged, never applied)."""
    ca = pd.read_csv(root / "config" / "corporate_actions.csv")
    ca["ticker"] = ca["ticker"].astype(str)
    rows = [{"ticker": r.ticker, "action": r.action, "ratio": r.ratio, "effective_date": r.effective_date,
             "status": "confirmed (config/corporate_actions.csv)", "source": r.doc_id, "notes": r.notes}
            for r in ca.itertuples()]
    thr = float(lcfg(root)["library"]["break_threshold"])
    for t, g in prices.sort_values("trading_date").groupby("ticker"):
        g = g[g["close"] > 0]
        rr = g["close"] / g["close"].shift()
        traded = g["volume"].fillna(0) > 0
        last_trade = g["trading_date"].where(traded).ffill().shift()
        for i in g.index[(rr > 1 + thr) | (rr < 1 / (1 + thr))]:
            d = g.loc[i, "trading_date"]
            known = any((x["ticker"] == t) and abs((pd.Timestamp(x["effective_date"]) - d).days) <= 5 for x in rows
                        if x["status"].startswith("confirmed"))
            gap = (d - last_trade[i]).days if pd.notna(last_trade[i]) else None
            rows.append({"ticker": t, "action": "price break", "ratio": float(rr[i]), "effective_date": str(d.date()),
                         "status": "explained by a confirmed action" if known else "candidate (unconfirmed; not applied)",
                         "source": "close-to-close ratio on consecutive DSE records", "notes":
                         f"previous close {g['close'].shift()[i]:,.2f}, close {g.loc[i, 'close']:,.2f}; "
                         + (f"{gap} calendar days since the previous trade" if gap is not None else
                            "no earlier trade on record (listing reference price)")
                         + ("" if known else "; no split, bonus or consolidation notice among the DSE documents"),
                         "days_since_previous_trade": gap})
    return pd.DataFrame(rows)


def adjust(prices: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """Split-adjusted columns (confirmed splits only): prices before the effective date / ratio, volume x ratio."""
    p = prices.copy()
    p["split_factor"] = 1.0
    for r in ca[(ca["action"] == "split") & ca["status"].str.startswith("confirmed")].itertuples():
        m = (p["ticker"] == r.ticker) & (p["trading_date"] < pd.Timestamp(r.effective_date))
        p.loc[m, "split_factor"] *= float(r.ratio)
    for c in ("open", "low", "high", "close"):
        p[f"adj_{c}"] = p[c] / p["split_factor"]
    p["adj_volume"] = p["volume"] * p["split_factor"]
    return p


def holidays(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "config" / "tz_public_holidays.csv", parse_dates=["date"])


def classify_missing(missing: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    d = missing.copy()
    d["missing_weekday"] = pd.to_datetime(d["missing_weekday"])
    exact = hol.set_index("date")["holiday"].to_dict()
    approx = {r.date: r.holiday for r in hol.itertuples() if r.date_precision != "exact"}
    cls, why = [], []
    for x in d["missing_weekday"]:
        if x in exact:
            cls.append("public holiday")
            why.append(exact[x])
        elif any(abs((x - k).days) <= 1 for k in approx):
            cls.append("public holiday (moon-sighting date, +/- 1 day)")
            why.append(next(v for k, v in approx.items() if abs((x - k).days) <= 1))
        else:
            cls.append("unexplained (not a listed holiday)")
            why.append("")
    d["classification"], d["holiday"] = cls, why
    return d


def index_history(root: Path | None = None) -> pd.DataFrame:
    """Daily index levels embedded in the indices page (one year per index, as published; Observed)."""
    body, meta = _latest("site", "indices", root)
    if body is None:
        return pd.DataFrame()
    t = th.flight_text(body.decode("utf-8"))
    rows, seen = [], set()
    for m in re.finditer(r'\{"name":"([A-Za-z]+)","close":([-\d.]+)[^{}\[]*?"series":', t):
        name = m.group(1)
        if name in seen:
            continue
        arr = _arrays(t[m.end() - len('"series":'):], r'"series":\[')
        if not arr:
            continue
        seen.add(name)
        for ts, level in arr[0]:
            rows.append({"index": name, "trading_date": str(dt.datetime.fromtimestamp(ts, dt.UTC).date()),
                         "level": float(level), "close_on_page": float(m.group(2)),
                         "snapshot_file": meta["body_file"], "collected_at_utc": meta["collected_at_utc"],
                         "source_url": meta.get("source_url", BASE + SITE_PAGES["indices"]), "evidence_class": "Observed"})
    return pd.DataFrame(rows)


def index_proxy(prices: pd.DataFrame, shares: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Capitalisation-weighted price index (Derived): all equities (DSEI-like) and domestic only (TSI-like); ETFs
    excluded. Daily return = sum(shares_i x close_i,t) / sum(shares_i x close_i,t-1) over securities with a close
    on both days, shares from the latest overview (split-adjusted backwards); base 1,000 on the first date."""
    eq = sec[sec["market"] != "etf"]
    sh = shares.set_index("ticker")["shares_in_issue_current"].to_dict()
    cross = set(shares.loc[shares["listing"].astype(str).str.contains("Cross", case=False), "ticker"])
    p = prices[prices["ticker"].isin(eq["symbol"]) & prices["ticker"].isin(sh)].copy()
    p["shares_t"] = p["ticker"].map(sh) / p["split_factor"]  # pre-split share counts
    p["mcap"] = p["shares_t"] * p["close"]
    out = []
    for name, sel in (("all equities (DSEI-like)", p), ("domestic equities (TSI-like)", p[~p["ticker"].isin(cross)]),
                      ("all equities ex NMB and CRDB (event-study market proxy)",
                       p[~p["ticker"].isin(["NMB", "CRDB"])])):
        w = sel.pivot_table(index="trading_date", columns="ticker", values="mcap").sort_index()
        prev = w.shift()
        both = w.notna() & prev.notna()
        r = (w.where(both).sum(axis=1) / prev.where(both).sum(axis=1)).fillna(1.0)
        lvl = 1000 * r.cumprod()
        out.append(pd.DataFrame({"index": name, "trading_date": lvl.index, "level": lvl.values,
                                 "daily_return": r.values - 1, "constituents": both.sum(axis=1).values,
                                 "total_mcap_tzs": w.sum(axis=1).values}))
    d = pd.concat(out, ignore_index=True)
    d["method"] = ("capitalisation-weighted; shares in issue from the latest DSE overview, split-adjusted backwards; "
                   "constant share counts otherwise (rights / bonus issues not modelled)")
    d["evidence_class"] = "Derived"
    return d


def build(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or find_root()
    c = lcfg(root)["library"]
    b = th.build(root)
    prices = b["data"].copy()
    sec = securities_from_snapshot(root)
    facts = pd.DataFrame([r for t in sec["symbol"] for r in overview_facts(t, root)])
    shares = facts.sort_values("collected_at_utc").groupby("ticker").tail(1).copy() if len(facts) else facts
    if len(shares):
        shares["shares_in_issue_current"] = shares["Shares in Issue"].map(_num)
        shares["listing"] = shares["Listing"]
    ca = corporate_actions(root, prices)
    prices = adjust(prices, ca)
    prices = prices.sort_values(["ticker", "trading_date"])
    prev_close = prices.groupby("ticker")["close"].shift()
    prices["stale"] = (prices["volume"].fillna(0) == 0) | ((prices["close"] == prev_close)
                                                            & prices["validation_flags"].str.contains("carried", na=False))
    prices["stale_reason"] = np.where(prices["volume"].fillna(0) == 0, "no volume (close carried forward)",
                                      np.where(prices["stale"], "close carried forward outside the day's range", ""))
    tr = prices["turnover"].where(prices["volume"] > 0)
    med = tr.groupby(prices["ticker"]).transform(
        lambda x: x.dropna().rolling(60, min_periods=20).median().shift(1).reindex(x.index).ffill())
    prices["block_proxy"] = (prices["turnover"] > float(c["block_multiple"]) * med) | \
        prices["validation_flags"].str.contains("vwap_outside", na=False)
    prices["block_proxy_rule"] = (f"turnover > {c['block_multiple']}x the ticker's trailing median turnover (previous "
                                  "60 records with trades), or turnover / volume outside the day's low-high")
    sh_rows = []
    for r in facts.to_dict("records") if len(facts) else []:
        sh_rows.append({"ticker": r["ticker"], "observed_utc": r["collected_at_utc"],
                        "shares_in_issue": _num(r["Shares in Issue"]), "basis": "DSE overview page (as displayed)",
                        "evidence_class": "Observed"})
    for r in ca[(ca["action"] == "split") & ca["status"].str.startswith("confirmed")].itertuples():
        cur = shares.loc[shares["ticker"] == r.ticker, "shares_in_issue_current"]
        if len(cur):
            sh_rows.append({"ticker": r.ticker, "observed_utc": f"before {r.effective_date}",
                            "shares_in_issue": float(cur.iloc[0]) / float(r.ratio),
                            "basis": f"Derived: current shares / {r.ratio} split ratio", "evidence_class": "Derived"})
    docs = pd.concat([document_rows(t, root) for t in sec["symbol"]], ignore_index=True) if len(sec) else pd.DataFrame()
    if len(docs):
        st = []
        for r in docs.to_dict("records"):
            f = _doc_file(r, root)
            mf = f.with_suffix(f.suffix + ".json")
            m = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
            st.append({"local_file": f.relative_to(root).as_posix() if f.exists() else "",
                       "download_status": m.get("status", "not downloaded"), "sha256": m.get("sha256", ""),
                       "bytes": m.get("bytes", np.nan)})
        docs = pd.concat([docs.reset_index(drop=True), pd.DataFrame(st)], axis=1)
    rt = pd.concat([recent_trades(t, root) for t in sec["symbol"]], ignore_index=True) if len(sec) else pd.DataFrame()
    hol = holidays(root)
    miss = classify_missing(b["missing"], hol)
    idx = index_proxy(prices, shares, sec) if len(shares) else pd.DataFrame()
    for d in (prices, ca, docs, rt, miss):
        if len(d) and "evidence_class" not in d:
            d["evidence_class"] = "Observed"
    sec["evidence_class"] = "Observed"
    if len(shares):
        sec = sec.merge(shares[["ticker", *FACTS]].rename(columns={"ticker": "symbol"}), on="symbol", how="left")
    return {"daily_prices": prices, "securities": sec,
            "shares_in_issue": pd.DataFrame(sh_rows), "corporate_actions": ca, "documents_index": docs,
            "recent_trades": rt, "index_proxy": idx, "index_history": index_history(root), "missing_weekdays": miss,
            "crosscheck": b["crosscheck"]}


README = """# DSE data library (`data/library/dse/`)

Market data for every security listed on the Dar es Salaam Stock Exchange, collected from the public DSE website and
kept for reuse beyond this project. Rebuilt offline from the raw snapshots by `python -m tzbank run ingest`; refreshed
by `python -m tzbank.ingest.dse_library fetch` (network, by hand) followed by `run ingest`. Built {built}.

## Files (CSV; `daily_prices` also as parquet)

- `daily_prices` (security x trading date): ticker, trading_date, open, low, high, close, volume, turnover (as
  published); split_factor and adj_open / adj_low / adj_high / adj_close / adj_volume; stale, stale_reason;
  block_proxy; validation_status, validation_flags; provenance (source_url, snapshot_file, payload_index,
  pagination_ref, collected_at_utc).
- `securities` (security): symbol, name, market (main / etf), last, mcap (snapshot), Shares in Issue, Sector,
  Listing, Market Segment.
- `shares_in_issue` (security x observation): shares_in_issue, observed_utc, basis (displayed, or derived through a
  split).
- `corporate_actions`: action, ratio, effective_date, status (confirmed / explained / candidate; only confirmed
  actions adjust prices).
- `documents_index` (document): ticker, date, title, kind, url, local_file, sha256, download_status.
- `recent_trades`: the overview page's Recent Trades panel as published (daily summaries, not trade-level).
- `index_history` (index x date): DSEI, TSI and the sector indices as embedded in the indices page (the last year
  only; Observed). Longer index history is a paid DSE data product.
- `index_proxy` (index x date): level (base 1,000), daily_return, constituents (Derived; capitalisation-weighted).
- `missing_weekdays` (security x weekday without a record): classification against `config/tz_public_holidays.csv`.

## Sources

- `https://dse.co.tz/en/all-equities` (securities), `/en/main-market/company-profile/<T>` and its `trading-history`,
  `financial-reports`, `disclosures` and `dividends` pages; `/en/investors/products/indices` (checked; no history).
- Each page is fetched with one GET and saved unchanged under `data/raw/dse/` (gzip + metadata JSON, MANIFEST).
  PDFs are saved under `data/raw/dse/documents/<T>/` (git-ignored, third-party copyright) with sha256.

## Validation rules

Rows are never altered or imputed. Flags: invalid / weekend / future dates, missing or negative values, open or close
outside low-high (a close equal to the previous close is the DSE closing-price rule, labelled), VWAP outside low-high,
volume / turnover inconsistencies, duplicates and revisions across snapshots (first-collected kept). Price breaks
above {thr:.0%} close-to-close are listed as candidate corporate actions; only confirmed actions adjust prices.

## Terms

DSE website content and documents are third-party material. Raw snapshots, documents and library files are excluded
from any public release unless the DSE terms permit (OPEN_ISSUES 19).
"""


def write_library(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    b = build(root)
    out = lib(root)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for k, d in b.items():
        if k == "crosscheck":
            continue
        files.append(write_csv(d, out / f"{k}.csv"))
    p = b["daily_prices"].copy()
    for col in p.columns:
        if p[col].dtype == object:
            p[col] = p[col].astype("string")
    p.to_parquet(out / "daily_prices.parquet", index=False)
    files.append(out / "daily_prices.parquet")
    built = str(b["securities"]["snapshot_collected_utc"].max())[:10] if len(b["securities"]) else ""
    files.append(write_text(README.format(built=built,
                                          thr=float(lcfg(root)["library"]["break_threshold"])), out / "README.md"))
    return files


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="dse_library")
    ap.add_argument("mode", choices=["fetch", "build"])
    ap.add_argument("--ticker", action="append")
    ap.add_argument("--no-documents", action="store_true")
    a = ap.parse_args(argv)
    root = find_root()
    if a.mode == "fetch":
        print(fetch(root, a.ticker, documents=not a.no_documents))
    for f in write_library(root):
        print(Path(f).relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
