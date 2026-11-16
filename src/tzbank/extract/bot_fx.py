"""TZS per USD, end of period (December), from the "selected economic indicators" table of each BoT Annual Report
(prompt 11 task 1; blueprint chapter 12). The BoT research database holds TZS/USD only to 2014.

Each table prints two rows, "Exchange rate (end of period)" and "Exchange rate (annual average)", for a run of years
whose header does not extract reliably from the PDF text. The years are therefore fixed by alignment, not by the
header: a table is placed at the offset where its annual-average row agrees with the already-known annual averages
(seed: the BoT research database series A4.7, annual average, to 2014) within 0.1% on at least two overlapping years
and on every overlap; placed tables extend the known series and the loop repeats. A table that cannot be placed
uniquely (e.g. fiscal-year columns, which are close to calendar averages but not within 0.1%) is not used and is
listed in the check report. Where several reports print the same year, the latest-published report wins (revisions,
cf. D5) and every overlap is listed with its relative difference. Publication date: the upload timestamp embedded in
the BoT file name, a conservative proxy (old reports were re-uploaded in 2020, so the proxy is never earlier than the
actual publication). Output: ``data/interim/bot_fx_eop.csv`` (Observed) and ``reports/validation/bot_fx_check.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.pdftext import _doc
from tzbank.io import write_csv

NUMS = r"((?:\d{1,2},\d{3}\.\d\s+){3,14})"
EOP = re.compile(r"Exchange rate(?: \(TZS per USD\))? \(end of (?:the )?period\)\s+" + NUMS, re.I)  # AR and MER tables
AVG = re.compile(r"Exchange rate(?: \(TZS per USD\))? \((?:annual|period) average\)\s+" + NUMS, re.I)
TOL = 0.001
SEED = "BOTDB_EXCHANGE_RATES__TZS_PER_USD_ANNUAL"


def upload_date(name: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{8})\d{6,8}", name)
    try:
        return pd.Timestamp(m.group(1)) if m else None
    except ValueError:
        return None


def _vals(s: str) -> list[float]:
    return [float(v.replace(",", "")) for v in s.split()]


def tables(root: Path) -> list[dict]:
    """Every (end-of-period, annual-average) row pair of equal length, de-duplicated per report."""
    out, seen = [], set()
    docs = path("documents", root)
    # BoT Annual Reports plus BoT Monthly Economic Reviews held in documents/bot (D64, issue 34: end-2023)
    from tzbank.extract.bot_reports import bot_report_files

    for f in bot_report_files(root) + sorted(docs.glob("bot/MER - *.pdf")):
        d = _doc(str(f))
        for i in range(d.page_count):
            t = re.sub(r"\s+", " ", d[i].get_text())
            e, a = EOP.search(t), AVG.search(t)
            if not e or not a:
                continue
            ev, av = _vals(e.group(1)), _vals(a.group(1))
            n = min(len(ev), len(av))
            key = (f.name, tuple(ev[:n]))
            if n < 3 or key in seen:
                continue
            seen.add(key)
            out.append({"file": f.name, "page": i + 1, "eop": ev[:n], "avg": av[:n], "pub": upload_date(f.name)})
    return out


def align(tabs: list[dict], seed: dict[int, float]) -> tuple[list[dict], list[dict]]:
    """Place each table at the unique start year where its averages match the known averages (see module doc)."""
    known = dict(seed)
    placed, left = [], list(tabs)
    progress = True
    while progress and left:
        progress = False
        for tb in list(left):
            fits = []
            for start in range(1990, 2031):
                ov = [(start + k, v) for k, v in enumerate(tb["avg"]) if start + k in known]
                if len(ov) >= 2 and all(abs(v / known[y] - 1) <= TOL for y, v in ov):
                    fits.append(start)
            if len(fits) == 1:
                tb["start"] = fits[0]
                for k, v in enumerate(tb["avg"]):
                    known.setdefault(fits[0] + k, v)
                placed.append(tb)
                left.remove(tb)
                progress = True
    return placed, left


def build(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = root or find_root()
    b = pd.read_csv(path("interim", root) / "bot_long.csv", usecols=["series_id", "period_end", "value"],
                    low_memory=False)
    s = b[b["series_id"] == SEED]
    seed = {int(str(p)[:4]): float(v) for p, v in zip(s["period_end"], pd.to_numeric(s["value"]), strict=True)}
    placed, left = align(tables(root), seed)
    rows = [{"year": tb["start"] + k, "period_end": f"{tb['start'] + k}-12-31", "tzs_per_usd": v,
             "annual_average": tb["avg"][k], "file": tb["file"], "page": tb["page"],
             "publication_date": tb["pub"].date() if tb["pub"] is not None else None}
            for tb in placed for k, v in enumerate(tb["eop"])]
    raw = pd.DataFrame(rows).sort_values(["year", "publication_date", "file"])
    final = raw.groupby("year").tail(1).copy()
    final["series_id"] = "BOT_AR_TZS_PER_USD_EOP_DEC"
    final["source"] = ("BoT Annual Report, selected economic indicators, 'Exchange rate (end of period)'; years fixed by "
                       "aligning the annual-average row (module doc)")
    final["evidence_class"] = "Observed"
    keep = final.set_index("year")
    chk = [{"year": r.year, "file": r.file, "page": r.page, "tzs_per_usd": r.tzs_per_usd,
            "kept_value": keep.loc[r.year, "tzs_per_usd"], "rel_diff_vs_kept": r.tzs_per_usd / keep.loc[r.year,
                                                                                                         "tzs_per_usd"] - 1,
            "status": "kept (latest publication)" if r.file == keep.loc[r.year, "file"] else "superseded"}
           for r in raw.itertuples()]
    chk += [{"year": None, "file": tb["file"], "page": tb["page"], "tzs_per_usd": None, "kept_value": None,
             "rel_diff_vs_kept": None, "status": "not used: averages do not align uniquely with the known annual averages "
             f"(eop {' '.join(f'{v:,.1f}' for v in tb['eop'])})"} for tb in left]
    return final.reset_index(drop=True), pd.DataFrame(chk).assign(evidence_class="Observed")


def write_bot_fx(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    final, chk = build(root)
    return [write_csv(final, path("interim", root) / "bot_fx_eop.csv", require_evidence=True),
            write_csv(chk, path("validation", root) / "bot_fx_check.csv", require_evidence=True)]
