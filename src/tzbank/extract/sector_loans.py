"""Loans and advances to customers by industry sector, from the IFRS 7 credit-risk concentration note of each NMB and
CRDB annual report (prompt 10 task 3; blueprint chapter 2; feeds chapter 4.3's sector book).

The note is a matrix: industry columns across the page, asset classes down it. On the page whose note says "Industry
sectors" the header (from the report's own year-end date to "Total") is split into industry labels with a vocabulary
of known column names; the "Loans and advances to customers" row (or its sub-rows: corporate, SME, microfinance,
mortgage, personal ...) is read as runs of n + 1 numbers ("-" = 0). Gates: (1) each run's industry values sum to its
printed total within 0.5%; (2) the summed loans total is within 5% of the bank's held net loans for that year (the
note uses carrying amounts). The first table on the page for the report's own year is used (Group before Bank where
both are printed). Output: ``data/interim/loans_by_sector.csv`` (Observed); unreconciled reports are listed with
the reason in ``reports/validation/loans_by_sector_review.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.detail_lines import report_year
from tzbank.extract.pdftext import _doc
from tzbank.io import write_csv

VOCAB = sorted([
    "financial institutions", "manufacturing", "government", "trading and commercial", "trading and commerce",
    "trading", "transport and communication", "transport & communication", "transport and communications",
    "hotel and restaurant", "hotels and restaurants", "hotel & restaurant", "wholesale & retail", "wholesale and retail",
    "agriculture", "individuals", "individual", "others", "other", "building and construction", "construction",
    "real estate", "mining and quarrying", "mining", "energy", "tourism", "education", "health", "personal",
    "public sector", "services", "mortgage", "households", "electricity", "oil and gas", "telecommunication",
], key=len, reverse=True)
STOP = re.compile(r"\b(debt|other assets|investment|as at|off|guarantees|letters|total financial|derivative)\b", re.I)
NUM = re.compile(r"\(?[\d]{1,3}(?:,\d{3})+\)?|\(?\d+\)?|(?<![\w])-(?![\w])")


def _clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)  # de-hyphenate line breaks: "Agricul- ture" -> "Agriculture"
    return t


def _header(seg: str) -> list[str]:
    s = seg.lower()
    out, i = [], 0
    while i < len(s):
        m = next((v for v in VOCAB if s.startswith(v, i)), None)
        if m:
            out.append(m)
            i += len(m)
        else:
            i += 1
    return out


def _val(tok: str) -> float:
    if tok == "-":
        return 0.0
    neg = tok.startswith("(")
    v = float(tok.strip("()").replace(",", ""))
    return -v if neg else v


def parse_page(t: str, fy: int) -> list[dict]:
    """Every candidate industry table on the page dated at the report's own year end (the caller gates them)."""
    t = _clean(t)
    out: list[dict] = []
    if not re.search(r"industry sectors", t, re.I) or not re.search(r"loans and advances to customers", t, re.I):
        return out
    for dm in re.finditer(rf"31(?:st)? december,? {fy}", t, re.I):
        rest = t[dm.end():]
        tot = re.search(r"\btotal\b", rest, re.I)
        if not tot:
            continue
        cols = _header(re.sub(r"(\w)-(\w)", r"\1\2", rest[:tot.start()]))
        n = len(cols)
        if n < 4:
            continue
        lm = re.search(r"loans and advances to customers", rest[tot.end():], re.I)
        if not lm:
            continue
        body = rest[tot.end() + lm.end():]
        stop = STOP.search(body)
        body = body[: stop.start()] if stop else body[:1500]
        toks = re.findall(r"\(?[\d]{1,3}(?:,\d{3})+\)?|\(?\d+\)?|-|[A-Za-z]+", body)
        nums = []
        for k, tok in enumerate(toks):
            if re.match(r"[A-Za-z]", tok):
                continue
            if tok == "-" and k + 1 < len(toks) and re.match(r"[A-Za-z]", toks[k + 1]):
                continue  # bullet before a sub-row label, not a value
            nums.append(_val(tok))
        if len(nums) < n + 1:
            continue
        runs = [nums[j: j + n + 1] for j in range(0, len(nums) - len(nums) % (n + 1), n + 1)]
        ok = [r for r in runs if abs(sum(r[:n]) - r[n]) <= 0.005 * max(abs(r[n]), 1)]
        if not ok or len(ok) != len(runs):
            continue
        sums = [sum(r[i] for r in ok) for i in range(n + 1)]
        out.append({"columns": cols, "values": sums[:n], "total": sums[n], "rows": len(ok),
                    "header": rest[:tot.start()][:200]})
    return out


CAND_COLS = ["bank_id", "fiscal_year", "file", "page", "candidate", "sector_label", "value", "total_tzs", "sub_rows",
             "evidence_class"]


def candidates(root: Path | None = None) -> pd.DataFrame:
    """Every gate-1 table (row sums reconcile) per report, in page order (pdf stage; no processed-data dependency)."""
    root = root or find_root()
    docs = path("documents", root)
    rows = []
    for folder, bank in (("NMB", "B02"), ("CRDB", "B01")):
        for f in sorted((docs / folder).glob("*.pdf")):
            fy = report_year(f)
            if not fy:
                continue
            d = _doc(str(f))
            k = 0
            for i in range(d.page_count):
                for h in parse_page(d[i].get_text(), fy):
                    for c, v in zip(h["columns"], h["values"], strict=True):
                        rows.append({"bank_id": bank, "fiscal_year": fy, "file": f"{folder}/{f.name}", "page": i + 1,
                                     "candidate": k, "sector_label": c, "value": v * 1e6, "total_tzs": h["total"] * 1e6,
                                     "sub_rows": h["rows"], "evidence_class": "Observed"})
                    k += 1
            if k == 0:  # record the report so the review lists it
                rows.append({"bank_id": bank, "fiscal_year": fy, "file": f"{folder}/{f.name}", "page": None,
                             "candidate": -1, "sector_label": None, "value": None, "total_tzs": None, "sub_rows": None,
                             "evidence_class": "Observed"})
    return pd.DataFrame(rows, columns=CAND_COLS)


def select(cand: pd.DataFrame, net: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate 2 (build stage): the first candidate per bank-year whose total is within 5% of held net loans."""
    rows, review = [], []
    for (bank, fy), g in cand.groupby(["bank_id", "fiscal_year"], sort=True):
        held = net.get((bank, fy))
        hit, why = None, "no 'Industry sectors' loans matrix found"
        for _, c in g[g["candidate"] >= 0].groupby(["file", "candidate"], sort=False):
            tot = float(c["total_tzs"].iloc[0])
            if held and abs(tot - held) > 0.05 * held:
                why = f"total {tot / 1e9:,.0f}bn vs held net loans {held / 1e9:,.0f}bn (> 5%)"
                continue
            hit = c
            break
        review.append({"bank_id": bank, "fiscal_year": int(fy), "file": "; ".join(sorted(g["file"].unique())),
                       "page": int(hit["page"].iloc[0]) if hit is not None else None,
                       "status": "reconciled" if hit is not None else why, "evidence_class": "Observed"})
        if hit is None:
            continue
        tot = float(hit["total_tzs"].iloc[0])
        for r in hit.itertuples():
            rows.append({"bank_id": bank, "fiscal_year": int(fy), "sector_label": r.sector_label, "value": r.value,
                         "share": r.value / tot if tot else None, "total_tzs": tot, "file": r.file, "page": int(r.page),
                         "sub_rows": int(r.sub_rows),
                         "gate": "industry values sum to the printed row total (0.5%); total within 5% of held "
                                 "net loans", "evidence_class": "Observed"})
    return pd.DataFrame(rows), pd.DataFrame(review)


def write_sector_candidates(root: Path | None = None) -> list[Path]:
    """pdf stage job: data/interim/loans_by_sector_candidates.csv."""
    root = root or find_root()
    return [write_csv(candidates(root), path("interim", root) / "loans_by_sector_candidates.csv",
                      require_evidence=True)]


def write_sector_loans(root: Path | None = None) -> list[Path]:
    """build stage (after bank_year_wide): gate 2 and data/interim/loans_by_sector.csv plus the review."""
    root = root or find_root()
    cand = pd.read_csv(path("interim", root) / "loans_by_sector_candidates.csv", dtype={"bank_id": str})
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    d, rv = select(cand, w.set_index(["bank_id", "fiscal_year"])["net_loans"])
    return [write_csv(d, path("interim", root) / "loans_by_sector.csv", require_evidence=True),
            write_csv(rv, path("validation", root) / "loans_by_sector_review.csv", require_evidence=True)]
