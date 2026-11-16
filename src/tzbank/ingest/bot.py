"""BoT and financial-inclusion intake (prompt 02 section 4; blueprint §6 data audit: sector totals, rates, reserve
requirements, credit by activity, payments, inclusion).

Every ``data/raw/bot/*.csv`` that follows ``TEMPLATE_bot_series.csv`` is validated and stacked into
data/interim/bot_long.csv. The template file itself is skipped. Problems are reported in
reports/validation/bot_intake_issues.csv; a file with a wrong header is a critical failure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

TEMPLATE_COLUMNS = ["series_id", "series_name", "category", "scope", "bank_name", "frequency", "period_end", "value",
                    "unit", "source_document", "page_table", "publication_date", "confidence", "notes"]
CATEGORIES = {"sector_totals", "rates", "reserve_requirements", "credit_by_activity", "interest_rates_spreads",
              "regulation", "payments", "inclusion", "other"}
SCOPES = {"sector", "bank"}
FREQUENCIES = {"daily", "weekly", "monthly", "quarterly", "annual", "event"}
UNITS = {"TZS", "USD", "ratio", "count", "index", "number", "TZS million", "TZS billion", "text"}
CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
TEMPLATE_NAME = "TEMPLATE_bot_series.csv"


def bot_dir(root: Path | None = None) -> Path:
    return path("raw", root or find_root()) / "bot"


def write_template(root: Path | None = None) -> Path:
    """data/raw/bot/TEMPLATE_bot_series.csv (header only)."""
    f = bot_dir(root) / TEMPLATE_NAME
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(f, index=False, lineterminator="\n")
    return f


def validate_frame(df: pd.DataFrame, name: str) -> list[dict[str, object]]:
    """Issues in one template file (empty list when valid)."""
    issues: list[dict[str, object]] = []

    def blank(v) -> bool:
        return v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip()

    def add(row, col, msg, critical=False):
        issues.append({"file": name, "row": row, "column": col, "issue": msg, "critical": critical})

    if list(df.columns) != TEMPLATE_COLUMNS:
        add(None, None, f"header differs from template: {list(df.columns)}", True)
        return issues
    for i, r in df.iterrows():
        n = i + 2
        if r["category"] not in CATEGORIES:
            add(n, "category", f"unknown category {r['category']!r}", True)
        if r["scope"] not in SCOPES:
            add(n, "scope", f"unknown scope {r['scope']!r}", True)
        if r["scope"] == "bank" and blank(r["bank_name"]):
            add(n, "bank_name", "bank scope needs bank_name", True)
        if r["frequency"] not in FREQUENCIES:
            add(n, "frequency", f"unknown frequency {r['frequency']!r}")
        if pd.isna(pd.to_datetime(r["period_end"], errors="coerce")):
            add(n, "period_end", "not a date", True)
        if pd.notna(r["publication_date"]) and pd.isna(pd.to_datetime(r["publication_date"], errors="coerce")):
            add(n, "publication_date", "not a date", True)
        if r["unit"] not in UNITS:
            add(n, "unit", f"unit {r['unit']!r} not in {sorted(UNITS)}")
        if r["unit"] != "text" and pd.isna(pd.to_numeric(r["value"], errors="coerce")):
            add(n, "value", "not numeric", True)
        if r["unit"] == "ratio" and abs(pd.to_numeric(r["value"], errors="coerce") or 0) > 1.5:
            add(n, "value", "ratio above 1.5: entered as a percentage?")
        if blank(r["source_document"]):
            add(n, "source_document", "value with no source", True)
        if r["confidence"] not in CONFIDENCE:
            add(n, "confidence", f"unknown confidence {r['confidence']!r}")
    dup = df[df.duplicated(["series_id", "period_end"], keep=False)]
    for i in dup.index:
        add(i + 2, "series_id", "duplicate series_id + period_end", True)
    return issues


def read_bot(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bot_long, issues) from every template file in data/raw/bot/."""
    parts, issues = [], []
    for f in sorted(bot_dir(root).glob("*.csv")):
        if f.name == TEMPLATE_NAME:
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False).replace({"": None})
        issues += validate_frame(df, f.name)
        if list(df.columns) == TEMPLATE_COLUMNS:
            parts.append(df.assign(file=f"bot/{f.name}"))
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*TEMPLATE_COLUMNS, "file"])
    long["period_end"] = pd.to_datetime(long["period_end"], errors="coerce")
    long["publication_date"] = pd.to_datetime(long["publication_date"], errors="coerce")
    long["value_numeric"] = pd.to_numeric(long["value"], errors="coerce")
    long["evidence_class"] = "Observed"
    return long, pd.DataFrame(issues, columns=["file", "row", "column", "issue", "critical"])


def auction_yields_template(results: pd.DataFrame) -> pd.DataFrame:
    """Worked example (prompt 02 section 4.3): BoT bond auction yields in the template layout (category rates)."""
    r = results[results["waytm"].notna()]
    return pd.DataFrame({
        "series_id": "BOT_TBOND_WAYTM_" + r["auction_number"].astype(int).astype(str),
        "series_name": "Treasury bond weighted average yield to maturity, auction " + r["auction_number"].astype(
            int).astype(str) + " (" + r["bond_title"].fillna("") + ")",
        "category": "rates", "scope": "sector", "bank_name": None, "frequency": "event",
        "period_end": r["auction_date"].dt.strftime("%Y-%m-%d"), "value": r["waytm"].round(6),
        "unit": "ratio", "source_document": "BoT TBonds auction summary (" + r["file"] + ")",
        "page_table": "AuctionSummaries: WEIGHTED AVERAGE YIELD-TO-MATURITY",
        "publication_date": r["auction_date"].dt.strftime("%Y-%m-%d"), "confidence": "HIGH",
        "notes": "Generated from data/raw/bot/auctions by tzbank.ingest.bot.auction_yields_template",
    })[TEMPLATE_COLUMNS]


def write_bot(root: Path | None = None) -> list[Path]:
    """Template (if missing), data/interim/bot_long.csv and the intake-issue report."""
    from tzbank.ingest.bot_database import write_bot_database

    root = root or find_root()
    write_template(root)
    long, issues = read_bot(root)
    files = write_bot_database(root)  # D40: the owner's research database (v1 + v2 merged), template schema
    db = path("interim", root) / "bot_research_database.csv"
    if files and db.exists():
        d = pd.read_csv(db, dtype=str, keep_default_na=False).replace({"": None}).drop(columns=["evidence_class"])
        d["period_end"] = pd.to_datetime(d["period_end"], errors="coerce")
        d["value_numeric"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.assign(file="Tanzania_Banking_Research_Database (v1+v2, D40)", evidence_class="Observed")
        long = pd.concat([long, d], ignore_index=True)
    return [*files, write_csv(long, path("interim", root) / "bot_long.csv"),
            write_csv(issues, path("validation", root) / "bot_intake_issues.csv")]
