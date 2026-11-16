"""pandera schemas for every interim and processed table (prompt 01 section 6.1)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

BANK = Check.str_matches(r"^B0[1-7]$")
EVIDENCE = Check.isin(["Observed", "Derived", "Estimated", "Scenario", "Interpretation"])
YN = Check.isin(["Y", "N"])


def _s(cols: dict[str, Column], unique: list[str] | None = None, strict: bool = False) -> DataFrameSchema:
    return DataFrameSchema(cols, unique=unique, strict=strict, coerce=False)


PHASE1_LONG = _s({
    "bank_id": Column(str, BANK),
    "year": Column("Int64", Check.in_range(2000, 2030), nullable=False),
    "variable": Column(str),
    "value": Column(float, nullable=True),
    "unit": Column(str, nullable=True),
    "status": Column(str, nullable=True),
    "confidence": Column(str, nullable=True),
    "source_document": Column(str, nullable=True),
    "sheet": Column(str, Check.str_startswith("RAW_")),
}, unique=["sheet", "row_number"])

_INPUT_BANK_SHEET = {
    "Row ID": Column(str, unique=True),
    "Bank ID": Column(str, BANK),
    "Basis": Column(str, Check.isin(["GROUP", "BANK"])),
    "Period End": Column("datetime64[ns]", nullable=False),
    "Variable": Column(str),
    "value_numeric": Column(float, nullable=True),
    "Status": Column(str, Check.isin(["REPORTED", "COMPUTED", "TO COLLECT", "NOT DISCLOSED", "NOT YET PUBLISHED"])),
    "Doc ID": Column(str, nullable=True),
    "publication_date": Column("datetime64[ns]", nullable=True),
    "eligible_historical": Column(str, YN),
    "eligible_current": Column(str, YN),
}
INPUT_BANK_SHEETS = {s: _s(_INPUT_BANK_SHEET) for s in
                     ["Historical Financials", "Interim Results", "Quarterly Results", "Regulatory Capital",
                      "Market Data"]}

INPUT_OTHER = {
    "Macroeconomic Data": _s({"Row ID": Column(str, unique=True), "Variable": Column(str),
                              "value_numeric": Column(float, nullable=True), "eligible_historical": Column(str, YN),
                              "eligible_current": Column(str, YN)}),
    "Peer Banks": _s({"Row ID": Column(str, unique=True), "Peer": Column(str), "Variable": Column(str)}),
    "Assumptions": _s({"ID": Column(str, unique=True), "Valuation": Column(str), "Parameter": Column(str)}),
    "Sources": _s({"Doc ID": Column(str, unique=True), "Publication Date": Column("datetime64[ns]", nullable=True)}),
    "Phase1 Source Map": _s({"Bank ID": Column(str, BANK), "Phase 1 Source Label": Column(str)},
                            unique=["Bank ID", "Phase 1 Source Label"]),
    "Daily Market Data": _s({"Row ID": Column(str, unique=True), "Date": Column("datetime64[ns]"),
                             "Bank ID": Column(str, BANK), "split_factor": Column(float, Check.gt(0)),
                             "adj_close": Column(float, Check.gt(0), nullable=True)},
                            unique=["Bank ID", "Date"]),
}

BOND_AUCTIONS = _s({
    "auction_number": Column(int), "title": Column(str), "auction_date": Column("datetime64[ns]"),
    "tenor_years": Column(int, Check.isin([2, 5, 7, 10, 15, 20, 25])),
    "coupon": Column(float, Check.in_range(0.03, 0.25), nullable=True),
    "evidence_class": Column(str, EVIDENCE),
}, unique=["auction_number", "auction_date"])

NPS_LONG = _s({"sheet": Column(str), "table_title": Column(str), "particular": Column(str),
               "year": Column(int, Check.in_range(2015, 2030)), "value": Column(float, nullable=True),
               "evidence_class": Column(str, EVIDENCE)},
              unique=["sheet", "table_title", "particular", "year"])

FINSCOPE_VARIABLES = _s({"variable": Column(str, unique=True), "label": Column(str, nullable=True),
                         "evidence_class": Column(str, EVIDENCE)})

CANONICAL_LONG = _s({
    "bank_id": Column(str, BANK),
    "period_end": Column("datetime64[ns]"),
    "period_type": Column(str, Check.isin(["FY", "Q1", "H1", "9M", "AS AT", "DIV"])),
    "canonical_name": Column(str),
    "basis": Column(str, Check.isin(["GROUP", "BANK"])),
    "value": Column(float, nullable=True),
    "provenance": Column(str, Check.isin(["inputs_workbook", "phase1"])),
    "is_restated": Column(bool),
    "evidence_class": Column(str, EVIDENCE),
}, unique=["provenance", "source_row_id"])

BANK_PERIOD_LONG = CANONICAL_LONG

BANK_YEAR_WIDE = _s({"bank_id": Column(str, BANK), "fiscal_year": Column(int, Check.in_range(2000, 2030)),
                     "evidence_class": Column(str, EVIDENCE)}, unique=["bank_id", "fiscal_year"])

MARKET_DAILY = _s({"date": Column("datetime64[ns]"), "bank_id": Column(str, BANK),
                   "adj_close": Column(float, Check.gt(0), nullable=True), "split_factor": Column(float, Check.gt(0)),
                   "evidence_class": Column(str, EVIDENCE)}, unique=["bank_id", "date"])

MACRO = _s({"Row ID": Column(str, unique=True), "canonical_name": Column(str), "evidence_class": Column(str, EVIDENCE)})

SOURCES = INPUT_OTHER["Sources"]
ASSUMPTIONS = INPUT_OTHER["Assumptions"]


def validate(schema: DataFrameSchema, df, name: str) -> list[dict[str, str]]:
    """Run a schema lazily; return a list of error dicts (empty when valid)."""
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
        rows = []
        for _, r in fc.head(50).iterrows():
            rows.append({"table": name, "column": str(r.get("column")), "check": str(r.get("check")),
                         "failure_case": str(r.get("failure_case"))[:120], "index": str(r.get("index"))})
        if len(fc) > 50:
            rows.append({"table": name, "column": "", "check": f"... {len(fc) - 50} more failures", "failure_case": "",
                         "index": ""})
        return rows
