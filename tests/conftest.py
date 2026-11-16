"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tzbank.config import find_root


@pytest.fixture(scope="session")
def root() -> Path:
    return find_root()


# Public repository (prompt 26, D97): third-party raw files are not redistributed and generated folders are not
# committed. A test that fails only because such a file is absent is reported as skipped, with the file named.
NOT_PUBLISHED = ("data/raw/", "data/interim/", "data/processed/", "data/library/", "data/extraction/", "data/ocr/",
                 "reports/", "outputs/exhibits/", "outputs/market/", "outputs/reports/", "outputs/run_logs/",
                 "docs/RESEARCH_BLUEPRINT.md", "docs/OPEN_ISSUES.md", "drafts/")


def _missing_private(exc: BaseException) -> str | None:
    """The repository-relative path of a missing raw or generated file that ``exc`` reports, else None."""
    import re

    names = []
    if isinstance(exc, OSError) and exc.filename:
        names.append(str(exc.filename))
    names += re.findall(r"""['"]([^'"]+\.(?:csv|xlsx|json|md|parquet|pdf|txt|yaml))['"]""", str(exc))
    rt = find_root()
    for n in names:
        p = Path(n)
        p = p if p.is_absolute() else rt / p
        try:
            rel = p.resolve().relative_to(rt.resolve()).as_posix()
        except ValueError:
            continue
        if not p.exists() and rel.startswith(NOT_PUBLISHED):
            return rel
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is None or not rep.failed:
        return
    missing = _missing_private(call.excinfo.value)
    if missing:
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0,
                        f"Skipped: needs {missing}, which is not in the public repository (a third-party raw file or "
                        f"a pipeline output): fetch the raw files (docs/DATA_REQUIREMENTS.md) and run "
                        f"`python -m tzbank run all`")


def canon_row(**kw) -> dict:
    """A canonical-long row with defaults (inputs workbook, Group, FY, original)."""
    base = dict(bank_id="B01", entity="Group", basis="GROUP", period_type="FY",
                period_end=pd.Timestamp("2024-12-31"), canonical_name="x", category=None, value=1.0,
                provenance="inputs_workbook", is_restated=False, forecast_base_current=True,
                publication_date=pd.Timestamp("2025-03-27"), notes=None, source_row_id="r", doc_id="D")
    base.update(kw)
    return base


@pytest.fixture
def identity_fixture() -> pd.DataFrame:
    """One bank-year where every identity holds, one where each fails, hand-built."""
    good = {"interest_income": 1000.0, "interest_expense": 300.0, "net_interest_income": 700.0,
            "profit_before_tax": 400.0, "income_tax_expense": 120.0, "net_profit": 280.0,
            "profit_attributable": 270.0, "profit_nci": 10.0, "equity_attributable": 1900.0, "nci_equity": 100.0,
            "total_equity": 2000.0, "weighted_avg_shares": 100.0, "gross_loans": 5000.0,
            "loan_loss_allowance": 200.0, "net_loans": 4800.0, "tier1_capital": 150.0, "rwa_total": 1000.0,
            "tier1_ratio": 0.15}
    bad = dict(good, net_interest_income=690.0,          # 1000-300=700 vs 690: 1.45% off
               net_profit=290.0,                          # 400-120=280 vs 290 (3.4%); 270+10=280 vs 290
               total_equity=2050.0,                       # 1900+100=2000 vs 2050
               net_loans=4500.0,                          # 4800 vs 4500 (6.7%)
               tier1_ratio=0.16)                          # 0.15 vs 0.16 = 1.0 pp
    rows = []
    for year, vals in [(2023, good), (2024, bad)]:
        for k, v in vals.items():
            rows.append(canon_row(canonical_name=k, value=v, period_end=pd.Timestamp(f"{year}-12-31"),
                                  source_row_id=f"{year}-{k}"))
        eps = 2.70 if year == 2023 else 3.00             # 2.70 x 100 = 270 (pass); 3.00 x 100 = 300 vs 270 (fail)
        rows.append(canon_row(canonical_name="eps_basic", value=eps, provenance="phase1",
                              period_end=pd.Timestamp(f"{year}-12-31"), source_row_id=f"p1-{year}"))
    return pd.DataFrame(rows)
