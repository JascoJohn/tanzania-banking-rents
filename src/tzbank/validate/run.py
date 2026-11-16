"""Validation stage (prompt 01 section 6). Critical failures: schema errors, duplicate keys, a value with no source,
an unmapped variable, or a raw file that no longer matches MANIFEST.csv. The stage exits non-zero on any of them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from tzbank.clean.crosswalk import load_crosswalk, unmapped
from tzbank.clean.normalise import load_canonical
from tzbank.config import find_root, path, settings, valuation_dates
from tzbank.ingest.inputs_workbook import SHEETS, load_interim
from tzbank.ingest.manifest import verify_manifest
from tzbank.io import write_csv, write_text
from tzbank.pit import MissingPublicationDateWarning, eligible
from tzbank.validate import coverage, identities, schemas


def _read(f: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(f, low_memory=False, **kw)


def interim_schema_errors(root: Path) -> list[dict[str, str]]:
    """Validate every interim table against its pandera schema."""
    i = path("interim", root)
    errs: list[dict[str, str]] = []
    p1 = _read(i / "phase1_long.csv", dtype={"bank_id": str})
    p1["year"] = p1["year"].astype("Int64")
    errs += schemas.validate(schemas.PHASE1_LONG, p1, "phase1_long")
    for sheet in SHEETS:
        df = load_interim(sheet, root)
        sch = schemas.INPUT_BANK_SHEETS.get(sheet) or schemas.INPUT_OTHER.get(sheet)
        errs += schemas.validate(sch, df, f"inputs_{sheet}")
    au = _read(i / "bond_auctions.csv", parse_dates=["auction_date"])
    errs += schemas.validate(schemas.BOND_AUCTIONS, au, "bond_auctions")
    errs += schemas.validate(schemas.NPS_LONG, _read(i / "nps_long.csv"), "nps_long")
    errs += schemas.validate(schemas.FINSCOPE_VARIABLES, _read(i / "finscope_variables.csv"), "finscope_variables")
    errs += schemas.validate(schemas.CANONICAL_LONG, load_canonical(root), "canonical_long")
    return errs


def processed_schema_errors(root: Path) -> list[dict[str, str]]:
    """Validate every processed table (called at the end of the build stage)."""
    p = path("processed", root)
    errs: list[dict[str, str]] = []
    errs += schemas.validate(schemas.BANK_PERIOD_LONG, load_canonical(root, p / "bank_period_long.csv"),
                             "bank_period_long")
    errs += schemas.validate(schemas.BANK_YEAR_WIDE, _read(p / "bank_year_wide.csv", dtype={"bank_id": str}),
                             "bank_year_wide")
    errs += schemas.validate(schemas.MARKET_DAILY, _read(p / "market_daily.csv", parse_dates=["date"]),
                             "market_daily")
    errs += schemas.validate(schemas.MACRO, _read(p / "macro.csv"), "macro")
    errs += schemas.validate(schemas.SOURCES, _read(p / "sources.csv", parse_dates=["Publication Date"]), "sources")
    errs += schemas.validate(schemas.ASSUMPTIONS, _read(p / "assumptions.csv"), "assumptions")
    for name in ["interim_long", "quarterly_long", "capital_long"]:
        errs += schemas.validate(schemas.CANONICAL_LONG, load_canonical(root, p / f"{name}.csv"), name)
    return errs


def run_validation(root: Path | None = None) -> tuple[bool, list[Path]]:
    """Run all checks, write reports/validation/*, return (no critical failures, files written)."""
    root = root or find_root()
    s = settings(root)
    vdir = path("validation", root)
    files: list[Path] = []
    sheets = {sh: load_interim(sh, root) for sh in SHEETS}
    p1 = _read(path("interim", root) / "phase1_long.csv", dtype={"bank_id": str})
    canon = load_canonical(root)
    cw = load_crosswalk(root)

    manifest = verify_manifest(root)
    schema_errs = interim_schema_errors(root)
    dups = coverage.duplicate_keys(sheets, p1)
    nosrc = coverage.values_without_source(sheets, p1)
    missing_docs = coverage.doc_ids_missing_from_sources(sheets)
    pubcov = coverage.publication_date_coverage(canon)
    um = unmapped(cw, p1, sheets)
    checks = identities.run_identities(canon, s["identity_tolerances"])
    summary = identities.summarise(checks)
    conflicts = coverage.phase1_vs_inputs(canon, s["precedence_banks"], s["conflict_threshold"])
    compared = coverage.phase1_vs_inputs_compared(canon, s["precedence_banks"])
    unit_flags = canon[canon["unit_flag"].notna()][["bank_id", "fiscal_year", "canonical_name", "value", "unit",
                                                    "source_row_id", "unit_flag"]]
    pit_rows = []
    for name, d in valuation_dates(root).items():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", MissingPublicationDateWarning)
            el = eligible(canon[canon["value"].notna()], d)
        pit_rows.append({"valuation_date": name, "date": d.date().isoformat(), "rows_with_value":
                         int(canon["value"].notna().sum()), "eligible": len(el),
                         "excluded_no_publication_date": int(canon.loc[canon["value"].notna(), "publication_date"]
                                                             .isna().sum()),
                         "warning": str(w[0].message) if w else ""})
    pit = pd.DataFrame(pit_rows)

    critical = {
        "schema errors": len(schema_errs),
        "duplicate keys (two or more rows with a value)": int(dups["critical"].sum()) if len(dups) else 0,
        "values with no source (status claims an extracted value)": int(nosrc["critical"].sum()) if len(nosrc) else 0,
        "unmapped variables": len(um),
        "raw files changed or missing vs MANIFEST": int(manifest["status"].isin(["CHANGED", "MISSING"]).sum()),
    }
    files += [
        write_csv(checks, vdir / "identity_checks.csv", require_evidence=True),
        write_csv(summary, vdir / "identity_summary.csv"),
        write_csv(conflicts if len(conflicts) else pd.DataFrame(columns=["bank_id"]),
                  vdir / "phase1_vs_inputs_conflicts.csv"),
        write_csv(pd.DataFrame(schema_errs, columns=["table", "column", "check", "failure_case", "index"]),
                  vdir / "schema_errors.csv"),
        write_csv(dups, vdir / "duplicate_keys.csv"),
        write_csv(nosrc, vdir / "values_without_source.csv"),
        write_csv(missing_docs, vdir / "doc_ids_missing_from_sources.csv"),
        write_csv(pubcov, vdir / "publication_date_coverage.csv"),
        write_csv(um, vdir / "unmapped_variables.csv"),
        write_csv(unit_flags, vdir / "unit_flags.csv"),
        write_csv(pit, vdir / "pit_eligibility.csv"),
        write_csv(manifest, vdir / "manifest_check.csv"),
    ]
    report = render_report(critical, summary, checks, conflicts, compared, dups, nosrc, missing_docs, pubcov, unit_flags,
                           pit, manifest, schema_errs)
    from tzbank.validate.funding import funding_checks, publication_date_checks

    fchecks = pd.concat([funding_checks(canon), publication_date_checks(sheets["Sources"])], ignore_index=True)
    files.append(write_csv(fchecks, vdir / "funding_and_pubdate_checks.csv", require_evidence=True))
    from tzbank.validate.funding import publication_date_changes

    pchg = publication_date_changes(sheets["Sources"], valuation_dates(root))
    files.append(write_csv(pchg, vdir / "publication_date_changes.csv"))
    warn = fchecks[fchecks["status"] == "WARN"]
    report += "\n".join([
        "", "", "## Funding and publication-date checks (prompt 02 section 5.2; non-critical)", "",
        _md_table(fchecks.groupby("check")["status"].value_counts().unstack(fill_value=0).reset_index()), "",
        f"Publication dates changed by D16: {int((~pchg['new_document']).sum()) if len(pchg) else 0} existing "
        f"documents re-dated, {int(pchg['new_document'].sum()) if len(pchg) else 0} new documents dated; eligibility "
        "changed at a valuation date for an existing document: "
        f"{int(pchg['eligibility_changed'].sum()) if len(pchg) else 0} (`publication_date_changes.csv`).", "",
        "### Warnings", "",
        _md_table(warn.drop(columns=["evidence_class"]).head(60), "{:,.4f}"), ""])
    report += inclusion_section(root)
    files.append(write_text(report, vdir / "validation_report.md"))
    return sum(critical.values()) == 0, files


def inclusion_section(root: Path) -> str:
    """Financial inclusion and NPS panels (prompt 02c sections 2.3-2.4 and 4; non-critical, D33)."""
    vdir, idir = path("validation", root), path("interim", root)
    lines = ["", "## Financial inclusion and NPS panels (prompt 02c; non-critical)", ""]
    vc = vdir / "inclusion_vintage_check.csv"
    if vc.exists():
        v = pd.read_csv(vc)
        lines += ["### D31 vintage rule vs Panel_Wide", "",
                  _md_table(v["status"].value_counts().rename_axis("status").reset_index()), ""]
        bad = v[v["status"] != "MATCH"]
        if len(bad):
            lines += [_md_table(bad.drop(columns=["evidence_class"]).head(60), "{:,.4f}"), ""]
    di = idir / "inclusion_data_issues.csv"
    if di.exists():
        lines += ["### Known source issues (owner's Data_Issues sheet, carried as delivered)", "",
                  _md_table(pd.read_csv(di)), ""]
    cc = vdir / "inclusion_nps_crosscheck.csv"
    if cc.exists():
        c = pd.read_csv(cc)
        lines += ["### Independent PDF cross-check (verification only; panel values never changed)", "",
                  "Written by `python -m tzbank.extract.inclusion_crosscheck` (needs the git-ignored PDFs). Rows "
                  "that are not a clean match:", "",
                  _md_table(c.groupby(["panel", "check_status"]).size().rename("rows").reset_index()), ""]
    return "\n".join(lines)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_none_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        elif pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in d.astype(str).to_numpy()]
    return "\n".join([head, sep, *body])


def render_report(critical, summary, checks, conflicts, compared, dups, nosrc, missing_docs, pubcov, unit_flags, pit,
                  manifest, schema_errs) -> str:
    """validation_report.md text."""
    n_crit = sum(critical.values())
    fails = checks[(checks["vintage"] == "original") & (checks["status"] != "PASS")]
    lines = [
        "# Validation report",
        "",
        "Generated by `python -m tzbank run validate` (`tzbank.validate.run`). Prompt 01 section 6.",
        "",
        f"**Result: {'PASS - no critical failures' if n_crit == 0 else f'FAIL - {n_crit} critical failures'}.**",
        "",
        "## Critical checks",
        "",
        _md_table(pd.DataFrame({"check": list(critical), "count": list(critical.values())})),
        "",
        "## Accounting identities (original vintage)",
        "",
        "Tolerances: NII, PBT-tax, profit split and equity split 0.5%; EPS 2%; loans 2% (Bank-level gross loans "
        "flagged, not failed); Tier 1 ratio 0.2 pp. Full results, including the latest-restatement vintage, in "
        "`identity_checks.csv`.",
        "",
        _md_table(summary),
        "",
        "### Non-passing identity rows (non-critical)",
        "",
        _md_table(fails[["identity", "bank_id", "period_end", "period_type", "basis", "lhs", "rhs",
                         "relative_difference", "status", "flag_reason"]], "{:,.4f}"),
        "",
        "## Phase 1 vs inputs workbook (NMB/CRDB)",
        "",
        f"Value pairs compared (same canonical variable, FY, basis): {compared}. Differences above 1%: "
        f"{len(conflicts)}. Listed in `phase1_vs_inputs_conflicts.csv`; not resolved. The precedence rule "
        "(CLAUDE.md 3.8) uses the inputs workbook value in `data/processed`; every replaced Phase 1 row is in "
        "`provenance_replacements.csv`.",
        "",
        _md_table(conflicts[["bank_id", "canonical_name", "fiscal_year", "basis", "value_phase1", "value_inputs",
                             "relative_difference", "matches_other_vintage"]] if len(conflicts) else conflicts,
                  "{:,.4f}"),
        "",
        "## Coverage (non-critical)",
        "",
        "### Duplicate keys",
        "",
        "Rows sharing a key. Critical only when two or more of them carry a value.",
        "",
        _md_table(dups),
        "",
        "### Values without a source",
        "",
        _md_table(nosrc),
        "",
        "### Doc IDs missing from Sources",
        "",
        _md_table(missing_docs),
        "",
        "### Publication-date coverage (rows with a value)",
        "",
        "Phase 1 rows carry a publication date only where the inputs workbook's Phase1 Source Map assigns a Doc ID. "
        "Rows without one are excluded by `tzbank.pit.eligible` (blueprint limitation L9).",
        "",
        _md_table(pubcov),
        "",
        "### Point-in-time eligibility",
        "",
        _md_table(pit),
        "",
        "### Ratio values above 1.5 (verify; not changed)",
        "",
        _md_table(unit_flags),
        "",
        "### Raw files vs MANIFEST",
        "",
        _md_table(manifest["status"].value_counts().rename_axis("status").reset_index(name="files")),
        "",
        "### Schema errors",
        "",
        _md_table(pd.DataFrame(schema_errs)),
    ]
    return "\n".join(lines)
