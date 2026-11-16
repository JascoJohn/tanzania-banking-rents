"""Pipeline stages (CLAUDE.md section 4: one command rebuilds everything; each stage runs alone)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import rel, write_csv

log = logging.getLogger("tzbank")
SKIP_SLOW = False  # set by ``run all --skip-slow`` (prompt 02f section 5)


class StageFailed(RuntimeError):
    """A stage found a critical failure; the CLI exits non-zero."""


def stage_ingest(root: Path) -> list[Path]:
    """Raw -> interim (parse, recompute, canonicalise). Prompt 01 sections 3-4."""
    from tzbank.clean.normalise import write_canonical
    from tzbank.ingest import auction_results, auctions, bot, daily_original, finscope, inputs_workbook, nps, phase1
    from tzbank.ingest.manifest import verify_manifest

    check = verify_manifest(root)
    bad = check[check["status"].isin(["CHANGED", "MISSING"])]
    if len(bad):
        raise StageFailed(f"raw files changed or missing vs MANIFEST.csv: {bad['file'].tolist()}")
    files = [phase1.write_phase1_long(root)]
    files += inputs_workbook.write_inputs_interim(root)
    files += auctions.write_auctions(root)
    files += auction_results.write_results(root)
    files += bot.write_bot(root)
    # Prompt 15 pre-flight B: bot.write_bot rewrites bot_long.csv from the workbooks, which drops the SMR and
    # deposit-insurance rows the pdf stage appends. Re-apply them here (idempotent, from the last
    # bot_policy_text.csv) so a stand-alone `run ingest` leaves bot_long.csv complete.
    from tzbank.extract.bot_policy import merge_into_bot_long

    files += merge_into_bot_long(root)
    from tzbank.ingest.manual import write_manual

    files += write_manual(root)
    from tzbank.ingest.web_collected import write_web_collected

    files += write_web_collected(root)  # D64: web-collected macro forecasts and 2026 T-bill yields
    files += nps.write_nps(root)
    from tzbank.ingest.inclusion import write_inclusion

    files += write_inclusion(root)
    files += finscope.write_finscope(root)
    files += daily_original.write_daily_original(root)
    from tzbank.ingest.dse_trading_history import write_dataset as write_dse

    files += write_dse(root)  # D71: DSE trading history from saved snapshots (offline)
    from tzbank.ingest.dse_library import write_library

    files += write_library(root)  # D72: DSE data library (data/library/dse/) from saved snapshots (offline)
    from tzbank.ingest.damodaran import write_damodaran_archive

    files += write_damodaran_archive(root)  # D48
    files.append(write_canonical(root))
    return files


def stage_pdf(root: Path) -> list[Path]:
    """Work that reads the source PDFs (BoT report tables, inclusion/NPS supplements, PDF inspection and manual
    list), cached by input fingerprint; skipped without the PDFs. Prompt 02f section 5."""
    from tzbank.reporting.pdf_stage import run_pdf_stage

    return run_pdf_stage(root, SKIP_SLOW)


def stage_validate(root: Path) -> list[Path]:
    """Schemas, identities, coverage, conflicts. Prompt 01 section 6."""
    from tzbank.ingest.daily_original import crosscheck, read_original
    from tzbank.ingest.inputs_workbook import load_interim
    from tzbank.validate.run import run_validation

    ok, files = run_validation(root)
    cc = crosscheck(read_original(root=root), load_interim("Daily Market Data", root))
    files.append(write_csv(cc, path("validation", root) / "daily_market_crosscheck.csv"))
    if not ok:
        raise StageFailed("critical validation failures; see reports/validation/validation_report.md")
    return files


def stage_build(root: Path) -> list[Path]:
    """Interim -> processed tables, then processed schemas. Prompt 01 section 7."""
    from tzbank.ingest.adjudication import write_adjudication
    from tzbank.ingest.nps import write_nps_panel
    from tzbank.market.coe_inputs import write_coe_inputs
    from tzbank.panel.build import write_processed
    from tzbank.validate.plausibility import write_flags
    from tzbank.validate.run import processed_schema_errors

    files = write_processed(root) + write_coe_inputs(root) + write_nps_panel(root)
    files += write_adjudication(root) + write_flags(root)  # prompt 02d sections 1.2, 2.3, 3 and 7
    from tzbank.market.bot_tables import write_bot_tables

    files += write_bot_tables(root)  # prompt 02f section 3
    from tzbank.extract.sector_loans import write_sector_loans

    files += write_sector_loans(root)  # prompt 10: net-loans gate on the pdf-stage candidates (needs bank_year_wide)
    errs = processed_schema_errors(root)
    files.append(write_csv(pd.DataFrame(errs, columns=["table", "column", "check", "failure_case", "index"]),
                           path("validation", root) / "processed_schema_errors.csv"))
    if errs:
        raise StageFailed(f"{len(errs)} schema errors in processed tables")
    return files


def stage_audit(root: Path) -> list[Path]:
    """Data audit and generated docs. Prompt 01 sections 8 and 10."""
    from tzbank.reporting.audit import write_audit
    from tzbank.reporting.docs_gen import write_docs

    return write_audit(root) + write_docs(root)


def stage_profitability(root: Path) -> list[Path]:
    """Chapter 4 starter. Prompt 01 section 9."""
    from tzbank.chapters.ch04.business_model_map import write_business_model_map
    from tzbank.chapters.ch04.deployment import write_deployment
    from tzbank.chapters.ch04.dupont_attribution import write_dupont_attribution
    from tzbank.chapters.ch04.funding import write_funding
    from tzbank.chapters.ch04.operating_leverage import write_operating_leverage
    from tzbank.chapters.ch04.profitability import write_profitability
    from tzbank.chapters.ch04.revenue import write_revenue
    from tzbank.chapters.ch04.risk import write_risk
    from tzbank.chapters.ch04.transactions import write_transactions

    return (write_profitability(root) + write_dupont_attribution(root) + write_revenue(root) + write_funding(root)
            + write_deployment(root)  # prompt 03a: 4.0-4.3
            + write_transactions(root) + write_operating_leverage(root) + write_risk(root)
            + write_business_model_map(root))  # prompt 03b: 4.4-4.7 (4.7 last: it reads 4.0-4.6)


def stage_persistence(root: Path) -> list[Path]:
    """Chapter 5: annual COE, ER/RI, persistence models and calibration candidates (prompt 04)."""
    from tzbank.chapters.ch05.persistence import write_persistence

    return write_persistence(root)


def stage_funding(root: Path) -> list[Path]:
    """Chapter 6: SMR history, cost of deposits, deposit beta, net deposit franchise, liquidity (prompt 05)."""
    from tzbank.chapters.ch06_funding import write_funding

    return write_funding(root)


def stage_capital(root: Path) -> list[Path]:
    """Chapter 7: RORE, payout, capital headroom, leverage, distributable capacity (prompt 06)."""
    from tzbank.chapters.ch07_capital import write_capital

    return write_capital(root)


def stage_credit(root: Path) -> list[Path]:
    """Chapter 8: cost of risk through the cycle, IFRS 9 break, capital efficiency, stress lens (prompt 07)."""
    from tzbank.chapters.ch08_credit import write_credit

    return write_credit(root)


def stage_part1(root: Path) -> list[Path]:
    """Part I, chapters 1-3: concentration, credit allocation, systemic importance (prompt 10)."""
    from tzbank.chapters.ch01_concentration import write_part1

    return write_part1(root)


def stage_synthesis(root: Path) -> list[Path]:
    """Part II synthesis: consistency, rent sources, hypothesis register, calibration ranges (prompt 08)."""
    from tzbank.chapters.part2_synthesis import write_synthesis

    return write_synthesis(root)


def stage_valuation(root: Path) -> list[Path]:
    """Prompt 09a: proposed assumption register and the engine dry run (no valuation results published)."""
    from tzbank.valuation.run import write_valuation

    return write_valuation(root)


def stage_valuation_results(root: Path) -> list[Path]:
    """Prompt 09b: chapters 9-11 on the approved register (config/assumption_register.csv)."""
    from tzbank.chapters.ch09_valuation import write_valuation_results

    return write_valuation_results(root)


def stage_usd(root: Path) -> list[Path]:
    """Chapter 12: USD returns and implementation (prompt 11); after valuation_results (reads chapters 9 and 11)."""
    from tzbank.chapters.ch12_usd_implementation import write_ch12

    return write_ch12(root)


def stage_support(root: Path) -> list[Path]:
    """Prompt 13: report-support exhibits (chapter 13 page), after chapters 9-12."""
    from tzbank.chapters.ch13_report_support import write_ch13
    from tzbank.valuation.context import build_ctx

    return write_ch13(build_ctx(root))


def stage_market_ext(root: Path) -> list[Path]:
    """Prompt 14: market behaviour and valuation extensions (chapter 14 page), after the support stage."""
    from tzbank.chapters.ch13_analysts import write_analysts
    from tzbank.chapters.ch13_decisions import write_decisions
    from tzbank.chapters.ch13_exhibits import redraw_peer_exhibits
    from tzbank.chapters.ch13_p21_exhibits import write_p21
    from tzbank.chapters.ch13_peers import write_peers
    from tzbank.chapters.ch13_quarterly import write_quarterly
    from tzbank.chapters.ch13_text_exports import write_text_exports
    from tzbank.chapters.ch14_market_extensions import write_ch14
    from tzbank.valuation.context import build_ctx

    ctx = build_ctx(root)
    # text exports read chapter 14 (prompt 17 B); owner decisions read both (prompt 18)
    return (write_ch14(ctx) + write_text_exports(root) + write_decisions(ctx)
            + write_peers(ctx) + redraw_peer_exhibits(root)  # peer exhibits on the prompt 19 tables (prompt 23)
            + write_analysts(ctx) + write_quarterly(ctx)  # prompts 19-20 (D87-D89)
            + write_p21(ctx))  # prompt 21 exhibits E1-E6 (D90)


def stage_findings(root: Path) -> list[Path]:
    """Prompt 12a: findings pack (every figure and verdict the dashboard shows, with ids) after all chapters."""
    from tzbank.chapters.ch13_exhibits import refresh_index
    from tzbank.reporting.findings import write_findings
    from tzbank.reporting.hypotheses_all import write as write_hypotheses

    # exhibit index figure rows after every chart stage (chapter 14 draws after support); hypotheses_all before the
    # pack, which reads its manifest
    return refresh_index(root) + write_hypotheses(root) + write_findings(root)


def stage_review(root: Path) -> list[Path]:
    """Review pack for the owner (prompt 02b section 4)."""
    from tzbank.reporting.review_pack import write_review_pack

    return write_review_pack(root)


STAGES: dict[str, Callable[[Path], list[Path]]] = {
    "ingest": stage_ingest,
    "pdf": stage_pdf,
    "validate": stage_validate,
    "build": stage_build,
    "audit": stage_audit,
    "profitability": stage_profitability,
    "persistence": stage_persistence,
    "funding": stage_funding,
    "capital": stage_capital,
    "credit": stage_credit,
    "part1": stage_part1,
    "synthesis": stage_synthesis,
    "valuation": stage_valuation,
    "valuation_results": stage_valuation_results,
    "usd": stage_usd,
    "support": stage_support,
    "market_ext": stage_market_ext,
    "findings": stage_findings,
    "review": stage_review,
}

# Files that `run all` must produce (checked by tests/test_pipeline.py). Generated files outside outputs/ (data/,
# reports/, docs/, the proposed register) are checked for existence but never deleted by the tests (owner, ch01-03
# review A): inputs live beside them.
GENERATED_FILES = [
    "data/interim/phase1_long.csv", "data/interim/bond_auctions.csv", "data/interim/nps_long.csv",
    "data/interim/finscope_variables.csv", "data/interim/canonical_long.csv",
    "data/interim/daily_trading_original.csv", "data/interim/bond_auction_results.csv", "data/interim/bot_long.csv",
    "data/processed/dse_daily_prices.csv", "reports/validation/dse_validation_issues.csv",
    "data/interim/manual_values.csv", "data/interim/web_collected.csv",
    "data/library/dse/daily_prices.csv", "data/library/dse/daily_prices.parquet", "data/library/dse/index_proxy.csv",
    "data/library/dse/securities.csv", "data/library/dse/documents_index.csv", "data/library/dse/README.md",
    "reports/validation/web_collected_issues.csv", "data/interim/inclusion_long.csv", "data/interim/inclusion_data_issues.csv",
    "data/processed/inclusion_panel.csv", "data/processed/nps_panel.csv",
    "reports/validation/inclusion_vintage_check.csv",
    "data/processed/coe_inputs_annual.csv", "reports/validation/funding_and_pubdate_checks.csv",
    "data/processed/coe_long_bond_auctions_used.csv", "data/processed/coe_long_bond_comparison.csv",
    "reports/validation/plausibility_flags.csv", "reports/validation/outliers.csv",
    "data/processed/bank_year_flags.csv", "reports/inclusion_nps_adjudication.csv",
    "data/interim/bot_research_database.csv", "reports/validation/bot_v1_v2_diff.csv",
    "data/processed/rates_annual.csv", "data/processed/rates_quarterly.csv", "data/processed/sector_totals.csv",
    "data/processed/credit_by_activity.csv", "reports/bot_tables_summary.csv",
    *[f"data/interim/inputs_{s}.csv" for s in
      ["historical_financials", "interim_results", "quarterly_results", "regulatory_capital", "market_data", "macroeconomic_data",
       "peer_banks", "assumptions", "sources", "phase1_source_map", "daily_market_data"]],
    *[f"data/processed/{f}.csv" for f in ["bank_period_long", "bank_year_wide", "market_daily", "macro",
                                          "interim_long", "quarterly_long", "capital_long", "sources",
                                          "assumptions"]],
    "reports/data_audit_auction_yields_needed.csv", "reports/data_audit.md",
    "reports/data_audit_coverage.csv", "reports/data_audit_gaps.csv", "reports/data_audit_analysis_support.csv",
    "reports/validation/identity_checks.csv", "reports/validation/phase1_vs_inputs_conflicts.csv",
    "reports/validation/validation_report.md", "reports/validation/provenance_replacements.csv",
    "outputs/tables/ch04_dupont_panel.csv", "outputs/figures/ch04_roe_by_bank.png",
    "outputs/figures/ch04_roe_by_bank_data.csv", "outputs/notes/ch04_starter.md", "outputs/manifests/ch04.json",
    *[f"outputs/manifests/ch04_4_{i}.json" for i in range(8)], *[f"outputs/notes/ch04_4_{i}.md" for i in range(8)],
    "outputs/tables/ch04_4_0_components.csv", "outputs/tables/ch04_4_0_gap_attribution.csv",
    "outputs/tables/ch04_4_0_change_attribution.csv", "outputs/tables/ch04_4_1_composition.csv",
    "outputs/tables/ch04_4_2_funding.csv", "outputs/tables/ch04_4_3_deployment.csv",
    "outputs/tables/ch04_4_4_transactions.csv", "outputs/tables/ch04_4_4_market_context.csv",
    "outputs/tables/ch04_4_5_operating_leverage.csv", "outputs/tables/ch04_4_5_scale_elasticity.csv",
    "outputs/tables/ch04_4_6_risk.csv", "outputs/tables/ch04_business_model_map.csv",
    *[f"outputs/tables/ch05_{t}.csv" for t in ["coe_annual", "excess_returns", "persistence", "break_tests",
                                                "sign_persistence", "rank_persistence", "hypotheses",
                                                "calibration_candidates", "interval_coverage"]],
    "data/interim/damodaran_archive.csv",
    "data/interim/bot_policy_text.csv",
    *[f"outputs/tables/ch06_{t}.csv" for t in ["smr_annual", "rates_used", "deposits", "deposit_beta", "franchise",
                                                "liquidity", "hypotheses", "h61_parts"]],
    "outputs/notes/ch06.md", "outputs/manifests/ch06.json", "outputs/figures/ch06_franchise.png",
    *[f"outputs/tables/ch07_{t}.csv" for t in ["capital_raises", "rore", "payout", "capital_requirements", "capital",
                                                "leverage", "distributable_capacity", "hypotheses",
                                                "rore_windows", "rore_summary", "requirement_statements",
                                                "dsib_search", "distributable_capacity_total"]],
    "outputs/notes/ch07.md", "outputs/manifests/ch07.json", "outputs/figures/ch07_rore.png",
    *[f"outputs/tables/ch08_{t}.csv" for t in ["credit_history", "ttc_cost_of_risk", "ifrs9_transition",
                                                "ifrs9_transition_ratios", "h81_cost_of_risk_gap", "capital_efficiency",
                                                "panel_summary", "h82_rorwa_difference", "stress_lens", "hypotheses"]],
    "outputs/notes/ch08.md", "outputs/manifests/ch08.json", "outputs/figures/ch08_cost_of_risk.png",
    "outputs/figures/ch08_asset_quality.png", "outputs/figures/ch08_rorwa.png",
    "reports/part2_consistency.md", "reports/part2_consistency.csv", "reports/part2_executive_summary.md",
    "outputs/tables/part2_rent_sources.csv", "outputs/tables/hypotheses_status.csv",
    "outputs/tables/calibration_inputs.csv", "outputs/notes/part2.md", "outputs/manifests/part2.json",
    "config/assumption_register_proposed.csv", "outputs/valuation/dryrun_engine_checks.csv",
    "reports/valuation_09a_report.md", "outputs/valuation/manifest_09a.json",
    *[f"outputs/tables/ch09_{t}.csv" for t in ["valuations", "override_log", "sensitivity_grids", "icg_cap",
                                                "payout_path", "pit_verification", "hypotheses", "coe_ab", "dividends_12m",
                                                "coe_g_guard", "key_assumptions", "crdb_dividend_sensitivity"]],
    *[f"outputs/tables/ch10_{t}.csv" for t in ["implied", "iso_surface", "rerating", "bridge", "hypotheses"]],
    *[f"outputs/tables/ch11_{t}.csv" for t in ["scenarios", "scenario_evidence", "monte_carlo", "tested_call",
                                                "hypotheses"]],
    *[f"outputs/notes/ch{c}.md" for c in ("09", "10", "11")], *[f"outputs/manifests/ch{c}.json" for c in ("09", "10",
                                                                                                         "11")],
    *[f"outputs/tables/ch01_{t}.csv" for t in ["shares", "profit_vs_assets", "dse_shares", "dse_report",
                                                "hypotheses"]],
    *[f"outputs/tables/ch02_{t}.csv" for t in ["loan_mix", "bot_coverage", "dissimilarity", "credit_providers",
                                                "hypotheses"]],
    *[f"outputs/tables/ch03_{t}.csv" for t in ["dsib", "capital_buffers", "deposit_cost", "pb_roe", "hypotheses"]],
    *[f"outputs/notes/ch{c}.md" for c in ("01", "02", "03")], *[f"outputs/manifests/ch{c}.json" for c in ("01", "02",
                                                                                                         "03")],
    "data/interim/loans_by_sector.csv", "data/interim/loans_by_sector_candidates.csv",
    "reports/validation/loans_by_sector_review.csv",
    "data/interim/dse_market_report.csv",
    "reports/validation/bot_v1_v2_recheck.csv", "reports/validation/bot_crosswalk_check.csv",
    "reports/validation/bot_crosswalk_collisions.csv",
    *[f"outputs/tables/ch12_{t}.csv" for t in ["fx", "returns_annual", "returns_spans", "depreciation", "liquidity",
                                                "capacity", "expected_usd", "market_access", "hypotheses", "usd_steps",
                                                "cross_rates"]],
    "outputs/notes/ch12.md", "outputs/manifests/ch12.json",
    "outputs/findings/findings_pack_v1.17.csv", "outputs/findings/findings_pack_v1.17.md",
    "outputs/tables/ch13_text_fee_bases.csv", "outputs/tables/ch13_text_governance.csv",
    "outputs/tables/ch13_text_agri_support.csv", "outputs/tables/ch13_dec_coe_interval.csv",
    "outputs/tables/ch13_cr2_profit_2023_sensitivity.csv",
    "outputs/tables/ch13_constant_coe_sensitivity.csv",
    *[f"outputs/tables/ch13_text_{t}.csv" for t in ["loan_mix", "fee_mix", "implied_growth", "thesis_breaks",
                                                     "yield_minus_tbill", "real_excess", "rore", "usd_ytd"]],
    "outputs/tables/ch13_split_restated.csv", "outputs/manifests/ch13_text.json", "outputs/notes/ch13_text.md",
    *[f"outputs/tables/ch13_dec_{t}.csv" for t in ["interp_weights", "crossing_weights", "mc_conditional", "mc_inputs",
                                                    "configurations", "joint_grid", "bank_betas", "bank_coe",
                                                    "hist_coe", "capital_policies", "surplus_drag", "terminal_growth",
                                                    "price_requires", "implied_growth", "rate_roles", "rc_curr",
                                                    "event_robust", "outliers", "monitoring", "income_vs_return",
                                                    "peer_structure", "analyst_compare", "analyst_anchoring", "analyst_direction",
                                                    "rating_rule"]],
    "outputs/manifests/ch13_dec.json", "outputs/notes/ch13_dec.md",
    *[f"outputs/tables/ch13_peers_{t}.csv" for t in ["checks", "roe_oneoffs", "valuation", "table", "premium", "paths",
                                                     "annex"]],
    *[f"outputs/tables/ch13_analyst_{t}.csv" for t in ["methods", "targets", "accuracy", "anchoring", "our_model",
                                                        "revisions", "issues"]],
    *[f"outputs/tables/ch13_qtr_{t}.csv" for t in ["checks", "ttm", "monitor", "five", "ldr", "eps_split", "roll"]],
    "outputs/manifests/ch13_peers.json", "outputs/manifests/ch13_analysts.json", "outputs/manifests/ch13_qtr.json",
    *[f"outputs/figures/ch13_p21_{t}.png" for t in ["thesis", "roe_bridge", "dupont", "surplus", "rating_map",
                                                     "transmission"]],
    "outputs/manifests/ch13_p21.json", "outputs/notes/ch13_p21.md",
    "outputs/tables/hypotheses_all.csv", "outputs/manifests/hypotheses.json", "outputs/notes/hypotheses.md",
    *[f"outputs/tables/ch14_{t}.csv" for t in ["return_attribution", "event_study", "microstructure", "seasonality",
                                                "implied_growth", "price_lines_scenarios", "roe_lambda_grid",
                                                "real_roe_coe", "yield_minus_tbill", "annual_report_items", "catalysts",
                                                "thesis_breaks", "split_study", "split_objectives", "split_benchmarks",
                                                "comovement", "rolling_correlation", "index_proxy_check",
                                                "data_quality", "quarterly_fundamentals", "results_events",
                                                "daily_coverage"]],
    "outputs/notes/ch14.md", "outputs/manifests/ch14.json", "outputs/figures/ch14_return_attribution.png",
    "outputs/figures/ch14_split_study.png", "outputs/figures/ch14_seasonality.png",
    "outputs/figures/ch14_rolling_correlation.png", "reports/validation/quarterly_extraction_crosscheck.csv",
    "reports/validation/dse_library_reconciliation.csv", "reports/validation/quarterly_extraction_review.csv",
    "data/processed/quarterly_long.csv",
    "data/library/dse/index_history.csv",
    *[f"outputs/tables/ch13_{t}.csv" for t in ["roe_reconcile", "cr2_profit_direct", "implied_credit_shares",
                                                "earnings_bridge", "roe_waterfall", "roe_sustainable_bridge",
                                                "rating_flips", "stress_mixture", "rate_consistent", "named_scenarios",
                                                "deposit_migration", "rate_sensitivity", "casa_floor", "event_2018",
                                                "asset_beta", "rate_sensitivity_repricing", "repricing_profile",
                                                "valuation_bands", "kpi_table", "capacity_windows", "block_board",
                                                "franchise_breakeven", "surplus_drag", "financial_table",
                                                "price_series", "price_lines", "usd_exports"]],
    "outputs/notes/ch13.md", "outputs/manifests/ch13.json", "outputs/exhibits/index.csv", "outputs/figures/ch12_capacity.png",
    "outputs/tables/ch01_hhi_bound.csv",
    "data/interim/bot_fx_eop.csv", "reports/validation/bot_fx_check.csv",
    "outputs/tables/ch05_beta_sensitivity.csv", "outputs/tables/ch05_ifrs9_sensitivity.csv",
    "outputs/notes/ch05.md", "outputs/manifests/ch05.json", "outputs/figures/ch05_excess_returns.png",
    "outputs/figures/ch05_coe_annual.png",
    "docs/RESEARCH_HYPOTHESES.md", "docs/VARIABLE_DICTIONARY.md", "docs/DATA_REQUIREMENTS.md",
    "docs/SOURCE_REGISTER.md", "docs/EMPIRICAL_METHODOLOGY.md", "docs/EVIDENCE_AND_LIMITATIONS.md",
    "docs/RESEARCH_FRAME.md", "docs/NPS_PANEL_NOTES.md",
    "reports/review_pack/README.md", "reports/review_pack/pdf_inspection_summary.md",
    "reports/review_pack/validation_report.md", "reports/review_pack/coe_inputs_annual.csv",
]

# The only files the pipeline test may delete before `run all` (owner, ch01-03 review A): everything under outputs/.
EXPECTED_OUTPUTS = [f for f in GENERATED_FILES if f.startswith("outputs/")]


def run_stage(name: str, root: Path | None = None, skip_slow: bool = False) -> list[Path]:
    """Run one stage (or 'all') and write a run log per stage. ``skip_slow`` reuses cached PDF-stage outputs whose
    inputs are unchanged (prompt 02f section 5)."""
    from tzbank.reporting.runlog import write_run_log

    global SKIP_SLOW
    SKIP_SLOW = skip_slow

    root = root or find_root()
    names = list(STAGES) if name == "all" else [name]
    if any(n not in STAGES for n in names):
        raise KeyError(f"unknown stage {name!r}; choose from {', '.join([*STAGES, 'all'])}")
    produced: list[Path] = []
    for n in names:
        log.info("stage %s: start", n)
        try:
            files = STAGES[n](root)
        except StageFailed as exc:
            write_run_log(n, [], "FAILED", {"error": str(exc)}, root)
            raise
        produced += files
        rl = write_run_log(n, sorted({rel(f, root) for f in files}), "OK", root=root)
        log.info("stage %s: %d files; run log %s", n, len(files), rl.name)
    return produced
