"""End-to-end: `run all` on the real data, completeness rule, processed-table rules (prompt 01 section 12)."""

import json

import pandas as pd
import pytest

from tzbank.pipeline import EXPECTED_OUTPUTS, GENERATED_FILES, run_stage


def test_expected_outputs_lie_under_outputs():
    """The pipeline test deletes EXPECTED_OUTPUTS; nothing outside outputs/ may be on that list (ch01-03 review A)."""
    outside = [f for f in EXPECTED_OUTPUTS if not f.startswith("outputs/") or ".." in f]
    assert outside == []
    assert set(EXPECTED_OUTPUTS) <= set(GENERATED_FILES)


def test_every_config_input_is_committed_or_hashed(root):
    """Every file under config/ is tracked by git or hashed in data/raw/MANIFEST.csv (ch01-03 review A)."""
    import subprocess

    try:
        tracked = set(subprocess.run(["git", "ls-files", "config"], cwd=root, capture_output=True, text=True,
                                     check=True).stdout.split())
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    manifest = (root / "data" / "raw" / "MANIFEST.csv").read_text(encoding="utf-8")
    loose = [f"config/{p.name}" for p in (root / "config").iterdir() if p.is_file()
             and f"config/{p.name}" not in tracked and p.name not in manifest]
    assert loose == []


def _raw_missing(root) -> list[str]:
    """Raw files listed in MANIFEST but not on disk. In the public repository (no research blueprint) the end-to-end
    run needs every one of them, fetched as docs/DATA_REQUIREMENTS.md describes (prompt 26, D97)."""
    from tzbank.reporting import blueprint as bp

    if bp.is_full(root):
        return []  # the private repository: the MANIFEST check in `run ingest` decides
    man = pd.read_csv(root / "data" / "raw" / "MANIFEST.csv", dtype=str)
    return [f for f in man["file"] if not (root / "data" / "raw" / f).exists()]


@pytest.fixture(scope="module")
def ran(root):
    missing = _raw_missing(root)
    if missing:  # checked before anything is deleted, so a public clone keeps its published outputs
        pytest.skip(f"end-to-end run needs the raw files: {len(missing)} listed in data/raw/MANIFEST.csv are not "
                    f"present (e.g. {missing[0]}); fetch them as docs/DATA_REQUIREMENTS.md describes")
    for f in EXPECTED_OUTPUTS:  # outputs/ only (test_expected_outputs_lie_under_outputs)
        p = root / f
        if p.exists():
            p.unlink()
    run_stage("all", root, skip_slow=True)  # PDF stage cached by input fingerprint (prompt 02f)
    return root


@pytest.mark.slow
def test_run_all_produces_every_expected_file(ran):
    missing = [f for f in GENERATED_FILES if not (ran / f).exists()]
    assert missing == []


@pytest.mark.slow
def test_every_manifest_output_has_its_notes_file(ran):
    """Completeness rule (CLAUDE.md 6): each listed table/figure names a notes file that exists."""
    for m in (ran / "outputs" / "manifests").glob("*.json"):
        man = json.loads(m.read_text(encoding="utf-8"))
        assert man["notes"], m.name
        for item in man.get("tables", []) + man.get("figures", []):
            assert (ran / item["path"]).exists()
            assert item["note"] in man["notes"] and (ran / item["note"]).exists()
            assert item["evidence_class"] in {"Observed", "Derived", "Estimated", "Scenario", "Interpretation"}
        for note in man["notes"]:
            text = (ran / note).read_text(encoding="utf-8")
            for heading in ["## Research question", "## Variable definitions", "## Sources", "## Validation test",
                            "## Evidence class", "## Limitations"]:
                assert heading in text, (note, heading)


@pytest.mark.slow
def test_output_tables_carry_evidence_class(ran):
    for f in ["outputs/tables/ch04_dupont_panel.csv", "data/processed/bank_period_long.csv",
              "data/processed/bank_year_wide.csv", "data/processed/market_daily.csv"]:
        df = pd.read_csv(ran / f, low_memory=False)
        assert df["evidence_class"].notna().all(), f


@pytest.mark.slow
def test_precedence_inputs_over_phase1(ran):
    long = pd.read_csv(ran / "data/processed/bank_period_long.csv", low_memory=False)
    iw = long[(long.provenance == "inputs_workbook") & long.value.notna()]
    p1 = long[(long.provenance == "phase1") & long.value.notna()]
    key = ["bank_id", "canonical_name", "period_type", "period_end", "basis"]
    clash = p1.merge(iw[iw.entity.isin(["Group", "Bank solo", "CRDB Bank (Tanzania, parent)"])][key].drop_duplicates(),
                     on=key)
    assert clash.empty
    replaced = pd.read_csv(ran / "reports/validation/provenance_replacements.csv")
    assert len(replaced) > 0
    # D26: B03-B07 keep Phase 1 for every line the workbook does not hold
    others = long[~long.bank_id.isin(["B01", "B02"])]
    assert "phase1" in set(others.provenance)


@pytest.mark.slow
def test_validation_has_no_critical_failures(ran):
    text = (ran / "reports/validation/validation_report.md").read_text(encoding="utf-8")
    assert "PASS - no critical failures" in text


@pytest.mark.slow
def test_run_is_deterministic(ran):
    """Re-running the build and profitability stages gives identical tables."""
    files = ["data/processed/bank_period_long.csv", "data/processed/bank_year_wide.csv",
             "outputs/tables/ch04_dupont_panel.csv"]
    before = {f: (ran / f).read_bytes() for f in files}
    run_stage("build", ran)
    run_stage("profitability", ran)
    assert all((ran / f).read_bytes() == before[f] for f in files)


@pytest.mark.slow
def test_dashboard_pages_render(ran):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ran / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) >= 4  # prompt 12a: landing page headline figures from the findings pack
    assert not any(m.value == "n/a" for m in at.metric)
    pages = sorted(p.relative_to(ran / "dashboard").as_posix() for p in (ran / "dashboard" / "pages").rglob("*.py"))
    assert len(pages) == 22  # chapters 1-12, chapter 4 subpages, page 13 report support (D67), page 14 (D70)
    for page in pages:
        at.switch_page(page).run()
        assert not at.exception, page
