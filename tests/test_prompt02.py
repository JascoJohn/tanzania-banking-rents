"""Prompt 02: OCR/extraction helpers, publication-date rules, BoT intake, COE inputs, funding checks."""

from pathlib import Path

import pandas as pd
import pytest

from tzbank.extract.pdftext import Row, merge_orphans, parse_number
from tzbank.ingest.bot import TEMPLATE_COLUMNS, auction_yields_template, validate_frame
from tzbank.market.coe_inputs import coe_inputs_annual
from tzbank.pit.pubdates import ASSUMED, REG_2001_A, REG_2014_A, REG_2014_Q, annual_rule, quarterly_rule, resolve
from tzbank.validate.funding import funding_checks


@pytest.mark.parametrize("tok,val", [("(1,234)", -1234.0), ("1,234", 1234.0), ("-", 0.0), ("12.5", 12.5),
                                     ("(547,277)", -547277.0), ("2,936,631", 2936631.0), ("abc", None)])
def test_parse_number_brackets_separators_decimals(tok, val):
    assert parse_number(tok) == val


def test_merge_orphans_joins_ocr_number_row_to_label_row():
    """OCR puts amounts a few points above their label (MCB 2019 p21 pilot)."""
    rows = [Row(21, "2,013,917 2,680,672", "", [2013917.0, 2680672.0], 100.0),
            Row(21, "Net interest income", "Net interest income", [], 104.0),
            Row(21, "Staff costs (3,337,225)", "Staff costs", [-3337225.0], 140.0)]
    out = merge_orphans(rows)
    assert out[0].label == "Net interest income" and out[0].numbers == [2013917.0, 2680672.0]
    assert len(out) == 2 and out[1].label == "Staff costs"


def test_merge_orphans_respects_gap():
    rows = [Row(1, "100 200", "", [100.0, 200.0], 100.0), Row(1, "Far label", "Far label", [], 130.0)]
    assert merge_orphans(rows)[0].label == ""


def test_ocr_routing_and_targets(root, tmp_path):
    from tzbank.extract.ocr import extraction_source, needs_ocr, ocr_target

    docs = root / "data" / "raw" / "documents"
    if not (docs / "MCB" / "2019.pdf").exists():
        pytest.skip("source PDFs are not committed")
    scanned, text = docs / "MCB" / "2019.pdf", docs / "MCB" / "2025.pdf"
    assert needs_ocr(scanned) and not needs_ocr(text)
    assert ocr_target(scanned, root) == root / "data" / "ocr" / "MCB" / "2019.pdf"
    assert extraction_source(text, root) == text  # text-layer PDFs keep the original


def test_ocr_values_need_arithmetic_gate(root):
    """On an OCR copy only II/IE/NII (identity-gated) pass by themselves; other lines are flagged."""
    from tzbank.extract.gated import candidates_for

    if not (root / "data" / "ocr" / "MCB" / "2019.pdf").exists():
        pytest.skip("OCR copy not present")
    c = candidates_for("B06", root / "data" / "raw" / "documents" / "MCB", root)
    o = c[c["ocr"] & (c["file"] == "MCB/2019.pdf")]
    ii = o[(o["variable"] == "Interest Income") & (o["fiscal_year"] == 2019)].iloc[0]
    assert ii["value"] == 2_936_631_000 and ii["gate_passed"] and ii["page"] == 21
    ie = o[(o["variable"] == "Interest Expense") & (o["fiscal_year"] == 2019)].iloc[0]
    assert ie["value"] == 922_714_000  # 547,277 + 375,437 (TZS '000)
    other = o[~o["gate"].str.startswith(("II-IE=NII", "gross - allowance"))]
    assert other["gate"].str.contains("OCR: manual verification").all()


def test_quarterly_rules():
    d, r = quarterly_rule(pd.Timestamp("2026-09-30"))
    assert d == pd.Timestamp("2026-10-30") and r.regulatory_basis == REG_2014_Q
    d, r = quarterly_rule(pd.Timestamp("2014-12-31"))
    assert d == pd.Timestamp("2015-02-14") and "2014 quarter" in r.flag  # ambiguous: longer deadline
    d, r = quarterly_rule(pd.Timestamp("2006-06-30"))
    assert d == pd.Timestamp("2006-08-14")


def test_annual_rule_hand_checked():
    fye = pd.Timestamp("2020-12-31")
    d, r = annual_rule(fye, pd.Timestamp("2021-03-19"))
    assert d == pd.Timestamp("2021-04-03") and r.regulatory_basis == REG_2014_A  # approval + 15 < FYE + 105
    d, _ = annual_rule(fye, pd.Timestamp("2021-04-10"))
    assert d == pd.Timestamp("2021-04-15")  # capped at FYE + 105
    d, r = annual_rule(fye, None)
    assert d == pd.Timestamp("2021-04-15") and r.name == "fye+105d"
    assert annual_rule(pd.Timestamp("2005-12-31"), None)[1].regulatory_basis == REG_2001_A
    assert annual_rule(pd.Timestamp("2010-12-31"), None)[1].regulatory_basis.startswith(ASSUMED)


def test_resolve_prefers_actual_and_never_uses_approval_as_publication():
    base = {"Document Type": "Annual report", "Period End": pd.Timestamp("2024-12-31"),
            "FS Board Approval Date": pd.Timestamp("2025-03-27")}
    est = resolve(pd.Series(base))
    assert est["publication_date_type"] == "estimated" and est["publication_date"] == pd.Timestamp("2025-04-11")
    act = resolve(pd.Series(base | {"Actual Publication Date": pd.Timestamp("2025-04-02")}))
    assert act["publication_date_type"] == "actual" and act["publication_date"] == pd.Timestamp("2025-04-02")


def test_bot_template_validation():
    good = pd.DataFrame([dict(zip(TEMPLATE_COLUMNS, ["S1", "CBR", "rates", "sector", None, "quarterly", "2025-12-31",
                                                     "0.0575", "ratio", "BoT MPC", "p1", "2025-10-02", "HIGH", ""],
                              strict=True))])
    assert validate_frame(good, "x.csv") == []
    bad = good.assign(value="5.75", source_document=None)
    issues = validate_frame(bad, "x.csv")
    assert any(i["critical"] and i["column"] == "source_document" for i in issues)
    assert any(i["column"] == "value" and "percentage" in i["issue"] for i in issues)
    assert validate_frame(good.drop(columns=["notes"]), "x.csv")[0]["critical"]


def test_auction_yield_template_and_coe_inputs():
    res = pd.DataFrame({"auction_number": [693, 692], "auction_date": pd.to_datetime(["2025-12-17", "2025-12-10"]),
                        "bond_title": ["13.00% 20-YEAR", "x"], "waytm": [0.120193, None], "status": "SUCCESS",
                        "weighted_average_price": [107.3346, None], "coupon_yield": [0.121117, None],
                        "file": ["bot/auctions/693_1.json", "bot/auctions/692_1.json"]})
    t = auction_yields_template(res)
    assert list(t.columns) == TEMPLATE_COLUMNS and len(t) == 1 and t.iloc[0]["value"] == 0.120193
    assert validate_frame(t, "t.csv") == []
    au = pd.DataFrame({"auction_number": [693, 688], "auction_date": pd.to_datetime(["2025-12-17", "2025-11-12"]),
                       "tenor_years": [20, 15], "coupon": [0.13, 0.1275]})
    c, used = coe_inputs_annual(au, res, 2011, 2025)
    old_rule = c[c["series"] == "alt_longest_tenor_nearest_year_end"].set_index("year")
    assert old_rule.loc[2025, "bond_number"] == "693"
    assert old_rule.loc[2011, "status"] == "needs BoT Monthly Economic Review"
    assert "auction_number" not in c.columns


def test_funding_checks_hand_checked():
    from tests.conftest import canon_row

    rows = []
    for y, dep, ied, ie in [(2023, 1000.0, 30.0, 50.0), (2024, 1200.0, 44.0, 40.0)]:
        pe = pd.Timestamp(f"{y}-12-31")
        rows += [canon_row(canonical_name="customer_deposits", value=dep, period_end=pe, source_row_id=f"d{y}"),
                 canon_row(canonical_name="interest_expense_deposits", value=ied, period_end=pe, source_row_id=f"e{y}"),
                 canon_row(canonical_name="interest_expense", value=ie, period_end=pe, source_row_id=f"i{y}"),
                 canon_row(canonical_name="deposits_current", value=dep * 0.5, period_end=pe, source_row_id=f"c{y}"),
                 canon_row(canonical_name="deposits_time", value=dep * 0.45, period_end=pe, source_row_id=f"t{y}")]
    df = pd.DataFrame(rows)
    df["fiscal_year"] = df["period_end"].dt.year
    c = funding_checks(df)
    cod = c[c["check"].str.startswith("cost")].iloc[0]
    assert cod["fiscal_year"] == 2024 and abs(cod["value"] - 44 / 1100) < 1e-12 and cod["status"] == "PASS"
    mix = c[c["check"].str.startswith("deposit mix")]
    assert (mix["status"] == "WARN").all()  # 95% only
    share = c[c["check"].str.startswith("deposit interest")].set_index("fiscal_year")["status"]
    assert share[2023] == "PASS" and share[2024] == "WARN"  # 44 > 40


def test_document_paths_are_relative(root):
    assert not Path("data/ocr").is_absolute()


def test_ie_note_reconciles_to_audited_total(root):
    """NMB FY2019 interest-expense note (p.177): lines sum exactly to the audited TZS 130,820m and 109,615m."""
    from tzbank.extract.ledger import ie_note_components

    pdf = root / "data" / "raw" / "documents" / "NMB" / "2019.pdf"
    if not pdf.exists():
        pytest.skip("source PDFs are not committed")
    cur = ie_note_components(pdf, 130_820e6, 0)
    assert cur["page"] == 177
    parts = {k: v[0] / 1e6 for k, v in cur["parts"].items()}
    assert parts["Interest Expense on Time Deposits"] == 52_777
    assert parts["Interest Expense on Bank Deposits"] == 24  # small amount next to a note-like position
    assert ie_note_components(pdf, 130_000e6, 0) is None  # 0.6% off: rejected


def test_restatement_tie_break_d21():
    from tzbank.clean.normalise import flag_restatements

    base = dict(bank_id="B03", entity="Bank solo", basis="BANK", period_type="FY",
                period_end=pd.Timestamp("2021-12-31"), canonical_name="interest_income", category=None,
                publication_date=pd.Timestamp("2022-04-15"), notes=None, fiscal_year=2021)
    df = pd.DataFrame([dict(base, doc_id="DCB-AR-FY2022", source_row_id="a", value=2.0),
                       dict(base, doc_id="DCB-AR-FY2021", source_row_id="b", value=1.0)])
    out = flag_restatements(df).set_index("doc_id")["is_restated"]
    assert out["DCB-AR-FY2021"] == False and out["DCB-AR-FY2022"] == True  # noqa: E712


def test_manual_entries_validation(root, tmp_path, monkeypatch):
    """D29: valid rows pass in full TZS; a row without page or checker is critical and not passed on."""
    import tzbank.ingest.manual as m

    f = tmp_path / "manual_values.csv"
    rows = pd.DataFrame([
        dict(bank_id="B06", fiscal_year=2023, period_type="FY", variable="Interest Income", value="10512",
             unit="TZS million", basis="BANK", file="MCB/2023.pdf", page="1", as_reported_label="Mapato ya Riba",
             checked_by="owner", checked_on="2026-11-07", confidence="MEDIUM", notes=""),
        dict(bank_id="B06", fiscal_year=2023, period_type="FY", variable="Interest Expense", value="4919",
             unit="TZS million", basis="BANK", file="MCB/2023.pdf", page=None, as_reported_label="Gharama za Riba",
             checked_by=None, checked_on=None, confidence="MEDIUM", notes=""),
    ], columns=m.MANUAL_COLUMNS)
    rows.to_csv(f, index=False)
    monkeypatch.setattr(m, "manual_file", lambda root=None: f)
    valid, issues = m.read_manual(root)
    assert len(valid) == 1 and valid.iloc[0]["value_full"] == 10_512e6
    assert valid.iloc[0]["extraction_method"] == "manual"
    assert issues["critical"].any() and set(issues["row"]) == {3}


def test_failed_auctions_excluded_from_coe(root):
    r = pd.read_csv(root / "data" / "interim" / "bond_auction_results.csv")
    failed = r[r["status"] == "FAILURE"]
    assert set(failed["auction_number"]) == {337, 382, 383, 397, 404, 502}
    assert failed["waytm"].isna().all()
    used = pd.read_csv(root / "data" / "processed" / "coe_long_bond_auctions_used.csv")
    assert not (set(zip(used["bond_number"], used["auction_date"], strict=False))
                & set(zip(failed["auction_number"], failed["auction_date"].astype(str), strict=False)))
