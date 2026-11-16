"""Prompt 02c: inclusion and NPS panels (D30-D33), caption inventory parser, PDF cross-check helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.extract.captions import _refs, parse_caption
from tzbank.extract.inclusion_crosscheck import _join_thousands, check_obs, equals, label_words, numbers
from tzbank.ingest.inclusion import latest_vintage, panel, read_long, read_wide_cached, vintage_check, workbook_path
from tzbank.ingest.nps import canonical_name as nps_name
from tzbank.ingest.nps import inferred_source_report


@pytest.fixture(scope="module")
def long(root):
    return read_long(workbook_path(root))


# --- D31: latest-vintage rule -------------------------------------------------------------------------------------

def _obs(sid, year, report, value, include=1, oid=None):
    return {"Obs_ID": oid or f"{sid}{year}{report}", "Series_ID": sid, "Domain": "d", "Indicator": "i",
            "Breakdown": None, "Year": year, "Std_Value": value, "Std_Unit": "count", "Source_Report": report,
            "Source_Table_or_Section": "Table 1", "PDF_Page": 1, "Include_in_Panel": include}


def test_latest_vintage_hand_built():
    df = pd.DataFrame([_obs("S", 2022, 2023, 10.0), _obs("S", 2022, 2024, 11.0), _obs("S", 2022, 2025, 99.0, 0),
                       _obs("S", 2023, 2023, 5.0)])
    flag = latest_vintage(df)
    assert flag.tolist() == [0, 1, 0, 1]  # newest included report wins; an excluded newer row never does
    df["Is_Latest_Vintage"] = flag
    p = panel(df).set_index("Year")
    assert p.loc[2022, "value"] == 11.0 and p.loc[2022, "source_report"] == 2024
    assert p.loc[2023, "value"] == 5.0


def test_vintage_rule_reproduces_panel_wide(root, long):
    vc = vintage_check(panel(long), read_wide_cached(workbook_path(root)))
    assert (vc["status"] == "MATCH").all(), vc[vc["status"] != "MATCH"].head()
    assert len(vc) > 1000


def test_std_value_recomputed_not_read_from_cache(long):
    ok = long["Reported_Value"].notna() & long["Factor_to_Std"].notna()
    assert (long.loc[ok, "Std_Value"] == long.loc[ok, "Reported_Value"] * long.loc[ok, "Factor_to_Std"]).all()


# --- D32: definition breaks never merged -------------------------------------------------------------------------

def test_definition_break_series_stay_separate(long):
    p = panel(long)
    suffixed = p[p["basis_suffix"].notna()]
    assert len(suffixed) > 0
    assert (suffixed["definition_break_note"].str.contains("D32")).all()
    # each suffixed series exists on its own and is never folded into its stem
    stems = suffixed["Series_ID"].str.replace(r"_(A|B|R25)$", "", regex=True)
    assert not set(suffixed["Series_ID"]) & set(stems)
    assert p.groupby(["Series_ID", "Year"]).size().max() == 1


def test_processed_panels_carry_evidence_and_crosswalk(root):
    cw = pd.read_csv(path("crosswalk", root), dtype=str).fillna("")
    for f, col in (("inclusion_panel.csv", "canonical_name"), ("nps_panel.csv", "canonical_name")):
        df = pd.read_csv(path("processed", root) / f)
        assert df["evidence_class"].notna().all()
        assert set(df[col]) <= set(cw["canonical_name"]), f
    nps = pd.read_csv(path("processed", root) / "nps_panel.csv")
    assert nps.groupby("canonical_name")["series_id"].nunique().max() == 1  # slug is one-to-one


# --- NPS source-report inference -----------------------------------------------------------------------------------

def test_nps_source_report_inference():
    assert inferred_source_report("Source: 2024 NPS Annual Report", 2021) == (2024, "stated: single report")
    note = "2023 report (2019-2023); 2025 report (2021-2025)"
    assert inferred_source_report(note, 2020)[0] == 2023
    assert inferred_source_report(note, 2022)[0] == 2025
    assert inferred_source_report("Annual Reports 2023-2025", 2019)[0] == 2023
    assert inferred_source_report("no reports named", 2020)[0] is None
    assert nps_name("12_X|T|% number") != nps_name("12_X|T|Number")


# --- caption inventory parser ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("Chart 2.7: Number of Loans Accounts Banks and SACCOs", ("chart", "2.7", "Number of Loans Accounts Banks and SACCOs")),
    ("Table 2. 1: Saving Accounts", ("table", "2.1", "Saving Accounts")),
    ("Appendix 4: Access Points", ("appendix", "4", "Access Points")),
    ("Table E1: Card Transactions", ("table", "E1", "Card Transactions")),
])
def test_parse_caption(line, expected):
    kind, num, title = parse_caption(line)
    assert (kind, num) == expected[:2] and title.startswith(expected[2])


@pytest.mark.parametrize("line", [
    "Table 2.1: Total Number of Financial Access Point ........ 19",  # table-of-contents leader dots
    "Table 2.1 shows that access improved",  # a sentence, not a caption
    "The general growth in uptake",
])
def test_parse_caption_rejects_non_captions(line):
    assert parse_caption(line) is None


def test_reference_ranges_expand():
    assert set(_refs("Charts 3.1-3.2")) >= {("chart", "3.1"), ("chart", "3.2")}
    assert ("table", "2.1") in _refs("Table 2.1")


# --- PDF cross-check helpers ------------------------------------------------------------------------------------

def test_printed_number_equality():
    assert equals(609956.0, numbers("609,956")[0])
    assert equals(0.81, numbers("81")[0])  # ratio printed as a percentage
    assert equals(-2.92, numbers("(2.92)")[0])  # negatives printed in brackets
    assert not equals(45792.0, numbers("45,785")[0])
    assert equals(5.51, numbers("5.51")[0]) and not equals(5.5, numbers("5.51")[0])


def test_thousands_and_broken_decimals():
    vals = [v for v, _ in numbers(_join_thousands("Branches 957 969 990 981 1,011"))]
    assert {957.0, 969.0, 990.0, 1011.0} <= set(vals)
    assert 1215033.0 in [v for v, _ in numbers(_join_thousands("Total 1 215 033"))]
    assert 277.62 in [v for v, _ in numbers(_join_thousands("scheme 230.12 277. 62 384.11"))]
    assert 2006.0 in [v for v, _ in numbers(_join_thousands("SIDO NEDF 2,006. 5.51"))]


def test_check_obs_statuses():
    pages = {5: ["Table 2.2: Access points", "Number of Bank branches 957 969 990", "Total 29,371 41,431"],
             6: ["Number of Bank agents 28,358"]}
    lab = label_words("Number of bank branches")
    assert check_obs(pages, 5, 969.0, lab)[0] == "match"
    assert check_obs(pages, 5, 41431.0, label_words("Total banking access points"), "Total banking access points")[0] \
        == "match"
    assert check_obs(pages, 5, 28358.0, lab)[0] == "value on adjacent page"
    assert check_obs(pages, 5, 12345.0, lab)[0] == "NOT FOUND"


def test_parse_caption_number_without_dot():
    assert parse_caption("Table 2 1: Saving Accounts and Amounts in Banks and SACCOs")[:2] == ("table", "2.1")
