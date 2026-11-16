"""Prompt 20 item 1 (D88): merge the analyst register.

Inputs: the owner's seed ``data/raw/analyst_reports/analyst_register_seed.csv`` (17 documents), the prompt 18 register
(nine views) and the field transcription ``data/raw/manual_entries/analyst_notes_extracted.csv`` (page-referenced).
Where they disagree the PDF decides; every disagreement is written to
``data/raw/manual_entries/analyst_register_merge.csv``. The four Solomon views flagged in prompt 18 are confirmed by
the owner (prompt 20). Tanzania Securities (ids 1-2): PDFs not in the folder, owner values kept, unverified.

Run once: ``python -m tzbank.extract.session_20a``; then ``python -m tzbank manifest`` and the pipeline.
"""

from __future__ import annotations

import pandas as pd

from tzbank.config import find_root

# data_as_of: the date of the market data (price) printed in the note, when it differs from the note's date
COLS = ["id", "broker", "analyst_names", "date", "date_label", "date_basis", "data_as_of", "bank", "entity", "doc_type",
        "rating", "fair_value_reported", "per_share_basis_reported", "fair_value_post_split", "price_reported",
        "price_post_split", "method", "method_weights", "coe", "coe_note", "terminal_g", "g_basis", "horizon_months",
        "explicit_forecast_years", "forecast_notes", "split_adjusted", "file", "document_status", "source_note",
        "evidence_class"]
PRE = "pre-split (NMB 1:10 split effective 24 Aug 2026)"

ROWS = [
    (1, "Tanzania Securities", "not disclosed", "2023-07-11", "11 Jul 2023", "valuation date printed (p1)", "",
     "NMB", "NMB", "valuation note", "Buy", 4019, PRE, 401.9, 3460, 346.0, "DDM + market P/B (p11)",
     "DDM 60%, market comparable 40% (p11)", 0.1847,
     "CAPM 20.5% less a 2% liquidity factor = 18.47% used (p9); rf 9.85% (p8 '5-year bond, last auction', p9 table "
     "'10-year yields'); weekly beta 0.69, adjusted 0.80 vs TSI (p9); ERP 13.35% Damodaran Jul 2023 (p8-9)", 0.055,
     "real (average GDP growth, p9)", "", "", "fees growth 15%; terminal multiple 1.3x book at 31 Dec 2027 (p10)",
     "yes (values / 10)", "TSL_NMB_2023-07-11.pdf", "held (verified prompt 21)", "owner seed; PDF verified",
     "Observed"),
    (2, "Tanzania Securities", "not disclosed", "2024-06-03", "3 Jun 2024", "valuation date printed (p2)",
     "2024-05-31", "CRDB", "CRDB", "valuation note", "Buy", 588, "as reported", 588.0, 510, 510.0,
     "5-year DDM + P/B (1.0x)", "not stated", 0.2065,
     "CAPM 20.65% (p7): rf 12.03% 10-year bond; adjusted beta 0.68 (unadjusted 0.52) vs BI index; ERP 12.64% Damodaran",
     0.055, "real (sustainable GDP growth, p7; p8 also cites 'about 7.5%')", "12", "5",
     "LDR 85%; comparable P/B 1.0x = average of 103 African banks; p8 says 'overpriced by 15.3%' where it means "
     "underpriced", "not needed", "TSL_CRDB_2024-06-03.pdf",
     "held (verified prompt 21; received as 'TSL NMB 2024.pdf', mislabelled)", "owner seed; PDF verified",
     "Observed"),
    (3, "Solomon", "not disclosed", "2024-06-30", "June 2024 (AGM update)", "month printed", "", "NMB", "NMB",
     "AGM update", "none", None, "", None, None, None, "commentary", "", None, "not disclosed", None, "", "", "", "",
     "", "SOLOMON_NMB_2024-06_AGM.pdf", "held", "owner seed; PDF", "Observed"),
    (4, "Solomon", "Emmanuel Matunda, ACSI", "2024-09-27", "Sep 2024 (initiation)", "price date printed (27 Sep 2024)",
     "", "other", "MKCB", "initiation", "Strong Buy", 1019.70, "as reported", 1019.70, 560, 560.0,
     "Excess return + DDM + relative (P/E, P/B)", "1/3 each", 0.2163,
     "rf 11.70% (interpolated 5y); unlevered beta 0.78 (Damodaran EM regional banks) used in CAPM; ERP 12.73% = MKCB "
     "ROE 24.43% minus rf (non-standard); Damodaran TZ premium 12.64% considered and rejected", 0.0594,
     "real (sustainable growth tied to GDP projections)", "", "5 (2024F-2028F)", "", "not needed",
     "SOLOMON_MKCB_2024-09.pdf", "held", "owner seed; PDF (price and g added)", "Observed"),
    (5, "iTrust", "Frank Bunuma; Neema Ndunguru", "2025-02-21", "21 Feb 2025", "printed", "", "CRDB", "CRDB",
     "results tear sheet", "none", None, "", None, 730, 730.0, "data only", "", None, "not disclosed", None, "", "",
     "", "NPL coverage 45.0% FY2024; CIR 45.7%; ROAE 27.7% (unaudited; restated in the Aug 2025 sheet)", "not needed",
     "ITRUST_CRDB_2025-02-21.pdf", "held", "owner seed; PDF", "Observed"),
    (6, "iTrust", "Frank Bunuma; Neema Ndunguru", "2025-08-05", "5 Aug 2025", "printed", "", "CRDB", "CRDB",
     "results tear sheet", "none", None, "", None, 1210, 1210.0, "data only", "", None, "not disclosed", None, "", "",
     "", "52-week high printed (1,180) below the price (1,210)", "not needed", "ITRUST_CRDB_2025-08-05.pdf", "held",
     "owner seed; PDF", "Observed"),
    (7, "Kadoo", "Abraham Abubakar Ukhotya; Erick Sindatuma", "2025-11-06", "Q3 2025 review (undated)",
     "undated; uses Q3 results published 28 Oct 2025; first DSE close at the printed price 1,170 after that is 6 Nov 2025",
     "", "CRDB", "CRDB", "results review", "Buy", 1716.56, "as reported", 1716.56, 1170, 1170.0,
     "Two-stage DDM", "", None, "not printed (present values imply about 15.15%, a derivation, not used)", 0.08,
     "nominal assumed (stage-2 stable growth)", "", "5", "stage-1 growth 20%", "not needed", "KADOO_CRDB_2025-Q3.pdf",
     "held", "owner seed; PDF (date fixed from results date and DSE price)", "Observed"),
    (8, "iTrust", "Frank Bunuma; Neema Ndunguru", "2025-11-20", "20 Nov 2025", "printed", "", "CRDB", "CRDB",
     "results tear sheet", "none", None, "", None, 1160, 1160.0, "data only", "", None, "not disclosed", None, "", "",
     "", "", "not needed", "ITRUST_CRDB_2025-11-20.pdf", "held", "owner seed; PDF", "Observed"),
    (9, "Kadoo", "Abraham Ukhotya; Erick Sindatuma", "2026-01-31", "31 Jan 2026", "printed", "", "CRDB", "CRDB",
     "equity note", "Hold (accumulate on dips)", 2283.27, "as reported (blended; text prints 2,281.77)", 2283.27,
     None, None, "DDM + forward P/E (peer-based)", "DDM 2,641.69 at 50%", None, "not disclosed", None, "", "", "",
     "forward P/E peer median 4.8x; P/E 7.4x, P/B 1.92x; ROE 29.3% p1 vs 26.0% elsewhere; loan growth 36.6% text vs "
     "32.6% table", "not needed", "KADOO_CRDB_2026-01-31.pdf", "held", "owner seed; PDF (table value)", "Observed"),
    (10, "Solomon", "Emmanuel Matunda", "2026-02-02", "Q4 2025 results (cover: Dec 2025)",
     "price date printed (2 Feb 2026); cover shows the reporting quarter", "", "other", "DCB", "earnings highlights",
     "none", None, "", None, 315, 315.0, "data only (multiples)", "", None, "not disclosed", None, "", "", "",
     "P/B 1.76x; P/E 553.6x printed", "not needed", "SOLOMON_DCB_2026-Q1.pdf", "held", "owner seed; PDF (date)",
     "Observed"),
    (11, "Solomon", "Emmanuel Matunda", "2026-04-08", "Monthly Rally, March 2026",
     "data as of 31 Mar 2026; cites moves as of 8 Apr 2026, so published on or after 8 Apr", "2026-03-31", "CRDB",
     "CRDB", "Monthly Rally", "Buy", 2934.63, "as reported", 2934.63, 2700, 2700.0, "not stated", "", 0.1497,
     "14.97% (stated; inputs not disclosed)", None, "", "", "2026-2030", "earnings CAGR 2026-30 24.77%", "not needed",
     "SOLOMON_MONTHLY_2026-03-31.pdf", "held", "owner confirmed (prompt 20); PDF (publication date)", "Observed"),
    (12, "Solomon", "Emmanuel Matunda", "2026-04-08", "Monthly Rally, March 2026",
     "data as of 31 Mar 2026; cites moves as of 8 Apr 2026, so published on or after 8 Apr", "2026-03-31", "NMB", "NMB",
     "Monthly Rally", "Buy", 16347.02, PRE, 1634.702, 13980, 1398.0, "not stated", "", 0.1497,
     "14.97% (stated; inputs not disclosed)", None, "", "", "2026-2030", "earnings CAGR 2026-30 19.90%",
     "yes (values / 10)", "SOLOMON_MONTHLY_2026-03-31.pdf", "held", "owner confirmed (prompt 20); PDF (publication date)",
     "Observed"),
    (13, "Solomon", "Emmanuel Matunda", "2026-05-06", "Q1 2026 results", "price date printed (6 May 2026)", "", "other",
     "MKCB", "earnings highlights", "none", None, "", None, 4600, 4600.0, "data only", "", None, "not disclosed", None,
     "", "", "", "P/B 2.06x; 'TZS 413 billion written off' against total assets of TZS 333bn (unit error in the note)",
     "not needed", "SOLOMON_MKCB_2026-04.pdf", "held", "owner seed; PDF (date)", "Observed"),
    (14, "Solomon", "Emmanuel Matunda", "2026-04-30", "30 Apr 2026 (Q1 note)", "printed", "", "CRDB", "CRDB",
     "earnings highlights", "Sell", 2756.40, "as reported", 2756.40, 2830, 2830.0, "not stated", "", None,
     "not disclosed", None, "", "", "", "target cut 6.07% from 2,934.63 (negative jaws)", "not needed",
     "SOLOMON_CRDB_2026-04-30.pdf", "held", "owner confirmed (prompt 20)", "Observed"),
    (15, "Solomon", "Emmanuel Matunda", "2026-04-30", "30 Apr 2026 (Q1 note)", "printed", "", "NMB", "NMB",
     "earnings highlights", "Hold", 13544.03, PRE, 1354.403, 13510, 1351.0, "not stated", "", None, "not disclosed",
     None, "", "", "", "target cut 17.15% from 16,347.02 (weaker revenue; funding costs)", "yes (values / 10)",
     "SOLOMON_NMB_2026-04-30.pdf", "held", "owner confirmed (prompt 20)", "Observed"),
    (16, "Kadoo", "Abraham Ukhotya; Erick Sindatuma", "2026-06-25", "25 Jun 2026", "printed", "", "NMB", "NMB",
     "equity note", "Hold/Buy", 16100, PRE, 1610.0, 16090, 1609.0, "RIM", "", None, "not disclosed", None, "", "", "",
     "equity value TZS 8.0tn (8,007,308m / 500m shares = 16,015, below the printed 16,100); PV terminal 3.2tn; P/E "
     "10.41x on annualised Q1 2026 EPS (unlabelled); P/B 2.43x",
     "yes (values / 10; the note's own restatement is inconsistent: 16,090 -> '1,690', 16,015 -> 1,601)",
     "KADOO_NMB_2026-06-25.pdf", "held (same file as documents/benchmarks/Kadoo_NMB_Equity_Research_Note_2026-06.pdf)",
     "owner seed; PDF (price added)", "Observed"),
    (17, "Vertex", "not disclosed", "2026-04-02", "2025 capital market review",
     "no date printed; mentions an ETF listing of 28 Jan 2026; PDF created 2 Apr 2026", "2025-12-31", "market",
     "DSE", "capital market review", "none", None, "", None, None, None, "market review", "", None, "not disclosed",
     None, "", "", "", "NMB market cap TZS 4,205bn (largest); total market cap TZS 23,995.45bn (+34.3%); turnover TZS "
     "663.75bn (+190.31%) (OCR checked against the page image)", "not needed", "VERTEX_MARKET_2025.pdf",
     "held (scanned; OCR copy in data/ocr/analyst_reports)", "owner seed; PDF (OCR)", "Observed"),
]

DISAGREEMENTS = [
    (4, "price_in_note", "none", "", "TZS 560 at 27 Sep 2024 (p2, p33)", "PDF"),
    (4, "terminal_g", "SGR", "", "5.94%, real, tied to GDP projections (p32)", "PDF"),
    (7, "date", "2025-Q4 (undated)", "2025-09-30 (plotted at the quarter end)",
     "undated; uses Q3 results published 28 Oct 2025; price 1,170 = DSE close on 6, 7, 10-12 Nov and 8 Dec 2025",
     "PDF + DSE: 2025-11-06 (earliest date consistent with both); the prompt 18 date was before the results were public"),
    (9, "target_or_fair_value", "2283.27", "2283 (about)", "table 2,283.27; text 2,281.77 (p6)", "PDF table: 2,283.27"),
    (9, "price_in_note", "none", "none", "not stated; P/E 7.4 x EPS 277 = 2,050 = DSE close 30 Jan 2026", "DSE price"),
    (11, "date", "2026-03-31", "2026-03-31", "data as of 31 Mar 2026; cites moves as of 8 Apr 2026 (p14)",
     "PDF: dated 2026-04-08 (publication), data_as_of 2026-03-31"),
    (12, "date", "2026-03-31", "2026-03-31", "as id 11", "PDF: dated 2026-04-08, data_as_of 2026-03-31"),
    (10, "date", "2026-Q1", "", "price as of 2 Feb 2026; cover 'Dec 2025' is the reporting quarter", "PDF: 2026-02-02"),
    (13, "date", "2026-04", "", "price as of 6 May 2026 (p3)", "PDF: 2026-05-06"),
    (16, "price_in_note", "none", "none", "TZS 16,090 pre-split (p8) = DSE close 25 Jun 2026", "PDF"),
    (16, "target_or_fair_value", "16100", "16100", "printed 16,100; equity value 8,007,308m / 500m = 16,015; the "
     "note's split text uses 16,015", "printed value kept (16,100); inconsistency listed"),
    (16, "rating", "Hold/Buy", "Hold/Buy", "'HOLD/BUY' p1, 'BUY/HOLD' p7", "p1 kept"),
    (17, "date", "2025 (annual)", "", "no date printed; ETF listing 28 Jan 2026 mentioned; PDF metadata 2 Apr 2026",
     "2026-04-02 (publication proxy), data_as_of 2025-12-31"),
    (1, "method_weights", "not stated", "", "DDM 60%, market comparable 40% (p11)", "PDF"),
    (1, "rf_tenor", "5y bond (last auction)", "", "p8 says 5-year, p9 table says 10-year (9.85%)", "both recorded"),
    (2, "rf_rate", "not in seed", "", "12.03%, 10-year bond (p7)", "PDF"),
    (2, "data_as_of", "", "", "price as at 31 May 2024 (p7)", "PDF: data_as_of 2024-05-31"),
]


def run() -> dict[str, int]:
    root = find_root()
    reg = pd.DataFrame([dict(zip(COLS, r, strict=True)) for r in ROWS])
    reg.to_csv(root / "data/raw/analyst_reports/register.csv", index=False)
    pd.DataFrame(DISAGREEMENTS, columns=["id", "field", "seed", "prior_register", "pdf", "resolution"]).to_csv(
        root / "data/raw/manual_entries/analyst_register_merge.csv", index=False)
    return {"register_rows": len(reg), "disagreements": len(DISAGREEMENTS)}


if __name__ == "__main__":
    print(run())
