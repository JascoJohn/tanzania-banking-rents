# Publication date rules

Implemented in `tzbank.pit.pubdates`; the regulation texts are
held in `data/raw/documents/regulation/` (git-ignored, hashed in `data/raw/MANIFEST.csv`).

## Order of evidence

1. **Actual.** A dated newspaper publication notice or press release, a DSE announcement date, a dated web posting,
   or a dated AGM notice inside the report (evidence the report was public by then). Recorded in the Sources
   column `Actual Publication Date` with the evidence in `How Publication Date Established` / `Date Evidence`.
2. **Estimated.** The regulatory deadline below. Never earlier than the rule without evidence.

Board approval of the financial statements (`FS Board Approval Date`) and the auditor's report date
(`Auditor Report Date`) are recorded separately and are not publication dates. The 15-day rule counts from the
board's approval of the financial statements, not of a later integrated report. Market, statistical and
dataset sources (DSE, BoT market data, NBS, Damodaran, US Treasury) are dated by their own posting or release date
(actual).

## Regulations

| Regulation | Provision | Commencement | Periods applied | Deadline | Status |
| --- | --- | --- | --- | --- | --- |
| Banking and Financial Institutions (Financial Statements) Regulations, 2001, GN 103 of 2001 | reg. 6 (quarterly) | 1 Sep 2000 (TanzLII metadata) | periods ending before 12 Dec 2008 | quarter-end + 45 days | verified from text |
| same | reg. 8 (annual) | same | FY up to 2007 | earlier of FS board approval + 15 days and FYE + 105 days; FYE + 105 days if no approval date | verified from text |
| Banking and Financial Institutions (Publication of Financial Statements) Regulations, 2008, GN 368 of 2008 | text not held | 12 Dec 2008 (Gazette No. 50, notice 632, Supplement No. 41; inferred from the gazette date) | periods ending 12 Dec 2008 - 21 Aug 2014 (FY2008-FY2013; quarters to Jun 2014) | the 2001 deadlines are assumed | `assumed_2001_deadlines_gn368_text_not_held` |
| Banking and Financial Institutions (Disclosures) Regulations, 2014, GN 289 of 2014 | reg. 7(3) (quarterly) | published in the Gazette 22 Aug 2014; no commencement clause, so taken to operate from publication (inference) | quarters ending from 31 Mar 2015 | quarter-end + 30 days | verified from text |
| same | reg. 8(3) (annual) | same | FY2014 onward | earlier of FS board approval + 15 days and FYE + 105 days | verified from text |

**GN 289 of 2014 commencement (an inference).** The Regulations have no commencement clause; they are taken to
operate from their gazette date, 22 Aug 2014. This is an inference from the general rule that subsidiary legislation
takes effect on publication unless it says otherwise; it is not stated in the text. Quarters ending in 2014 (Q3 and
Q4, after gazettement) keep the longer 45-day deadline (conservative) and are flagged; 30 days applies from Q1 2015.

**Three periods.** GN 289 of 2014 revokes GN 368 of 2008 (reg. 20), not the 2001 Regulations. Government
Gazette No. 50 of 12 Dec 2008 (notice No. 632; `documents/regulation/tz-government-gazette-dated-2008-12-12-no-50.pdf`,
Sources REG-GAZ-2008-50) records that the BFI (Publication of Financial Statements) Regulations were published as
GN 368 of 2008 in Subsidiary Legislation Supplement No. 41 of 12 Dec 2008. Commencement on 12 Dec 2008 is an
inference from the gazette date. The rule is applied by period end:

1. before 12 Dec 2008: the 2001 Regulations (verified);
2. 12 Dec 2008 - 21 Aug 2014: GN 368's text is not held; the 2001 deadlines (45 days quarterly; annual 15 days after
   FS approval, capped at 105 days) are assumed and flagged `assumed_2001_deadlines_gn368_text_not_held`. These are
   the longer, conservative deadlines;
3. from 22 Aug 2014: GN 289 of 2014 (verified; quarters ending in 2014 keep 45 days).

**The flag cannot change eligibility.** Every FY2008-FY2013 estimated date falls in 2009-2014, years before both
valuation dates (31 Dec 2025 and 31 Oct 2026). Whatever deadline GN 368 of 2008 set, a document for those years is
eligible at both cut-offs. The GN 368 collection task is closed as "text unavailable; resolved by assumption".

The file placed on 2026-11-06 as GN 368 (`tz-act-gn-2009-368-publication-document.pdf`) is GN 368 of **2009**, the
Tanzania Court of Appeal Rules, 2009. It is unrelated and is not used.

DSE and CMSA listing-rule deadlines are not applied. None was found in the documents examined.

## Provisions (short extracts)

- GN 103 of 2001, reg. 6: statements for each quarter "shall be published within forty-five days after the end of
  the quarter".
- GN 103 of 2001, reg. 8: annual audited statements "within fifteen days after approval of the board of directors
  ... but not later than 105 days after end of the financial year".
- GN 289 of 2014, reg. 7(3): quarterly statements "shall be published within thirty days after the end of the
  quarter".
- GN 289 of 2014, reg. 8(3): annual audited statements "within fifteen days after approval of the board of
  directors ... but not later than one hundred and five days after the end of the financial year".

## Phase 1 rows

A Phase 1 row takes its document's date when the Phase1 Source Map maps its source label to a Doc ID. Rows with
no mappable document take FYE + 105 days for their fiscal year, flagged `pubdate_basis = phase1_unmapped`.

## Recording

`data/processed/sources.csv` carries for every document: `publication_date`, `publication_date_type`
(actual / estimated / lower_bound / missing), `rule_applied`, `period` (Period End), `regulatory_basis`,
`fs_board_approval_date` (FS Board Approval Date), `auditor_report_date` (Auditor Report Date), and the
value before these rules in `Legacy Publication Date (before the publication-date rules)`.

## Lower bounds

A date known only as a lower bound (for example a PDF's last-modified date: the file cannot have been posted before
it) is recorded in Sources as `Lower-Bound Publication Date` and typed `lower_bound`. `tzbank.pit.eligible` and the
workbook eligibility columns treat it as ineligible at every valuation date until an actual or estimated date
replaces it; descriptive history may still use it. The four BoT NPS annual reports (BOT-NPSAR-2022..2025) are dated
this way; confirming their posting dates is an owner collection task (`reports/inclusion_nps_collection_tasks.csv`).
The inclusion reports' file-name document IDs remain estimated posting dates.
