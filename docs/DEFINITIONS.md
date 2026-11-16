# Definitions: Phase 1 vs audited statements

Diagnosis of the Phase 1 vs inputs-workbook differences found when the data layer was built
(`reports/validation/phase1_vs_inputs_conflicts.csv`), and the definition used by the processed panel.

## Inputs-workbook definitions (used for all seven banks where extracted)

| Line | Definition |
| --- | --- |
| Non-interest income | Net fee and commission income + foreign exchange income + income from securities trading + other operating income (total operating income less net interest income). Fee and commission **expense** is netted against fee income. |
| Operating expenses | Employee benefits + depreciation and amortisation + other operating expenses. Excludes impairment charges and fee and commission expense. |
| Customer deposits | Deposits from customers on the statement of financial position. Excludes deposits from banks. |

## CRDB (B01)

Evidence: CRDB Group and Bank Annual Report FY2019, statement of profit or loss (PDF p.135) and statement of
financial position (p.137).

1. **Operating expenses (Phase 1 higher by 3-12%).** Phase 1 adds fee and commission expense to operating
   expenses. FY2018: Phase 1 448,313 vs audited 428,880, difference 19,433 = fee and commission expense FY2018
   (19,433). FY2019: 507,530 vs 481,482, difference 26,048 = fee and commission expense FY2019 (26,048) (TZS m).
2. **Non-interest income (Phase 1 higher by 6-27%).** Phase 1 uses gross fee and commission income. FY2018: audited
   non-interest income 200,476 + fee expense 19,433 = 219,909 vs Phase 1 219,255 (0.3% apart). In other years the
   Phase 1 figure (taken from secondary aggregators: MarketScreener, stockanalysis) exceeds even the gross-fee total
   (FY2019: gross-fee total 229,629 vs Phase 1 251,523). The aggregator figures cannot be reconciled line by line;
   the audited net-fee definition is used.
3. **Customer deposits (Phase 1 11% higher in FY2018 and FY2021).** Not a definition difference: a one-year column
   shift. Phase 1 FY2018 = 5,202,247, which is the **FY2019** balance in AR 2019 (current-year column); the FY2018
   balance in the same statement is 4,687,174 (= inputs workbook). Deposits from banks (FY2018: 3,994) are not the
   cause. FY2021 (stockanalysis) shows the same pattern.

Resolution: precedence already replaces these Phase 1 values with the audited ones in
`data/processed`. No Phase 1 value was edited.

## B03-B07

Each line was re-extracted from the banks' own reports for every held year (`data/extraction/ledger.csv`) and
compared with Phase 1 (`reports/validation/phase1_vs_inputs_conflicts.csv`, now covering all seven banks). The
results for the earliest, middle and latest held year of each bank are in the section below, generated from the
ledger.

**Finding: Phase 1 uses the reports' own definitions for B03-B07; no line needed full re-extraction.**

For each bank, the extractor's own-year value of non-interest income ("Total non-interest income"), operating
expenses ("Total operating expenses") and customer deposits (statement of financial position) was compared with
Phase 1 for every year it located (`data/extraction/candidates_all.csv`, columns `phase1_value`,
`phase1_rel_diff`). Earliest, middle and latest years are covered for each bank where the line was located.

| Bank | Line | Years compared | Within 1% | Median abs. difference |
| --- | --- | --- | --- | --- |
| B03 DCB | Customer deposits | 8 | 7 | 0.0% |
| B03 DCB | Non-interest income | 7 | 6 | 0.0% |
| B03 DCB | Operating expenses | 6 | 5 | 0.0% |
| B04 MBP | Operating expenses | 7 | 6 | 0.0% |
| B05 MKCB | Customer deposits / Non-interest income / Operating expenses | 3 each | 3 each | 0.0% |
| B06 MCB | Customer deposits | 6 | 5 | 0.0% |
| B06 MCB | Operating expenses | 8 | 8 | 0.0% |
| B07 MUCOBA | Customer deposits | 6 | 5 | 0.0% |
| B07 MUCOBA | Non-interest income | 6 | 5 | 0.0% |

The exceptions are not definition differences:

- DCB FY2021 non-interest income (report 7,180,753 vs Phase 1 6,890,197, TZS '000) and operating expenses
  (18,447,224 vs 18,724,544): Phase 1 holds the **restated** FY2021 comparative from the FY2022 report (p.66), not
  the original. Same pattern for Maendeleo FY2021 operating expenses (Phase 1 = FY2022 report comparative).
- DCB FY2021 deposits (p.77), MCB FY2016 deposits (p.54) and MUCOBA FY2022 lines: extractor mis-picks (a maturity
  table, a page without a unit line, a quarterly publication with misaligned columns). These candidates failed the
  gate and were not written.

Where the extractor did not locate a line for a bank-year, Phase 1 remains the source (the workbook replaces Phase 1 only
where the workbook holds a value).

## MCB (B06) ratios above 1.5

All verified from the reports (genuine, not percentage typos; `reports/validation/unit_flags.csv`). The FY2018 ratios were verified from the OCR copy of the scanned FY2018 report.

| Year | Ratio | Phase 1 | Evidence |
| --- | --- | --- | --- |
| 2015 | Tier 1 and total capital ratio | 4.12 | MCB AR FY2016 capital note (PDF p.64): Tier 1 29,569,412 / RWA 7,172,967 (TZS '000) = 4.12. A newly licensed bank with little lending. |
| 2016 | Tier 1 and total capital ratio | 3.21 | Same note: 25,509,043 / 7,957,470 = 3.21. |
| 2017-2020 | Cost-to-income | 2.25-3.01 | Operating expenses above total operating income in loss-making years, e.g. FY2019 (OCR of the scanned AR, p.21, checked against the page image): operating expenses 7,360,013 / net operating income 2,547,639 = 2.89 (Phase 1 2.8890). |
| 2018 | NPL coverage | 3.02 | Verified: loss allowance 368,123 (MCB AR FY2018, OCR copy p.66, gated: gross 18,253,665 - allowance = net loans 17,885,542 on the balance sheet) / gross NPLs 121,717 (TZS '000; Phase 1 note) = 3.02. |
| 2018 | Gross loans to deposits | 2.25 | Verified, with a label correction: 2.25 = **net** loans 17,885,542 / customer deposits 7,948,974 (balance sheet p.25, OCR). On gross loans (18,253,665) the ratio is 2.30. The Phase 1 variable name says gross; the value is net-based. Lending far exceeded deposits in FY2018. |

No value was changed.
