# Dominant Banks, Conditional Rents: dataset

Dataset of *Dominant Banks, Conditional Rents: NMB, CRDB and the price of dominance in Tanzania* (Jasco John,
2026), findings pack v1.17, valuation date 31 October 2026. Licence: CC BY 4.0 (`LICENSE-DATA.md` in the code
repository). The same files are in the public code repository and in the Zenodo dataset record.

## Contents

| Path | What it holds | Files |
| --- | --- | --- |
| `outputs/tables/` | Chapter tables (chapters 1-14) | 238 |
| `outputs/valuation/` | Valuation results (Part III) | 3 |
| `outputs/findings/findings_pack_v1.17.csv` (+ `.md`) | Every figure and verdict the dashboard shows | 2 |
| `data/raw/MANIFEST.csv` | Every raw input with its sha256 hash; source documents are listed, not included | 1 |
| `DATA_SHA256SUMS.txt` | sha256 of every file above | 1 |

The findings pack gives each figure an id, unit, bank, date, chapter, source table and dashboard page; the four
reports' figures are a subset of it.

Every table has a header row and an `evidence_class` column (Observed, Derived, Estimated, Scenario or
Interpretation). Money is in full Tanzanian shillings (TZS), ratios are decimals, per-share values are as reported
(NMB's 1:10 split, effective 24 August 2026, is applied in code). Variables are defined in
`docs/VARIABLE_DICTIONARY.md` of the code repository.

## Sources

Audited annual reports and BoT-format quarterly statements of the banks; Bank of Tanzania publications and statistics;
Dar es Salaam Stock Exchange daily reports; IMF and World Bank reports; broker notes and peer-bank reports. MANIFEST
lists every input file with its hash; `docs/SOURCE_REGISTER.md` (code repository) gives each document's URL. The
documents themselves, the DSE price library and the FinScope Tanzania microdata are third-party material, not
included, and keep their own terms.

## How it was produced

`python -m tzbank run all` (Python 3.11, pinned requirements) in the code repository: ingest, validation, the
point-in-time filter (only data published on or before each valuation date), chapters 1-14, the valuation and the
findings stage. Each run writes a run log with the git commit, input hashes and configuration.

Not investment advice.
