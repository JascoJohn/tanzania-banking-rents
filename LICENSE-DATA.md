# Licence for data, tables and reports

Copyright (c) 2026 Jasco John.

The code in this repository (`src/`, `scripts/`, `tests/`, `dashboard/`) is licensed under the MIT
licence in [`LICENSE`](LICENSE).

Everything else that is the author's own work is licensed under the **Creative Commons Attribution 4.0
International licence (CC BY 4.0)**, <https://creativecommons.org/licenses/by/4.0/>:

- the derived tables, figures and notes in `outputs/` (including the findings pack), `DATA_README.md` and
  `DATA_SHA256SUMS.txt`;
- the owner-compiled input workbooks and manual-entry files in `data/raw/` (values transcribed from published
  sources, each with its document and page), except the DSE daily trading data in them (below);
- the documentation in `docs/` and the four research reports (PDF, on Zenodo).

You may share and adapt this material for any purpose, provided you give appropriate credit (see
[`CITATION.cff`](CITATION.cff)), link to the licence and indicate whether changes were made.

## Third-party sources keep their own terms

The figures here are derived from published sources: bank annual and quarterly reports, Bank of Tanzania
publications, Dar es Salaam Stock Exchange (DSE) data, IMF and World Bank reports, broker notes, peer-bank reports and
the FinScope Tanzania survey. CC BY 4.0 covers the author's compilation and analysis. It does not cover the underlying
documents or datasets, which remain under their owners' terms.

- **Source documents are not redistributed.** PDFs and other third-party files are git-ignored. Each is listed in
  `data/raw/MANIFEST.csv` with its sha256 hash and described, with its URL, in `docs/SOURCE_REGISTER.md`.
- **DSE daily trading data:** source Dar es Salaam Stock Exchange; included for non-commercial research reproducibility; remains subject to the DSE’s own terms and is not covered by this project’s CC BY 4.0 licence. It is in
  `data/raw/NMB_CRDB_52wk_and_trading_value.xlsx` and the "Daily Market Data" sheet of
  `data/raw/NMB_CRDB_Valuation_Inputs.xlsx`. Otherwise the repository carries derived figures and short price series;
  the DSE site snapshots and the price library built from them are not included.
- **FinScope Tanzania microdata and data map** are FSD Tanzania's and are not included. Request them from FSD
  Tanzania.

Nothing here is investment advice. See the disclaimer in [`README.md`](README.md).
