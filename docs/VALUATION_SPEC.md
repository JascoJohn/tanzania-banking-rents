# Bank Equity Valuation Framework — Design Specification (NMB & CRDB)

Version as of 2026-11-05 · Author: Jasco John

> **Repository note.** This document is the detailed method for Part III (chapters 9–12) of the research. In the repository, the package referred to here as `bankval` is implemented inside `src/tzbank/` (mainly `banks/` and `valuation/`).

## 1. Scope and fixed decisions

The framework values NMB and CRDB on two dates with strictly separated information sets: a historical valuation at 31 Dec 2025 and a current valuation at 31 Oct 2026. It is built to publish as an independent research note, and any DSE-listed bank can be added through a configuration file.

| Decision | Choice | Implication |
| --- | --- | --- |
| Banks (v1) | NMB (B02), CRDB (B01) | NMB is the build case; CRDB is the generalisation test (foreign subsidiaries, NCI) |
| Purpose | Public research note, plus investor presentation | Every input is cited or entered as a named assumption; methods are disclosed in an appendix |
| Historical valuation | 31 Dec 2025 | Only information public on or before 31 Dec 2025; audited FY2025 results are excluded |
| Current valuation | 31 Oct 2026 | All information public on or before 31 Oct 2026, including audited FY2025 and 2026 interims |
| Information cut-off | Every input carries a publication date | An input enters a valuation only if published on or before that valuation date |
| Reporting basis | Group, NCI deducted | Value attributable to ordinary shareholders only; Bank-basis rows are flagged and not mixed in |
| Currency | Nominal TZS | Cash flows, cost of equity and growth all in TZS nominal terms |
| Horizon | 5-year explicit forecast, then 10-year ROE fade, then terminal | Avoids a terminal value built on peak ROE |
| Share counts | Weighted-average shares for EPS only; ordinary shares outstanding at the valuation date (ex-treasury, adjusted for bonus issues and splits) for value per share | The two measures are never interchanged |
| Source data | Phase 1 workbook, read-only | Valuation inputs live in a separate workbook; nothing writes back to the manuscript dataset |
| Deliverables | (1) financial data workbook, (2) valuation model, (3) research report, (4) investor presentation | The Python package produces 2–4 |

## 2. Valuation architecture

The intrinsic value is a weighted blend of two equity-side models. Multiples cross-check it and the implied cost of equity diagnoses it. Enterprise-value DCF is excluded, because deposits and borrowings are operating items for a bank.

| Method | Role | Default weight | Why |
| --- | --- | --- | --- |
| Residual income model (RIM) | Primary | 60% | Anchored on audited book value; makes value creation explicit as (ROE − COE) × book value |
| Capital-constrained dividend discount model (DDM) | Primary | 40% | Payout limited by what BoT capital rules allow; links value to RWA growth |
| Justified P/B and P/E | Cross-check | 0% | Tests whether the blend is consistent with sustainable ROE |
| Peer multiples (East African banks) | Cross-check | 0% | Market-relative sense check; shown on the valuation range chart |
| Implied COE and implied ROE at market price | Diagnostic | n/a | Shows what the price already assumes |

The weights are the central case, held in the assumption register (section 8), not hard-coded. Each model's value is published separately, and the note explains any material disagreement. RIM and DDM share one forecast, so a gap between them above 10% is flagged as a sign of an inconsistent payout or capital assumption.

### Valuation dates and information cut-off policy

The two valuations use identical model architecture but different information sets, opening balances and market assumptions. This eliminates look-ahead bias.

| | Historical valuation | Current valuation |
| --- | --- | --- |
| Valuation date | 31 Dec 2025 | 31 Oct 2026 |
| Information set | Public on or before 31 Dec 2025 | Public on or before 31 Oct 2026 |
| Financial base | Latest audited annual statements (FY2024) plus the latest interim published by the date (expected 9M 2025) | Audited FY2025 plus the latest 2026 interim published by the date |
| First forecast year | FY2025, built from interim actuals plus the remaining quarter | FY2026, built from year-to-date actuals |
| Price | DSE close on 31 Dec 2025 | DSE close on 31 Oct 2026 |
| Market inputs | Last bond auction, ERP and CRP dataset versions published by the date | The same, as at 31 Oct 2026 |
| Audited FY2025 results | Excluded | Included |

**Information-set integrity.** Every input stores two dates: the financial period end and the publication date. The loader assigns each input's eligibility automatically by comparing its publication date with the valuation date; eligibility is never entered by hand. Audited FY2025 results stay in the historical data store and enter only the current valuation.

**Valuation bridge.** The bridge from the historical to the current value is attributed sequentially in this fixed order:

1. Passage of time: historical value accreted at the historical COE.
2. Dividends paid between the two dates.
3. Newly available information: FY2025 audited and 2026 interim actuals versus the historical forecast for those periods.
4. Forecast revisions: changes to forward driver paths.
5. Change in cost of equity.

The order is stated in the note because sequential attribution depends on it. Any residual is reported, not absorbed.

## 3. Historical analytics

A bank DuPont over the full listed history (NMB from FY2008, CRDB from FY2009), extracted on one Group basis, sets the evidence base for every forecast driver. Covering more than one credit and rate cycle keeps through-the-cycle cost of risk and sustainable ROE from being calibrated on a single benign period. Each driver's forecast must be justified against its own history and against sector data from BoT.

ROE breaks down into margin, efficiency, credit cost, tax and leverage:

```latex
ROE = \underbrace{\frac{NII + NonII - OpEx - Impairment}{\overline{TA}}}_{\text{pre-tax ROA}} \times (1 - t) \times \underbrace{\frac{\overline{TA}}{\overline{Equity}}}_{\text{leverage}}
```

| Metric | Definition | Fallback if input missing |
| --- | --- | --- |
| Net interest margin (NIM) | NII ÷ average interest-earning assets | NII ÷ average total assets (labelled as such) |
| Fee share | Non-interest income ÷ total operating income | — |
| Cost-to-income | OpEx ÷ (NII + NonII) | Reported ratio |
| Cost of risk | Impairment charge ÷ average gross loans | ÷ average net loans |
| Effective tax rate | Tax ÷ PBT | (PBT − net profit) ÷ PBT |
| ROA / ROE | Attributable profit ÷ average assets / average ordinary equity | — |
| Leverage | Average assets ÷ average equity | — |
| RWA density | RWA ÷ total assets | Derived from capital ratio and capital amount |
| Payout | DPS × shares ÷ attributable profit | — |
| Internal capital generation | ROE × (1 − payout) | — |

**Normalisation rules**

- Averages use opening and closing balances; quarterly averages where interim data exists.
- One-off items (disposal gains, regulatory penalties) are stripped from the forecasting base and logged.
- Structural breaks are flagged, not smoothed: IFRS 9 adoption (FY2018) and CRDB's FY2015 rights issue are examples.
- The forecasting base uses the latest restated comparatives published by the valuation date (approved). Original figures are never overwritten. The manuscript workbook keeps own-year audited figures (Decision D-13), so the mapping layer records which convention each value follows.

## 4. Forecast engine

One generic, driver-based engine projects every bank; only driver values in each bank's configuration differ. Loans and deposits are forecast separately, and the balance sheet closes through liquid assets or wholesale funding. Regulatory capital, not accounting equity, limits distributions.

| Driver | Symbol | Forecast form | Anchored on |
| --- | --- | --- | --- |
| Net loan growth | g_L | Path FY+1 to FY+5, converging to terminal g | History, credit-to-GDP, nominal GDP growth |
| Deposit growth | g_D | Path | History, sector deposit growth, broad money |
| Other liabilities ÷ assets | ω | Constant or path | History |
| Fixed and other non-earning assets ÷ assets | θ | Constant | History |
| Minimum liquid assets ratio | ℓ_min | Constant | BoT liquidity requirement plus bank buffer |
| Net interest margin | NIM | Path | History, BoT policy rate cycle |
| Wholesale funding spread over deposit cost | s_W | Constant | Interbank and term borrowing costs |
| Non-interest income ÷ assets | φ | Path | History, digital-fee trend |
| Cost-to-income | CIR | Path | History, management guidance |
| Cost of risk | CoR | Path, through-the-cycle by terminal | History, NPL trend |
| Effective tax rate | t | Constant | Statutory 30%, history |
| NCI share of profit | ν | Constant (Group); by subsidiary for CRDB | Ownership percentages |
| RWA density | ρ | Path | History, Basel II/III transition |
| Regulatory deductions ÷ equity | δ | Constant or path | History: (equity − Tier 1 capital) ÷ equity |
| Target Tier 1 ratio | k* | Constant | BoT minimum + capital conservation buffer + bank buffer |
| Payout cap | π_max | Constant | Dividend policy |

**Projection equations** (t = forecast year; bars denote averages)

```latex
\begin{aligned}
&\textbf{Balance sheet} \\
L_t &= L_{t-1}(1+g_{L,t}), \qquad D_t = D_{t-1}(1+g_{D,t}), \qquad OL_t = \omega\, TA_t \\
Gap_t &= L_t + \theta\, TA_t + LA^{min}_t - (D_t + OL_t + E_t) \\
W_t &= \max(0,\, Gap_t), \qquad LA_t = LA^{min}_t + \max(0,\, -Gap_t) \\
TA_t &= L_t + LA_t + \theta\, TA_t \\
&\textbf{Income} \\
NII_t &= NIM_t \cdot \overline{(L + LA)}_t - s_W \cdot \overline{W}_t \\
NonII_t &= \varphi_t \cdot \overline{TA}_t, \qquad OpEx_t = CIR_t\,(NII_t + NonII_t) \\
Imp_t &= CoR_t \cdot \overline{L}_t \\
NP_t &= (NII_t + NonII_t - OpEx_t - Imp_t)(1-t), \qquad NP^{attr}_t = NP_t(1-\nu) \\
&\textbf{Accounting equity (used by RIM)} \\
E_t &= E_{t-1} + NP^{attr}_t - Div_t \\
&\textbf{Regulatory capital (used by DDM)} \\
T1_t &= E_t(1-\delta_t), \qquad RWA_t = \rho_t\, TA_t \\
Div_t &= \min\!\left(\pi_{max} NP^{attr}_t,\; \max\!\left(0,\; \frac{(E_{t-1} + NP^{attr}_t)(1-\delta_t) - k^{*} RWA_t}{1-\delta_t}\right)\right)
\end{aligned}
```

LA^min is the liquid-asset floor implied by ℓ_min. A funding shortfall is met by wholesale funding W at a spread, so loan growth that outruns deposits visibly costs margin. Surplus funding sits in liquid assets. TA appears on both sides and is solved algebraically each year.

**Fade and terminal.** After the five-year explicit period (FY2025–29 historical, FY2026–30 current), ROE moves linearly over 10 years to a sustainable ROE*, and growth moves to terminal g. ROE* is set per scenario (bear, base, bull) from normalised multi-year ROE, East African peer profitability, sector competition and growth, capital requirements, and its spread over COE. It is never the current ROE carried forward. The terminal year must satisfy g = ROE* × (1 − payout*), so growth is funded internally.

> Rule: ROE* and the fade speed are set only after chapter 5 persistence estimates exist; the estimated persistence parameters (λ, ER*) then inform this fade.

**Engine checks.** Assets equal liabilities plus equity every year. Tier 1 capital never falls below k* × RWA. Liquid assets never fall below the floor. The loan-to-deposit ratio and the wholesale funding share stay inside declared bounds. The capital test runs at Group level; Bank-solo capital is checked where data allows, and the tighter of the two binds. Approved exception: NMB discloses regulatory capital for the Bank only, so NMB's capital rule uses Bank-solo Tier 1 capital and RWA. Any breach stops the run with a named error.

### CRDB subsidiary layer

For groups with material foreign operations, profit and distributions are modelled in four layers:

| Layer | What is modelled | Why |
| --- | --- | --- |
| Parent bank (Tanzania) | Full driver set above | Largest share of profit and the dividend-paying entity |
| Subsidiaries (Burundi, DRC, others) | Profit growth, TZS translation at a forecast FX path, ownership share | Different growth, risk and currency exposure |
| NCI | Each subsidiary's profit × minority ownership | Removes profit not owned by CRDB shareholders |
| Distributable to CRDB shareholders | Parent profit + upstream ratio u_j × subsidiary dividends, subject to parent capital | Foreign subsidiaries' profit is not cash at the parent until upstreamed, and local capital rules or FX controls can block it |

The RIM uses consolidated attributable profit and equity. The DDM uses the parent's distributable capacity. Scenarios shock subsidiary FX and credit separately from Tanzania.

## 5. Cost of equity

COE is estimated two independent ways in nominal TZS, then reconciled into a central value and a range. The range feeds the sensitivity grids, and the market-implied COE (section 6) tests it.

**Approach A — TZS build-up.** The TZS long bond yield includes sovereign default risk. That default spread is removed first so it is not counted twice when the country risk premium is added.

```latex
COE_{TZS}^{A} = \left(y_{TZS}^{long} - DS_{TZ}\right) + \beta_L \left(ERP_{mature} + CRP_{TZ}\right) + LP
```

**Approach B — USD CAPM, converted by relative inflation.**

```latex
COE_{USD} = r_{f,US} + \beta_L \left(ERP_{mature} + CRP_{TZ}\right) + LP
\qquad
COE_{TZS}^{B} = (1 + COE_{USD})\,\frac{1 + \pi_{TZS}}{1 + \pi_{US}} - 1
```

| Input | Symbol | Source | Notes |
| --- | --- | --- | --- |
| Long TZS government bond yield | y_TZS | BoT Treasury bond auction results (15-, 20- and 25-year) | Use the yield nearest the valuation date; record the auction date |
| Sovereign default spread | DS_TZ | Rating-based spread (Damodaran country risk dataset) | Tanzania rating as at valuation date |
| US risk-free rate | r_f,US | US 10-year Treasury yield | Valuation date |
| Mature-market ERP | ERP_mature | Damodaran implied ERP | Latest monthly update |
| Country risk premium | CRP_TZ | Default spread × equity/bond volatility ratio | Damodaran method |
| Bank beta | β_L | Bottom-up: emerging-market / African bank betas, relevered | DSE regression beta (Dimson-adjusted for thin trading) reported as a diagnostic only |
| Illiquidity premium | LP | Central 1.5%, sensitised over 0–3% | DSE turnover and free float justify a non-zero value |
| Inflation, TZS and US | π | NBS CPI and BoT outlook; US long-run expectations | Forward-looking, not trailing |

**Reconciliation rule.** If A and B differ by less than 150 bp, the central COE is their midpoint. If the gap is larger, the input causing it is investigated and the choice is written into the register with reasons. COE is held constant in the explicit period. A separate terminal COE is allowed if the TZS rate environment is expected to normalise.

## 6. Valuation model equations

All models run off the forecast in section 4 and discount at r = COE. τ_t is the year fraction from the valuation date to the end of period t, which allows mid-year dates for the roll-forward. B is ordinary equity attributable to shareholders, and N is the last fade year.

**Residual income model**

```latex
V_0 = B_0 + \sum_{t=1}^{N} \frac{(ROE_t - r)\,B_{t-1}}{(1+r)^{\tau_t}} + \frac{(ROE^{*} - r)\,B_N}{(r-g)\,(1+r)^{\tau_N}}
```

This relies on clean surplus: all equity changes pass through profit or dividends. Material other comprehensive income movements (FVOCI reserves, FX translation for CRDB's subsidiaries) are tested for in history and flagged. If ROE* = r, the terminal term is zero, which is the conservative default.

**Capital-constrained DDM**

```latex
V_0 = \sum_{t=1}^{N} \frac{Div_t}{(1+r)^{\tau_t}} + \frac{Div_{N+1}}{(r-g)\,(1+r)^{\tau_N}}
```

Div_t comes from the capital rule in section 4. It is never the historical payout carried forward.

**Justified multiples**

```latex
\frac{P}{B}^{*} = \frac{ROE^{*} - g}{r - g}
\qquad
\frac{P}{E}^{*} = \frac{\pi^{*}(1+g)}{r - g}
```

**Market-implied diagnostics.** These are solved numerically with a bracketed root-finder.

- Implied COE: the r at which the RIM value equals the market price, holding the forecast fixed.
- Implied sustainable ROE: the ROE* at which the value equals the price, holding r fixed.

> Rule: single implied values are not identified on their own. Chapter 10 reports the full iso-value surface of (λ, ER*) pairs consistent with the price, overlays the chapter 5 estimates, and states the extra assumption behind any conclusion.

**Peer multiples.** P/B is regressed on ROE across East African listed banks. The bank's residual from that line is reported; peer P/B is never applied mechanically.

**Outputs per bank and date**

- **Value per share** = (w_RIM × V_RIM + w_DDM × V_DDM) ÷ ordinary shares outstanding at the valuation date.
- **12-month target price** = V₁ per share. V₁ is the blended value re-computed at the valuation date plus 12 months, using projected opening equity, regulatory capital and the remaining forecast cash flows and residual incomes. It is divided by projected shares outstanding at that date.
- **Identity check:** with an unchanged forecast and COE, V₁ = V₀(1 + r) − Div₁ must hold within rounding. A breach signals an engine error, so the run stops.
- **Expected total return** = (target price − current price + expected DPS over the 12 months) ÷ current price.
- **Downside measure:** bear-scenario value per share and its implied loss versus the current price, reported beside the expected return.

## 7. Uncertainty

The note publishes a value range, not a point estimate alone. Three layers of uncertainty analysis produce that range.

| Layer | Design | Published output |
| --- | --- | --- |
| Sensitivity grids | COE × terminal g; COE × ROE*; NIM × cost of risk, each on value per share; ±200 bp in 50 bp steps | Three heat-map tables |
| Scenarios | Bear, base and bull, each a coherent macro story with a full driver set, not single-variable shocks | Value per share and FY2026–28 EPS and ROE per scenario |
| Monte Carlo | 10,000 draws over NIM, CoR, CIR, g_A, COE and LP; triangular or truncated-normal distributions; correlation matrix (e.g. NIM with policy rate, CoR against growth); fixed seed | Value distribution, P5–P95, probability that value exceeds price |

**Scenario stories (to calibrate)**

- **Bear:** BoT easing compresses NIM, a credit cycle lifts CoR, and TZS depreciation hits CRDB's foreign subsidiaries.
- **Base:** drivers follow the central register values.
- **Bull:** digital-fee growth and continued cost efficiency, with credit quality holding.

The valuation range chart combines the RIM, DDM, scenario, Monte Carlo P10–P90, justified P/B and peer ranges against the current price and 52-week range.

## 8. Assumption register

Every judgement in the model is one row in a versioned register. The code reads values only from there, so a published number can always be traced to a stated assumption and its evidence.

| Field | Content |
| --- | --- |
| id | Stable key, e.g. NMB.NIM.2026 |
| scope | Global, market or bank |
| bank | Ticker, or ALL |
| parameter / symbol | Name and symbol as used in sections 4–6 |
| period | Year, a range, or terminal |
| base / bear / bull | Scenario values |
| distribution | Type and parameters for Monte Carlo |
| rationale | One to three sentences on why |
| evidence | Historical metric and/or source citation (document, page, URL, access date) |
| set_on / version | Date and register version |

The structure is shown below; values are set in the calibration step, not here.

```csv
id,scope,bank,parameter,symbol,period,base,bear,bull,distribution,rationale,evidence,set_on,version
NMB.NIM.2026,bank,NMB,Net interest margin,NIM,2026,TBD,TBD,TBD,triangular,TBD,DuPont FY2021-25; BoT MPC statement,TBD,v0.1
MKT.ERP,global,ALL,Mature-market ERP,ERP_mature,all,TBD,TBD,TBD,fixed,TBD,Damodaran implied ERP (monthly),TBD,v0.1
CRDB.NCI,bank,CRDB,NCI share of profit,nu,all,TBD,TBD,TBD,fixed,TBD,CRDB FY2025 audited FS,TBD,v0.1
```

The register lives in version control. Each published note records the register version and the input-data hash used.

## 9. Data specification

The Phase 1 workbook covers headline P&L, balance sheet, capital and market series for both banks over FY2016–FY2025, but it records no publication dates. Seventeen further inputs are needed; eight of them block a first run.

**Available in Phase 1 (read through the mapping layer)**

- **P&L:** NII, non-interest income, OpEx, impairment, PBT, net profit, EPS.
- **Balance sheet:** total assets, net loans, customer deposits, total equity.
- **Capital and risk:** Tier 1 and total capital ratios, NPL ratio, NPL coverage, LDR.
- **Market:** year-end price, shares outstanding, market capitalisation, DPS.

Known holes in the history: NMB Tier 1 ratio FY2023, CRDB total capital ratio FY2019, NMB NPL coverage (7 of 18 years), and a shorter CRDB price and market-cap history.

**Gaps to fill in the separate valuation-inputs workbook**

| # | Item | Needed for | Source | Priority |
| --- | --- | --- | --- | --- |
| 1 | Publication date for every input, including existing Phase 1 values | Cut-off policy, both valuations | DSE announcements, newspaper publication, bank IR pages | Blocking |
| 2 | Latest interim published by 31 Dec 2025 (expected 9M 2025) | Historical valuation base | Published quarterly statements | Blocking |
| 3 | 2026 interims published by 31 Oct 2026 | Current valuation base | Published quarterly statements | Blocking |
| 4 | Profit attributable to shareholders and NCI; NCI in equity | Group basis with NCI deducted | Audited FS FY2021–25 | Blocking |
| 5 | Tax expense | Effective tax rate | Audited FS | Blocking |
| 6 | RWA, Tier 1 capital and regulatory deductions (TZS) | Regulatory capital rule, δ | Capital management note | Blocking |
| 7 | Long TZS bond yield, with auction date, at both dates | COE | BoT auction results | Blocking |
| 8 | DSE close at both dates; 52-week range; turnover and free float | Prices, illiquidity premium evidence | DSE daily market reports | Blocking |
| 9 | Interest income and expense; interest-earning and liquid assets; wholesale borrowings | NIM, funding gap | Audited FS | High |
| 10 | Gross loans and Stage 3 loans | Cost of risk on gross basis | IFRS 9 credit-risk note | High |
| 11 | CRDB segment profit, subsidiary ownership, dividends upstreamed | Subsidiary layer | Segment and investment notes | High |
| 12 | Dividend amounts, record and payment dates | Valuation bridge, expected DPS | DSE corporate actions | High |
| 13 | Intangibles and deferred tax assets | Tangible book cross-check | Audited FS | Medium |
| 14 | OCI items (FVOCI reserve, FX translation) | Clean-surplus test | Statement of changes in equity | Medium |
| 15 | CPI (TZ, US), BoT policy rate, FX (TZS/USD, BIF, CDF) | COE approach B, scenarios, subsidiary translation | NBS, BoT, US BLS, central banks | Medium |
| 16 | ERP, CRP, default spread, bank betas, by dataset version date | COE | Damodaran datasets (archived versions) | Medium |
| 17 | Peer bank price, book value and ROE | Peer multiples | NSE, USE, RSE filings | Low |

Each new value is stored in the Phase 1 long format, extended with two date fields: bank, period end, publication date, variable, value, unit, basis, status, source document, page, URL and confidence. Restated figures are stored as new rows with their own publication date, so both the original and the restated value remain available and each valuation sees only what was public at its date.

> Status note (Nov 2026): items 1–8 and most of 9–16 are now filled in `NMB_CRDB_Valuation_Inputs_v0.6.xlsx`; remaining gaps are listed on its Validation sheet and in the repository data audit.

## 10. Validation and software architecture

The system is a Python package (`bankval`, implemented in the repository as `src/tzbank/`), with a hard validation gate between data and modelling. A run that fails any check produces no valuation.

**Architecture (inputs to outputs).** Every published number passes the validation gate first:

1. **Inputs:** Phase 1 workbook (read-only source), valuation-inputs workbook (gap-fill), bank configuration (one YAML per bank), assumption register (versioned).
2. **Data layer:** mapping to a canonical long schema.
3. **Validation gate:** schema, identities, basis, units. The bank configuration and assumption register are validated here too.
4. **Analysis:** historical analytics (bank DuPont) feeds the forecast engine (driver-based, capital rule).
5. **Cost of equity:** TZS build-up plus USD CAPM. It feeds the valuation models, not the forecast.
6. **Valuation models:** RIM, capital DDM, multiples and implied COE, which then feed the uncertainty layer (grids, scenarios, Monte Carlo).
7. **Outputs:** Excel model, charts, research note, and a run log with the input hash.

**Validation rules (run before any model)**

| Check | Rule | On failure |
| --- | --- | --- |
| Information cut-off | No input with a publication date after the valuation date enters that valuation; inputs without a publication date are excluded | Stop |
| Schema | Required variables present for every base year; numeric types; units declared | Stop |
| Basis | All rows used for one bank share one basis (Group) | Stop |
| EPS identity | EPS × weighted shares within 2% of attributable profit | Warn; stop above 5% |
| Capital consistency | Total capital ratio ≥ Tier 1 ratio; Tier 1 capital ≤ accounting equity | Stop |
| Market identity | Price × shares outstanding within 1% of market capitalisation | Warn |
| Equity roll | E_t ≈ E_t−1 + profit − dividends ± OCI, within 3% | Warn; triggers the clean-surplus flag |
| Target-price identity | V₁ = V₀(1 + r) − Div₁ with unchanged forecast and COE | Stop |
| Confidence | Inputs rated LOW are listed in the run log | Report |
| Register | Every driver has base, bear and bull values within declared bounds | Stop |

**Package layout (original design; see the repository note at the top for where each part lives in `src/tzbank/`)**

```
bankval/
  config/           banks/nmb.yaml, banks/crdb.yaml, market.yaml
  register/         assumptions.csv (versioned)
  data/             loaders, Phase 1 mapping, canonical schema
  validation/       pandera schemas, identity checks
  analytics/        DuPont, normalisation
  forecast/         driver engine, capital rule, fade
  cost_of_capital/  approach A, approach B, reconciliation
  models/           rim.py, ddm_capital.py, multiples.py, implied.py
  risk/             sensitivity.py, scenarios.py, montecarlo.py
  reporting/        excel export, charts, Jinja2 note template
  cli.py            bankval run --bank NMB --date 2025-12-31
tests/              pytest: identities, golden-file regression per bank
runs/<timestamp>/   outputs, input hash, register version, log
```

**Stack:** pandas, numpy, scipy (root-finding), pandera, pydantic, PyYAML, matplotlib, xlsxwriter, Jinja2, pytest. Notebooks are for exploration only and never feed a published number.

## 11. Research note standard and open decisions

Each bank gets an initiation-style note generated from the same run outputs. No figure in the note is typed by hand.

**Note structure**

1. **Page 1:** rating, 12-month target, expected total return, and a key-metrics table (FY2024A–FY2028F: EPS, DPS, BVPS, ROE, P/E, P/B, yield).
2. **Investment thesis:** three points, each tied to a driver.
3. **Valuation:** method blend, value bridge between the two dates, valuation range chart, and sensitivities.
4. **Forecasts and DuPont history.**
5. **Risks:** scenarios and the probability that value exceeds price.
6. **Appendix:** methodology (this spec, condensed), assumption register extract, sources, disclosures.

**Rating rule.** Expected total return is compared with COE, and downside is measured as the bear-case value per share against the current market price, not against base-case fair value.

| Rating | Conditions |
| --- | --- |
| Buy | Expected return exceeds COE by more than 5 percentage points, and bear-case downside is no worse than 25% |
| Hold | Buy conditions not met, and Sell condition not met |
| Sell | Expected return is more than 5 percentage points below COE |

A failed downside test turns a Buy into a Hold, never automatically into a Sell. Any analyst override of the rule-based rating is recorded in the run log with its rationale and disclosed in the note.

**Peers.** NMB and CRDB are direct peers of each other. KCB, Equity, Co-op, Stanbic Uganda and Bank of Kigali are contextual peers. They differ in footprint, business mix and liquidity, so their multiples are read alongside ROE, growth and risk, never applied mechanically.

**Investor presentation.** A short deck per bank is generated from the same run: thesis, valuation range, bridge, and risks.

**Disclosures.** State analyst independence, any holdings in the covered stocks, compensation (none), and the academic affiliation. Include a not-investment-advice statement.

**Open decisions**

- [x] Model weights: RIM 60% / DDM 40% as the central case, with each model shown separately.
- [x] Illiquidity premium: 0–3% range, 1.5% central, supported by turnover and free-float evidence.
- [x] Rating rule: COE ± 5 points; Buy also requires bear-case downside no worse than 25%.
- [x] Peer set: direct and contextual peers as above.
- [x] Restated comparatives as the forecasting base, originals retained.
- [x] Information cut-off policy and dual-date design (section 2).
- [x] Terminal ROE: evidence-based bear, base and bull values with the sustainable-growth constraint (section 4).
- [ ] Confirm with CMSA or Tanzanian counsel whether publishing ratings and target prices requires an investment adviser licence, and which disclosures apply.
- [ ] Presentation audience: professional investors, analysts and researchers, or both.
