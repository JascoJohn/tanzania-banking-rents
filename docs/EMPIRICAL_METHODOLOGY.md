# Empirical methodology

Seeded from the research blueprint (section 2; not published) and maintained with the code.

The code implements these formulas exactly; any deviation is recorded here.

## Core definitions and methods (locked)

**Economic rent.** High ROE is not rent. Rent is the return above the cost of equity, measured two ways:

```latex
ER_{i,t} = ROE_{i,t} - COE_t \qquad RI_{i,t} = NI_{i,t} - COE_t \cdot B_{i,t-1}
```

ROE uses profit attributable to ordinary shareholders over average attributable equity (Group basis). ER is the excess return rate; RI is the shilling value created above the equity charge.

**Historical cost of equity.** Rents over 2008–2025 need a year-by-year COE, not today's. Each year uses the TZS build-up from the valuation specification:

- the long-bond yield at each year-end (the average WAYTM of all Treasury bond auctions with tenor >= 15 years in the final quarter to the date; fallback the most recent such auction before it; the "longest tenor nearest year-end" and the "most recent long-tenor auction" are kept as sensitivity series), net of the sovereign default spread where available;
- plus beta × (mature-market ERP + Tanzania country risk premium);
- plus the illiquidity premium.

The annual COE is kept as a table with every input documented and sourced. Every rent result is re-run with COE shifted ±200 bp. A conclusion that flips inside that band is reported as not robust.

**Persistence model.** Excess returns follow a partial-adjustment process, estimated bank by bank:

```latex
ER_{i,t} = \alpha_i + \lambda_i \, ER_{i,t-1} + \varepsilon_{i,t}, \qquad ER_i^{*} = \frac{\alpha_i}{1-\lambda_i}, \qquad \text{half-life}_i = \frac{\ln 0.5}{\ln \lambda_i}
```

λ is the share of this year's excess return that survives into next year; ER\* is the long-run rent level.

- **Status of the AR(1).** It is a descriptive benchmark, not the model of record. With about 17 annual observations per bank, a single AR(1) can mistake a structural break (IFRS 9 in FY2018, the 2015–2017 credit episode, a rights issue) for persistence.
- **Valid range.** The half-life is reported only for 0 < λ < 1. If λ ≤ 0 there is no positive persistence to express as a half-life. If λ ≥ 1, or the confidence interval includes 1, mean reversion is not established and no half-life is quoted.
- **Uncertainty:** moving-block bootstrap confidence intervals. Bootstrapping quantifies uncertainty; it does not create independent observations.
- **Structural breaks:** test for breaks (known-date Chow tests at FY2018 and the 2015–2017 episode; a sup-F test where the sample allows). Re-estimate on sub-periods where a break is found.
- **Alternative specifications:** AR(1) with a break dummy; mean-reversion towards a time-varying sector benchmark; sign persistence (share of years with ER > 0); rank persistence across the seven banks.
- **Pooled estimate:** a fixed-effects panel is reported as secondary only, flagged for small-sample (Nickell) bias.
- **Robustness:** excluding FY2018, Bank versus Group basis, and the COE band.
- **Extension:** East African peer banks as a benchmark, not a prerequisite.

**Implied persistence (the headline test).** In the residual income model, excess returns fade as ER\_t = ER\* + λᵗ(ER₀ − ER\*). Many combinations of long-run rent and fade speed reproduce the same price, so a single implied λ is not identified. Chapter 10 therefore:

1. Computes the full iso-value surface: every (λ, ER\*) pair that reproduces the market price, for each bank and valuation date, across the COE range.
2. Overlays the chapter 5 estimate (λ̂, ER\*̂) and its confidence region.
3. States the additional assumption needed to read a conclusion from the gap. For example: "if long-run rent is held at the estimated ER\*, the price implies λ = x"; or "if fade speed is held at λ̂, the price implies ER\* = y".

The conclusion is stated only for the regions where the estimate lies clearly on one side of the surface.

**Incremental capital productivity** (chapter 7), over k = 3 and 5 years. The primary measure uses attributable earnings and retained equity, which share issuance, splits and buybacks cannot distort:

```latex
RORE^{primary}_{t,k} = \frac{NI^{attr}_t - NI^{attr}_{t-k}}{\sum_{j=t-k}^{t-1} \left(NI^{attr}_j - Div_j\right)} \qquad \Delta ROE_{t,k} = \frac{NI^{attr}_t - NI^{attr}_{t-k}}{B_t - B_{t-k}}
```

The EPS-based version, (EPS\_t − EPS\_{t−k}) ÷ Σ(EPS\_j − DPS\_j), is kept as supplementary, on split-adjusted per-share data. Periods containing a capital raise (e.g. CRDB's FY2015 rights issue) are flagged, and retained equity is adjusted for new capital.

Both measures are descriptive, not causal. They are read against COE: above it, retention created value; below it, retention only enlarged the balance sheet.

Windows (t−k, t] containing a raise have no primary RORE; windows starting in FY2015–18 are flagged and medians are shown with and without them. The H7.2 slope of incremental ROE on log average assets is a within-bank (bank-demeaned) OLS slope; its 90% interval is the moving-block bootstrap (block 3, 2,000 replicates, percentile), resampling consecutive windows within each bank so that window overlap is carried into the replicates; the OLS t interval is reported alongside. Capital headroom is computed against both the Tier 1 and the total capital requirement (stated level + 2.5% conservation buffer from August 2017 where the stated level excludes it), and the binding constraint is the smaller headroom.

**Credit risk and capital efficiency** (chapter 8). Cost of risk is impairment ÷ average gross loans in each year, as computed; the through-the-cycle level is the mean over the primary cycle FY2010–25 (one full cycle with the 2015–18 stress), with every cycle starting FY2010–14 and ending FY2021–25 as a boundary sensitivity. H8.1 compares FY2025 with that level; the interval comes from a moving-block bootstrap of the years (block 3). Risk-adjusted return on RWA is (PBT + impairment − TTC cost of risk × average gross loans) ÷ average RWA. The IFRS 9 transition in FY2018 is not smoothed: the opening adjustment to equity is reported separately and the two regimes are summarised separately as well as together. The stress lens replaces Imp/TA in the 4.0 tree with the 2015–18 cost of risk scaled by gross loans ÷ total assets and holds every other component fixed.

**Gross deposit funding advantage and deposit beta** (chapters 4.2 and 6):

```latex
GDFA_t = (y^{bill}_t - c^{dep}_t) \cdot \bar{D}_t \qquad c^{dep}_t = \frac{\text{interest expense on deposits}_t}{\bar{D}_t} \qquad \beta^{dep} = \frac{\Delta c^{dep}}{\Delta y^{bill}}
```

GDFA is a **proxy**, not net economic rent. It ignores:

- the cost of reserve requirements and required liquid assets;
- the operating cost of running the deposit franchise (branches, agents, staff);
- deposit insurance premiums;
- the risk of the assets the deposits fund.

Chapter 6 therefore also estimates a **net deposit franchise contribution**: GDFA minus statutory reserve cost, minus the share of operating expenses attributable to deposit gathering (allocation rule stated and sensitised), minus deposit insurance. Only the net measure is discussed as a source of rent. A low deposit beta means a sticky, cheap franchise.

**Credit and capital efficiency** (chapter 8):

- cost of risk = impairment ÷ average gross loans;
- through-the-cycle cost of risk = the mean across a full cycle, including CRDB's 2015–2017 episode;
- return on RWA = net income ÷ average RWA.

**Concentration** (chapter 1): CR2 = the NMB + CRDB share of sector totals, which needs only sector totals. HHI is computed only if every bank's share is available.

## Implementation notes

- Averages for ratio denominators are (opening + closing) / 2 and require the immediately preceding fiscal year (`tzbank.chapters.ch04.profitability.average`).
- ROE for the five non-focus banks uses net profit / average total equity because Phase 1 holds no NCI split; the column `roe_basis` records which definition each row uses.
- Cost of risk follows the definition (average gross loans). A net-loans version is written as a separately labelled proxy, never in place of the defined measure.

## Implementation notes

- Long-bond yield: `tzbank.market.coe_inputs` writes `data/processed/coe_inputs_annual.csv` (three series per
  year-end and valuation date), `coe_long_bond_auctions_used.csv` (every auction entering a series) and
  `coe_long_bond_comparison.csv` (the primary rule against the alternatives). The final quarter is the three
  calendar months ending on the date. The primary is a simple, not amount-weighted, average of auction WAYTMs.
- Cost of deposits uses deposit interest admitted only by the note reconciliation for NMB and CRDB (note lines sum to
  the audited total interest expense within 0.1%; `reports/validation/deposit_interest_reconciliation.csv`).
- Bank-years with an open or excluding plausibility flag (`reports/validation/plausibility_flags.csv`,
  `outliers.csv`) carry `accepted_for_analysis = False` in `data/processed/bank_year_flags.csv`; chapter 5 estimates
  use only accepted bank-years, and report the effect of including the verified outliers.

**Valuation engine (VALUATION_SPEC §4-7).** The forecast engine follows VALUATION_SPEC §4 exactly:
loans and deposits are projected separately, liquid assets or wholesale funding close the balance sheet (solved each
year by fixed-point iteration because equity enters total assets), and dividends follow the capital rule
Div = min(pi_max NP_attr, max(0, ((E_-1 + NP_attr)(1 - delta) - k* RWA)/(1 - delta))). Two additions are stated here
because they go beyond the spec text: NIM moves with the T-bill path through asset and deposit betas,
and the fade uses ROE_t - ROE* = lambda (ROE_t-1 - ROE*) with the break-adjusted lambda (owner pre-flight D; the spec's
linear fade remains a configuration option). RIM and DDM run off one forecast; with clean surplus and a terminal payout
of 1 - g/ROE*, they agree exactly, so any gap above 10% signals an inconsistent register. Monte Carlo draws use a
Gaussian copula with triangular marginals whose bounds are the register's bear and bull shifts.

