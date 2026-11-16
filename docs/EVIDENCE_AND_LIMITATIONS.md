# Evidence discipline and limitations

Seeded from the research blueprint (sections 6-7; not published) and maintained with the code.

## Evidence classes (the research blueprint, section 6)

Every number, chart and sentence in either output carries one evidence class. The class is stored as a column in every output table.

| Class | Meaning | Example |
| --- | --- | --- |
| Observed | Reported directly in a source document or market record | NMB FY2024 net profit |
| Derived | Calculated deterministically from observed data | Cost of deposits; ROE |
| Estimated | Statistical or econometric estimate, reported with uncertainty | λ and its confidence interval |
| Scenario | Conditional on stated assumptions | Bear-case value per share |
| Interpretation | A plausible explanation, not directly established | Why NMB's funding is cheaper |

The text may not present an Interpretation in the language of an Observed or Estimated result.

## Limitations register (the research blueprint, section 7)

| # | Limitation | Effect | Mitigation | Disclosed in |
| --- | --- | --- | --- | --- |
| L1 | Seven listed banks out of about 40 in the system | No economy-wide causal claims; listed banks may differ systematically (selection) | Bank-level and comparative claims only; sector totals for system context | Ch. 1, 5 |
| L2 | Short panel per bank (about 17 years, fewer for some) | Wide confidence intervals; dynamic-panel bias; breaks can masquerade as persistence | Bank-by-bank estimates, bootstrap intervals, peer benchmark | Ch. 5 |
| L3 | Historical COE is itself estimated | Rent levels shift with COE | ±200 bp band; report rent sign and ranking robustness | Ch. 2 method, 5 |
| L4 | Accounting breaks: IAS 39 → IFRS 9 (FY2018), restatements, Group vs Bank basis | Discontinuities in credit and equity series | Restatement rows kept; transition year flagged and excluded in robustness | Ch. 4, 8 |
| L5 | Disclosure gaps (e.g. NMB impaired loans FY2008–09; deposit mix in early years) | Missing observations | Recorded as NOT DISCLOSED; never imputed silently | All |
| L6 | Descriptive capital-allocation measures (RORE, incremental ROE) | Cannot attribute causality to retention | Stated as descriptive; read against COE | Ch. 7 |
| L7 | Short daily price history (from Aug 2025) | Return statistics and volatility are noisy | Used for liquidity and capacity only, not for return forecasts | Ch. 12 |
| L8 | Market-access facts change and need legal confirmation | Risk of stale or wrong regulatory statements | Every claim cited with date; market-access findings checked by counsel; licensing position re-assessed if the work becomes commercial | Ch. 12 |
| L9 | Point-in-time discipline relies on publication dates, some estimated as upper bounds | Small risk of eligibility error near cut-offs | Upper-bound rule; near-cut-off documents confirmed | Ch. 9–10 |

## Limitations found when the data layer was built

- Phase 1 rows carry a publication date only where the Phase1 Source Map assigns a Doc ID; the rest are excluded by `tzbank.pit.eligible` (L9).
- The five non-focus banks lack the interest income/expense split, gross loans, RWA and Tier 1 capital in Phase 1 (see reports/data_audit.md).
- Interest expense on deposits is not held for any bank (cost of deposits, chapters 4.2 and 6).
- FinScope survey year is not documented in the delivered files.
