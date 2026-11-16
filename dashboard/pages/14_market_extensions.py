"""Chapter 14 page: market behaviour and valuation extensions (prompts 14-15). Display only (CLAUDE.md section 7)."""

import streamlit as st
from _common import render_outputs

st.caption("Market behaviour on the DSE data library: return attribution, event study against a "
           "market proxy (all DSE equities ex NMB and CRDB), the NMB split study (DiD vs CRDB and the market), "
           "seasonality 2015-2026 with HAC tests, NMB-CRDB co-movement, microstructure (unfiltered and "
           "stale-filtered), proxy and data-quality checks, quarterly fundamentals; valuation extensions (implied "
           "growth, scenario price lines, ROE* x persistence grid, real returns, yield gap), annual-report items, "
           "catalysts and thesis-break indicators. Descriptive results are labelled so; tests report their "
           "statistics. Register v1.3, no register changes. Figures carry findings-pack ids (CH14.*).")
render_outputs("14")
