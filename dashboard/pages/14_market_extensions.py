"""Chapter 14 page: market behaviour and valuation extensions. Display only (CLAUDE.md section 7)."""

import _views as V
import streamlit as st
from _common import render_outputs

st.caption("Market behaviour from the DSE data: prices, turnover and liquidity; return attribution, event study "
           "against a market proxy (all DSE equities except NMB and CRDB), the NMB split study, seasonality, "
           "NMB-CRDB co-movement and microstructure; valuation extensions. Descriptive results are labelled so; "
           "tests report their statistics. Figures carry findings-pack ids (CH14.*).")
V.market()
render_outputs("14")
