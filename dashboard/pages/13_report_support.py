"""Chapter 13 page: report support (prompt 13). Display only (CLAUDE.md section 7)."""

import streamlit as st
from _common import render_outputs

st.caption("Report support: exhibits computed in the pipeline for the four reports. Each figure has a "
           "findings-pack id (CH13.*).")
render_outputs("13")
st.divider()
st.markdown("#### Report text exports")
render_outputs("13_text")
st.divider()
st.markdown("#### Decisions on contested analytical choices")
render_outputs("13_dec")
st.divider()
st.markdown("#### Peer banks")
render_outputs("13_peers")
st.markdown("#### Local analyst register (prompts 19-20)")
render_outputs("13_analysts")
st.markdown("#### Quarterly statements, Q4 2025 - Q3 2026")
render_outputs("13_qtr")
