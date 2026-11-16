"""Chapter 13 page: report support. Display only (CLAUDE.md section 7): selects and re-plots published tables."""

import _views as V
import streamlit as st
from _common import render_outputs

st.caption("Exhibits computed in the pipeline for the four reports: African peers, analyst views, quarterly "
           "statements and the supporting tables. Each figure has a findings-pack id (CH13.*).")
peers, analysts, quarterly, other = st.tabs(["African peers", "Analyst views", "Quarterly statements",
                                             "All report-support tables"])
with peers:
    V.peers("ch13")
    render_outputs("13_peers", heading="")
with analysts:
    V.analysts()
    render_outputs("13_analysts", heading="")
with quarterly:
    V.quarterly()
    render_outputs("13_qtr", heading="")
with other:
    render_outputs("13", heading="Report exhibits")
    render_outputs("13_text", heading="Report text exports")
    render_outputs("13_dec", heading="Contested analytical choices")
    render_outputs("13_p21", heading="Report exhibit tables")
