"""Shared rendering for dashboard pages (Output C, CLAUDE.md section 7).

The dashboard is the interface, not the engine: it reads blueprint text and files listed in
outputs/manifests/*.json and displays them. It computes no financial quantity.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing tzbank (Streamlit Cloud)

from tzbank.config import find_root  # noqa: E402
from tzbank.reporting import blueprint as bp  # noqa: E402

ROOT = find_root()
OUTPUTS = ROOT / "outputs"


@st.cache_data
def manifest(chapter: str) -> dict | None:
    """outputs/manifests/ch<NN>.json, or None when the chapter is not built."""
    f = OUTPUTS / "manifests" / f"ch{chapter}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_data
def run_log_for(paths: tuple[str, ...]) -> str | None:
    """Latest run-log ID whose outputs include any of ``paths``."""
    best = None
    for f in sorted((OUTPUTS / "run_logs").glob("*.json")) + sorted((OUTPUTS / "run_logs" / "latest").glob("*.json")):
        log = json.loads(f.read_text(encoding="utf-8"))
        if set(paths) & set(log.get("outputs", [])):
            best = log.get("run_id", f.stem)
    return best


def limitations_for(chapter: int) -> pd.DataFrame:
    """Blueprint section 7 limitations whose 'Disclosed in' mentions the chapter."""
    tabs = [t for t in bp.tables(bp.sections(ROOT)["7"]) if t and "Limitation" in t[0]]
    rows = []
    for r in tabs[0] if tabs else []:
        where = r.get("Disclosed in", "")
        nums = {int(n) for n in re.findall(r"\b(\d{1,2})\b", where)}
        if chapter in nums or "All" in where:
            rows.append({k: r[k] for k in ["#", "Limitation", "Effect", "Mitigation"] if k in r})
    return pd.DataFrame(rows)


def show_png(f, caption: str) -> None:
    """Embed a figure inline (base64) instead of through Streamlit's media endpoint, which fails on this runtime
    (ASGI 'http.response.start' error, prompt 12a); display only."""
    import base64

    b64 = base64.b64encode(f.read_bytes()).decode("ascii")
    st.markdown(f'<img src="data:image/png;base64,{b64}" style="max-width:100%" alt="{f.name}">',
                unsafe_allow_html=True)
    st.caption(caption)


def render_outputs(chapter: str) -> None:
    """Show tables, figures and notes listed in the chapter manifest, or 'not yet built'."""
    m = manifest(chapter)
    if not m:
        st.info("Not yet built: no outputs in `outputs/manifests/` for this page.")
        return
    st.caption(f"Status: {m.get('status', '')} · module `{m.get('module', '')}`")
    paths = tuple([t["path"] for t in m.get("tables", [])] + [f["path"] for f in m.get("figures", [])])
    st.caption(f"Run-log ID: `{run_log_for(paths) or 'unknown'}`")
    for fig in m.get("figures", []):
        st.subheader(fig["description"])
        show_png(ROOT / fig["path"], f"Evidence class: {fig['evidence_class']}")
    for tab in m.get("tables", []):
        st.subheader(tab["description"])
        st.caption(f"`{tab['path']}` · each row carries its evidence_class")
        st.dataframe(pd.read_csv(ROOT / tab["path"]), hide_index=True)
    for note in m.get("notes", []):
        with st.expander(f"Notes, interpretation and limitations: {note}"):
            st.markdown((ROOT / note).read_text(encoding="utf-8"))


def render_chapter(number: int) -> None:
    """Research question, hypotheses, outputs and limitations for one chapter."""
    ch = {c["number"]: c for c in bp.chapters(ROOT)}[number]
    hyp = bp.hypotheses(ROOT).get(number, {})
    st.caption(f"Part {ch['part']}")
    st.markdown(f"**Research question.** {ch['question']}")
    st.markdown(f"**Primary output.** {ch['output']}")
    if hyp:
        st.markdown("**Hypotheses**")
        st.markdown("\n".join(f"- {h}" for h in hyp["hypotheses"]))
        st.markdown(f"**Test.** {hyp['test']}")
    st.divider()
    st.markdown("#### Results")
    render_outputs(f"{number:02d}")
    lim = limitations_for(number)
    if len(lim):
        st.markdown("#### Limitations (blueprint section 7)")
        st.dataframe(lim, hide_index=True)


def render_subsection(number: str) -> None:
    """Chapter 4 subsection page (4.0-4.7)."""
    sub = {s["number"]: s for s in bp.ch4_subsections(ROOT)}[number]
    hyp = bp.hypotheses(ROOT).get(4, {})
    st.markdown(f"**Investigation.** {sub['investigation']}")
    st.markdown(f"**Core measures.** {sub['measures']}")
    if hyp:
        st.markdown("**Chapter 4 hypotheses**")
        st.markdown("\n".join(f"- {h}" for h in hyp["hypotheses"]))
    st.divider()
    st.markdown("#### Results")
    render_outputs(f"04_{number.replace('.', '_')}")
    lim = limitations_for(4)
    if len(lim):
        st.markdown("#### Limitations (blueprint section 7)")
        st.dataframe(lim, hide_index=True)
