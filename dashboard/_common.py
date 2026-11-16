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

try:  # private repository: the release tool's cleaner; the public copies are already cleaned by it
    sys.path.insert(0, str(ROOT / "release"))
    from sanitise import clean_line as _clean_line  # type: ignore[import-not-found]
except ImportError:  # public repository (no release/ folder)
    _clean_line = None

_IDS = re.compile(r"\s*\((?:D\d+|prompts? \d+[a-z]?|TO_VERIFY [A-Z]?\d+|(?:open )?issues? \d+)"
                  r"(?:[,;/ ]+(?:D\d+|prompts? \d+[a-z]?|(?:open )?issues? \d+))*\)")


def plain(text: str) -> str:
    """On-screen text without internal decision ids, prompt numbers or module names (plain labels only)."""
    if not isinstance(text, str):
        return text
    out = "\n".join(_clean_line(ln) for ln in text.split("\n")) if _clean_line else text
    return _IDS.sub("", out)


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


def render_outputs(chapter: str, *, heading: str = "Published tables, figures and notes") -> None:
    """Tables, figures and notes listed in the chapter manifest, each table with a CSV download, or 'not yet built'."""
    m = manifest(chapter)
    if not m:
        st.info("Not yet built: no outputs in `outputs/manifests/` for this page.")
        return
    paths = tuple([t["path"] for t in m.get("tables", [])] + [f["path"] for f in m.get("figures", [])])
    st.caption(f"Run-log ID: `{run_log_for(paths) or 'unknown'}`")
    if heading:
        st.markdown(f"#### {heading}")
    figs = m.get("figures", [])
    if figs:
        with st.expander(f"Report figures as published ({len(figs)})"):
            for fig in figs:
                st.markdown(f"**{plain(fig['description'])}**")
                show_png(ROOT / fig["path"], f"Evidence class: {fig['evidence_class']}")
                if fig.get("data") and (ROOT / fig["data"]).exists():
                    st.download_button("Download the plotted data (CSV)", (ROOT / fig["data"]).read_bytes(),
                                       file_name=Path(fig["data"]).name, mime="text/csv", key=f"dlf_{fig['path']}")
    tabs = m.get("tables", [])
    if tabs:
        with st.expander(f"Published tables ({len(tabs)})"):
            for tab in tabs:
                st.markdown(f"**{plain(tab['description'])}**")
                d = read_table(tab["path"])
                st.dataframe(d, hide_index=True)
                st.caption(f"Evidence class: {tab['evidence_class']} · Source table: {Path(tab['path']).name}")
                st.download_button("Download this table (CSV)", d.to_csv(index=False).encode("utf-8"),
                                   file_name=Path(tab["path"]).name, mime="text/csv", key=f"dlt_{tab['path']}")
    notes = m.get("notes", [])
    for note in notes if isinstance(notes, list) else [notes]:
        if note and (ROOT / note).exists():
            with st.expander("Notes: interpretation and limitations"):
                st.markdown(plain((ROOT / note).read_text(encoding="utf-8")))


@st.cache_data(show_spinner=False)
def read_table(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel, low_memory=False)


def render_chapter(number: int, explore=None) -> None:
    """Research question, hypotheses, interactive views, published outputs and limitations for one chapter."""
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
    if explore:
        st.markdown("#### Explore the results")
        explore()
    render_outputs(f"{number:02d}")
    lim = limitations_for(number)
    if len(lim):
        st.markdown("#### Limitations")
        st.dataframe(lim.map(plain), hide_index=True)


def render_subsection(number: str, explore=None) -> None:
    """Chapter 4 subsection page (4.0-4.7)."""
    sub = {s["number"]: s for s in bp.ch4_subsections(ROOT)}[number]
    hyp = bp.hypotheses(ROOT).get(4, {})
    st.markdown(f"**Investigation.** {sub['investigation']}")
    st.markdown(f"**Core measures.** {sub['measures']}")
    if hyp:
        st.markdown("**Chapter 4 hypotheses**")
        st.markdown("\n".join(f"- {h}" for h in hyp["hypotheses"]))
    st.divider()
    if explore:
        st.markdown("#### Explore the results")
        explore()
    render_outputs(f"04_{number.replace('.', '_')}")
    lim = limitations_for(4)
    if len(lim):
        st.markdown("#### Limitations")
        st.dataframe(lim.map(plain), hide_index=True)
