"""Research intelligence dashboard (Output C). Run: streamlit run dashboard/app.py

Landing page plus one page per blueprint chapter (1-12) and chapter 4 subpages 4.0-4.7 (CLAUDE.md section 7).
Reads docs/RESEARCH_BLUEPRINT.md and outputs/manifests/; computes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from _common import ROOT, manifest, plain  # noqa: E402

from tzbank.reporting import blueprint as bp  # noqa: E402

st.set_page_config(page_title="Economics of Dominance and Rents", layout="wide")

PAGE_DIR = Path(__file__).parent / "pages"
SLUGS = {1: "01_concentration", 2: "02_credit_allocation", 3: "03_systemic_importance",
         4: "04_banking_economics/04_0_overview", 5: "05_persistence", 6: "06_funding_architecture",
         7: "07_capital_allocation", 8: "08_credit_risk", 9: "09_comparative_valuation", 10: "10_market_implied",
         11: "11_strategic_scenarios", 12: "12_usd_implementation"}
SUBS = {"4.0": "04_0a_dupont_attribution",
        "4.1": "04_1_revenue", "4.2": "04_2_funding", "4.3": "04_3_deployment", "4.4": "04_4_transactions",
        "4.5": "04_5_operating_leverage", "4.6": "04_6_risk", "4.7": "04_7_business_model_map"}

CHAPTERS = {c["number"]: c for c in bp.chapters(ROOT)}
SUB_TITLES = {s["number"]: s["title"] for s in bp.ch4_subsections(ROOT)}
PAGES = {n: st.Page(str(PAGE_DIR / f"{s}.py"), title=f"{n}. {CHAPTERS[n]['title']}", url_path=f"ch{n:02d}")
         for n, s in SLUGS.items()}
SUBPAGES = [st.Page(str(PAGE_DIR / "04_banking_economics" / f"{s}.py"), title=f"{k} {SUB_TITLES[k]}",
                    url_path=f"ch{k.replace('.', '_')}") for k, s in SUBS.items()]


@st.cache_data
def findings_pack() -> pd.DataFrame | None:
    """The latest outputs/findings/findings_pack_v*.csv (prompt 12a), or None."""
    files = sorted((ROOT / "outputs" / "findings").glob("findings_pack_v*.csv"),
                   key=lambda p: [int(x) for x in p.stem.split("_v")[-1].split(".")])  # v1.10 after v1.9
    if not files:
        return None
    p = pd.read_csv(files[-1], low_memory=False)
    p.attrs["file"] = files[-1].relative_to(ROOT).as_posix()
    return p


def landing() -> None:
    """Central proposition, the three questions and links to Parts I-III."""
    st.title("The Economics of Dominance and Rents")
    st.caption("Market Structure, Profitability Persistence and the Investment Value of CRDB and NMB")
    st.markdown(f"**Central proposition.** {bp.central_proposition(ROOT)}")
    st.markdown(
        "1. **Why** do NMB and CRDB earn returns above their cost of equity?\n"
        "2. **Will** they keep earning them, and for how long?\n"
        "3. **Is** the market paying an appropriate price for that persistence?")
    st.markdown("#### Headline findings")
    pack = findings_pack()
    if pack is None:
        st.info("No findings pack yet: run `python -m tzbank run all` (stage `findings`).")
    else:
        h = pack[pack["tier"] == "headline"]
        st.caption(f"From `{pack.attrs['file']}` (register {h['register_version'].iloc[0]}, input hash "
                   f"`{h['input_hash'].iloc[0]}`). Every figure here is a pack row; nothing is computed on this page.")

        def val(i: str) -> str:
            r = pack.loc[pack["id"].str.startswith(i + ".") | (pack["id"] == i)]
            if r.empty:
                return "n/a"
            v, u = r["value"].iloc[0], r["unit"].iloc[0]
            if u in ("text", "verdict"):
                return str(v)
            v = float(v)
            return f"{v:.0%}" if u == "ratio" else f"{v:,.0f}"

        for vd, lab in (("HIST", "31 Dec 2025 (point in time)"), ("CURR", "31 Oct 2026")):
            st.markdown(f"**Valuation at {lab}** (TZS per share; Monte Carlo base-centred)")
            cols = st.columns(2)
            for col, bank in zip(cols, ("NMB", "CRDB"), strict=True):
                k = f"{vd}.{bank}"
                col.metric(f"{bank}: value vs price {val(f'CH09.VALUATIONS.PRICE.{k}')}",
                           val(f"CH09.VALUATIONS.VALUE_PER_SHARE.{k}"),
                           f"rating {val(f'CH09.VALUATIONS.RATING.{k}')}", delta_color="off")
                col.caption(f"P(value > price) {val(f'CH11.MONTE_CARLO.PROB_VALUE_ABOVE_PRICE.{k}.base_centred_primary')}"
                            f"; MC P10-P90 {val(f'CH11.MONTE_CARLO.P10.{k}.base_centred_primary')} - "
                            f"{val(f'CH11.MONTE_CARLO.P90.{k}.base_centred_primary')}; expected 12-month return "
                            f"{val(f'CH09.VALUATIONS.EXPECTED_TOTAL_RETURN.{k}')}")
        v = pack[(pack["unit"] == "verdict") & pack["id"].str.endswith(".VERDICT") & (pack["chapter"] != "hypotheses")]
        st.markdown("**Hypothesis verdicts** (chapter registers; Interpretation)")
        st.dataframe(v.assign(hypothesis=v["id"].str.split(".").str[1].str.replace("_", "."),
                              value=v["value"].map(plain))[["chapter", "hypothesis", "value"]]
                     .rename(columns={"value": "verdict"}), hide_index=True)
        with st.expander(f"All {len(h)} headline figures (findings pack, headline tier)"):
            st.dataframe(h[["id", "value", "unit", "bank", "date", "chapter", "source_file"]], hide_index=True)
        hyp = manifest("hypotheses")
        if hyp:
            with st.expander("All registered hypotheses with their verdicts (hypotheses_all)"):
                for tab in hyp.get("tables", []):
                    st.caption(tab["description"])
                    st.dataframe(pd.read_csv(ROOT / tab["path"]), hide_index=True)
        part2 = manifest("part2")
        if part2:
            with st.expander("Part II synthesis: hypothesis status, rent sources, calibration"):
                for tab in part2.get("tables", []):
                    st.caption(tab["description"])
                    st.dataframe(pd.read_csv(ROOT / tab["path"]), hide_index=True)
    parts: dict[str, list[dict]] = {}
    for c in CHAPTERS.values():
        parts.setdefault(c["part"], []).append(c)
    for part, chs in parts.items():
        st.markdown(f"#### Part {part}")
        for c in chs:
            st.page_link(PAGES[c["number"]], label=f"Chapter {c['number']}. {c['title']}")


nav = st.navigation({
    "": [st.Page(landing, title="Overview", default=True)],
    "Part I. Architecture of dominance": [PAGES[n] for n in (1, 2, 3)],
    "Part II. Economics of banking profits": [PAGES[4], *SUBPAGES, *(PAGES[n] for n in (5, 6, 7, 8))],
    "Part III. Investment valuation": [PAGES[n] for n in (9, 10, 11, 12)],
    "Report support": [st.Page(str(PAGE_DIR / "13_report_support.py"), title="13. Report support (exhibits for the "
                                                                              "reports)", url_path="ch13"),
                       st.Page(str(PAGE_DIR / "14_market_extensions.py"), title="14. Market behaviour and valuation "
                                                                                 "extensions", url_path="ch14")],
})
if nav.title != "Overview":
    st.title(nav.title)
nav.run()
