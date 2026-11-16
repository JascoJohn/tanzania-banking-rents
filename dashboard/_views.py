"""Interactive views for each chapter page (Output C, CLAUDE.md section 7).

One function per page. Each one filters, selects and re-plots published tables (``outputs/tables``,
``outputs/valuation``); none estimates anything. A chart's download button writes the frame the chart draws.
"""

from __future__ import annotations

import _explore as X
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================================================================
# shared references
# ============================================================================================================================


def sector_roa(metric: str, years: tuple[int, int]) -> pd.DataFrame | None:
    """BoT commercial-bank sector ROA (published in ch01_profit_vs_assets) as a reference on ROA charts."""
    if metric != "roa":
        return None
    d = X.table("ch01_profit_vs_assets")
    d = d[d["year"].between(*years)]
    out = pd.DataFrame({"bank": "Sector (BoT, June year-end)", "year": d["year"], "value": d["sector_roa"],
                        "evidence_class": d["evidence_class"]})
    out.attrs["source"] = "ch01_profit_vs_assets"
    return out


def smaller_deposit_cost(metric: str, years: tuple[int, int]) -> pd.DataFrame | None:
    """Median deposit cost of the smaller listed banks (published in ch03_deposit_cost)."""
    if metric != "cost_of_deposits":
        return None
    d = X.table("ch03_deposit_cost")
    d = d[d["fiscal_year"].between(*years)]
    out = pd.DataFrame({"bank": "Smaller listed banks (median)", "year": d["fiscal_year"],
                        "value": d["smaller_median"], "evidence_class": d["evidence_class"]})
    out.attrs["source"] = "ch03_deposit_cost"
    return out


def mckinsey(metric: str) -> list[tuple[float, str]]:
    """McKinsey Global Banking Annual Review 2025 global averages, from the published benchmark table."""
    b = X.table("ch13_benchmarks")
    b = b[b["document"].str.contains("McKinsey", na=False) & (b["scope"] == "global banking")]
    key = {"roe": "ROE", "pb": "price-to-book"}.get(metric)
    r = b[b["metric"] == key]
    if r.empty:
        return []
    return [(float(r["value"].iloc[0]), f"Global banks, McKinsey 2025 ({r['period'].iloc[0]})")]


# ============================================================================================================================
# Part I
# ============================================================================================================================


def ch01() -> None:
    d = X.table("ch01_shares")
    with st.container(border=True):
        st.markdown("**NMB and CRDB shares of the banking sector**")
        c1, c2 = st.columns([2, 3])
        meas = c1.segmented_control("Measure", sorted(d["measure"].unique()), default="assets", key="m_ch01") or "assets"
        yr = X.year_slider("ch01", d["year"])
        s = d[(d["measure"] == meas) & d["year"].between(*yr)]
        shown = s.melt(id_vars=["year", "evidence_class"], value_vars=["nmb_share", "crdb_share", "cr2"],
                       var_name="series", value_name="share")
        shown["series"] = shown["series"].map({"nmb_share": "NMB", "crdb_share": "CRDB", "cr2": "NMB + CRDB (CR2)"})
        fig = go.Figure()
        for nm, col in (("NMB", X.HOUSE["NMB"]), ("CRDB", X.HOUSE["CRDB"]), ("NMB + CRDB (CR2)", X.HOUSE["sector"])):
            r = shown[shown["series"] == nm]
            fig.add_trace(go.Scatter(x=r["year"], y=r["share"], name=nm, mode="lines+markers", line={"color": col}))
        X.style(fig, y_unit="pct", y_title=f"Share of sector {meas}")
        X.show_chart(fig, shown, "ch01_shares", "ch01_shares", "Sector: BoT commercial banks, December")
    p = X.table("ch01_profit_vs_assets")
    with st.container(border=True):
        st.markdown("**Profit share against asset share (NMB + CRDB)**")
        shown = p[["year", "asset_share", "profit_share_pretax", "sector_roa", "cr2_roa_pretax", "cr2_roa_posttax",
                   "evidence_class"]]
        pick = st.multiselect("Series", ["asset_share", "profit_share_pretax", "sector_roa", "cr2_roa_pretax",
                                         "cr2_roa_posttax"], default=["asset_share", "profit_share_pretax"],
                              key="s_ch01p", format_func=lambda c: {
                                  "asset_share": "Asset share", "profit_share_pretax": "Profit share (pre-tax)",
                                  "sector_roa": "Sector ROA (BoT)", "cr2_roa_pretax": "NMB + CRDB ROA (pre-tax)",
                                  "cr2_roa_posttax": "NMB + CRDB ROA (post-tax)"}[c])
        fig = go.Figure()
        for i, c in enumerate(pick):
            fig.add_trace(go.Scatter(x=shown["year"], y=shown[c], name=c.replace("_", " "), mode="lines+markers",
                                     line={"color": X.OTHERS[i]}))
        X.style(fig, y_unit="pct")
        X.show_chart(fig, shown[["year", *pick, "evidence_class"]], "ch01_profit", "ch01_profit_vs_assets")


def ch02() -> None:
    d = X.table("ch02_loan_mix")
    with st.container(border=True):
        st.markdown("**Loan book by sector: NMB, CRDB and the commercial-bank system**")
        c1, c2 = st.columns(2)
        books = c1.multiselect("Loan books", list(d["book"].unique()), default=["NMB", "CRDB", "BoT commercial banks"],
                               key="b_ch02")
        cats = sorted(d["category"].unique())
        cat = c2.selectbox("Sector", cats, index=cats.index("agriculture") if "agriculture" in cats else 0, key="c_ch02")
        yr = X.year_slider("ch02", d["fiscal_year"])
        shown = d[d["book"].isin(books) & (d["category"] == cat) & d["fiscal_year"].between(*yr)][
            ["book", "fiscal_year", "category", "share", "evidence_class"]]
        cmap = {"NMB": X.HOUSE["NMB"], "CRDB": X.HOUSE["CRDB"], "BoT commercial banks": X.HOUSE["sector"],
                "NMB + CRDB": X.HOUSE["global"]}
        fig = go.Figure([go.Scatter(x=r["fiscal_year"], y=r["share"], name=b, mode="lines+markers",
                                    line={"color": cmap.get(b)}) for b, r in shown.groupby("book")])
        X.style(fig, y_unit="pct", y_title=f"Share of loans: {cat}")
        X.show_chart(fig, shown, "ch02_mix", "ch02_loan_mix")
    s = X.table("ch02_dissimilarity")
    with st.container(border=True):
        st.markdown("**How different are the loan books? (dissimilarity index)**")
        fig = go.Figure([go.Scatter(x=r["fiscal_year"], y=r["dissimilarity"], name=c, mode="lines+markers",
                                    line={"color": X.OTHERS[i]}) for i, (c, r) in enumerate(s.groupby("comparison"))])
        X.style(fig, y_unit="pct", y_title="Dissimilarity (0 = identical mix)")
        X.show_chart(fig, s, "ch02_dis", "ch02_dissimilarity")


def ch03() -> None:
    X.explore_panel("ch03_buf", "ch03_capital_buffers",
                    ["binding_headroom", "tier1_ratio", "total_capital_ratio", "tier1_headroom", "total_headroom"],
                    title="Capital buffers: listed banks, FY2015-2025")
    d = X.panel("ch03_pb_roe")
    with st.container(border=True):
        st.markdown("**Price-to-book against ROE, listed banks**")
        sel = X.bank_picker("ch03_pb", [b for b in X.TZ_PANEL if b in set(d["bank"])],
                            default=[b for b in X.TZ_PANEL if b in set(d["bank"])])
        yr = X.year_slider("ch03_pb", d["year"])
        shown = d[d["bank"].isin(sel) & d["year"].between(*yr) & d["pb"].notna()][
            ["bank", "year", "roe", "pb", "systemic", "evidence_class"]]
        cmap = X.colour_map(sel)
        fig = go.Figure([go.Scatter(x=r["roe"], y=r["pb"], name=b, mode="markers", text=r["year"],
                                    marker={"color": cmap[b], "size": 9},
                                    hovertemplate="%{text}: ROE %{x:.1%}, P/B %{y:.2f}<extra>" + b + "</extra>")
                         for b, r in shown.groupby("bank")])
        X.style(fig, y_unit="x", x_title="Return on equity", y_title="Price-to-book").update_layout(hovermode="closest")
        fig.update_xaxes(tickformat=".0%")
        X.show_chart(fig, shown, "ch03_pb", "ch03_pb_roe")
    X.explore_panel("ch03_dep", "ch06_deposits", ["cost_of_deposits"], title="Deposit cost: systemic against smaller "
                    "listed banks", reference=smaller_deposit_cost)


# ============================================================================================================================
# Part II: chapter 4 and subsections
# ============================================================================================================================
PROFIT = ["roe", "roa", "nii_to_avg_assets", "non_interest_income_share", "cost_to_income", "cost_of_risk", "leverage"]


def ch04() -> None:
    X.explore_panel("ch04", "ch04_dupont_panel", PROFIT, title="Profitability and its drivers, listed banks",
                    reference=sector_roa, note="Sector reference: BoT commercial-bank ROA where ROA is selected")


def ch04_0a() -> None:
    X.explore_panel("ch040c", "ch04_4_0_components",
                    ["roe_tree", "y_iea", "c_fund", "nii_ta", "nonii_ta", "opex_ta", "imp_ta", "tax_rate", "ta_e"],
                    title="ROE tree components by bank", default_metric="roe_tree")
    g = X.table("ch04_4_0_gap_attribution")
    with st.container(border=True):
        st.markdown("**What explains the NMB - CRDB ROE gap? Contribution by component**")
        yr = X.year_slider("ch040g", g["fiscal_year"])
        shown = g[g["fiscal_year"].between(*yr)][["fiscal_year", "component", "subsection", "contribution", "gap",
                                                   "evidence_class"]]
        fig = go.Figure([go.Bar(x=r["fiscal_year"], y=r["contribution"], name=c, marker_color=X.OTHERS[i % 9])
                         for i, (c, r) in enumerate(shown.groupby("subsection"))])
        gap = shown.drop_duplicates("fiscal_year")
        fig.add_trace(go.Scatter(x=gap["fiscal_year"], y=gap["gap"], name="ROE gap (NMB - CRDB)", mode="lines+markers",
                                 line={"color": X.TEXT}))
        X.style(fig, y_unit="pct", y_title="ROE points").update_layout(barmode="relative")
        X.show_chart(fig, shown, "ch040_gap", "ch04_4_0_gap_attribution")


def ch04_1() -> None:
    X.explore_panel("ch041", "ch04_4_1_composition",
                    ["nii_share", "nonii_share", "fee_share_of_nonii", "fee_share_of_income", "total_income",
                     "growth_total_income", "growth_nii", "growth_nonii"], title="Revenue composition, listed banks")


def ch04_2() -> None:
    X.explore_panel("ch042", "ch04_4_2_funding", ["cost_of_deposits", "cost_of_funds", "casa_share", "gdfa_spread",
                                                  "gdfa_proxy_to_avg_assets", "avg_deposits"],
                    title="Funding economics, listed banks", reference=smaller_deposit_cost,
                    note="Gross deposit funding advantage is a proxy; rent is discussed with the net franchise "
                         "(chapter 6)")


def ch04_3() -> None:
    X.explore_panel("ch043", "ch04_4_3_deployment", ["loan_yield", "securities_yield", "iea_yield", "loans_to_assets",
                                                     "securities_to_assets", "loans_to_deposits"],
                    title="Asset deployment, listed banks")
    d = X.panel("ch04_4_3_sector_book")
    with st.container(border=True):
        st.markdown("**Loan book by sector, NMB and CRDB**")
        banks = X.two_bank_picker("ch043s")
        secs = sorted(d["sector"].unique())
        sec = st.multiselect("Sectors", secs, default=secs[:4], key="sec_ch043")
        shown = d[d["bank"].isin(banks) & d["sector"].isin(sec)][["bank", "year", "sector", "share", "evidence_class"]]
        fig = go.Figure()
        for (b, s), r in shown.groupby(["bank", "sector"]):
            fig.add_trace(go.Scatter(x=r["year"], y=r["share"], name=f"{b}: {s}", mode="lines+markers",
                                     line={"color": X.OTHERS[sec.index(s) % 9], "dash": "solid" if b == "NMB" else "dot"}))
        X.style(fig, y_unit="pct", y_title="Share of loans")
        X.show_chart(fig, shown, "ch043_sector", "ch04_4_3_sector_book", "NMB solid lines, CRDB dotted")


def ch04_4() -> None:
    X.explore_panel("ch044", "ch04_4_4_transactions", ["fee_to_avg_assets", "fx_to_avg_assets", "fee_share_of_nonii",
                                                       "fee_income"], title="Transaction franchise, NMB and CRDB")


def ch04_5() -> None:
    d = X.panel("ch04_4_5_operating_leverage")
    extra = [c for c in d.columns if c.endswith(("_per_employee", "_per_branch", "_per_agent"))]
    X.explore_panel("ch045", "ch04_4_5_operating_leverage",
                    ["cost_to_income", "opex_to_avg_assets", "staff_share_of_opex", "incremental_margin", *extra],
                    title="Operating leverage and productivity, listed banks")
    e = X.panel("ch04_4_5_scale_elasticity")
    with st.container(border=True):
        st.markdown("**Scale elasticity of cost (below 1 = scale economies)**")
        shown = e[["bank", "years", "n", "elasticity", "ci_low", "ci_high", "reading", "evidence_class"]]
        cmap = X.colour_map(shown["bank"])
        fig = go.Figure(go.Scatter(x=shown["bank"], y=shown["elasticity"], mode="markers",
                                   marker={"size": 12, "color": [cmap[b] for b in shown["bank"]]},
                                   error_y={"type": "data", "symmetric": False,
                                            "array": shown["ci_high"] - shown["elasticity"],
                                            "arrayminus": shown["elasticity"] - shown["ci_low"]}))
        fig.add_hline(y=1, line_dash="dash", line_color=X.HOUSE["global"])
        X.style(fig, y_title="Cost elasticity to assets").update_layout(hovermode="closest")
        X.show_chart(fig, shown, "ch045_scale", "ch04_4_5_scale_elasticity", "Bars: published interval bounds")


def ch04_6() -> None:
    X.explore_panel("ch046", "ch04_4_6_risk", ["cost_of_risk", "npl_ratio", "coverage", "nim", "risk_adjusted_nim",
                                               "return_on_rwa", "impairment_to_avg_assets"],
                    title="Risk and risk-adjusted margins, listed banks")


def ch04_7() -> None:
    d = X.table("ch04_business_model_map")
    with st.container(border=True):
        st.markdown("**Business model map**")
        b = st.segmented_control("Bank", ["NMB", "CRDB", "both"], default=None, key="b_ch047",
                                 format_func=lambda x: "Shared" if x == "both" else x)
        shown = d if b is None else d[d["bank"] == b]
        X.show_table(shown, "ch047_map", "ch04_business_model_map")


# ============================================================================================================================
# Part II: chapters 5-8
# ============================================================================================================================


SPECS = {  # plain labels for the persistence specifications (ch05_persistence.specification)
    "ar1_benchmark": "AR(1) benchmark, all years", "ar1_break_ifrs9": "AR(1), IFRS 9 dummy from FY2018",
    "ar1_break_episode": "AR(1), 2015-18 episode dummy", "mean_reversion_peer_mean": "AR(1) vs other banks' mean",
    "mean_reversion_bot_sector": "AR(1) vs BoT sector (2019-25)", "robust_coe_minus_200bp": "AR(1), COE - 200 bp",
    "robust_coe_plus_200bp": "AR(1), COE + 200 bp", "robust_excl_fy2018": "AR(1) without FY2018",
    "robust_d41_adjusted_2008_12": "AR(1), term-spread-adjusted COE 2008-12",
    "robust_coe_2025_vintage": "AR(1), single COE vintage", "robust_beta_2012_replaced": "AR(1), 2012 beta replaced",
    "robust_original_vintage": "AR(1), originally published figures", "robust_bank_basis": "AR(1), Bank-solo ROE",
    "ar1_break_episode_excl_fy2018": "AR(1), episode dummy, without FY2018",
    "through_cycle_mean": "Through-the-cycle mean (no model)", "pooled_fixed_effects": "Pooled fixed-effects AR(1)",
}


def ch05() -> None:
    X.explore_panel("ch05", "ch05_excess_returns", ["er", "roe", "coe"], title="Excess returns: ROE minus cost of "
                    "equity, listed banks", note="Years without a computable ROE are excluded, not filled")
    p = X.panel("ch05_persistence")
    with st.container(border=True):
        st.markdown("**Persistence estimates by bank and specification**")
        c1, c2 = st.columns(2)
        banks = [b for b in X.TZ_PANEL if b in set(p["bank"])] + ["pooled"]
        sel = c1.multiselect("Banks", banks, default=["NMB", "CRDB"], key="b_ch05p")
        specs = list(p["specification"].unique())
        spec = c2.multiselect("Specifications", specs, default=specs[:3], key="s_ch05p",
                              format_func=lambda s: SPECS.get(s, s))
        shown = p[p["bank"].isin(sel) & p["specification"].isin(spec)][
            ["bank", "specification", "years", "n", "lambda", "lambda_lo", "lambda_hi", "er_star", "er_star_lo",
             "er_star_hi", "half_life_status", "half_life", "evidence_class"]]
        cmap = X.colour_map(sel)
        fig = go.Figure()
        for b, r in shown.groupby("bank", sort=False):
            fig.add_trace(go.Scatter(x=r["specification"].map(lambda s: SPECS.get(s, s)), y=r["lambda"], name=b,
                                     mode="markers",
                                     marker={"size": 11, "color": cmap.get(b, X.HOUSE["global"])},
                                     error_y={"type": "data", "symmetric": False, "array": r["lambda_hi"] - r["lambda"],
                                              "arrayminus": r["lambda"] - r["lambda_lo"]}))
        fig.add_hline(y=1, line_dash="dash", line_color=X.HOUSE["global"], annotation_text="no mean reversion")
        X.style(fig, y_title="Persistence (lambda), 90% interval").update_layout(hovermode="closest",
                                                                                 scattermode="group")
        X.show_chart(fig, shown, "ch05_persist", "ch05_persistence",
                     "A half-life is quoted only when 0 < lambda < 1 and the interval excludes 1")
    s = X.panel("ch05_ifrs9_sensitivity")
    with st.container(border=True):
        st.markdown("**Sensitivity: persistence without FY2018 (IFRS 9 transition year)**")
        X.show_table(s[["bank", *[c for c in s.columns if c not in ("bank", "bank_id")]]], "ch05_ifrs9",
                     "ch05_ifrs9_sensitivity")


def ch06() -> None:
    X.explore_panel("ch06d", "ch06_deposits", ["cost_of_deposits", "casa_share", "share_current", "share_savings",
                                               "share_time", "deposit_growth", "avg_deposits"],
                    title="Deposits: cost and mix, listed banks", reference=smaller_deposit_cost)
    X.explore_panel("ch06f", "ch06_franchise", ["net_franchise_central_roe_pts", "net_franchise_central_pct_assets",
                                                "net_franchise_low_roe_pts", "net_franchise_high_roe_pts"],
                    title="Net deposit franchise contribution (low / central / high)")
    X.explore_panel("ch06l", "ch06_liquidity", ["ldr", "wholesale_share", "loan_growth", "deposit_growth", "growth_gap",
                                                "funding_gap_cost_roe"], title="Liquidity and funding gap")


def ch07() -> None:
    X.explore_panel("ch07c", "ch07_capital", ["tier1_ratio", "total_capital_ratio", "equity_to_assets",
                                              "tier1_headroom", "total_headroom", "binding_headroom", "rwa_growth"],
                    title="Capital ratios and headroom, listed banks")
    X.explore_panel("ch07p", "ch07_payout", ["payout_total", "payout_ordinary", "dps_ordinary"],
                    title="Dividend payout, listed banks")
    r = X.panel("ch07_rore")
    with st.container(border=True):
        st.markdown("**Return on retained earnings by window length**")
        c1, c2 = st.columns(2)
        sel = c1.multiselect("Banks", [b for b in X.TZ_PANEL if b in set(r["bank"])], default=["NMB", "CRDB"],
                             key="b_ch07r")
        k = c2.segmented_control("Window (years)", sorted(r["k"].unique()), default=sorted(r["k"].unique())[0],
                                 key="k_ch07r")
        shown = r[r["bank"].isin(sel) & (r["k"] == k) & r["rore_primary"].notna()][
            ["bank", "year", "k", "window", "rore_primary", "rore_adjusted", "raise_flag", "evidence_class"]]
        cmap = X.colour_map(sel)
        fig = go.Figure([go.Scatter(x=g["year"], y=g["rore_primary"], name=b, mode="lines+markers",
                                    line={"color": cmap[b]}) for b, g in shown.groupby("bank")])
        X.style(fig, y_unit="pct", y_title="RORE (attributable earnings, retained equity)")
        X.show_chart(fig, shown, "ch07_rore", "ch07_rore", "Window ends in the year shown")


def ch08() -> None:
    X.explore_panel("ch08c", "ch08_credit_history", ["cost_of_risk", "npl_ratio", "coverage", "npl_ratio_reported",
                                                     "coverage_reported"], title="Credit history, listed banks",
                    note="IFRS 9 from FY2018: NPL definitions change (see the accounting regime in the data)")
    X.explore_panel("ch08e", "ch08_capital_efficiency", ["return_on_rwa", "risk_adjusted_rorwa", "ppop_to_rwa",
                                                         "rwa_density"], title="Capital efficiency, listed banks")


# ============================================================================================================================
# Part III and peers
# ============================================================================================================================


def peers(key: str) -> None:
    """African peers against NMB and CRDB: cross-section by valuation date and ROE paths with/without one-offs."""
    t = X.table("ch13_peers_table")
    names = X.bank_full_names()
    with st.container(border=True):
        st.markdown("**NMB and CRDB against African listed banks**")
        vd = X.date_picker(f"{key}_pt")
        d = t[t["valuation"] == vd]
        all_b = list(d["bank"])
        sel = st.multiselect("Banks", all_b, default=all_b, key=f"b_{key}_pt",
                             format_func=lambda b: f"{b} ({names[b]})" if b in names and not names[b].startswith(b) else b)
        d = d[d["bank"].isin(sel)]
        m = X.metric_picker(f"{key}_pt", ["pb", "pe", "roe", "coe", "roe_minus_coe", "justified_pb", "capital_ratio",
                                          "npl_ratio"])
        shown = d[["bank", "country", m, "flagged_year", "evidence_class"]].reset_index(drop=True)

        def txt(r: pd.Series) -> str:
            if pd.isna(r[m]):
                return "no traded price" if m in ("pb", "pe") else "not published"
            return X.fmt(r[m], X.unit(m)) + (" (one-off year)" if r["flagged_year"] is True else "")

        lab = shown.apply(txt, axis=1)
        cmap = X.colour_map(shown["bank"])
        fig = go.Figure(go.Bar(x=shown["bank"], y=shown[m].fillna(0), text=lab, textposition="outside",
                               marker_color=[cmap[b] for b in shown["bank"]], hovertext=lab, hoverinfo="x+text"))
        for y, nm in mckinsey(m):
            fig.add_hline(y=y, line_dash="dash", line_color=X.HOUSE["global"], annotation_text=nm,
                          annotation_position="top right")
        X.style(fig, y_unit=X.unit(m), y_title=X.label(m)).update_layout(hovermode="closest", showlegend=False)
        X.show_chart(fig, shown.assign(display=lab), f"{key}_peers", ["ch13_peers_table", "ch13_benchmarks"],
                     f"Peers' figures at {X.DATES[vd]}; McKinsey global average shown where the measure matches")
    o = X.table("ch13_peers_roe_oneoffs")
    with st.container(border=True):
        st.markdown("**Peer ROE over time, with or without one-off items**")
        c1, c2 = st.columns([3, 2])
        pb = sorted(o["bank"].unique())
        sel = c1.multiselect("Peers", pb, default=pb, key=f"b_{key}_po",
                             format_func=lambda b: f"{b} ({names.get(b, b)})")
        basis = c2.segmented_control("ROE basis", ["As reported", "Without flagged one-offs"], default="As reported",
                                     key=f"o_{key}_po") or "As reported"
        col = "roe" if basis == "As reported" else "roe_ex_oneoffs"
        shown = o[o["bank"].isin(sel)][["bank", "fy", col, "oneoffs_pretax", "policy", "evidence_class"]]
        shown = shown.assign(one_off_year=shown["oneoffs_pretax"].fillna(0) != 0)
        cmap = X.colour_map(sel)
        fig = go.Figure()
        for b, r in shown.groupby("bank"):
            fig.add_trace(go.Scatter(x=r["fy"], y=r[col], name=b, mode="lines+markers", line={"color": cmap[b]},
                                     marker={"symbol": ["star" if f else "circle" for f in r["one_off_year"]],
                                             "size": [14 if f else 7 for f in r["one_off_year"]]}))
        for y, nm in mckinsey("roe"):
            fig.add_hline(y=y, line_dash="dash", line_color=X.HOUSE["global"], annotation_text=nm,
                          annotation_position="top right")
        X.style(fig, y_unit="pct", y_title=f"ROE ({basis.lower()})")
        X.show_chart(fig, shown, f"{key}_oneoffs", ["ch13_peers_roe_oneoffs", "ch13_benchmarks"],
                     "Stars mark the flagged one-off years (GCB government-bond impairment 2022-23; GTCO FX "
                     "revaluation gains 2023-24)")


def valuation_headline(key: str, vd: str, banks: list[str]) -> None:
    v = X.table("ch09_valuations")
    cols = st.columns(len(banks))
    for c, b in zip(cols, banks, strict=True):
        r = v[(v["valuation"] == vd) & (v["bank"] == b)].iloc[0]
        c.metric(f"{b}: value per share (TZS)", f"{r['value_per_share']:,.0f}",
                 f"price {r['price']:,.0f} · rating {r['rating']}", delta_color="off", border=True)
        c.caption(f"Expected 12-month return {r['expected_total_return']:.1%}; cost of equity {r['coe']:.1%}; "
                  f"bear {r['bear_value_per_share']:,.0f} / bull {r['bull_value_per_share']:,.0f}")
    st.caption("Source table: ch09_valuations.csv · Evidence class: Estimated · Prices and values in TZS per share "
               "as at the valuation date (NMB before its 1:10 split on 31 Dec 2025)")


def ch09() -> None:
    c1, c2 = st.columns(2)
    with c1:
        banks = X.two_bank_picker("ch09")
    with c2:
        vd = X.date_picker("ch09")
    valuation_headline("ch09", vd, banks)

    cfg = X.table("ch13_dec_configurations")
    with st.container(border=True):
        st.markdown("**The three valuation cases: value per share against price**")
        shown = cfg[(cfg["valuation"] == vd) & cfg["bank"].isin(banks)][
            ["bank", "configuration", "case", "er_star", "lambda", "value_per_share", "price", "value_to_price",
             "expected_return", "rating", "evidence_class"]]
        fig = go.Figure()
        for b, r in shown.groupby("bank", sort=False):
            fig.add_trace(go.Bar(x=r["case"], y=r["value_per_share"], name=f"{b} value", marker_color=X.HOUSE[b],
                                 text=r["rating"], textposition="outside"))
            fig.add_hline(y=float(r["price"].iloc[0]), line_dash="dash", line_color=X.HOUSE[b],
                          annotation_text=f"{b} price {r['price'].iloc[0]:,.0f}", annotation_position="top left")
        X.style(fig, y_unit="tzs", y_title="TZS per share").update_layout(barmode="group", hovermode="closest")
        X.show_chart(fig, shown, "ch09_cases", "ch13_dec_configurations", "Dashed lines: market price")

    w = X.table("ch13_dec_interp_weights")
    cw = X.table("ch13_dec_crossing_weights")
    with st.container(border=True):
        st.markdown("**Weight on the 2015-18 exceptional-episode reading: value against price**")
        shown = w[(w["valuation"] == vd) & w["bank"].isin(banks)][
            ["bank", "weight_on_B", "er_star", "value_per_share", "price", "value_to_price", "expected_return",
             "rating", "evidence_class"]]
        fig = go.Figure()
        for b, r in shown.groupby("bank", sort=False):
            fig.add_trace(go.Scatter(x=r["weight_on_B"], y=r["value_to_price"], name=b, mode="lines+markers",
                                     line={"color": X.HOUSE[b]}, text=r["rating"],
                                     hovertemplate="weight %{x:.0%}: value / price %{y:.2f} (%{text})"))
            c = cw[(cw["valuation"] == vd) & (cw["bank"] == b)]
            if len(c) and pd.notna(c["crossing_weight"].iloc[0]):
                fig.add_vline(x=float(c["crossing_weight"].iloc[0]), line_dash="dot", line_color=X.HOUSE[b],
                              annotation_text=f"{b} crossing {c['crossing_weight'].iloc[0]:.0%}")
        fig.add_hline(y=1, line_dash="dash", line_color=X.HOUSE["global"], annotation_text="value = price")
        X.style(fig, y_unit="x", x_title="Weight on the exceptional-episode reading", y_title="Value / price")
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(hovermode="closest")
        X.show_chart(fig, shown, "ch09_weights", "ch13_dec_interp_weights")
        crossing = cw[(cw["valuation"] == vd) & cw["bank"].isin(banks)][
            ["bank", "price", "er_star_A", "er_star_B", "crossing_weight", "status", "evidence_class"]]
        X.show_table(crossing, "ch09_crossing", "ch13_dec_crossing_weights",
                     "A weight on an interpretation of the past, not the probability of a crisis")

    rm = X.table("ch13_p21_rating_map")
    with st.container(border=True):
        st.markdown(f"**Rating map ({X.DATES['CURR']}): expected return against the rating thresholds**")
        shown = rm[rm["bank"].isin(banks)][["bank", "expected_return", "sell_threshold", "buy_threshold", "rating",
                                            "bear_downside", "crossing_weight", "evidence_class"]]
        fig = go.Figure()
        t0 = shown.iloc[0]
        fig.add_vrect(x0=-1, x1=t0["sell_threshold"], fillcolor="#f3d9d0", opacity=0.5, line_width=0,
                      annotation_text="Sell", annotation_position="top left")
        fig.add_vrect(x0=t0["sell_threshold"], x1=t0["buy_threshold"], fillcolor="#eeeeea", opacity=0.6,
                      line_width=0, annotation_text="Hold", annotation_position="top left")
        fig.add_vrect(x0=t0["buy_threshold"], x1=1, fillcolor="#d7ebdf", opacity=0.6, line_width=0,
                      annotation_text="Buy", annotation_position="top left")
        fig.add_trace(go.Scatter(x=shown["expected_return"], y=shown["bank"], mode="markers+text", text=shown["rating"],
                                 textposition="top center", marker={"size": 16, "color": [X.HOUSE[b] for b in
                                                                                          shown["bank"]]}))
        lo = min(-0.4, float(shown["expected_return"].min()) - 0.05)
        X.style(fig, height=260, x_title="Expected 12-month return").update_layout(showlegend=False,
                                                                                   hovermode="closest")
        fig.update_xaxes(tickformat=".0%", range=[lo, 0.4])
        X.show_chart(fig, shown, "ch09_rating", "ch13_p21_rating_map")

    g = X.table("ch09_sensitivity_grids")
    with st.container(border=True):
        st.markdown("**Sensitivity of value per share**")
        grids = {"coe_x_g": ("coe", "g", "Cost of equity", "Long-run growth"),
                 "coe_x_roe_star": ("coe", "roe_star", "Cost of equity", "Sustainable ROE"),
                 "nim_x_cor": ("nim_shift", "cor_shift", "NIM shift", "Cost-of-risk shift")}
        c1, c2 = st.columns(2)
        gk = c1.selectbox("Grid", list(grids), key="g_ch09",
                          format_func=lambda k: f"{grids[k][2]} x {grids[k][3]}")
        b = c2.segmented_control("Bank", banks, default=banks[0], key="gb_ch09") or banks[0]
        xc, yc, xl, yl = grids[gk]
        shown = g[(g["grid"] == gk) & (g["valuation"] == vd) & (g["bank"] == b)][
            ["bank", "valuation", xc, yc, "value_per_share", "evidence_class"]]
        piv = shown.pivot_table(index=yc, columns=xc, values="value_per_share")
        price = X.table("ch09_valuations").set_index(["valuation", "bank"]).loc[(vd, b), "price"]
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="RdYlGn", zmid=price,
                                   colorbar={"title": "TZS"},
                                   hovertemplate=f"{xl} %{{x:.2%}}<br>{yl} %{{y:.2%}}<br>value %{{z:,.0f}}<extra></extra>"))
        X.style(fig, x_title=xl, y_title=yl).update_layout(hovermode="closest")
        fig.update_xaxes(tickformat=".1%")
        fig.update_yaxes(tickformat=".1%")
        X.show_chart(fig, shown, "ch09_grid", "ch09_sensitivity_grids",
                     f"Colour centred on the price ({price:,.0f}): green above, red below")

    bands = X.table("ch13_valuation_bands")
    with st.container(border=True):
        st.markdown("**Price-to-book and price-to-earnings history**")
        mult = st.segmented_control("Multiple", ["pb", "pe"], default="pb", key="m_ch09b",
                                    format_func=X.label) or "pb"
        shown = bands[(bands["kind"] == "history") & bands["bank"].isin(banks)][["bank", "fiscal_year", mult,
                                                                                  "evidence_class"]]
        fig = go.Figure([go.Scatter(x=r["fiscal_year"], y=r[mult], name=b, mode="lines+markers",
                                    line={"color": X.HOUSE[b]}) for b, r in shown.groupby("bank")])
        X.style(fig, y_unit="x", y_title=X.label(mult))
        X.show_chart(fig, shown, "ch09_bands", "ch13_valuation_bands")

    peers("ch09")


def ch10() -> None:
    c1, c2 = st.columns(2)
    with c1:
        banks = X.two_bank_picker("ch10")
    with c2:
        vd = X.date_picker("ch10")
    imp = X.table("ch10_implied")
    X.show_table(imp[(imp["valuation"] == vd) & imp["bank"].isin(banks)][
        ["bank", "coe", "lambda_ch5", "er_star_ch5", "implied_er_star_at_ch5_lambda", "implied_lambda_at_ch5_er_star",
         "implied_coe", "implied_roe_star", "conclusion", "additional_assumption", "evidence_class"]],
        "ch10_implied", "ch10_implied")
    s = X.table("ch10_iso_surface")
    with st.container(border=True):
        st.markdown("**Iso-value surface: value / price across persistence and long-run excess return**")
        b = st.segmented_control("Surface for", banks, default=banks[0], key="sb_ch10") or banks[0]
        shown = s[(s["valuation"] == vd) & (s["bank"] == b)][["bank", "valuation", "lambda", "er_star",
                                                               "value_to_price", "evidence_class"]]
        piv = shown.pivot_table(index="er_star", columns="lambda", values="value_to_price")
        fig = go.Figure(go.Contour(z=piv.values, x=piv.columns, y=piv.index, colorscale="RdYlGn", zmid=1,
                                   contours={"showlabels": True}, colorbar={"title": "value / price"},
                                   hovertemplate="lambda %{x:.2f}<br>ER* %{y:.1%}<br>value / price %{z:.2f}"
                                                 "<extra></extra>"))
        X.style(fig, x_title="Persistence (lambda)", y_title="Long-run excess return (ER*)").update_layout(
            hovermode="closest")
        fig.update_yaxes(tickformat=".0%")
        X.show_chart(fig, shown, "ch10_surface", "ch10_iso_surface",
                     "The contour at 1.0 is where value equals price; any reading needs an assumption on one axis")
    br = X.table("ch10_bridge")
    with st.container(border=True):
        st.markdown(f"**Value bridge: {X.DATES['HIST']} to {X.DATES['CURR']}**")
        b2 = st.segmented_control("Bridge for", banks, default=banks[0], key="bb_ch10") or banks[0]
        shown = br[br["bank"] == b2][["bank", "step", "tzs", "evidence_class"]].reset_index(drop=True)
        meas = ["absolute"] + ["relative"] * (len(shown) - 2) + ["total"]
        fig = go.Figure(go.Waterfall(x=shown["step"], y=shown["tzs"], measure=meas,
                                     increasing={"marker": {"color": X.HOUSE["sector"]}},
                                     decreasing={"marker": {"color": "#b5523b"}},
                                     totals={"marker": {"color": X.HOUSE[b2]}}))
        X.style(fig, y_unit="tzs", y_title="Equity value (TZS)").update_layout(showlegend=False)
        X.show_chart(fig, shown, "ch10_bridge", "ch10_bridge")


def ch11() -> None:
    c1, c2 = st.columns(2)
    with c1:
        banks = X.two_bank_picker("ch11")
    with c2:
        vd = X.date_picker("ch11")
    sc = X.table("ch11_scenarios")
    with st.container(border=True):
        st.markdown("**Bear, base and bull scenarios: value per share against price**")
        shown = sc[(sc["valuation"] == vd) & sc["bank"].isin(banks)][
            ["bank", "scenario", "value_per_share", "price", "roe_2026", "roe_2027", "evidence_class"]]
        fig = go.Figure()
        for b, r in shown.groupby("bank", sort=False):
            fig.add_trace(go.Bar(x=r["scenario"], y=r["value_per_share"], name=b, marker_color=X.HOUSE[b]))
            fig.add_hline(y=float(r["price"].iloc[0]), line_dash="dash", line_color=X.HOUSE[b],
                          annotation_text=f"{b} price {r['price'].iloc[0]:,.0f}", annotation_position="top left")
        X.style(fig, y_unit="tzs", y_title="TZS per share").update_layout(barmode="group", hovermode="closest")
        X.show_chart(fig, shown, "ch11_scen", "ch11_scenarios")
    mc = X.table("ch11_monte_carlo")
    with st.container(border=True):
        st.markdown("**Monte Carlo: published percentiles of value per share**")
        shown = mc[(mc["valuation"] == vd) & mc["bank"].isin(banks)][
            ["bank", "variant", "draws", "p5", "p10", "p50", "p90", "p95", "mean", "price", "prob_value_above_price",
             "evidence_class"]]
        fig = go.Figure()
        for (b, var), r in shown.groupby(["bank", "variant"], sort=False):
            r = r.iloc[0]
            nm = f"{b}: {var}"
            fig.add_trace(go.Box(name=nm, x=[nm], q1=[r["p10"]], median=[r["p50"]], q3=[r["p90"]], lowerfence=[r["p5"]],
                                 upperfence=[r["p95"]], mean=[r["mean"]], marker_color=X.HOUSE[b],
                                 opacity=1 if "primary" in var else 0.55, showlegend=False))
            fig.add_trace(go.Scatter(x=[nm], y=[r["price"]], mode="markers", name="price", showlegend=False,
                                     marker={"symbol": "line-ew-open", "size": 50, "color": X.TEXT,
                                             "line": {"width": 3}}))
        X.style(fig, y_unit="tzs", y_title="TZS per share", height=460).update_layout(hovermode="closest")
        X.show_chart(fig, shown, "ch11_mc", "ch11_monte_carlo",
                     "Box: P10-P90 with median; whiskers P5-P95; black bar: price. Draws are not published, only "
                     "these percentiles")
    cm = X.table("ch13_dec_mc_conditional")
    with st.container(border=True):
        st.markdown("**Probability that value exceeds price, by weight on the 2015-18 reading**")
        shown = cm[(cm["valuation"] == vd) & cm["bank"].isin(banks)]
        fig = go.Figure([go.Scatter(x=r["weight_on_B"], y=r["prob_value_above_price_mixture"], name=b,
                                    mode="lines+markers", line={"color": X.HOUSE[b]})
                         for b, r in shown.groupby("bank", sort=False)])
        X.style(fig, y_unit="pct", x_title="Weight on the exceptional-episode reading",
                y_title="P(value > price)")
        fig.update_xaxes(tickformat=".0%")
        X.show_chart(fig, shown, "ch11_mccond", "ch13_dec_mc_conditional")
    tor = X.table("outputs/valuation/register_tornado.csv")
    with st.container(border=True):
        st.markdown("**Which assumptions move value most? (bear and bull settings, one at a time)**")
        b = st.segmented_control("Bank", banks, default=banks[0], key="tb_ch11") or banks[0]
        n = st.slider("Show the largest", 5, 34, 12, key="tn_ch11")
        shown = tor[(tor["valuation"] == vd) & (tor["bank"] == b)].nlargest(n, "swing")[
            ["bank", "parameter", "base", "bear", "bull", "bear_impact", "bull_impact", "swing", "evidence_class"]]
        shown = shown.iloc[::-1]
        fig = go.Figure([go.Bar(y=shown["parameter"], x=shown["bear_impact"], orientation="h", name="bear setting",
                                marker_color="#b5523b"),
                         go.Bar(y=shown["parameter"], x=shown["bull_impact"], orientation="h", name="bull setting",
                                marker_color=X.HOUSE["sector"])])
        X.style(fig, height=max(320, 28 * len(shown)), x_title="Change in value per share").update_layout(
            barmode="overlay", hovermode="closest")
        fig.update_xaxes(tickformat=".0%")
        X.show_chart(fig, shown.iloc[::-1], "ch11_tornado", "outputs/valuation/register_tornado.csv")


def ch12() -> None:
    c1, c2 = st.columns(2)
    with c1:
        banks = X.two_bank_picker("ch12")
    with c2:
        vd = X.date_picker("ch12")
    dep = X.table("ch12_depreciation")
    e = X.table("ch12_expected_usd")
    with st.container(border=True):
        st.markdown("**Expected 12-month USD return under each depreciation case**")
        cases = list(dep.loc[dep["valuation"] == vd, "scenario"])
        sel = st.multiselect("Depreciation cases", cases, default=cases, key="c_ch12",
                             format_func=lambda c: f"{c} ({dep.set_index(['valuation', 'scenario']).loc[(vd, c), 'depreciation']:.1%})")
        pos = sorted(e["position_usd"].unique())
        p = (st.select_slider("Position (USD)", pos, value=pos[0], key="p_ch12", format_func=lambda v: f"{v:,.0f}")
             if len(pos) > 1 else pos[0])
        st.caption(f"Position: USD {p:,.0f}")
        shown = e[(e["valuation"] == vd) & e["bank"].isin(banks) & e["scenario"].isin(sel) & (e["position_usd"] == p)][
            ["bank", "scenario", "depreciation", "expected_tzs_return_12m", "translation_effect", "expected_usd_return",
             "round_trip_cost", "expected_usd_return_after_cost", "usd_hurdle", "excess_over_hurdle",
             "breakeven_depreciation", "evidence_class"]].drop_duplicates(["bank", "scenario"])
        fig = go.Figure([go.Bar(x=r["scenario"], y=r["expected_usd_return_after_cost"], name=b,
                                marker_color=X.HOUSE[b]) for b, r in shown.groupby("bank", sort=False)])
        if len(shown):
            fig.add_hline(y=float(shown["usd_hurdle"].iloc[0]), line_dash="dash", line_color=X.HOUSE["global"],
                          annotation_text=f"USD hurdle {shown['usd_hurdle'].iloc[0]:.1%}")
        X.style(fig, y_unit="pct", y_title="Expected USD return after round-trip cost").update_layout(
            barmode="group", hovermode="closest")
        X.show_chart(fig, shown, "ch12_usd", "ch12_expected_usd")
        X.show_table(dep[dep["valuation"] == vd], "ch12_dep", "ch12_depreciation")
    a = X.table("ch12_returns_annual")
    with st.container(border=True):
        st.markdown("**Annual total returns in TZS and USD**")
        cur = st.segmented_control("Currency", ["tzs_return", "usd_return"], default="usd_return", key="cu_ch12",
                                   format_func=lambda c: c[:3].upper()) or "usd_return"
        a = a[pd.to_numeric(a["period"], errors="coerce").notna()].assign(year=lambda x: x["period"].astype(int))
        yr = X.year_slider("ch12", a["year"])
        shown = a[a["bank"].isin(banks) & a["year"].between(*yr)][
            ["bank", "year", "price_start", "price_end", "dividends", "tzs_return", "fx_change", "usd_return",
             "evidence_class"]]
        fig = go.Figure([go.Bar(x=r["year"], y=r[cur], name=b, marker_color=X.HOUSE[b])
                         for b, r in shown.groupby("bank", sort=False)])
        X.style(fig, y_unit="pct", y_title=f"Total return ({cur[:3].upper()})").update_layout(barmode="group")
        X.show_chart(fig, shown, "ch12_annual", "ch12_returns_annual")
        sp = X.table("ch12_returns_spans")
        X.show_table(sp[sp["bank"].isin(banks)], "ch12_spans", "ch12_returns_spans")


# ============================================================================================================================
# Report support (chapter 13) and market (chapter 14)
# ============================================================================================================================


def analysts() -> None:
    d = X.table("ch13_dec_analyst_compare")
    with st.container(border=True):
        st.markdown("**Analyst views: broker targets and this study's values against the price**")
        c1, c2 = st.columns(2)
        with c1:
            banks = X.two_bank_picker("an")
        srcs = list(d["source"].unique())
        sel = c2.multiselect("Brokers and this study", srcs, default=srcs, key="s_an")
        shown = d[d["bank"].isin(banks) & d["source"].isin(sel)][
            ["source", "date", "date_label", "bank", "rating", "value_or_target", "price_at_date", "value_to_price",
             "method", "coe", "terminal_g", "evidence_class"]].sort_values("date")
        symbols = {s: sym for s, sym in zip(srcs, ["circle", "diamond", "square", "triangle-up", "star", "x"],
                                            strict=False)}
        fig = go.Figure()
        for (b, s), r in shown.groupby(["bank", "source"], sort=False):
            fig.add_trace(go.Scatter(x=pd.to_datetime(r["date"]), y=r["value_or_target"], name=f"{b}: {s}",
                                     mode="markers", text=r["rating"],
                                     marker={"symbol": symbols.get(s, "circle"), "size": 13, "color": X.HOUSE[b],
                                             "line": {"width": 1, "color": X.TEXT}},
                                     hovertemplate="%{x|%d %b %Y}: %{y:,.0f} (%{text})"))
        ps = X.table("ch13_price_series")
        price = ps[ps["bank"].isin(banks)][["bank", "date", "adj_close"]]
        for b, r in price.groupby("bank", sort=False):
            fig.add_trace(go.Scatter(x=pd.to_datetime(r["date"]), y=r["adj_close"], name=f"{b} price (DSE)",
                                     mode="lines", line={"color": X.HOUSE[b], "width": 1.2}, opacity=0.6))
        at = shown.dropna(subset=["price_at_date"])
        fig.add_trace(go.Scatter(x=pd.to_datetime(at["date"]), y=at["price_at_date"], name="price on the report date",
                                 mode="markers", marker={"symbol": "line-ew-open", "size": 18, "color": X.TEXT,
                                                         "line": {"width": 2}}))
        X.style(fig, y_unit="tzs", y_title="TZS per share (post-split basis)", height=480).update_layout(
            hovermode="closest")
        prices = price.rename(columns={"adj_close": "price_at_date"}).assign(source="DSE daily price")
        X.show_chart(fig, pd.concat([shown, prices], ignore_index=True), "an_compare",
                     ["ch13_dec_analyst_compare", "ch13_price_series"],
                     "Daily price line covers 3 Nov 2025 - 30 Oct 2026 (the published series)")


def quarterly() -> None:
    q = X.table("ch13_qtr_five")
    with st.container(border=True):
        st.markdown("**Quarterly statements: four quarter-ends to 30 Sep 2026**")
        c1, c2 = st.columns(2)
        with c1:
            banks = X.two_bank_picker("qt")
        ms = ["roe_quarter_annualised", "npl_ratio", "cost_to_income_quarter", "cet1_or_core_ratio", "net_ldr",
              "gross_ldr_printed", "loans_to_assets_printed", "customer_deposits_tzs_m", "net_loans_tzs_m",
              "deposits_qoq", "loans_qoq"]
        lab = {"roe_quarter_annualised": "ROE (quarter, annualised)", "cost_to_income_quarter": "Cost-to-income "
               "(quarter)", "cet1_or_core_ratio": "Core capital ratio", "net_ldr": "Net loan-to-deposit",
               "gross_ldr_printed": "Gross loan-to-deposit (printed)", "loans_to_assets_printed": "Loans / assets "
               "(printed)", "customer_deposits_tzs_m": "Customer deposits (TZS m)", "net_loans_tzs_m": "Net loans "
               "(TZS m)", "deposits_qoq": "Deposit growth q/q", "loans_qoq": "Loan growth q/q", "npl_ratio": "NPL ratio"}
        m = c2.selectbox("Measure", ms, key="m_qt", format_func=lab.get)
        shown = q[q["bank"].isin(banks)][["bank", "period_end", m, "evidence_class"]]
        u = "tzs_m" if m.endswith("_tzs_m") else "pct"
        fig = go.Figure([go.Scatter(x=r["period_end"], y=r[m], name=b, mode="lines+markers",
                                    line={"color": X.HOUSE[b], "width": 2.5}) for b, r in shown.groupby("bank",
                                                                                                         sort=False)])
        X.style(fig, y_unit=u, y_title=lab[m])
        fig.update_xaxes(type="category")
        X.show_chart(fig, shown, "qt_five", "ch13_qtr_five")
        r = X.table("ch13_qtr_roll")
        X.show_table(r[r["bank"].isin(banks)], "qt_roll", "ch13_qtr_roll",
                     "Value per share with year 1 re-based on each quarter (Scenario)")


def market() -> None:
    ps = X.table("ch13_price_series").assign(date=lambda x: pd.to_datetime(x["date"]))
    with st.container(border=True):
        st.markdown("**DSE daily prices and turnover, NMB and CRDB**")
        c1, c2 = st.columns(2)
        with c1:
            banks = X.two_bank_picker("mk")
        lo, hi = ps["date"].min().date(), ps["date"].max().date()
        rng = c2.date_input("Dates", (lo, hi), min_value=lo, max_value=hi, key="d_mk")
        rng = rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi)
        series = st.segmented_control("Series", ["Price", "Turnover"], default="Price", key="s_mk") or "Price"
        col = "adj_close" if series == "Price" else "turnover"
        shown = ps[ps["bank"].isin(banks) & ps["date"].dt.date.between(*rng)][
            ["bank", "date", "adj_close", "turnover", "traded_day", "evidence_class"]]
        fig = go.Figure()
        for b, r in shown.groupby("bank", sort=False):
            tr = go.Scatter if series == "Price" else go.Bar
            kw = {"mode": "lines", "line": {"color": X.HOUSE[b]}} if series == "Price" else {"marker_color": X.HOUSE[b]}
            fig.add_trace(tr(x=r["date"], y=r[col], name=b, **kw))
        sd = X.split_date()
        if pd.Timestamp(rng[0]) <= sd <= pd.Timestamp(rng[1]):
            fig.add_vline(x=sd, line_dash="dot", line_color=X.HOUSE["NMB"])
            fig.add_annotation(x=sd, yref="paper", y=1, text=f"{X.SPLIT_NOTE} ({sd:%d %b %Y})", showarrow=False,
                               xanchor="left", font={"color": X.HOUSE["NMB"]})
        X.style(fig, y_unit="tzs", y_title="Adjusted close (TZS)" if series == "Price" else "Turnover (TZS)")
        X.show_chart(fig, shown.assign(date=shown["date"].dt.date), "mk_daily", "ch13_price_series",
                     "Prices adjusted for the NMB 1:10 split (pre-split prices divided by 10)")
    m = X.table("ch13_price_monthly")
    with st.container(border=True):
        st.markdown("**Monthly closes, ranges and turnover**")
        X.show_table(m[m["bank"].isin(banks)], "mk_monthly", "ch13_price_monthly")
    cov = X.table("ch14_daily_coverage")
    with st.container(border=True):
        st.markdown("**Liquidity: share of trading days with a trade and with a price change, 2015-2026**")
        meas = st.segmented_control("Measure", ["traded_share", "nonstale_share"], default="nonstale_share",
                                    key="lq_mk", format_func=lambda c: {"traded_share": "Days traded",
                                                                        "nonstale_share": "Days with a new price"}[c])
        meas = meas or "nonstale_share"
        shown = cov[cov["bank"].isin(banks)][["bank", "year", "trading_days", "traded_days", "nonstale_days", meas,
                                              "coverage", "evidence_class"]]
        fig = go.Figure([go.Bar(x=r["year"], y=r[meas], name=b, marker_color=X.HOUSE[b])
                         for b, r in shown.groupby("bank", sort=False)])
        fig.add_hline(y=float(cov["threshold"].iloc[0]), line_dash="dash", line_color=X.HOUSE["global"],
                      annotation_text="adequacy threshold")
        X.style(fig, y_unit="pct").update_layout(barmode="group")
        X.show_chart(fig, shown, "mk_cov", "ch14_daily_coverage")
        lq = X.table("ch12_liquidity")
        X.show_table(lq[lq["bank"].isin(banks)], "mk_liq", "ch12_liquidity",
                     "Average daily value traded, block trades and the Amihud illiquidity measure")
        ms = X.table("ch14_microstructure")
        tests = list(ms["test"].unique())
        t = st.multiselect("Microstructure tests", tests, default=tests[:2], key="ms_mk")
        X.show_table(ms[ms["bank"].isin(banks) & ms["test"].isin(t)], "mk_micro", "ch14_microstructure")
    sb = X.table("ch14_split_benchmarks")
    with st.container(border=True):
        st.markdown("**Other DSE equities used as split benchmarks**")
        X.show_table(sb, "mk_splitb", "ch14_split_benchmarks")
