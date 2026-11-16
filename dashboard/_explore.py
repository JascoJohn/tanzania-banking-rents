"""Interactive building blocks for the dashboard (Output C, CLAUDE.md section 7).

Selectors (bank, period, metric), Plotly charts in the house colours and a "download the data shown" button for every
chart and table. Everything here filters, selects and re-plots published tables in ``outputs/``; nothing is estimated.
The frame a chart draws is the frame its download button writes, so every number on screen is a published value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from _common import OUTPUTS, ROOT

# --- house style (src/tzbank/reporting/chart_theme.py) -------------------------------------------------------------------
HOUSE = {"NMB": "#eb6834", "CRDB": "#2a78d6", "market": "#66A88A", "sector": "#006B3F", "global": "#6B736F"}
# the other Tanzanian panel banks and the African peers: distinguishable muted tones next to the two house colours
OTHERS = ["#006B3F", "#8c6d1f", "#6b736f", "#66A88A", "#7a5c8f", "#a8aca9", "#b5523b", "#3f6f6a", "#9c8f3c"]
FONT = "Calibri, Carlito, 'Source Sans Pro', sans-serif"
GRID, TEXT = "#E3ECE6", "#252B28"
DATES = {"HIST": "31 Dec 2025", "CURR": "31 Oct 2026"}  # the two valuation dates (point in time / current)
SPLIT_NOTE = "NMB 1:10 share split"

# --- column labels and units -----------------------------------------------------------------------------------------------
# unit: pct (decimal ratio shown as %), x (multiple), tzs (full TZS), tzs_m (TZS million), num (plain number)
LABELS: dict[str, tuple[str, str]] = {
    "roe": ("Return on equity", "pct"), "roa": ("Return on assets", "pct"),
    "nii_to_avg_assets": ("Net interest income / average assets", "pct"),
    "non_interest_income_share": ("Non-interest income share of income", "pct"),
    "cost_to_income": ("Cost-to-income ratio", "pct"), "cost_of_risk": ("Cost of risk", "pct"),
    "leverage": ("Leverage (assets / equity)", "x"), "nim": ("Net interest margin", "pct"),
    "npl_ratio": ("NPL ratio", "pct"), "coverage": ("NPL coverage", "pct"),
    "risk_adjusted_nim": ("Risk-adjusted NIM", "pct"), "return_on_rwa": ("Return on RWA", "pct"),
    "impairment_to_avg_assets": ("Impairment / average assets", "pct"),
    "nii_share": ("Net interest income share", "pct"), "nonii_share": ("Non-interest income share", "pct"),
    "fee_share_of_nonii": ("Fees share of non-interest income", "pct"),
    "fee_share_of_income": ("Fees share of total income", "pct"), "total_income": ("Total income (TZS)", "tzs"),
    "growth_total_income": ("Growth in total income", "pct"), "growth_nii": ("Growth in net interest income", "pct"),
    "growth_nonii": ("Growth in non-interest income", "pct"),
    "cost_of_deposits": ("Cost of deposits", "pct"), "cost_of_funds": ("Cost of funds", "pct"),
    "casa_share": ("Current and savings share of deposits", "pct"), "avg_deposits": ("Average deposits (TZS)", "tzs"),
    "gdfa_spread": ("T-bill yield minus deposit cost", "pct"),
    "gdfa_proxy_to_avg_assets": ("Gross deposit funding advantage / average assets (proxy)", "pct"),
    "loan_yield": ("Loan yield", "pct"), "securities_yield": ("Securities yield", "pct"),
    "iea_yield": ("Yield on interest-earning assets", "pct"), "loans_to_assets": ("Loans / assets", "pct"),
    "securities_to_assets": ("Securities / assets", "pct"), "loans_to_deposits": ("Loans / deposits", "pct"),
    "fee_income": ("Fee income (TZS)", "tzs"), "fee_to_avg_assets": ("Fee income / average assets", "pct"),
    "fx_to_avg_assets": ("FX income / average assets", "pct"),
    "opex_to_avg_assets": ("Operating cost / average assets", "pct"), "opex": ("Operating cost (TZS)", "tzs"),
    "staff_share_of_opex": ("Staff cost share of operating cost", "pct"),
    "deposit_growth": ("Deposit growth", "pct"), "share_current": ("Current accounts share", "pct"),
    "share_savings": ("Savings share", "pct"), "share_time": ("Time deposits share", "pct"),
    "ldr": ("Loan-to-deposit ratio", "pct"), "wholesale_share": ("Wholesale funding share", "pct"),
    "loan_growth": ("Loan growth", "pct"), "growth_gap": ("Loan growth minus deposit growth", "pct"),
    "net_franchise_central_roe_pts": ("Net deposit franchise, ROE points (central)", "pct"),
    "net_franchise_central_pct_assets": ("Net deposit franchise / assets (central)", "pct"),
    "equity_to_assets": ("Equity / assets", "pct"), "tier1_ratio": ("Tier 1 ratio", "pct"),
    "total_capital_ratio": ("Total capital ratio", "pct"), "tier1_headroom": ("Tier 1 headroom over requirement", "pct"),
    "total_headroom": ("Total capital headroom over requirement", "pct"), "rwa_growth": ("RWA growth", "pct"),
    "payout_total": ("Payout ratio (total)", "pct"), "payout_ordinary": ("Payout ratio (ordinary)", "pct"),
    "dps_ordinary": ("Ordinary DPS (TZS, as reported)", "num"),
    "rore_primary": ("Return on retained earnings (primary)", "pct"),
    "rore_adjusted": ("Return on retained earnings (raise-adjusted)", "pct"),
    "rwa_density": ("RWA density", "pct"), "risk_adjusted_rorwa": ("Risk-adjusted return on RWA", "pct"),
    "ppop_to_rwa": ("Pre-provision profit / RWA", "pct"),
    "coe": ("Cost of equity", "pct"), "er": ("Excess return (ROE minus cost of equity)", "pct"),
    "pb": ("Price-to-book", "x"), "pe": ("Price-to-earnings", "x"),
    "justified_pb": ("Justified price-to-book", "x"), "roe_minus_coe": ("ROE minus cost of equity", "pct"),
    "capital_ratio": ("Capital ratio", "pct"), "g": ("Long-run growth", "pct"),
    "binding_headroom": ("Binding capital headroom", "pct"),
}


def label(col: str) -> str:
    return LABELS.get(col, (col.replace("_", " ").capitalize(), "num"))[0]


def unit(col: str) -> str:
    return LABELS.get(col, ("", "pct" if re.search(r"(ratio|share|yield|roe|roa|_pct|margin)$", col) else "num"))[1]


def axis_format(u: str) -> dict:
    return {"pct": {"tickformat": ".0%", "hoverformat": ".2%"}, "x": {"tickformat": ".2~f", "hoverformat": ".2f"},
            "tzs": {"tickformat": ",.3s", "hoverformat": ",.0f"}, "tzs_m": {"tickformat": ",.0f", "hoverformat": ",.0f"},
            }.get(u, {"hoverformat": ",.4g"})


def fmt(v, u: str) -> str:
    """Display formatting only (no arithmetic beyond scaling for the % sign)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, str):
        return v
    return {"pct": f"{v:.1%}", "x": f"{v:.2f}x", "tzs": f"{v:,.0f}", "tzs_m": f"{v:,.0f}"}.get(u, f"{v:,.4g}")


# --- data access -----------------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def table(rel: str) -> pd.DataFrame:
    """A published table by path relative to the repository (``outputs/tables/x.csv``) or bare stem (``x``)."""
    p = ROOT / rel if "/" in rel else OUTPUTS / "tables" / f"{rel}.csv"
    return pd.read_csv(p, low_memory=False)


@st.cache_data(show_spinner=False)
def bank_names() -> dict[str, str]:
    """bank_id -> ticker for the Tanzanian panel (config/banks.yaml)."""
    cfg = yaml.safe_load((ROOT / "config" / "banks.yaml").read_text(encoding="utf-8"))
    return {b["bank_id"]: b["ticker"] for b in cfg["banks"]}


@st.cache_data(show_spinner=False)
def bank_full_names() -> dict[str, str]:
    cfg = yaml.safe_load((ROOT / "config" / "banks.yaml").read_text(encoding="utf-8"))
    out = {b["ticker"]: b["name"] for b in cfg["banks"]}
    v = table("ch13_peers_valuation")
    out.update(dict(zip(v["bank"], v["name"], strict=True)))
    return out


@st.cache_data(show_spinner=False)
def split_date() -> pd.Timestamp:
    ca = pd.read_csv(ROOT / "config" / "corporate_actions.csv")
    return pd.Timestamp(ca.loc[(ca["ticker"] == "NMB") & (ca["action"] == "split"), "effective_date"].iloc[0])


@st.cache_data(show_spinner=False)
def panel(rel: str) -> pd.DataFrame:
    """A bank-year table with ``bank`` (ticker) and ``year`` columns added (bank_id / fiscal_year kept)."""
    d = table(rel).copy()
    if "bank_id" in d:
        d["bank"] = d["bank_id"].map(bank_names()).fillna(d["bank_id"])
    yc = next((c for c in ("fiscal_year", "year", "fy") if c in d), None)
    if yc:
        d["year"] = pd.to_numeric(d[yc], errors="coerce").astype("Int64")
    return d


def colour_map(banks: Iterable[str]) -> dict[str, str]:
    out, i = {}, 0
    for b in banks:
        if b in HOUSE:
            out[b] = HOUSE[b]
        else:
            out[b] = OTHERS[i % len(OTHERS)]
            i += 1
    return out


TZ_PANEL = ["NMB", "CRDB", "DCB", "MBP", "MKCB", "MCB", "MUCOBA"]


# --- layout helpers --------------------------------------------------------------------------------------------------------
def style(fig: go.Figure, *, y_unit: str = "num", height: int = 420, x_title: str | None = None,
          y_title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, font={"family": FONT, "color": TEXT, "size": 13},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10}, hovermode="x unified",
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF")
    fig.update_xaxes(gridcolor=GRID, title=x_title)
    fig.update_yaxes(gridcolor=GRID, title=y_title, **axis_format(y_unit))
    return fig


def evidence_caption(d: pd.DataFrame, source: str | Sequence[str], extra: str = "") -> None:
    ev = ", ".join(sorted(d["evidence_class"].dropna().astype(str).unique())) if "evidence_class" in d else ""
    src = ", ".join(Path(s).name if "/" in s else f"{s}.csv" for s in ([source] if isinstance(source, str) else source))
    st.caption(" · ".join(x for x in (f"Evidence class: {ev}" if ev else "", f"Source table: {src}", extra) if x))


def _sources(source: str | Sequence[str]) -> list[str]:
    return [s if "/" in s else f"outputs/tables/{s}.csv" for s in ([source] if isinstance(source, str) else source)]


def download(d: pd.DataFrame, key: str, name: str, source: str | Sequence[str] = ()) -> None:
    """The 'download the data shown' button: writes exactly the frame on screen."""
    payload = d.to_csv(index=False).encode("utf-8")
    st.download_button("Download the data shown (CSV)", payload, file_name=f"{name}.csv", mime="text/csv",
                       key=f"dl_{key}", icon=":material/download:")
    if st.session_state.get("_record"):  # set by tests/test_dashboard.py only: payload and its published tables
        st.session_state.setdefault("_shown", {})[key] = (payload, _sources(source))


def show_chart(fig: go.Figure, d: pd.DataFrame, key: str, source: str | Sequence[str], extra: str = "") -> None:
    st.plotly_chart(fig, key=f"fig_{key}", config={"displaylogo": False})
    if st.session_state.get("_record"):  # tests: the numbers drawn must be in the downloaded frame
        st.session_state.setdefault("_figs", {})[key] = fig.to_plotly_json()["data"]
    evidence_caption(d, source, extra)
    download(d, key, key, source)


def show_table(d: pd.DataFrame, key: str, source: str | Sequence[str], extra: str = "", **kw) -> None:
    st.dataframe(d, hide_index=True, **kw)
    evidence_caption(d, source, extra)
    download(d, key, key, source)


# --- selectors -------------------------------------------------------------------------------------------------------------
def bank_picker(key: str, options: Sequence[str], default: Sequence[str] = ("NMB", "CRDB"),
                label_: str = "Banks") -> list[str]:
    names = bank_full_names()
    return st.multiselect(label_, list(options), default=[b for b in default if b in options], key=f"banks_{key}",
                          format_func=lambda b: f"{b} ({names[b]})" if b in names and not names[b].startswith(b) else b)


def two_bank_picker(key: str, label_: str = "Bank") -> list[str]:
    """NMB, CRDB or both (default both)."""
    pick = st.segmented_control(label_, ["Both", "NMB", "CRDB"], default="Both", key=f"bank2_{key}")
    return ["NMB", "CRDB"] if pick in (None, "Both") else [pick]


def date_picker(key: str, options: Sequence[str] = ("HIST", "CURR"), default: str = "CURR") -> str:
    return st.segmented_control("Valuation date", list(options), default=default, key=f"vd_{key}",
                                format_func=lambda v: DATES.get(v, v)) or default


def year_slider(key: str, years: Sequence[int], default: tuple[int, int] | None = None) -> tuple[int, int]:
    ys = sorted({int(y) for y in years if pd.notna(y)})
    if len(ys) < 2:
        return (ys[0], ys[0]) if ys else (0, 0)
    return st.slider("Years", ys[0], ys[-1], default or (ys[0], ys[-1]), key=f"years_{key}")


def metric_picker(key: str, cols: Sequence[str], default: str | None = None) -> str:
    cols = list(cols)
    return st.selectbox("Measure", cols, index=cols.index(default) if default in cols else 0, key=f"metric_{key}",
                        format_func=label)


# --- composite explorers ---------------------------------------------------------------------------------------------------
Reference = Callable[[str, tuple[int, int]], pd.DataFrame | None]  # (metric, years) -> frame with bank, year, value


def explore_panel(key: str, rel: str, metrics: Sequence[str], *, title: str, banks: Sequence[str] = TZ_PANEL,
                  default_metric: str | None = None, reference: Reference | None = None, note: str = "",
                  default_banks: Sequence[str] = ("NMB", "CRDB")) -> None:
    """Bank x year explorer: pick banks, a measure and a year range; overlaid or side-by-side Plotly chart."""
    d = panel(rel)
    metrics = [m for m in metrics if m in d and d[m].notna().any()]
    if not metrics:
        return
    with st.container(border=True):
        st.markdown(f"**{title}**")
        c1, c2 = st.columns([3, 2])
        with c1:
            sel = bank_picker(key, [b for b in banks if b in set(d["bank"])], default_banks)
        with c2:
            m = metric_picker(key, metrics, default_metric)
        c3, c4 = st.columns([3, 2])
        have = d.loc[d[m].notna(), "year"]
        with c3:
            yr = year_slider(key, have)
        with c4:
            layout = st.segmented_control("View", ["Overlaid", "Side by side"], default="Overlaid", key=f"lay_{key}")
        shown = d.loc[d["bank"].isin(sel) & d["year"].between(*yr) & d[m].notna(),
                      ["bank", "year", m] + (["evidence_class"] if "evidence_class" in d else [])]
        shown = shown.sort_values(["bank", "year"]).reset_index(drop=True)
        ref = reference(m, yr) if reference else None
        if shown.empty and (ref is None or ref.empty):
            st.info("No published values for this selection.")
            return
        fig = go.Figure()
        cmap = colour_map(sel)
        side = layout == "Side by side" and len(sel) > 1
        if side:
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=len(sel), shared_yaxes=True, subplot_titles=sel)
        for i, b in enumerate(sel):
            s = shown[shown["bank"] == b]
            tr = go.Scatter(x=s["year"], y=s[m], name=b, mode="lines+markers", line={"color": cmap[b], "width": 2.5})
            fig.add_trace(tr, row=1, col=i + 1) if side else fig.add_trace(tr)
        if ref is not None and not ref.empty:
            for nm, r in ref.groupby("bank", sort=False):
                tr = go.Scatter(x=r["year"], y=r["value"], name=nm, mode="lines",
                                line={"color": HOUSE["sector"], "dash": "dash", "width": 2})
                if side:
                    for i in range(len(sel)):
                        fig.add_trace(tr.update(showlegend=i == 0), row=1, col=i + 1)
                else:
                    fig.add_trace(tr)
            shown = pd.concat([shown, ref.rename(columns={"value": m})], ignore_index=True)
        style(fig, y_unit=unit(m), y_title=label(m))
        fig.update_xaxes(dtick=2 if yr[1] - yr[0] > 10 else 1, tickformat="d")
        srcs = [rel] + ([ref.attrs["source"]] if ref is not None and "source" in ref.attrs else [])
        show_chart(fig, shown, key, srcs, note)


def explore_cross_section(key: str, d: pd.DataFrame, metrics: Sequence[str], *, title: str, cat: str = "bank",
                          source: str, note: str = "", text: Callable[[pd.Series], str] | None = None,
                          hlines: Callable[[str], list[tuple[float, str]]] | None = None) -> None:
    """Bar chart of one measure across banks (or other categories) with a measure selector."""
    metrics = [m for m in metrics if m in d]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        m = metric_picker(key, metrics)
        shown = d[[cat, m] + [c for c in ("flagged_year", "evidence_class") if c in d]].reset_index(drop=True)
        cmap = colour_map(shown[cat])
        lab = shown.apply(text, axis=1) if text else shown[m].map(lambda v: fmt(v, unit(m)))
        fig = go.Figure(go.Bar(x=shown[cat], y=shown[m].fillna(0), marker_color=[cmap[b] for b in shown[cat]],
                               text=lab, textposition="outside", hovertext=lab, hoverinfo="x+text"))
        for y, nm in (hlines(m) if hlines else []):
            fig.add_hline(y=y, line_dash="dash", line_color=HOUSE["global"], annotation_text=nm,
                          annotation_position="top right")
        style(fig, y_unit=unit(m), y_title=label(m)).update_layout(hovermode="closest", showlegend=False)
        show_chart(fig, shown, key, source, note)
