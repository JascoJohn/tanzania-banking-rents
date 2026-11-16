"""Exhibit charts for the owner's drafts (fill-review pre-flight E; D65, D69).

Every chart exhibit the drafts in ``drafts/`` mark ``[Chart: pipeline export ...]`` is written as a 300-dpi PNG to
``outputs/exhibits/<report>/<exhibit_no>.png`` with the plotted data beside it (``<exhibit_no>_data.csv``), so the owner
can insert it. ``outputs/exhibits/index.csv`` lists report, exhibit number, title, file, source and evidence class.
Reports: ``main`` (Dominant Banks, Conditional Rents), ``nmb`` and ``crdb`` (investment reports), ``foreign`` (Tanzanian
bank equities for foreign investors). Charts that already exist as chapter figures are copied (same figure, same data);
the others are drawn here from pipeline tables only (CLAUDE.md 5: no calculations beyond plotting and grouping).
Table exhibits (scorecards, KPI tables, financial tables) are not charts and are filled from the findings pack.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path
from tzbank.io import write_csv

NAME = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB", "B07": "MUCOBA"}
BID = {v: k for k, v in NAME.items()}
REPORTS = {"main": "Dominant Banks, Conditional Rents (research report)", "nmb": "NMB Bank Plc (investment report)",
           "crdb": "CRDB Bank Plc (investment report)", "foreign": "Tanzanian Bank Equities for Foreign Investors"}
PEER_GREY = "#8c8c86"


def _t(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, low_memory=False, dtype={"bank_id": str})


def _foot(src: str, ev: str) -> str:
    return f"Source: {src}. Evidence class: {ev}."


# --- drawn exhibits ----------------------------------------------------------------------------------------------------
def profit_share(root, png):
    c = _t(root, "ch13_cr2_profit_direct.csv")
    allb = c[c["sector_basis"].str.startswith("all banking")]
    a = allb[allb["numerator_basis"].str.startswith("NMB + CRDB Group")].sort_values("year")
    s = allb[allb["numerator_basis"].str.contains("Bank-solo")].sort_values("year")
    k = c[c["sector_basis"].str.startswith("commercial")].sort_values("year")
    fig, ax = plt_subplots(figsize=(7.5, 4.0))
    ax.plot(a["year"], a["cr2_asset_share"] * 100, color="#1f1f1d", lw=2, label="asset share (all banking institutions)")
    ax.plot(a["year"], a["cr2_pbt_share"] * 100, color=COLORS["B02"], lw=2, marker="o", ms=3,
            label="profit-before-tax share (BoT appendices; 2023 not published in usable form)")
    ax.plot(k["year"], k["cr2_pbt_share"] * 100, color=COLORS["B01"], lw=1, ls="--", marker="s", ms=3,
            label="commercial banks, Estimated via BoT sector ROA (ch01)")
    if len(s):  # CRDB parent bank in the numerator (edit pack A B3): Group shares are upper bounds
        ax.plot(s["year"], s["cr2_pbt_share"] * 100, color="#06452E", lw=1.2, ls=":", marker="D", ms=3,
                label="profit share, CRDB bank-solo numerator")
        ax.plot(s["year"], s["cr2_asset_share"] * 100, color="#5a5a55", lw=1.2, ls=":",
                label="asset share, CRDB bank-solo numerator")
    style(ax, "NMB + CRDB share of sector profit before tax and of assets", "percent")
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, png, _foot("BoT supervision report appendices 2013-2025; bank annual reports (Group)", "Derived; "
                         "dashed series Estimated"))
    return pd.concat([a.assign(series="all banking institutions"), s.assign(series="all banking, CRDB bank-solo"),
                      k.assign(series="commercial banks (Estimated)")])[
        ["series", "year", "cr2_asset_share", "cr2_pbt_share", "evidence_class"]], "Derived"


def dse_shares(root, png):
    d = _t(root, "ch01_dse_shares.csv")
    s = _t(root, "ch01_shares.csv")
    a = s[(s["measure"] == "assets")].sort_values("year").iloc[-1]
    rows = [(f"commercial-bank assets ({int(a['year'])})", float(a["cr2"]))]
    for r in d.itertuples():
        if r.measure.startswith("market cap share of domestic") or (r.measure == "equity turnover share" and
                                                                     r.period == "Jul-Sep 2026"):
            rows.append((f"{r.measure.replace('market cap share of domestic', 'domestic market capitalisation')} "
                         f"({r.period})", float(r.share)))
    x = pd.DataFrame(rows, columns=["measure", "share"])
    fig, ax = plt_subplots(figsize=(7.0, 3.4))
    ax.barh(x["measure"], x["share"] * 100, color=["#1f1f1d", COLORS["B02"], COLORS["B01"]][:len(x)])
    for i, v in enumerate(x["share"]):
        ax.text(v * 100 + 1, i, f"{v:.1%}", va="center", fontsize=8)
    style(ax, "NMB + CRDB: share of banking and of the listed equity market", "")
    ax.set_xlabel("percent", fontsize=8)
    ax.invert_yaxis()
    save(fig, png, _foot("DSE market report, 30 October 2026; BoT", "Derived"))
    return x.assign(evidence_class="Derived"), "Derived"


def headroom(root, png):
    c = _t(root, "ch03_capital_buffers.csv")
    c = c[c["fiscal_year"].between(2018, 2025)]
    g = c.groupby("bank")["binding_headroom"].agg(["mean", "count"]).sort_values("mean").reset_index()
    fig, ax = plt_subplots(figsize=(7.0, 3.6))
    ax.barh(g["bank"], g["mean"] * 100, color=[COLORS.get(BID.get(b, ""), PEER_GREY) for b in g["bank"]])
    style(ax, "Binding capital headroom over the BoT minimum + conservation buffer, mean FY2018-25", "")
    ax.set_xlabel("percentage points", fontsize=8)
    save(fig, png, _foot("bank annual reports; BoT Capital Adequacy Regulations (chapter 3)", "Derived"))
    return g.rename(columns={"mean": "binding_headroom_mean", "count": "years"}).assign(evidence_class="Derived"), \
        "Derived"


def excess_returns(root, png):
    e = _t(root, "ch05_excess_returns.csv")
    g = e.dropna(subset=["er"]).groupby("bank_id").agg(mean_er=("er", "mean"), first=("fiscal_year", "min"),
                                                         last=("fiscal_year", "max"), n=("er", "size")).reset_index()
    g["bank"] = g["bank_id"].map(NAME)
    g = g.sort_values("mean_er")
    fig, ax = plt_subplots(figsize=(7.0, 3.6))
    ax.barh([f"{b} (FY{f}-{la})" for b, f, la in zip(g["bank"], g["first"], g["last"], strict=True)], g["mean_er"] * 100,
            color=[COLORS.get(b, PEER_GREY) for b in g["bank_id"]])
    ax.axvline(0, color="#1f1f1d", lw=0.8)
    style(ax, "Mean excess return (ROE - annual cost of equity), seven-bank panel", "")
    ax.set_xlabel("percentage points", fontsize=8)
    save(fig, png, _foot("chapter 5 estimates", "Derived"))
    return g[["bank", "first", "last", "n", "mean_er"]].assign(evidence_class="Derived"), "Derived"


def deposit_migration(root, png):
    m = _t(root, "ch13_deposit_migration.csv")
    fig, ax = plt_subplots(figsize=(7.0, 3.8))
    for b, g in m.groupby("bank"):
        ax.plot(g["casa_repriced"] * 100, g["roe_change_pts"] * 100, color=COLORS[BID[b]], lw=2, marker="o", label=b)
    style(ax, "ROE effect of repricing x% of current and savings balances to the T-bill (FY2025 base)",
          "ROE, percentage points")
    ax.set_xlabel("share of CASA repriced to the T-bill, percent", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    save(fig, png, _foot("chapter 6 model, scenario (T-bill on the v1.3 path); NMB FY2025 mix from the annual report "
                         "note (D29 manual entry)", "Scenario"))
    return m[["bank", "casa_repriced", "casa_tzs", "casa_cost", "tbill_v13", "roe_change_pts", "evidence_class"]], \
        "Scenario"


def distributable(root, png):
    d = _t(root, "ch07_distributable_capacity.csv")
    d = d[d["requirement_reading"].str.startswith("A")]
    d["bank"] = d["bank_id"].astype(str).map(NAME)
    fig, ax = plt_subplots(figsize=(7.0, 3.8))
    banks = list(d["bank"].unique())
    w = 0.18
    for i, buf in enumerate(sorted(d["management_buffer"].unique())):
        g = d[d["management_buffer"] == buf].set_index("bank").reindex(banks)
        ax.bar(np.arange(len(banks)) + (i - 1) * w, g["payout_max_steady_state"] * 100, width=w,
               color=["#c9c9c4", "#9a9a94", "#5f5f5a"][i % 3], label=f"ceiling, {buf * 100:.0f} pp buffer")
    act = d.groupby("bank")["payout_actual"].first().reindex(banks)
    ax.scatter(np.arange(len(banks)), act * 100, color="#c0392b", zorder=3, label="FY2025 actual payout")
    ax.set_xticks(np.arange(len(banks)), banks)
    style(ax, "FY2025 payout against modelled payout ceilings (Tier 1, reading A)", "percent of profit")
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, png, _foot("chapter 7 distributable-capacity scenarios (NMB Bank-solo capital, D2)", "Scenario"))
    return d[["bank", "management_buffer", "payout_max_steady_state", "payout_actual", "evidence_class"]], "Scenario"


def earnings_bridge(root, png, bank=None, scen=("base", "bear", "bull")):
    e = _t(root, "ch13_earnings_bridge.csv")
    e = e[e["scenario"].isin(scen) & ((e["bank"] == bank) if bank else True)]
    banks = [bank] if bank else ["NMB", "CRDB"]
    fig, axs = plt_subplots(1, len(banks), figsize=(7.5 if len(banks) == 1 else 10, 4.0), squeeze=False)
    for ax, b in zip(axs[0], banks, strict=True):
        g = e[e["bank"] == b]
        steps = g[g["scenario"] == scen[0]].sort_values("step_no")["step"].tolist()
        w = 0.8 / len(scen)
        for i, s in enumerate(scen):
            v = g[g["scenario"] == s].set_index("step").reindex(steps)["tzs"] / 1e9
            ax.bar(np.arange(len(steps)) + (i - (len(scen) - 1) / 2) * w, v, width=w,
                   color={"base": COLORS[BID[b]], "bear": "#5f5f5a", "bull": "#c9c9c4"}[s], label=s)
        ax.set_xticks(np.arange(len(steps)), [textwrap.fill(s, 18) for s in steps], rotation=60, ha="right", fontsize=6)
        style(ax, f"{b}: attributable profit bridge FY2025A -> FY2027E", "TZS bn")
        ax.legend(fontsize=6.5, frameon=False)
    save(fig, png, _foot("forecast engine, register v1.3 (end-steps are levels, middle steps are driver changes)",
                         "Scenario"))
    return e[["bank", "scenario", "step_no", "step", "tzs", "evidence_class"]], "Scenario"


def roe_bridge(root, png, bank):
    b = _t(root, "ch13_roe_sustainable_bridge.csv")
    g = b[b["bank"] == bank].sort_values("step_no")
    v = g["roe_pts"].to_numpy() * 100
    fig, ax = plt_subplots(figsize=(7.5, 4.0))
    run = v[0]
    for i, val in enumerate(v):
        if i in (0, len(v) - 1):
            ax.bar(i, val, color=COLORS[BID[bank]])
            top, lab = val, f"{val:.1f}"
        else:
            ax.bar(i, val, bottom=run, color="#2e8b57" if val >= 0 else "#c0392b")
            top, lab = max(run, run + val), f"{val:+.1f}"
            run += val
        ax.text(i, top + 0.3, lab, ha="center", fontsize=7)
    ax.set_xticks(range(len(v)), [textwrap.fill(s, 16) for s in g["step"]], rotation=0, ha="center", fontsize=6)
    ax.set_ylim(0, v.max() * 1.12)
    style(ax, f"{bank}: from reported FY2025 ROE to sustainable ROE*", "ROE, percent / percentage points")
    save(fig, png, _foot("research report chapters 4, 6-8; register v1.3", "Scenario"))
    return g[["bank", "step_no", "step", "roe_pts", "evidence_class"]], "Scenario"


GLOBAL_LABEL = "Global bank average (McKinsey, 2025)"


def _peer_curr(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(NMB and CRDB, African peers) at 31 Oct 2026 from the prompt 19 peer table (P/B on reported book), with the
    peers' names; empty frames before the peer stage has run (the market_ext stage redraws these exhibits)."""
    f = path("tables", root) / "ch13_peers_table.csv"
    if not f.exists():
        return pd.DataFrame(columns=["bank", "roe", "pb"]), pd.DataFrame(columns=["bank", "name", "roe", "pb"])
    t = _t(root, "ch13_peers_table.csv")
    t = t[t["valuation"] == "CURR"]
    tz = t[t["bank"].isin(["NMB", "CRDB"])].dropna(subset=["pb", "roe"])
    pe = t[~t["country"].fillna("").isin(["Tanzania", ""])].dropna(subset=["pb", "roe"]).copy()
    v = path("tables", root) / "ch13_peers_valuation.csv"
    names = _t(root, "ch13_peers_valuation.csv").drop_duplicates("bank").set_index("bank")["name"] if v.exists() \
        else pd.Series(dtype=str)
    pe["name"] = pe["bank"].map(names).fillna(pe["bank"])
    return tz, pe


def _global_pb(root: Path) -> tuple[float, float] | None:
    """(ROE, P/B) of the global banking benchmark (McKinsey Global Banking Annual Review 2025), if recorded."""
    f = path("tables", root) / "ch13_benchmarks.csv"
    if not f.exists():
        return None
    g = _t(root, "ch13_benchmarks.csv")
    g = g[g["scope"].astype(str).eq("global banking")]
    roe, pb = g[g["metric"] == "ROE"]["value"], g[g["metric"] == "price-to-book"]["value"]
    return (float(roe.iloc[0]), float(pb.iloc[0])) if len(roe) and len(pb) else None


def pb_roe_panel(root, png, highlight=None, own_history=False):
    """P/B against ROE at 31 Oct 2026: NMB, CRDB and the registered African peers (prompt 19 figures, P/B on reported
    book) and the global bank average. The other DSE-listed Tanzanian banks are not peers and are not drawn
    (prompt 23)."""
    tz, pe = _peer_curr(root)
    fig, ax = plt_subplots(figsize=(7.5, 4.6))
    rows = []
    for r in tz.itertuples():
        hl = highlight in (None, r.bank)
        ax.scatter(r.roe * 100, r.pb, color=COLORS[BID[r.bank]], s=46 if hl else 22,
                   edgecolor="#1f1f1d" if hl else "none", zorder=4)
        ax.annotate(f"{r.bank} (31 Oct 2026 price, FY2025 book)", (r.roe * 100, r.pb), fontsize=6.5, xytext=(4, 4),
                    textcoords="offset points")
        rows.append({"group": "NMB and CRDB (31 Oct 2026)", "bank": r.bank, "roe": r.roe, "pb": r.pb})
    if len(pe):
        ax.scatter(pe["roe"] * 100, pe["pb"], marker="^", color=PEER_GREY, s=24,
                   label="African peers (30 Oct 2026 price, reported book)")
        for r in pe.itertuples():
            ax.annotate(str(r.name).split()[0], (r.roe * 100, r.pb), fontsize=6, color=PEER_GREY, xytext=(3, -8),
                        textcoords="offset points")
            rows.append({"group": "African peers (30 Oct 2026)", "bank": r.name, "roe": r.roe, "pb": r.pb})
    gl = _global_pb(root)
    if gl:
        ax.scatter(gl[0] * 100, gl[1], marker="*", s=90, color="#1f1f1d", label=GLOBAL_LABEL)
        rows.append({"group": GLOBAL_LABEL, "bank": "global banking", "roe": gl[0], "pb": gl[1]})
    if own_history:
        v = _t(root, "ch13_valuation_bands.csv")
        h = v[(v["kind"] == "history") & v["bank"].isin([highlight] if highlight else ["NMB", "CRDB"])]
        d4 = _t(root, "ch04_dupont_panel.csv")
        d4["bank"] = d4["bank_id"].astype(str).map(NAME)
        h = h.merge(d4[["bank", "fiscal_year", "roe"]], on=["bank", "fiscal_year"], how="left").dropna(subset=["roe"])
        for b, g in h.groupby("bank"):
            ax.scatter(g["roe"] * 100, g["pb"], color=COLORS[BID[b]], alpha=0.35, s=14,
                       label=f"{b} own history, year-end (FY{int(g['fiscal_year'].min())}-{int(g['fiscal_year'].max())})")
            rows += [{"group": f"{b} own history", "bank": f"{b} FY{r.fiscal_year}", "roe": r.roe, "pb": r.pb}
                     for r in g.itertuples()]
    style(ax, "Price-to-book against ROE" + (f": {highlight}" if highlight else ""), "P/B")
    ax.set_xlabel("ROE, percent", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    save(fig, png, _foot("peer annual reports and exchange prices (P/B on reported book); DSE; company reports; "
                         "McKinsey Global Banking Annual Review 2025", "Derived"))
    return pd.DataFrame(rows).assign(evidence_class="Derived"), "Derived"


def own_history_roe(root, png, bank):
    """P/B since listing with the own-history band, the 31 Oct 2026 point, the African peer median and the global
    bank average (top), and ROE (bottom), one bank (prompt 17 A9; prompt 23: the bank reports' Exhibit 4)."""
    v = _t(root, "ch13_valuation_bands.csv")
    h = v[(v["kind"] == "history") & (v["bank"] == bank)].sort_values("fiscal_year")
    b = v[(v["kind"] == "band") & (v["bank"] == bank) & (v["multiple"] == "P/B")].iloc[0]
    d = _t(root, "ch04_dupont_panel.csv")
    r = d[d["bank_id"] == BID[bank]].sort_values("fiscal_year").dropna(subset=["roe"])
    # start at the first year with both P/B and ROE (prompt 24: NMB has P/B from FY2008, ROE from FY2009)
    first = max(float(h.dropna(subset=["pb"])["fiscal_year"].min()), float(r["fiscal_year"].min()))
    h, r = h[h["fiscal_year"] >= first], r[r["fiscal_year"] >= first]
    tz, pe = _peer_curr(root)
    c = COLORS[BID[bank]]
    fig, (a1, a2) = plt_subplots(2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(h["fiscal_year"], h["pb"], color=c, lw=2, marker="o", ms=3, label=f"{bank} P/B (year-end)")
    for lvl, ls, lab in ((b["mean"], "-", "own mean"), (b["plus_1sd"], ":", "own mean +1 sd"),
                         (b["minus_1sd"], ":", "own mean -1 sd")):
        a1.axhline(lvl, color=c, lw=0.8, ls=ls, alpha=0.7, label=f"{lab} {lvl:.2f}")
    data = [h[["fiscal_year", "pb"]].assign(series=f"{bank} P/B")]
    x_now = float(h["fiscal_year"].max()) + 10 / 12  # 31 Oct 2026 on the fiscal-year axis
    now = tz[tz["bank"] == bank]
    if len(now):
        pb_now = float(now["pb"].iloc[0])
        a1.scatter([x_now], [pb_now], marker="D", s=40, color=c, edgecolor="#1f1f1d", zorder=5,
                   label="31 Oct 2026 price on FY2025 book")
        a1.annotate(f"31 Oct 2026: {pb_now:.2f}", (x_now, pb_now), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7, color=c, fontweight="bold")
        data.append(pd.DataFrame([{"fiscal_year": x_now, "pb": pb_now, "series": f"{bank} P/B, 31 Oct 2026"}]))
    if len(pe):
        med = float(pe["pb"].median())
        a1.axhline(med, color=PEER_GREY, lw=1, ls="--", label=f"African peer median {med:.2f} (30 Oct 2026)")
        data.append(pd.DataFrame([{"series": "African peer median P/B", "pb": med}]))
    gl = _global_pb(root)
    if gl:
        a1.axhline(gl[1], color="#1f1f1d", lw=0.8, ls="--", label=f"{GLOBAL_LABEL} {gl[1]:.1f}")
        data.append(pd.DataFrame([{"series": GLOBAL_LABEL, "pb": gl[1]}]))
    style(a1, f"{bank}: P/B since listing", "P/B")
    a2.bar(r["fiscal_year"], r["roe"] * 100, color=c, alpha=0.75, label=f"{bank} ROE")
    a2.axhline(0, color="#1f1f1d", lw=0.6)
    style(a2, f"{bank}: return on equity", "percent")
    a2.set_xlim(float(h["fiscal_year"].min()) - 0.7, x_now + 0.7)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.5, frameon=False)
    save(fig, png, _foot("DSE; company reports; peer annual reports (P/B on reported book); McKinsey Global Banking "
                         "Annual Review 2025; ROE from the chapter 4 panel", "Derived"))
    data.append(r[["fiscal_year", "roe"]].assign(series=f"{bank} ROE"))
    return pd.concat(data, ignore_index=True).assign(evidence_class="Derived"), "Derived"


def sector_mix(root, png, bank):
    m = _t(root, "ch02_loan_mix.csv")
    y = 2023 if (m["fiscal_year"] == 2023).any() else int(m["fiscal_year"].max())
    g = m[(m["fiscal_year"] == y) & m["book"].isin([bank, "BoT commercial banks"])]
    p = g.pivot_table(index="category", columns="book", values="share").fillna(0).sort_values(bank)
    fig, ax = plt_subplots(figsize=(7.0, 4.0))
    yy = np.arange(len(p))
    ax.barh(yy + 0.2, p[bank] * 100, height=0.4, color=COLORS[BID[bank]], label=bank)
    ax.barh(yy - 0.2, p["BoT commercial banks"] * 100, height=0.4, color=PEER_GREY, label="commercial-bank system")
    ax.set_yticks(yy, [c.replace("_", " ") for c in p.index], fontsize=7)
    style(ax, f"Loan book by sector, FY{y}: {bank} against the commercial-bank system", "")
    ax.set_xlabel("percent of loans", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    save(fig, png, _foot(f"{bank} IFRS 7 industry note; BoT credit by economic activity (chapter 2)", "Observed"))
    return p.reset_index().assign(fiscal_year=y, evidence_class="Observed"), "Observed"


def rerating(root, png):
    r = _t(root, "ch10_rerating.csv")
    cols = [("earnings_and_time_log", "earnings and time"), ("discount_rate_log", "cost of equity"),
            ("residual_log", "residual")]
    fig, ax = plt_subplots(figsize=(7.0, 3.8))
    left = np.zeros(len(r))
    for (c, lab), colr in zip(cols, ["#2e8b57", "#2a78d6", "#9a9a94"], strict=True):
        ax.barh(r["bank"], r[c], left=left, color=colr, label=lab)
        left += r[c].to_numpy()
    ax.scatter(r["market_total_return_log"], r["bank"], color="#1f1f1d", zorder=3, label="total (log)")
    style(ax, "Decomposition of the log price change, 31 Dec 2025 to 31 Oct 2026", "")
    ax.set_xlabel("log points", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, png, _foot("chapter 10 (NMB post-split basis)", "Estimated"))
    return r[["bank", "market_total_return_log"] + [c for c, _ in cols] + ["evidence_class"]], "Estimated"


def tested_call(root, png):
    t = _t(root, "ch11_tested_call.csv")
    steps = [("hist_value_per_share", "31 Dec 2025 value"), ("bridge_time", "time"), ("bridge_dividends", "dividends"),
             ("bridge_new_information", "new information"), ("bridge_forecast_revisions", "forecast revisions"),
             ("bridge_coe", "cost of equity"), ("curr_value_same_basis", "31 Oct 2026 value")]
    fig, axs = plt_subplots(1, len(t), figsize=(10, 4.0), squeeze=False)
    rows = []
    for ax, r in zip(axs[0], t.itertuples(), strict=True):
        v = [getattr(r, c) for c, _ in steps]
        run = v[0]
        for i, val in enumerate(v):
            if i in (0, len(v) - 1):
                ax.bar(i, val, color=COLORS[BID[r.bank]])
            else:
                ax.bar(i, val, bottom=run, color="#2e8b57" if val >= 0 else "#c0392b")
                run += val
            rows.append({"bank": r.bank, "step": steps[i][1], "tzs_per_share": val})
        ax.axhline(r.actual_price_31oct2026_same_basis, color="#1f1f1d", ls="--", lw=0.8)
        ax.set_xticks(range(len(v)), [s for _, s in steps], rotation=45, ha="right", fontsize=6.5)
        style(ax, f"{r.bank}: value bridge (dashed: 31 Oct 2026 price)", "TZS per share")
    save(fig, png, _foot("chapter 11 tested-call bridge, register v1.3", "Estimated"))
    return pd.DataFrame(rows).assign(evidence_class="Estimated"), "Estimated"


def translation(root, png):
    s = _t(root, "ch12_returns_spans.csv")
    s = s[s["period"].str.startswith("31 Dec 2025")]
    fig, ax = plt_subplots(figsize=(7.0, 3.6))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s["tzs_return"] * 100, width=0.4, color="#2a78d6", label="TZS")
    ax.bar(x + 0.2, s["usd_return"] * 100, width=0.4, color="#9a9a94", label="USD")
    for i, r in enumerate(s.itertuples()):
        ax.text(i, max(r.tzs_return, r.usd_return) * 100 + 3, f"cost {(r.tzs_return - r.usd_return) * 100:.1f} pts",
                ha="center", fontsize=7)
    ax.set_xticks(x, s["bank"])
    style(ax, "Total return in TZS and USD, 31 Dec 2025 to 31 Oct 2026", "percent")
    ax.legend(fontsize=7, frameon=False)
    save(fig, png, _foot("chapter 12 (BoT end-period TZS/USD)", "Derived"))
    return s[["bank", "period", "tzs_return", "usd_return", "fx_start", "fx_end", "evidence_class"]], "Derived"


def depreciation(root, png):
    u = _t(root, "ch13_usd_exports.csv")
    e = u[(u["measure"] == "expected USD return") & (u["valuation"] == "CURR")]
    fig, ax = plt_subplots(figsize=(7.0, 3.8))
    dd = np.linspace(0, 0.25, 51)
    rows = []
    from tzbank.chapters.ch12_usd_implementation import _cfg, usd_total_return

    w = float(_cfg(root)["div_wht_foreign"])
    x = _t(root, "ch12_expected_usd.csv")
    dv = _t(root, "ch09_dividends_12m.csv")
    for b, g in e.groupby("bank"):
        pr = float(x[(x["valuation"] == "CURR") & (x["bank"] == b)]["tzs_price_return"].iloc[0])
        divs = [(float(r.dps) / float(r.price), float(r.years_from_valuation))
                for r in dv[(dv["valuation"] == "CURR") & (dv["bank"] == b)].itertuples()]
        curve = [usd_total_return(pr, divs, w, d) for d in dd]  # chapter 12 method (prompt 24)
        ax.plot(dd * 100, np.array(curve) * 100, color=COLORS[BID[b]], lw=2, label=b)
        ax.scatter(g["depreciation"] * 100, g["value"] * 100, color=COLORS[BID[b]], s=18, zorder=3)
        rows += [{"bank": b, "depreciation": d, "expected_usd_return": c} for d, c in zip(dd, curve, strict=True)]
    h = float(e["usd_hurdle"].iloc[0])
    ax.axhline(h * 100, color="#1f1f1d", ls="--", lw=0.8)
    ax.text(0, h * 100, f" USD hurdle {h:.1%}", fontsize=7, va="bottom")
    style(ax, "Expected 12-month USD return against annual depreciation (31 Oct 2026; dots: 0, 2.3, 3.8, 24.5%)",
          "percent")
    ax.set_xlabel("TZS depreciation against the USD, percent a year", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    save(fig, png, _foot("chapter 12 (USD return before trading cost)", "Scenario"))
    return pd.DataFrame(rows).assign(usd_hurdle=h, evidence_class="Scenario"), "Scenario"


def price_12m(root, png, bank):
    """One bank's 12-month close with the value and rating price lines (report page-1 sidebar; prompt 16)."""
    px, ln = _t(root, "ch13_price_series.csv"), _t(root, "ch13_price_lines.csv")
    g = px[px["bank"] == bank].sort_values("date")
    lines = ln[(ln["bank"] == bank) & ln["line"].str.match(r"(value per share|Sell above)")]
    # a Buy needs both tests (return above COE + 5 pp and bear-case downside within 25%): the Buy line is the lower of
    # the two prices (prompt 24; D4)
    buy = ln[(ln["bank"] == bank) & ln["line"].str.startswith("Buy below")]
    if len(buy):
        lines = pd.concat([lines, pd.DataFrame([{"bank": bank, "line": "Buy below (both tests)",
                                                 "price": float(buy["price"].min()), "evidence_class": "Scenario"}])],
                          ignore_index=True)
    fig, ax = plt_subplots(figsize=(4.2, 3.4))
    ax.plot(pd.to_datetime(g["date"]), g["adj_close"], color=COLORS[BID[bank]], lw=1.3, label=f"{bank} close")
    tone = {"v": ("#1f1f1d", ":"), "S": ("#d1453b", "-"), "B": ("#1baf7a", "-")}
    for r in lines.sort_values("price", ascending=False).itertuples():
        c, ls = tone[r.line[0]]
        lab = "Buy below (both tests)" if r.line.startswith("Buy") else r.line.split(" (")[0]
        ax.axhline(r.price, color=c, ls=ls, lw=0.8, label=f"{lab} {r.price:,.0f}")
    style(ax, f"{bank}: 12-month price (TZS, post-split basis)", "TZS")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    save(fig, png, _foot("DSE daily closes; value and rating lines from the base-case value path (D4)", "Observed; lines "
                         "Scenario"))
    out = pd.concat([g.assign(series="close"), lines.rename(columns={"line": "series", "price": "adj_close"})])
    return out.assign(evidence_class=lambda d: d["evidence_class"].fillna("Observed")), "Observed"


# --- the exhibit map: (report, exhibit no, title, source); a source is a figure file or (function, kwargs) ------------
EXHIBITS = [
    ("main", 1, "Combined NMB + CRDB share of commercial-bank assets, deposits and loans", "ch01_cr2.png"),
    ("main", 2, "Profit share against asset share, 2011-2025", (profit_share, {})),
    ("main", 3, "Share of banking and of the listed equity market", (dse_shares, {})),
    ("main", 4, "Loan-book mix by sector against the commercial-bank system", "ch02_loan_mix.png"),
    ("main", 5, "Binding capital headroom, seven-bank panel", (headroom, {})),
    ("main", 6, "Shapley attribution of the NMB-CRDB ROE gap", "ch04_4_0_gap_waterfall.png"),
    ("main", 8, "Mean excess return, seven-bank panel", (excess_returns, {})),
    ("main", 9, "Net deposit franchise against the T-bill rate", "ch06_franchise.png"),
    ("main", 10, "ROE effect of CASA migration to the T-bill", (deposit_migration, {})),
    ("main", 11, "FY2025 payout against modelled payout ceilings", (distributable, {})),
    ("main", 12, "Cost of risk, FY2010-25, with the through-the-cycle mean", "ch08_cost_of_risk.png"),
    ("main", 13, "Risk-adjusted return on RWA", "ch08_rorwa.png"),
    ("main", 15, "Forward profit bridge FY2025 to FY2027 by driver, base, bear and bull", (earnings_bridge, {})),
    ("main", 17, "P/B against ROE: panel, own history, African peers and the global anchor",
     (pb_roe_panel, {"own_history": True})),
    ("main", 18, "Decomposition of the log price change", (rerating, {})),
    ("main", 19, "P(value above price) against the weight on the stress regime", "ch13_stress_mixture.png"),
    ("main", 20, "Bridge from the 31 Dec 2025 value to the 31 Oct 2026 value", (tested_call, {})),
    ("main", 21, "Total return in TZS and USD", (translation, {})),
    ("main", 22, "Expected USD return against depreciation", (depreciation, {})),
    ("nmb", 3, "NMB loan book by sector against the system", (sector_mix, {"bank": "NMB"})),
    ("nmb", 5, "NMB ROE and P/B against listed African banks", (pb_roe_panel, {"highlight": "NMB"})),
    ("nmb", 6, "NMB: from reported to sustainable ROE", (roe_bridge, {"bank": "NMB"})),
    ("nmb", 9, "NMB P/B and ROE since listing (own history)", (own_history_roe, {"bank": "NMB"})),
    ("nmb", 10, "NMB forward profit bridge FY2025 to FY2027", (earnings_bridge, {"bank": "NMB",
                                                                                 "scen": ("base", "bear")})),
    ("nmb", 15, "NMB share price, 12 months, with the value and rating lines", (price_12m, {"bank": "NMB"})),
    ("crdb", 3, "CRDB loan book by sector against the system", (sector_mix, {"bank": "CRDB"})),
    ("crdb", 5, "CRDB ROE and P/B against listed African banks", (pb_roe_panel, {"highlight": "CRDB"})),
    ("crdb", 6, "CRDB: from reported to sustainable ROE", (roe_bridge, {"bank": "CRDB"})),
    ("crdb", 9, "CRDB P/B and ROE since listing (own history)", (own_history_roe, {"bank": "CRDB"})),
    ("crdb", 10, "CRDB forward profit bridge FY2025 to FY2027", (earnings_bridge, {"bank": "CRDB",
                                                                                   "scen": ("base", "bear")})),
    ("crdb", 15, "CRDB share price, 12 months, with the value and rating lines", (price_12m, {"bank": "CRDB"})),
    ("crdb", 16, "CRDB P/B since listing with ROE (own history)", (own_history_roe, {"bank": "CRDB"})),
    ("foreign", 2, "Share of banking and of the listed market", (dse_shares, {})),
    ("foreign", 5, "Tanzanian banks against listed African banks", (pb_roe_panel, {})),
    ("foreign", 7, "Total return in TZS and USD", (translation, {})),
    ("foreign", 9, "Expected USD return against depreciation", (depreciation, {})),
]


def figure_rows(root: Path) -> list[dict]:
    """One index row per chapter figure (``report = figure``), so a report may cite a figure stem directly
    (prompt 16)."""
    figd = path("figures", root)
    rows = []
    for png in sorted(figd.glob("*.png")):
        d = png.with_name(f"{png.stem}_data.csv")
        ev = pd.read_csv(d)["evidence_class"].iloc[0] if d.exists() else "Derived"
        rows.append({"report": "figure", "report_title": "chapter figure", "exhibit_no": pd.NA, "title": png.stem,
                     "file": png.relative_to(root).as_posix(),
                     "data": d.relative_to(root).as_posix() if d.exists() else "", "source": "chapter figure",
                     "evidence_class": ev, "export_key": "", "figure_stem": png.stem})
    return rows


def refresh_index(root: Path) -> list[Path]:
    """Rewrite the chapter-figure rows of outputs/exhibits/index.csv after the last chart stage (the support stage
    runs before chapter 14 draws its figures); the report exhibit rows are kept as written."""
    f = path("outputs", root) / "exhibits" / "index.csv"
    if not f.exists():
        return []
    i = pd.read_csv(f, dtype={"exhibit_no": "Int64"}, keep_default_na=False, na_values=[""])
    i = pd.concat([i[i["report"] != "figure"], pd.DataFrame(figure_rows(root))], ignore_index=True)
    return [write_csv(i.astype({"exhibit_no": "Int64"}), f, require_evidence=True)]


PEER_FNS = ("pb_roe_panel", "own_history_roe")


def redraw_peer_exhibits(root: Path) -> list[Path]:
    """Redraw the exhibits that read the prompt 19 peer tables, after the peer stage has written them (the support
    stage draws every exhibit first, before the peer tables exist on a clean run; prompt 23)."""
    out = path("outputs", root) / "exhibits"
    files = []
    for rep, no, _title, src in EXHIBITS:
        if isinstance(src, str) or src[0].__name__ not in PEER_FNS:
            continue
        fn, kw = src
        png = out / rep / f"{no:02d}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        df, _ev = fn(root, png, **kw)
        files += [png, write_csv(df, png.with_name(f"{no:02d}_data.csv"), require_evidence=True)]
    return files


def write_exhibits(root: Path) -> list[Path]:
    """Write every chart exhibit and outputs/exhibits/index.csv (pre-flight E)."""
    out = path("outputs", root) / "exhibits"
    figd = path("figures", root)
    files, idx = [], []
    for rep, no, title, src in EXHIBITS:
        png = out / rep / f"{no:02d}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        data_csv = png.with_name(f"{no:02d}_data.csv")
        if isinstance(src, str):
            shutil.copyfile(figd / src, png)
            d = figd / src.replace(".png", "_data.csv")
            if d.exists():
                shutil.copyfile(d, data_csv)
            ev = pd.read_csv(d)["evidence_class"].iloc[0] if d.exists() else "Derived"
            how = f"copy of outputs/figures/{src}"
        else:
            fn, kw = src
            df, ev = fn(root, png, **kw)
            write_csv(df, data_csv, require_evidence=True)
            how = f"drawn by tzbank.chapters.ch13_exhibits.{fn.__name__}"
        files += [png, data_csv]
        idx.append({"report": rep, "report_title": REPORTS[rep], "exhibit_no": no, "title": title,
                    "file": png.relative_to(root).as_posix(), "data": data_csv.relative_to(root).as_posix(),
                    "source": how, "evidence_class": ev, "export_key": f"{rep}/{no:02d}",
                    "figure_stem": src.removesuffix(".png") if isinstance(src, str) else f"{rep}_{no:02d}"})
    idx += figure_rows(root)
    files.append(write_csv(pd.DataFrame(idx).astype({"exhibit_no": "Int64"}), out / "index.csv",
                           require_evidence=True))
    return files
