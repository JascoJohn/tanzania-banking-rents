"""Prompt 21 report exhibits E1-E6 (D90), drawn from pipeline tables (never pasted images).

The owner's samples in ``reports/design/samples/`` are design references only. House rules: forest / bottle / teal
palette, gold only for thresholds and callouts, NMB orange and CRDB blue in every exhibit, no embedded titles (the
caption comes from the report's exhibit marker). E1 and E6 are diagrams (``fig._house_diagram``): the report theme only
rescales them.

- E1 ``ch13_p21_thesis``: thesis architecture, seven steps across Parts I-III, gate before valuation (main, landscape).
- E2 ``ch13_p21_roe_bridge``: reported -> sustainable ROE against the ROE the price requires (one panel per bank).
- E3 ``ch13_p21_dupont``: extended DuPont tree with every Shapley effect on the FY2025 NMB - CRDB ROE gap.
- E4 ``ch13_p21_surplus``: cost of holding surplus capital by hurdle (binding-headroom basis).
- E5 ``ch13_p21_rating_map``: 12-month expected return against the Sell / Hold / Buy bands, with COE switch points.
- E6 ``ch13_p21_transmission``: external shocks -> bank channels -> shareholder consequences, with sensitivities.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path
from tzbank.io import write_csv, write_json, write_text

FOREST, BOTTLE, TEAL, GOLD = "#006B3F", "#06452E", "#66A88A", "#E6B323"
PALE, BAND, INK, MUTED = "#E5F1E8", "#F3F7F4", "#1f1f1d", "#5a5a55"
NMB, CRDB = COLORS["B02"], COLORS["B01"]
BANK_C = {"NMB": NMB, "CRDB": CRDB}


def _t(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, low_memory=False, dtype={"bank_id": str})


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def _wrap(s: str, n: int) -> str:
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


# --- figures collected from the pack ----------------------------------------------------------------------------------
def facts(root: Path) -> dict:
    """Every number the exhibits print, read from pipeline tables (each traceable to a findings-pack id)."""
    f: dict = {}
    cr2 = _t(root, "ch13_cr2_profit_direct.csv")
    solo = cr2[cr2["sector_basis"].str.startswith("all banking") & cr2["numerator_basis"].str.contains("Bank-solo")
               & (cr2["year"] == 2025)].iloc[0]
    f["assets_share"], f["pbt_share"] = solo["cr2_asset_share"], solo["cr2_pbt_share"]
    dse = _t(root, "ch01_dse_shares.csv")
    dom = dse[dse["measure"].str.contains("domestic", case=False)].sort_values("period")
    f["dse_domestic"] = float(dom["share"].iloc[-1])
    ics = _t(root, "ch13_implied_credit_shares.csv")
    pair = ics[(ics["book"] == "NMB + CRDB") & (ics["fiscal_year"] == ics[ics["book"] == "NMB + CRDB"]["fiscal_year"]
                                                .max())].set_index("category")["implied_share"]
    f["agri_pair"], f["hh_pair"] = pair["agriculture"], pair["households and personal"]
    f["credit_fy"] = int(ics[ics["book"] == "NMB + CRDB"]["fiscal_year"].max())
    lm = _t(root, "ch13_text_loan_mix.csv").set_index(["book", "category"])["share_latest"]
    f["nmb_hh"], f["crdb_agri"] = lm[("NMB", "households and personal")], lm[("CRDB", "agriculture")]
    fin = _t(root, "ch13_financial_table.csv")
    fy = fin[fin["fiscal_year"] == "FY2025A"].set_index("bank")
    dp = _t(root, "ch04_dupont_panel.csv")
    d25 = dp[dp["fiscal_year"] == 2025].set_index("bank_id")
    cap = _t(root, "ch07_capital.csv")
    c25 = cap[cap["fiscal_year"] == 2025].set_index("bank_id")
    er = _t(root, "ch05_excess_returns.csv")
    e25 = er[er["fiscal_year"] == 2025].set_index("bank_id")["er"]
    reg = pd.read_csv(root / "config" / "assumption_register.csv")
    v = _t(root, "ch09_valuations.csv")
    vc = v[v["valuation"] == "CURR"].set_index("bank")
    cw = _t(root, "ch13_dec_crossing_weights.csv")
    cw = cw[cw["valuation"] == "CURR"].set_index("bank")["crossing_weight"]
    imp = _t(root, "ch10_implied.csv")
    imp = imp[imp["valuation"] == "CURR"].set_index("bank")
    for b, bid in (("NMB", "B02"), ("CRDB", "B01")):
        r = reg[(reg["valuation"] == "CURR") & (reg["bank"] == b) & (reg["symbol"] == "ER_star")].iloc[0]
        f[b] = {"roe": fy.loc[b, "ROE"], "cor": d25.loc[bid, "cost_of_risk"], "cir": fy.loc[b, "CIR"],
                "headroom": c25.loc[bid, "binding_headroom"], "er25": e25[bid], "er_full": r["base"],
                "er_break": r["bull"], "value": vc.loc[b, "value_per_share"], "price": vc.loc[b, "price"],
                "exp": vc.loc[b, "expected_total_return"], "rating": vc.loc[b, "rating"], "crossing": cw[b],
                "implied_er": imp.loc[b, "implied_er_star_at_ch5_lambda"], "implied_roe": imp.loc[b, "implied_roe_star"]}
    f["coe"] = float(vc["coe"].iloc[0])
    return f


# --- diagrams: drawn at their placed size, text 8 pt or more (prompt 22) ------------------------------------------------
CM_IN = 1 / 2.54
MIN_DIAGRAM_PT = 8.0


def _canvas(w_cm: float, h_cm: float):
    """A figure exactly w x h cm with one axes covering it in cm units (no scaling later: fonts stay as drawn)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w_cm * CM_IN, h_cm * CM_IN))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_cm)
    ax.set_ylim(0, h_cm)
    ax.axis("off")
    fig._house_diagram = True  # noqa: SLF001
    return fig, ax


def _txt(ax, x, y, s, size=8.0, **kw):
    return ax.text(x, y, s, fontsize=max(size, MIN_DIAGRAM_PT), clip_on=False, zorder=5, **kw)


def _bank_shares(root: Path) -> dict:
    """Each bank's share of sector assets and pre-tax profit, 2025 (CRDB bank-solo, NMB Group = Bank)."""
    cr2 = _t(root, "ch13_cr2_profit_direct.csv")
    sec = cr2[cr2["sector_basis"].str.startswith("all banking") & (cr2["year"] == 2025)].iloc[0]
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    w = w[w["fiscal_year"] == 2025].set_index("bank_id")
    return {"NMB": (w.loc["B02", "total_assets"] / sec["sector_total_assets"],
                    w.loc["B02", "profit_before_tax"] / sec["sector_pbt"]),
            "CRDB": (w.loc["B01", "bank_only_total_assets"] / sec["sector_total_assets"],
                     w.loc["B01", "bank_only_pbt"] / sec["sector_pbt"])}


# --- E1 ---------------------------------------------------------------------------------------------------------------
def e1(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    f = facts(root)
    n, c = f["NMB"], f["CRDB"]
    sh = _bank_shares(root)
    steps = ["Market\nposition", "Business\nfranchises", "Financial\nperformance", "Shareholder\nreturns",
             "Economic\nrent", "Valuation", "Investment\nconclusion"]
    question = ["Are they central to the banking system?", "What is each franchise built on?",
                "How do the franchises become profit?", "What return do shareholders get?",
                f"Does ROE beat the {pct(f['coe'])} cost of equity, and does it last?", "What is a lasting rent worth?",
                "Does value justify the price?"]
    together = [f"{pct(f['assets_share'])} of banking assets, {pct(f['pbt_share'])} of sector pre-tax profit (2025); "
                f"{pct(f['dse_domestic'])} of DSE domestic market value",
                f"{pct(f['agri_pair'], 0)} of commercial-bank agricultural and {pct(f['hh_pair'], 0)} of household "
                f"credit (FY{f['credit_fy']})", "", "", "", "", ""]

    def row(b, x, mix, earn):
        return [f"{pct(sh[b][0])} of banking assets, {pct(sh[b][1])} of sector pre-tax profit (2025)",
                mix, f"{earn}. Cost of risk {pct(x['cor'], 2)}; cost-to-income {pct(x['cir'], 1)}",
                f"FY2025 ROE {pct(x['roe'])}; capital headroom {x['headroom'] * 100:.2f} pts",
                f"{x['er25'] * 100:+.1f} pts over the cost of equity in FY2025; {pct(x['er_full'])} a year over the "
                "full cycle", f"Value TZS {x['value']:,.0f} per share",
                f"Value {x['value']:,.0f} vs price {x['price']:,.0f}; 12-month return {x['exp'] * 100:.1f}%. "
                f"{x['rating'].upper()}"]
    nrow = row("NMB", n, f"Households and personal {pct(f['nmb_hh'], 0)} of the book",
               "Earns on assets: yield, fees, cheap deposits")
    crow = row("CRDB", c, f"Agriculture {pct(f['crdb_agri'], 0)} of the book, policy-supported",
               "Earns on equity: leverage and a kind credit cycle")
    finding = ["Central, not monopolistic; the safety net protects depositors and the state",
               "Both have a cheap-deposit moat against smaller banks", "Net of credit losses the loan yields are close",
               "ROE = operating performance x leverage x capital allocation",
               "Conditional on the credit cycle and the bill rate",
               "Breaks on a repeat 2015-18 stress, the bill rate or the cost of equity",
               f"Price implies {pct(n['implied_er'])} (NMB) and {pct(c['implied_er'])} (CRDB) excess return; value "
               f"meets price only at {n['crossing'] * 100:.0f}% / {c['crossing'] * 100:.1f}% weight on 2015-18 being "
               "exceptional"]
    # 25 x 12.8 cm: with its caption and source line the exhibit fits the 17 cm body of the landscape page (prompt 22)
    W, H = 25.0, 12.8
    fig, ax = _canvas(W, H)
    x0, gap = 2.55, 0.08
    w = (W - x0 - 0.15 - 6 * gap) / 7
    xs = [x0 + i * (w + gap) for i in range(7)]
    for lab, a, b in (("PART I  Is the dominance real?", 0, 1), ("PART II  Do they earn rents, and do they last?", 2, 4),
                      ("PART III  Is it worth the price?", 5, 6)):
        ax.plot([xs[a] + 0.1, xs[b] + w - 0.1], [12.05, 12.05], color=TEAL, lw=1.0)
        _txt(ax, (xs[a] + xs[b] + w) / 2, 12.15, lab, 9, ha="center", va="bottom", color=FOREST, fontweight="bold")
    for i, (x, s) in enumerate(zip(xs, steps, strict=True)):
        col = BOTTLE if i < 5 else (FOREST if i == 5 else TEAL)
        ax.add_patch(Polygon([(x, 10.95), (x + w - 0.3, 10.95), (x + w, 11.45), (x + w - 0.3, 11.95), (x, 11.95),
                              (x + (0.3 if i else 0), 11.45)], closed=True, color=col, lw=0))
        _txt(ax, x + 0.45, 11.45, str(i + 1), 11, color="white", fontweight="bold", va="center")
        _txt(ax, x + 0.85, 11.45, s, 8.5, color="white", fontweight="bold", va="center")
    # (label, cells, top, height, fill, bold, colour): heights follow the longest wrapped cell of each row
    rows = [("Key question", question, 10.85, 1.45, PALE, True, INK),
            ("Together", together, 9.4, 1.95, "white", False, INK),
            ("NMB", nrow, 7.45, 1.75, BAND, False, INK), ("CRDB", crow, 5.7, 1.75, "white", False, INK),
            ("Report finding", finding, 3.95, 2.4, BAND, False, MUTED)]
    chars = int(w / 0.13)  # about 0.125 cm per character at 8 pt Calibri, measured on the 300 dpi render
    for lab, cells, top, h, bg, bold, colr in rows:
        y = top - h / 2
        ax.add_patch(FancyBboxPatch((0.1, top - h), W - 0.25, h, boxstyle="square,pad=0", color=bg, lw=0))
        _txt(ax, 0.25, y, _wrap(lab, 12), 9, color=BANK_C.get(lab, FOREST), fontweight="bold", va="center")
        for i, (x, s) in enumerate(zip(xs, cells, strict=True)):
            if not s:
                continue
            sell = lab in ("NMB", "CRDB") and i == 6
            _txt(ax, x + 0.08, y, _wrap(s, chars), 8, va="center", color=BOTTLE if sell else colr,
                 fontweight="bold" if (sell or bold) else None, style="italic" if lab == "Report finding" else None,
                 linespacing=1.15)
    gx = xs[5] - gap / 2
    ax.plot([gx, gx], [1.5, 11.95], color=GOLD, lw=1.6, ls=(0, (4, 2)))
    _txt(ax, gx, 1.33, "GATE", 8, color=GOLD, fontweight="bold", ha="center", va="center")
    ax.add_patch(FancyBboxPatch((0.1, 0.05), W - 0.25, 1.1, boxstyle="square,pad=0", color=BOTTLE, lw=0))
    _txt(ax, 0.25, 0.6, "So what", 9, color="white", fontweight="bold", va="center")
    _txt(ax, x0, 0.6, _wrap("Steps 1-5 show a real franchise earning high returns. The gate: a large, profitable bank is not "
         "the same as an attractively priced share. The price pays for more rent than either bank's full-cycle record "
         "supports, so both are rated Sell.", int((W - x0 - 0.3) / 0.13)), 8, color="white", va="center")
    data = pd.DataFrame({"row": [r[0] for r in rows for _ in range(7)], "step": list(range(1, 8)) * len(rows),
                         "text": [s for r in rows for s in r[1]]}).assign(evidence_class="Derived")
    return save(fig, figd / "ch13_p21_thesis.png", "Source: findings pack; chapters 1-12. Evidence class: Derived."), \
        data



# --- E2 ---------------------------------------------------------------------------------------------------------------
def e2(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    br = _t(root, "ch13_roe_sustainable_bridge.csv")
    imp = _t(root, "ch10_implied.csv")
    imp = imp[imp["valuation"] == "CURR"].set_index("bank")["implied_roe_star"]
    fig, axs = plt_subplots(1, 2, figsize=(11, 4.0), sharey=True)
    rows = []
    for ax, b in zip(axs, ("NMB", "CRDB"), strict=True):
        g = br[(br["bank"] == b) & (br["step_no"] != 2)].sort_values("step_no")  # the rate step is zero by construction
        v = g["roe_pts"].to_numpy() * 100
        r = np.round(v, 1)
        r[-2] = round(r[-1] - r[0] - r[1:-2].sum(), 1)  # residual absorbs rounding: bars add to the printed total
        labels = ["Reported\nFY2025", "Cost of risk to\nthrough-cycle", "Leverage to\nthe k* path",
                  "Fade to ROE*,\nmix, efficiency", "Sustainable\nROE*"]
        ticks = ["Reported\nFY2025", "Cost of\nrisk", "Leverage", "Fade to\nROE*", "ROE*"]  # 7 pt fits the bar pitch
        level = r[0]
        for i, (lab, x) in enumerate(zip(labels, r, strict=True)):
            if i in (0, len(r) - 1):
                ax.bar(i, x, color=BANK_C[b], width=0.62)
                ax.text(i, x + 0.4, f"{x:.1f}", ha="center", fontsize=8, fontweight="bold")
            else:  # label at the bar's own end (below a fall), so neighbouring labels never share a height
                ax.bar(i, x, bottom=level, color=TEAL, width=0.62)
                end = level + x
                ax.text(i, end - 0.35 if x < 0 else end + 0.35, f"{x:+.1f}", ha="center",
                        va="top" if x < 0 else "bottom", fontsize=8)
                level += x
            rows.append({"bank": b, "step": lab.replace("\n", " "), "roe_pts_exact": v[i], "roe_pts_printed": x,
                         "evidence_class": "Scenario"})
        req = imp[b] * 100
        ax.axhline(req, color=GOLD, lw=1.6, ls="--")
        ax.set_xlim(-0.6, len(r) + 0.9)  # room right of the last bar for the requirement label, outside the bars
        ax.text(len(r) - 0.55, req + 0.6, chr(10).join(["price", "requires", f"{req:.1f}"]), color=GOLD, ha="left",
                va="bottom", fontsize=8, fontweight="bold")  # above the line, right of the bars (prompt 22)
        ax.set_xticks(range(len(r)), ticks, fontsize=7)
        style(ax, b, "ROE, percent")
        ax.set_ylim(0, 36)  # headroom over the 30.0 bar for its value label and the panel title
        rows.append({"bank": b, "step": "ROE the 31 Oct 2026 price requires (ch10)", "roe_pts_exact": req,
                     "roe_pts_printed": round(req, 1), "evidence_class": "Estimated"})
    return save(fig, figd / "ch13_p21_roe_bridge.png", "Register v1.3. Steps: cost of risk to its "
                "through-cycle level; leverage to the k* path; "
                "fade to ROE*, mix and efficiency. The T-bill step is zero: the v1.3 path is flat "
                "at the 6.29% auction yield and NIM moves only with changes from it; the fall from 9.49% (2025) is "
                "already in the register NIM path. Evidence class: Scenario."), pd.DataFrame(rows)


# --- E3 ---------------------------------------------------------------------------------------------------------------
GROUPS = [("NII / average\nassets", ["y_iea", "iea_ta", "c_fund", "ibl_ta"], "nii_to_avg_assets", 2),
          ("Non-interest\nincome share", ["nonii_ta"], "non_interest_income_share", 1),
          ("Cost-to-income", ["opex_ta"], "cost_to_income", 1), ("Cost of risk", ["imp_ta"], "cost_of_risk", 2),
          ("Tax and other", ["tax_rate", "other_ta"], None, 1), ("Leverage", ["ta_e"], "leverage", 0),
          ("Minority share", ["attr_share"], None, 1)]


def e3(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    g = _t(root, "ch04_4_0_gap_attribution.csv")
    y = int(g["fiscal_year"].max())
    g = g[g["fiscal_year"] == y].set_index("component")
    gap = float(g["gap"].iloc[0])
    dp = _t(root, "ch04_dupont_panel.csv")
    d = dp[dp["fiscal_year"] == y].set_index("bank_id")
    contrib = {lab: float(g.loc[[k for k in comps if k in g.index], "contribution"].sum()) for lab, comps, *_ in GROUPS}
    raw = {k: v * 10000 for k, v in contrib.items()}  # printed effects (0.01 pt) add up to the printed gap
    base = {k: float(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:int(round(gap * 10000) - sum(base.values()))]:
        base[k] += 1
    shown = {k: v / 100 for k, v in base.items()}
    W, H = 16.5, 9.6
    fig, ax = _canvas(W, H)

    def box(x, yy, w, h, lines, edge=FOREST, fill="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=fill, ec=edge, lw=1.0, zorder=2))
        step = 0.42
        top = yy + step * (len(lines) - 1) / 2
        for i, (s, bold) in enumerate(lines):
            _txt(ax, x, top - i * step, s, 8, ha="center", va="center", color=BOTTLE if bold else INK,
                 fontweight="bold" if bold else None)

    def link(a, b):
        mid = (a[1] + b[1]) / 2
        ax.plot([a[0], a[0], b[0], b[0]], [a[1], mid, mid, b[1]], color=MUTED, lw=0.8, zorder=1)

    def v(bid, col, dec):
        if col is None:
            return ""
        x = float(d.loc[bid, col])
        return f"{x:.1f}x" if col == "leverage" else pct(x, dec)

    roe_n, roe_c = d.loc["B02", "roe"], d.loc["B01", "roe"]
    box(W / 2, 8.55, 6.4, 1.6, [(f"ROE (FY{y})", True), (f"NMB {pct(roe_n)}   CRDB {pct(roe_c)}", False),
                                (f"gap {gap * 100:+.2f} pts", False)], fill=PALE)
    roa_shown = sum(shown[k] for k, *_ in GROUPS[:5])
    lev_shown = shown["Leverage"] + shown["Minority share"]
    box(4.6, 5.9, 6.0, 1.6, [("ROA", True), (f"NMB {pct(d.loc['B02', 'roa'], 2)}   CRDB {pct(d.loc['B01', 'roa'], 2)}",
                                             False), (f"effect {roa_shown:+.2f} pts", False)])
    box(12.6, 5.9, 6.0, 1.6, [("x Leverage (assets / equity)", True),
                              (f"NMB {d.loc['B02', 'leverage']:.1f}x   CRDB {d.loc['B01', 'leverage']:.1f}x", False),
                              (f"effect {lev_shown:+.2f} pts", False)])
    link((W / 2, 7.75), (4.6, 6.7))
    link((W / 2, 7.75), (12.6, 6.7))
    bw, bgap = 2.15, 0.12  # seven boxes centred inside the 16.5 cm canvas, with room for the rounded corners
    x_left = (W - 7 * bw - 6 * bgap) / 2
    xs = [x_left + bw / 2 + i * (bw + bgap) for i in range(7)]
    rows = []
    for i, (x, (lab, _, col, dec)) in enumerate(zip(xs, GROUPS, strict=True)):
        lines = [(s, True) for s in lab.split("\n")]
        if col:
            lines += [(f"NMB {v('B02', col, dec)}", False), (f"CRDB {v('B01', col, dec)}", False)]
        lines.append((f"{shown[lab]:+.2f} pts", True))
        box(x, 2.85, bw, 2.75, lines, edge=TEAL)
        link((4.6 if i < 5 else 12.6, 5.1), (x, 4.22))
        rows.append({"fiscal_year": y, "effect": lab.replace("\n", " "), "contribution": contrib[lab],
                     "contribution_printed_pts": shown[lab],
                     "nmb_value": float(d.loc["B02", col]) if col else np.nan,
                     "crdb_value": float(d.loc["B01", col]) if col else np.nan, "evidence_class": "Derived"})
    total = sum(shown.values())
    _txt(ax, W / 2, 0.75, f"Shapley effects on the NMB - CRDB ROE gap add to {total:+.2f} pts, the gap. A positive "
         "effect favours NMB.", 8, ha="center", color=BOTTLE, fontweight="bold")
    rows.append({"fiscal_year": y, "effect": "sum (= ROE gap)", "contribution": sum(contrib.values()),
                 "contribution_printed_pts": total, "nmb_value": np.nan, "crdb_value": np.nan,
                 "evidence_class": "Derived"})
    roa_c, lev_c = float(d.loc["B01", "roa"]), float(d.loc["B01", "leverage"])
    return save(fig, figd / "ch13_p21_dupont.png", f"Chapter 4.0 Shapley attribution, FY{y}. Check: CRDB ROA "
                f"{pct(roa_c, 2)} x leverage {lev_c:.1f} = {roa_c * lev_c * 100:.1f}% against a reported "
                f"{pct(float(d.loc['B01', 'roe']))} (period averages). Evidence class: Derived."), pd.DataFrame(rows)



# --- E4 ---------------------------------------------------------------------------------------------------------------
ROLE = [("regulatory minimum", "Regulatory minimum\n(upper bound)"), ("conservation", "Minimum + conservation\nbuffer "
        "(headline)"), ("model target", "Model target\n(lower bound)")]


def e4(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    s = _t(root, "ch13_surplus_drag.csv")
    fig, ax = plt_subplots(1, 1, figsize=(8.5, 4.4))
    rows = []
    for b in ("NMB", "CRDB"):
        g = s[s["bank"] == b].sort_values("hurdle_ratio")
        ax.plot(g["hurdle_ratio"] * 100, g["drag_share_of_profit"] * 100, color=BANK_C[b], marker="o", lw=1.6,
                label=f"{b}: drag at each hurdle")
        actual = float(g["tier1_ratio"].iloc[0])
        ax.axvline(actual * 100, color=BANK_C[b], lw=0.9, ls=":", label=f"{b}: actual Tier 1 ratio {pct(actual)}")
        for r in g.itertuples():
            lab = next((v for k, v in ROLE if k in r.hurdle), r.hurdle)
            ax.annotate(f"{lab.split(chr(10))[0]}: {pct(r.drag_share_of_profit)}", (r.hurdle_ratio * 100,
                        r.drag_share_of_profit * 100), xytext=(6, 4), textcoords="offset points", fontsize=7,
                        color=BANK_C[b])
            rows.append({"bank": b, "hurdle": r.hurdle, "role": lab.replace("\n", " "), "hurdle_ratio": r.hurdle_ratio,
                         "headroom": r.headroom, "drag_share_of_profit": r.drag_share_of_profit,
                         "evidence_class": "Estimated"})
    style(ax, "", "drag, percent of FY2025 profit")
    ax.set_xlabel("hurdle: required capital ratio in Tier 1 terms on the binding (total-capital) constraint, percent",
                  fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return save(fig, figd / "ch13_p21_surplus.png", "Surplus capital = (ratio - hurdle) x RWA, earning the bill rate "
                "rather than the cost of equity, after tax. Headroom on the binding basis: the smaller of CET1 - 12.5% "
                "and total capital - 14.5%. Evidence class: Estimated."), pd.DataFrame(rows)


# --- E5 ---------------------------------------------------------------------------------------------------------------
def e5(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    r = _t(root, "ch13_dec_rating_rule.csv")
    r = r[r["valuation"] == "CURR"].set_index("bank")
    cw = _t(root, "ch13_dec_crossing_weights.csv")
    cw = cw[cw["valuation"] == "CURR"].set_index("bank")["crossing_weight"]
    lo, hi = float(r["sell_threshold"].iloc[0]), float(r["buy_threshold"].iloc[0])
    fig, ax = plt_subplots(1, 1, figsize=(9.5, 3.4))
    xmin, xmax = -35, 40
    for a, b_, c, lab in ((xmin, lo * 100, "#F4E3E3", "SELL"), (lo * 100, hi * 100, BAND, "HOLD"),
                          (hi * 100, xmax, PALE, "BUY")):
        ax.axvspan(a, b_, color=c, lw=0)
        ax.text((a + b_) / 2, 2.6, lab, ha="center", fontsize=8, fontweight="bold", color=BOTTLE)
    ax.text((hi * 100 + xmax) / 2, 1.9, "also needs a bear-case" + chr(10) + "downside no worse" + chr(10) + "than -25%",
            ha="center",
            fontsize=6.8, color=BOTTLE)
    for x in (lo, hi):
        ax.axvline(x * 100, color=GOLD, lw=1.4)
        ax.text(x * 100, 2.25, f"{pct(x, 2)}", ha="center", fontsize=7.5, color=GOLD, fontweight="bold")
    rows = []
    for yv, b in ((1.5, "NMB"), (0.6, "CRDB")):
        x = r.loc[b]
        ax.plot([x["expected_return"] * 100], [yv], marker="D", ms=9, color=BANK_C[b])
        ax.text(x["expected_return"] * 100, yv + 0.22, f"{b} {x['expected_return'] * 100:.1f}%: {x['rating']}",
                ha="center", fontsize=8, color=BANK_C[b], fontweight="bold")
        sw = x["coe_change_to_switch_down"]
        ax.text(x["expected_return"] * 100, yv - 0.3, f"Hold if COE {sw * 100:+.2f} pts; bear downside "
                f"{x['bear_downside'] * 100:.1f}%", ha="center", fontsize=7, color=MUTED)
        rows.append({"bank": b, "expected_return": x["expected_return"], "sell_threshold": lo, "buy_threshold": hi,
                     "rating": x["rating"], "coe_switch": sw, "bear_downside": x["bear_downside"],
                     "crossing_weight": cw[b], "evidence_class": "Scenario"})
    ax.set_xlim(xmin, xmax), ax.set_ylim(0, 3.0), ax.set_yticks([])
    style(ax, "", "")
    ax.set_xlabel("expected 12-month total return, percent (31 October 2026)", fontsize=8)
    return save(fig, figd / "ch13_p21_rating_map.png", f"D4 rule: COE {pct(float(r['coe'].iloc[0]), 2)} +/- 5 pts. "
                f"Value meets price at a {cw['NMB'] * 100:.0f}% (NMB) / {cw['CRDB'] * 100:.1f}% (CRDB) weight on "
                "2015-18 being exceptional. Evidence class: Scenario."), pd.DataFrame(rows)


# --- E6 ---------------------------------------------------------------------------------------------------------------
def e6(root: Path, figd: Path) -> tuple[Path, pd.DataFrame]:
    fb = _t(root, "ch13_franchise_breakeven.csv").drop_duplicates("bank").set_index("bank")["breakeven_tbill"]
    br = _t(root, "ch13_roe_sustainable_bridge.csv")
    cr = br[br["step_no"] == 1].set_index("bank")["roe_pts"]
    cap = _t(root, "ch07_capital.csv")
    hr = cap[cap["fiscal_year"] == 2025].set_index("bank_id")["binding_headroom"]
    tb = _t(root, "ch13_text_thesis_breaks.csv")
    beta = tb[tb["indicator"].str.startswith("deposit beta")].set_index("bank")["current"]
    usd = _t(root, "ch12_expected_usd.csv")
    pace = float(usd[(usd["valuation"] == "CURR") & (usd["scenario"] == "realised_ytd_annualised")]
                 ["depreciation"].iloc[0])
    lm = _t(root, "ch13_text_loan_mix.csv").set_index(["book", "category"])["share_latest"]
    cp = _t(root, "ch13_dec_capital_policies.csv")
    dsib = cp[(cp["bank"] == "CRDB") & cp["policy"].str.contains("2.0")]["value_per_share"].iloc[0]
    shocks = [("Lower T-bill rates", f"Bill 6.29% vs break-even {pct(fb['NMB'], 2)} (NMB), {pct(fb['CRDB'], 2)} (CRDB)",
               "Margin compression; smaller deposit franchise", "both"),
              ("Credit deterioration", f"Cost of risk to through-cycle: {cr['NMB'] * 100:+.1f} / {cr['CRDB'] * 100:+.1f} "
               "ROE pts", "Higher impairments, lower earnings", "both"),
              ("Higher capital requirements", f"Headroom {hr['B02'] * 100:.2f} / {hr['B01'] * 100:.2f} pts; CRDB value "
               f"{dsib:,.0f} at a 2-pt D-SIB surcharge", "Less headroom and dividend capacity", "both"),
              ("Deposit repricing", f"Deposit beta {beta['NMB']:.2f} / {beta['CRDB']:.2f} vs a 0.10 review trigger",
               "Higher funding costs", "both"),
              ("Currency depreciation", f"2026 pace {pct(pace)} a year vs 3.8% assumed",
               "FX translation of subsidiaries", "CRDB"),
              ("Withdrawal of public risk-sharing", f"Agriculture {pct(lm[('CRDB', 'agriculture')], 0)} of the CRDB book",
               "Higher agricultural credit risk", "CRDB")]
    W, H = 16.5, 13.4
    fig, ax = _canvas(W, H)
    # band titles sit rotated in a left margin so no arrow crosses them (prompt 22)
    bands = ((8.0, 5.3, "External shocks, with the report's sensitivity", "#FBF5E1", 26),
             (4.45, 2.95, "Bank transmission channels", BAND, 18), (0.1, 3.75, "Shareholder consequences", PALE, 14))
    for y0, h, lab, fill, n in bands:
        ax.add_patch(FancyBboxPatch((0.05, y0), W - 0.1, h, boxstyle="square,pad=0", fc=fill, ec="none", zorder=0))
        _txt(ax, 0.62, y0 + h / 2, _wrap(lab, n), 8.5, fontweight="bold", color=BOTTLE, rotation=90, ha="center",
             va="center", multialignment="center")
    bw, gap = 2.4, 0.12
    x0 = W - 0.2 - 6 * bw - 5 * gap
    rows = []
    for i, (s, q, ch, who) in enumerate(shocks):
        x = x0 + i * (bw + gap)
        edge = CRDB if who == "CRDB" else FOREST
        ax.add_patch(FancyBboxPatch((x, 8.25), bw, 4.75, boxstyle="round,pad=0.03,rounding_size=0.15", fc="white",
                                    ec=edge, lw=1.1, zorder=2))
        _txt(ax, x + bw / 2, 12.8, _wrap(s, 14), 8.5, ha="center", va="top", fontweight="bold", color=BOTTLE)
        _txt(ax, x + bw / 2, 11.15, _wrap(q, 15), 8, ha="center", va="top", color=FOREST if who == "both" else CRDB)
        ax.add_patch(FancyBboxPatch((x, 4.7), bw, 2.1, boxstyle="round,pad=0.03,rounding_size=0.15", fc="white",
                                    ec=edge, lw=1.1, zorder=2))
        _txt(ax, x + bw / 2, 5.75, _wrap(ch + (" (CRDB)" if who == "CRDB" else ""), 15), 8, ha="center",
             va="center", color=BOTTLE)
        for a, b in ((8.2, 6.85), (4.65, 3.3)):
            ax.add_patch(FancyArrowPatch((x + bw / 2, a), (x + bw / 2, b), arrowstyle="-|>", mutation_scale=8,
                                         color=MUTED, lw=0.8, zorder=1))
        rows.append({"shock": s, "sensitivity": q, "channel": ch, "bank": who, "evidence_class": "Derived"})
    cgap = 0.45
    cw = (6 * bw + 5 * gap - 2 * cgap) / 3
    for i, lab in enumerate(("Lower sustainable ROE", "Lower excess returns and distributable dividends",
                             "Lower intrinsic value")):
        x = x0 + i * (cw + cgap)
        ax.add_patch(FancyBboxPatch((x, 0.45), cw, 2.75, boxstyle="round,pad=0.03,rounding_size=0.15", fc="white",
                                    ec=FOREST, lw=1.3, zorder=2))
        _txt(ax, x + cw / 2, 1.82, _wrap(lab, 24), 9, ha="center", va="center", fontweight="bold", color=BOTTLE)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + cw + 0.05, 1.82), (x + cw + cgap - 0.05, 1.82), arrowstyle="-|>",
                                         mutation_scale=9, color=FOREST, lw=1.1, zorder=3))
    return save(fig, figd / "ch13_p21_transmission.png", "Blue: CRDB-specific links. Sensitivities from chapters 6-8 "
                "and 11-12. Evidence class: Derived."), pd.DataFrame(rows)


NOTE = """# Prompt 21 report exhibits (D90)

Generated by `tzbank.chapters.ch13_p21_exhibits` (stage `market_ext`).

## Research question
Report exhibits E1-E6: thesis architecture, ROE bridge, DuPont tree, surplus capital, rating map, transmission map.

## Variable definitions
As in the source tables (ch13_cr2_profit_direct, ch13_roe_sustainable_bridge, ch04_4_0_gap_attribution,
ch13_surplus_drag, ch13_dec_rating_rule, ch13_franchise_breakeven); capital headroom on the binding basis (D90).

## Sources
Pipeline tables only; owner samples in reports/design/samples are design references.

## Validation test
`tests/test_p21_exhibits.py`.

## Results
{results}

## Evidence class
Derived, Scenario and Estimated as each exhibit's source line states.

## Limitations
- E1 and E6 are diagrams: their text is a summary of the report, every number from the tables above.
"""


def write_p21(ctx) -> list[Path]:
    root = ctx.root
    figd, tab = path("figures", root), path("tables", root)
    nf = path("notes", root) / "ch13_p21.md"
    specs = [(e1, "E1. Thesis architecture", "Derived"), (e2, "E2. ROE bridge", "Scenario"),
             (e3, "E3. DuPont tree with Shapley effects", "Derived"), (e4, "E4. Surplus capital by hurdle", "Estimated"),
             (e5, "E5. Rating decision map", "Scenario"), (e6, "E6. Transmission map", "Derived")]
    files, man_f, res = [], [], []
    for fn, desc, ev in specs:
        png, data = fn(root, figd)
        d = write_csv(data, png.with_name(f"{png.stem}_data.csv"), require_evidence=True)
        t = write_csv(ctx.stamp(data), tab / f"{png.stem}.csv", require_evidence=True)
        files += [png, d, t]
        man_f.append({"path": png.relative_to(root).as_posix(), "data": d.relative_to(root).as_posix(),
                      "description": desc, "evidence_class": ev, "note": nf.relative_to(root).as_posix()})
        res.append(f"- {desc}: {png.stem}")
    fk = forecast_kpis(ctx)
    files.append(write_csv(ctx.stamp(fk), tab / "ch13_p21_forecast_kpis.csv", require_evidence=True))
    files.append(write_text(NOTE.format(results="\n".join(res)), nf))
    m = {"chapter": "13_p21", "title": "Prompt 21 report exhibits", "status": "complete", "hypotheses": [],
         "tables": [{"path": f"outputs/tables/{Path(f['path']).stem}.csv", "description": f["description"],
                     "evidence_class": f["evidence_class"], "note": f["note"]} for f in man_f]
         + [{"path": "outputs/tables/ch13_p21_forecast_kpis.csv", "description": "NPL, coverage and LDR: actuals, "
             "30 Sep 2026 and FY2026-30 forecasts", "evidence_class": "Scenario", "note": nf.relative_to(root).as_posix()}],
         "figures": man_f, "notes": [nf.relative_to(root).as_posix()], "inputs": [],
         "module": "tzbank.chapters.ch13_p21_exhibits", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_p21.json"))
    return files


# --- prompt 21 section 5: NPL, coverage and LDR in the bank reports' key-forecast table ------------------------------
def forecast_kpis(ctx) -> pd.DataFrame:
    """FY2024A-FY2025A actuals, 30 Sep 2026 TTM / quarter-end, FY2026E-FY2030E. LDR (net loans / deposits) from the
    model's projected loans and deposits; NPL ratio and coverage are not model drivers: the NPL ratio moves from FY2025
    towards the bank's through-cycle mean (FY2010-FY2025, chapter 8) by the same share of the way as the projected cost
    of risk moves to its through-cycle level, and coverage stays at FY2025, labelled 'assumption, not modelled'.
    If the through-cycle NPL ratio were below FY2025 while the cost of risk rises to its through-cycle mean,
    the two would contradict and the NPL rows are not printed (the status says so)."""
    from tzbank.banks.engine import project

    root = ctx.root
    fin = _t(root, "ch13_financial_table.csv")
    h8 = _t(root, "ch08_credit_history.csv")
    q = _t(root, "ch13_qtr_five.csv")
    qp = pd.read_csv(path("processed", root) / "quarterly" / "quarterly_panel.csv", low_memory=False)
    rows = []
    for b, bid in (("NMB", "B02"), ("CRDB", "B01")):
        c = ctx.cases[("CURR", bid, "base")]
        fc = project(c.op, c.dr).rows
        a = fin[fin["bank"] == b].set_index("fiscal_year")
        hist = h8[(h8["bank_id"] == bid) & h8["fiscal_year"].between(2010, 2025)]
        npl_ttc, cov25 = float(hist["npl_ratio"].mean()), float(a.loc["FY2025A", "NPL_coverage"])
        cor_up = float(c.dr.CoR[-1]) > float(a.loc["FY2025A", "CoR"])
        ok = not (npl_ttc < float(a.loc["FY2025A", "NPL"]) and cor_up)
        qq = q[(q["bank"] == b) & (q["period_end"] == "2026-09-30")].iloc[0]
        g = qp[(qp["bank"] == b) & (qp["basis"] == "group") & (qp["period_end"] == "2026-09-30")
               & (qp["column_type"] == "current_quarter")].set_index("field")["value"]
        cov_q = float(g["allowances"] / g["npl"]) if {"allowances", "npl"} <= set(g.index) else np.nan
        cols = {"FY2024A": (a.loc["FY2024A", "NPL"], a.loc["FY2024A", "NPL_coverage"], a.loc["FY2024A", "LDR_net"]),
                "FY2025A": (a.loc["FY2025A", "NPL"], a.loc["FY2025A", "NPL_coverage"], a.loc["FY2025A", "LDR_net"]),
                "30 Sep 2026": (qq["npl_ratio"], cov_q, qq["net_ldr"])}
        npl25, cor25, cor_ttc = float(a.loc["FY2025A", "NPL"]), float(a.loc["FY2025A", "CoR"]), float(c.dr.CoR[-1])
        for i, y in enumerate(range(2026, 2031)):
            r = fc[fc["year"] == y]
            ldr = float(r["L"].iloc[0] / r["D"].iloc[0]) if len(r) and pd.notna(r["L"].iloc[0]) else np.nan
            cor_t = float(c.dr.CoR[min(i, len(c.dr.CoR) - 1)])
            frac = float(np.clip((cor_t - cor25) / (cor_ttc - cor25), 0, 1)) if cor_ttc != cor25 else 1.0
            npl_t = npl25 + frac * (npl_ttc - npl25)  # same share of the way to through-cycle as the cost of risk
            cols[f"FY{y}E"] = (npl_t if ok else np.nan, cov25 if ok else np.nan, ldr)
        for i, (metric, basis) in enumerate((
                ("NPL ratio", "actual: stage 3 / gross loans (ch. 8); 30 Sep 2026: BoT statement; forecast: moves "
                 "from FY2025 towards the FY2010-25 mean by the same share of the way as the projected cost of risk "
                 "(assumption, not modelled)"),
                ("NPL coverage", "actual: allowance / stage 3; 30 Sep 2026: line 40 / line 39; forecast: FY2025 level "
                 "(assumption, not modelled)"),
                ("Loan-to-deposit ratio (net)", "net loans / customer deposits; forecast: model's projected loans "
                 "and deposits"))):
            rows.append({"bank": b, "metric": metric, **{k: float(v[i]) for k, v in cols.items()}, "basis": basis,
                         "status": "printed" if ok or i == 2 else "not printed: the through-cycle NPL ratio is below "
                         "FY2025 while the cost of risk rises (contradiction)", "evidence_class": "Scenario"})
    return pd.DataFrame(rows)
