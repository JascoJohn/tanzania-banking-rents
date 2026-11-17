"""Build the Excel companion model (D104): sheets in reading order, live chapter blocks, engine runs, Summary, README.

Python remains the source of truth: every live block reproduces a pipeline table and is checked against it after
recalculation (``tzbank.excel.check``). Statistics, simulations and data-heavy steps enter as grey static blocks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook

from tzbank.config import find_root, path, settings
from tzbank.excel import engine as EG
from tzbank.excel.book import Sheet, a1, define
from tzbank.excel.chapters import Checks, Live, live_table, statics
from tzbank.excel.figures import add_figures, heat
from tzbank.excel.inputs import DATE_LABEL, TICK, build_inputs
from tzbank.valuation.context import build_ctx

DATES = ("HIST", "CURR")
BANKS = ("B02", "B01")
SCEN = ("base", "bear", "bull")
GRID = [x / 1e4 for x in range(-200, 201, 50)]
LAMS = [round(x, 3) for x in np.round(np.arange(0.0, 0.951, 0.05), 3)]
ERS = [round(x, 4) for x in np.round(np.arange(-0.05, 0.1501, 0.005), 4)]
BISECT = 60
LAM_SHAPE = '"lambda"'  # iso surface and implied searches: geometric fade
TITLE = "Tanzania Banking Rents: Excel companion model"
GRID_MAPS: list[dict] = []  # colour-scaled live grids, for the figure map


def _grid_row(rng: str, what: str) -> dict:
    return {"figure": "(live grid shown in the chapter; heatmap)", "exhibit_id": "", "title": what,
            "sheet": rng.split("!")[0], "kind": "heatmap", "chart_type": "colour scale",
            "source_range": rng, "live_links": 0, "data_file": "", "evidence_class": "Derived"}

CHAPTERS = [  # sheet, title, part, script
    ("Ch01", "1. Concentration and market position", "I", "scripts/14_part1.py"),
    ("Ch02", "2. Credit allocation patterns", "I", "scripts/14_part1.py"),
    ("Ch03", "3. Systemic importance", "I", "scripts/14_part1.py"),
    ("Ch04", "4. Banking economics (4.0 DuPont attribution, 4.1-4.7)", "II", "scripts/05_profitability.py"),
    ("Ch05", "5. Persistence and economic rents", "II", "scripts/07_persistence.py"),
    ("Ch06", "6. Funding architecture", "II", "scripts/08_funding.py"),
    ("Ch07", "7. Capital allocation", "II", "scripts/09_capital.py"),
    ("Ch08", "8. Credit risk and capital efficiency", "II", "scripts/10_credit.py"),
    ("Ch09", "9. Comparative valuation", "III", "scripts/13_valuation_results.py"),
    ("Ch10", "10. Market-implied expectations", "III", "scripts/13_valuation_results.py"),
    ("Ch11", "11. Strategic divergence scenarios", "III", "scripts/13_valuation_results.py"),
    ("Ch12", "12. USD returns and implementation", "III", "scripts/15_usd.py"),
    ("Ch13", "13. Report support (exhibits for the reports)", "III", "scripts/16_support.py"),
    ("Ch14", "14. Market behaviour and valuation extensions", "III", "scripts/17_market_ext.py"),
]


def _tables(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(path("tables", root).glob("ch*.csv")):
        out[f.stem] = pd.read_csv(f, low_memory=False)
    return out


def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --- Part I and II: live blocks from chapter tables -------------------------------------------------------------------
def part1_2(sh: dict, ds: Sheet, T: dict, checks: Checks, live: Live) -> None:
    c = sh["Ch01"]
    live_table(c, ds, T["ch01_shares"], ["year", "measure"], ["sector_tzs", "nmb_tzs", "crdb_tzs"],
               {"nmb_share": "{nmb_tzs}/{sector_tzs}", "crdb_share": "{crdb_tzs}/{sector_tzs}",
                "cr2": "{nmb_share}+{crdb_share}"}, "ch01_shares (CR2 and bank shares)",
               "outputs/tables/ch01_shares.csv", checks, live, fmt={"sector_tzs": "tzs", "nmb_tzs": "tzs",
                                                                     "crdb_tzs": "tzs", "nmb_share": "pct",
                                                                     "crdb_share": "pct", "cr2": "pct"})
    live_table(c, ds, T["ch01_profit_vs_assets"], ["year"], ["sector_roa", "cr2_roa_pretax", "cr2_roa_posttax",
                                                             "asset_share"],
               {"profit_share_pretax": "{asset_share}*{cr2_roa_pretax}/{sector_roa}",
                "profit_minus_asset_share_pretax": "{profit_share_pretax}-{asset_share}",
                "profit_minus_asset_share_posttax": "{asset_share}*{cr2_roa_posttax}/{sector_roa}-{asset_share}"},
               "ch01_profit_vs_assets (profit share vs asset share)", "outputs/tables/ch01_profit_vs_assets.csv",
               checks, live, fmt={k: "pct" for k in ("sector_roa", "cr2_roa_pretax", "cr2_roa_posttax", "asset_share",
                                                     "profit_share_pretax", "profit_minus_asset_share_pretax",
                                                     "profit_minus_asset_share_posttax")})
    statics(c, T, ["ch01_hhi_bound", "ch01_dse_shares", "ch01_dse_report", "ch01_hypotheses"], CHAP["Ch01"], live)
    statics(sh["Ch02"], T, ["ch02_dissimilarity", "ch02_loan_mix", "ch02_credit_providers", "ch02_bot_coverage",
                            "ch02_hypotheses"], CHAP["Ch02"], live)
    c = sh["Ch03"]
    live_table(c, ds, T["ch03_deposit_cost"], ["fiscal_year"], ["systemic_mean", "smaller_median", "n_smaller"],
               {"advantage": "{smaller_median}-{systemic_mean}"}, "ch03_deposit_cost (deposit-cost advantage)",
               "outputs/tables/ch03_deposit_cost.csv", checks, live, fmt={"systemic_mean": "pct",
                                                                          "smaller_median": "pct", "advantage": "pct",
                                                                          "n_smaller": "int"})
    live_table(c, ds, T["ch03_capital_buffers"], ["bank", "fiscal_year"], ["tier1_ratio", "tier1_requirement",
                                                                           "total_capital_ratio", "total_requirement"],
               {"tier1_headroom": "{tier1_ratio}-{tier1_requirement}",
                "total_headroom": "{total_capital_ratio}-{total_requirement}"},
               "ch03_capital_buffers (headroom over the stated requirement)",
               "outputs/tables/ch03_capital_buffers.csv", checks, live, fmt={k: "pct" for k in (
                   "tier1_ratio", "tier1_requirement", "total_capital_ratio", "total_requirement", "tier1_headroom",
                   "total_headroom")})
    statics(c, T, ["ch03_dsib", "ch03_pb_roe", "ch03_hypotheses"], CHAP["Ch03"], live)
    c = sh["Ch04"]
    live_table(c, ds, T["ch04_4_0_components"], ["bank_id", "fiscal_year"],
               ["nii_ta", "nonii_ta", "opex_ta", "imp_ta", "other_ta", "tax_rate", "ta_e", "attr_share",
                "roe_reported"],
               {"roe_tree": "({nii_ta}+{nonii_ta}-{opex_ta}-{imp_ta}+{other_ta})*(1-{tax_rate})*{ta_e}*{attr_share}"},
               "ch04_4_0_components (extended DuPont tree: ROE rebuilt from its components)",
               "outputs/tables/ch04_4_0_components.csv", checks, live,
               fmt={k: "pct" for k in ("nii_ta", "nonii_ta", "opex_ta", "imp_ta", "other_ta", "tax_rate",
                                       "attr_share", "roe_reported", "roe_tree")} | {"ta_e": "num"})
    statics(c, T, ["ch04_dupont_panel", "ch04_4_0_gap_attribution", "ch04_4_0_change_attribution",
                   "ch04_4_0_subsection_ranking", "ch04_4_1_composition", "ch04_4_2_funding", "ch04_4_3_deployment",
                   "ch04_4_4_transactions", "ch04_4_5_operating_leverage", "ch04_4_5_scale_elasticity",
                   "ch04_4_6_risk", "ch04_business_model_map"], CHAP["Ch04"], live)
    c = sh["Ch05"]
    live_table(c, ds, T["ch05_coe_annual"], ["date_label", "year"],
               ["y_long", "default_spread", "beta_levered", "erp_mature", "crp", "illiquidity_premium"],
               {"coe": "({y_long}-{default_spread})+{beta_levered}*({erp_mature}+{crp})+{illiquidity_premium}",
                "coe_minus_200bp": "{coe}-0.02", "coe_plus_200bp": "{coe}+0.02"},
               "ch05_coe_annual (cost of equity, approach A, by year)", "outputs/tables/ch05_coe_annual.csv", checks,
               live, fmt={k: "pct" for k in ("y_long", "default_spread", "erp_mature", "crp", "illiquidity_premium",
                                             "coe", "coe_minus_200bp", "coe_plus_200bp")} | {"beta_levered": "num"})
    live_table(c, ds, T["ch05_excess_returns"], ["bank_id", "fiscal_year"], ["ni", "b_open", "avg_equity", "coe"],
               {"roe": "{ni}/{avg_equity}", "er": "{roe}-{coe}", "ri": "{ni}-{coe}*{b_open}"},
               "ch05_excess_returns (ROE, excess return and residual income)",
               "outputs/tables/ch05_excess_returns.csv", checks, live,
               fmt={"ni": "tzs", "b_open": "tzs", "avg_equity": "tzs", "coe": "pct", "roe": "pct", "er": "pct",
                    "ri": "tzs"})
    statics(c, T, ["ch05_persistence", "ch05_break_tests", "ch05_interval_coverage", "ch05_calibration_candidates",
                   "ch05_rank_persistence", "ch05_sign_persistence", "ch05_beta_sensitivity",
                   "ch05_ifrs9_sensitivity", "ch05_hypotheses"], CHAP["Ch05"], live)
    c = sh["Ch06"]
    live_table(c, ds, T["ch06_franchise"], ["bank_id", "fiscal_year"],
               ["avg_deposits", "cost_of_deposits", "tbill", "smr_avg", "deposit_insurance", "opex_ex_di"],
               {"gdfa": "({tbill}-{cost_of_deposits})*{avg_deposits}",
                "reserve_cost": "{smr_avg}*{avg_deposits}*{tbill}",
                "net_franchise_low": "{gdfa}-{reserve_cost}-1.5*{deposit_insurance}-0.75*{opex_ex_di}",
                "net_franchise_central": "{gdfa}-{reserve_cost}-{deposit_insurance}-0.5*{opex_ex_di}",
                "net_franchise_high": "{gdfa}-{reserve_cost}-0.5*{deposit_insurance}-0.25*{opex_ex_di}"},
               "ch06_franchise (gross deposit funding advantage, a proxy, and the net deposit franchise)",
               "outputs/tables/ch06_franchise.csv", checks, live,
               fmt={"avg_deposits": "tzs", "cost_of_deposits": "pct", "tbill": "pct", "smr_avg": "pct",
                    "deposit_insurance": "tzs", "opex_ex_di": "tzs", "gdfa": "tzs", "reserve_cost": "tzs",
                    "net_franchise_low": "tzs", "net_franchise_central": "tzs", "net_franchise_high": "tzs"})
    d = T["ch06_deposits"]
    live_table(c, ds, d[d["share_current"].notna()], ["bank_id", "fiscal_year"], ["share_current", "share_savings",
                                                                                   "share_time"],
               {"casa_share": "{share_current}+{share_savings}"}, "ch06_deposits (CASA share)",
               "outputs/tables/ch06_deposits.csv", checks, live, fmt={k: "pct" for k in (
                   "share_current", "share_savings", "share_time", "casa_share")})
    statics(c, T, ["ch06_deposit_beta", "ch06_rates_used", "ch06_smr_annual", "ch06_liquidity", "ch06_h61_parts",
                   "ch06_hypotheses"], CHAP["Ch06"], live)
    c = sh["Ch07"]
    k = T["ch07_capital"]
    live_table(c, ds, k[k["tier1_requirement"].notna()], ["bank_id", "fiscal_year"],
               ["tier1_ratio", "tier1_requirement", "total_capital_ratio", "total_requirement"],
               {"tier1_headroom": "{tier1_ratio}-{tier1_requirement}",
                "total_headroom": "{total_capital_ratio}-{total_requirement}"},
               "ch07_capital (headroom over the requirement, primary reading)", "outputs/tables/ch07_capital.csv",
               checks, live, fmt={x: "pct" for x in ("tier1_ratio", "tier1_requirement", "total_capital_ratio",
                                                     "total_requirement", "tier1_headroom", "total_headroom")})
    p = T["ch07_payout"].copy()
    p = p[p["dps_ordinary"].notna()].copy()
    p["dps_special_0"] = p["dps_special"].fillna(0.0)
    live_table(c, ds, p, ["bank_id", "fiscal_year"], ["ni_attr", "shares", "dps_ordinary", "dps_special_0"],
               {"div_ordinary": "{dps_ordinary}*{shares}", "div_special": "{dps_special_0}*{shares}",
                "div_total": "{div_ordinary}+{div_special}", "payout_total": "{div_total}/{ni_attr}"},
               "ch07_payout (dividends declared for the year and payout)", "outputs/tables/ch07_payout.csv", checks,
               live, fmt={"ni_attr": "tzs", "shares": "tzs", "dps_ordinary": "ps", "dps_special_0": "ps",
                          "div_ordinary": "tzs", "div_special": "tzs", "div_total": "tzs", "payout_total": "pct"},
               python_cols={"div_special": "_none_"})
    live_table(c, ds, T["ch07_distributable_capacity"], ["bank_id", "fiscal_year", "measure", "management_buffer"],
               ["requirement", "tier1_ratio", "rwa"],
               {"tier1_target": "{requirement}+{management_buffer}",
                "excess_tier1_tzs": "({tier1_ratio}-{tier1_target})*{rwa}"},
               "ch07_distributable_capacity (Tier 1 above the target)",
               "outputs/tables/ch07_distributable_capacity.csv", checks, live,
               fmt={"requirement": "pct", "tier1_ratio": "pct", "rwa": "tzs", "tier1_target": "pct",
                    "excess_tier1_tzs": "tzs"})
    statics(c, T, ["ch07_rore", "ch07_rore_windows", "ch07_rore_summary", "ch07_leverage",
                   "ch07_capital_requirements", "ch07_capital_raises", "ch07_distributable_capacity_total",
                   "ch07_hypotheses"], CHAP["Ch07"], live)
    c = sh["Ch08"]
    h = live_table(c, ds, T["ch08_credit_history"], ["bank_id", "fiscal_year"], ["impairment", "avg_gross_loans"],
                   {"cost_of_risk": "{impairment}/{avg_gross_loans}"}, "ch08_credit_history (cost of risk)",
                   "outputs/tables/ch08_credit_history.csv", checks, live,
                   fmt={"impairment": "tzs", "avg_gross_loans": "tzs", "cost_of_risk": "pct"})
    # through-the-cycle cost of risk: mean of the live annual ratios over each window
    tt = T["ch08_ttc_cost_of_risk"]
    tt = tt[tt["definition_type"].isin(["primary", "sub-period", "boundary grid"])].reset_index(drop=True)
    c.heading("Live: ch08_ttc_cost_of_risk (mean of the annual cost of risk over each window)", 8)
    c.note("AVERAGEIFS over the live cost-of-risk column above (robustness variants are static).")
    r0 = c.row
    for j, n in enumerate(("bank_id", "definition_type", "definition", "start", "end", "ttc_mean"), 1):
        c.put(r0, j, n, "label", bold=True)
    rr = h["rows"]
    pos = h["pos"]
    bank_rng = c.rng(rr[0], pos["bank_id"], rr[-1], pos["bank_id"], c)
    yr_rng = c.rng(rr[0], pos["fiscal_year"], rr[-1], pos["fiscal_year"], c)
    cor_rng = c.rng(rr[0], pos["cost_of_risk"], rr[-1], pos["cost_of_risk"], c)
    ttc_cell = {}
    for i, rec in tt.iterrows():
        r = r0 + 1 + i
        for j, n in enumerate(("bank_id", "definition_type", "definition", "start", "end"), 1):
            c.put(r, j, rec[n], "value")
        f = (f"=IFERROR(AVERAGEIFS({cor_rng},{bank_rng},{a1(r, 1)},{yr_rng},\">=\"&{a1(r, 4)},{yr_rng},\"<=\"&"
             f"{a1(r, 5)}),\"\")")
        c.put(r, 6, f, "formula", "pct")
        checks.add(f"Ch08 ttc_cost_of_risk {rec['bank_id']} {rec['definition'][:40]}", c, r, 6, rec["ttc_mean"])
        if rec["definition_type"] == "primary":
            ttc_cell[rec["bank_id"]] = a1(r, 6, True)
    live.add("Ch08", n_live=len(tt))
    c.row = r0 + len(tt) + 2
    ce = T["ch08_capital_efficiency"].copy()
    live_table(c, ds, ce, ["bank_id", "fiscal_year"], ["net_profit", "ppop", "total_assets", "rwa", "avg_rwa",
                                                       "avg_gross_loans", "ttc_cost_of_risk"],
               {"return_on_rwa": "{net_profit}/{avg_rwa}", "ppop_to_rwa": "{ppop}/{avg_rwa}",
                "risk_adjusted_rorwa": "({ppop}-{ttc_cost_of_risk}*{avg_gross_loans})/{avg_rwa}",
                "rwa_density": "{rwa}/{total_assets}"},
               "ch08_capital_efficiency (return on RWA, risk-adjusted)", "outputs/tables/ch08_capital_efficiency.csv",
               checks, live, fmt={x: "tzs" for x in ("net_profit", "ppop", "total_assets", "rwa", "avg_rwa",
                                                     "avg_gross_loans")} | {
                   x: "pct" for x in ("ttc_cost_of_risk", "return_on_rwa", "ppop_to_rwa", "risk_adjusted_rorwa",
                                      "rwa_density")})
    statics(c, T, ["ch08_ifrs9_transition", "ch08_ifrs9_transition_ratios", "ch08_stress_lens",
                   "ch08_h81_cost_of_risk_gap", "ch08_h82_rorwa_difference", "ch08_panel_summary", "ch08_hypotheses"],
            CHAP["Ch08"], live)


CHAP = {s: f"script {Path(sc).name}" for s, _, _, sc in CHAPTERS}


# --- Part III ------------------------------------------------------------------------------------------------------
def build_runs(wb, IN) -> tuple[dict, dict]:
    runs, bridge = {}, {}
    for val in DATES:
        for bank in BANKS:
            base_central = None
            for scen in SCEN:
                drivers, coe = EG.link_cells(IN, val, TICK[bank], scen)
                coe["r_base"] = base_central
                run = EG.build_run(wb, IN, val, bank, scen, f"Run {val} {TICK[bank]} {scen}", drivers=drivers, coe=coe,
                                   roe_link=None, icg=scen == "base", vd_val=val, opening_val=val,
                                   grid_g=GRID if scen == "base" else None)
                if scen == "base":
                    base_central = run.x("coe_central", None)
                runs[(val, bank, scen)] = run
    for bank in BANKS:  # bridge step 3: the historical drivers on the current opening, at the historical COE
        h = runs[("HIST", bank, "base")]
        drv, _ = EG.link_cells(IN, "CURR", TICK[bank], "base")
        hp = {s: IN.path("HIST", TICK[bank], s, "base", None) for s in EG.PATHS}
        for s in EG.PATHS:  # ext(xs) = (xs[1:] + [xs[-1]] * n)[:n]
            xs = hp[s]
            drv[s] = (xs[1:] + [xs[-1]] * EG.T)[:EG.T]
        link = {k: h.x(k, None) for k in ("r", "roe_star", "payout_star", "lam", "g")}
        bridge[bank] = EG.build_run(wb, IN, "CURR", bank, "base", f"Run bridge {TICK[bank]}", drivers=drv, coe=None,
                                    roe_link=link, icg=True, vd_val="CURR", opening_val="CURR",
                                    note="Bridge step 3 (VALUATION_SPEC §2): the historical register's drivers "
                                         "(shifted one year) on the current opening balances, valued at the "
                                         "historical COE.")
    return runs, bridge


def ch09(sh: Sheet, runs: dict, IN, T: dict, checks: Checks, live: Live) -> dict:
    v = T["ch09_valuations"]
    sh.heading("Live: ch09_valuations (blended RIM 60% / DDM 40%; D3)", 34)
    sh.note("Each row links its engine run sheet; bear and bull values come from their own runs. Rating rule D4: Buy "
            "when the expected return exceeds COE + 5 pp and the bear downside is no worse than -25%; Sell below "
            "COE - 5 pp; Hold otherwise.")
    cols = ["valuation", "date", "bank", "anchor_fy", "coe", "rim_equity_tzs", "ddm_equity_tzs", "blend_equity_tzs",
            "rim_ddm_gap", "shares", "value_per_share", "rim_per_share", "ddm_per_share", "price", "value_to_price",
            "target_price_12m", "expected_dps_12m", "expected_total_return", "identity_residual",
            "bear_value_per_share", "bull_value_per_share", "bear_downside", "rating", "market_pb", "market_pe",
            "model_pb", "justified_pb", "justified_pe", "value_per_share_post_split_basis", "price_post_split_basis"]
    r0 = sh.row
    for j, c in enumerate(cols, 1):
        sh.put(r0, j, c, "label", bold=True)
    pos = {c: j for j, c in enumerate(cols, 1)}
    cells = {}
    i = 0
    for val in DATES:
        for bank in BANKS:
            i += 1
            r = r0 + i
            run = runs[(val, bank, "base")]
            x = lambda k, run=run: run.x(k, sh)  # noqa: E731
            q = lambda k, r=r: a1(r, pos[k])  # noqa: E731
            bear, bull = runs[(val, bank, "bear")], runs[(val, bank, "bull")]
            ic = lambda k, val=val, bank=bank: IN.c(val, bank, k, sh)  # noqa: E731
            f = {"valuation": val, "date": f"={x('vd')}", "bank": TICK[bank], "anchor_fy": f"={ic('anchor_fy')}",
                 "coe": f"={x('r_val')}", "rim_equity_tzs": f"={x('v0_rim')}", "ddm_equity_tzs": f"={x('v0_ddm')}",
                 "blend_equity_tzs": f"={x('v0_blend')}", "rim_ddm_gap": f"={x('v0_gap')}",
                 "shares": f"={x('shares')}", "value_per_share": f"={x('v0_per_share')}",
                 "rim_per_share": f"={q('rim_equity_tzs')}/{q('shares')}",
                 "ddm_per_share": f"={q('ddm_equity_tzs')}/{q('shares')}", "price": f"={x('price')}",
                 "value_to_price": f"={q('value_per_share')}/{q('price')}",
                 "target_price_12m": f"={x('v1_per_share')}", "expected_dps_12m": f"={x('dps_window')}",
                 "expected_total_return": f"={x('exp_return')}", "identity_residual": f"={x('identity_res')}",
                 "bear_value_per_share": f"={bear.x('v0_per_share', sh)}",
                 "bull_value_per_share": f"={bull.x('v0_per_share', sh)}",
                 "bear_downside": f"={q('bear_value_per_share')}/{q('price')}-1",
                 "rating": f"=IF({q('expected_total_return')}>{q('coe')}+0.05,IF({q('bear_downside')}>=-0.25,\"Buy\","
                           f"\"Hold\"),IF({q('expected_total_return')}<{q('coe')}-0.05,\"Sell\",\"Hold\"))",
                 "market_pb": f"={q('price')}*{q('shares')}/{ic('equity_attr')}",
                 "market_pe": f"={q('price')}*{q('shares')}/{ic('profit_attr')}",
                 "model_pb": f"={q('blend_equity_tzs')}/{ic('equity_attr')}",
                 "justified_pb": f"=({x('roe_star')}-{x('g')})/({q('coe')}-{x('g')})",
                 "justified_pe": f"={x('payout_star')}*(1+{x('g')})/({q('coe')}-{x('g')})",
                 "value_per_share_post_split_basis": f"={q('value_per_share')}/{ic('split')}",
                 "price_post_split_basis": f"={q('price')}/{ic('split')}"}
            fm = {"coe": "pct4", "rim_ddm_gap": "num6", "identity_residual": "num6", "date": "date", "anchor_fy": "int",
                  "value_to_price": "num", "expected_total_return": "pct", "bear_downside": "pct", "market_pb": "num",
                  "market_pe": "num", "model_pb": "num", "justified_pb": "num", "justified_pe": "num",
                  "shares": "tzs"}
            py = v[(v["valuation"] == val) & (v["bank"] == TICK[bank])].iloc[0]
            for k, fx in f.items():
                kind = "value" if not str(fx).startswith("=") else "auto"
                fmt = fm.get(k, "tzs" if k.endswith("_tzs") else "ps")
                sh.put(r, pos[k], fx, kind, fmt)
                if k in ("valuation", "bank", "date"):
                    continue
                pv = py[k]
                kd = "text" if k == "rating" else ("ps" if k in ("value_per_share", "rim_per_share", "ddm_per_share",
                                                                 "target_price_12m", "expected_dps_12m",
                                                                 "bear_value_per_share", "bull_value_per_share",
                                                                 "value_per_share_post_split_basis") else
                                                   "abs" if k in ("identity_residual", "rim_ddm_gap") else "ratio")
                checks.add(f"Ch09 valuation {val} {TICK[bank]}: {k}", sh, r, pos[k], pv, kd)
            cells[(val, bank)] = {k: (r, pos[k]) for k in cols}
    live.add("Ch09", n_live=i * (len(cols) - 3))
    sh.row = r0 + i + 2
    # payout path and ICG cap
    pp = T["ch09_payout_path"]
    sh.heading("Live: ch09_payout_path (profit, dividends, capital in the explicit years, base case)", 12)
    hdr = ["valuation", "bank", "year", "np_attr_tzs", "dividend_tzs", "payout", "tier1_ratio", "k_target", "roe",
           "loan_growth", "icg_cap_applied"]
    r0 = sh.row
    for j, c in enumerate(hdr, 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    src = {"np_attr_tzs": "NP_attr", "dividend_tzs": "Div", "payout": "payout", "tier1_ratio": "T1_ratio",
           "k_target": "k_target", "roe": "ROE", "loan_growth": "loan_growth", "icg_cap_applied": "icg_flag"}
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            for t in range(1, EG.T + 1):
                r += 1
                sh.put(r, 1, val)
                sh.put(r, 2, TICK[bank])
                sh.put(r, 3, f"={run.at('year', t, sh)}", "link", "int")
                for j, k in enumerate(hdr[3:], 4):
                    sh.put(r, j, f"={run.at(src[k], t, sh)}", "link", "tzs" if k.endswith("tzs") else "pct")
                py = pp[(pp["valuation"] == val) & (pp["bank"] == TICK[bank])].sort_values("year")
                if len(py) >= t:
                    rec = py.iloc[t - 1]
                    for j, k in enumerate(hdr[3:-1], 4):
                        checks.add(f"Ch09 payout path {val} {TICK[bank]} {int(rec['year'])}: {k}", sh, r, j, rec[k])
    live.add("Ch09", n_live=(r - r0) * 8)
    sh.row = r + 2
    ic = T["ch09_icg_cap"]
    sh.heading("Live: ch09_icg_cap (base-case growth capped by internal capital generation, D56)", 8)
    r0 = sh.row
    for j, c in enumerate(["valuation", "bank", "year", "register_loan_growth", "capped_loan_growth",
                           "payout_after_cap"], 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for rec in ic.itertuples():
        bank = "B02" if rec.bank == "NMB" else "B01"
        run = runs[(rec.valuation, bank, "base")]
        anchor = int(T["ch09_payout_path"].query("valuation == @rec.valuation and bank == @rec.bank")["year"].min()) - 1
        t = int(rec.year) - anchor
        r += 1
        sh.put(r, 1, rec.valuation)
        sh.put(r, 2, rec.bank)
        sh.put(r, 3, int(rec.year), "value", "int")
        sh.put(r, 4, f"={run.sheet.ref(run.rows['icg_gL_reg'], EG.tc(t), sh)}", "link", "pct")
        sh.put(r, 5, f"={run.at('loan_growth', t, sh)}", "link", "pct")
        sh.put(r, 6, f"={run.at('payout', t, sh)}", "link", "pct")
        for j, k in ((4, "register_loan_growth"), (5, "capped_loan_growth"), (6, "payout_after_cap")):
            checks.add(f"Ch09 ICG cap {rec.valuation} {rec.bank} {rec.year}: {k}", sh, r, j, getattr(rec, k))
    live.add("Ch09", n_live=(r - r0) * 3)
    sh.row = r + 2
    # dividends in the 12-month window
    dv = T["ch09_dividends_12m"]
    sh.heading("Live: ch09_dividends_12m (dividends paid in the 12 months after the date; payment lag D62)", 8)
    r0 = sh.row
    for j, c in enumerate(["valuation", "bank", "fiscal_year", "pay_date", "years_from_valuation", "dps"], 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            r += 1
            pay = run.rowref("pay", 1, EG.N, sh)
            cond = f"({pay}+1>{run.x('v0_date', sh)})*({pay}+1<={run.x('v1_date', sh)})"
            sh.put(r, 1, val)
            sh.put(r, 2, TICK[bank])
            sh.put(r, 3, f"=SUMPRODUCT({cond}*{run.rowref('year', 1, EG.N, sh)})", "link", "int")
            sh.put(r, 4, f"=SUMPRODUCT({cond}*{pay})", "link", "date")
            sh.put(r, 5, f"=({a1(r, 4)}-{run.x('vd', sh)})/365.25", "formula", "num6")
            sh.put(r, 6, f"={run.x('dps_window', sh)}", "link", "ps")
            py = dv[(dv["valuation"] == val) & (dv["bank"] == TICK[bank])]
            if len(py):
                checks.add(f"Ch09 dividends {val} {TICK[bank]}: years_from_valuation", sh, r, 5,
                           py["years_from_valuation"].iloc[0])
                checks.add(f"Ch09 dividends {val} {TICK[bank]}: dps", sh, r, 6, py["dps"].iloc[0], "ps")
            cells[(val, bank)]["div_years"] = (r, 5)
            cells[(val, bank)]["div_dps"] = (r, 6)
    live.add("Ch09", n_live=(r - r0) * 4)
    sh.row = r + 2
    # COE A / B and the guard
    ca = T["ch09_coe_ab"]
    sh.heading("Live: ch09_coe_ab (COE approaches A and B; registered reconciliation rule, D93)", 10)
    r0 = sh.row
    for j, c in enumerate(["valuation", "bank", "coe_a", "coe_b", "a_minus_b", "pi_tz", "pi_us", "coe_used"], 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            r += 1
            sh.put(r, 1, val)
            sh.put(r, 2, TICK[bank])
            for j, k in ((3, "coe_a"), (4, "coe_b"), (6, "pi_tz"), (7, "pi_us"), (8, "r")):
                sh.put(r, j, f"={run.x(k, sh)}", "link", "pct4")
            sh.put(r, 5, f"={a1(r, 3)}-{a1(r, 4)}", "formula", "pct4")
            py = ca[(ca["valuation"] == val) & (ca["bank"] == TICK[bank])].iloc[0]
            for j, k in ((3, "coe_a"), (4, "coe_b"), (5, "a_minus_b"), (8, "coe_used")):
                checks.add(f"Ch09 COE {val} {TICK[bank]}: {k}", sh, r, j, py[k])
    live.add("Ch09", n_live=(r - r0) * 6)
    sh.row = r + 2
    gd = T["ch09_coe_g_guard"]
    sh.heading("Live: ch09_coe_g_guard (COE - g at least 4 pp in every scenario, D64 amended)", 10)
    r0 = sh.row
    for j, c in enumerate(["valuation", "bank", "scenario", "coe", "g", "coe_minus_g", "min_gap", "status"], 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            for scen in SCEN:
                run = runs[(val, bank, scen)]
                r += 1
                for j, x in ((1, val), (2, TICK[bank]), (3, scen)):
                    sh.put(r, j, x)
                sh.put(r, 4, f"={run.x('r', sh)}", "link", "pct4")
                sh.put(r, 5, f"={run.x('g', sh)}", "link", "pct4")
                sh.put(r, 6, f"={a1(r, 4)}-{a1(r, 5)}", "formula", "pct4")
                sh.put(r, 7, f"={IN.g('coe_g_min_gap', sh)}", "link", "pct")
                sh.put(r, 8, f"={run.x('guard', sh)}", "link")
                py = gd[(gd["valuation"] == val) & (gd["bank"] == TICK[bank]) & (gd["scenario"] == scen)]
                if len(py):
                    checks.add(f"Ch09 guard {val} {TICK[bank]} {scen}: coe_minus_g", sh, r, 6,
                               py["coe_minus_g"].iloc[0])
                    checks.add(f"Ch09 guard {val} {TICK[bank]} {scen}: status", sh, r, 8, py["status"].iloc[0],
                               "text")
    live.add("Ch09", n_live=(r - r0) * 5)
    sh.row = r + 2
    # sensitivity grids
    grids = T["ch09_sensitivity_grids"]
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            for gname in ("coe_x_g", "coe_x_roe_star"):
                py = grids[(grids["valuation"] == val) & (grids["bank"] == TICK[bank]) & (grids["grid"] == gname)]
                sh.heading(f"Live: sensitivity grid {gname}, {TICK[bank]} {DATE_LABEL[val]}: value per share (TZS)",
                           12)
                sh.note("Rows: COE; columns: terminal g (or ROE*). Each cell re-runs the fade and terminal value "
                        "with the engine's explicit years fixed (DDM form, equal to the blend by clean surplus).")
                r0 = sh.row
                sh.put(r0, 1, "COE \\ " + ("g" if gname == "coe_x_g" else "ROE*"), "label", bold=True)
                for j, d in enumerate(GRID):
                    col_ = 2 + j
                    base = run.x("g", sh) if gname == "coe_x_g" else run.x("roe_star", sh)
                    sh.put(r0, col_, f"={base}+{d}", "link", "pct")
                for i_, dr in enumerate(GRID):
                    r = r0 + 1 + i_
                    sh.put(r, 1, f"={run.x('r', sh)}+{dr}", "link", "pct")
                    for j, _d in enumerate(GRID):
                        col_ = 2 + j
                        rr, hh = f"$A{r}", f"{a1(r0, col_)}".replace(str(r0), f"${r0}")
                        if gname == "coe_x_g":
                            val_ = run.value_expr(sh, rr, run.x("roe_star", sh), run.x("lam", sh), gpath=j)
                            f = f"=IF({rr}>{hh},{val_}/{run.x('shares', sh)},\"\")"
                        else:
                            val_ = run.value_expr(sh, rr, hh, run.x("lam", sh))
                            f = (f"=IF(AND({hh}>{run.x('g', sh)},{rr}>{run.x('g', sh)}),{val_}/"
                                 f"{run.x('shares', sh)},\"\")")
                        sh.put(r, col_, f, "link", "ps")
                        pv = py.iloc[i_ * len(GRID) + j]["value_per_share"] if len(py) == 81 else None
                        checks.add(f"Ch09 grid {gname} {val} {TICK[bank]} [{i_},{j}]", sh, r, col_, _f(pv), "ps")
                live.add("Ch09", n_live=81)
                GRID_MAPS.append(_grid_row(heat(sh, r0 + 1, 2, r0 + len(GRID), 1 + len(GRID)),
                                           f"sensitivity grid {gname}, {TICK[bank]} {val}"))
                sh.row = r0 + len(GRID) + 2
    nim = grids[grids["grid"] == "nim_x_cor"]
    statics(sh, {"ch09_sensitivity_grids (NIM x cost of risk)": nim}, ["ch09_sensitivity_grids (NIM x cost of risk)"],
            CHAP["Ch09"] + "; each cell re-runs the full engine (81 runs per bank and date)", live)
    statics(sh, T, ["ch09_key_assumptions", "ch09_crdb_dividend_sensitivity", "ch09_pit_verification",
                    "ch09_override_log", "ch09_hypotheses"], CHAP["Ch09"], live)
    return cells


def _bisect(ws: Sheet, label: str, lo: str, hi: str, fx, checks: Checks, py, item: str) -> str:
    """Unrolled bisection (``BISECT`` steps) on the search sheet; returns the address of the root ('' if no sign
    change in [lo, hi], as Python's bracketed root returns NaN)."""
    r0 = ws.row
    ws.put(r0, 1, label, "label", bold=True)
    for j, h in enumerate(("lo", "hi", "f(lo)", "mid", "f(mid)", "f(hi)"), 2):
        ws.put(r0, j, h, "label", bold=True)
    r = r0 + 1
    ws.put(r, 2, f"={lo}", "link", "num6")
    ws.put(r, 3, f"={hi}", "link", "num6")
    ws.put(r, 4, f"={fx(a1(r, 2))}", "auto", "tzs")
    ws.put(r, 7, f"={fx(a1(r, 3))}", "auto", "tzs")
    for i in range(BISECT):
        rr = r + i
        if i:
            p = rr - 1
            ws.put(rr, 2, f"=IF({a1(p, 4)}*{a1(p, 6)}<=0,{a1(p, 2)},{a1(p, 5)})", "formula", "num6")
            ws.put(rr, 3, f"=IF({a1(p, 4)}*{a1(p, 6)}<=0,{a1(p, 5)},{a1(p, 3)})", "formula", "num6")
            ws.put(rr, 4, f"=IF({a1(p, 4)}*{a1(p, 6)}<=0,{a1(p, 4)},{a1(p, 6)})", "formula", "tzs")
        ws.put(rr, 5, f"=({a1(rr, 2)}+{a1(rr, 3)})/2", "formula", "num6")
        ws.put(rr, 6, f"={fx(a1(rr, 5))}", "auto", "tzs")
    last = r + BISECT - 1
    res = r0
    ws.put(res, 8, "root", "label", bold=True)
    ws.put(res, 9, f"=IF({a1(r, 4)}*{a1(r, 7)}<0,({a1(last, 2)}+{a1(last, 3)})/2,\"\")", "formula", "num6")
    ws.row = last + 2
    if py is not None:
        checks.add(item, ws, res, 9, py)
    return ws.ref(res, 9, None)


def ch10(sh: Sheet, srch: Sheet, runs: dict, bridge: dict, IN, T: dict, cells9: dict, checks: Checks,
         live: Live) -> dict:
    imp = T["ch10_implied"]
    sh.heading("Static – chapter 5 estimates used by the implied values (computed in Python, script "
               "07_persistence.py / 13_valuation_results.py)", 12)
    sh.note("Source: outputs/tables/ch10_implied.csv (31 Dec 2025: re-estimated on ER through FY2024; 31 Oct 2026: "
            "ch05_calibration_candidates). Statistical estimates enter as values; the searches below are live.")
    r0 = sh.row
    est_cols = ["lambda_ch5", "lambda_ch5_lo", "lambda_ch5_hi", "er_star_ch5", "er_star_ch5_lo", "er_star_ch5_hi",
                "er_star_ttc_ch5"]
    for j, c in enumerate(["valuation", "bank", *est_cols], 1):
        sh.put(r0, j, c, "label", bold=True)
    est = {}
    for i, rec in enumerate(imp.itertuples(), 1):
        r = r0 + i
        sh.put(r, 1, rec.valuation)
        sh.put(r, 2, rec.bank)
        for j, c in enumerate(est_cols, 3):
            sh.put(r, j, _f(getattr(rec, c)), "static", "num6")
        est[(rec.valuation, rec.bank)] = {c: sh.ref(r, j, None) for j, c in enumerate(est_cols, 3)}
    live.add("Ch10", n_static=len(imp) * len(est_cols))
    sh.row = r0 + len(imp) + 2
    # implied values (live searches on the search sheet)
    srch.title_block("Chapter 10 searches: implied ER*, lambda, COE and ROE* by unrolled bisection",
                     f"Each search halves the bracket {BISECT} times (Python: scipy brentq on the same bracket; an "
                     "empty root means no sign change, as Python's NaN).", 9)
    sh.heading("Live: ch10_implied (market-implied ER*, lambda, COE and ROE*; searches on 'Ch10 searches')", 12)
    r0 = sh.row
    hdr = ["valuation", "bank", "coe", "implied_er_star_at_ch5_lambda", "implied_lambda_at_ch5_er_star",
           "implied_coe", "implied_roe_star"]
    for j, c in enumerate(hdr, 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            e = est[(val, TICK[bank])]
            py = imp[(imp["valuation"] == val) & (imp["bank"] == TICK[bank])].iloc[0]
            pe = f"({run.x('price', srch)}*{run.x('shares', srch)})"
            rs, gs, lam, X = run.x("r", srch), run.x("g", srch), run.x("lam", srch), run.x("roe_star", srch)
            lab = f"{TICK[bank]} {val}"
            srch.heading(lab, 9)
            lam5, er5 = e["lambda_ch5"], e["er_star_ch5"]

            def f_er(x: str, run=run, rs=rs, lam5=lam5, pe=pe) -> str:
                return f"{run.value_expr(srch, rs, '(' + rs + '+' + x + ')', lam5, shape=LAM_SHAPE)}-{pe}"

            def f_lam(x: str, run=run, rs=rs, er5=er5, pe=pe) -> str:
                return f"{run.value_expr(srch, rs, '(' + rs + '+' + er5 + ')', x, shape=LAM_SHAPE)}-{pe}"

            def f_coe(x: str, run=run, X=X, lam=lam, pe=pe) -> str:
                return f"{run.value_expr(srch, x, X, lam)}-{pe}"

            def f_roe(x: str, run=run, rs=rs, lam=lam, pe=pe) -> str:
                return f"{run.value_expr(srch, rs, x, lam)}-{pe}"
            res = {}
            res["er"] = _bisect(srch, "implied ER* at the chapter 5 lambda", f"{gs}-{rs}+0.001", "0.4", f_er, checks,
                                _f(py["implied_er_star_at_ch5_lambda"]),
                                f"Ch10 implied {val} {TICK[bank]}: ER* at the chapter 5 lambda")
            res["lam"] = _bisect(srch, "implied lambda at the chapter 5 ER*", "0", "0.999", f_lam, checks,
                                 _f(py["implied_lambda_at_ch5_er_star"]),
                                 f"Ch10 implied {val} {TICK[bank]}: lambda at the chapter 5 ER*")
            res["coe"] = _bisect(srch, "implied COE (forecast fixed)", f"{gs}+0.0001", "0.6", f_coe, checks,
                                 _f(py["implied_coe"]), f"Ch10 implied {val} {TICK[bank]}: COE")
            res["roe"] = _bisect(srch, "implied ROE* (payout* = 1 - g / ROE*)", f"MAX(-0.1,{gs}+0.0001)", "0.8",
                                 f_roe, checks, _f(py["implied_roe_star"]), f"Ch10 implied {val} {TICK[bank]}: ROE*")
            r += 1
            sh.put(r, 1, val)
            sh.put(r, 2, TICK[bank])
            sh.put(r, 3, f"={run.x('r', sh)}", "link", "pct4")
            for j, k in ((4, "er"), (5, "lam"), (6, "coe"), (7, "roe")):
                sh.put(r, j, f"={res[k]}", "link", "pct4" if k != "lam" else "num")
    live.add("Ch10", n_live=(r - r0) * 4 + 16 * BISECT * 6)
    sh.row = r + 2
    # iso-value surfaces
    iso = T["ch10_iso_surface"]
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            py = iso[(iso["valuation"] == val) & (iso["bank"] == TICK[bank])]
            sh.heading(f"Live: iso-value surface {TICK[bank]} {DATE_LABEL[val]}: value / price over (lambda, ER*)",
                       min(len(ERS) + 3, 45))
            sh.note("Rows: lambda; columns: ER*; forecast and COE fixed (blueprint D12). Last column: the ER* at which "
                    "value = price (first crossing, linear interpolation, as Python).")
            r0 = sh.row
            sh.put(r0, 1, "lambda \\ ER*", "label", bold=True)
            for j, er in enumerate(ERS):
                sh.put(r0, 2 + j, er, "value", "pct")
            cx = 2 + len(ERS)
            sh.put(r0, cx, "er_star_at_price", "label", bold=True)
            pe = f"({run.x('price', sh)}*{run.x('shares', sh)})"
            rs = run.x("r", sh)
            # crossing helper block to the right
            hx = cx + 2
            sh.put(r0, hx, "crossing candidates (helper)", "note")
            for i, lam in enumerate(LAMS):
                r = r0 + 1 + i
                sh.put(r, 1, lam, "value", "num")
                for j, er in enumerate(ERS):
                    c = 2 + j
                    X = f"({rs}+{a1(r0, c).replace(str(r0), '$' + str(r0))})"
                    f = f"={run.value_expr(sh, rs, X, f'$A{r}', shape=LAM_SHAPE)}/{pe}"
                    sh.put(r, c, f, "link", "num")
                    m = py[(np.isclose(py["lambda"], lam)) & (np.isclose(py["er_star"], er))]
                    if len(m):
                        checks.add(f"Ch10 iso {val} {TICK[bank]} lambda {lam} ER* {er}", sh, r, c,
                                   m["value_to_price"].iloc[0], "ratio")
                for j in range(len(ERS) - 1):
                    v0, v1 = a1(r, 2 + j), a1(r, 3 + j)
                    e0, e1 = a1(r0, 2 + j), a1(r0, 3 + j)
                    sh.put(r, hx + j, f"=IF(AND(({v0}-1)*({v1}-1)<=0,{v1}<>{v0}),{e0}+(1-{v0})*({e1}-{e0})/({v1}-{v0}),"
                                      f"\"\")", "formula", "num6")
                cand = sh.rng(r, hx, r, hx + len(ERS) - 2, sh)
                sh.put(r, cx, f"=IFERROR(INDEX({cand},MATCH(TRUE,INDEX(ISNUMBER({cand}),0),0)),\"\")", "formula",
                       "pct4")
                m = py[np.isclose(py["lambda"], lam)]
                if len(m):
                    checks.add(f"Ch10 iso {val} {TICK[bank]} lambda {lam}: ER* at price", sh, r, cx,
                               _f(m["er_star_at_price"].iloc[0]))
            live.add("Ch10", n_live=len(LAMS) * (2 * len(ERS)))
            GRID_MAPS.append(_grid_row(heat(sh, r0 + 1, 2, r0 + len(LAMS), 1 + len(ERS)),
                                       f"iso-value surface, {TICK[bank]} {val}"))
            sh.row = r0 + len(LAMS) + 2
    # bridge and rerating
    br = T["ch10_bridge"]
    rr_ = T["ch10_rerating"]
    out = {}
    sh.heading("Live: ch10_bridge (five-step valuation bridge 31 Dec 2025 -> 31 Oct 2026, fixed order; VALUATION_SPEC "
               "§2) and ch10_rerating", 14)
    for bank in BANKS:
        h, cu, b3 = runs[("HIST", bank, "base")], runs[("CURR", bank, "base")], bridge[bank]
        rh = h.x("r", sh)
        r0 = sh.row
        sh.put(r0, 1, f"{TICK[bank]}", "label", bold=True)
        lines = [
            ("v0 historical value", f"={h.x('v0_blend', sh)}"),
            ("v1 after the passage of time", f"=B{r0 + 1}*(1+{rh})^((YEAR({cu.x('vd', sh)})-YEAR({h.x('vd', sh)}))+("
                                             f"{cu.x('vd', sh)}-DATE(YEAR({cu.x('vd', sh)}),MONTH({h.x('vd', sh)}),"
                                             f"DAY({h.x('vd', sh)})))/365.25)"),
            ("dividend paid between the dates (final DPS x shares)",
             f"={IN.c('CURR', bank, 'final_dps', sh)}*{IN.c('CURR', bank, 'shares_out', sh)}"),
            ("its payment date", f"=EDATE(DATE({IN.c('CURR', bank, 'anchor_fy', sh)},12,31),{IN.g('lag_months', sh)})"),
            ("carry factor to the current date at the historical COE",
             f"=IF({IN.g('lag_months', sh)}>0,(1+{rh})^((YEAR({cu.x('vd', sh)})-YEAR(B{r0 + 4}))+({cu.x('vd', sh)}-"
             f"DATE(YEAR({cu.x('vd', sh)}),MONTH(B{r0 + 4}),DAY(B{r0 + 4})))/365.25),1)"),
            ("v2 after the dividend", f"=B{r0 + 2}-B{r0 + 3}*B{r0 + 5}"),
            ("v3 new information (historical drivers on the current opening, historical COE)",
             f"={b3.x('v0_blend', sh)}"),
            ("v4 forecast revisions (current forecast at the historical COE)",
             f"={cu.value_expr(sh, rh, cu.x('roe_star', sh), cu.x('lam', sh))}"),
            ("v5 current value", f"={cu.x('v0_blend', sh)}"),
        ]
        for i, (lab, f) in enumerate(lines, 1):
            sh.put(r0 + i, 1, lab)
            sh.put(r0 + i, 2, f, "auto", "date" if "payment date" in lab else "num6" if "carry" in lab else "tzs")
        base = r0 + len(lines) + 1
        steps = [("historical value", f"=B{r0 + 1}"), ("1 passage of time", f"=B{r0 + 2}-B{r0 + 1}"),
                 ("2 dividends paid", f"=B{r0 + 6}-B{r0 + 2}"), ("3 new information", f"=B{r0 + 7}-B{r0 + 6}"),
                 ("4 forecast revisions", f"=B{r0 + 8}-B{r0 + 7}"), ("5 change in COE", f"=B{r0 + 9}-B{r0 + 8}"),
                 ("current value", f"=B{r0 + 9}"), ("residual (reported, not absorbed)",
                                                    f"=B{base + 6}-SUM(B{base}:B{base + 5})")]
        py = br[br["bank"] == TICK[bank]].set_index("step")["tzs"]
        for i, (lab, f) in enumerate(steps):
            sh.put(base + i, 1, f"step: {lab}")
            sh.put(base + i, 2, f, "formula", "tzs")
            if lab in py.index and not lab.startswith("residual"):
                checks.add(f"Ch10 bridge {TICK[bank]}: {lab}", sh, base + i, 2, py[lab], "ratio")
        rb = base + len(steps) + 1
        pyr = rr_[rr_["bank"] == TICK[bank]].iloc[0]
        mv0 = f"{h.x('price', sh)}*{h.x('shares', sh)}"
        mv1 = f"{cu.x('price', sh)}*{cu.x('shares', sh)}"
        rer = [("market_cap_hist", f"={mv0}"), ("market_cap_curr", f"={mv1}"), ("dividends_paid", f"=B{r0 + 3}"),
               ("market_total_return_log", f"=LN((B{rb + 1}+B{rb + 2})/B{rb})"),
               ("earnings_and_time_log", f"=LN((B{r0 + 8}+B{rb + 2})/B{r0 + 1})"),
               ("discount_rate_log", f"=IF(AND(B{r0 + 8}>0,B{r0 + 9}>0),LN(B{r0 + 9}/B{r0 + 8}),\"\")"),
               ("residual_log", f"=B{rb + 3}-B{rb + 4}-B{rb + 5}"), ("coe_hist", f"={rh}"),
               ("coe_curr", f"={cu.x('r', sh)}"), ("value_to_price_hist", f"={h.x('v0_per_share', sh)}/{h.x('price', sh)}"),
               ("value_to_price_curr", f"={cu.x('v0_per_share', sh)}/{cu.x('price', sh)}")]
        for i, (lab, f) in enumerate(rer):
            sh.put(rb + i, 1, f"rerating: {lab}")
            sh.put(rb + i, 2, f, "auto", "tzs" if lab.startswith(("market_cap", "dividends")) else "num6")
            checks.add(f"Ch10 rerating {TICK[bank]}: {lab}", sh, rb + i, 2, _f(pyr[lab]))
        out[bank] = {"steps": (base, 2), "rerating": (rb, 2)}
        live.add("Ch10", n_live=len(lines) + len(steps) + len(rer))
        sh.row = rb + len(rer) + 2
    statics(sh, T, ["ch10_hypotheses"], CHAP["Ch10"], live)
    return out


def ch11(sh: Sheet, runs: dict, T: dict, checks: Checks, live: Live) -> None:
    sc = T["ch11_scenarios"]
    sh.heading("Live: ch11_scenarios (bear / base / bull: value per share, first three forecast years' EPS and ROE)",
               12)
    r0 = sh.row
    hdr = ["valuation", "bank", "scenario", "value_per_share", "eps_y1", "eps_y2", "eps_y3", "roe_y1", "roe_y2",
           "roe_y3", "price"]
    for j, c in enumerate(hdr, 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            for scen in SCEN:
                run = runs[(val, bank, scen)]
                r += 1
                for j, x in ((1, val), (2, TICK[bank]), (3, scen)):
                    sh.put(r, j, x)
                sh.put(r, 4, f"={run.x('v0_per_share', sh)}", "link", "ps")
                for t in (1, 2, 3):
                    sh.put(r, 4 + t, f"={run.at('NP_attr', t, sh)}/{run.x('shares', sh)}", "link", "ps")
                    sh.put(r, 7 + t, f"={run.at('ROE', t, sh)}", "link", "pct")
                sh.put(r, 11, f"={run.x('price', sh)}", "link", "ps")
                py = sc[(sc["valuation"] == val) & (sc["bank"] == TICK[bank]) & (sc["scenario"] == scen)]
                if len(py):
                    rec = py.iloc[0]
                    checks.add(f"Ch11 scenario {val} {TICK[bank]} {scen}: value_per_share", sh, r, 4,
                               rec["value_per_share"], "ps")
                    yrs = sorted(int(c.split("_")[1]) for c in sc.columns if c.startswith("eps_") and pd.notna(rec[c]))
                    for t, y in enumerate(yrs[:3], 1):
                        checks.add(f"Ch11 scenario {val} {TICK[bank]} {scen}: eps_{y}", sh, r, 4 + t,
                                   rec[f"eps_{y}"], "ps")
                        checks.add(f"Ch11 scenario {val} {TICK[bank]} {scen}: roe_{y}", sh, r, 7 + t,
                                   rec[f"roe_{y}"])
    live.add("Ch11", n_live=(r - r0) * 8)
    sh.row = r + 2
    statics(sh, T, ["ch11_monte_carlo", "ch11_tested_call", "ch11_scenario_evidence", "ch11_hypotheses"],
            CHAP["Ch11"] + " (seeded Monte Carlo, 10,000 draws)", live)


def ch12(sh: Sheet, srch: Sheet, ch9: Sheet, runs: dict, IN, T: dict, cells9: dict, cfg: dict, checks: Checks,
         live: Live) -> None:
    exp = T["ch12_expected_usd"]
    dep = T["ch12_depreciation"]
    capc = T["ch12_capacity"]
    sh.heading("Inputs for chapter 12 (blue): withholding tax and depreciation scenarios", 10)
    r = sh.row
    sh.put(r, 1, "Dividend withholding tax, foreign holder (DSE-listed)")
    sh.put(r, 2, float(cfg["div_wht_foreign"]), "input", "pct")
    wht_pos = (r, 2)
    sh.row += 1
    depcell = {}
    for val in DATES:
        for rec in dep[dep["valuation"] == val].itertuples():
            r = sh.row
            sh.put(r, 1, f"Depreciation a year, {DATE_LABEL[val]}: {rec.scenario}")
            if rec.scenario == "ppp":
                f = f"=(1+{IN.c(val, 'B02', 'infl_tz', sh)})/(1+{IN.c(val, 'B02', 'infl_us', sh)})-1"
                sh.put(r, 2, f, "link", "pct4")
                checks.add(f"Ch12 depreciation {val}: ppp", sh, r, 2, rec.depreciation)
            else:
                sh.put(r, 2, float(rec.depreciation), "input", "pct4")
            sh.put(r, 3, str(rec.basis)[:200], "note")
            depcell[(val, rec.scenario)] = (r, 2)
            sh.row += 1
    live.add("Ch12", n_live=2)
    sh.row += 1
    sh.heading("Live: USD hurdle = US 10-year + beta (ERP + CRP) + LP (VALUATION_SPEC §5, USD leg; D60 amended)", 10)
    hurdle = {}
    r0 = sh.row
    for j, c in enumerate(("valuation", "bank", "rf_us", "beta", "erp", "crp", "LP", "usd_hurdle"), 1):
        sh.put(r0, j, c, "label", bold=True)
    r = r0
    for val in DATES:
        for bank in BANKS:
            r += 1
            sh.put(r, 1, val)
            sh.put(r, 2, TICK[bank])
            sh.put(r, 3, f"={IN.c(val, bank, 'rf_us', sh)}", "link", "pct")
            sh.put(r, 4, f"={IN.one(val, TICK[bank], 'beta', 'base', sh)}", "link", "num")
            sh.put(r, 5, f"={IN.c(val, bank, 'erp', sh)}", "link", "pct")
            sh.put(r, 6, f"={IN.c(val, bank, 'crp', sh)}", "link", "pct")
            sh.put(r, 7, f"={IN.one(val, TICK[bank], 'LP', 'base', sh)}", "link", "pct")
            sh.put(r, 8, f"={a1(r, 3)}+{a1(r, 4)}*({a1(r, 5)}+{a1(r, 6)})+{a1(r, 7)}", "formula", "pct4")
            py = exp[(exp["valuation"] == val) & (exp["bank"] == TICK[bank])]
            if len(py):
                checks.add(f"Ch12 hurdle {val} {TICK[bank]}", sh, r, 8, py["usd_hurdle"].iloc[0])
            hurdle[(val, bank)] = (r, 8)
    live.add("Ch12", n_live=(r - r0))
    sh.row = r + 2
    # breakeven depreciation by bisection (search sheet)
    srch.heading("Chapter 12: breakeven depreciation (USD return = hurdle)", 9)
    be = {}
    wht_s = sh.ref(*wht_pos, srch)
    for val in DATES:
        for bank in BANKS:
            run = runs[(val, bank, "base")]
            pr = f"({run.x('v1_per_share', srch)}/{run.x('price', srch)}-1)"
            dy = f"({run.x('dps_window', srch)}/{run.x('price', srch)})"
            tt = ch9.ref(*cells9[(val, bank)]["div_years"], srch)
            hh = sh.ref(*hurdle[(val, bank)], srch)

            def fx(d: str, pr=pr, dy=dy, tt=tt, hh=hh) -> str:
                return f"((1+{pr})/(1+{d})+{dy}*(1-{wht_s})/(1+{d})^{tt}-1)-{hh}"
            pyb = exp[(exp["valuation"] == val) & (exp["bank"] == TICK[bank])]
            be[(val, bank)] = _bisect(srch, f"breakeven depreciation {TICK[bank]} {val}", "-0.95", "3", fx, checks,
                                      _f(pyb["breakeven_depreciation"].iloc[0]) if len(pyb) else None,
                                      f"Ch12 breakeven depreciation {val} {TICK[bank]}")
    sh.heading("Live: ch12_expected_usd (12-month USD total return per date, bank, position and depreciation scenario)",
               20)
    sh.note("USD return = (1 + TZS price return) / (1 + d) + dividend yield x (1 - WHT) / (1 + d)^t - 1, t = years to "
            "the payment date (prompt 24); round-trip cost from ch12_capacity (static, daily trading data).")
    hdr = ["valuation", "bank", "position_usd", "scenario", "price_tzs", "target_price_12m", "tzs_price_return",
           "expected_dividend_yield", "dividend_wht", "expected_tzs_return_12m", "depreciation", "translation_effect",
           "expected_usd_return", "usd_hurdle", "excess_over_hurdle", "breakeven_depreciation", "round_trip_cost",
           "expected_usd_return_after_cost"]
    r0 = sh.row
    for j, c in enumerate(hdr, 1):
        sh.put(r0, j, c, "label", bold=True)
    pos = {c: j for j, c in enumerate(hdr, 1)}
    wht = sh.ref(*wht_pos, sh)
    r = r0
    for rec in exp.itertuples():
        bank = "B02" if rec.bank == "NMB" else "B01"
        run = runs[(rec.valuation, bank, "base")]
        r += 1
        q = lambda k, r=r: a1(r, pos[k])  # noqa: E731
        cap = capc[(capc["valuation"] == rec.valuation) & (capc["bank"] == rec.bank) &
                   (capc["position_usd"] == rec.position_usd) & np.isclose(capc["participation"],
                                                                         float(cfg["base_participation"]))]
        tt = ch9.ref(*cells9[(rec.valuation, bank)]["div_years"], sh)
        f = {"valuation": rec.valuation, "bank": rec.bank, "position_usd": int(rec.position_usd),
             "scenario": rec.scenario, "price_tzs": f"={run.x('price', sh)}",
             "target_price_12m": f"={run.x('v1_per_share', sh)}",
             "tzs_price_return": f"={q('target_price_12m')}/{q('price_tzs')}-1",
             "expected_dividend_yield": f"={run.x('dps_window', sh)}/{q('price_tzs')}",
             "dividend_wht": f"={wht}*{q('expected_dividend_yield')}",
             "expected_tzs_return_12m": f"={q('tzs_price_return')}+{q('expected_dividend_yield')}*(1-{wht})",
             "depreciation": f"={sh.ref(*depcell[(rec.valuation, rec.scenario)], sh)}",
             "expected_usd_return": f"=(1+{q('tzs_price_return')})/(1+{q('depreciation')})+"
                                    f"{q('expected_dividend_yield')}*(1-{wht})/(1+{q('depreciation')})^{tt}-1",
             "translation_effect": f"={q('expected_usd_return')}-{q('expected_tzs_return_12m')}",
             "usd_hurdle": f"={sh.ref(*hurdle[(rec.valuation, bank)], sh)}",
             "excess_over_hurdle": f"={q('expected_usd_return')}-{q('usd_hurdle')}",
             "breakeven_depreciation": f"={be[(rec.valuation, bank)]}",
             "round_trip_cost": float(cap["round_trip_cost"].iloc[0]) if len(cap) else None,
             "expected_usd_return_after_cost": f"={q('expected_usd_return')}-{q('round_trip_cost')}/(1+"
                                               f"{q('depreciation')})"}
        for k, fx in f.items():
            kind = "static" if k == "round_trip_cost" else ("value" if not str(fx).startswith("=") else "auto")
            fmt = "ps" if k in ("price_tzs", "target_price_12m") else "tzs" if k == "position_usd" else "pct4"
            sh.put(r, pos[k], fx, kind, fmt if k not in ("valuation", "bank", "scenario") else None)
            if k in ("tzs_price_return", "expected_dividend_yield", "expected_tzs_return_12m", "depreciation",
                     "translation_effect", "expected_usd_return", "excess_over_hurdle", "breakeven_depreciation",
                     "expected_usd_return_after_cost"):
                checks.add(f"Ch12 USD {rec.valuation} {rec.bank} {rec.position_usd} {rec.scenario}: {k}", sh, r,
                           pos[k], getattr(rec, k), "abs" if k == "translation_effect" else "ratio")
    live.add("Ch12", n_live=(r - r0) * 13 + 4 * BISECT * 5, n_static=(r - r0))
    sh.row = r + 2
    statics(sh, T, ["ch12_capacity", "ch12_liquidity", "ch12_fx", "ch12_depreciation", "ch12_returns_annual",
                    "ch12_returns_spans", "ch12_cross_rates", "ch12_usd_steps", "ch12_market_access",
                    "ch12_hypotheses"], CHAP["Ch12"], live)


def _diff(live: str, pub: str) -> str:
    """Live - published, shown as 0 within 1e-9 relative (floating-point noise; the check's tolerance is 1e-4)."""
    return f"=IF(ABS({live}-{pub})<=MAX(1E-9*ABS({pub}),1E-12),0,{live}-{pub})"


def summary(sh: Sheet, runs: dict, bridge: dict, cells9: dict, br: dict, T: dict, ch9: Sheet, ch10: Sheet) -> None:
    """Live values beside the published pipeline values, with a difference cell (red when non-zero), and the
    convergence flags of every engine run (owner review of step 20)."""
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import PatternFill

    red = CellIsRule(operator="notEqual", formula=["0"], fill=PatternFill("solid", fgColor="F8CBAD"))
    sh.title_block("Summary: value per share, target and rating by bank and date; the valuation bridge",
                   "Live columns link the chapter sheets (green) and move with the inputs; Published columns hold the "
                   "pipeline's figures (grey); Difference = live - published, red when not zero.", 10)
    sh.width(1, 46)
    for c in range(2, 9):
        sh.width(c, 16)
    items = [("value_per_share", "Value per share (TZS)", "ps"), ("rim_per_share", "RIM per share", "ps"),
             ("ddm_per_share", "DDM per share", "ps"), ("price", "Market price (TZS)", "ps"),
             ("value_to_price", "Value / price", "num"), ("target_price_12m", "12-month target price", "ps"),
             ("expected_dps_12m", "Expected DPS in 12 months", "ps"),
             ("expected_total_return", "Expected total return", "pct"), ("coe", "Cost of equity (r)", "pct"),
             ("bear_value_per_share", "Bear value per share", "ps"), ("bull_value_per_share", "Bull value per share",
                                                                      "ps"),
             ("bear_downside", "Bear downside vs price", "pct"), ("rating", "Rating (D4 rule)", None),
             ("value_per_share_post_split_basis", "Value per share, post-split basis (NMB 1:10)", "ps")]
    v = T["ch09_valuations"]

    def header(r0: int) -> None:
        for j, bank in enumerate(BANKS):
            for k, h in enumerate(("live", "published", "difference")):
                sh.put(r0, 2 + 3 * j + k, f"{TICK[bank]} {h}", "label", bold=True)

    for val in DATES:
        sh.heading(f"{DATE_LABEL[val]}" + (" – point in time: information public by 31 Dec 2025" if val == "HIST"
                                           else ""), 8)
        r0 = sh.row
        header(r0)
        for i, (k, lab, fm) in enumerate(items, 1):
            r = r0 + i
            sh.put(r, 1, lab)
            for j, bank in enumerate(BANKS):
                c = 2 + 3 * j
                rc = cells9[(val, bank)][k]
                sh.put(r, c, f"={ch9.ref(*rc, sh)}", "link", fm)
                define(sh.wb, f"S_{val}_{TICK[bank]}_{k}", sh, r, c)
                pv = v[(v["valuation"] == val) & (v["bank"] == TICK[bank])][k].iloc[0]
                pv = pv.item() if hasattr(pv, "item") else pv
                sh.put(r, c + 1, pv, "static", fm)
                if k == "rating":
                    f = f"=IF({a1(r, c)}={a1(r, c + 1)},0,1)"
                else:
                    f = _diff(a1(r, c), a1(r, c + 1))
                sh.put(r, c + 2, f, "formula", "ps" if fm == "ps" else "num6")
                sh.ws.conditional_formatting.add(a1(r, c + 2), red)
        sh.row = r0 + len(items) + 2
    sh.heading("Valuation bridge 31 Dec 2025 -> 31 Oct 2026 (TZS; VALUATION_SPEC §2)", 8)
    r0 = sh.row
    header(r0)
    labels = ["historical value", "1 passage of time", "2 dividends paid", "3 new information",
              "4 forecast revisions", "5 change in COE", "current value", "residual (reported, not absorbed)"]
    bt = T["ch10_bridge"]
    for i, lab in enumerate(labels, 1):
        r = r0 + i
        sh.put(r, 1, lab)
        for j, bank in enumerate(BANKS):
            c = 2 + 3 * j
            rr, cc = br[bank]["steps"]
            sh.put(r, c, f"={ch10.ref(rr + i - 1, cc, sh)}", "link", "tzs")
            pv = bt[(bt["bank"] == TICK[bank]) & (bt["step"] == lab)]["tzs"]
            sh.put(r, c + 1, float(pv.iloc[0]) if len(pv) else None, "static", "tzs")
            if lab.startswith("residual"):
                sh.put(r, c + 2, None)
                continue
            sh.put(r, c + 2, _diff(a1(r, c), a1(r, c + 1)), "formula", "tzs")
            sh.ws.conditional_formatting.add(a1(r, c + 2), red)
    sh.row = r0 + len(labels) + 2
    sh.heading("Convergence of the engine runs (fixed-point solve and growth-cap search; see each run sheet)", 8)
    r0 = sh.row
    for j, s in enumerate(("base", "bear", "bull"), 2):
        sh.put(r0, j, s, "label", bold=True)
    first = r0 + 1
    r = r0
    for val in DATES:
        for bank in BANKS:
            r += 1
            sh.put(r, 1, f"{TICK[bank]}, {DATE_LABEL[val]}")
            for j, s in enumerate(("base", "bear", "bull"), 2):
                sh.put(r, j, f"={runs[(val, bank, s)].x('converged', sh)}", "link")
    for bank in BANKS:
        r += 1
        sh.put(r, 1, f"{TICK[bank]}, bridge run (step 3)")
        sh.put(r, 2, f"={bridge[bank].x('converged', sh)}", "link")
    r += 1
    sh.put(r, 1, "All engine runs converged", "label", bold=True)
    sh.put(r, 2, f"=IF(COUNTIF({a1(first, 2)}:{a1(r - 1, 4)},\"FAIL\")=0,\"PASS\",\"FAIL\")", "formula", bold=True)
    define(sh.wb, "S_all_runs_converged", sh, r, 2)
    sh.ws.conditional_formatting.add(f"{a1(first, 2)}:{a1(r, 4)}",
                                     CellIsRule(operator="equal", formula=['"FAIL"'],
                                                fill=PatternFill("solid", fgColor="F8CBAD")))
    sh.row = r + 2
    sh.ws.freeze_panes = "B4"


def readme(sh: Sheet, root: Path, live: Live, n_checks: int, sheets: list[tuple[str, str]]) -> None:
    rep = {}
    rp = root / "config" / "reports.yaml"
    if rp.exists():
        rep = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
    rl = {}
    lp = path("outputs", root) / "run_logs" / "latest" / "valuation_results.json"
    if lp.exists():
        rl = json.loads(lp.read_text(encoding="utf-8"))
    packs = sorted(path("outputs", root).glob("findings/findings_pack_v*.csv"),
                   key=lambda p: [int(x) for x in p.stem.split("_v")[1].split(".")])
    reg = pd.read_csv(root / "config" / "assumption_register.csv", usecols=["version"])["version"].iloc[0]
    sh.title_block(TITLE, "Dominant Banks, Conditional Rents: NMB, CRDB and the price of dominance in Tanzania", 8)
    sh.width(1, 34)
    sh.width(2, 110)
    rows = [
        ("Purpose", "A chapter-organised Excel companion to the Python pipeline, so a reader without Python can open "
                    "the analysis, change inputs and see the results recalculate. It reproduces the pipeline; it does "
                    "not reinterpret it."),
        ("Source of truth", "The Python pipeline (`python -m tzbank run all`) is the source of truth. This workbook is "
                            "rebuilt by `python -m tzbank run excel_model` (scripts/20_excel_model.py), recalculated "
                            "and checked cell by cell against the Python tables (outputs/valuation/"
                            "excel_model_check.csv; tolerance 0.01% or TZS 0.01 per share)."),
        ("How to use", "Change a blue cell on the Inputs sheet (assumptions, point-in-time inputs, register drivers "
                       "for base / bear / bull). The engine run sheets, the chapter sheets and the Summary "
                       "recalculate. Keep calculation on Automatic. Named ranges: g_<setting>, <date>_<bank>_<input>, "
                       "R_<date>_<bank>_<symbol>_<period>_<scenario>, S_<date>_<bank>_<output>."),
        ("Colour legend", "Blue font on cream = input. Black = formula on the same sheet. Green = formula or link that "
                          "reads another sheet. Grey fill = static value computed in Python (statistics, simulations, "
                          "data-heavy steps), with the script and the source table named above each block."),
        ("Live vs static", "Live: ratios, DuPont tree, cost of equity, excess returns, deposit franchise, capital "
                           "headroom and payout, cost of risk and return on RWA, the forecast engine, RIM / DDM / "
                           "justified P/B, ICG cap, payout path, valuation bridge, sensitivity grids (COE x g, COE x "
                           "ROE*), implied ER* / lambda / COE / ROE*, iso-value surfaces, rerating, scenarios, USD "
                           "returns and breakeven. Static: AR(1) persistence and break tests, interval coverage, "
                           "bootstraps, Monte Carlo, beta estimation, the NIM x cost-of-risk grid (81 full engine "
                           "runs per bank and date), liquidity and capacity from daily trading data, chapters 13-14."),
        ("Point in time", "31 Dec 2025 (historical) uses only information public by that date (FY2024 audited + 9M "
                          "2025 interim); 31 Oct 2026 (current) uses FY2025 audited + 9M 2026 interim. The two dates "
                          "sit in separate blocks on the Inputs sheet and the Summary (Blueprint D1)."),
        ("Data vintage", f"Findings pack {packs[-1].stem.replace('findings_pack_', '') if packs else 'n/a'}; "
                         f"assumption register {reg}; valuation run log {rl.get('run_id', 'n/a')} (git "
                         f"{str(rl.get('git_commit', 'n/a'))[:12]})."),
        ("Dataset", str(rep.get("zenodo_doi") or "see the repository README")),
        ("Code", f"{rep.get('code_doi') or ''}  {rep.get('github_url') or ''}".strip() or "see the repository README"),
        ("Engine runs", "One sheet per bank, date and scenario (12), plus two bridge runs. Explicit years are solved "
                        f"by fixed-point iteration ({EG.ITER} unrolled passes); the base-case ICG growth cap by an "
                        f"unrolled secant search ({EG.SECANT} steps); implied values by {BISECT}-step bisection."),
        ("Checks", f"{n_checks} live cells are compared with the Python outputs after every build."),
        ("Published figures", "The Published columns on the Summary hold the pipeline's figures; they correspond "
                              "to the default inputs, so with the inputs unchanged every Difference cell is zero "
                              "(red otherwise)."),
        ("Convergence", "Each engine run flags whether its fixed-point solve and growth-cap search converged; the "
                        "Summary rolls the flags up (All engine runs converged)."),
        ("Report figures", "Every report exhibit sits on its chapter sheet, numbered and titled as in the report: a "
                           "native Excel chart where faithful (green data cells are live), a colour scale on grids, "
                           "or the report image (captioned 'Static image from report')."),
        ("No macros", "No macros, no external links; .xlsx only. Excel 2019 or later, or LibreOffice 7.x."),
        ("Disclaimer", "This is independent research, prepared for information and research purposes. It is not "
                       "investment advice, an offer or a solicitation to buy or sell any security, and it does not "
                       "take into account the objectives, financial situation or needs of any reader. Ratings, "
                       "values and target prices are the outputs of the stated models and rules under the stated "
                       "assumptions; they are not recommendations to transact. Data come from public sources "
                       "believed to be reliable but are not guaranteed. Past performance is not a guide to future "
                       "returns, and investments in Tanzanian equities carry market, liquidity and currency risks."),
    ]
    for lab, txt in rows:
        r = sh.row
        sh.put(r, 1, lab, "label", bold=True)
        sh.put(r, 2, txt, "value", wrap=True)
        sh.ws.row_dimensions[r].height = 15 * max(1, math.ceil(len(txt) / 120))
        sh.row += 1
    sh.row += 1
    sh.heading("Sheets", 4)
    r0 = sh.row
    for j, h in enumerate(("sheet", "content", "live cells", "static cells"), 1):
        sh.put(r0, j, h, "label", bold=True)
    for i, (name, desc) in enumerate(sheets, 1):
        sh.put(r0 + i, 1, name)
        sh.put(r0 + i, 2, desc)
        sh.put(r0 + i, 3, live.live.get(name, 0) or None, "value", "int")
        sh.put(r0 + i, 4, live.static.get(name, 0) or None, "value", "int")
    sh.width(3, 12)
    sh.width(4, 12)


def build_model(root: Path | None = None) -> tuple[Workbook, Checks, Live, list[dict]]:
    root = root or find_root()
    st = settings(root)
    ctx = build_ctx(root)
    T = _tables(root)
    ca = pd.read_csv(root / "config" / "corporate_actions.csv")
    splits = {r.bank_id: float(r.ratio) for r in ca.itertuples() if r.action == "split"}
    wb = Workbook()
    wb.remove(wb.active)
    checks, live = Checks(), Live()
    rd = Sheet(wb, "README")
    sm = Sheet(wb, "Summary")
    IN = build_inputs(wb, ctx, st, splits)
    dp = Sheet(wb, "Data_Panel")
    dc = Sheet(wb, "Data_Chapters")
    dc.title_block("Data for the chapter formulas (values as computed in Python, with their source tables)", "", 14)
    chs = {}
    for s, title, part, _ in CHAPTERS:
        chs[s] = Sheet(wb, s)
        chs[s].title_block(f"Chapter {title} (Part {part})", "Live blocks: formulas reproducing the pipeline table "
                           "named in each heading. Grey blocks: static values computed in Python.", 12)
        chs[s].width(1, 30)
    srch = Sheet(wb, "Ch10 searches")
    panel(dp, root)
    part1_2(chs, dc, T, checks, live)
    GRID_MAPS.clear()
    runs, bridge = build_runs(wb, IN)
    cells9 = ch09(chs["Ch09"], runs, IN, T, checks, live)
    br = ch10(chs["Ch10"], srch, runs, bridge, IN, T, cells9, checks, live)
    ch11(chs["Ch11"], runs, T, checks, live)
    ch12(chs["Ch12"], srch, chs["Ch09"], runs, IN, T, cells9, st["ch12"], checks, live)
    statics(chs["Ch13"], T, sorted(k for k in T if k.startswith("ch13_")), CHAP["Ch13"], live, 300)
    statics(chs["Ch14"], T, sorted(k for k in T if k.startswith("ch14_")), CHAP["Ch14"], live, 300)
    grid_maps = list(GRID_MAPS)
    fig_rows = grid_maps + add_figures(root, chs, checks.items)
    summary(sm, runs, bridge, cells9, br, T, chs["Ch09"], chs["Ch10"])
    # exact counts: formula cells (live) and grey cells (static) on every sheet; print-ready page setup
    live = Live()
    for ws in wb.worksheets:
        n_f = n_s = 0
        for row in ws.iter_rows():
            for c in row:
                if c.data_type == "f" or (isinstance(c.value, str) and c.value.startswith("=") and c.data_type != "s"):
                    n_f += 1
                elif c.fill is not None and c.fill.fgColor is not None and c.fill.fgColor.rgb in ("00E7E6E6",
                                                                                                    "FFE7E6E6"):
                    n_s += 1
        live.add(ws.title, n_live=n_f, n_static=n_s)
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.print_title_rows = "1:2"
    sheets =[("README", "this sheet"), ("Summary", "values, targets, ratings and the bridge"),
              ("Inputs", "every changeable assumption (blue)"), ("Data_Panel", "NMB and CRDB bank-year panel"),
              ("Data_Chapters", "inputs of the chapter formulas")] + \
             [(s, t) for s, t, _, _ in CHAPTERS] + [("Ch10 searches", "bisection workings of the implied values")] + \
             [(r.sheet.title, "engine run") for r in runs.values()] + \
             [(r.sheet.title, "bridge engine run") for r in bridge.values()]
    readme(rd, root, live, len(checks.items), sheets)
    for s in chs.values():
        s.ws.freeze_panes = "B4"
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.calcMode = "auto"
    return wb, checks, live, fig_rows


def panel(dp: Sheet, root: Path) -> None:
    w = pd.read_csv(path("processed", root) / "bank_year_wide.csv", dtype={"bank_id": str}, low_memory=False)
    w = w[w["bank_id"].isin(BANKS)].sort_values(["bank_id", "fiscal_year"])
    meta = ["bank_id", "fiscal_year", "reporting_basis", "provenance"]
    num = [c for c in w.columns if c not in meta and pd.api.types.is_numeric_dtype(w[c])]
    dp.title_block("Bank-year panel, NMB (B02) and CRDB (B01), full listed history (data/processed/bank_year_wide.csv)",
                   "Values as validated by the pipeline; money in full TZS, ratios as decimals, per-share as reported. "
                   "Row 4: units; column D: provenance (source tag).", len(meta) + len(num))

    def unit(c: str) -> str:
        if c.startswith("kpi_") or c in ("number_of_shareholders",):
            return "count"
        if c in ("dps", "dps_special", "dps_total", "eps_basic", "share_price_close"):
            return "TZS per share"
        if c.startswith("shares"):
            return "shares"
        if any(x in c for x in ("ratio", "share", "rate", "coverage", "cost_to_income")):
            return "decimal"
        return "TZS"
    cols = meta + num
    for j, c in enumerate(cols, 1):
        dp.put(4, j, unit(c) if c in num else "", "note")
        dp.put(5, j, c, "label", bold=True)
    for i, rec in enumerate(w[cols].itertuples(index=False), 6):
        for j, v in enumerate(rec, 1):
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            dp.put(i, j, v, "value", None if j <= len(meta) else ("num6" if isinstance(v, float) and abs(v) < 10
                                                                    else "tzs"))
    dp.ws.freeze_panes = "E6"
