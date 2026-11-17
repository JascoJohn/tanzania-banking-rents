"""Inputs sheet (D104): every user-changeable assumption, one cell each, with a named range and a source or rule.

Three blocks: global settings (config/settings.yaml ``valuation``), the point-in-time case inputs for each valuation
date and bank (opening balances, interim profit, price, shares, macro inputs; Blueprint D1, VALUATION_SPEC §3), and the
approved assumption register (config/assumption_register.csv; base, bear and bull per driver and year).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tzbank.excel.book import Sheet, define

DATE_LABEL = {"HIST": "31 Dec 2025 (historical)", "CURR": "31 Oct 2026 (current)"}
TICK = {"B02": "NMB", "B01": "CRDB"}
SCEN = ("base", "bear", "bull")

GLOBALS = [  # key, label, value getter (settings), format, rule
    ("w_rim", "RIM weight in the blend", lambda vc: vc["weights"]["rim"], "pct", "D3; settings valuation.weights"),
    ("w_ddm", "DDM weight in the blend", lambda vc: vc["weights"]["ddm"], "pct", "D3; settings valuation.weights"),
    ("lag_months", "Dividend payment lag (months after year-end)", lambda vc: int(vc["dividend_pay_lag_months"]),
     "int", "D62 amended; settings valuation.dividend_pay_lag_months"),
    ("coe_reconcile_bp", "COE reconciliation threshold |A - B| (bp)", lambda vc: float(vc["coe_reconcile_bp"]),
     "int", "VALUATION_SPEC §5; midpoint when |A - B| < threshold"),
    ("coe_central", "Central COE rule (reconcile | A)", lambda vc: str(vc.get("coe_central", "A")), "text",
     "D93: the registered reconciliation rule"),
    ("payout_floor", "Payout floor under the base-case ICG growth cap", lambda vc: float(vc.get("payout_floor", .2)),
     "pct", "D56"),
    ("phase_in_max_gap", "k* phase-in allowed when the year-1 gap is at most", lambda vc: 0.01, "pct",
     "D56 (engine default)"),
    ("fade_shape", "Fade shape (lambda | linear)", lambda vc: str(vc["fade_shape"]), "text",
     "Owner pre-flight D; D8"),
    ("coe_g_min_gap", "Guard: COE - g at least", lambda vc: float(vc.get("coe_g_min_gap", 0.04)), "pct",
     "D64 amended"),
    ("fade_years", "Fade years (structural: the sheets have 10 fade columns)", lambda vc: int(vc["fade_years"]),
     "int", "Structural; changing it needs a rebuild in Python"),
]

CASE = [  # key, label, getter(b, o), format, source key in b.sources (fiscal-year prefixed) or note
    ("vd", "Valuation date", lambda b, o: b.vd.to_pydatetime(), "date", "Blueprint D1"),
    ("anchor_fy", "Anchor fiscal year (audited)", lambda b, o: int(b.anchor_fy), "int", "latest audited year public by the date"),
    ("price", "Share price at the date (TZS)", lambda b, o: float(b.price), "ps", "DSE daily report at the date"),
    ("shares", "Shares outstanding used for per-share values", lambda b, o: float(b.shares), "tzs", "shares_outstanding"),
    ("net_loans", "Net loans (anchor year, TZS)", lambda b, o: float(b.fy["net_loans"]), "tzs", "net_loans"),
    ("customer_deposits", "Customer deposits (TZS)", lambda b, o: float(b.fy["customer_deposits"]), "tzs",
     "customer_deposits"),
    ("iea", "Interest-earning assets (TZS)", lambda b, o: float(b.fy["interest_earning_assets"]), "tzs",
     "interest_earning_assets"),
    ("borrowings", "Borrowings (wholesale funding, TZS)", lambda b, o: float(b.fy["borrowings"]), "tzs", "borrowings"),
    ("nci_equity", "Non-controlling interests (TZS)", lambda b, o: float(b.fy["nci_equity"]), "tzs", "nci_equity"),
    ("total_assets", "Total assets (TZS)", lambda b, o: float(b.fy["total_assets"]), "tzs", "total_assets"),
    ("equity_attr", "Equity attributable (TZS)", lambda b, o: float(b.fy["equity_attributable"]), "tzs",
     "equity_attributable"),
    ("profit_attr", "Profit attributable, anchor year (TZS)", lambda b, o: float(b.fy["profit_attributable"]), "tzs",
     "profit_attributable"),
    ("final_dps", "Final DPS declared for the anchor year (TZS)", lambda b, o: float(b.final_dps_anchor), "ps", "dps"),
    ("shares_out", "Shares outstanding, anchor year-end", lambda b, o: float(b.fy["shares_outstanding"]), "tzs",
     "shares_outstanding"),
    ("ytd_profit", "Interim profit attributable, year to date (TZS)",
     lambda b, o: float(b.interim["profit_attributable"]), "tzs", "interim"),
    ("ytd_months", "Interim months", lambda b, o: int(b.interim_months or 0), "int", "interim"),
    ("y_long", "TZS long-bond yield (D34)", lambda b, o: b.macro["y_long"], "pct", "macro"),
    ("ds", "Tanzania default spread", lambda b, o: b.macro["ds"], "pct", "macro (Damodaran)"),
    ("erp", "Mature-market implied ERP", lambda b, o: b.macro["erp"], "pct", "macro (Damodaran)"),
    ("crp", "Tanzania country risk premium", lambda b, o: b.macro["crp"], "pct", "macro (Damodaran)"),
    ("rf_us", "US 10-year Treasury yield", lambda b, o: b.macro["rf_us"], "pct", "macro"),
    ("infl_tz", "Tanzania forward inflation (approach B)", lambda b, o: b.macro.get("infl_tz_fwd"), "pct",
     "IMF WEO / REO at the date (D64)"),
    ("infl_us", "US forward inflation (approach B)", lambda b, o: b.macro.get("infl_us_fwd"), "pct",
     "IMF WEO at the date (D64)"),
    ("split", "Share split factor to the post-split basis", None, "num", "config/corporate_actions.csv"),
]


@dataclass
class Inputs:
    sheet: Sheet
    glob: dict = field(default_factory=dict)          # key -> (r, c)
    case: dict = field(default_factory=dict)          # (val, bank) -> {key: (r, c)}
    reg: dict = field(default_factory=dict)           # (val, tick, symbol, period) -> {scen: (r, c)}
    reg_rows: dict = field(default_factory=dict)      # (val, tick, symbol, period) -> register row (Series)

    def g(self, key: str, other: Sheet) -> str:
        return self.sheet.ref(*self.glob[key], other)

    def c(self, val: str, bank: str, key: str, other: Sheet) -> str:
        return self.sheet.ref(*self.case[(val, bank)][key], other)

    def periods(self, val: str, tick: str, symbol: str) -> list[str]:
        return sorted(p for (v, t, s, p) in self.reg if v == val and t == tick and s == symbol)

    def path(self, val: str, tick: str, symbol: str, scen: str, other: Sheet) -> list[str]:
        return [self.sheet.ref(*self.reg[(val, tick, symbol, p)][scen], other) for p in self.periods(val, tick, symbol)]

    def one(self, val: str, tick: str, symbol: str, scen: str, other: Sheet) -> str | None:
        ps = self.periods(val, tick, symbol)
        return self.sheet.ref(*self.reg[(val, tick, symbol, ps[0])][scen], other) if ps else None


def build_inputs(wb, ctx, settings: dict, splits: dict[str, float]) -> Inputs:
    vc = settings["valuation"]
    sh = Sheet(wb, "Inputs")
    sh.title_block("Inputs: every user-changeable assumption (blue cells)",
                   "Change a blue cell and every chapter sheet and the Summary recalculate. Named ranges: "
                   "g_<key>, <date>_<bank>_<key>, R_<date>_<bank>_<symbol>_<period>_<scenario>.", 14)
    for c, w in ((1, 48), (2, 18), (3, 18), (4, 18), (5, 14), (6, 60)):
        sh.width(c, w)
    out = Inputs(sh)
    # globals
    sh.heading("1. Global settings (config/settings.yaml, valuation)", 14)
    for key, label, get, fmt, rule in GLOBALS:
        r = sh.row
        sh.put(r, 1, label)
        sh.put(r, 2, get(vc), "input" if key != "fade_years" else "value", fmt)
        sh.put(r, 6, rule, "note")
        out.glob[key] = (r, 2)
        define(wb, f"g_{key}", sh, r, 2)
        sh.row += 1
    sh.row += 1
    # case inputs
    sh.heading("2. Point-in-time case inputs per valuation date and bank (keep the two dates apart: Blueprint D1)", 14)
    sh.note("31 Dec 2025 uses only information published on or before 31 Dec 2025 (FY2024 audited + 9M 2025 interim); "
            "31 Oct 2026 uses FY2025 audited + 9M 2026 interim. Publication dates in column F.")
    for val in ("HIST", "CURR"):
        for bank in ("B02", "B01"):
            b, o = ctx.inputs[(val, bank)], ctx.inputs[(val, bank, "opening")]
            r = sh.heading(f"2.{'1' if val == 'HIST' else '2'} {TICK[bank]}, {DATE_LABEL[val]}", 6)
            cells = {}
            for key, label, get, fmt, src in CASE:
                r = sh.row
                v = splits.get(bank, 1.0) if key == "split" and val == "HIST" else (1.0 if key == "split" else
                                                                                     get(b, o))
                sh.put(r, 1, label)
                sh.put(r, 2, v, "input", fmt)
                pub = b.sources.get(f"FY{b.anchor_fy}.{src}") if isinstance(src, str) else None
                sh.put(r, 6, f"{src}" + (f" (FY{b.anchor_fy}, published {pub})" if pub else ""), "note")
                cells[key] = (r, 2)
                define(wb, f"{val}_{TICK[bank]}_{key}", sh, r, 2)
                sh.row += 1
            # derived openings (formulas)
            for key, label, f, fmt in (
                    ("liquid", "Opening liquid assets = IEA - net loans", f"={sh.ref(*cells['iea'], sh)}-"
                     f"{sh.ref(*cells['net_loans'], sh)}", "tzs"),
                    ("e_ex", "Opening equity ex the declared final dividend = equity - final DPS x shares",
                     f"={sh.ref(*cells['equity_attr'], sh)}-{sh.ref(*cells['final_dps'], sh)}*"
                     f"{sh.ref(*cells['shares_out'], sh)}", "tzs"),
                    ("kappa", "NCI share of total equity (held constant)", None, "pct")):
                r = sh.row
                if key == "kappa":
                    f = (f"={sh.ref(*cells['nci_equity'], sh)}/({sh.ref(*cells['e_ex'], sh)}+"
                         f"{sh.ref(*cells['nci_equity'], sh)})")
                sh.put(r, 1, label)
                sh.put(r, 2, f, "formula", fmt)
                cells[key] = (r, 2)
                sh.row += 1
            out.case[(val, bank)] = cells
            sh.row += 1
    # register
    reg = ctx.reg.copy()
    sh.heading("3. Assumption register (config/assumption_register.csv, approved " +
               str(reg["version"].iloc[0]) + "): base, bear, bull", 14)
    sh.note("Path drivers have one row per forecast year; scalars one row. Rationale and evidence as in the register.")
    hdr = ["id", "valuation", "bank", "symbol", "period", "parameter", "base", "bear", "bull", "distribution",
           "range_low", "range_high", "information_cutoff", "rationale (short)"]
    r0 = sh.row
    for j, h in enumerate(hdr, 1):
        sh.put(r0, j, h, "label", bold=True)
    sh.row = r0 + 1
    for rec in reg.itertuples(index=False):
        r = sh.row
        vals = [rec.id, rec.valuation, rec.bank, rec.symbol, rec.period, rec.parameter]
        for j, v in enumerate(vals, 1):
            sh.put(r, j, v)
        key = (rec.valuation, rec.bank, rec.symbol, rec.period)
        out.reg[key] = {}
        for j, s in enumerate(SCEN, 7):
            v = float(getattr(rec, s))
            sh.put(r, j, v, "input", "num6")
            out.reg[key][s] = (r, j)
            define(wb, f"R_{rec.valuation}_{rec.bank}_{rec.symbol}_{rec.period}_{s}", sh, r, j)
        sh.put(r, 10, rec.distribution)
        for j, k in ((11, "range_low"), (12, "range_high")):
            v = getattr(rec, k)
            sh.put(r, j, float(v) if pd.notna(v) else None, "value", "num6")
        sh.put(r, 13, str(rec.information_cutoff)[:200], "note")
        sh.put(r, 14, str(rec.rationale)[:240], "note")
        out.reg_rows[key] = rec
        sh.row += 1
    sh.ws.freeze_panes = "B4"
    return out
