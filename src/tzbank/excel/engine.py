"""Engine run sheets (D104): the driver-based forecast and the RIM / DDM valuation as live Excel formulas.

Mirrors ``tzbank.banks.engine.project`` and ``tzbank.valuation.models`` (VALUATION_SPEC §4 and §6) for one bank, date
and scenario per sheet:

- explicit years t = 1..5: loans and deposits grow by the register paths; liquid assets or wholesale funding balance
  the sheet; the year is solved by fixed-point iteration, unrolled here as ``ITER`` passes (the map contracts by about
  1% a pass, so 8 passes reach machine precision); NIM moves with the rate path through the asset and deposit betas;
  first year = interim actual + remaining months; CRDB subsidiary share with its FX path and distributable capacity;
  dividends = min(payout cap x profit, capital-rule capacity, distributable);
- the base case caps growth from year 2 by internal capital generation (D56): where the capital rule leaves less than
  the payout floor, loan (and deposit) growth is lowered to the rate at which it just allows the floor. Python finds it
  by bisection; the sheet solves the same equation with an unrolled secant search (``SECANT`` steps);
- k* phase-in from the year-1 ratio (D56); fade years t = 6..15 (ROE to ROE*, growth to terminal g); terminal value;
- valuation at any date with the dividend payment lag (D62 amended): RIM and DDM, blend, target price, dividends in
  the 12-month window.

``value_expr`` gives the value at a date in the DDM form (equal to RIM by clean surplus, VALUATION_SPEC §6) for any
discount rate, ROE*, fade speed and growth path, as one formula: the sensitivity grids, the iso-value surface and the
implied-value searches use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tzbank.excel.book import Sheet, a1, col
from tzbank.excel.inputs import TICK, Inputs

T, FADE = 5, 10
N = T + FADE
ITER = 8          # fixed-point passes per year
SECANT = 8        # secant steps for the ICG growth cap
PATHS = ("g_L", "g_D", "NIM", "phi", "CIR", "CoR", "rho", "rate")
C0 = 3            # column of t = 0


def tc(t: int) -> int:
    return C0 + t


@dataclass
class Run:
    sheet: Sheet
    val: str
    bank: str
    scen: str
    kind: str                      # "case" or "bridge"
    rows: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)   # scalar name -> (r, c)
    gvar: list = field(default_factory=list)     # grid g-variant rows: (g cell, gf row, Ep row, EN cell)

    @property
    def tick(self) -> str:
        return TICK[self.bank]

    def x(self, name: str, other: Sheet) -> str:
        return self.sheet.ref(*self.cells[name], other)

    def rowref(self, name: str, t0: int, t1: int, other: Sheet) -> str:
        r = self.rows[name]
        return self.sheet.rng(r, tc(t0), r, tc(t1), other)

    def at(self, name: str, t: int, other: Sheet) -> str:
        return self.sheet.ref(self.rows[name], tc(t), other)

    def value_expr(self, other: Sheet, r: str, X: str, lam: str, g: str | None = None, gpath: int | None = None,
                   which: str = "v0", shape: str | None = None) -> str:
        """Equity value at the valuation date (``v0``) or one year later (``v1``), DDM form, for discount rate ``r``,
        ROE* ``X``, fade speed ``lam`` and terminal growth ``g`` (default: the run's; ``gpath`` = index of a grid
        g-variant whose growth path and E path are used). Excel expression without the leading '='."""
        s = self
        k, f = s.x(f"{which}_k", other), s.x(f"{which}_f", other)
        flag, held, yrs = s.x(f"{which}_flag", other), s.x(f"{which}_heldamt", other), s.x(f"{which}_yrs", other)
        divk = s.x(f"{which}_divk", other)
        lag_l, lagm = s.x("lag_l", other), s.x("lag_m", other)
        tE, divE, yE = s.rowref("t", 1, T, other), s.rowref("Div", 1, T, other), s.rowref("y", 1, T, other)
        F = s.rowref("F", T + 1, N, other)
        yF = s.at("y", N, other)
        roeT = s.x("roe_T", other)
        if gpath is None:
            gf, Ep, EN = s.rowref("g_E", T + 1, N, other), s.rowref("E_prev", T + 1, N, other), s.at("E", N, other)
            g = g or s.x("g", other)
        else:
            gcell, grow, erow, encell = s.gvar[gpath]
            gf = s.sheet.rng(grow, tc(T + 1), grow, tc(N), other)
            Ep = s.sheet.rng(erow, tc(T + 1), erow, tc(N), other)
            EN = s.sheet.ref(*encell, other)
            g = s.sheet.ref(*gcell, other)
        shp = shape or s.x("shape", other)
        # a boolean switch, not IF: IF over an array inside SUMPRODUCT needs array entry in Excel
        roe_f = (f"(({shp}=\"linear\")*({roeT}+({X}-{roeT})*{F}/{FADE})+({shp}<>\"linear\")*"
                 f"({X}+{lam}^{F}*({roeT}-{X})))")
        phiF = f"((1+{yF})/(1+{r}))^{lag_l}"
        explicit = f"SUMPRODUCT(({tE}>{k})*{divE}*((1+{yE})/(1+{r}))^{lag_l}/(1+{r})^({tE}-{k}))"
        fade = f"SUMPRODUCT(({roe_f}-{gf})*{Ep}/(1+{r})^({T}+{F}-{k}))"
        term = f"({X}-{g})*{EN}/(({r}-{g})*(1+{r})^({N}-{k}))"
        add = f"IF({flag},IF({lagm}>0,{held}/(1+{r})^{yrs},{divk}),0)"
        return f"(({explicit}+{phiF}*({fade}+{term}))*(1+{r})^{f}+{add})"


def _years(a: str, b: str) -> str:
    """Excel twin of models._years: whole anniversaries + days / 365.25."""
    day = f"IF(AND(MONTH({a})=2,DAY({a})=29),28,DAY({a}))"
    return f"((YEAR({b})-YEAR({a}))+({b}-DATE(YEAR({b}),MONTH({a}),{day}))/365.25)"


class Layout:
    def __init__(self, start: int) -> None:
        self.r = start
        self.rows: dict[str, int] = {}

    def add(self, name: str, gap: int = 0) -> int:
        self.r += gap
        self.rows[name] = self.r
        self.r += 1
        return self.rows[name]


def build_run(wb, IN: Inputs, val: str, bank: str, scen: str, title: str, *, drivers: dict, coe: dict | None,
              roe_link: dict | None, icg: bool, vd_val: str, opening_val: str, r_override: str | None = None,
              grid_g: list[float] | None = None, note: str = "") -> Run:
    """One engine sheet. ``drivers``: symbol -> list of 5 Inputs refs (paths) or one ref (scalars), refs relative to
    Inputs. ``coe``: register refs to build the run's COE (case runs); ``roe_link``: refs to another run's r / ROE* /
    payout* / lambda / g (bridge run). ``vd_val`` / ``opening_val``: which date's case inputs give the valuation date,
    price, shares and the opening balances."""
    sh = Sheet(wb, title)
    run = Run(sh, val, bank, scen, "bridge" if roe_link else "case")
    sh.title_block(f"Engine run: {TICK[bank]}, {val} ({vd_val}), {scen}", note or
                   "Live formulas reproducing tzbank.banks.engine.project and tzbank.valuation.models. Drivers link "
                   "to the Inputs sheet (green); results feed the chapter sheets and the Summary.", tc(N))
    sh.width(1, 46)
    sh.width(2, 12)
    for t in range(N + 1):
        sh.width(tc(t), 16)
    S = run.cells

    def put_scalar(name: str, label: str, v, fmt: str = "num6", kind: str = "auto") -> None:
        r = sh.row
        sh.put(r, 1, label)
        sh.put(r, C0, v, kind, fmt)
        S[name] = (r, C0)
        sh.row += 1

    # --- drivers ------------------------------------------------------------------------------------------------
    sh.heading("Drivers (links to Inputs; years across)", tc(N))
    r = sh.row
    sh.put(r, 1, "forecast year t", bold=True)
    for t in range(1, T + 1):
        sh.put(r, tc(t), t, "value", "int")
    sh.row += 1
    for sym in PATHS:
        r = sh.row
        sh.put(r, 1, sym)
        for t, ref in enumerate(drivers[sym], 1):
            sh.put(r, tc(t), f"={ref}", "link", "num6")
        run.rows[f"drv_{sym}"] = r
        sh.row += 1
    for sym, label in (("rate0", "rate at the anchor (base scenario, year 1)"), ("omega", "other liabilities / TA"),
                       ("theta", "non-earning assets / TA"), ("l_min", "liquid-asset floor / deposits"),
                       ("s_W", "spread paid on wholesale funding"), ("t", "tax rate"), ("nu", "NCI share of profit"),
                       ("delta", "Tier 1 deductions / equity"), ("k_star", "Tier 1 target k*"),
                       ("pi_max", "payout cap"), ("k_req", "Tier 1 requirement (year-1 check)"),
                       ("asset_beta", "asset beta"), ("deposit_beta", "deposit beta"),
                       ("g_fade", "fade-start growth (D64 amended)"), ("s_sub", "subsidiary share of profit"),
                       ("u", "upstream ratio of subsidiary profit"), ("fx", "subsidiary FX depreciation a year")):
        if sym in drivers and drivers[sym] is not None:
            put_scalar(sym, f"{sym}: {label}", f"={drivers[sym]}", "num6")
    if "s_sub" in S:
        put_scalar("payout_sub", "subsidiary payout (structural, run.case)", 1.0, "num6", "value")
    put_scalar("lag_m", "dividend payment lag (months)", f"={IN.g('lag_months', sh)}", "int")
    put_scalar("w_rim", "RIM weight", f"={IN.g('w_rim', sh)}", "pct")
    put_scalar("w_ddm", "DDM weight", f"={IN.g('w_ddm', sh)}", "pct")
    put_scalar("shape", "fade shape", f"={IN.g('fade_shape', sh)}", "text")
    put_scalar("floor", "payout floor (ICG cap)", f"={IN.g('payout_floor', sh)}", "pct")
    put_scalar("phase_max", "k* phase-in maximum gap", f"={IN.g('phase_in_max_gap', sh)}", "pct")
    put_scalar("icg", "ICG growth cap applies (base case = 1)", 1 if icg else 0, "int", "value")
    ov = opening_val
    put_scalar("ytd", "interim profit attributable, year to date", f"={IN.c(ov, bank, 'ytd_profit', sh)}", "tzs")
    put_scalar("months", "interim months", f"={IN.c(ov, bank, 'ytd_months', sh)}", "int")
    put_scalar("kappa", "NCI share of total equity", f"={IN.c(ov, bank, 'kappa', sh)}", "pct")
    # COE and long-run parameters
    sh.heading("Cost of equity and long-run parameters (VALUATION_SPEC §4-5; D93)", tc(N))
    if coe:
        cv = vd_val
        put_scalar("beta", "beta (register, this scenario)", f"={coe['beta']}")
        put_scalar("LP", "illiquidity premium LP (register, this scenario)", f"={coe['LP']}")
        put_scalar("y_long", "TZS long-bond yield", f"={IN.c(cv, bank, 'y_long', sh)}")
        put_scalar("ds", "default spread", f"={IN.c(cv, bank, 'ds', sh)}")
        put_scalar("erp", "mature ERP", f"={IN.c(cv, bank, 'erp', sh)}")
        put_scalar("crp", "country risk premium", f"={IN.c(cv, bank, 'crp', sh)}")
        put_scalar("rf", "US 10-year yield", f"={IN.c(cv, bank, 'rf_us', sh)}")
        put_scalar("pi_tz", "Tanzania forward inflation", f"={IN.c(cv, bank, 'infl_tz', sh)}")
        put_scalar("pi_us", "US forward inflation", f"={IN.c(cv, bank, 'infl_us', sh)}")
        x = {k: sh.ref(*S[k], sh) for k in ("beta", "LP", "y_long", "ds", "erp", "crp", "rf", "pi_tz", "pi_us")}
        put_scalar("coe_a", "COE approach A = (y_long - DS) + beta (ERP + CRP) + LP",
                   f"=({x['y_long']}-{x['ds']})+{x['beta']}*({x['erp']}+{x['crp']})+{x['LP']}")
        put_scalar("coe_a_reg", "COE A in the register (check)", f"={coe['COE_A']}")
        put_scalar("coe_b", "COE approach B = (1 + rf + beta (ERP + CRP) + LP)(1 + pi_TZ)/(1 + pi_US) - 1",
                   f"=IF(OR({x['pi_tz']}=\"\",{x['pi_us']}=\"\"),\"\",(1+{x['rf']}+{x['beta']}*({x['erp']}+"
                   f"{x['crp']})+{x['LP']})*(1+{x['pi_tz']})/(1+{x['pi_us']})-1)")
        A, B, bp = sh.ref(*S["coe_a"], sh), sh.ref(*S["coe_b"], sh), IN.g("coe_reconcile_bp", sh)
        put_scalar("coe_central", "central COE: midpoint when |A - B| < threshold, else A",
                   f"=IF({B}=\"\",{A},IF(ABS({A}-{B})*10000<{bp},({A}+{B})/2,{A}))")
        rule = IN.g("coe_central", sh)
        put_scalar("r", "COE used (r)", f"=IF({rule}=\"A\",{A},{sh.ref(*S['coe_central'], sh)})", "pct4")
        put_scalar("r_base", "base-scenario central COE (for ROE*, D93)",
                   f"={coe['r_base']}" if coe.get("r_base") else f"={sh.ref(*S['coe_central'], sh)}", "pct4")
        put_scalar("er_star", "ER* (register, this scenario)", f"={coe['ER_star']}")
        put_scalar("g", "terminal growth g (register)", f"={coe['g']}")
        put_scalar("lam", "fade speed lambda (register)", f"={coe['lambda']}")
        rr, er, g = sh.ref(*S["r_base"], sh), sh.ref(*S["er_star"], sh), sh.ref(*S["g"], sh)
        put_scalar("roe_star", "ROE* = base central COE + ER* (D93; register ROE* when the rule is A)",
                   f"=IF({rule}=\"A\",{coe['ROE_star']},{rr}+{er})", "pct4")
        put_scalar("payout_star", "payout* = 1 - g / ROE* (register payout* when the rule is A)",
                   f"=IF({rule}=\"A\",{coe['payout_star']},1-{g}/{sh.ref(*S['roe_star'], sh)})", "pct4")
        put_scalar("guard", "guard: COE - g >= minimum gap (D64 amended)",
                   f"=IF({sh.ref(*S['r'], sh)}-{g}>={IN.g('coe_g_min_gap', sh)}-1E-12,\"PASS\",\"FAIL\")", "text")
    else:
        for k, label in (("r", "COE used (r): the historical COE (bridge step 3)"), ("roe_star", "ROE* (historical)"),
                         ("payout_star", "payout* (historical)"), ("lam", "lambda (historical)"),
                         ("g", "terminal g (historical)")):
            put_scalar(k, label, f"={roe_link[k]}", "pct4")
    if r_override:
        put_scalar("r_val", "COE used to value this run", f"={r_override}", "pct4")
    else:
        put_scalar("r_val", "COE used to value this run", f"={sh.ref(*S['r'], sh)}", "pct4")
    put_scalar("shares", "shares", f"={IN.c(vd_val, bank, 'shares', sh)}", "tzs")
    put_scalar("price", "price (TZS)", f"={IN.c(vd_val, bank, 'price', sh)}", "ps")
    put_scalar("vd", "valuation date", f"={IN.c(vd_val, bank, 'vd', sh)}", "date")
    sh.row += 1

    # --- layout of the results, valuation and workings blocks -------------------------------------------------------
    L = Layout(sh.row)
    L.add("h_results")
    for name in ("t", "year", "phase", "F", "L", "D", "LA", "W", "TA", "NIM_eff", "NII", "NonII", "OpEx", "Imp", "PBT",
                 "NP", "NP_attr", "Div", "E", "E_prev", "E_tot", "RWA", "T1", "T1_ratio", "k_target", "cap_div",
                 "distributable", "payout", "ROE", "g_E", "loan_growth", "icg_flag", "fx_f", "s_eff"):
        L.add(name)
    L.add("h_val", 1)
    for name in ("y", "pay", "phi"):
        L.add(name)
    L.add("h_valsc", 1)
    vrows = ("date", "k", "anchor", "f", "pay_k", "flag", "yrs", "divk", "yk", "heldamt", "rim_vex", "ddm_vex",
             "rim", "ddm", "blend", "gap", "per_share", "rim_ps", "ddm_ps")
    for name in vrows:
        L.add(f"v_{name}")
    L.add("h_misc", 1)
    for name in ("lag_l", "roe_T", "g_T", "E_N", "phiT", "divs_window", "dps_window", "exp_return",
                 "identity_out", "identity_res", "chk_balance", "chk_tier1", "chk_clean", "conv_fp", "conv_icg",
                 "converged"):
        L.add(name)
    if grid_g:
        L.add("h_grid", 1)
        for j in range(len(grid_g)):
            L.add(f"gv{j}_g")
            L.add(f"gv{j}_gf")
            L.add(f"gv{j}_Ep")
    # workings
    evals = ["fin"] if not icg else ["reg", "lo", *[f"x{j}" for j in range(1, SECANT + 2)], "fin"]
    L.add("h_work", 1)
    if icg:
        for name in ("gL_reg", "gD_spread", "slack_reg", "bind", "slack_lo", "g_new", "slack_fin"):
            L.add(f"icg_{name}")
    for e in evals:
        L.add(f"{e}_head", 1)
        if e.startswith("x"):
            L.add(f"{e}_g")
        for item in ("L", "D", "lamin", "taw", "avgd", "GL", "GD"):
            L.add(f"{e}_{item}")
        for j in range(1, ITER + 1):
            for item in ("etot", "gap", "W", "TA", "LA", "avgiea", "nim", "nii", "nonii", "npa", "dist", "rwa", "capdiv",
                         "div", "enew"):
                L.add(f"{e}_{j}_{item}")
        L.add(f"{e}_slack")
    R = L.rows
    run.rows.update({k: v for k, v in R.items()})

    def X(name: str) -> str:
        return sh.ref(*S[name], sh)

    def cell(name: str, t: int, absolute: bool = False) -> str:
        return a1(R[name], tc(t), absolute)

    def drv(sym: str, t: int) -> str:
        return a1(run.rows[f"drv_{sym}"], tc(t))

    # --- results block -------------------------------------------------------------------------------------------
    sh.heading("Results by year (t = 0 anchor; 1-5 explicit; 6-15 fade)", tc(N), R["h_results"])
    labels = {"t": "t", "year": "fiscal year", "phase": "phase", "F": "fade year index", "L": "net loans", "D": "deposits",
              "LA": "liquid assets", "W": "wholesale funding", "TA": "total assets", "NIM_eff": "NIM (with the rate "
              "path)", "NII": "net interest income", "NonII": "non-interest income", "OpEx": "operating expenses",
              "Imp": "impairment", "PBT": "profit before tax", "NP": "net profit", "NP_attr": "profit attributable",
              "Div": "dividend for the year", "E": "equity attributable (year-end)", "E_prev": "equity, prior year-end",
              "E_tot": "total equity", "RWA": "risk-weighted assets", "T1": "Tier 1 capital", "T1_ratio": "Tier 1 ratio",
              "k_target": "k* this year (phase-in)", "cap_div": "dividend capacity under k*",
              "distributable": "distributable (CRDB: parent + upstream)", "payout": "payout",
              "ROE": "ROE (on prior equity)", "g_E": "equity growth (fade)", "loan_growth": "loan growth used",
              "icg_flag": "ICG cap applied (1 = yes)", "fx_f": "subsidiary FX factor", "s_eff": "effective sub share"}
    for k, v in labels.items():
        sh.put(R[k], 1, v)
    oc = lambda k: IN.c(opening_val, bank, k, sh)  # noqa: E731
    for t in range(N + 1):
        c = tc(t)
        sh.put(R["t"], c, t, "value", "int")
        sh.put(R["year"], c, f"={oc('anchor_fy')}+{cell('t', t)}", "link", "int")
        sh.put(R["phase"], c, "anchor" if t == 0 else "explicit" if t <= T else "fade", "value")
        if t > T:
            sh.put(R["F"], c, t - T, "value", "int")
    # anchor column
    c = tc(0)
    for k, ref in (("L", oc("net_loans")), ("D", oc("customer_deposits")), ("LA", oc("liquid")), ("W", oc("borrowings")),
                   ("TA", oc("total_assets")), ("E", oc("e_ex"))):
        sh.put(R[k], c, f"={ref}", "link", "tzs")
    sh.put(R["E_tot"], c, f"={oc('e_ex')}+{oc('nci_equity')}", "link", "tzs")
    sh.put(R["fx_f"], c, 1.0, "value", "num6")
    # explicit years: results read the final evaluation
    kap, dl, tax, nu = X("kappa"), X("delta"), X("t"), X("nu")
    for t in range(1, T + 1):
        p = t - 1
        last = f"fin_{ITER}"
        m = {"L": "fin_L", "D": "fin_D", "LA": f"{last}_LA", "W": f"{last}_W", "TA": f"{last}_TA",
             "NIM_eff": f"{last}_nim", "NII": f"{last}_nii", "NonII": f"{last}_nonii", "NP_attr": f"{last}_npa",
             "Div": f"{last}_div", "E": f"{last}_enew", "RWA": f"{last}_rwa", "cap_div": f"{last}_capdiv"}
        for k, src in m.items():
            sh.put(R[k], tc(t), f"={cell(src, t)}", "formula", "pct4" if k == "NIM_eff" else "tzs")
        sh.put(R["OpEx"], tc(t), f"={drv('CIR', t)}*({cell('NII', t)}+{cell('NonII', t)})", "formula", "tzs")
        sh.put(R["Imp"], tc(t), f"={drv('CoR', t)}*({cell('L', p)}+{cell('L', t)})/2", "formula", "tzs")
        sh.put(R["PBT"], tc(t), f"={cell('NII', t)}+{cell('NonII', t)}-{cell('OpEx', t)}-{cell('Imp', t)}", "formula",
               "tzs")
        sh.put(R["NP"], tc(t), f"={cell('PBT', t)}*(1-{tax})", "formula", "tzs")
        sh.put(R["E_tot"], tc(t), f"={cell('E', t)}/(1-{kap})", "formula", "tzs")
        sh.put(R["T1"], tc(t), f"={cell('E', t)}*(1-{dl})", "formula", "tzs")
        sh.put(R["T1_ratio"], tc(t), f"={cell('T1', t)}/{cell('RWA', t)}", "formula", "pct")
        sh.put(R["distributable"], tc(t), f"={cell(f'{last}_dist', t)}", "formula", "tzs")
        sh.put(R["payout"], tc(t), f"=IF({cell('NP_attr', t)}=0,\"\",{cell('Div', t)}/{cell('NP_attr', t)})",
               "formula", "pct")
        sh.put(R["loan_growth"], tc(t), f"={cell('L', t)}/{cell('L', p)}-1", "formula", "pct")
        fx = X("fx") if "fx" in S else None
        sh.put(R["fx_f"], tc(t), f"={cell('fx_f', p)}*(1-{fx})" if fx else 1.0, "formula" if fx else "value", "num6")
        if fx:
            s = X("s_sub")
            sh.put(R["s_eff"], tc(t), f"={s}*{cell('fx_f', t)}/(1-{s}+{s}*{cell('fx_f', t)})", "formula", "num6")
        # k path (D56 phase-in from the year-1 ratio)
        if t == 1:
            sh.put(R["k_target"], tc(t), f"={X('k_star')}", "formula", "pct4")
        else:
            gap = f"({X('k_star')}-{cell('T1_ratio', 1, True)})"
            sh.put(R["k_target"], tc(t), f"=IF(AND({gap}>0,{gap}<={X('phase_max')}+1E-12),"
                   f"{cell('T1_ratio', 1, True)}+{gap}*{t - 1}/{T - 1},{X('k_star')})", "formula", "pct4")
    for t in range(1, N + 1):
        sh.put(R["E_prev"], tc(t), f"={cell('E', t - 1)}", "formula", "tzs")
        sh.put(R["ROE"], tc(t), f"={cell('NP_attr', t)}/{cell('E_prev', t)}", "formula", "pct")
    # fade
    lam, rs, g, shape = X("lam"), X("roe_star"), X("g"), X("shape")
    for t in range(T + 1, N + 1):
        f = cell("F", t)
        sh.put(R["ROE"], tc(t), f"=IF({shape}=\"linear\",{a1(R['roe_T'], C0, True)}+("
               f"{rs}-{a1(R['roe_T'], C0, True)})*{f}/{FADE},{rs}+{lam}^{f}*({a1(R['roe_T'], C0, True)}-{rs}))",
               "formula", "pct")
        sh.put(R["g_E"], tc(t), f"={a1(R['g_T'], C0, True)}+({g}-{a1(R['g_T'], C0, True)})*{f}/{FADE}", "formula",
               "pct")
        sh.put(R["NP_attr"], tc(t), f"={cell('ROE', t)}*{cell('E_prev', t)}", "formula", "tzs")
        sh.put(R["Div"], tc(t), f"={cell('NP_attr', t)}-{cell('g_E', t)}*{cell('E_prev', t)}", "formula", "tzs")
        sh.put(R["E"], tc(t), f"={cell('E_prev', t)}*(1+{cell('g_E', t)})", "formula", "tzs")

    # --- workings: explicit-year evaluations ------------------------------------------------------------------------
    sh.heading("Workings: explicit years solved by fixed-point iteration (unrolled); ICG growth cap by secant search "
               "(base case)", tc(N), R["h_work"])
    if icg:
        for k, lab in (("gL_reg", "register loan growth"), ("gD_spread", "deposit growth minus loan growth (kept)"),
                       ("slack_reg", "capacity above the payout floor at the register growth"),
                       ("bind", "cap binds (year >= 2 and capacity < floor - 1e-6)"),
                       ("slack_lo", "capacity above the floor at growth -50% (Python's search bound)"),
                       ("g_new", "loan growth used"), ("slack_fin", "capacity above the floor at the growth used")):
            sh.put(R[f"icg_{k}"], 1, lab)
    floor = X("floor")

    def evaluation(e: str, t: int, GL: str, GD: str) -> None:
        p = t - 1
        cpv = lambda k: cell(k, p)  # noqa: E731
        sh.put(R[f"{e}_GL"], tc(t), f"={GL}", "formula", "pct4")
        sh.put(R[f"{e}_GD"], tc(t), f"={GD}", "formula", "pct4")
        sh.put(R[f"{e}_L"], tc(t), f"={cpv('L')}*(1+{cell(f'{e}_GL', t)})", "formula", "tzs")
        sh.put(R[f"{e}_D"], tc(t), f"={cpv('D')}*(1+{cell(f'{e}_GD', t)})", "formula", "tzs")
        Lc, Dc = cell(f"{e}_L", t), cell(f"{e}_D", t)
        sh.put(R[f"{e}_lamin"], tc(t), f"={X('l_min')}*{Dc}", "formula", "tzs")
        sh.put(R[f"{e}_taw"], tc(t), f"=({Lc}+{cell(f'{e}_lamin', t)})/(1-{X('theta')})", "formula", "tzs")
        sh.put(R[f"{e}_avgd"], tc(t), f"=({cpv('D')}+{Dc})/2", "formula", "tzs")
        lamin, taw, avgd = cell(f"{e}_lamin", t), cell(f"{e}_taw", t), cell(f"{e}_avgd", t)
        k_t = cell("k_target", t)
        for j in range(1, ITER + 1):
            q = lambda k, j=j: cell(f"{e}_{j}_{k}", t)  # noqa: E731
            Eg = cpv("E") if j == 1 else cell(f"{e}_{j - 1}_enew", t)
            put = lambda k, f, fm="tzs", j=j: sh.put(R[f"{e}_{j}_{k}"], tc(t), f, "formula", fm)  # noqa: E731
            put("etot", f"={Eg}/(1-{kap})")
            put("gap", f"={Lc}+{X('theta')}*{taw}+{lamin}-({Dc}+{X('omega')}*{taw}+{q('etot')})")
            put("W", f"=MAX(0,{q('gap')})")
            put("TA", f"=IF({q('gap')}>0,{taw},({Dc}+{q('etot')})/(1-{X('omega')}))")
            put("LA", f"=IF({q('gap')}>0,{lamin},{q('TA')}*(1-{X('theta')})-{Lc})")
            put("avgiea", f"=(({cpv('L')}+{cpv('LA')})+({Lc}+{q('LA')}))/2")
            put("nim", f"={drv('NIM', t)}+({X('asset_beta')}-{X('deposit_beta')}*{avgd}/{q('avgiea')})*"
                       f"({drv('rate', t)}-{X('rate0')})", "pct4")
            put("nii", f"={q('nim')}*{q('avgiea')}-{X('s_W')}*({cpv('W')}+{q('W')})/2")
            put("nonii", f"={drv('phi', t)}*({cpv('TA')}+{q('TA')})/2")
            base = (f"(({q('nii')}+{q('nonii')})*(1-{drv('CIR', t)})-{drv('CoR', t)}*({cpv('L')}+{Lc})/2)"
                    f"*(1-{tax})*(1-{nu})")
            if "s_sub" in S:
                base = f"{base}*(1-{X('s_sub')}+{X('s_sub')}*{cell('fx_f', t)})"
            if t == 1:
                base = f"IF({X('months')}>0,{X('ytd')}+(12-{X('months')})/12*{base},{base})"
            put("npa", f"={base}")
            if "s_sub" in S:
                put("dist", f"={q('npa')}*(1-{cell('s_eff', t)})+{X('u')}*{X('payout_sub')}*{q('npa')}*"
                            f"{cell('s_eff', t)}")
            put("rwa", f"={drv('rho', t)}*{q('TA')}")
            put("capdiv", f"=MAX(0,(({cpv('E')}+{q('npa')})*(1-{dl})-{k_t}*{q('rwa')})/(1-{dl}))")
            mn = f"MIN({X('pi_max')}*{q('npa')},{q('capdiv')}" + (f",{q('dist')})" if "s_sub" in S else ")")
            put("div", f"=MAX(0,{mn})")
            put("enew", f"={cpv('E')}+{q('npa')}-{q('div')}")
        sh.put(R[f"{e}_slack"], tc(t), f"={cell(f'{e}_{ITER}_capdiv', t)}-{floor}*{cell(f'{e}_{ITER}_npa', t)}",
               "formula", "tzs")

    for e in evals:
        sh.put(R[f"{e}_head"], 1, f"evaluation '{e}'", "label", bold=True)
        for item, lab in (("L", "loans"), ("D", "deposits"), ("lamin", "liquid-asset floor"), ("taw", "TA if wholesale"),
                          ("avgd", "average deposits"), ("GL", "loan growth"), ("GD", "deposit growth")):
            sh.put(R[f"{e}_{item}"], 1, f"  {lab}")
        for j in range(1, ITER + 1):
            for item in ("etot", "gap", "W", "TA", "LA", "avgiea", "nim", "nii", "nonii", "npa", "dist", "rwa",
                         "capdiv", "div", "enew"):
                sh.put(R[f"{e}_{j}_{item}"], 1, f"  pass {j}: {item}")
        sh.put(R[f"{e}_slack"], 1, "  capacity above the payout floor")
        if e.startswith("x"):
            sh.put(R[f"{e}_g"], 1, "  secant point: loan growth")
    for t in range(1, T + 1):
        gl, gd = drv("g_L", t), drv("g_D", t)
        if not icg:
            evaluation("fin", t, gl, gd)
            continue
        sh.put(R["icg_gL_reg"], tc(t), f"={gl}", "formula", "pct4")
        sh.put(R["icg_gD_spread"], tc(t), f"={gd}-{gl}", "formula", "pct4")
        spread = cell("icg_gD_spread", t)
        if t == 1:
            evaluation("fin", t, gl, gd)
            sh.put(R["icg_g_new"], tc(t), f"={gl}", "formula", "pct4")
            sh.put(R["icg_bind"], tc(t), "=FALSE", "formula")
            continue
        evaluation("reg", t, gl, gd)
        sh.put(R["icg_slack_reg"], tc(t), f"={cell('reg_slack', t)}", "formula", "tzs")
        sh.put(R["icg_bind"], tc(t), f"=AND({X('icg')}=1,{cell('reg_slack', t)}<-1E-6)", "formula")
        evaluation("lo", t, "-0.5", f"-0.5+{spread}")
        sh.put(R["icg_slack_lo"], tc(t), f"={cell('lo_slack', t)}", "formula", "tzs")
        # secant search on loan growth: x1 = register growth - 5 pp, x2.. by secant from (register, x1)
        for j in range(1, SECANT + 2):
            e = f"x{j}"
            if j == 1:
                gx = f"{gl}-0.05"
                prev_g, prev_s, pp_g, pp_s = None, None, None, None
            else:
                prev_g = cell(f"x{j - 1}_g", t)
                prev_s = cell(f"x{j - 1}_slack", t)
                pp_g = gl if j == 2 else cell(f"x{j - 2}_g", t)
                pp_s = cell("reg_slack", t) if j == 2 else cell(f"x{j - 2}_slack", t)
                gx = (f"IF(ABS({prev_s}-{pp_s})<1E-300,{prev_g},{prev_g}-{prev_s}*({prev_g}-{pp_g})/"
                      f"({prev_s}-{pp_s}))")
            sh.put(R[f"{e}_g"], tc(t), f"={gx}", "formula", "pct4")
            evaluation(e, t, cell(f"{e}_g", t), f"{cell(f'{e}_g', t)}+{spread}")
        last = cell(f"x{SECANT + 1}_g", t)
        sh.put(R["icg_g_new"], tc(t), f"=IF(AND({cell('icg_bind', t)},{cell('icg_slack_lo', t)}>=0),{last},{gl})",
               "formula", "pct4")
        evaluation("fin", t, cell("icg_g_new", t), f"{cell('icg_g_new', t)}+{spread}")
        sh.put(R["icg_slack_fin"], tc(t), f"={cell('fin_slack', t)}", "formula", "tzs")
    for t in range(1, T + 1):  # results row: 1 where the ICG cap changed the growth
        sh.put(R["icg_flag"], tc(t), f"=IF({cell('icg_bind', t)},1,0)" if icg else 0, "formula" if icg else "value",
               "int")

    # --- misc scalars, valuation ---------------------------------------------------------------------------------
    for k, lab in (("lag_l", "payment lag in years"), ("roe_T", "ROE in the last explicit year"),
                   ("g_T", "fade-start growth"), ("E_N", "equity at the terminal year"),
                   ("phiT", "lag factor for the terminal stream"), ("divs_window", "dividends paid in (vd, vd + 12m]"),
                   ("dps_window", "expected DPS in the 12 months"), ("exp_return", "expected total return"),
                   ("identity_out", "amounts leaving the value in (vd, vd + 12m], carried"),
                   ("identity_res", "target-price identity residual (RIM)"),
                   ("chk_balance", "check: balance sheet closes (max |gap| / TA)"),
                   ("chk_tier1", "check: Tier 1 >= k* x RWA (min headroom)"),
                   ("chk_clean", "check: clean surplus (max error)"),
                   ("conv_fp", "convergence: fixed-point residual, last pass (max over years, relative)"),
                   ("conv_icg", "convergence: growth-cap search, capacity above the floor at the growth used "
                                "(max over capped years, relative to profit)"),
                   ("converged", "converged (fixed point <= 1e-10 and growth-cap search <= 1e-9)")):
        sh.put(R[k], 1, lab)
    S.update({k: (R[k], C0) for k in ("lag_l", "roe_T", "g_T", "E_N", "phiT", "divs_window", "dps_window",
                                       "exp_return", "identity_res", "conv_fp", "conv_icg", "converged")})
    lagm = X("lag_m")
    base_d = "DATE(2000,12,31)"
    pay0 = f"EDATE({base_d},{lagm})"
    sh.put(R["lag_l"], C0, f"=IF({lagm}=0,0,{_years(base_d, pay0)})", "formula", "num6")
    sh.put(R["roe_T"], C0, f"={cell('NP_attr', T)}/{cell('E', T - 1)}", "formula", "pct4")
    gfad = X("g_fade") if "g_fade" in S else '""'
    sh.put(R["g_T"], C0, f"=IF({gfad}=\"\",{cell('E', T)}/{cell('E', T - 1)}-1,{gfad})", "formula", "pct4")
    sh.put(R["E_N"], C0, f"={cell('E', N)}", "formula", "tzs")
    r = X("r_val")
    sh.heading("Valuation (VALUATION_SPEC §6; payment lag D62 amended)", tc(N), R["h_val"])
    for k, lab in (("y", "liquid-asset yield in the payment year (T-bill path)"), ("pay", "dividend payment date"),
                   ("phi", "lag factor ((1 + y)/(1 + r))^l")):
        sh.put(R[k], 1, lab)
    rate_rng = run.sheet.rng(run.rows["drv_rate"], tc(1), run.rows["drv_rate"], tc(T), sh)
    for t in range(N + 1):
        sh.put(R["y"], tc(t), f"=INDEX({rate_rng},MIN({cell('t', t)},{T - 1})+1)", "formula", "pct4")
        sh.put(R["pay"], tc(t), f"=EDATE(DATE({cell('year', t)},12,31),{lagm})", "formula", "date")
        if t >= 1:
            sh.put(R["phi"], tc(t), f"=((1+{cell('y', t)})/(1+{r}))^{a1(R['lag_l'], C0, True)}", "formula", "num6")
    sh.put(R["phiT"], C0, f"=((1+INDEX({rate_rng},{T}))/(1+{r}))^{a1(R['lag_l'], C0, True)}", "formula", "num6")
    sh.heading("Value at the valuation date (column C) and one year later (column D)", tc(N), R["h_valsc"])
    vlabels = {"date": "date", "k": "anchor index k (last year-end on or before the date)", "anchor": "anchor date",
               "f": "years from the anchor to the date", "pay_k": "payment date of dividend k",
               "flag": "dividend k declared, not yet paid", "yrs": "years from the date to its payment",
               "divk": "dividend k", "yk": "yield until paid", "heldamt": "dividend k plus the yield until paid",
               "rim_vex": "RIM value ex dividend at the anchor", "ddm_vex": "DDM value ex dividend at the anchor",
               "rim": "RIM equity value at the date", "ddm": "DDM equity value at the date",
               "blend": "blended equity value (RIM / DDM weights)", "gap": "|RIM - DDM| / RIM",
               "per_share": "value per share (TZS)", "rim_ps": "RIM per share", "ddm_ps": "DDM per share"}
    for k, lab in vlabels.items():
        sh.put(R[f"v_{k}"], 1, lab)
    y0 = cell("year", 0, True)
    tR, npR, eR, epR, dvR, phR = (sh.rng(R[k], tc(1), R[k], tc(N), sh) for k in ("t", "NP_attr", "E", "E_prev", "Div",
                                                                                  "phi"))
    eAll = sh.rng(R["E"], tc(0), R["E"], tc(N), sh)
    EN, phiT, lagl = a1(R["E_N"], C0, True), a1(R["phiT"], C0, True), a1(R["lag_l"], C0, True)
    for j, which in enumerate(("v0", "v1")):
        c = C0 + j
        v = lambda k, c=c: a1(R[f"v_{k}"], c)  # noqa: E731
        sh.put(R["v_date"], c, f"={X('vd')}" if j == 0 else f"=EDATE({X('vd')},12)", "formula", "date")
        d = v("date")
        sh.put(R["v_k"], c, f"=MAX(0,MIN({N - 1},YEAR({d})-{y0}-IF(AND(MONTH({d})=12,DAY({d})=31),0,1)))", "formula",
               "int")
        sh.put(R["v_anchor"], c, f"=DATE({y0}+{v('k')},12,31)", "formula", "date")
        sh.put(R["v_f"], c, f"={_years(v('anchor'), d)}", "formula", "num6")
        sh.put(R["v_pay_k"], c, f"=EDATE({v('anchor')},{lagm})", "formula", "date")
        sh.put(R["v_flag"], c, f"={d}<={v('pay_k')}", "formula")
        sh.put(R["v_yrs"], c, f"={_years(d, v('pay_k'))}", "formula", "num6")
        sh.put(R["v_divk"], c, f"=IF({v('k')}=0,0,INDEX({dvR},{v('k')}))", "formula", "tzs")
        sh.put(R["v_yk"], c, f"=INDEX({rate_rng},MIN({v('k')},{T - 1})+1)", "formula", "pct4")
        sh.put(R["v_heldamt"], c, f"={v('divk')}*(1+{v('yk')})^{lagl}", "formula", "tzs")
        k = v("k")
        tvd = f"({X('roe_star')}-{X('g')})*{EN}/(({r}-{X('g')})*(1+{r})^({N}-{k}))"
        sh.put(R["v_rim_vex"], c, f"=INDEX({eAll},{k}+1)+SUMPRODUCT(({tR}>{k})*({npR}-{r}*{epR})/(1+{r})^({tR}-{k}))"
               f"+({X('roe_star')}-{r})*{EN}/(({r}-{X('g')})*(1+{r})^({N}-{k}))"
               f"+SUMPRODUCT(({tR}>{k})*{dvR}*({phR}-1)/(1+{r})^({tR}-{k}))+{tvd}*({phiT}-1)", "formula", "tzs")
        sh.put(R["v_ddm_vex"], c, f"=SUMPRODUCT(({tR}>{k})*{dvR}*{phR}/(1+{r})^({tR}-{k}))+{tvd}*{phiT}", "formula",
               "tzs")
        add = f"IF({v('flag')},IF({lagm}>0,{v('heldamt')}/(1+{r})^{v('yrs')},{v('divk')}),0)"
        sh.put(R["v_rim"], c, f"={v('rim_vex')}*(1+{r})^{v('f')}+{add}", "formula", "tzs")
        sh.put(R["v_ddm"], c, f"={v('ddm_vex')}*(1+{r})^{v('f')}+{add}", "formula", "tzs")
        sh.put(R["v_blend"], c, f"={X('w_rim')}*{v('rim')}+{X('w_ddm')}*{v('ddm')}", "formula", "tzs")
        sh.put(R["v_gap"], c, f"=ABS({v('rim')}-{v('ddm')})/ABS({v('rim')})", "formula", "pct4")
        sh.put(R["v_per_share"], c, f"={v('blend')}/{X('shares')}", "formula", "ps")
        sh.put(R["v_rim_ps"], c, f"={v('rim')}/{X('shares')}", "formula", "ps")
        sh.put(R["v_ddm_ps"], c, f"={v('ddm')}/{X('shares')}", "formula", "ps")
        for key in ("k", "f", "flag", "heldamt", "yrs", "divk", "per_share", "blend", "rim", "ddm", "gap", "date"):
            S[f"{which}_{key}"] = (R[f"v_{key}"], c)
    S["v0_value"] = S["v0_blend"]
    payR = sh.rng(R["pay"], tc(1), R["pay"], tc(N), sh)
    vd, v1d = a1(R["v_date"], C0, True), a1(R["v_date"], C0 + 1, True)
    sh.put(R["divs_window"], C0, f"=SUMPRODUCT(({payR}+1>{vd})*({payR}+1<={v1d})*{dvR})", "formula", "tzs")
    sh.put(R["dps_window"], C0, f"={a1(R['divs_window'], C0)}/{X('shares')}", "formula", "ps")
    sh.put(R["exp_return"], C0, f"=({a1(R['v_per_share'], C0 + 1)}+{a1(R['dps_window'], C0)}-{X('price')})/"
           f"{X('price')}", "formula", "pct")
    # target identity (RIM): V1 = V0 (1 + r)^yrs - sum of amounts leaving, each carried from its payment date
    yR = sh.rng(R["y"], tc(1), R["y"], tc(N), sh)
    sh.put(R["identity_out"], C0, f"=SUMPRODUCT(({payR}+1>{vd})*({payR}+1<={v1d})*{dvR}*(1+{yR})^{lagl}*(1+{r})^"
           f"((YEAR({v1d})-YEAR({payR}))+({v1d}-DATE(YEAR({v1d}),MONTH({payR}),DAY({payR})))/365.25))",
           "formula", "tzs")
    sh.put(R["identity_res"], C0, f"=({a1(R['v_rim'], C0 + 1)}-({a1(R['v_rim'], C0)}*(1+{r})^"
           f"{_years(vd, v1d)}-{a1(R['identity_out'], C0)}))/{a1(R['v_rim'], C0)}", "formula", "num6")
    # engine checks (VALUATION_SPEC §4, §10)
    ex = lambda k: sh.rng(R[k], tc(1), R[k], tc(T), sh)  # noqa: E731
    sh.put(R["chk_balance"], C0, f"=MAX(INDEX(ABS({ex('TA')}-({ex('D')}+{ex('TA')}*{X('omega')}+{ex('E_tot')}+"
           f"{ex('W')}))/{ex('TA')},0))", "formula", "num6")
    sh.put(R["chk_tier1"], C0, f"=MIN(INDEX(({ex('T1')}-{ex('k_target')}*{ex('RWA')})/{ex('RWA')},0))", "formula",
           "num6")
    eR = sh.rng(R["E"], tc(1), R["E"], tc(N), sh)
    sh.put(R["chk_clean"], C0, f"=MAX(INDEX(ABS({eR}-({epR}+{npR}-{dvR}))/ABS({eR}),0))", "formula", "num6")
    # convergence flags (owner review of step 20): the last fixed-point pass moved E by at most 1e-10 (Python's
    # stopping rule); where the ICG cap binds, the capacity above the payout floor at the growth used is ~0
    e8 = sh.rng(R[f"fin_{ITER}_enew"], tc(1), R[f"fin_{ITER}_enew"], tc(T), sh)
    e7 = sh.rng(R[f"fin_{ITER - 1}_enew"], tc(1), R[f"fin_{ITER - 1}_enew"], tc(T), sh)
    sh.put(R["conv_fp"], C0, f"=MAX(INDEX(ABS({e8}-{e7})/ABS({e8}),0))", "formula", "num6")
    if icg:
        rr = lambda k: sh.rng(R[k], tc(2), R[k], tc(T), sh)  # noqa: E731
        sh.put(R["conv_icg"], C0, f"=MAX(INDEX({rr('icg_bind')}*({rr('icg_slack_lo')}>=0)*ABS({rr('icg_slack_fin')})/"
               f"ABS({sh.rng(R['NP_attr'], tc(2), R['NP_attr'], tc(T), sh)}),0))", "formula", "num6")
    else:
        sh.put(R["conv_icg"], C0, 0, "value", "num6")
    sh.put(R["converged"], C0, f"=IF(AND({a1(R['conv_fp'], C0)}<=1E-10,{a1(R['conv_icg'], C0)}<=1E-9),\"PASS\","
           f"\"FAIL\")", "formula")

    # --- grid growth variants (COE x g grid) ---------------------------------------------------------------------
    if grid_g:
        sh.heading("Fade paths for the COE x terminal-g grid (one per g)", tc(N), R["h_grid"])
        gT = a1(R["g_T"], C0, True)
        for j, dg in enumerate(grid_g):
            rg_, rf_, re_ = R[f"gv{j}_g"], R[f"gv{j}_gf"], R[f"gv{j}_Ep"]
            sh.put(rg_, 1, f"g = base g {dg * 1e4:+.0f} bp")
            sh.put(rg_, C0, f"={X('g')}+{dg}", "formula", "pct4")
            sh.put(rf_, 1, "  equity growth path")
            sh.put(re_, 1, "  equity, prior year-end (last cell: terminal equity in column B)")
            for t in range(T + 1, N + 1):
                sh.put(rf_, tc(t), f"={gT}+({a1(rg_, C0, True)}-{gT})*{cell('F', t, True)}/{FADE}", "formula", "pct4")
                prev = cell("E", T, True) if t == T + 1 else f"{a1(re_, tc(t - 1))}*(1+{a1(rf_, tc(t - 1))})"
                sh.put(re_, tc(t), f"={prev}", "formula", "tzs")
            sh.put(re_, 2, f"={a1(re_, tc(N))}*(1+{a1(rf_, tc(N))})", "formula", "tzs")
            run.gvar.append(((rg_, C0), rf_, re_, (re_, 2)))
    sh.ws.freeze_panes = sh.ws.cell(row=4, column=C0)
    # row grouping: workings collapsed
    sh.ws.row_dimensions.group(R["h_work"] + 1, sh.ws.max_row, hidden=False, outline_level=1)
    return run


def link_cells(IN: Inputs, val: str, tick: str, scen: str) -> tuple[dict, dict]:
    """Driver refs and COE refs of one case from the Inputs register (always cross-sheet references)."""
    drivers = {s: IN.path(val, tick, s, scen, None) for s in PATHS}
    drivers["rate0"] = IN.path(val, tick, "rate", "base", None)[0]
    for s in ("omega", "theta", "l_min", "s_W", "t", "nu", "delta", "k_star", "pi_max", "k_req", "asset_beta",
              "deposit_beta", "g_fade", "s_sub", "u", "fx"):
        drivers[s] = IN.one(val, tick, s, scen, None)
    coe = {k: IN.one(val, tick, k, scen, None) for k in ("beta", "LP", "COE_A", "ER_star", "g", "lambda", "ROE_star",
                                                        "payout_star")}
    return drivers, coe


def col_letter(t: int) -> str:
    return col(tc(t))
