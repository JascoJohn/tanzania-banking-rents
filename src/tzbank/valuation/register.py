"""Proposed assumption register (VALUATION_SPEC §8; prompt 09a task 5; D8, D50, D55).

One row per driver, bank and valuation date with base / bear / bull values, the evidence range the values must sit in,
and the rationale. Information sets are kept apart (D1):

- HIST (31 Dec 2025): anchors, ranges, lambda, ER* and betas are re-computed from data eligible at 31 Dec 2025 only
  (fiscal years up to FY2024, the 9M 2025 interim, macro rows published by the date); `calibration_inputs.csv` is NOT
  used for HIST because it contains FY2025.
- CURR (31 Oct 2026): ranges from `outputs/tables/calibration_inputs.csv` where it covers the driver, chapter tables
  otherwise.

Values are proposals for the owner; the approved file is `config/assumption_register.csv` (versioned).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tzbank.chapters.ch05 import models as m5
from tzbank.chapters.ch06_funding import beta_ols
from tzbank.config import find_root, path, settings
from tzbank.valuation import coe as coe_mod
from tzbank.valuation.inputs import BaseInputs, base_inputs, eligible_long, pick

TICK = {"B01": "CRDB", "B02": "NMB"}
DATES = {"HIST": "2025-12-31", "CURR": "2026-10-31"}
VERSION = "v0.1-proposed"
STRESS = (2015, 2018)


def _tab(root, name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables", root) / name, dtype={"bank_id": str}, low_memory=False)


def _hist(root, bank: str, upto: int) -> dict[str, pd.Series]:
    """Annual history per driver, fiscal years <= ``upto`` (eligible at the valuation date)."""
    def s(df: pd.DataFrame, col: str) -> pd.Series:
        x = df[(df["bank_id"] == bank) & (df["fiscal_year"] <= upto)].set_index("fiscal_year")[col]
        return x.dropna().sort_index()
    r46, ol = _tab(root, "ch04_4_6_risk.csv"), _tab(root, "ch04_4_5_operating_leverage.csv")
    c40, cap = _tab(root, "ch04_4_0_components.csv"), _tab(root, "ch07_capital.csv")
    pay, liq = _tab(root, "ch07_payout.csv"), _tab(root, "ch06_liquidity.csv")
    dep, dpl = _tab(root, "ch06_deposits.csv"), _tab(root, "ch04_4_3_deployment.csv")
    ch = _tab(root, "ch08_credit_history.csv")
    return {"NIM": s(r46, "nim"), "CIR": s(ol, "cost_to_income"), "phi": s(c40, "nonii_ta"),
            "CoR": s(ch, "cost_of_risk"), "rho": s(cap, "rwa_density"), "payout": s(pay, "payout_total"),
            "tax": s(c40, "tax_rate"), "cost_dep": s(dep, "cost_of_deposits"),
            "cost_borr": s(liq, "cost_of_borrowings"), "iea_yield": s(dpl, "iea_yield"),
            "loan_g": s(liq, "loan_growth"), "dep_g": s(liq, "deposit_growth")}


def _pit_persistence(root, bank: str, e: pd.DataFrame, upto: int) -> dict:
    """Break-adjusted Andrews lambda, ER* (Kilian interval) and the through-the-cycle mean ER on ER through ``upto``,
    with ROE from rows eligible at the valuation date (D1) and the year's COE (ch05_coe_annual, FY rows)."""
    cfg, seed = settings(root)["persistence"], int(settings(root)["random_seed"])
    coe = _tab(root, "ch05_coe_annual.csv")
    coe = coe[coe["date_label"].str.startswith("FY")].set_index("year")["coe"]
    roe = {}
    for fy in range(2009, upto + 1):
        npa, _ = pick(e, bank, "profit_attributable", "FY", fy)
        npa = npa if np.isfinite(npa) else pick(e, bank, "net_profit", "FY", fy)[0]
        e1, _ = pick(e, bank, "equity_attributable", "FY", fy)
        e0, _ = pick(e, bank, "equity_attributable", "FY", fy - 1)
        if not np.isfinite(e1):
            e1 = pick(e, bank, "total_equity", "FY", fy)[0]
        if not np.isfinite(e0):
            e0 = pick(e, bank, "total_equity", "FY", fy - 1)[0]
        if np.isfinite(npa) and np.isfinite(e1) and np.isfinite(e0):
            roe[fy] = npa / ((e0 + e1) / 2)
    er = (pd.Series(roe) - coe.reindex(list(roe))).dropna()
    base = m5.pairs(er)
    dm = pd.DataFrame({"d_episode": base["year"].between(*STRESS).astype(int).to_numpy()}, index=base["year"].to_numpy())
    d = m5.pairs(er, dm[["d_episode"]])
    a = m5.andrews(d, cfg["interval"], cfg["andrews_sims"], [seed, 903, 9], ["d_episode"], [0.0])
    k = m5.kilian(d, cfg["bootstrap_reps"], cfg["bootstrap_reps"], cfg["interval"], [seed, 9, 2], ["d_episode"], [0.0])
    mu, lo, hi = m5.mean_interval(er, cfg["bootstrap_reps"], cfg["block_length"], cfg["interval"], [seed, 9, 3])
    return {"lam": a["lambda"], "lam_lo": a["lambda_lo"], "lam_hi": a["lambda_hi"], "er_brk": a["er_star"],
            "er_brk_lo": k["er_star_lo"], "er_brk_hi": k["er_star_hi"], "er_ttc": mu, "er_ttc_lo": lo, "er_ttc_hi": hi,
            "years": f"{int(er.index.min())}-{int(er.index.max())}", "er": er}


def _tbill(root, upto_year: int) -> pd.Series:
    t = _tab(root, "ch06_rates_used.csv").set_index("year")["tbill"].dropna()
    return t[t.index <= upto_year]


class Reg:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, val: str, bank: str, param: str, symbol: str, period: str, base, bear, bull, lo, hi, rng_src: str,
            rationale: str, evidence: str, dist: str = "triangular", scope: str = "bank", cutoff: str = "") -> None:
        vals = [v for v in (base, bear, bull) if isinstance(v, (int, float)) and np.isfinite(v)]
        inside = (all(lo - 1e-12 <= v <= hi + 1e-12 for v in vals)
                  if np.isfinite(lo) and np.isfinite(hi) else None)
        self.rows.append({"id": f"{val}.{TICK.get(bank, bank)}.{symbol}.{period}", "valuation": val, "scope": scope,
                          "bank": TICK.get(bank, bank), "parameter": param, "symbol": symbol, "period": period,
                          "base": base, "bear": bear, "bull": bull, "distribution": dist, "range_low": lo,
                          "range_high": hi, "range_source": rng_src,
                          "inside_range": {True: "yes", False: "NO", None: "no range (judgement)"}[inside],
                          "rationale": rationale, "evidence": evidence, "information_cutoff": cutoff,
                          "set_on": "2026-11-09", "version": VERSION, "status": "PROPOSED (owner approval pending)"})


def _path(start: float, end: float, n: int) -> list[float]:
    return [start + (end - start) * i / (n - 1) for i in range(n)] if n > 1 else [start]


def _cal(cal: pd.DataFrame, bank: str, p: str) -> tuple[float, float, float, float]:
    r = cal[(cal["bank_id"] == bank) & (cal["parameter"] == p)].iloc[0]
    return float(r["range_low"]), float(r["range_high"]), float(r["interval_low"]), float(r["interval_high"])


def build(root=None) -> tuple[pd.DataFrame, dict]:
    """Proposed register for both banks and both dates; also returns the opening inputs used."""
    root = root or find_root()
    st = settings(root)
    vcfg, rules = st["valuation"], st["capital_rules"]
    cal = _tab(root, "calibration_inputs.csv")
    reg = Reg()
    inputs: dict = {}
    for val, vd in DATES.items():
        e = eligible_long(root, vd)
        for bank in ("B02", "B01"):
            b: BaseInputs = base_inputs(bank, vd, root)
            inputs[(val, bank)] = b
            a = b.anchor_fy
            T = int(vcfg["explicit_end"][vd]) - a
            yrs = [str(a + i) for i in range(1, T + 1)]
            h = _hist(root, bank, a)
            cut = f"eligible at {vd}: FY<={a}, {b.interim_months}M {a + 1} interim, macro as at {vd}"
            ev_fy = f"FY{a} audited ({b.sources.get(f'FY{a}.net_profit', '')}); {b.interim_months}M{a + 1} interim"
            curr = val == "CURR"
            # --- growth
            ng = b.macro.get("ngdp_tz", float("nan"))
            pz = b.macro.get("infl_tz_fwd", float("nan"))
            real_lr = float(vcfg.get("terminal_real_growth", 0.03))
            if np.isfinite(pz):  # D64 amended: WEO final-year Tanzanian inflation + 3% long-run real growth
                g_term = {"base": pz + real_lr, "bear": pz + real_lr - 0.01, "bull": pz + real_lr + 0.01}
            else:
                g_term = {"base": 0.07, "bear": 0.06, "bull": 0.08}
            g_fs = ng if np.isfinite(ng) else g_term["base"]  # fade-start growth: IMF nominal GDP growth
            fy, it = b.fy, b.interim
            gl1 = (it.get("net_loans", np.nan) / fy["net_loans"]) ** (12 / b.interim_months) - 1
            gd1 = (it.get("customer_deposits", np.nan) / fy["customer_deposits"]) ** (12 / b.interim_months) - 1
            lpath = None
            for sym, g1, hs, nm in (("g_L", gl1, h["loan_g"], "Net loan growth"),
                                    ("g_D", gd1, h["dep_g"], "Deposit growth")):
                lo, hi = float(hs[hs.index >= 2015].min()), float(hs[hs.index >= 2015].max())
                g1c = float(np.clip(g1, lo, hi))
                base = _path(g1c, g_fs, T)  # explicit path ends at the fade-start growth (D64 amended)
                stress = float(hs[hs.index.to_series().between(*STRESS)].mean())  # 2015-18 experience (09a review C)
                if sym == "g_L":
                    lpath, lstress = base, stress
                    why = ("Year 1 from the annualised interim growth (clipped to history); linear convergence to "
                           "the fade-start growth (IMF nominal GDP growth, D64 amended); bull +2 pp after year 1; bear "
                           "from year 2 = the bank's mean loan growth in the "
                           f"2015-18 stress ({stress:.3f}), never above base (D56 ordering).")
                    bear = [g1c] + [min(float(np.clip(stress, lo, hi)), x) for x in base[1:]]  # bear <= base (D56)
                else:  # deposits fund loan growth one-for-one after year 1: the LDR holds at its year-1 level (D56)
                    base = [g1c] + [float(np.clip(x, lo, hi)) for x in lpath[1:]]
                    why = ("Year 1 from the annualised interim growth; from year 2 deposits grow with loans, so the "
                           "loan-to-deposit ratio holds at its year-1 level (D56); bull +2 pp after year 1; bear from "
                           f"year 2 = the stress-period loan growth ({lstress:.3f}; stress deposit growth {stress:.3f}), "
                           "so the LDR also holds in the bear case.")
                    bear = [g1c] + [min(float(np.clip(lstress, lo, hi)), x) for x in base[1:]]  # bear <= base
                for i, y in enumerate(yrs):
                    reg.add(val, bank, nm, sym, f"FY{y}", base[i], bear[i],
                            min(hi, base[i] + (0 if i == 0 else 0.02)), lo, hi, f"history FY2015-{a} (ch06_liquidity)",
                            why, f"{ev_fy}: {nm.lower()} annualised {g1:.3f}; ch06_liquidity."
                            f"{'loan' if sym == 'g_L' else 'deposit'}_growth FY2015-18 mean {stress:.3f}", cutoff=cut)
            # --- NIM, phi, CIR, CoR, rho paths
            if curr:
                nlo, nhi, _, _ = _cal(cal, bank, "NIM (NII / avg interest-earning assets)")
                clo, chi, _, _ = _cal(cal, bank, "cost-to-income path")
                nsrc, csrc = "calibration_inputs.csv (NIM, FY2019-25)", "calibration_inputs.csv (CIR, FY2019-25)"
            else:
                rr = h["NIM"][h["NIM"].index >= 2019]
                nlo, nhi = float(rr.min()), float(rr.max())
                cc = h["CIR"][h["CIR"].index >= 2019]
                clo, chi = float(cc.min()), float(cc.max())
                nsrc, csrc = f"history FY2019-{a} (ch04_4_6_risk.nim)", f"history FY2019-{a} (ch04_4_5 CIR)"
            nim0, cir0 = float(h["NIM"].iloc[-1]), float(h["CIR"].iloc[-1])
            for i, y in enumerate(yrs):
                reg.add(val, bank, "Net interest margin (at the base rate path)", "NIM", f"FY{y}", nim0,
                        max(nlo, nim0 - 0.005 * (i + 1) / T * 2), min(nhi, nim0 + 0.0025), nlo, nhi, nsrc,
                        "Held at the anchor-year NIM; rate changes enter through the rate path and betas (open issue "
                        "11); bear erodes up to 100 bp by the last explicit year, bull +25 bp; all inside the range.",
                        f"ch04_4_6_risk.nim FY{a} = {nim0:.4f}", cutoff=cut)
                cbase = max(clo, cir0 - 0.005 * i)
                reg.add(val, bank, "Cost-to-income", "CIR", f"FY{y}", cbase, min(chi, cir0 + 0.03),
                        max(clo, cir0 - 0.01 * (i + 1)), clo, chi, csrc,
                        "Base continues half the FY2019-anchor improvement (-0.5 pp a year) floored at the range low; "
                        "bear +3 pp; bull -1 pp a year, floored.", f"ch04_4_5_operating_leverage FY{a} = {cir0:.4f}",
                        cutoff=cut)
            phi = h["phi"]
            p0, plo, phi_hi = float(phi.iloc[-1]), float(phi[phi.index >= 2015].min()), float(phi[phi.index >= 2015].max())
            for y in yrs:
                reg.add(val, bank, "Non-interest income / avg assets", "phi", f"FY{y}", p0, max(plo, p0 * 0.9),
                        min(phi_hi, p0 * 1.1), plo, phi_hi, f"history FY2015-{a} (ch04_4_0 nonii_ta)",
                        "Anchor-year level; bear -10% (mobile-money competition, 4.7 q8), bull +10% (digital fees).",
                        f"ch04_4_0_components.nonii_ta FY{a} = {p0:.4f}", cutoff=cut)
            # CoR: interim -> TTC by the last explicit year; TTC recomputed on eligible years
            cor = h["CoR"]
            cor_ttc = float(cor[cor.index >= 2010].mean())
            cor_st = float(cor[cor.index.to_series().between(*STRESS)].mean())
            cor_ifrs = float(cor[cor.index >= 2018].mean())
            cor1 = it.get("impairment_charges", np.nan) * 12 / b.interim_months / (
                (fy["gross_loans"] + it.get("gross_loans", np.nan)) / 2)
            alo, ahi = float(cor[cor.index >= 2010].min()), float(cor[cor.index >= 2010].max())
            cor1 = float(np.clip(cor1, alo, ahi))
            bp, bearp = _path(cor1, cor_ttc, T), [min(ahi, cor1 + (cor_st - cor1) * min(1, (i + 1) / 3))
                                                   for i in range(T)]
            bull = _path(cor1, min(cor_ttc, cor_ifrs), T)
            for i, y in enumerate(yrs):
                reg.add(val, bank, "Cost of risk", "CoR", f"FY{y}", bp[i], bearp[i], bull[i], alo, ahi,
                        f"annual history FY2010-{a} (ch08_credit_history)",
                        "Year 1 from the annualised interim; base converges to the through-the-cycle mean by the last "
                        "explicit year (VALUATION_SPEC §4); bear reaches the 2015-18 stress mean by year 3; bull ends at "
                        "the lower of TTC and the IFRS 9-regime mean.",
                        f"interim CoR {cor1:.4f}; TTC FY2010-{a} {cor_ttc:.4f}; stress {cor_st:.4f}; IFRS 9 mean "
                        f"{cor_ifrs:.4f}", cutoff=cut)
            if curr:
                tlo, thi, _, _ = _cal(cal, bank, "through-the-cycle cost of risk")
                tsrc = "calibration_inputs.csv (25 cycle boundaries)"
            else:
                g_ = [float(cor[(cor.index >= s) & (cor.index <= e_)].mean()) for s in range(2010, 2015)
                      for e_ in range(2021, a + 1)]
                tlo, thi, tsrc = min(g_), max(g_), f"cycle boundaries FY2010-14 x FY2021-{a} (eligible years)"
            reg.add(val, bank, "Through-the-cycle cost of risk (terminal)", "CoR_ttc", "terminal", cor_ttc,
                    min(thi, max(tlo, cor_st)) if tlo <= cor_st <= thi else thi, tlo if tlo <= cor_ttc else cor_ttc,
                    tlo, thi, tsrc, "Primary TTC mean; bear at the top of the boundary range (stress-weighted), bull "
                    "at its bottom.", f"ch08_credit_history FY2010-{a}", cutoff=cut)
            rho = h["rho"]
            r0, rlo, rhi = float(rho.iloc[-1]), float(rho[rho.index >= 2015].min()), float(rho[rho.index >= 2015].max())
            for y in yrs:
                reg.add(val, bank, "RWA density", "rho", f"FY{y}", r0, min(rhi, r0 + 0.02), max(rlo, r0 - 0.02), rlo,
                        rhi, f"history FY2015-{a} (ch07_capital)", "Anchor-year density; bear +2 pp (Basel II/III "
                        "transition), bull -2 pp." + (" NMB: Bank-solo RWA over Group assets (D2)." if bank == "B02"
                                                      else ""), f"ch07_capital.rwa_density FY{a} = {r0:.4f}", cutoff=cut)
            # --- rate path and betas (open issue 11)
            tb = _tbill(root, int(vd[:4]))
            tb0 = float(tb.iloc[-1])
            for i, y in enumerate(yrs):
                proxy = int(tb.index.max()) == 2026  # D64 amended: the 2026 value is the 364-day auction proxy
                reg.add(val, bank, "T-bill path (BoT overall T-bill rate; rate driver)", "rate", f"FY{y}", tb0,
                        max(0.0, tb0 - 0.02 * min(1, (i + 1) / 2)), tb0 + 0.01 * min(1, (i + 1) / 2),
                        float(tb.min()), float(tb.max()), f"T-bill history {int(tb.index.min())}-{int(tb.index.max())}"
                        " (ch06_rates_used)", "Base flat at the latest annual T-bill; bear: BoT easing -200 bp over two "
                        "years; bull +100 bp." + (" 2026 = PROXY (accepted by the owner, D64 amended): the 364-day "
                        "auction yield at the Q3 2026 quarter-end auction stands in for the overall T-bill rate (Q1-Q2 "
                        "2026 not collected)." if proxy else ""), f"ch06_rates_used {int(tb.index.max())} = {tb0:.4f}; CBR at vd "
                        f"{b.macro['cbr']:.4f}", cutoff=cut)
            full = h["cost_dep"].to_frame("c").join(tb.rename("r"), how="inner")
            cons = full.index.to_series().diff() == 1
            ob = beta_ols(full["c"].diff()[cons].to_numpy(), full["r"].diff()[cons].to_numpy())
            if curr:
                blo, bhi, _, _ = _cal(cal, bank, "deposit beta (all deposits)")
                bsrc = "calibration_inputs.csv (deposit beta, all deposits)"
                blo, bhi = min(blo, ob["lo"]), max(bhi, ob["hi"])
            else:
                blo, bhi, bsrc = ob["lo"], ob["hi"], f"90% interval, OLS on annual changes, FY<= {a}"
            reg.add(val, bank, "Deposit beta (cost of deposits on T-bill)", "deposit_beta", "all", ob["beta"],
                    min(bhi, max(ob["beta"], 0.10)), max(blo, ob["beta"]), blo, bhi, bsrc,
                    "Point estimate; bear assumes deposit costs follow rates more (0.10 or the upper bound), bull the "
                    "point estimate.", f"ch06 method on cost_of_deposits and T-bill, FY<= {a}, n = {ob['n']}",
                    cutoff=cut)
            ay = h["iea_yield"].to_frame("y").join(tb.rename("r"), how="inner")
            cy = ay.index.to_series().diff() == 1
            oa = beta_ols(ay["y"].diff()[cy].to_numpy(), ay["r"].diff()[cy].to_numpy())
            reg.add(val, bank, "Asset beta (IEA yield on T-bill)", "asset_beta", "all", oa["beta"],
                    min(oa["hi"], oa["beta"] + 0.1) if np.isfinite(oa["hi"]) else oa["beta"],
                    max(oa["lo"], oa["beta"] - 0.1) if np.isfinite(oa["lo"]) else oa["beta"], oa["lo"], oa["hi"],
                    f"90% interval, OLS on annual changes, FY<= {a}",
                    "Repricing of earning-asset yields with the T-bill; with the deposit beta it ties NIM to the rate "
                    "path (open issue 11).", f"ch04_4_3_deployment.iea_yield, FY<= {a}, n = {oa['n']}", cutoff=cut)
            # --- balance-sheet constants
            ta, iea = fy["total_assets"], fy["interest_earning_assets"]
            theta = (ta - iea) / ta
            e_ex = fy["equity_attributable"] - (b.final_dps_anchor * fy["shares_outstanding"]
                                                if np.isfinite(b.final_dps_anchor) else 0)
            omega = (ta - fy["customer_deposits"] - fy["borrowings"] - fy["equity_attributable"] - fy["nci_equity"]) / ta
            reg.add(val, bank, "Other liabilities / assets", "omega", "all", omega, omega, omega, np.nan, np.nan, "",
                    "Anchor-year ratio held constant.", f"FY{a}: (TA - deposits - borrowings - equity) / TA",
                    "fixed", cutoff=cut)
            reg.add(val, bank, "Non-earning assets / assets", "theta", "all", theta, theta, theta, np.nan, np.nan, "",
                    "Anchor-year (TA - IEA) / TA, so loans + liquid assets = interest-earning assets.",
                    f"FY{a} interest_earning_assets", "fixed", cutoff=cut)
            lmin, _ = pick(e, bank, "bot_min_liquid_assets_ratio", "AS AT", period_end=vd)
            lmin = lmin if np.isfinite(lmin) else 0.20
            reg.add(val, bank, "Minimum liquid assets / deposits", "l_min", "all", lmin, lmin, lmin, np.nan, np.nan,
                    "", "BoT minimum liquid assets ratio applied to deposits (floor; banks hold more).",
                    "bot_min_liquid_assets_ratio (AS AT)", "fixed", cutoff=cut)
            sw = float((h["cost_borr"] - h["cost_dep"]).dropna().iloc[-3:].mean())
            reg.add(val, bank, "Wholesale funding spread over deposit cost", "s_W", "all", sw, sw + 0.01, max(0, sw - 0.01),
                    float((h["cost_borr"] - h["cost_dep"]).dropna().min()),
                    float((h["cost_borr"] - h["cost_dep"]).dropna().max()), f"history FY<= {a} (ch06)",
                    "Mean of the last three years' cost of borrowings - cost of deposits; bear +100 bp, bull -100 bp.",
                    "ch06_liquidity.cost_of_borrowings, ch06_deposits.cost_of_deposits", cutoff=cut)
            tx = h["tax"]
            t0, tlo2, thi2 = float(tx.iloc[-3:].mean()), float(tx[tx.index >= 2015].min()), float(tx[tx.index >= 2015].max())
            reg.add(val, bank, "Effective tax rate", "t", "all", t0, min(thi2, t0 + 0.02), max(tlo2, t0 - 0.02), tlo2,
                    thi2, f"history FY2015-{a}", "Three-year mean (statutory 30%); +/- 2 pp.",
                    "ch04_4_0_components.tax_rate", "fixed", cutoff=cut)
            nu = 1 - fy["profit_attributable"] / fy["net_profit"]
            reg.add(val, bank, "NCI share of profit", "nu", "all", nu, nu, nu, np.nan, np.nan, "",
                    "Anchor year; CRDB's NCI is the 45% minority in CRDB DRC (loss-making, so nu < 0).",
                    f"FY{a} profit_attributable / net_profit", "fixed", cutoff=cut)
            delta = 1 - fy["tier1_capital"] / fy["equity_attributable"]
            reg.add(val, bank, "Regulatory deductions / equity", "delta", "all", delta, delta, delta, np.nan, np.nan, "",
                    "1 - Tier 1 / attributable equity at the anchor" + (" (Bank-solo Tier 1 over Group equity, D2)"
                                                                       if bank == "B02" else ""),
                    f"FY{a} tier1_capital, equity_attributable", "fixed", cutoff=cut)
            # --- capital target (deferred-fix parameters, D55)
            q = _tab(root, "ch07_capital_requirements.csv")
            q = q[(q["bank_id"] == bank) & (q["fiscal_year"] == min(a, int(q["fiscal_year"].max())))].iloc[0]
            t2 = fy["total_capital_ratio"] - fy["tier1_ratio"]
            req = {"tier1": q["tier1_requirement"], "total": q["total_requirement"] - t2,
                   "binding": max(q["tier1_requirement"], q["total_requirement"] - t2)}[rules["constraint"]]
            vis = bool(rules["dsib_visible_at"].get(vd, {}).get(bank, False))
            sur = float(rules["dsib_surcharge_pp"]) if vis else 0.0
            sc = rules["dsib_surcharge_scenarios"]
            head = min(fy["tier1_ratio"] - q["tier1_requirement"], fy["total_capital_ratio"] - q["total_requirement"])
            cp = _tab(root, "ch07_capital.csv")
            hr = cp[(cp["bank_id"] == bank) & cp["fiscal_year"].between(2015, a)]["binding_headroom"].dropna()
            buf_mean = float(hr.mean())
            # base (owner, 09b pre-flight): min(mean FY2015-anchor headroom, anchor-year headroom); if the anchor-year
            # headroom is more than 1 pp below the mean, flag a STOP for the owner instead of capping
            buf = min(buf_mean, head)
            buf_stop = head < buf_mean - 0.01
            inputs[(val, bank, "buffer")] = {"mean": buf_mean, "anchor": head, "base": buf, "stop": buf_stop}
            bull_buf = float(np.clip(head / 2, 0.0, 0.02))      # bull: half the anchor-year headroom, at most 2 pp
            bear_add = max(sc) if vis else 0.0                  # bear: base + D-SIB scenario where visible (D1)
            reg.add(val, bank, "Target Tier 1 ratio (requirement + bank buffer + D-SIB)", "k_star", "all",
                    req + buf + sur, req + buf + bear_add + sur, req + bull_buf + sur, req,
                    req + max(buf, bull_buf) + max(sc), "requirement (ch07, D52 accepted) to + mean headroom + the "
                    "largest D-SIB scenario (config capital_rules)",
                    f"Requirement {req:.4f} on the '{rules['constraint']}' constraint (Tier 2 held at {t2:.4f} of RWA), "
                    f"reading {rules['crdb_requirement_reading']}, buffer from {rules['ccb_start']}. Bank buffer (D56): "
                    f"base = min(mean binding headroom FY2015-{a} ({buf_mean:.4f}, {len(hr)} years), FY{a} headroom "
                    f"({head:.4f})) = {buf:.4f}" + (" - STOP: anchor headroom > 1 pp below the mean, owner decision"
                                                     if buf_stop else "") + f"; bull = half the FY{a} "
                    f"headroom ({head:.4f}), at most 2 pp ({bull_buf:.4f}); bear = base + the largest D-SIB scenario "
                    f"where the designation is visible at vd ({vis}: +{bear_add:.2%}). Dividends stay within the payout "
                    "cap pi_max.", "ch07_capital.csv binding_headroom; ch07_capital_requirements.csv; config "
                    "capital_rules (D55)", "fixed", cutoff=cut)
            reg.add(val, bank, "Capital requirement (Tier 1 terms; nowcast-year check)", "k_req", "all", req + sur,
                    req + sur, req + sur, np.nan, np.nan, "", "Requirement on the configured constraint plus any visible "
                    "D-SIB surcharge; the nowcast-year capital check uses it, k* applies from year 2 (D56).",
                    "ch07_capital_requirements.csv; config capital_rules", "fixed", cutoff=cut)
            pp = h["payout"]
            plo2, phi2 = float(pp[pp.index >= 2010].min()), float(pp[pp.index >= 2010].max())
            reg.add(val, bank, "Payout cap", "pi_max", "all", float(np.clip(pp.iloc[-1], plo2, phi2)), plo2, phi2, plo2,
                    phi2, f"payout history FY2010-{a} (ch07_payout)", "Anchor-year payout as the policy cap; bear the "
                    "historical minimum, bull the maximum.", f"ch07_payout FY{a} = {pp.iloc[-1]:.4f}", "fixed",
                    cutoff=cut)
            # --- persistence-based long-run inputs (D8, D50): PIT re-estimation for HIST
            if curr:
                c5 = _tab(root, "ch05_calibration_candidates.csv")
                c5 = c5[c5["bank_id"] == bank].set_index("specification")
                ps = {"lam": c5.loc["ar1_break_episode", "lambda"], "lam_lo": c5.loc["ar1_break_episode", "lambda_lo"],
                      "lam_hi": c5.loc["ar1_break_episode", "lambda_hi"],
                      "er_brk": c5.loc["ar1_break_episode", "er_star"], "er_ttc": c5.loc["through_cycle_mean", "er_star"],
                      "er_ttc_lo": c5.loc["through_cycle_mean", "er_star_lo"],
                      "er_brk_hi": c5.loc["ar1_break_episode", "er_star_hi"], "years": c5.loc["ar1_break_episode", "years"]}
                elo, ehi, eilo, eihi = _cal(cal, bank, "long-run excess return ER*")
                esrc = "calibration_inputs.csv (ER*)"
            else:
                ps = _pit_persistence(root, bank, e, a)
                elo, ehi = min(0.0, ps["er_ttc"]), ps["er_brk"]
                esrc = f"re-estimated on ER FY{ps['years']} (eligible at {vd}): bear 0 to break-adjusted"
            inputs[(val, bank, "persistence")] = ps
            lp = {"base": 0.015, "bear": 0.03, "bull": 0.0}
            beta = b.macro["beta"]
            coe = {k: coe_mod.approach_a(b.macro["y_long"], b.macro["ds"], beta + {"base": 0, "bear": 0.1,
                                                                                  "bull": -0.1}[k],
                                         b.macro["erp"], b.macro["crp"], lp[k]) for k in lp}
            er = {"base": ps["er_ttc"], "bear": min(0.0, ps["er_ttc"]), "bull": ps["er_brk"]}  # D56 ordering rule
            reg.add(val, bank, "Long-run excess return ER* (D50)", "ER_star", "terminal", er["base"], er["bear"],
                    er["bull"], elo, ehi, esrc, "D50 (confirmed): base = through-the-cycle mean ER, bear = min(0, base ER*) "
                    "(D56 ordering rule, owner 2026-11-09), bull = break-adjusted ER*.", f"ER years {ps['years']}", "fixed", cutoff=cut)
            roe = {k: coe["base"] + er[k] for k in er}
            reg.add(val, bank, "Sustainable ROE (COE base + ER*)", "ROE_star", "terminal", roe["base"], roe["bear"],
                    roe["bull"], coe["base"] + elo, coe["base"] + ehi, "COE (approach A, base) + ER* range",
                    "ROE* = base COE + ER* case; never the current ROE carried forward (VALUATION_SPEC §4).",
                    f"COE A base {coe['base']:.4f}", "fixed", cutoff=cut)
            reg.add(val, bank, "Terminal growth (nominal TZS)", "g", "terminal", g_term["base"], g_term["bear"],
                    g_term["bull"], 0.03, 0.12, "IMF final-year Tanzanian inflation + 3% (D64 amended)"
                    if np.isfinite(pz) else "judgement (placeholder)", ("Owner rule (D64 amended): base = IMF WEO "
                    "final-year Tanzanian CPI inflation of the vintage public at the date (MA-0007 / MA-0029) + 3% "
                    "long-run real growth (`valuation.terminal_real_growth`); bear / bull = base -/+ 1 pp; guard "
                    "COE - g >= 4 pp in every scenario (`valuation.coe_g_min_gap`, the stage stops on violation).")
                    if np.isfinite(pz) else "PLACEHOLDER pending the inflation forecast.",
                    f"Tanzanian inflation forecast at vd {pz:.4f} + {real_lr:.2f}" if np.isfinite(pz)
                    else f"CPI at vd {b.macro['cpi_tz']:.3f}", "fixed", cutoff=cut)
            reg.add(val, bank, "Fade-start growth (nominal TZS)", "g_fade", "fade_start", g_fs, g_fs, g_fs, 0.03,
                    0.15, "IMF nominal GDP growth (MA-0010 / MA-0032, D64)", "Owner rule (D64 amended): the fade starts "
                    "from the IMF projection of Tanzanian nominal GDP growth, (1 + real) (1 + CPI) - 1, and converges "
                    "linearly to terminal g over the fade; the explicit loan path converges to it.",
                    f"nominal GDP growth at vd {g_fs:.4f}", "fixed", cutoff=cut)
            reg.add(val, bank, "Terminal payout", "payout_star", "terminal", 1 - g_term["base"] / roe["base"],
                    1 - g_term["bear"] / roe["bear"], 1 - g_term["bull"] / roe["bull"], 0.0, 1.0,
                    "derived: 1 - g / ROE*", "Set so terminal growth is internally funded (VALUATION_SPEC §4).",
                    "derived", "fixed", cutoff=cut)
            reg.add(val, bank, "Fade speed lambda (break-adjusted, D50)", "lambda", "fade", ps["lam"], ps["lam"],
                    ps["lam"], ps["lam_lo"], ps["lam_hi"], "Andrews 90% interval (" + ("calibration_inputs.csv" if curr
                                                                                       else "re-estimated, eligible ER")
                    + ")", "Break-adjusted lambda for all cases (D50); the through-the-cycle lambda is 1.00, so it "
                    "cannot serve as a fade speed (open issue 3).", f"ER years {ps['years']}", "fixed", cutoff=cut)
            reg.add(val, bank, "Illiquidity premium", "LP", "all", lp["base"], lp["bear"], lp["bull"], 0.0, 0.03,
                    "VALUATION_SPEC §5 / §11 (0-3%, central 1.5%)", "Spec range; bear 3%, bull 0%.",
                    "VALUATION_SPEC §11 open decisions (approved)", "triangular", "market", cut)
            reg.add(val, bank, "Bank beta (levered, EM banks)", "beta", "all", beta, beta + 0.1, beta - 0.1,
                    beta - 0.15, beta + 0.15, "Damodaran EM bank beta +/- 0.15", "Bottom-up EM bank beta at vd; "
                    "bear/bull +/- 0.1.", f"macro bank beta at vd {beta:.4f} (pub {b.sources.get('macro.beta')})",
                    "triangular", "market", cut)
            reg.add(val, bank, "Cost of equity, approach A (derived)", "COE_A", "all", coe["base"], coe["bear"],
                    coe["bull"], np.nan, np.nan, "", "Derived from the market inputs, beta and LP (VALUATION_SPEC §5); "
                    "approach B not computable until forward inflation is collected.",
                    f"y_long {b.macro['y_long']:.4f}, DS {b.macro['ds']:.4f}, ERP {b.macro['erp']:.4f}, CRP "
                    f"{b.macro['crp']:.4f}", "derived", "market", cut)
            if bank == "B01":
                lg = e[(e["bank_id"] == bank) & (e["canonical_name"].isin(["segment_pat", "dividends_upstreamed"]))
                       & (e["fiscal_year"] == a)]
                bur = lg[(lg["canonical_name"] == "segment_pat") & lg["entity"].str.contains("Burundi", na=False)]
                drc = lg[(lg["canonical_name"] == "segment_pat") & lg["entity"].str.contains("DRC", na=False)]
                up = lg[(lg["canonical_name"] == "dividends_upstreamed") & lg["entity"].str.contains("Burundi", na=False)]
                sp = float(bur["value"].sum()) + 0.55 * float(drc["value"].sum())
                s_sub = sp / fy["profit_attributable"]
                u = float(up["value"].sum()) / float(bur["value"].sum()) if float(bur["value"].sum()) else 0.0
                reg.add(val, bank, "Subsidiary share of attributable profit", "s_sub", "all", s_sub, s_sub, s_sub,
                        np.nan, np.nan, "", "Burundi (100%) + 55% of DRC segment profit over attributable profit.",
                        f"segment_pat FY{a}", "fixed", cutoff=cut)
                reg.add(val, bank, "Upstream ratio (subsidiary dividends / profit)", "u", "all", u, u * 0.5, u, 0.0,
                        1.0, "0-1", "Burundi dividends upstreamed over its profit; bear halves it (FX controls).",
                        f"dividends_upstreamed FY{a}", "fixed", cutoff=cut)
                pz, pb, pc = (b.macro.get(k, float("nan")) for k in ("infl_tz_fwd", "infl_bdi", "infl_cod"))
                wb, wd = float(bur["value"].sum()), 0.55 * float(drc["value"].sum())
                if all(np.isfinite([pz, pb, pc])) and wb + wd > 0:  # D64: relative PPP vs TZS, profit-weighted
                    fxb = (wb * (1 - (1 + pz) / (1 + pb)) + wd * (1 - (1 + pz) / (1 + pc))) / (wb + wd)
                    fx_note = (f"LOW CONFIDENCE (vintage swing: 19.5% a year at 31 Dec 2025 vs 6.1% at 31 Oct 2026, "
                               f"Burundi inflation forecast): relative PPP vs TZS from IMF inflation forecasts (TZ "
                               f"{pz:.3f}, BDI {pb:.3f}, COD {pc:.3f}), weighted by Burundi and 55% DRC profit")
                else:
                    fxb, fx_note = 0.0, "judgement (FX paths TO COLLECT)"
                reg.add(val, bank, "Subsidiary currency depreciation vs TZS (a year)", "fx", "all", fxb,
                        min(fxb + 0.10, 0.50), fxb, 0.0, 0.50, fx_note, "Base: subsidiary currencies lose value "
                        "against TZS at the inflation differential (D64, issue 16); bear: a further 10% a year "
                        "(VALUATION_SPEC §7 bear story); bull = base.", fx_note, "fixed", cutoff=cut)
            for p_, v_, note in (("RIM weight", vcfg["weights"]["rim"], "VALUATION_SPEC §2 (approved)"),
                                 ("DDM weight", vcfg["weights"]["ddm"], "VALUATION_SPEC §2 (approved)")):
                reg.add(val, bank, p_, p_.split()[0].lower() + "_w", "all", v_, v_, v_, np.nan, np.nan, "", note, note,
                        "fixed", "global", cut)
            inputs[(val, bank, "opening")] = {"e_ex": e_ex, "T": T, "years": yrs}
    return pd.DataFrame(reg.rows), inputs
