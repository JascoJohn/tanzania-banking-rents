"""Peer banks: extraction checks, market data, cost of equity and peer outputs (prompt 19 Parts A-C; D87).

Inputs (raw, MANIFEST-hashed): transcriptions of the peer annual reports in ``data/raw/manual_entries/peers/``
(``<TICKER>_financials.csv``, ``_structure.csv``; schema in that folder's README) and the collected market and macro
data (``peer_market_collected.csv``, ``peer_macro_collected.csv``). The PDFs themselves sit in ``data/raw/peers/``
(git-ignored, hashed).

Outputs:
- ``data/processed/peers/peer_financials.csv``: long (bank, fy, field, value in full currency units, unit as reported,
  currency, basis, source_file, page, column, check_status) and ``peer_financials_wide.csv`` (current-year values).
- ``outputs/tables/ch13_peers_checks.csv``: every check per bank-year (ROE, ROA, payout, equity bridge, BVPS x shares,
  restatements) with pass / fail / not computable.
- ``ch13_peers_valuation.csv``: per bank and date, P/B on reported book, P/E, dividend yield, market cap in USD, COE on
  the study's method (approach A with the register's common beta and LP), ROE - COE, justified P/B.
- ``ch13_peers_table.csv`` (C1), ``ch13_peers_premium.csv`` (C4), ``ch13_peers_paths.csv`` (C3) and the charts
  ``ch13_peers_scatter.png`` (C2) and ``ch13_peers_paths.png`` (C3); manifest ``ch13_peers.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import write_csv, write_json, write_text

ENTRIES = Path("data/raw/manual_entries/peers")
SCALE = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
MONEY = ["total_assets", "gross_loans", "net_loans", "customer_deposits", "equity_attributable", "net_interest_income",
         "fee_commission_income", "total_operating_income", "impairment_charge", "operating_expenses",
         "profit_before_tax", "profit_attributable", "equity_opening", "dividends_paid", "oci_attributable",
         "other_equity_movements", "oneoff_fx_revaluation_gain", "oneoff_govt_bond_impairment", "oneoff_other",
         "hyperinflation_adjustment"]
DATES = {"HIST": "2025-12-31", "CURR": "2026-10-30"}


def _cfg(root: Path) -> dict:
    return settings(root)["peers"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


# --- A. financials -----------------------------------------------------------------------------------------------------
def load_financials(root: Path) -> pd.DataFrame:
    """All transcribed peer rows, values converted to full currency units (money) or kept (ratios, counts)."""
    frames = []
    for f in sorted((root / ENTRIES).glob("*_financials.csv")):
        d = pd.read_csv(f, dtype=str)
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["bank", "fy", "field", "value"])
    d = pd.concat(frames, ignore_index=True)
    d["fy"] = pd.to_numeric(d["fy"], errors="coerce").astype("Int64")
    d["value_reported"] = _num(d["value"])
    unit = d["unit"].str.strip().str.lower()
    d["value"] = d["value_reported"] * unit.map(SCALE).fillna(1.0)
    d["column"] = d.get("column", pd.Series("current", index=d.index)).fillna("current")
    return d.dropna(subset=["fy", "value"])


BANK_ONLY = {"cet1_ratio", "total_capital_ratio", "min_cet1_ratio", "min_total_capital_ratio", "npl_ratio",
             "npl_coverage"}


def wide(fin: pd.DataFrame) -> pd.DataFrame:
    """Current-year values, gaps filled from the next report's comparative column; ratios reported for the bank only
    (not the group) carry a ``_bank`` suffix so they never pass as group figures."""
    f = fin.copy()
    if "basis" in f:
        f.loc[f["field"].isin(BANK_ONLY) & f["basis"].fillna("").str.lower().eq("bank"), "field"] += "_bank"
    cur = f[f["column"] == "current"].pivot_table(index=["bank", "fy"], columns="field", values="value", aggfunc="first")
    comp = f[f["column"] == "comparative"].pivot_table(index=["bank", "fy"], columns="field", values="value",
                                                      aggfunc="last")
    return cur.combine_first(comp)


def _chk(rows, bank, fy, check, value, expected, tol, note=""):
    if pd.isna(value) or pd.isna(expected) or expected == 0:
        st = "not computable"
        dev = np.nan
    else:
        dev = value / expected - 1
        st = "pass" if abs(dev) <= tol else "fail"
    rows.append({"bank": bank, "fy": int(fy), "check": check, "computed": value, "reference": expected,
                 "deviation": dev, "tolerance": tol, "status": st, "note": note, "evidence_class": "Derived"})


def checks(fin: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Blueprint data layer: recomputed ratios and identities per bank-year (prompt 19 Part A rules)."""
    c = _cfg(root)
    tol, rtol = float(c["check_tolerance"]), float(c["restatement_tolerance"])
    w = wide(fin)
    rows = []
    for (b, fy), r in w.iterrows():
        prev = w.loc[(b, fy - 1)] if (b, fy - 1) in w.index else pd.Series(dtype=float)
        e0 = prev.get("equity_attributable", r.get("equity_opening", np.nan))
        avg_e = np.nanmean([e0, r.get("equity_attributable", np.nan)]) if pd.notna(e0) else np.nan
        a0 = prev.get("total_assets", np.nan)
        avg_a = np.nanmean([a0, r.get("total_assets", np.nan)]) if pd.notna(a0) else np.nan
        npa = r.get("profit_attributable", np.nan)
        rows.append({"bank": b, "fy": int(fy), "check": "ROE (profit / average equity)",
                     "computed": npa / avg_e if avg_e else np.nan, "reference": np.nan, "deviation": np.nan,
                     "tolerance": np.nan, "status": "computed" if pd.notna(avg_e) else "not computable",
                     "note": "opening equity from the prior year's report, else the equity bridge",
                     "evidence_class": "Derived"})
        rows.append({"bank": b, "fy": int(fy), "check": "ROA (profit / average assets)",
                     "computed": npa / avg_a if avg_a else np.nan, "reference": np.nan, "deviation": np.nan,
                     "tolerance": np.nan, "status": "computed" if pd.notna(avg_a) else "not computable",
                     "note": "", "evidence_class": "Derived"})
        sh = r.get("shares_outstanding_ye", np.nan)
        rows.append({"bank": b, "fy": int(fy), "check": "payout (DPS x shares / profit)",
                     "computed": r.get("dps_declared", np.nan) * sh / npa if npa else np.nan, "reference": np.nan,
                     "deviation": np.nan, "tolerance": np.nan,
                     "status": "computed" if pd.notna(r.get("dps_declared", np.nan) * sh) else "not computable",
                     "note": "", "evidence_class": "Derived"})
        bridge = (r.get("equity_opening", np.nan) + npa - r.get("dividends_paid", np.nan)
                  + r.get("oci_attributable", 0.0) + r.get("other_equity_movements", 0.0))
        _chk(rows, b, fy, "equity bridge (opening + profit - dividends + OCI + other = closing)", bridge,
             r.get("equity_attributable", np.nan), tol)
        _chk(rows, b, fy, "BVPS x shares = equity", r.get("bvps_reported", np.nan) * sh,
             r.get("equity_attributable", np.nan), tol, "only where the report prints BVPS")
        _chk(rows, b, fy, "EPS x weighted shares = profit", r.get("eps_basic", np.nan) * r.get("shares_weighted", np.nan),
             npa, tol)
    comp = fin[fin["column"] == "comparative"]
    cur = fin[fin["column"] == "current"].groupby(["bank", "fy", "field"])["value"].first()
    for r in comp.drop_duplicates(["bank", "fy", "field"], keep="last").itertuples():
        k = (r.bank, r.fy, r.field)
        if k in cur.index:
            _chk(rows, r.bank, r.fy, f"restatement: {r.field} (comparative in the next report vs own report)",
                 r.value, float(cur.loc[k]), rtol, str(getattr(r, "note", "") or ""))
    return pd.DataFrame(rows)


def processed(fin: pd.DataFrame, chk: pd.DataFrame, root: Path) -> list[Path]:
    out = path("processed", root) / "peers"
    out.mkdir(parents=True, exist_ok=True)
    fails = chk[chk["status"] == "fail"].groupby(["bank", "fy"])["check"].apply(lambda s: "; ".join(s))
    long = fin.assign(check_status=[("fail: " + fails[(b, y)]) if (b, y) in fails.index else "ok"
                                    for b, y in zip(fin["bank"], fin["fy"], strict=True)])
    cols = ["bank", "fy", "field", "value", "value_reported", "unit", "currency", "basis", "source_file", "page",
            "column", "note", "check_status"]
    long = long[[c for c in cols if c in long.columns]].assign(evidence_class="Observed")
    wd = wide(fin).reset_index().assign(evidence_class="Observed")
    return [write_csv(long, out / "peer_financials.csv", require_evidence=True),
            write_csv(wd, out / "peer_financials_wide.csv", require_evidence=True)]


# --- B. market data and COE ---------------------------------------------------------------------------------------------
def _market(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = root / ENTRIES / "peer_market_collected.csv"
    x = root / ENTRIES / "peer_macro_collected.csv"
    mk = pd.read_csv(m, dtype=str) if m.exists() else pd.DataFrame(columns=["bank", "date", "field", "value"])
    mc = pd.read_csv(x, dtype=str) if x.exists() else pd.DataFrame(columns=["country", "date", "field", "value"])
    for d in (mk, mc):
        d["value"] = _num(d["value"])
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return mk, mc


def _at(df: pd.DataFrame, key: str, who: str, field: str, date: pd.Timestamp, back_days: int = 45,
        ahead_days: int = 0) -> tuple[float, str]:
    """Latest value dated within ``back_days`` before the date; ``ahead_days`` admits a month-end label falling on a
    non-trading day just after it (31 Oct 2026 is a Saturday)."""
    x = df[(df[key] == who) & (df["field"] == field) & (df["date"] <= date + pd.Timedelta(days=ahead_days))
           & (df["date"] >= date - pd.Timedelta(days=back_days))].sort_values("date")
    return (float(x["value"].iloc[-1]), x["date"].iloc[-1].date().isoformat()) if len(x) else (np.nan, "")


def valuation(fin: pd.DataFrame, chk: pd.DataFrame, ctx) -> pd.DataFrame:
    """Per bank and date (31 Dec 2025, 30 Oct 2026): multiples on reported book and the study's COE (prompt 19 B)."""
    root = ctx.root
    c = _cfg(root)
    mk, mc = _market(root)
    w = wide(fin)
    roe = chk[chk["check"].str.startswith("ROE")].set_index(["bank", "fy"])["computed"]
    reg = ctx.reg
    f = path("tables", root) / "ch13_peer_coe.csv"
    prior = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["country", "valuation"])
    rows = []
    for val, ds in DATES.items():
        d = pd.Timestamp(ds)
        beta = float(reg[(reg["valuation"] == val) & (reg["symbol"] == "beta")]["base"].iloc[0])
        lp = float(reg[(reg["valuation"] == val) & (reg["symbol"] == "LP")]["base"].iloc[0])
        for tick, (name, country, ccy, exch) in c["banks"].items():
            fy = 2025 if (tick, 2025) in w.index else (max([y for b, y in w.index if b == tick]) if tick in
                                                      w.index.get_level_values(0) else np.nan)
            r = w.loc[(tick, fy)] if pd.notna(fy) else pd.Series(dtype=float)
            price, pdate = _at(mk, "bank", tick, "close_price", d)
            sh = r.get("shares_outstanding_ye", np.nan)
            sh_m, _ = _at(mk, "bank", tick, "shares_outstanding", d, 400)
            sh = sh if pd.notna(sh) else sh_m
            eq = r.get("equity_attributable", np.nan)
            y10, ydate = _at(mc, "country", country, "bond_10y", d, 120, 2)
            fx, _ = _at(mc, "country", country, "usd_fx", d, 10, 2)
            crp, _ = _at(mc, "country", country, "damodaran_crp", d, 800)
            erp, _ = _at(mc, "country", country, "damodaran_erp_mature", d, 800)
            ds_, _ = _at(mc, "country", country, "damodaran_default_spread", d, 800)
            # fall back on the pack's country inputs at the same date and vintage (ch13_peer_coe, D67)
            pc = prior[(prior["country"] == country) & (prior["valuation"] == val)]
            if len(pc):
                pc = pc.iloc[0]
                if pd.isna(y10) and pd.notna(pc["bond_10y"]):
                    y10, ydate = float(pc["bond_10y"]), f"{pc['date']} (ch13_peer_coe)"
                crp = float(pc["crp"]) if pd.isna(crp) else crp
                erp = float(pc["erp_mature"]) if pd.isna(erp) else erp
                ds_ = float(pc["default_spread"]) if pd.isna(ds_) else ds_
            coe = (y10 - ds_) + beta * (erp + crp) + lp
            infl, _ = _at(mc, "country", country, "imf_inflation_final_year", d, 400, 2)
            vc = settings(root)["valuation"]
            if c.get("terminal_g_rule") == "imf_inflation_plus_real" and pd.notna(infl):
                g, g_basis = infl + float(vc["terminal_real_growth"]), "IMF final-year inflation + real growth"
            else:
                g, g_basis = float(c["terminal_g"][country]), "fallback (peers.terminal_g)"
            guard_ok = coe - g >= float(vc.get("coe_g_min_gap", 0.04)) - 1e-12
            rr = float(roe.get((tick, fy), np.nan)) if pd.notna(fy) else np.nan
            mcap = price * sh
            rows.append({"valuation": val, "date": ds, "bank": tick, "name": name, "country": country, "exchange": exch,
                         "currency": ccy, "fy_basis": fy, "price": price, "price_date": pdate, "shares": sh,
                         "market_cap": mcap, "usd_fx": fx, "market_cap_usd": mcap / fx if fx else np.nan,
                         "equity_attributable": eq, "pb": mcap / eq if eq else np.nan,
                         "pe": price / r.get("eps_basic", np.nan) if r.get("eps_basic", np.nan) else np.nan,
                         "dividend_yield": r.get("dps_declared", np.nan) / price if price else np.nan,
                         "roe": rr, "bond_10y": y10, "bond_date": ydate, "default_spread": ds_, "crp": crp,
                         "erp_mature": erp, "beta": beta, "lp": lp, "coe": coe, "roe_minus_coe": rr - coe, "g": g,
                         "imf_inflation": infl, "g_basis": g_basis,
                         "justified_pb": (rr - g) / (coe - g) if guard_ok else np.nan,
                         "guard": "ok" if guard_ok else "COE - g below the 4-point guard: justified P/B not computed",
                         "flagged_year": fy in c["flagged_years"].get(tick, []),
                         "status": "computed" if pd.notna(price) and pd.notna(coe) else
                         "not computable: " + ", ".join(k for k, v in {"price": price, "10-year yield": y10,
                                                                         "country premium": crp}.items() if pd.isna(v)),
                         "evidence_class": "Estimated"})
    v = pd.DataFrame(rows)
    v["pb_minus_justified"] = v["pb"] - v["justified_pb"]
    return v


def oneoff_roe(fin: pd.DataFrame, chk: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
    """ROE with and without one-offs (pre-tax amounts as printed). ``policy`` (settings peers.oneoff_policy, owner
    decision on issue 73) lists, per bank, the one-off fields and years removed for ``roe_ex_oneoffs``; every
    transcribed one-off is also shown in ``roe_ex_all_oneoffs``."""
    policy = policy or {}
    w = wide(fin)
    roe = chk[chk["check"].str.startswith("ROE")].set_index(["bank", "fy"])["computed"]
    rows = []
    for (b, fy), r in w.iterrows():
        v = {k: float(np.nan_to_num(r.get(k, 0.0))) for k in ("oneoff_fx_revaluation_gain",
                                                               "oneoff_govt_bond_impairment", "oneoff_other")}
        sign = {"oneoff_fx_revaluation_gain": 1, "oneoff_govt_bond_impairment": -1, "oneoff_other": 1}
        adj_all = sum(sign[k] * v[k] for k in v)
        pol = policy.get(b, {})
        adj = sum(sign[k] * v[k] for k in v if k in pol.get("fields", []) and int(fy) in pol.get("years", []))
        npa = r.get("profit_attributable", np.nan)
        rr = roe.get((b, fy), np.nan)
        avg_e = npa / rr if rr else np.nan
        rows.append({"bank": b, "fy": int(fy), "roe": rr, "oneoffs_pretax": adj,
                     "roe_ex_oneoffs": (npa - adj) / avg_e if avg_e else np.nan,
                     "oneoffs_all_pretax": adj_all, "roe_ex_all_oneoffs": (npa - adj_all) / avg_e if avg_e else np.nan,
                     "policy": "; ".join(f"{k} {pol.get('years')}" for k in pol.get("fields", [])) or "none removed",
                     "basis": "one-offs as printed, pre-tax (profit less FX revaluation gains plus government-bond "
                              "impairment, less other one-offs)", "evidence_class": "Derived"})
    return pd.DataFrame(rows)


# --- C. outputs ----------------------------------------------------------------------------------------------------------
def peer_table(v: pd.DataFrame, fin: pd.DataFrame, root: Path, val: str = "HIST") -> pd.DataFrame:
    w = wide(fin)
    x = v[v["valuation"] == val].copy()
    def pick(b, fy, f):
        if (b, fy) not in w.index:
            return np.nan, ""
        for col, basis in ((f, "group"), (f"{f}_bank", "bank only")):
            if col in w and pd.notna(w.loc[(b, fy), col]):
                return float(w.loc[(b, fy), col]), basis
        return np.nan, ""
    cap = [pick(b, fy, "cet1_ratio") for b, fy in zip(x["bank"], x["fy_basis"], strict=True)]
    npl = [pick(b, fy, "npl_ratio") for b, fy in zip(x["bank"], x["fy_basis"], strict=True)]
    x["capital_ratio"], x["capital_basis"] = [c[0] for c in cap], [c[1] for c in cap]
    x["npl_ratio"], x["npl_basis"] = [c[0] for c in npl], [c[1] for c in npl]
    tz = pd.read_csv(path("tables", root) / "ch13_peer_market.csv")
    tz = tz[(tz["valuation"] == val) & (tz["country"] == "Tanzania")]
    k = pd.read_csv(path("tables", root) / "ch13_kpi_table.csv")
    g_tz = 0.07 if val == "CURR" else 0.065
    for r in tz.itertuples():
        kk = k[k["bank"] == r.bank]
        x = pd.concat([x, pd.DataFrame([{"valuation": val, "date": DATES[val], "bank": r.bank, "name": r.bank,
                                         "country": "Tanzania", "pb": r.pb, "pe": r.pe, "roe": r.roe,
                                         "coe": r.coe_approach_a, "roe_minus_coe": r.roe - r.coe_approach_a,
                                         "g": g_tz, "justified_pb": (r.roe - g_tz) / (r.coe_approach_a - g_tz),
                                         "capital_ratio": float(kk["tier1_ratio"].iloc[0]) if len(kk) else np.nan,
                                         "npl_ratio": float(kk["npl_ratio"].iloc[0]) if len(kk) else np.nan,
                                         "flagged_year": False, "evidence_class": "Estimated"}])],
                      ignore_index=True)
    x["pb_minus_justified"] = x["pb"] - x["justified_pb"]
    cols = ["pb", "pe", "roe", "coe", "roe_minus_coe", "justified_pb", "pb_minus_justified", "capital_ratio",
            "npl_ratio"]
    peers = x[~x["country"].eq("Tanzania")]
    med = []
    for lab, s in (("peer median", peers), ("peer median ex GTCO", peers[peers["bank"] != "GTCO"])):
        med.append({"valuation": val, "bank": lab, "country": "", **{c: float(s[c].median()) for c in cols},
                    "evidence_class": "Derived"})
    return pd.concat([x, pd.DataFrame(med)], ignore_index=True)[
        ["valuation", "bank", "country", *cols, "g", "capital_basis", "npl_basis", "flagged_year", "evidence_class"]]


def premium(t: pd.DataFrame) -> pd.DataFrame:
    x = t[~t["bank"].str.startswith("peer median")].copy()
    x["pb_over_justified"] = x["pb"] / x["justified_pb"]
    peers = x[x["country"] != "Tanzania"]
    mx = peers["pb_minus_justified"].max()
    x["larger_than_every_peer"] = np.where(x["country"] == "Tanzania", x["pb_minus_justified"] > mx, np.nan)
    x["max_peer_premium"] = mx
    return x[["valuation", "bank", "country", "pb", "justified_pb", "pb_minus_justified", "pb_over_justified",
              "max_peer_premium", "larger_than_every_peer", "flagged_year"]].assign(evidence_class="Derived")


def paths(v: pd.DataFrame, fin: pd.DataFrame, chk: pd.DataFrame, root: Path) -> pd.DataFrame:
    """C3: five-year ROE and year-end P/B per bank (P/B where a year-end price is held)."""
    mk, _ = _market(root)
    w = wide(fin)
    roe = chk[chk["check"].str.startswith("ROE")].set_index(["bank", "fy"])["computed"]
    rows = []
    for (b, fy), r in w.iterrows():
        price, _ = _at(mk, "bank", b, "close_price", pd.Timestamp(int(fy), 12, 31))
        eq, sh = r.get("equity_attributable", np.nan), r.get("shares_outstanding_ye", np.nan)
        rows.append({"bank": b, "fy": int(fy), "roe": roe.get((b, fy), np.nan),
                     "pb": price * sh / eq if eq else np.nan, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def scatter(t: pd.DataFrame, figd: Path) -> Path:
    fig, ax = plt_subplots(1, 1, figsize=(7.5, 4.6))
    x = t[~t["bank"].str.startswith("peer median")].dropna(subset=["pb", "roe_minus_coe"])
    for r in x.itertuples():
        tz = r.country == "Tanzania"
        col = COLORS["B02" if r.bank == "NMB" else "B01"] if tz else "#8c8c86"
        ax.scatter(r.roe_minus_coe * 100, r.pb, color=col, s=36 if tz else 24, zorder=3,
                   marker="D" if tz else ("x" if r.flagged_year else "o"))
        ax.annotate(r.bank, (r.roe_minus_coe * 100, r.pb), fontsize=7, xytext=(4, 3), textcoords="offset points")
    gap = float((x["coe"] - x["g"]).median()) if len(x) else 0.08
    lo, hi = (x["roe_minus_coe"].min() * 100, x["roe_minus_coe"].max() * 100) if len(x) else (-5.0, 15.0)
    xs = np.linspace(min(-5, lo - 1), hi + 2, 50)
    ax.plot(xs, 1 + xs / 100 / gap, color="#1f1f1d", ls="--", lw=0.9,
            label=f"justified P/B on current ROE = 1 + (ROE − COE) / (COE − g), median COE − g = {gap:.1%}")
    ax.axhline(1, color="#bdbdb8", lw=0.6)
    style(ax, "P/B against ROE less cost of equity", "P/B (times)")
    ax.set_xlabel("ROE − COE, percentage points (x: flagged currency-event year)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    return save(fig, figd / "ch13_peers_scatter.png", "Source: peer annual reports (transcribed, page-referenced); "
                "exchange prices; COE on the study's approach A. Evidence class: Estimated.")


def paths_figure(p: pd.DataFrame, figd: Path) -> Path:
    banks = sorted(p["bank"].unique())
    fig, axs = plt_subplots(2, max(1, len(banks)), figsize=(1.7 * max(1, len(banks)), 4.2), sharex=True)
    axs = np.atleast_2d(axs)
    for i, b in enumerate(banks):
        g = p[p["bank"] == b].sort_values("fy")
        axs[0, i].plot(g["fy"], g["roe"] * 100, color="#06452E", marker="o", ms=2)
        axs[1, i].plot(g["fy"], g["pb"], color="#E6B323", marker="o", ms=2)
        axs[0, i].set_title(b, fontsize=8)
        for a in axs[:, i]:
            a.tick_params(labelsize=6)
    axs[0, 0].set_ylabel("ROE, %", fontsize=7)
    axs[1, 0].set_ylabel("P/B", fontsize=7)
    return save(fig, figd / "ch13_peers_paths.png", "Source: peer annual reports; year-end prices. Evidence class: "
                "Derived.")


NOTE = """# Peer banks (prompt 19, D87)

Generated by `tzbank.chapters.ch13_peers` (stage `market_ext`).

## Research question
Do NMB and CRDB trade at a premium to what their returns justify, measured against African peers on the same method?

## Variable definitions
P/B on reported book value (price x year-end shares / equity attributable); ROE = profit attributable / average
equity; COE = (10-year local yield - default spread) + beta (ERP + CRP) + LP with the register's common beta and LP;
justified P/B = (ROE - g) / (COE - g), g per country in `config/settings.yaml` (`peers.terminal_g`).

## Sources
Peer annual reports (data/raw/peers, transcriptions in data/raw/manual_entries/peers with page references); exchange
prices and macro inputs (peer_market_collected.csv, peer_macro_collected.csv); ch13_peer_market and ch13_kpi_table
for NMB and CRDB.

## Validation test
`tests/test_peers.py`.

## Results
{results}

## Evidence class
Observed (transcribed statements), Derived (ratios, checks), Estimated (COE, justified P/B).

## Limitations
- Book values and ROE are transcribed from published reports; check failures are listed in `ch13_peers_checks`.
- GTCO (naira devaluation 2023-24) and GCB (Ghana DDEP 2022-23) years are flagged; ROE is also shown ex one-offs.
- One growth rate per country is an assumption (open issue 69). BK Group's figures come from results press releases.
"""


def annex_table(v: pd.DataFrame, tbl: pd.DataFrame, fin: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Peer annex of the bank reports (prompt 23): NMB, CRDB and the African peers at 31 Oct 2026 on the prompt 19
    figures. P/B on reported book (``ch13_peers_valuation``); NPL from ``ch13_peers_table``; cost-to-income =
    operating expenses / total operating income of the FY basis (peers' annual reports); NMB and CRDB ratios and
    dividend yield from the chapter 13 KPI table, FY2025 (Blueprint ch. 13, peer comparison)."""
    w = wide(fin)
    t = tbl[tbl["valuation"] == "CURR"].set_index("bank")
    k = pd.read_csv(path("tables", root) / "ch13_kpi_table.csv")
    k = k[k["fiscal_year"] == 2025].set_index("bank")
    rows = []
    for b in ("NMB", "CRDB"):
        rows.append({"bank": b, "country": "Tanzania", "fy_basis": "FY2025", "roe": t.loc[b, "roe"],
                     "pb": t.loc[b, "pb"], "pe": t.loc[b, "pe"], "dividend_yield": k.loc[b, "dividend_yield"],
                     "npl_ratio": k.loc[b, "npl_ratio"], "cost_to_income": k.loc[b, "cost_to_income"],
                     "cir_basis": "chapter 13 KPI table", "pb_note": "",
                     "pb_basis": "price 31 Oct 2026 / FY2025 book", "evidence_class": "Estimated"})
    comp = {}  # cost lines from the income statement where no opex subtotal is printed (prompt 24: KCB FY2025)
    for f in (path("raw", root) / "manual_entries" / "peers").glob("*_opex_components_FY*.csv"):
        c = pd.read_csv(f)
        for (b, fy), g in c.groupby(["bank", "fy"]):
            x = g.set_index("item")["value"]
            comp[(b, int(fy))] = (float(x.drop("total_income").sum()) / float(x["total_income"]),
                                  "staff + depreciation + other operating expenses over total income, from the "
                                  f"FY{int(fy)} income statement (no printed subtotal)")
    for r in v[v["valuation"] == "CURR"].sort_values(["country", "name"]).itertuples():
        fy = int(r.fy_basis)
        cir, cir_basis = np.nan, ""
        if (r.bank, fy) in w.index and {"operating_expenses", "total_operating_income"} <= set(w.columns):
            ox, ti = w.loc[(r.bank, fy), "operating_expenses"], w.loc[(r.bank, fy), "total_operating_income"]
            cir = abs(float(ox)) / float(ti) if pd.notna(ox) and pd.notna(ti) and float(ti) else np.nan
            cir_basis = "operating expenses / total operating income" if pd.notna(cir) else ""
        if pd.isna(cir) and (r.bank, fy) in comp:
            cir, cir_basis = comp[(r.bank, fy)]
        rows.append({"bank": r.name, "country": r.country, "fy_basis": f"FY{fy}", "roe": r.roe, "pb": r.pb,
                     "pe": r.pe, "dividend_yield": r.dividend_yield,
                     "npl_ratio": t.loc[r.bank, "npl_ratio"] if r.bank in t.index else np.nan,
                     "cost_to_income": cir, "cir_basis": cir_basis or "not reported on a comparable basis",
                     "pb_note": "" if pd.notna(r.pb) else "no traded price at 31 Oct 2026",
                     "pb_basis": "price 30 Oct 2026 / reported book (peer annual report)",
                     "evidence_class": "Estimated"})
    return pd.DataFrame(rows)


def write_peers(ctx) -> list[Path]:
    root = ctx.root
    fin = load_financials(root)
    if fin.empty:
        return []
    tab, figd = path("tables", root), path("figures", root)
    chk = checks(fin, root)
    files = processed(fin, chk, root)
    v = valuation(fin, chk, ctx)
    t_hist, t_curr = peer_table(v, fin, root, "HIST"), peer_table(v, fin, root, "CURR")
    tbl = pd.concat([t_hist, t_curr], ignore_index=True)
    prem = pd.concat([premium(t_hist), premium(t_curr)], ignore_index=True)
    pth = paths(v, fin, chk, root)
    one = oneoff_roe(fin, chk, _cfg(root).get("oneoff_policy"))
    nf = path("notes", root) / "ch13_peers.md"
    tables = [(chk, "ch13_peers_checks.csv", "A. Extraction checks per bank-year", "Derived"),
              (one, "ch13_peers_roe_oneoffs.csv", "A. ROE with and without one-offs", "Derived"),
              (v, "ch13_peers_valuation.csv", "B. Multiples, COE and justified P/B per bank and date", "Estimated"),
              (tbl, "ch13_peers_table.csv", "C1. Peer table with NMB and CRDB; medians with and without GTCO",
               "Estimated"),
              (prem, "ch13_peers_premium.csv", "C4. Premium to justified P/B", "Derived"),
              (annex_table(v, tbl, fin, root), "ch13_peers_annex.csv", "Peer annex of the bank reports (prompt 23)",
               "Estimated"),
              (pth, "ch13_peers_paths.csv", "C3. ROE and P/B paths", "Derived")]
    man_t = []
    for df, name, desc, ev in tables:
        f = write_csv(ctx.stamp(df), tab / name, require_evidence=True)
        files.append(f)
        man_t.append({"path": f.relative_to(root).as_posix(), "description": desc, "evidence_class": ev,
                      "note": nf.relative_to(root).as_posix()})
    man_f = []
    for p, data, desc, ev in ((scatter(tbl[tbl["valuation"] == "CURR"], figd), tbl, "C2. P/B against ROE - COE",
                               "Estimated"), (paths_figure(pth, figd), pth, "C3. ROE and P/B paths", "Derived")):
        dd = write_csv(data, p.with_name(f"{p.stem}_data.csv"), require_evidence=True)
        files += [p, dd]
        man_f.append({"path": p.relative_to(root).as_posix(), "data": dd.relative_to(root).as_posix(),
                      "description": desc, "evidence_class": ev, "note": nf.relative_to(root).as_posix()})
    fails = chk[chk["status"] == "fail"]
    res = [f"- Checks: {len(chk)} run, {len(fails)} failed (listed in ch13_peers_checks)."]
    for r in prem[prem["country"] == "Tanzania"].itertuples():
        res.append(f"- {r.valuation} {r.bank}: P/B {r.pb:.2f} vs justified {r.justified_pb:.2f}; larger premium than "
                   f"every peer: {r.larger_than_every_peer}")
    files.append(write_text(NOTE.format(results="\n".join(res)), nf))
    m = {"chapter": "13_peers", "title": "Peer banks (prompt 19)", "status": "complete", "hypotheses": [],
         "tables": man_t, "figures": man_f, "notes": [nf.relative_to(root).as_posix()],
         "inputs": [str(ENTRIES)], "module": "tzbank.chapters.ch13_peers", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_peers.json"))
    return files
