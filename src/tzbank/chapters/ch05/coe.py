"""Annual cost of equity, approach A (blueprint §2 "Historical cost of equity"; VALUATION_SPEC §5).

COE_t = (y_long,t - DS_TZ) + beta_L (ERP_mature + CRP_TZ) + LP

- y_long: D34 primary long-bond yield (`coe_inputs_annual.csv`, series `d34_primary`); FY2008-12 the D41 10-year
  fallback (flagged) plus the D41 term-spread-adjusted alternative as a second column.
- DS_TZ, CRP_TZ, ERP_mature, beta_L: for FY2008-24 the Damodaran version public at each year-end (archived January
  release, `data/interim/damodaran_archive.csv`, D48); where no archived version covers the year (Tanzania is absent
  before the January 2016 release; EM bank betas before January 2012) the nearest archived version is used and
  flagged. FY2025 and the current valuation date use the versions held in the inputs workbook (Macroeconomic Data).
  The previous constant-vintage series (2025 vintage for every year) is kept as `coe_2025_vintage` for comparison.
- LP: the illiquidity premium from the assumption register (D3: 1.5% central).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tzbank.config import path, settings

DAMODARAN = {"ds": "Tanzania Sovereign Default Spread", "crp": "Tanzania Country Risk Premium",
             "erp": "Mature-Market Implied ERP", "beta": "Bank Beta, Emerging Markets (levered)"}


def damodaran_versions(root: Path) -> dict[str, dict[str, tuple[float, str]]]:
    """{'HIST'|'CURR': {input: (value, 'Row ID / Doc ID')}} from the inputs workbook Macroeconomic Data sheet."""
    m = pd.read_csv(path("interim", root) / "inputs_macroeconomic_data.csv")
    dates = settings(root)["valuation_dates"]
    out: dict[str, dict[str, tuple[float, str]]] = {}
    for label, d in (("HIST", dates["historical"]), ("CURR", dates["current"])):
        x = m[m["As At (Valuation Date)"].astype(str).str[:10] == str(d)]
        out[label] = {}
        for k, var in DAMODARAN.items():
            r = x[x["Variable"] == var]
            if len(r) and pd.notna(r["value_numeric"].iloc[0]):
                out[label][k] = (float(r["value_numeric"].iloc[0]), f"{r['Row ID'].iloc[0]} / {r['Doc ID'].iloc[0]}")
    return out


def archive(root: Path) -> pd.DataFrame:
    f = path("interim", root) / "damodaran_archive.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["input", "fiscal_year", "value", "version"])


def archived_input(a: pd.DataFrame, k: str, year: int) -> tuple[float, str, bool]:
    """(value, source, version-matched?) for input ``k`` in fiscal year ``year``: the version public at year-end,
    else the nearest archived version (flagged)."""
    s = a[a["input"] == k]
    if s.empty:
        return float("nan"), "not held", False
    hit = s[s["fiscal_year"] == year]
    if len(hit):
        r = hit.iloc[0]
        return float(r["value"]), f"{r['version']} ({r['file']})", True
    r = s.iloc[(s["fiscal_year"] - year).abs().argsort().iloc[0]]
    return float(r["value"]), f"{r['version']} ({r['file']}; nearest archived version)", False


def illiquidity_premium(root: Path, label: str = "HIST") -> tuple[float, str]:
    a = pd.read_csv(path("processed", root) / "assumptions.csv")
    r = a[a["ID"] == f"{label}.ALL.LP"]
    if len(r) and pd.notna(r["Base"].iloc[0]):
        return float(r["Base"].iloc[0]), f"{label}.ALL.LP (assumption register; D3)"
    return 0.015, "D3 central value (assumption register row not held)"


def build_coe(y: float, ds: float, crp: float, erp: float, beta: float, lp: float) -> float:
    """Approach A build-up (VALUATION_SPEC §5). All inputs are decimals."""
    return (y - ds) + beta * (erp + crp) + lp


def coe_annual(root: Path) -> pd.DataFrame:
    """One row per date label in `coe_inputs_annual.csv` (FY2008-FY2025 plus the two valuation dates)."""
    c = pd.read_csv(path("processed", root) / "coe_inputs_annual.csv")
    prim = c[c["series"] == "d34_primary"].set_index("date_label")
    alt = c[c["series"] == "d41_term_spread_adjusted"].set_index("date_label")["waytm"]
    ver = damodaran_versions(root)
    arc = archive(root)
    band = settings(root)["persistence"]["coe_band_bp"] / 10_000
    rows = []
    for lab, r in prim.iterrows():
        year = int(r["year"])
        v = "CURR" if lab == "valuation_current" else "HIST"
        held = ver[v]
        lp, lp_src = illiquidity_premium(root, v)
        flags, srcs, inp = [], [], {}
        use_archive = v == "HIST" and year < 2025
        for k in ("ds", "crp", "erp", "beta"):
            if use_archive:
                val, src, matched = archived_input(arc, k, year)
                if not matched:
                    flags.append(f"{k}: nearest archived Damodaran version (none covers FY{year})")
            else:
                val, src = held[k]
                src = f"inputs workbook {src}"
            inp[k] = val
            srcs.append(f"{k}: {src}")
        if str(r["status"]).endswith("(flagged)"):
            flags.append(f"D41: {int(r['tenor_years'])}-year rate below the D34 tenor floor")
        if use_archive and inp["beta"] < 0.4:
            flags.append(f"beta {inp['beta']:.2f} is an outlier against 0.61-0.85 in the other archived versions")
        coe = build_coe(r["waytm"], inp["ds"], inp["crp"], inp["erp"], inp["beta"], lp)
        # D51 sensitivity: the January 2012 beta (betaemerg11, outlier) replaced by the January 2013 value
        beta_alt = inp["beta"]
        if use_archive and "betaemerg11" in srcs[3]:
            beta_alt = archived_input(arc, "beta", 2013)[0]
        coe_beta_alt = build_coe(r["waytm"], inp["ds"], inp["crp"], inp["erp"], beta_alt, lp)
        y_alt = alt.get(lab)
        coe_alt = build_coe(y_alt, inp["ds"], inp["crp"], inp["erp"], inp["beta"], lp) if pd.notna(y_alt) \
            else float("nan")
        h = held["ds"][0], held["crp"][0], held["erp"][0], held["beta"][0]
        rows.append({
            "date_label": lab, "year": year, "date": r["date"], "y_long": r["waytm"],
            "y_long_source": f"coe_inputs_annual d34_primary ({r['method']}; tenor {r['tenor_years']})",
            "y_long_d41_adjusted": y_alt, "default_spread": inp["ds"], "crp": inp["crp"], "erp_mature": inp["erp"],
            "beta_levered": inp["beta"], "illiquidity_premium": lp,
            "damodaran_basis": "archived version public at year-end (D48)" if use_archive else f"inputs workbook {v}",
            "damodaran_sources": "; ".join(srcs), "lp_source": lp_src, "coe": coe, "coe_d41_adjusted": coe_alt,
            "coe_minus_200bp": coe - band, "coe_plus_200bp": coe + band,
            "coe_2025_vintage": build_coe(r["waytm"], *h, lp), "beta_2012_replaced": beta_alt,
            "coe_beta2012_replaced": coe_beta_alt, "approach": "A (TZS build-up), primary",
            "flags": "; ".join(flags) or "none", "evidence_class": "Estimated"})
    return pd.DataFrame(rows)
