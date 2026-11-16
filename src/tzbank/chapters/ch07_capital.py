"""Chapter 7: capital allocation and incremental returns (blueprint §2 incremental capital productivity, §4 chapter 7;
H7.1-H7.3; CLAUDE.md §8 retained-earnings guardrail).

RORE_primary(t, k) = (NI_attr,t - NI_attr,t-k) / sum_{j=t-k}^{t-1} (NI_attr,j - Div_j)
dROE(t, k)         = (NI_attr,t - NI_attr,t-k) / (B_t - B_t-k)

Div_j = declared dividend for fiscal year j (DPS x shares outstanding, ordinary plus special). New capital (rights
issues, IPO) found in the equity statements is listed in ``CAPITAL_RAISES``; a window (t-k, t] that contains a raise
has no primary RORE (unadjusted), and the adjusted series adds the raise to the denominator. EPS-based RORE (split-
adjusted per-share data) is supplementary. Capital: Tier 1 headroom on the basis of D2/D17 (NMB Bank-solo; CRDB
Bank-solo FY2009-14, Group from FY2015) against the requirement stated in each bank's annual report
(``bot_policy_text.csv``, kind ``tier1_requirement`` and ``capital_conservation_buffer``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tzbank.chapters.ch04.common import COLORS, FOCUS, avg, col, div, pct, plt_subplots, save, style, tickers, wide
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text

# New capital found in the equity statements (prompt 06 task 1): TZS, fiscal year, source
CAPITAL_RAISES = [
    {"bank_id": "B01", "fiscal_year": 2009, "amount": 23_711_596e3, "kind": "rights issue and IPO",
     "source": "CRDB/2010.pdf p.34-36 (statement of changes in equity, prior-year column; cash flow 'Proceeds from "
               "issue of ordinary shares' 23,711,596 (TZS 000))"},
    {"bank_id": "B01", "fiscal_year": 2015, "amount": 150_432e6, "kind": "rights issue (435,306,424 shares at TZS 350)",
     "source": "CRDB/2015.pdf p.34, p.39 (equity statement and cash flow 'Rights issue' 150,432 TZS m), p.131"},
]
LEVEL = 0.90
BUFFERS = (0.0, 0.02, 0.04)  # management buffers above the requirement for distributable capacity (Scenario)


def raises() -> pd.DataFrame:
    return pd.DataFrame(CAPITAL_RAISES).assign(evidence_class="Observed")


def _split_factor(root: Path) -> dict[str, float]:
    """Per-share divisor for pre-split data (config/corporate_actions.csv); FY data up to 2025 precede NMB's 2026 split,
    so every per-share series of a bank is divided by the same factor and RORE ratios are invariant to it."""
    ca = pd.read_csv(root / "config" / "corporate_actions.csv")
    return {r.bank_id: float(r.ratio) for r in ca.itertuples() if r.action == "split"}


# --- earnings and payout -------------------------------------------------------------------------------------------


def payout(w: pd.DataFrame) -> pd.DataFrame:
    ni = col(w, "profit_attributable").fillna(col(w, "net_profit"))
    sh = col(w, "shares_outstanding")
    out = pd.DataFrame({"bank_id": w["bank_id"], "fiscal_year": w["fiscal_year"], "ni_attr": ni, "shares": sh,
                        "dps_ordinary": col(w, "dps"), "dps_special": col(w, "dps_special")})
    out["div_ordinary"] = out["dps_ordinary"] * sh
    out["div_special"] = out["dps_special"] * sh
    out["div_total"] = out["div_ordinary"] + out["div_special"].fillna(0)
    out["payout_ordinary"] = div(out["div_ordinary"], ni)
    out["payout_special"] = div(out["div_special"], ni)
    out["payout_total"] = div(out["div_total"], ni)
    out["dividend_basis"] = "declared for the fiscal year: DPS x shares outstanding at year-end"
    out["evidence_class"] = "Derived"
    return out


def rore(w: pd.DataFrame, pay: pd.DataFrame, rz: pd.DataFrame, split: dict[str, float]) -> pd.DataFrame:
    rows = []
    eq = col(w, "equity_attributable").fillna(col(w, "total_equity"))
    eps = col(w, "eps_basic")
    for b, g in w.assign(_eq=eq, _eps=eps).groupby("bank_id"):
        g = g.set_index("fiscal_year")
        p = pay[pay["bank_id"] == b].set_index("fiscal_year")
        f = split.get(b, 1.0)
        for k in (3, 5):
            for t in g.index:
                if t - k not in g.index:
                    continue
                yrs = list(range(t - k, t))
                ni_t, ni_0 = p["ni_attr"].get(t), p["ni_attr"].get(t - k)
                ret = [p["ni_attr"].get(j, np.nan) - p["div_total"].get(j, np.nan) for j in yrs]
                retained = float(np.sum(ret)) if all(pd.notna(ret)) else np.nan
                rz_in = rz[(rz["bank_id"] == b) & (rz["fiscal_year"] > t - k) & (rz["fiscal_year"] <= t)]
                raised = float(rz_in["amount"].sum())
                d_ni = ni_t - ni_0 if pd.notna(ni_t) and pd.notna(ni_0) else np.nan
                d_b = g["_eq"].get(t, np.nan) - g["_eq"].get(t - k, np.nan)
                # EPS-based (supplementary): split-adjusted per-share data
                e_t, e_0 = g["_eps"].get(t, np.nan) / f, g["_eps"].get(t - k, np.nan) / f
                e_ret = [g["_eps"].get(j, np.nan) / f - (p["dps_ordinary"].get(j, np.nan)
                                                          + (p["dps_special"].get(j, 0) if pd.notna(
                                                              p["dps_special"].get(j, np.nan)) else 0)) / f
                         for j in yrs]
                e_den = float(np.sum(e_ret)) if all(pd.notna(e_ret)) else np.nan
                rows.append({
                    "bank_id": b, "fiscal_year": int(t), "k": k, "window": f"{t - k}-{t}", "d_ni_attr": d_ni,
                    "retained_equity": retained, "capital_raised_in_window": raised,
                    "raise_flag": "; ".join(f"FY{r.fiscal_year} {r.kind}" for r in rz_in.itertuples()) or "none",
                    "rore_primary": d_ni / retained if raised == 0 and retained and retained > 0 else np.nan,
                    "rore_adjusted": d_ni / (retained + raised) if retained and retained + raised > 0 else np.nan,
                    "d_roe": d_ni / d_b if raised == 0 and d_b and d_b > 0 else np.nan,
                    "d_roe_adjusted": d_ni / (d_b - raised) if d_b and d_b - raised > 0 else np.nan,
                    "rore_eps_supplementary": (e_t - e_0) / e_den if raised == 0 and e_den and e_den > 0 else np.nan,
                    "split_divisor_applied": f,
                    "avg_assets_t": float(avg(w[w["bank_id"] == b], "total_assets").set_axis(
                        w.loc[w["bank_id"] == b, "fiscal_year"]).get(t, np.nan))})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


# --- capital ------------------------------------------------------------------------------------------------------


def capital_basis(root: Path) -> pd.DataFrame:
    """Tier 1 ratio, total capital ratio, Tier 1 capital and RWA on the D2/D17 basis."""
    c = pd.read_csv(path("processed", root) / "capital_long.csv", low_memory=False, dtype={"bank_id": str})
    c = c[(c["period_type"] == "FY") & c["canonical_name"].isin(["tier1_ratio", "total_capital_ratio", "rwa_total",
                                                                  "tier1_capital"])]
    def rule(b: str, y: int) -> str:
        if b == "B02":
            return "BANK"
        if b == "B01":
            return "BANK" if y <= 2014 else "GROUP"
        return "BANK"
    c = c[[r.basis == rule(r.bank_id, int(r.fiscal_year)) for r in c.itertuples()]]
    c = c.sort_values("publication_date").drop_duplicates(["bank_id", "fiscal_year", "canonical_name"], keep="last")
    out = c.pivot_table(index=["bank_id", "fiscal_year"], columns="canonical_name", values="value",
                        aggfunc="first").reset_index()
    out["capital_basis"] = [("Bank-solo (D2)" if b == "B02" else "Bank-solo (D17)" if y <= 2014 else "Group")
                            if b in FOCUS else "Bank (= Group)" for b, y in zip(out["bank_id"], out["fiscal_year"],
                                                                                strict=True)]
    return out


READING_A = "A: stated level includes the conservation buffer (primary)"
READING_B = "B: stated level is minimum + bank-specific add-on; conservation buffer stacks on top (sensitivity)"
CCB_RATE = 0.025
STRESS_START = (2015, 2018)  # window start years in the stress trough / CRDB post-raise (prompt 07b task 4b)


def _stated(p: pd.DataFrame, kind: str, bank: str) -> pd.DataFrame:
    st = p[(p["kind"] == kind) & (p["scope"] == bank) & (p["gate_result"] == "PASS")].copy()
    st["fy"] = st["report_fy"].astype(int)
    return st.sort_values(["fy", "file", "pdf_page"])


def requirements(root: Path, years: range) -> pd.DataFrame:
    """Effective Tier 1 and total capital requirement per bank-year (D52 as amended, prompt 07b).

    Stated levels come from each bank's annual report (``bot_policy_text.csv``: ``tier1_requirement``,
    ``total_capital_requirement``), carried forward to unstated years, BoT's 10% / 12% before the first statement.
    Conservation buffer (CCB, 2.5% of RWA, Tier 1): added where the stated Tier 1 level excludes it (below 12.5%), from
    the EARLIER stated start date (August 2017, NMB FY2017 report: conservative, D52 amended); the later stated date
    (August 2018) is kept as ``*_later_ccb``. CRDB states 12.5% / 14.5% from FY2015 and never names the buffer, so two
    readings are carried: A (stated level already includes the buffer, primary) and B (stated level = 10% / 12% minimum
    + 2.5% bank-specific add-on, buffer stacks from August 2017: 15% / 17%). For NMB both readings coincide.
    """
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    ccb = p[p["kind"] == "capital_conservation_buffer"]
    starts = sorted(pd.to_datetime(ccb["event_date"]).unique())
    early, late = (pd.Timestamp(starts[0]), pd.Timestamp(starts[-1])) if starts else (None, None)
    rules = settings(root).get("capital_rules", {})
    if rules.get("ccb_start"):  # deferred-fix parameter (CLAUDE.md section 10, D55)
        early = pd.Timestamp(rules["ccb_start"])
    reading = str(rules.get("crdb_requirement_reading", "A")).upper()
    rows = []
    for b in FOCUS:
        s1 = _stated(p, "tier1_requirement", b).groupby("fy")["value"].max()
        s2 = _stated(p, "total_capital_requirement", b).groupby("fy")["value"].max()
        t1 = t2 = np.nan
        src1 = src2 = ""
        for y in years:
            if y in s1.index:
                t1, src1 = float(s1[y]), f"stated in FY{y} annual report"
            elif np.isnan(t1):
                t1, src1 = 0.10, "BoT statutory core-capital minimum 10% (BoT Annual Report 2011/12 p.12); not stated"
            elif not src1.startswith("carried"):
                src1 = f"carried forward from the last stated year ({src1})"
            if y in s2.index:
                t2, src2 = float(s2[y]), f"stated in FY{y} annual report"
            elif np.isnan(t2):
                t2, src2 = 0.12, "BoT statutory total-capital minimum 12%; not stated"
            elif not src2.startswith("carried"):
                src2 = f"carried forward from the last stated year ({src2})"
            fye = pd.Timestamp(y, 12, 31)
            on_early = early is not None and fye >= early
            on_late = late is not None and fye >= late
            excl = t1 < 0.125  # stated level excludes the buffer
            ccb_a = CCB_RATE if excl and on_early else 0.0
            ccb_late = CCB_RATE if excl and on_late else 0.0
            ccb_b = CCB_RATE if on_early else 0.0  # reading B: the buffer stacks on any stated level
            flag = []
            if y == 2017 and b == "B02":
                flag.append("CCB start stated as Aug 2017 (FY2017 report) and Aug 2018 (FY2018-20 reports); earlier "
                            "date used (D52 amended)")
            if b == "B01" and not excl and y >= 2015:
                flag.append("CRDB does not say whether 12.5% / 14.5% includes the conservation buffer; owner decision "
                            "(D52 amended): reading A primary, reading B sensitivity")
            rows.append({"bank_id": b, "fiscal_year": y, "tier1_stated": t1, "total_stated": t2,
                         "ccb_added": ccb_a, "tier1_requirement": t1 + ccb_a, "total_requirement": t2 + ccb_a,
                         "ccb_added_later_date": ccb_late, "tier1_requirement_later_ccb": t1 + ccb_late,
                         "total_requirement_later_ccb": t2 + ccb_late,
                         "tier1_requirement_reading_b": t1 + ccb_b, "total_requirement_reading_b": t2 + ccb_b,
                         "ccb_start_primary": early.date().isoformat() if early is not None else None,
                         "ccb_start_later": late.date().isoformat() if late is not None else None,
                         "tier1_source": src1, "total_source": src2, "requirement_flag": "; ".join(flag) or "none",
                         "rule": "effective = stated minimum + 2.5% CCB where the stated Tier 1 level is below 12.5%, "
                                 "from the earlier stated start (Aug 2017); reading B adds the CCB to any stated level"})
    out = pd.DataFrame(rows)
    if reading == "B":  # config: reading B primary for CRDB, A kept in the *_reading_b columns as the alternative
        m = out["bank_id"] == "B01"
        for c in ("tier1", "total"):
            a = out.loc[m, f"{c}_requirement"].copy()
            out.loc[m, f"{c}_requirement"] = out.loc[m, f"{c}_requirement_reading_b"]
            out.loc[m, f"{c}_requirement_reading_b"] = a
    out["crdb_reading_primary"] = reading
    out["evidence_class"] = "Derived"
    return out


def requirement_statements(root: Path) -> pd.DataFrame:
    """Prompt 07b task 2: Tier 1 and total requirement exactly as stated in each report, FY2015-25, with page quotes,
    plus the sentences that explain CRDB's 12.5% / 14.5% (Observed)."""
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    rows = []
    for b in FOCUS:
        s1, s2 = _stated(p, "tier1_requirement", b), _stated(p, "total_capital_requirement", b)
        for y in range(2015, 2026):
            a, c = s1[s1["fy"] == y], s2[s2["fy"] == y]
            r1 = a.iloc[0] if len(a) else None
            same = c[c["file"] == r1["file"]] if r1 is not None else c.iloc[0:0]
            r2 = same.iloc[0] if len(same) else (c.iloc[0] if len(c) else None)
            rows.append({"bank_id": b, "fiscal_year": y, "statement": "requirement as stated",
                         "tier1_stated": r1["value"] if r1 is not None else np.nan,
                         "total_stated": r2["value"] if r2 is not None else np.nan,
                         "file": r1["file"] if r1 is not None else "NOT FOUND",
                         "pdf_page": r1["pdf_page"] if r1 is not None else np.nan,
                         "quote": r1["quote"] if r1 is not None else "TO COLLECT",
                         "total_file_page": f"{r2['file']} p.{int(r2['pdf_page'])}" if r2 is not None else "NOT FOUND"})
    ctx = p[p["kind"] == "capital_requirement_context"].sort_values(["scope", "report_fy", "pdf_page"])
    for r in ctx.itertuples():
        rows.append({"bank_id": r.scope, "fiscal_year": int(r.report_fy), "statement": "context of the requirement",
                     "tier1_stated": np.nan, "total_stated": np.nan, "file": r.file, "pdf_page": r.pdf_page,
                     "quote": r.quote, "total_file_page": ""})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Observed"
    return out


def dsib_search(root: Path) -> pd.DataFrame:
    """Prompt 07b task 2 / blueprint ch. 3: every D-SIB or capital-surcharge passage found, and the search record
    (documents searched, hits) per scope (Observed)."""
    p = pd.read_csv(path("interim", root) / "bot_policy_text.csv")
    d = p[p["kind"].isin(["dsib_search", "dsib_statement"])][["kind", "scope", "report_fy", "file", "pdf_page",
                                                                "value", "quote"]].copy()
    found = np.where(d["value"] > 0, "passages found (listed below)",
                     "NOT FOUND: no D-SIB designation or capital surcharge named")
    d["finding"] = np.where(d["kind"] == "dsib_search", found, "passage")
    d["evidence_class"] = "Observed"
    return d.sort_values(["kind", "scope", "report_fy"], ascending=[False, True, True]).reset_index(drop=True)


def _binding(t1: float, tot: float) -> str:
    if pd.isna(t1) and pd.isna(tot):
        return "not held"
    if pd.isna(tot):
        return "Tier 1 (total ratio not held)"
    if pd.isna(t1):
        return "total capital (Tier 1 ratio not held)"
    return "total capital" if tot < t1 else "Tier 1"


def capital(w: pd.DataFrame, cb: pd.DataFrame, req: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """Capital ratios, headroom over the Tier 1 and total requirements (primary, later CCB date, CRDB reading B) and the
    binding constraint per year (prompt 07b task 3)."""
    base = w[["bank_id", "fiscal_year"]].copy()
    eq = col(w, "equity_attributable").fillna(col(w, "total_equity"))
    base["equity"] = eq.to_numpy()
    base["total_assets"] = col(w, "total_assets").to_numpy()
    base["total_equity"] = col(w, "total_equity").to_numpy()
    base["equity_to_assets"] = div(col(w, "total_equity"), col(w, "total_assets")).to_numpy()
    base["roe"] = div(pay["ni_attr"], avg(w.assign(_e=eq), "_e")).to_numpy()
    out = base.merge(cb, on=["bank_id", "fiscal_year"], how="left").merge(
        req, on=["bank_id", "fiscal_year"], how="left").merge(
        pay[["bank_id", "fiscal_year", "payout_total"]], on=["bank_id", "fiscal_year"], how="left")
    out = out.sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)
    g = out.groupby("bank_id")
    prev = g["fiscal_year"].diff() == 1
    out["rwa_growth"] = (out["rwa_total"] / g["rwa_total"].shift(1) - 1).where(prev)
    out["equity_growth"] = (out["equity"] / g["equity"].shift(1) - 1).where(prev)
    out["tier1_growth"] = (out["tier1_capital"] / g["tier1_capital"].shift(1) - 1).where(prev)
    out["tier1_headroom"] = out["tier1_ratio"] - out["tier1_requirement"]
    out["total_headroom"] = out["total_capital_ratio"] - out["total_requirement"]
    out["tier1_headroom_later_ccb"] = out["tier1_ratio"] - out["tier1_requirement_later_ccb"]
    out["total_headroom_later_ccb"] = out["total_capital_ratio"] - out["total_requirement_later_ccb"]
    out["tier1_headroom_reading_b"] = out["tier1_ratio"] - out["tier1_requirement_reading_b"]
    out["total_headroom_reading_b"] = out["total_capital_ratio"] - out["total_requirement_reading_b"]
    out["binding_constraint"] = [_binding(a, b) for a, b in zip(out["tier1_headroom"], out["total_headroom"],
                                                                strict=True)]
    out["binding_headroom"] = out[["tier1_headroom", "total_headroom"]].min(axis=1)
    out["binding_constraint_reading_b"] = [_binding(a, b) for a, b in zip(
        out["tier1_headroom_reading_b"], out["total_headroom_reading_b"], strict=True)]
    out["tier2_ratio"] = out["total_capital_ratio"] - out["tier1_ratio"]
    out["rwa_density"] = out["rwa_total"] / out["total_assets"]
    out["internal_capital_generation"] = out["roe"] * (1 - out["payout_total"])
    out["evidence_class"] = "Derived"
    return out


def leverage_decomposition(cap: pd.DataFrame) -> pd.DataFrame:
    """E/TA = (T1/RWA) x (RWA/TA) x (E/T1): log gap NMB - CRDB split into capital ratio (computed T1 / RWA), RWA
    density and E/T1; years without Tier 1 capital or RWA on the D2/D17 basis have no decomposition. NMB's E and TA are
    Group while its Tier 1 and RWA are Bank-solo (D2), so its E/T1 (and RWA/TA) terms include a basis effect."""
    c = cap[cap["bank_id"].isin(FOCUS)].set_index(["fiscal_year", "bank_id"])
    rows = []
    for y in sorted(c.index.get_level_values(0).unique()):
        if (y, "B01") not in c.index or (y, "B02") not in c.index:
            continue
        n, r = c.loc[(y, "B02")], c.loc[(y, "B01")]
        # computed T1 / RWA (exact identity; reported ratios are rounded)
        parts = {"tier1_ratio": np.log((n["tier1_capital"] / n["rwa_total"]) / (r["tier1_capital"] / r["rwa_total"])),
                 "rwa_density": np.log(n["rwa_density"] / r["rwa_density"]),
                 "equity_to_tier1": np.log((n["total_equity"] / n["tier1_capital"])
                                           / (r["total_equity"] / r["tier1_capital"]))}
        tot = np.log(n["equity_to_assets"] / r["equity_to_assets"])
        rows.append({"fiscal_year": int(y), "log_gap_equity_to_assets": tot, **{f"log_gap_{k}": v for k, v in
                                                                                 parts.items()},
                     "residual": tot - sum(parts.values()), "nmb_equity_to_assets": n["equity_to_assets"],
                     "crdb_equity_to_assets": r["equity_to_assets"], "nmb_tier1_headroom": n["tier1_headroom"],
                     "crdb_tier1_headroom": r["tier1_headroom"],
                     "basis_note": "NMB: equity and assets Group, Tier 1 and RWA Bank-solo (D2): the equity / Tier 1 "
                                   "and RWA density terms include a Group-vs-Bank basis effect; CRDB "
                                   + ("Group throughout" if y >= 2015 else "Bank-solo capital (D17)")})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


def distributable_capacity(cap: pd.DataFrame, pay: pd.DataFrame, measure: str = "tier1") -> pd.DataFrame:
    """09a DDM input: excess capital over requirement + management buffer (TZS) and the payout consistent with holding
    the ratio while RWA grows at its 2019-25 mean (Scenario).

    Tier 1 (primary): payout_max = 1 - (req + buffer) x RWA_t x g / NI_t.
    Total capital (09a sensitivity, prompt 07b task 3): Tier 2 is held at its latest share of RWA, so retained earnings
    fund only the Tier 1 part: payout_max = 1 - (req_total + buffer - T2/RWA) x RWA_t x g / NI_t.
    CRDB rows are given under requirement readings A (primary) and B."""
    rows = []
    for b in FOCUS:
        ratio = "tier1_ratio" if measure == "tier1" else "total_capital_ratio"
        full = cap[cap["bank_id"] == b]
        c = full.dropna(subset=[ratio, "rwa_total"])
        last = c.iloc[-1]
        g = full.loc[full["fiscal_year"].between(2019, 2025), "rwa_growth"].mean()  # same years for both measures
        ni = pay[(pay["bank_id"] == b) & (pay["fiscal_year"] == last["fiscal_year"])]["ni_attr"].iloc[0]
        t2 = last["tier2_ratio"] if measure == "total" else 0.0
        readings = [(READING_A, "")] + ([(READING_B, "_reading_b")] if b == "B01" else [])
        for reading, sfx in readings:
            req = last[f"{measure}_requirement{sfx}"]
            for buf in BUFFERS:
                target = req + buf
                rows.append({"bank_id": b, "fiscal_year": int(last["fiscal_year"]), "measure": measure,
                             "requirement_reading": reading if b == "B01" else "A = B for NMB",
                             "management_buffer": buf, "requirement": req, "target_ratio": target,
                             "ratio": last[ratio], "rwa": last["rwa_total"],
                             "excess_capital_tzs": (last[ratio] - target) * last["rwa_total"],
                             "tier2_share_of_rwa_held": t2 if measure == "total" else np.nan,
                             "rwa_growth_2019_25_mean": g,
                             "payout_max_steady_state": 1 - (target - t2) * last["rwa_total"] * g / ni,
                             "payout_actual": last["payout_total"], "capital_basis": last["capital_basis"],
                             "evidence_class": "Scenario"})
    out = pd.DataFrame(rows)
    if measure == "tier1":  # keep the column names 09a already reads
        out = out.rename(columns={"target_ratio": "tier1_target", "ratio": "tier1_ratio",
                                  "excess_capital_tzs": "excess_tier1_tzs"})
    return out


# --- hypotheses ------------------------------------------------------------------------------------------------------


def _coe_fy(coe: pd.DataFrame) -> pd.Series:
    return coe[coe["date_label"].str.startswith("FY")].set_index("year")["coe"]


def rore_windows(rr: pd.DataFrame, pay: pd.DataFrame, rz: pd.DataFrame, coe: pd.DataFrame) -> pd.DataFrame:
    """Prompt 07b task 4a/4b: every NMB and CRDB window (k = 3, 5) with the reason it is in or out of the primary and
    raise-adjusted series, the stress-start flag (t-k in FY2015-18) and the window-average COE used by H7.1."""
    c = _coe_fy(coe)
    rows = []
    for r in rr[rr["bank_id"].isin(FOCUS)].itertuples():
        t, k, b = int(r.fiscal_year), int(r.k), r.bank_id
        p = pay[pay["bank_id"] == b].set_index("fiscal_year")
        miss_ni = [y for y in (t - k, t) if pd.isna(p["ni_attr"].get(y, np.nan))]
        miss_ret = [j for j in range(t - k, t) if pd.isna(p["ni_attr"].get(j, np.nan))
                    or pd.isna(p["div_total"].get(j, np.nan))]
        miss_div = [j for j in range(t - k, t) if pd.isna(p["div_total"].get(j, np.nan))]
        if miss_ni:
            data = f"excluded: attributable profit not held for FY{', FY'.join(map(str, miss_ni))}"
        elif miss_ret:
            data = ("excluded: retained equity not computable (dividend not held for FY"
                    + ", FY".join(map(str, miss_div)) + ")") if miss_div else \
                "excluded: retained equity not computable (profit not held)"
        elif not r.retained_equity > 0:
            data = "excluded: retained equity <= 0"
        else:
            data = "included"
        raised = r.capital_raised_in_window > 0
        prim = f"excluded: {r.raise_flag} in (t-k, t] (D52)" if raised else data
        adj = data if data != "included" or (r.retained_equity + r.capital_raised_in_window) > 0 else \
            "excluded: denominator <= 0"
        cw = c.reindex(range(t - k, t + 1)).mean() if c.reindex(range(t - k, t + 1)).notna().all() else np.nan
        raise_years = set(rz.loc[rz["bank_id"] == b, "fiscal_year"].astype(int))
        rows.append({"bank_id": b, "k": k, "window": r.window, "start_year": t - k, "end_year": t,
                     "primary_included": prim == "included", "primary_reason": prim,
                     "adjusted_included": adj == "included", "adjusted_reason": adj,
                     "stress_start_flag": STRESS_START[0] <= t - k <= STRESS_START[1],
                     "raise_in_base_year": (t - k) in raise_years,
                     "rore_primary": r.rore_primary, "rore_adjusted": r.rore_adjusted, "d_roe": r.d_roe,
                     "d_roe_adjusted": r.d_roe_adjusted, "rore_eps_supplementary": r.rore_eps_supplementary,
                     "coe_window": cw, "in_h71": adj == "included" and pd.notna(cw)})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


SERIES = {"rore_adjusted": "RORE, raise-adjusted (headline for CRDB)", "rore_primary": "RORE, primary (D52)",
          "d_roe_adjusted": "dROE, raise-adjusted", "d_roe": "dROE, primary",
          "rore_eps_supplementary": "EPS-based RORE, split-adjusted (supplementary)"}


def rore_summary(win: pd.DataFrame) -> pd.DataFrame:
    """Medians per bank, k and series, with and without stress-start windows (prompt 07b task 4b/4d)."""
    rows = []
    for (b, k), g in win.groupby(["bank_id", "k"]):
        for s, label in SERIES.items():
            x = g[s].dropna()
            xs = g.loc[~g["stress_start_flag"], s].dropna()
            rows.append({"bank_id": b, "k": int(k), "series": s, "label": label, "n_windows": len(x),
                         "median": x.median() if len(x) else np.nan, "n_excl_stress_start": len(xs),
                         "median_excl_stress_start": xs.median() if len(xs) else np.nan,
                         "latest_window": g["window"].iloc[-1], "latest": g[s].iloc[-1]})
    out = pd.DataFrame(rows)
    out["evidence_class"] = "Derived"
    return out


def _within_slope(ls: np.ndarray, y: np.ndarray, bank: np.ndarray) -> float:
    xd, yd = ls.copy(), y.copy()
    for b in np.unique(bank):
        m = bank == b
        xd[m] -= ls[m].mean()
        yd[m] -= y[m].mean()
    return float((xd * yd).sum() / (xd * xd).sum())


def slope_bootstrap(x: pd.DataFrame, reps: int, block: int, seed: list[int], level: float) -> dict[str, tuple]:
    """D46 moving-block bootstrap (block 3, percentile) of the H7.2 slopes, resampling consecutive windows within each
    bank so the overlap between windows stays in the replicates (prompt 07b task 5)."""
    from tzbank.chapters.ch05.models import block_indices, interval

    rng = np.random.default_rng(seed)
    x = x.sort_values(["bank_id", "fiscal_year"])
    parts = {b: g for b, g in x.groupby("bank_id")}
    pooled, per = np.full(reps, np.nan), {b: np.full(reps, np.nan) for b in parts}
    for i in range(reps):
        ls, y, bk = [], [], []
        for b, g in parts.items():
            ix = block_indices(len(g), block, rng)
            ls.append(g["ls"].to_numpy()[ix])
            y.append(g["rore_adjusted"].to_numpy()[ix])
            bk.append(np.full(len(ix), b))
            if np.ptp(ls[-1]) > 0:
                per[b][i] = np.polyfit(ls[-1], y[-1], 1)[0]
        lsa, ya, bka = np.concatenate(ls), np.concatenate(y), np.concatenate(bk)
        pooled[i] = _within_slope(lsa, ya, bka)
    out = {"pooled": interval(pd.Series(pooled), level)}
    for b, v in per.items():
        out[b] = interval(pd.Series(v), level)
    return out


def hypotheses(rr: pd.DataFrame, win: pd.DataFrame, cap: pd.DataFrame, t: dict[str, str],
               boot: dict[str, object]) -> pd.DataFrame:
    rows = []
    # H7.1: RORE (raise-adjusted headline, k = 5 decides; k = 3 and primary alongside) vs window-average COE, +/- 200 bp
    ev, per = [], {}
    for b in FOCUS:
        res = []
        for k in (5, 3):
            x = win[(win["bank_id"] == b) & (win["k"] == k) & win["in_h71"]]
            ex = x["rore_adjusted"] - x["coe_window"]
            above_hi, below_lo = float((ex > 0.02).mean()), float((ex < -0.02).mean())
            med = float(ex.median())
            xp = x[x["primary_included"]]
            medp = float((xp["rore_primary"] - xp["coe_window"]).median()) if len(xp) else np.nan
            xs = x[~x["stress_start_flag"]]
            meds = float((xs["rore_adjusted"] - xs["coe_window"]).median()) if len(xs) else np.nan
            res.append((k, len(x), med, above_hi, below_lo))
            ev.append(f"{t[b]} k={k}: raise-adjusted {len(x)} windows, median RORE - COE {pct(med)}, above COE + 200 bp "
                      f"in {above_hi:.0%}, below COE - 200 bp in {below_lo:.0%}; primary {len(xp)} windows, median "
                      f"{pct(medp)}; excluding stress-start windows {pct(meds)}")
        k5 = res[0]
        per[b] = "supported" if k5[2] > 0.02 and k5[3] >= 0.5 else "rejected" if k5[2] < -0.02 and k5[4] >= 0.5 \
            else "inconclusive"
        x5 = win[(win["bank_id"] == b) & (win["k"] == 5) & win["in_h71"]]
        xs5 = x5[~x5["stress_start_flag"]]
        above = x5[x5["rore_adjusted"] > x5["coe_window"] + 0.02]
        if per[b] == "inconclusive" and len(xs5) and (xs5["rore_adjusted"] - xs5["coe_window"]).median() < 0 \
                and len(above) and above["end_year"].min() >= 2019:
            # owner-accepted wording (prompt 09a pre-flight A; open issue 4)
            per[b] = (f"above COE only in windows ending in the post-2018 recovery (FY{int(above['end_year'].min())}"
                      f"-{int(above['end_year'].max())}), so not supported through the cycle; inconclusive overall")
    v1 = "supported" if all(v == "supported" for v in per.values()) else "rejected" if all(
        v == "rejected" for v in per.values()) else "inconclusive"
    by_bank = ", ".join(f"{t[b]} {v}" for b, v in per.items())
    rows.append({"hypothesis": "H7.1", "verdict": v1, "verdict_by_bank": by_bank,
                 "evidence": "; ".join(ev) + "; per bank (k = 5): " + by_bank,
                 "rule": "per bank (k = 5, capital-raise-adjusted RORE): supported when the median excess over the "
                         "window COE exceeds 200 bp and at least half the windows clear COE + 200 bp; rejected when the "
                         "mirror image holds; the joint verdict needs both banks"})
    # H7.2: incremental ROE vs scale (log average assets), within-bank slope, k = 3 adjusted
    x = rr[(rr["k"] == 3) & rr["bank_id"].isin(FOCUS)].dropna(subset=["rore_adjusted", "avg_assets_t"]).copy()
    x = x[x["rore_adjusted"].abs() < 2]
    x["ls"] = np.log(x["avg_assets_t"])
    xd = x["ls"] - x.groupby("bank_id")["ls"].transform("mean")
    yd = x["rore_adjusted"] - x.groupby("bank_id")["rore_adjusted"].transform("mean")
    bcoef = float((xd * yd).sum() / (xd * xd).sum())
    e = yd - bcoef * xd
    dfree = len(x) - 3
    se = float(np.sqrt((e @ e) / dfree / (xd @ xd)))
    tc = stats.t.ppf(1 - (1 - LEVEL) / 2, dfree)
    lo, hi = bcoef - tc * se, bcoef + tc * se
    bi = slope_bootstrap(x, int(boot["reps"]), int(boot["block"]), boot["seed"], LEVEL)  # type: ignore[arg-type]
    blo, bhi = bi["pooled"]
    per_bank = []
    for b in FOCUS:
        xb = x[x["bank_id"] == b]
        r = stats.linregress(xb["ls"], xb["rore_adjusted"])
        tb = stats.t.ppf(1 - (1 - LEVEL) / 2, len(xb) - 2)
        plo, phi = bi[b]
        per_bank.append(f"{t[b]} {r.slope:+.2f}, block bootstrap [{plo:+.2f}, {phi:+.2f}] (t interval "
                        f"[{r.slope - tb * r.stderr:+.2f}, {r.slope + tb * r.stderr:+.2f}], n={len(xb)})")
    v2 = "supported" if bhi < 0 else "rejected" if blo > 0 else "inconclusive"
    v2_t = "supported" if hi < 0 else "rejected" if lo > 0 else "inconclusive"
    rows.append({"hypothesis": "H7.2", "verdict": v2, "verdict_by_bank": "pooled within-bank slope",
                 "evidence": f"within-bank slope of incremental ROE (k = 3, raise-adjusted RORE) on log average assets: "
                             f"{bcoef:+.2f}; 90% moving-block bootstrap interval [{blo:+.2f}, {bhi:+.2f}] (block "
                             f"{boot['block']}, {boot['reps']:,} replicates, windows resampled in blocks within each "
                             f"bank); OLS t interval [{lo:+.2f}, {hi:+.2f}] (verdict on the t interval: {v2_t}; change: "
                             f"{'none' if v2 == v2_t else v2_t + ' -> ' + v2}); n = {len(x)} overlapping windows; per "
                             f"bank: " + "; ".join(per_bank),
                 "rule": "supported when the 90% block-bootstrap interval of the slope lies below zero; rejected when "
                         "above zero (D46 bootstrap allows for window overlap; descriptive)"})
    # H7.3: growth vs distribution
    pay = cap[cap["bank_id"].isin(FOCUS) & cap["fiscal_year"].between(2010, 2025)]
    a, b_ = pay[pay["bank_id"] == "B02"], pay[pay["bank_id"] == "B01"]
    tp = stats.ttest_ind(a["payout_total"].dropna(), b_["payout_total"].dropna(), equal_var=False)
    tg = stats.ttest_ind(a["rwa_growth"].dropna(), b_["rwa_growth"].dropna(), equal_var=False)
    diff = tp.pvalue < 0.10 or tg.pvalue < 0.10
    rows.append({"hypothesis": "H7.3", "verdict": "supported" if diff else "inconclusive", "verdict_by_bank": "joint",
                 "evidence": f"mean payout 2010-25: NMB {pct(a['payout_total'].mean())}, CRDB "
                             f"{pct(b_['payout_total'].mean())} (Welch p = {tp.pvalue:.2f}); mean RWA growth: NMB "
                             f"{pct(a['rwa_growth'].mean())}, CRDB {pct(b_['rwa_growth'].mean())} (p = {tg.pvalue:.2f}); "
                             f"mean Tier 1 headroom: NMB {pct(a['tier1_headroom'].mean())}, CRDB "
                             f"{pct(b_['tier1_headroom'].mean())}",
                 "rule": "supported when payout or RWA growth differs between the banks at p < 0.10 (Welch); "
                         "inconclusive otherwise (a descriptive comparison, not causal)"})
    h = pd.DataFrame(rows)
    h["evidence_class"] = "Interpretation"
    return h


NOTE = """# Chapter 7: capital allocation and incremental returns

Generated by `tzbank.chapters.ch07_capital` (stage `capital`). Blueprint §2 (incremental capital productivity) and
§4 chapter 7; hypotheses H7.1-H7.3; CLAUDE.md §8: primary RORE on attributable earnings and retained equity, EPS-based
RORE supplementary. Rules: D52 as amended (prompt 07b).

## Research question

Have the banks reinvested retained earnings at returns above the cost of equity, are incremental returns falling
with scale, and how do they split capital between growth and distribution? Why is CRDB more levered than NMB?

## Variable definitions

- RORE (primary) = (NI_attr,t - NI_attr,t-k) / sum over j = t-k..t-1 of (NI_attr,j - dividends declared for j),
  k = 3, 5; not computed for windows (t-k, t] containing a capital raise (D52); the raise-adjusted series adds the raise
  to the denominator and is the CRDB headline. dROE = change in NI_attr / change in attributable equity (raise-adjusted
  variant subtracts the raise).
- EPS-based RORE (supplementary) = change in EPS / sum of (EPS - DPS), split-adjusted (`config/corporate_actions.csv`).
- Stress-start window: start year t-k in FY2015-18 (the 2015-18 stress trough and, for CRDB, the years after the
  FY2015 rights issue). Medians are shown with and without them.
- Payout = DPS x shares outstanding / attributable profit; ordinary and special (NMB FY2025) shown separately.
- Requirement (D52 amended): the Tier 1 and total capital minimum stated in each bank's report, plus the 2.5%
  conservation buffer (CCB) where the stated Tier 1 level excludes it, from the EARLIER stated start (August 2017).
  CRDB reading A (primary): its stated 12.5% / 14.5% already includes the buffer; reading B: it is the 10% / 12% minimum
  plus a 2.5% add-on and the buffer stacks (15% / 17% from FY2017).
- Headroom = ratio - requirement, Tier 1 and total capital, capital on the D2/D17 basis; the binding constraint is the
  one with less headroom. Internal capital generation = ROE x (1 - payout).
- Leverage decomposition: ln(E/TA) gap NMB - CRDB = ln(T1/RWA) gap + ln(RWA/TA) gap + ln(E/T1) gap.

## Sources and transformation trail

`data/processed/bank_year_wide.csv`, `capital_long.csv` (basis per D2/D17), `data/interim/bot_policy_text.csv`
(requirements, context sentences and the D-SIB search, `tzbank.extract.bot_policy.capital_requirements`), capital
raises from the equity statements (`CAPITAL_RAISES`, page-cited), `outputs/tables/ch05_coe_annual.csv`. Functions:
`payout`, `rore`, `rore_windows`, `rore_summary`, `capital_basis`, `requirements`, `requirement_statements`,
`dsib_search`, `capital`, `leverage_decomposition`, `distributable_capacity`, `slope_bootstrap`, `hypotheses`.

## Validation test

`tests/test_ch07.py`: RORE hand case; primary RORE is missing for every window with an unadjusted raise; the window
list reconciles with the RORE table (CRDB k = 5: 7 windows pass the raise rule, 6 computed); headroom uses the D2/D17
basis and the earlier CCB date; total headroom and binding constraint; readings A and B; the H7.2 bootstrap is
reproducible; payout hand case; leverage decomposition adds up.

## Results

### Capital raises (Observed)

{raises}

### Incremental returns (Derived)

CRDB headline: raise-adjusted RORE (primary alongside). k = 5 and k = 3; medians over all windows and excluding
stress-start windows (start year FY2015-18). Full window list with reasons: `ch07_rore_windows.csv`.

{rore}

{recon}

### Payout (Derived)

{payout}

### Capital requirements as stated (Observed)

{req_stated}

{req_ctx}

**D-SIB designation or capital surcharge (blueprint ch. 3).** {dsib}

**Conservation buffer start (D52 amended).** {ccb}

### Capital and headroom (Derived)

Headroom over the Tier 1 and total requirement (primary: reading A, CCB from August 2017); the binding constraint is
the smaller headroom. CRDB reading B in the last columns.

{capital}

{capital_txt}

### Leverage: constraint or choice? (Derived)

{lev}

### Distributable capacity, input to the 09a DDM (Scenario)

Tier 1 (primary):

{dist}

Total capital (09a sensitivity: {dist_why}):

{dist_total}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

RORE, dROE, payout, capital ratios, headroom and the leverage decomposition: **Derived**. Stated requirements, context
sentences and the D-SIB search: **Observed**. Distributable capacity under management buffers: **Scenario**.
Hypothesis verdicts: **Interpretation**.

## Limitations

- RORE is descriptive, not causal: earnings growth also reflects rates, credit cycles and acquisitions.
- Windows overlap; the H7.2 interval now uses the D46 moving-block bootstrap (block 3), which allows for the overlap but
  is still approximate with about 13-15 windows per bank. k = 3 windows are noisy around the 2015-18 trough.
- Under D52 a raise in the window's start year t-k does not exclude the window; base-year earnings then partly reflect
  the new capital (CRDB 2015-2020 and 2015-2018 windows); these windows are also stress-start windows.
- CRDB's reports do not say whether its 12.5% / 14.5% requirement includes the conservation buffer (reading A) or a
  bank-specific add-on on which the buffer stacks (reading B). Owner decision (D52 amended): A primary, B sensitivity.
- The NMB conservation-buffer start is stated inconsistently (Aug 2017 vs Aug 2018); the earlier date is primary
  (conservative headroom).
- Capital bases differ (NMB Bank-solo, D2; CRDB Group from FY2015, D17); dividends are declared amounts, not cash paid.
  CRDB's FY2019 total capital ratio and FY2009 DPS are not held (TO COLLECT).

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

RORE persistently above COE is consistent with a durable franchise (**low-cost deposits**, **scale economies**);
falling incremental returns with scale would point to **limited reinvestment opportunities** rather than eroding rent.
Leverage raises ROE without creating rent; where it is set by the regulatory constraint it is not a strategic
advantage.
"""


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_capital(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    t = tickers(root)
    cfg = settings(root)
    w = wide(root)
    rz = raises()
    pay = payout(w)
    rr = rore(w, pay, rz, _split_factor(root))
    coe = pd.read_csv(path("tables", root) / "ch05_coe_annual.csv")
    win = rore_windows(rr, pay, rz, coe)
    summ = rore_summary(win)
    cb = capital_basis(root)
    req = requirements(root, range(2008, 2026))
    stated = requirement_statements(root)
    dsib = dsib_search(root)
    cap = capital(w, cb, req, pay)
    lev = leverage_decomposition(cap)
    dist = distributable_capacity(cap, pay, "tier1")
    dist_t = distributable_capacity(cap, pay, "total")
    pc = cfg["persistence"]
    boot = {"reps": int(pc["bootstrap_reps"]), "block": int(pc["block_length"]), "seed": [int(cfg["random_seed"]), 72]}
    hyp = hypotheses(rr, win, cap, t, boot)
    tab, figd = path("tables", root), path("figures", root)
    tables = [(rz, "ch07_capital_raises.csv", "Capital raises from the equity statements", "Observed"),
              (rr, "ch07_rore.csv", "RORE and dROE (k = 3, 5), primary, adjusted, EPS-based", "Derived"),
              (win, "ch07_rore_windows.csv", "Every RORE window with included/excluded flags, reasons and the "
                                             "stress-start flag", "Derived"),
              (summ, "ch07_rore_summary.csv", "RORE, dROE and EPS-based medians, with and without stress-start "
                                              "windows", "Derived"),
              (pay, "ch07_payout.csv", "Payout: ordinary and special", "Derived"),
              (req, "ch07_capital_requirements.csv", "Tier 1 and total requirement history (stated, CCB dates, "
                                                     "readings A/B)", "Derived"),
              (stated, "ch07_requirement_statements.csv", "Requirements as stated in each report FY2015-25, with "
                                                          "page quotes and context", "Observed"),
              (dsib, "ch07_dsib_search.csv", "D-SIB designation / capital surcharge search record", "Observed"),
              (cap, "ch07_capital.csv", "Capital ratios, Tier 1 and total headroom, binding constraint, growth",
               "Derived"),
              (lev, "ch07_leverage.csv", "Leverage decomposition NMB - CRDB", "Derived"),
              (dist, "ch07_distributable_capacity.csv", "Distributable capacity for the 09a DDM (Tier 1)", "Scenario"),
              (dist_t, "ch07_distributable_capacity_total.csv", "Distributable capacity, total-capital sensitivity "
                                                                "for 09a", "Scenario"),
              (hyp, "ch07_hypotheses.csv", "H7.1-H7.3 verdicts", "Interpretation")]
    files = [write_csv(df, tab / name, require_evidence=True) for df, name, _, _ in tables]
    figs = []
    c5 = coe[coe["date_label"].str.startswith("FY")].set_index("year")
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in FOCUS:
        x = rr[(rr["bank_id"] == b) & (rr["k"] == 5)]
        ax.plot(x["fiscal_year"], x["rore_adjusted"] * 100, color=COLORS[b], lw=2, marker="o", ms=3,
                label=f"{t[b]} RORE k=5 (raise-adjusted)")
    ax.plot(c5.index, c5["coe"] * 100, color="#1f1f1d", lw=1.2, label="COE")
    ax.fill_between(c5.index, c5["coe_minus_200bp"] * 100, c5["coe_plus_200bp"] * 100, color="#e4e3dd",
                    label="COE +/- 200 bp")
    ax.set_xlim(2012.5, 2025.5)
    style(ax, "Return on retained equity (5-year windows) against the cost of equity", "percent")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch07_rore.png", "Evidence class: Derived. RORE = change in attributable profit / retained "
               "earnings over the window; CRDB windows with the FY2015 rights issue are raise-adjusted.")
    data = write_csv(rr[(rr["k"] == 5) & rr["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "rore_primary",
                                                                     "rore_adjusted", "evidence_class"]],
                     figd / "ch07_rore_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "RORE against COE", "evidence_class": "Derived"})
    fig, ax = plt_subplots(figsize=(8, 4.4))
    for b in FOCUS:
        x = cap[cap["bank_id"] == b]
        ax.plot(x["fiscal_year"], x["equity_to_assets"] * 100, color=COLORS[b], lw=2, label=f"{t[b]} equity / assets")
        # one headroom everywhere (prompt 21, D90): the smaller of CET1/Tier 1 - 12.5% and total - 14.5%
        ax.plot(x["fiscal_year"], x["binding_headroom"] * 100, color=COLORS[b], lw=1.2, ls="--",
                label=f"{t[b]} capital headroom (binding)")
    ax.axhline(0, color="#5a5a55", lw=0.8)
    ax.set_xlim(2008.5, 2025.5)
    style(ax, "Equity to assets and binding capital headroom", "percent")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch07_leverage.png", "Evidence class: Derived. Headroom: the smaller of (CET1 / Tier 1 ratio "
               "- 12.5%) and (total capital ratio - 14.5%), i.e. the disclosed minima 10.0% / 12.0% plus the 2.5-point "
               "conservation buffer from Aug 2017 (D52 amended; CRDB reading A; D90).")
    data = write_csv(cap[cap["bank_id"].isin(FOCUS)][["bank_id", "fiscal_year", "equity_to_assets", "tier1_headroom",
                                                      "total_headroom", "binding_headroom", "evidence_class"]],
                     figd / "ch07_leverage_data.csv", require_evidence=True)
    figs.append({"path": png, "data": data, "description": "Equity to assets, Tier 1 and total headroom",
                 "evidence_class": "Derived"})
    files += [f["path"] for f in figs] + [f["data"] for f in figs]
    # --- note text ---
    rz_txt = "\n".join(f"- {t[r.bank_id]} FY{r.fiscal_year}: TZS {r.amount / 1e9:,.1f}bn (source quotation: {r.kind}; "
                       f"{r.source})"
                       for r in rz.itertuples()) + "\n- NMB: no share issue in FY2008-25 (500m shares throughout)."
    order = list(SERIES)
    sm = summ.assign(bank=summ["bank_id"].map(t), o=summ["series"].map(order.index)).sort_values(["bank", "k", "o"])
    rore_md = _md(pd.DataFrame({
        "Bank": sm["bank"], "k": sm["k"], "Series": sm["label"], "Windows": sm["n_windows"],
        "Median": [pct(v) for v in sm["median"]], "Windows excl. stress start": sm["n_excl_stress_start"],
        "Median excl. stress start": [pct(v) for v in sm["median_excl_stress_start"]],
        "Latest": [f"{w_}: {pct(v)}" for w_, v in zip(sm["latest_window"], sm["latest"], strict=True)]}))
    rl = []
    for b in FOCUS:
        for k in (5, 3):
            x = win[(win["bank_id"] == b) & (win["k"] == k)]
            raised = x["primary_reason"].str.contains("(D52)", regex=False)
            rest = x[~raised]
            gap = rest[~rest["primary_included"]]
            rl.append(f"- {t[b]} k={k}: {len(x)} windows listed; {int(raised.sum())} excluded by the raise rule; "
                      f"{len(rest)} pass it; {int(x['primary_included'].sum())} computed (primary)"
                      + ("; not computed for data gaps: " + "; ".join(f"{r.window} ({r.primary_reason})"
                                                                      for r in gap.itertuples()) if len(gap) else "")
                      + f"; stress-start windows: {', '.join(x.loc[x['stress_start_flag'], 'window'])}.")
    recon = ("**Window reconciliation (task 4a).** D52's (t-k, t] rule excludes a window when a raise falls after its "
             "start year and up to its end year.\n\n" + "\n".join(rl) +
             "\n\nCRDB k = 5: the raise rule leaves 7 windows (2009-2014 and 2015-2020 to 2020-2025), as the owner "
             "counted; 6 are computed because 2009-2014 has no retained equity (the FY2009 DPS is not held, TO "
             "COLLECT). The earlier note reported 6 without saying why.")
    pl = []
    for b in FOCUS:
        x = pay[pay["bank_id"] == b].dropna(subset=["payout_ordinary"])
        last = x.iloc[-1]
        pl.append(f"- {t[b]}: mean payout 2010-25 {pct(x.loc[x['fiscal_year'] >= 2010, 'payout_total'].mean())}; "
                  f"FY{int(last['fiscal_year'])} ordinary {pct(last['payout_ordinary'])}"
                  + (f" plus special {pct(last['payout_special'])} (total {pct(last['payout_total'])})"
                     if pd.notna(last["payout_special"]) else "") + ".")
    st = stated[(stated["statement"] == "requirement as stated") & (stated["bank_id"] == "B01")]
    req_md = "CRDB, FY2015-25, exactly as stated in its own report:\n\n" + _md(pd.DataFrame({
        "FY": st["fiscal_year"], "Tier 1 (core) stated": [pct(v) for v in st["tier1_stated"]],
        "Total stated": [pct(v) for v in st["total_stated"]],
        "Source": [f"{f} p.{int(pg)}" if pd.notna(pg) else f for f, pg in zip(st["file"], st["pdf_page"], strict=True)],
        "Quote": ["'..." + str(q)[max(0, str(q).find("maintain a ratio") if "maintain a ratio" in str(q) else
                                     str(q).lower().find("ratio")):][:170].replace("|", "/") + "...'"
                  for q in st["quote"]]}))
    nm = stated[(stated["statement"] == "requirement as stated") & (stated["bank_id"] == "B02")]
    req_md += ("\n\nNMB states 10% (core) and 12% (total) in every report FY2015-25 ("
               f"{int(nm['tier1_stated'].notna().sum())} of 11 years found), plus the 2.5% conservation buffer from "
               "FY2017.")
    ctx = stated[stated["statement"] == "context of the requirement"]
    lg = pd.read_csv(path("processed", root) / "bank_period_long.csv", low_memory=False, dtype={"bank_id": str})
    wb = lg[(lg["bank_id"] == "B01") & lg["canonical_name"].isin(["capital_conservation_buffer",
                                                                    "bank_specific_buffer"])]
    wb = wb.sort_values("fiscal_year").drop_duplicates("canonical_name")
    wb_txt = (" Inputs workbook (audited primary; " + "; ".join(
        f"`{r.canonical_name}`, {r.source_document}: '{str(r.notes).split(' | ')[-1]}'" for r in wb.itertuples())
        + ").") if len(wb) else ""
    ctx_txt = "**What CRDB's 12.5% / 14.5% is.** " + " ".join(
        f"{t.get(r.bank_id, r.bank_id)} FY{r.fiscal_year} ({r.file} p.{int(r.pdf_page)}): '{str(r.quote)[:260]}'."
        for r in ctx.itertuples()) + (
        " Reading: the step from 10% / 12% to 12.5% / 14.5% equals the 2.5% conservation buffer, CRDB ties it to the "
        "August 2014 BoT regulation and its FY2017 report calls the levels a 'required buffer', and from FY2018 (buffer "
        "in force for every bank) CRDB still states 12.5% / 14.5%, not 15% / 17%. That points to reading A (minimum + "
        "buffer, applied from August 2015 in CRDB's reading against NMB's August 2017), but no report says so in terms "
        "or names an add-on. **Owner decision (D52 amended): reading A primary, reading B sensitivity.** Indirect "
        "evidence: under reading B "
        "CRDB would have been below its Tier 1 requirement in FY2017-18 (see the table below), yet it declared "
        "dividends for both years and reports no breach or capital-restoration plan, while the buffer rule quoted in "
        "NMB's reports bars dividends for a bank inside its buffer (Interpretation)." + wb_txt)
    ds = dsib[dsib["kind"] == "dsib_search"]
    hits = dsib[dsib["kind"] == "dsib_statement"]
    sc = {"BoT": "BoT Annual Reports", "B01": "CRDB reports", "B02": "NMB reports"}
    dsib_txt = "Searched every page for 'domestic systemically important', 'systemically important bank / financial " \
               "institution', 'D-SIB', 'capital surcharge' and 'systemic (risk) buffer': " + "; ".join(
                   f"{sc.get(r.scope, r.scope)} ({r.file.split('(')[-1].rstrip(')')}): {int(r.value)} passage(s)"
                   for r in ds.itertuples()) + ". "
    hu = hits.drop_duplicates(["scope", "report_fy", "pdf_page"])
    dsib_txt += " ".join(f"{sc.get(r.scope, r.scope)} FY{r.report_fy} ({r.file} p.{int(r.pdf_page)}): "
                         f"'...{str(r.quote)[150:330].strip()}...'." for r in hu.itertuples())
    desig = hits[(hits["scope"] == "B02") & hits["quote"].str.contains("designat", case=False, na=False)]
    crdb_hits = int(ds.loc[ds["scope"] == "B01", "value"].sum())
    dsib_txt += (
        "\n\n**Finding.** BoT issued a Framework for Identification of D-SIBs in 2021 (BoT Annual Report 2021/22). "
        + (f"NMB reports its designation as a D-SIB by BoT in December 2025 ({desig['file'].iloc[0]} "
           f"p.{', p.'.join(str(int(x)) for x in sorted(desig['pdf_page'].unique()))}) and 'anticipates additional "
           "capital buffer requirements', with no rate stated; earlier NMB reports (FY2023-24) describe the bank as "
           "systemically important without a designation. The designation post-dates FY2025 capital, so FY2015-25 "
           "headroom is unchanged; the surcharge rate is an open item for 09a (the 2% and 4% management-buffer rows "
           "can be read as surcharge scenarios). " if len(desig) else "No designation found for NMB. ")
        + ("CRDB: no D-SIB designation or capital surcharge in any of its reports (not found); this does not "
           "explain its 12.5% / 14.5% from FY2015, which pre-dates the 2021 framework." if crdb_hits == 0 else
           "CRDB: passages found, listed above."))
    r17 = req[(req["bank_id"] == "B02") & (req["fiscal_year"] == 2017)].iloc[0]
    c17 = cap[(cap["bank_id"] == "B02") & (cap["fiscal_year"] == 2017)].iloc[0]
    ccb_txt = (f"Primary start: {r17['ccb_start_primary']} (NMB FY2017 report), the earlier of the two stated dates, "
               f"because it gives the lower headroom. NMB FY2017 under both dates: Tier 1 {pct(c17['tier1_ratio'])} vs "
               f"{pct(r17['tier1_requirement'])} (Aug 2017 start, headroom {pct(c17['tier1_headroom'])}) or "
               f"{pct(r17['tier1_requirement_later_ccb'])} (Aug 2018 start, headroom "
               f"{pct(c17['tier1_headroom_later_ccb'])}); total {pct(c17['total_capital_ratio'])} vs "
               f"{pct(r17['total_requirement'])} (headroom {pct(c17['total_headroom'])}) or "
               f"{pct(r17['total_requirement_later_ccb'])} (headroom {pct(c17['total_headroom_later_ccb'])}). Only "
               "FY2017 differs. CRDB (reading A) is unaffected; under reading B the buffer stacks from FY2017.")
    cc = cap[cap["bank_id"].isin(FOCUS) & cap["fiscal_year"].between(2015, 2025)].copy()
    cc["bank"] = cc["bank_id"].map(t)
    cc = cc.sort_values(["bank", "fiscal_year"])
    cap_md = _md(pd.DataFrame({
        "Bank": cc["bank"], "FY": cc["fiscal_year"], "Tier 1": [pct(v) for v in cc["tier1_ratio"]],
        "T1 req.": [pct(v) for v in cc["tier1_requirement"]], "T1 headroom": [pct(v) for v in cc["tier1_headroom"]],
        "Total": [pct(v) for v in cc["total_capital_ratio"]], "Total req.": [pct(v) for v in cc["total_requirement"]],
        "Total headroom": [pct(v) for v in cc["total_headroom"]], "Binding": cc["binding_constraint"],
        "T1 headroom (B)": [pct(v) if b == "B01" else "= A" for v, b in zip(cc["tier1_headroom_reading_b"],
                                                                            cc["bank_id"], strict=True)],
        "Total headroom (B)": [pct(v) if b == "B01" else "= A" for v, b in zip(cc["total_headroom_reading_b"],
                                                                               cc["bank_id"], strict=True)],
        "Binding (B)": [v if b == "B01" else "= A" for v, b in zip(cc["binding_constraint_reading_b"], cc["bank_id"],
                                                                   strict=True)]}))
    late = cc[cc["fiscal_year"].between(2019, 2025)]
    tb = {t[b]: sorted(late.loc[(late["bank_id"] == b) & (late["binding_constraint"] == "total capital"),
                                "fiscal_year"].tolist()) for b in FOCUS}
    cl = []
    for b in FOCUS:
        x = cap[cap["bank_id"] == b].dropna(subset=["tier1_ratio"])
        last = x.iloc[-1]
        cl.append(f"- {t[b]} FY{int(last['fiscal_year'])} ({last['capital_basis']}): binding constraint "
                  f"{last['binding_constraint']} (headroom {pct(last['binding_headroom'])}); mean RWA growth 2019-25 "
                  f"{pct(x.loc[x['fiscal_year'] >= 2019, 'rwa_growth'].mean())} vs equity growth "
                  f"{pct(x.loc[x['fiscal_year'] >= 2019, 'equity_growth'].mean())}; internal capital generation "
                  f"{pct(last['internal_capital_generation'])}.")
    cl.append("- Total capital binds in FY2019-25 for " + "; ".join(
        f"{k}: {', '.join(f'FY{y}' for y in v) if v else 'no year'}" for k, v in tb.items())
              + ". CRDB FY2019 total capital ratio not held (binding constraint Tier 1 by default).")
    last = lev.iloc[-1]
    lev_txt = (f"- FY{int(last['fiscal_year'])}: equity / assets NMB {pct(last['nmb_equity_to_assets'])}, CRDB "
               f"{pct(last['crdb_equity_to_assets'])}; log gap {last['log_gap_equity_to_assets']:+.3f} = Tier 1 ratio "
               f"{last['log_gap_tier1_ratio']:+.3f} + RWA density {last['log_gap_rwa_density']:+.3f} + equity / Tier 1 "
               f"{last['log_gap_equity_to_tier1']:+.3f}.\n- Basis: NMB's equity and assets are Group while its Tier 1 "
               "and RWA are Bank-solo (D2), so for NMB the equity / Tier 1 term (and the RWA density term) include a "
               "Group-vs-Bank basis effect, not only capital composition; CRDB is Group on all four from FY2015 (D17)."
               f"\n- Tier 1 headroom FY{int(last['fiscal_year'])}: NMB {pct(last['nmb_tier1_headroom'])}, CRDB "
               f"{pct(last['crdb_tier1_headroom'])}; mean 2015-25 NMB "
               f"{pct(lev.loc[lev['fiscal_year'] >= 2015, 'nmb_tier1_headroom'].mean())}, CRDB "
               f"{pct(lev.loc[lev['fiscal_year'] >= 2015, 'crdb_tier1_headroom'].mean())}.")

    def _dl(d: pd.DataFrame, ex: str, tgt: str) -> str:
        return "\n".join(
            f"- {t[r['bank_id']]} FY{r['fiscal_year']}" + (f", reading {r['requirement_reading'][0]}"
                                                         if r["bank_id"] == "B01" else "")
            + f", buffer {r['management_buffer']:.0%}: target {pct(r[tgt])}, excess TZS {r[ex] / 1e9:,.0f}bn; "
            f"steady-state payout ceiling {pct(r['payout_max_steady_state'])} at RWA growth "
            f"{pct(r['rwa_growth_2019_25_mean'])} (actual payout {pct(r['payout_actual'])})"
            for _, r in d.iterrows())
    any_total = any(bool(v) for v in tb.values())
    dist_why = ("total capital binds in FY2019-25 for " + ", ".join(k for k, v in tb.items() if v)) if any_total else \
        "total capital does not bind in FY2019-25; shown for completeness"
    hyp_txt = []
    for r in hyp.itertuples():
        head = r.verdict.upper()
        if r.hypothesis == "H7.1":
            head = f"{r.verdict.upper()} jointly; by bank: {r.verdict_by_bank.replace('supported', 'SUPPORTED')}"
        hyp_txt.append(f"- **{r.hypothesis}: {head}.** {r.evidence}. Rule: {r.rule}.")
    s5 = summ[(summ["bank_id"] == "B01") & (summ["k"] == 5) & (summ["series"] == "rore_adjusted")].iloc[0]
    interp = ("**Interpretation.** NMB has reinvested retained earnings above its cost of equity "
              "(H7.1 supported for NMB). CRDB's raise-adjusted record is inconclusive: its 5-year windows run through "
              f"the 2015-18 trough and the FY2015 rights issue (median {pct(s5['median'])}; "
              f"{pct(s5['median_excl_stress_start'])} without stress-start windows, which leaves mostly the "
              "raise-adjusted pre-2015-start windows), while the windows starting after FY2018 are strong. "
              "The leverage decomposition separates a regulatory reading (CRDB runs close to its requirement, and "
              "total capital is the tighter constraint in most recent years for both banks) from a business-model "
              "reading; for NMB part of the equity / Tier 1 term is a basis effect. The distributable-capacity rows "
              "give 09a the payout ceiling under the Tier 1 and total-capital constraints, each management buffer and, "
              "for CRDB, both requirement readings.")
    text = NOTE.format(raises=rz_txt, rore=rore_md, recon=recon, payout="\n".join(pl), req_stated=req_md,
                       req_ctx=ctx_txt, dsib=dsib_txt, ccb=ccb_txt, capital=cap_md, capital_txt="\n".join(cl),
                       lev=lev_txt, dist=_dl(dist, "excess_tier1_tzs", "tier1_target"), dist_why=dist_why,
                       dist_total=_dl(dist_t, "excess_capital_tzs", "target_ratio"), hyp="\n".join(hyp_txt),
                       interp=interp)
    nf = write_text(text, path("notes", root) / "ch07.md")
    files.append(nf)
    m = {"chapter": "07", "title": "Capital allocation and incremental returns", "hypotheses": ["H7.1", "H7.2", "H7.3"],
         "status": "built",
         "tables": [{"path": rel(tab / name, root), "description": d, "evidence_class": c_, "note": rel(nf, root)}
                    for _, name, d, c_ in tables],
         "figures": [{"path": rel(f["path"], root), "data": rel(f["data"], root), "description": f["description"],
                      "evidence_class": f["evidence_class"], "note": rel(nf, root)} for f in figs],
         "notes": [rel(nf, root)],
         "inputs": ["data/processed/bank_year_wide.csv", "data/processed/capital_long.csv",
                    "data/interim/bot_policy_text.csv", "outputs/tables/ch05_coe_annual.csv"],
         "module": "tzbank.chapters.ch07_capital", "run_log_stage": "capital"}
    files.append(write_json(m, path("manifests", root) / "ch07.json"))
    return files
