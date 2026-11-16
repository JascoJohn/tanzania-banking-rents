"""Accounting identity checks (prompt 01 section 6.2; CLAUDE.md section 4 'tests must cover accounting identities').

Each identity is tested within one vintage so restatements are not mixed: ``original`` uses first-published rows,
``latest`` uses the latest restatement (forecast base at the current valuation date). Results are non-critical:
failures are listed in the validation report, never silently corrected.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = ["identity", "bank_id", "period_end", "period_type", "entity", "basis", "vintage", "lhs", "rhs",
                    "difference", "relative_difference", "tolerance", "tolerance_type", "status", "flag_reason",
                    "inputs", "evidence_class"]
KEYS = ["bank_id", "entity", "basis", "period_type", "period_end"]
_BANK_LEVEL = re.compile(r"bank-level|bank level|bank-only|bank only", re.I)


@dataclass(frozen=True)
class Identity:
    """lhs(row) ≈ rhs(row) within tolerance."""

    name: str
    inputs: tuple[str, ...]
    lhs: Callable[[pd.DataFrame], pd.Series]
    rhs: Callable[[pd.DataFrame], pd.Series]
    tolerance: float
    tolerance_type: str = "relative"  # or "absolute"


def identities(tol: dict[str, float]) -> list[Identity]:
    """The seven identities of prompt 01 section 6.2."""
    return [
        Identity("interest income - interest expense = NII", ("interest_income", "interest_expense",
                                                               "net_interest_income"),
                 lambda d: d["interest_income"] - d["interest_expense"], lambda d: d["net_interest_income"],
                 tol["nii"]),
        Identity("PBT - tax = net profit", ("profit_before_tax", "income_tax_expense", "net_profit"),
                 lambda d: d["profit_before_tax"] - d["income_tax_expense"], lambda d: d["net_profit"], tol["pbt_tax"]),
        Identity("attributable profit + NCI profit = net profit", ("profit_attributable", "profit_nci", "net_profit"),
                 lambda d: d["profit_attributable"] + d["profit_nci"], lambda d: d["net_profit"], tol["profit_split"]),
        Identity("attributable equity + NCI = total equity", ("equity_attributable", "nci_equity", "total_equity"),
                 lambda d: d["equity_attributable"] + d["nci_equity"], lambda d: d["total_equity"],
                 tol["equity_split"]),
        Identity("EPS x weighted shares = attributable profit", ("eps_basic", "weighted_avg_shares",
                                                                  "profit_attributable"),
                 lambda d: d["eps_basic"] * d["weighted_avg_shares"], lambda d: d["profit_attributable"], tol["eps"]),
        Identity("gross loans - allowance = net loans", ("gross_loans", "loan_loss_allowance", "net_loans"),
                 lambda d: d["gross_loans"] - d["loan_loss_allowance"], lambda d: d["net_loans"], tol["loans"]),
        Identity("Tier 1 / RWA = Tier 1 ratio", ("tier1_capital", "rwa_total", "tier1_ratio"),
                 lambda d: d["tier1_capital"] / d["rwa_total"], lambda d: d["tier1_ratio"], tol["tier1_ratio_pp"],
                 "absolute"),
    ]


def vintage_frame(canon: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Wide frame (one row per bank/entity/basis/period) of inputs-workbook values for one vintage.

    Phase 1 EPS is joined for the EPS identity (the inputs workbook holds no FY EPS).
    """
    iw = canon[(canon["provenance"] == "inputs_workbook") & canon["value"].notna()]
    if vintage == "original":
        iw = iw[~iw["is_restated"]]
    elif vintage == "latest":
        iw = iw[iw["forecast_base_current"]]
    else:
        raise ValueError(vintage)
    iw = iw.sort_values("publication_date").drop_duplicates([*KEYS, "canonical_name"], keep="last")
    wide = iw.pivot_table(index=KEYS, columns="canonical_name", values="value", aggfunc="first")
    notes = iw[iw["canonical_name"] == "gross_loans"].set_index(KEYS)["notes"].rename("gross_loans_notes")
    wide = wide.join(notes, how="left").reset_index()
    eps = canon[(canon["canonical_name"] == "eps_basic") & canon["value"].notna() & (canon["period_type"] == "FY")]
    eps = eps.drop_duplicates(["bank_id", "period_end"]).set_index(["bank_id", "period_end"])["value"].rename(
        "eps_basic")
    wide = wide.join(eps, on=["bank_id", "period_end"], how="left") if "eps_basic" not in wide else wide
    return wide


def run_identities(canon: pd.DataFrame, tol: dict[str, float], vintages: tuple[str, ...] = ("original", "latest"),
                   ) -> pd.DataFrame:
    """Evaluate every identity on every bank-period where all its inputs exist."""
    rows = []
    for vintage in vintages:
        wide = vintage_frame(canon, vintage)
        for ident in identities(tol):
            if any(c not in wide.columns for c in ident.inputs):
                continue
            sub = wide.dropna(subset=list(ident.inputs))
            if ident.name.startswith("EPS"):
                sub = sub[(sub["period_type"] == "FY") & (sub["entity"] == "Group")]
            if sub.empty:
                continue
            lhs, rhs = ident.lhs(sub), ident.rhs(sub)
            diff = lhs - rhs
            rel = diff / rhs.abs().replace(0, np.nan)
            measure = diff.abs() if ident.tolerance_type == "absolute" else rel.abs()
            ok = measure <= ident.tolerance + 1e-12
            status = np.where(ok, "PASS", "FAIL")
            flag_reason = pd.Series([None] * len(sub), index=sub.index, dtype=object)
            if ident.name.startswith("gross loans"):
                bank_level = sub["gross_loans_notes"].fillna("").str.contains(_BANK_LEVEL) | sub["basis"].eq("BANK")
                status = np.where(~ok & bank_level.to_numpy(), "FLAG", status)
                flag_reason[bank_level & ~ok] = "Bank-level gross loans (flagged, not failed)"
            if ident.name.startswith("EPS"):
                flag_reason[:] = "EPS from Phase 1 (inputs workbook holds no FY EPS); as reported, not split-adjusted"
            for i, r in sub.iterrows():
                rows.append({
                    "identity": ident.name, "bank_id": r["bank_id"], "period_end": r["period_end"],
                    "period_type": r["period_type"], "entity": r["entity"], "basis": r["basis"], "vintage": vintage,
                    "lhs": lhs[i], "rhs": rhs[i], "difference": diff[i], "relative_difference": rel[i],
                    "tolerance": ident.tolerance, "tolerance_type": ident.tolerance_type,
                    "status": status[sub.index.get_loc(i)], "flag_reason": flag_reason[i],
                    "inputs": ";".join(ident.inputs), "evidence_class": "Derived",
                })
    out = pd.DataFrame(rows, columns=IDENTITY_COLUMNS)
    return out.sort_values(["identity", "bank_id", "period_end", "period_type", "basis", "vintage"]).reset_index(
        drop=True)


def summarise(checks: pd.DataFrame) -> pd.DataFrame:
    """Pass rates by identity (original vintage, Group entity)."""
    sub = checks[(checks["vintage"] == "original")]
    g = sub.groupby("identity")["status"]
    return pd.DataFrame({"tested": g.size(), "pass": g.apply(lambda s: (s == "PASS").sum()),
                         "fail": g.apply(lambda s: (s == "FAIL").sum()),
                         "flag": g.apply(lambda s: (s == "FLAG").sum())}).assign(
        pass_rate=lambda d: d["pass"] / d["tested"]).reset_index()
