"""Register approval by exception (owner, 09a review D): v1.0 = the proposed register with the review rules applied;
later edits are config changes (D55). Also the key-driver tornado that the owner reviews instead of every row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv, write_text
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.run import case, dry_run

APPROVED = "config/assumption_register.csv"
VERSION = "v1.0"
TOL = 1e-9
NO_IMPACT = {"ER_star", "CoR_ttc", "beta", "LP", "rim_w", "ddm_w", "payout_star", "k_req"}  # derived or not read


def freeze_register(root: Path | None = None, force: bool = False, version: str = VERSION,
                    status: str = "APPROVED v1.0 (owner 2026-11-09, by exception: proposal with the 09a review rules "
                                  "A-C; later edits are config changes, D55)") -> Path:
    """Write config/assumption_register.csv once (never overwrites an approved file unless ``force``)."""
    root = root or find_root()
    out = root / APPROVED
    if out.exists() and not force:
        return out
    _, reg, _ = dry_run(root)
    reg = reg.assign(version=version, set_on="2026-11-09", status=status,
                     evidence_class=np.where(reg["distribution"] == "derived", "Derived", "Scenario"))
    return write_csv(reg, out, require_evidence=True)


def load_register(root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((root or find_root()) / APPROVED)


def input_hash(root: Path | None = None) -> str:
    """sha256 over the approved register, settings and the processed inputs the engine reads (quality gate 09b)."""
    root = root or find_root()
    h = hashlib.sha256()
    for f in (APPROVED, "config/settings.yaml", "data/processed/bank_period_long.csv", "data/processed/macro.csv"):
        h.update((root / f).read_bytes())
    return h.hexdigest()[:16]


GROUPS = {"g_L": ("g_L", "g_D"), "g_D": ("g_L", "g_D")}  # deposits follow loans (D56): swapped together


def _swap(reg: pd.DataFrame, val: str, bank: str, sym: str, scen: str) -> pd.DataFrame:
    r = reg.copy()
    m = (r["valuation"] == val) & (r["bank"] == rg.TICK[bank]) & (r["symbol"].isin(GROUPS.get(sym, (sym,))))
    r.loc[m, "base"] = r.loc[m, scen]
    if sym in ("ROE_star", "g"):
        k = (r["valuation"] == val) & (r["bank"] == rg.TICK[bank])
        roe = float(r.loc[k & (r["symbol"] == "ROE_star"), "base"].iloc[0])
        g = float(r.loc[k & (r["symbol"] == "g"), "base"].iloc[0])
        r.loc[k & (r["symbol"] == "payout_star"), "base"] = 1 - g / roe
    return r


def key_drivers(root: Path | None = None, top: int = 15) -> list[Path]:
    """Tornado on the base value (both banks, both dates): each driver set to its bear and bull values alone."""
    root = root or find_root()
    reg = load_register(root)
    _, inputs = rg.build(root)
    rows = []
    for val in rg.DATES:
        for bank in ("B02", "B01"):
            base = un.value(case(reg, inputs, val, bank, "base", root))
            x = reg[(reg["valuation"] == val) & (reg["bank"] == rg.TICK[bank])]
            for sym, g in x.groupby("symbol"):
                if sym == "g_D":
                    continue  # reported with g_L (grouped)
                if sym in NO_IMPACT or (g["bear"] == g["base"]).all() and (g["bull"] == g["base"]).all():
                    continue
                v = {}
                for s in ("bear", "bull"):
                    c = case(_swap(reg, val, bank, sym, s), inputs, val, bank, "base", root)
                    v[s] = un.value(c) / base - 1
                g = g.sort_values("period")
                rows.append({"valuation": val, "bank": rg.TICK[bank], "symbol": "g_L+g_D" if sym == "g_L" else sym,
                             "parameter": "Loan and deposit growth" if sym == "g_L" else g["parameter"].iloc[0],
                             "bear_impact": v["bear"], "bull_impact": v["bull"],
                             "swing": abs(v["bull"] - v["bear"]),
                             "base": _fmt(g["base"]), "bear": _fmt(g["bear"]), "bull": _fmt(g["bull"]),
                             "rationale": g["rationale"].iloc[0], "evidence_class": "Scenario"})
    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    rank = df.groupby("symbol")["swing"].mean().sort_values(ascending=False)
    keep = list(rank.index[:top])
    lines = ["# Register v1.0: key drivers (approval by exception)", "",
             f"Tornado on the base blended value: each driver alone set to its bear and to its bull register values, "
             f"all other drivers at base (`{APPROVED}`, {VERSION}). The {len(keep)} drivers with the largest mean swing "
             "across both banks and both dates are listed; the full table is `outputs/valuation/register_tornado.csv`. "
             "Paths show the first -> last explicit year.", ""]
    for sym in keep:
        d = df[df["symbol"] == sym]
        lines += [f"## {d['parameter'].iloc[0]} (`{sym}`), mean swing {rank[sym]:.1%}", "",
                  "| Date | Bank | Bear | Base | Bull | Value impact bear / bull |", "| --- | --- | --- | --- | --- | --- |"]
        for r in d.sort_values(["valuation", "bank"]).itertuples():
            lines.append(f"| {r.valuation} | {r.bank} | {r.bear} | {r.base} | {r.bull} | {r.bear_impact:+.1%} / "
                         f"{r.bull_impact:+.1%} |")
        lines += ["", f"Rationale: {d['rationale'].iloc[0]}", ""]
    vdir = path("outputs", root) / "valuation"
    return [write_csv(df, vdir / "register_tornado.csv", require_evidence=True),
            write_text("\n".join(lines) + "\n", vdir / "register_key_drivers.md")]


def ordering_check(tornado: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """D56 ordering rule (owner, ch09-11 review A): for every value-driving input, the bear value alone must not raise
    and the bull value alone must not lower the base value (bear <= base <= bull in value terms). Rows that break it
    are returned with status FAIL; the stage stops on any FAIL."""
    t = tornado.copy()
    t["ordering_ok"] = (t["bear_impact"] <= tol) & (t["bull_impact"] >= -tol)
    pol = t["symbol"].isin(POLICY)
    t["status"] = np.where(pol, "EXEMPT (policy driver; direction reported)", np.where(t["ordering_ok"], "PASS", "FAIL"))
    t["measured_direction"] = np.select(
        [t["ordering_ok"], (t["bear_impact"] > tol) & (t["bull_impact"] < -tol)],
        ["as labelled (bear lowers, bull raises value)", "reversed (bear raises, bull lowers value)"],
        "mixed (both move value the same way)")
    return t


POLICY = {"g_L+g_D", "g", "pi_max"}  # D56 (owner, ch09-11 review): growth paths, terminal g, payout cap exempt


def scenario_ordering(ctx) -> pd.DataFrame:
    """D56 scenario-level check (owner): bear value <= base value <= bull value per bank and date; stop on violation."""
    rows = []
    for val in rg.DATES:
        for bank in ("B02", "B01"):
            v = {s: un.value(ctx.cases[(val, bank, s)]) for s in ("bear", "base", "bull")}
            ok = v["bear"] <= v["base"] + TOL and v["base"] <= v["bull"] + TOL
            rows.append({"valuation": val, "bank": rg.TICK[bank], "check": "scenario values: bear <= base <= bull",
                         "status": "PASS" if ok else "FAIL", "bear_to_base": v["bear"] / v["base"] - 1,
                         "bull_to_base": v["bull"] / v["base"] - 1, "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def _fmt(s: pd.Series) -> str:
    v = s.astype(float).to_numpy()
    return f"{v[0]:.4f}" if len(v) == 1 or np.allclose(v, v[0]) else f"{v[0]:.4f} -> {v[-1]:.4f}"
