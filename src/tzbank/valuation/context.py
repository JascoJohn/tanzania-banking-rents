"""Shared run context for chapters 9-11 (prompt 09b): the approved register, point-in-time inputs, one Case per bank,
date and scenario, the input hash, and the seeded Monte Carlo (computed once, reused by chapters 9 and 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, settings
from tzbank.valuation import register as rg
from tzbank.valuation import uncertainty as un
from tzbank.valuation.approve import input_hash, load_register
from tzbank.valuation.run import SCEN, case

BANKS = ("B02", "B01")


@dataclass
class Ctx:
    root: Path
    reg: pd.DataFrame
    inputs: dict
    cases: dict
    version: str
    hash: str
    mc: dict = field(default_factory=dict)

    def stamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every published value carries the register version and input hash (quality gate)."""
        return df.assign(register_version=self.version, input_hash=self.hash)

    def monte_carlo(self, val: str, bank: str, centred: bool = True) -> pd.DataFrame:
        """Seeded Monte Carlo. ``centred`` (primary, owner 2026-11-09): every ER* draw is shifted by (base ER* - median
        of the break-adjusted ER* draws), keeping the paired lambda, the spread and the correlation; the unshifted run
        is the labelled 'bull-centred' sensitivity."""
        key = (val, bank) if centred else (val, bank, "bull-centred")
        if key not in self.mc:
            st = settings(self.root)
            vc = st["valuation"]
            cs = {s: self.cases[(val, bank, s)] for s in SCEN}
            coe_sh = (cs["bull"].r - cs["base"].r - (0.0 - 0.015), cs["bear"].r - cs["base"].r - 0.015)
            bounds = un.mc_bounds(cs["base"].dr, cs["bear"].dr, cs["bull"].dr, (min(coe_sh), max(coe_sh)),
                                  (-0.015, 0.015))
            seed = [int(st["random_seed"]), 11, list(rg.DATES).index(val), BANKS.index(bank)]
            pd_ = self.persistence_draws(val, bank)
            if centred:
                pd_ = pd_.assign(er_star=pd_["er_star"] + self.er_shift(val, bank))
            self.mc[key] = un.monte_carlo(cs["base"], bounds, vc["mc_correlation"], int(vc["mc_draws"]), seed,
                                          persist=pd_, coe_base=cs["base"].r)
        return self.mc[key]

    def er_shift(self, val: str, bank: str) -> float:
        """Base ER* (register) minus the median of the break-adjusted bootstrap ER* draws."""
        base_er = self.cases[(val, bank, "base")].dr.roe_star - self.cases[(val, bank, "base")].r
        return float(base_er - self.persistence_draws(val, bank)["er_star"].median())

    def persistence_draws(self, val: str, bank: str) -> pd.DataFrame:
        """Chapter 5 Kilian (1998) bias-corrected bootstrap replicates (D47; D58 amended, owner issue 25) of the
        break-adjusted (lambda, ER*) on the ER eligible at the date (31 Dec 2025: FY<=2024 re-estimation; 31 Oct 2026:
        the chapter 5 sample). Kept: 0 <= lambda < 1 and -10% <= ER* <= 30% (the replicates outside are lambda >= 1,
        where ER* is undefined, or explosive). `valuation.mc_persistence_method` = kilian (default) | moving_block."""
        key = ("persist", val, bank)
        if key not in self.mc:
            from tzbank.chapters.ch05 import models as m5

            st = settings(self.root)
            pc = st["persistence"]
            if val == "HIST":
                er = self.inputs[(val, bank, "persistence")]["er"]
            else:
                e = pd.read_csv(self.root / "outputs" / "tables" / "ch05_excess_returns.csv", dtype={"bank_id": str})
                e = e[(e["bank_id"] == bank) & e["in_estimation"].astype(bool)]
                er = e.set_index("fiscal_year")["er"].dropna()
            base = m5.pairs(er)
            dm = pd.DataFrame({"d_episode": base["year"].between(2015, 2018).astype(int).to_numpy()},
                              index=base["year"].to_numpy())
            d = m5.pairs(er, dm[["d_episode"]])
            seed = [int(st["random_seed"]), 12, list(rg.DATES).index(val), BANKS.index(bank)]
            if st["valuation"].get("mc_persistence_method", "kilian") == "kilian":
                reps = int(pc["bootstrap_reps"])
                bs = m5.kilian_draws(d, reps, reps, seed, ["d_episode"], [0.0])
            else:
                bs = m5.bootstrap(d, int(pc["bootstrap_reps"]), int(pc["block_length"]), seed, ["d_episode"], [0.0])
            ok = bs["lambda"].between(0, 0.999999) & bs["er_star"].between(-0.10, 0.30)
            self.mc[key] = bs[ok].reset_index(drop=True).assign(kept_share=float(ok.mean()))
        return self.mc[key]


def build_ctx(root: Path | None = None) -> Ctx:
    root = root or find_root()
    reg = load_register(root)
    _, inputs = rg.build(root)
    cases = {(v, b, s): case(reg, inputs, v, b, s, root) for v in rg.DATES for b in BANKS for s in SCEN}
    return Ctx(root, reg, inputs, cases, str(reg["version"].iloc[0]), input_hash(root))


def shares_post_split(ctx: Ctx, val: str, bank: str) -> float:
    """Shares on the post-split basis (NMB 1:10 effective 24 Aug 2026, config/corporate_actions.csv)."""
    ca = pd.read_csv(ctx.root / "config" / "corporate_actions.csv")
    f = {r.bank_id: float(r.ratio) for r in ca.itertuples() if r.action == "split"}.get(bank, 1.0)
    c = ctx.cases[(val, bank, "base")]
    return c.shares * (f if val == "HIST" else 1.0)


def pct(v: float, d: int = 1) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}%}"
