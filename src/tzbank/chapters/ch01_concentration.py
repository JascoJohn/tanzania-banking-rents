"""Chapter 1: concentration and market position (blueprint §1, §4 chapter 1; H1.1, H1.2; limitation L1).

CR2 = (NMB + CRDB) / commercial-bank sector total for assets, deposits and loans (BoT A3.6 / A3.12, December, after
the series crosswalk of prompt 10), annually. Profit: the BoT publishes no sector profit level, only the sector return
on assets (financial soundness indicators, end-June, FY2019-25; narrative ROA FY2011-14), so the profit share is a
labelled derived series: profit share = asset share x (CR2 ROA / sector ROA), and profit share - asset share = asset
share x (CR2 ROA / sector ROA - 1) (Estimated). HHI needs every bank's share; only a lower bound is reported from the 7
banks held (ch01-03 review C). DSE: market-cap and turnover shares of NMB + CRDB from the DSE market report held (30 Oct 2026)
and the daily trading data. Bank-level and comparative statements only (L1).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, FOCUS, avg, col, pct, plt_subplots, save, style, tickers, wide
from tzbank.config import find_root, path, settings
from tzbank.io import rel, write_csv, write_json, write_text

SECTOR = {"assets": "BOTDB_BANK_ASSETS__TOTAL_ASSET", "deposits": "BOTDB_BANK_DEPOSITS__TOTAL_DEPOSIT",
          "loans": "BOTDB_BANK_ASSETS__LOAN_BILL"}
BANKVAR = {"assets": "total_assets", "deposits": "customer_deposits", "loans": "gross_loans"}
DSE_DOC = "dse/30_Oct_2026_DSE_Mkt_Info.pdf"


def shares(root: Path) -> pd.DataFrame:
    s = pd.read_csv(path("processed", root) / "sector_totals.csv")
    w = wide(root)
    rows = []
    for k, sid in SECTOR.items():
        sec = s[s["series_id"] == sid].set_index("year")["value"] * 1e6  # TZS million -> TZS
        f = w[w["bank_id"].isin(FOCUS)].pivot(index="fiscal_year", columns="bank_id", values=BANKVAR[k])
        for y, tot in sec.items():
            if y not in f.index or f.loc[y].isna().any():
                continue
            rows.append({"year": int(y), "measure": k, "sector_tzs": tot, "nmb_tzs": f.loc[y, "B02"],
                         "crdb_tzs": f.loc[y, "B01"], "nmb_share": f.loc[y, "B02"] / tot,
                         "crdb_share": f.loc[y, "B01"] / tot, "cr2": (f.loc[y, "B02"] + f.loc[y, "B01"]) / tot,
                         "sector_series": sid, "bank_basis": "Group (CRDB includes Burundi and DRC subsidiaries)",
                         "sector_basis": "commercial banks, BoT A3.6 / A3.12, December", "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def hhi_bound(root: Path) -> pd.DataFrame:
    """HHI lower bound (ch01-03 review C), 0-10,000 scale. Held banks' squared shares are summed (bound A); the
    unheld remainder R = 1 - sum of held shares contributes at least R^2 / (N - n) (equal split minimises the sum of
    squares; N = all licensed banks, which only lowers the bound), giving bound B. Neither is the HHI."""
    s = pd.read_csv(path("processed", root) / "sector_totals.csv")
    nb = s[s["series_id"] == "BOTDB_SECTOR_STRUCTURE__LICENSED_BANK_ALL_BANK_TOTAL"].set_index("year")["value"]
    w = wide(root)
    rows = []
    for k, sid in SECTOR.items():
        sec = s[s["series_id"] == sid].set_index("year")["value"] * 1e6
        f = w.pivot(index="fiscal_year", columns="bank_id", values=BANKVAR[k])
        for y, tot in sec.items():
            if y not in f.index or f.loc[y, FOCUS].isna().any():
                continue
            sh = (f.loc[y].dropna() / tot)
            held = float(sh.sum())
            n_lic = nb.get(y)
            rest = (1 - held) ** 2 / (n_lic - len(sh)) if n_lic and n_lic > len(sh) and held < 1 else 0.0
            rows.append({"year": int(y), "measure": k, "banks_held": len(sh),
                         "banks_held_ids": " ".join(sorted(sh.index)), "held_share": held,
                         "hhi_lower_bound_held_only": 1e4 * float((sh ** 2).sum()),
                         "licensed_banks": n_lic, "hhi_lower_bound": 1e4 * (float((sh ** 2).sum()) + rest),
                         "label": "LOWER BOUND, not the HHI: unheld banks' shares unknown",
                         "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def profit_gap(root: Path, sh: pd.DataFrame) -> pd.DataFrame:
    """Profit share - asset share from the sector ROA (Estimated; see module docstring)."""
    b = pd.read_csv(path("interim", root) / "bot_long.csv", low_memory=False)
    roa = b[b["series_id"].isin(["BOTDB_FINANCIAL_SOUNDNESS__RETURN_ON_ASSET_FY_JUNE",
                                 "BOTDB_NARRATIVE__BANKING_SECTOR_RETURN_ON_ASSET__PCT_FY_JUNE",
                                 "BOTDB_NARRATIVE__RETURN_ON_ASSET__PCT_FY_JUNE"])].copy()
    roa["year"] = pd.to_datetime(roa["period_end"]).dt.year
    roa["value"] = pd.to_numeric(roa["value"], errors="coerce")
    roa = roa.sort_values("series_id").drop_duplicates("year").set_index("year")
    w = wide(root)
    f = w[w["bank_id"].isin(FOCUS)].copy()
    f["avg_ta"] = avg(f, "total_assets").to_numpy()
    g = f.groupby("fiscal_year")[["profit_before_tax", "net_profit", "avg_ta"]].sum(min_count=2)
    a = sh[sh["measure"] == "assets"].set_index("year")["cr2"]
    rows = []
    for y, r in roa.iterrows():
        if y not in g.index or y not in a.index or not np.isfinite(r["value"]):
            continue
        cr2_pre, cr2_post = g.loc[y, "profit_before_tax"] / g.loc[y, "avg_ta"], g.loc[y, "net_profit"] / g.loc[y, "avg_ta"]
        rows.append({"year": int(y), "sector_roa": r["value"], "sector_roa_source": r["series_id"],
                     "cr2_roa_pretax": cr2_pre, "cr2_roa_posttax": cr2_post, "asset_share": a[y],
                     "profit_share_pretax": a[y] * cr2_pre / r["value"],
                     "profit_minus_asset_share_pretax": a[y] * (cr2_pre / r["value"] - 1),
                     "profit_minus_asset_share_posttax": a[y] * (cr2_post / r["value"] - 1),
                     "note": "BoT sector ROA is end-June; banks' ROA is the calendar fiscal year; BoT's ROA definition "
                             "(pre- or post-tax) is not stated in the extract: both CR2 versions shown",
                     "evidence_class": "Estimated"})
    return pd.DataFrame(rows).sort_values("year", ignore_index=True)


def dse_report(root: Path) -> pd.DataFrame:
    """Market totals from the DSE market report held (extracted from the PDF when present; else the interim copy)."""
    out = path("interim", root) / "dse_market_report.csv"
    pdf = path("documents", root) / DSE_DOC
    if not pdf.exists():
        return pd.read_csv(out) if out.exists() else pd.DataFrame()
    from tzbank.extract.pdftext import _doc

    d = _doc(str(pdf))
    pages = [re.sub(r"\s+", " ", d[i].get_text()) for i in range(d.page_count)]
    txt = " ".join(pages)
    rows = []

    def num(x: str) -> float:
        return float(x.replace(",", ""))
    m = re.search(r"Total Market Capitalisation \(TZS bln\) ([\d,.]+) ([\d,.]+) ([\d,.]+)", txt)
    dates = ["2026-10-30", "2026-09-30", "2025-10-31"]
    for name, rx in (("total_market_cap", r"Total Market Capitalisation \(TZS bln\) ([\d,.]+) ([\d,.]+) ([\d,.]+)"),
                     ("domestic_market_cap", r"Domestic Market Capitalisation \(TZS bln\) ([\d,.]+) ([\d,.]+) ([\d,.]+)")):
        m = re.search(rx, txt)
        if m:
            for dt, v in zip(dates, m.groups(), strict=True):
                rows.append({"item": name, "entity": "DSE", "date": dt, "value_tzs": num(v) * 1e9})
    m = re.search(r"Equities Market Turnover \(in million TZS\).*?Total Turnover ([\d,.]+) ([\d,.]+) ([\d,.]+) ([\d,.]+)",
                  txt)
    if m:
        for per, v in zip(["2026-10-30 (day)", "Oct-Dec 2026 (to date)", "Jul-Sep 2026", "Oct-Dec 2025"], m.groups(),
                          strict=True):
            rows.append({"item": "equity_turnover", "entity": "DSE", "date": per, "value_tzs": num(v) * 1e6})
    for tk in ("NMB", "CRDB"):
        m = re.search(rf"\b{tk} [\d,]+ [\d,]+ [\d,]+ [\d,]+ ([\d,]+) [\d,]+ [\d,]+ ([\d,.]+) ", txt)
        if m:
            rows.append({"item": "counter_market_cap", "entity": tk, "date": "2026-10-30", "value_tzs": num(m.group(2)) * 1e9})
            rows.append({"item": "counter_turnover", "entity": tk, "date": "2026-10-30 (day)",
                         "value_tzs": num(m.group(1))})
    df = pd.DataFrame(rows).assign(source=DSE_DOC, evidence_class="Observed")
    write_csv(df, out, require_evidence=True)
    return df


def dse_shares(root: Path, rep: pd.DataFrame) -> pd.DataFrame:
    md = pd.read_csv(path("processed", root) / "market_daily.csv", parse_dates=["date"])
    r = rep.set_index(["item", "entity", "date"])["value_tzs"]
    rows = []
    cap = sum(r.get(("counter_market_cap", t, "2026-10-30"), np.nan) for t in ("NMB", "CRDB"))
    for base in ("total_market_cap", "domestic_market_cap"):
        rows.append({"measure": f"market cap share of {base.replace('_', ' ')}", "period": "2026-10-30",
                     "cr2_value_tzs": cap, "market_value_tzs": r.get((base, "DSE", "2026-10-30")),
                     "share": cap / r.get((base, "DSE", "2026-10-30"), np.nan)})
    for per, (a, b) in (("Jul-Sep 2026", ("2026-07-01", "2026-09-30")), ("Oct-Dec 2025", ("2025-10-01", "2025-12-31"))):
        x = md[md["date"].between(a, b)]
        t = float(x["turnover"].sum())
        rows.append({"measure": "equity turnover share", "period": per, "cr2_value_tzs": t,
                     "market_value_tzs": r.get(("equity_turnover", "DSE", per)),
                     "share": t / r.get(("equity_turnover", "DSE", per), np.nan)})
    day = sum(r.get(("counter_turnover", t, "2026-10-30 (day)"), np.nan) for t in ("NMB", "CRDB"))
    rows.append({"measure": "equity turnover share", "period": "2026-10-30 (day)", "cr2_value_tzs": day,
                 "market_value_tzs": r.get(("equity_turnover", "DSE", "2026-10-30 (day)")),
                 "share": day / r.get(("equity_turnover", "DSE", "2026-10-30 (day)"), np.nan)})
    out = pd.DataFrame(rows)
    out["source"] = f"{DSE_DOC}; data/processed/market_daily.csv"
    out["evidence_class"] = "Derived"
    return out


def hypotheses(sh: pd.DataFrame, pg: pd.DataFrame, ds: pd.DataFrame, basis: str = "pretax") -> pd.DataFrame:
    """H1.1 on the configured CR2 ROA basis (`part1.sector_roa_basis`, open issue 26); the other basis is reported."""
    alt = "posttax" if basis == "pretax" else "pretax"
    pos = pg[f"profit_minus_asset_share_{basis}"] > 0
    pos_alt = pg[f"profit_minus_asset_share_{alt}"] > 0
    v1 = "supported" if len(pg) >= 5 and pos.all() else "rejected" if len(pg) and (~pos).mean() >= 0.5 else "inconclusive"
    if v1 == "supported" and not pos_alt.all():  # D59 amended: the verdict names its profit basis
        v1 = f"supported {basis.replace('tax', '-tax')}; depends on profit basis (issue 26)"
    a = sh[sh["measure"] == "assets"].sort_values("year")
    last = a.iloc[-1]
    dom = ds[ds["measure"].str.contains("domestic")]["share"].iloc[0]
    turn = ds[(ds["measure"] == "equity turnover share") & (ds["period"] == "Jul-Sep 2026")]["share"].iloc[0]
    v2 = "supported" if dom > last["cr2"] and turn > last["cr2"] else "rejected" if dom < last["cr2"] and \
        turn < last["cr2"] else "inconclusive"
    return pd.DataFrame([
        {"hypothesis": "H1.1", "verdict": v1,
         "evidence": f"profit share - asset share ({basis} CR2 ROA vs sector ROA) positive in {int(pos.sum())} of "
                     f"{len(pg)} years with sector ROA ({', '.join(str(y) for y in pg['year'])}); range "
                     f"{pct(pg[f'profit_minus_asset_share_{basis}'].min())} to "
                     f"{pct(pg[f'profit_minus_asset_share_{basis}'].max())}; NOT robust to the {alt} basis: positive in "
                     f"{int(pos_alt.sum())} of {len(pg)} years (negative in "
                     f"{', '.join(str(y) for y in pg.loc[~pos_alt, 'year']) or 'none'}); the BoT FSI ROA basis is "
                     "unconfirmed (open issue 26)",
         "rule": "supported when the profit share exceeds the asset share in every year with sector ROA and at least 5 "
                 "such years; rejected when it falls short in half or more; descriptive (L1)"},
        {"hypothesis": "H1.2", "verdict": v2,
         "evidence": f"NMB + CRDB: {pct(dom)} of DSE domestic market cap (30 Oct 2026) and {pct(turn)} of equity "
                     f"turnover (Jul-Sep 2026) vs {pct(last['cr2'])} of commercial-bank assets ({int(last['year'])})",
         "rule": "supported when both the DSE market-cap share (domestic) and the turnover share exceed the latest CR2 "
                 "asset share; rejected when both fall short"}]).assign(evidence_class="Interpretation")


NOTE = """# Chapter 1: concentration and market position

Generated by `tzbank.chapters.ch01_concentration` (stage `part1`). Blueprint §4 chapter 1; H1.1, H1.2; limitation L1:
bank-level and comparative statements only, no economy-wide causal claim.

## Research question

How large are NMB and CRDB in Tanzanian banking and on the DSE, and do they earn a larger share of sector profit than
their share of sector assets?

## Variable definitions

CR2 = (NMB + CRDB) / commercial-bank sector total (assets, deposits, gross loans; BoT A3.6 / A3.12, December values
after the BoT series crosswalk). Profit share = asset share x (CR2 ROA / sector ROA) (Estimated: the BoT publishes the
sector ROA, not sector profit). DSE shares: market cap of NMB + CRDB over DSE total and domestic market capitalisation;
their turnover over DSE equity turnover.

## Sources and transformation trail

`data/processed/sector_totals.csv` (BoT, crosswalk `config/bot_series_crosswalk.csv`), `data/interim/bot_long.csv`
(sector ROA), `data/processed/bank_year_wide.csv`, `data/raw/documents/{dse}` (extracted to
`data/interim/dse_market_report.csv`), `data/processed/market_daily.csv`. Functions: `shares`, `profit_gap`,
`dse_report`, `dse_shares`, `hypotheses`.

## Validation test

`tests/test_part1.py`: shares between 0 and 1; CR2 = NMB share + CRDB share; sector totals match the BoT series;
profit-gap identity; no duplicate canonical BoT series periods.

## Results

### Concentration (Derived)

{conc}

HHI: **lower bound only** (the panel holds 7 listed banks out of {n_banks} licensed banks; BoT sector structure).
Bound = held banks' squared shares + the unheld remainder split equally across the other licensed banks (the split
that minimises the HHI); the true HHI is at or above it.

{hhi}

### Profit share against asset share (Estimated)

{profit}

### DSE concentration (Derived)

{dse_res}

### Hypotheses (Interpretation)

{hyp}

## Evidence class

Shares: **Derived** from Observed BoT and bank figures. Profit shares: **Estimated** (via sector ROA). Verdicts:
**Interpretation**.

## Limitations

- L1: the sector totals are commercial banks; NMB and CRDB are Group figures (CRDB includes its Burundi and DRC
  subsidiaries), so CR2 slightly overstates the domestic share.
- Sector profit is not published as a level; the profit share rests on the sector ROA (end-June, FY2019-25, plus
  narrative values FY2011-14) and its pre- or post-tax definition is not stated.
- DSE totals come from one market report (30 Oct 2026); the daily data cover NMB and CRDB only (Aug 2025-Oct 2026).

## Interpretation

{interp}

## Cross-cutting explanations of rent (blueprint §0)

A profit share above the asset share is consistent with **scale economies**, **low-cost deposits** and **limited
competition**; chapter 1 cannot tell them apart (chapters 4-6 do). The DSE share shows how much of the listed market's
value rests on these two banks' rents.
"""


def write_ch01(root: Path) -> list[Path]:
    t = tickers(root)
    sh = shares(root)
    pg = profit_gap(root, sh)
    rep = dse_report(root)
    ds = dse_shares(root, rep)
    hyp = hypotheses(sh, pg, ds, settings(root).get("part1", {}).get("sector_roa_basis", "pretax"))
    tab, figd = path("tables", root), path("figures", root)
    hb = hhi_bound(root)
    tables = [(sh, "ch01_shares.csv", "CR2 shares of sector assets, deposits and loans", "Derived"),
              (hb, "ch01_hhi_bound.csv", "HHI lower bound from the 7 banks held (a bound, not the HHI)", "Derived"),
              (pg, "ch01_profit_vs_assets.csv", "Profit share minus asset share (via sector ROA)", "Estimated"),
              (ds, "ch01_dse_shares.csv", "DSE market-cap and turnover shares", "Derived"),
              (rep, "ch01_dse_report.csv", "DSE market report figures used", "Observed"),
              (hyp, "ch01_hypotheses.csv", "H1.1-H1.2", "Interpretation")]
    files = [write_csv(df, tab / n, require_evidence=True) for df, n, _, _ in tables]
    fig, ax = plt_subplots(figsize=(8, 4.2))
    for k, ls in (("assets", "-"), ("deposits", "--"), ("loans", ":")):
        x = sh[sh["measure"] == k]
        ax.plot(x["year"], x["cr2"] * 100, color="#1f1f1d", ls=ls, lw=1.6, label=f"CR2 {k}")
    for b in FOCUS:
        x = sh[sh["measure"] == "assets"]
        ax.plot(x["year"], x["nmb_share" if b == "B02" else "crdb_share"] * 100, color=COLORS[b], lw=1.2,
                label=f"{t[b]} asset share")
    style(ax, "NMB + CRDB share of commercial-bank assets, deposits and loans", "percent of sector")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    png = save(fig, figd / "ch01_cr2.png", "Evidence class: Derived. BoT A3.6 / A3.12 commercial banks (December); "
               "banks on a Group basis.")
    fd = write_csv(sh[["year", "measure", "cr2", "nmb_share", "crdb_share", "evidence_class"]],
                   figd / "ch01_cr2_data.csv", require_evidence=True)
    files += [png, fd]
    s = pd.read_csv(path("processed", root) / "sector_totals.csv")
    nb = s[s["series_id"] == "BOTDB_SECTOR_STRUCTURE__LICENSED_BANK_ALL_BANK_TOTAL"].sort_values("year")
    n_banks = f"{int(nb['value'].iloc[-1])} ({int(nb['year'].iloc[-1])})" if len(nb) else "about 40"
    conc = []
    for k in ("assets", "deposits", "loans"):
        x = sh[sh["measure"] == k].sort_values("year")
        if len(x):
            conc.append(f"- {k}: CR2 {pct(x['cr2'].iloc[0])} ({int(x['year'].iloc[0])}) -> {pct(x['cr2'].iloc[-1])} "
                        f"({int(x['year'].iloc[-1])}); latest NMB {pct(x['nmb_share'].iloc[-1])}, CRDB "
                        f"{pct(x['crdb_share'].iloc[-1])}; FY2015-24 mean {pct(x.loc[x['year'] >= 2015, 'cr2'].mean())}.")
    hhi_lines = []
    for k in ("assets", "deposits", "loans"):
        g = hb[hb["measure"] == k].sort_values("year")
        if g.empty:
            continue
        a, z = g.iloc[0], g.iloc[-1]
        hhi_lines.append(f"- {k}: HHI >= {z['hhi_lower_bound']:,.0f} ({int(z['year'])}; {int(z['banks_held'])} banks "
                         f"held, {pct(z['held_share'])} of the sector; held banks alone "
                         f"{z['hhi_lower_bound_held_only']:,.0f}); {int(a['year'])}: >= {a['hhi_lower_bound']:,.0f}.")
    hhi = "\n".join(hhi_lines)
    prof = "\n".join(f"- {int(r.year)}: sector ROA {pct(r.sector_roa)}, CR2 ROA {pct(r.cr2_roa_pretax)} pre-tax "
                     f"({pct(r.cr2_roa_posttax)} post-tax); asset share {pct(r.asset_share)} -> profit share "
                     f"{pct(r.profit_share_pretax)}; gap {pct(r.profit_minus_asset_share_pretax, sign=True)} "
                     f"(post-tax basis {pct(r.profit_minus_asset_share_posttax, sign=True)})." for r in pg.itertuples())
    dse = "\n".join(f"- {r.measure}, {r.period}: {pct(r.share)} (TZS {r.cr2_value_tzs / 1e9:,.1f}bn of "
                    f"{r.market_value_tzs / 1e9:,.1f}bn)." for r in ds.itertuples())
    hyp_txt = "\n".join(f"- **{r.hypothesis}: {r.verdict.upper()}.** {r.evidence}. Rule: {r.rule}." for r in hyp.itertuples())
    interp = ("**Interpretation.** Two banks hold a large and broadly stable share of commercial-"
              "bank balance sheets and a larger share of sector profit, and they dominate the DSE's value and trading. "
              "These are bank-level facts; they do not establish that concentration produces the profit gap (L1). "
              "The profit-share result also depends on the sector ROA basis (pre- vs post-tax, open issue 26).")
    nf = write_text(NOTE.format(dse=DSE_DOC, dse_res=dse, conc="\n".join(conc), n_banks=n_banks, hhi=hhi,
                                profit=prof, hyp=hyp_txt,
                                interp=interp), path("notes", root) / "ch01.md")
    files.append(nf)
    m = {"chapter": "01", "title": "Concentration and market position", "hypotheses": ["H1.1", "H1.2"],
         "status": "built",
         "tables": [{"path": rel(tab / n, root), "description": d, "evidence_class": c, "note": rel(nf, root)}
                    for _, n, d, c in tables],
         "figures": [{"path": rel(png, root), "data": rel(fd, root), "description": "CR2 over time",
                      "evidence_class": "Derived", "note": rel(nf, root)}],
         "notes": [rel(nf, root)], "module": "tzbank.chapters.ch01_concentration", "run_log_stage": "part1"}
    files.append(write_json(m, path("manifests", root) / "ch01.json"))
    return files


def write_part1(root: Path | None = None) -> list[Path]:
    """Stage `part1`: chapters 1-3 (prompt 10)."""
    from tzbank.chapters.ch02_credit_allocation import write_ch02
    from tzbank.chapters.ch03_systemic_importance import write_ch03

    root = root or find_root()
    return write_ch01(root) + write_ch02(root) + write_ch03(root)


__all__ = ["shares", "profit_gap", "dse_report", "dse_shares", "write_ch01", "write_part1", "col"]
