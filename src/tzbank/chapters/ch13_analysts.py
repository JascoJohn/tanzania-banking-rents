"""Local analyst register and analyses (prompt 19 Part D, prompt 20 item 1; D88).

Inputs (raw, MANIFEST-hashed): the merged register ``data/raw/analyst_reports/register.csv`` (one row per view: the
owner's seed of 17 documents merged with the prompt 18 rows; where they disagreed the PDF decided, listed in
``data/raw/manual_entries/analyst_register_merge.csv``) and the field transcription with page references
``data/raw/manual_entries/analyst_notes_extracted.csv``. Prices at each note's date come from the DSE library
(``data/library/dse/daily_prices.csv``, split-adjusted ``adj_close``), not from the note.

Outputs: ``data/processed/analyst_reports/register.csv`` (register + DSE price at the date + page references) and
tables ``ch13_analyst_methods`` (D1), ``ch13_analyst_targets`` (D2), ``ch13_analyst_accuracy`` (D3),
``ch13_analyst_anchoring`` (D4), ``ch13_analyst_our_model`` (D5), ``ch13_analyst_revisions`` (D6),
``ch13_analyst_issues`` (D7); chart ``ch13_analyst_targets.png``; manifest ``ch13_analysts.json``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.chapters.ch04.common import COLORS, plt_subplots, save, style
from tzbank.config import path, settings
from tzbank.io import write_csv, write_json, write_text

REGISTER = Path("data/raw/analyst_reports/register.csv")
EXTRACT = Path("data/raw/manual_entries/analyst_notes_extracted.csv")
FOCUS = {"NMB": "B02", "CRDB": "B01"}
AS_OF = pd.Timestamp("2026-10-31")


def register(root: Path) -> pd.DataFrame:
    r = pd.read_csv(root / REGISTER)
    r["date_ts"] = pd.to_datetime(r["date"], errors="coerce")
    r["target_post_split"] = pd.to_numeric(r["fair_value_post_split"], errors="coerce")
    return r


def dse_prices(root: Path) -> pd.DataFrame:
    d = pd.read_csv(root / "data" / "library" / "dse" / "daily_prices.csv", parse_dates=["trading_date"],
                    usecols=["ticker", "trading_date", "adj_close"], low_memory=False)
    return d[d["ticker"].isin(list(FOCUS))].dropna(subset=["adj_close"]).sort_values("trading_date")


def price_on(px: pd.DataFrame, tick: str, date: pd.Timestamp) -> tuple[float, str]:
    """Last split-adjusted DSE close on or before the date (post-split basis for NMB)."""
    x = px[(px["ticker"] == tick) & (px["trading_date"] <= date)]
    return (float(x["adj_close"].iloc[-1]), x["trading_date"].iloc[-1].date().isoformat()) if len(x) else (np.nan, "")


def processed(root: Path) -> pd.DataFrame:
    r = register(root)
    px = dse_prices(root)
    # price at the note's data date where one is printed (e.g. Monthly Rally: data 31 Mar, published 8 Apr), else the
    # note's date; the publication date stays in `date` for point-in-time use
    asof = pd.to_datetime(r.get("data_as_of"), errors="coerce").fillna(r["date_ts"])
    r["price_date_basis"] = np.where(pd.to_datetime(r.get("data_as_of"), errors="coerce").notna(), "data_as_of", "date")
    p = [price_on(px, b, d) if b in FOCUS and pd.notna(d) else (np.nan, "") for b, d in zip(r["bank"], asof,
                                                                                         strict=True)]
    r["price_date_ts"] = asof
    r["price_dse_post_split"] = [x[0] for x in p]
    r["price_dse_date"] = [x[1] for x in p]
    r["target_to_dse_price"] = r["target_post_split"] / r["price_dse_post_split"]
    ex = root / EXTRACT
    if ex.exists():
        e = pd.read_csv(ex, dtype=str)
        pages = e.dropna(subset=["page"]).groupby("seed_id").apply(
            lambda g: "; ".join(f"{f}: p.{p}" for f, p in zip(g["field"], g["page"], strict=True)),
            include_groups=False)
        r["page_refs"] = r["id"].astype(str).map(pages)
    return r


def methods(r: pd.DataFrame) -> pd.DataFrame:
    v = r[r["target_post_split"].notna() | r["method"].notna()]
    v = v[~v["method"].fillna("").str.contains("data only|commentary|market review", case=False)]
    m = v["method"].fillna("").str.lower()
    rows = [{"item": f"notes using {k}", "value": int(m.str.contains(p).sum()), "of_notes": len(v)}
            for k, p in (("DDM", "ddm|dividend"), ("RIM / excess return", "rim|residual|excess return"),
                         ("multiples", "multiple|p/e|p/b|relative"), ("justified P/B", "justified"),
                         ("method not stated", "not stated"))]
    for col, lab in (("coe", "cost of equity"), ("terminal_g", "terminal growth"), ("horizon_months", "horizon, months")):
        s = pd.to_numeric(v.get(col), errors="coerce").dropna() if col in v else pd.Series(dtype=float)
        rows += [{"item": f"{lab}: notes disclosing", "value": len(s), "of_notes": len(v)},
                 {"item": f"{lab}: min", "value": s.min() if len(s) else np.nan, "of_notes": len(s)},
                 {"item": f"{lab}: median", "value": s.median() if len(s) else np.nan, "of_notes": len(s)},
                 {"item": f"{lab}: max", "value": s.max() if len(s) else np.nan, "of_notes": len(s)}]
    return pd.DataFrame(rows).assign(evidence_class="Derived")


def targets(r: pd.DataFrame, root: Path) -> pd.DataFrame:
    x = r[r["bank"].isin(list(FOCUS)) & r["target_post_split"].notna()].copy()
    x["source"] = x["broker"]
    v = pd.read_csv(path("tables", root) / "ch09_valuations.csv")
    ours = pd.DataFrame({"source": "this study", "bank": v["bank"], "date_ts": pd.to_datetime(v["date"]),
                         "rating": v["rating"], "target_post_split": v["value_per_share_post_split_basis"],
                         "price_dse_post_split": v["price_post_split_basis"]})
    out = pd.concat([x[["source", "bank", "date_ts", "rating", "target_post_split", "price_dse_post_split"]], ours])
    out["target_to_price"] = out["target_post_split"] / out["price_dse_post_split"]
    out["date"] = out["date_ts"].dt.date.astype(str)
    return out.drop(columns="date_ts").sort_values(["bank", "date"]).assign(evidence_class="Derived")


def accuracy(r: pd.DataFrame, root: Path) -> pd.DataFrame:
    """D3: for notes at least 12 months old at 31 Oct 2026, target against the DSE price 12 months later."""
    px = dse_prices(root)
    rows = []
    for x in r[r["bank"].isin(list(FOCUS)) & r["target_post_split"].notna()].itertuples():
        d1 = x.date_ts + pd.DateOffset(years=1)
        if d1 > AS_OF:
            rows.append({"id": x.id, "broker": x.broker, "bank": x.bank, "date": x.date, "status":
                         "less than 12 months old at 31 Oct 2026", "evidence_class": "Derived"})
            continue
        p1, p1d = price_on(px, x.bank, d1)
        rows.append({"id": x.id, "broker": x.broker, "bank": x.bank, "date": x.date, "rating": x.rating,
                     "target": x.target_post_split, "price_at_note": x.price_dse_post_split, "price_12m": p1,
                     "price_12m_date": p1d, "target_error": p1 / x.target_post_split - 1,
                     "price_return": p1 / x.price_dse_post_split - 1,
                     "direction_right": np.sign(x.target_post_split - x.price_dse_post_split) == np.sign(
                         p1 - x.price_dse_post_split), "status": "computed", "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def anchoring(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in FOCUS:
        x = r[(r["bank"] == b) & r["target_post_split"].notna() & r["price_dse_post_split"].notna()].sort_values(
            "date_ts")
        if len(x) < 2:
            continue
        ratio = x["target_post_split"] / x["price_dse_post_split"]
        rows.append({"bank": b, "notes": len(x), "first": x["date"].iloc[0], "last": x["date"].iloc[-1],
                     "log_target_change": float(np.log(x["target_post_split"].iloc[-1] / x["target_post_split"].iloc[0])),
                     "log_price_change": float(np.log(x["price_dse_post_split"].iloc[-1]
                                                      / x["price_dse_post_split"].iloc[0])),
                     "ratio_min": float(ratio.min()), "ratio_median": float(ratio.median()),
                     "ratio_max": float(ratio.max()), "ratio_cv": float(ratio.std() / ratio.mean()),
                     "corr_log_target_log_price": float(np.corrcoef(np.log(x["target_post_split"]),
                                                                    np.log(x["price_dse_post_split"]))[0, 1])
                     if len(x) > 2 else np.nan,
                     "reading": "exploratory: few notes; a stable ratio while price moves is consistent with "
                                "anchoring on price", "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def our_model(r: pd.DataFrame, ctx) -> pd.DataFrame:
    """D5: our value per share at each broker's disclosed COE (and terminal g where disclosed), other inputs at our base
    case; separates assumptions from method."""
    from tzbank.chapters.ch13_report_support import quote

    rows = []
    for x in r[r["bank"].isin(list(FOCUS)) & pd.to_numeric(r["coe"], errors="coerce").notna()].itertuples():
        val = "CURR" if x.date_ts.year >= 2026 else "HIST"
        b = FOCUS[x.bank]
        c = ctx.cases[(val, b, "base")]
        coe = float(x.coe)
        g_b = pd.to_numeric(x.terminal_g, errors="coerce")
        real_g = float(settings(ctx.root)["valuation"].get("terminal_real_growth", 0.03))
        # a real broker g is made nominal with our inflation assumption (our g = inflation + real growth, D64)
        g = c.dr.g if pd.isna(g_b) else (c.dr.g - real_g + float(g_b) if "real" in str(x.g_basis) else float(g_b))
        guard = coe - g >= 0.04
        dr = replace(c.dr, g=g, payout_star=1 - g / c.dr.roe_star) if g != c.dr.g else c.dr
        q = quote(ctx, val, b, dr, r=coe) if guard else {"value_per_share": np.nan, "rating": "n/a"}
        split = 10 if (x.bank == "NMB" and val == "HIST") else 1
        rows.append({"id": x.id, "broker": x.broker, "bank": x.bank, "note_date": x.date, "our_date": val,
                     "broker_coe": coe, "broker_g_nominal": g, "our_coe": c.r, "our_g": c.dr.g,
                     "our_value_base_post_split": quote(ctx, val, b)["value_per_share"] / split,
                     "our_value_at_broker_inputs_post_split": q["value_per_share"] / split,
                     "broker_target_post_split": x.target_post_split, "rating_at_broker_inputs": q["rating"],
                     "broker_g_basis": x.g_basis, "note": "terminal g: the broker's where disclosed (a real g made "
                     "nominal with our inflation assumption), else ours; ROE* held, payout* re-solved",
                     "evidence_class": "Scenario"})
    return pd.DataFrame(rows)


def revisions(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = r[r["bank"].isin(list(FOCUS)) & r["target_post_split"].notna()].sort_values("date_ts")
    for (br, b), g in x.groupby(["broker", "bank"]):
        g = g.sort_values("price_date_ts").reset_index(drop=True)
        for i in range(1, len(g)):
            a, z = g.iloc[i - 1], g.iloc[i]
            rows.append({"broker": br, "bank": b, "from": a["price_date_ts"].date().isoformat(),
                         "to": z["price_date_ts"].date().isoformat(),
                         "target_from": a["target_post_split"], "target_to": z["target_post_split"],
                         "target_change": z["target_post_split"] / a["target_post_split"] - 1,
                         "price_from": a["price_dse_post_split"], "price_to": z["price_dse_post_split"],
                         "price_change": z["price_dse_post_split"] / a["price_dse_post_split"] - 1,
                         "target_to_price_after": z["target_post_split"] / z["price_dse_post_split"] - 1,
                         "rating_from": a["rating"], "rating_to": z["rating"], "evidence_class": "Derived"})
    return pd.DataFrame(rows)


def issues(root: Path) -> pd.DataFrame:
    f = root / "data" / "raw" / "manual_entries" / "analyst_register_merge.csv"
    m = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["id", "field", "seed", "prior", "pdf", "resolution"])
    return m.assign(evidence_class="Observed")


def chart(t: pd.DataFrame, figd: Path) -> Path:
    fig, axs = plt_subplots(1, 2, figsize=(10, 3.8))
    for ax, b in zip(axs, FOCUS, strict=True):
        g = t[t["bank"] == b].copy()
        g["d"] = pd.to_datetime(g["date"])
        br, us = g[g["source"] != "this study"], g[g["source"] == "this study"]
        for s, gg in br.groupby("source"):
            ax.scatter(gg["d"], gg["target_post_split"], s=22, label=f"{s}")
        ax.scatter(us["d"], us["target_post_split"], color=COLORS[FOCUS[b]], marker="D", s=30, label="this study")
        pr = g.dropna(subset=["price_dse_post_split"]).sort_values("d")
        ax.plot(pr["d"], pr["price_dse_post_split"], color="#1f1f1d", lw=0.8, ls=":", marker="x", ms=4,
                label="DSE price at the date")
        style(ax, f"{b}: targets and fair values against price (post-split TZS)", "TZS per share")
        ax.legend(fontsize=6, frameon=False)
        ax.tick_params(axis="x", rotation=30)
    return save(fig, figd / "ch13_analyst_targets.png", "Source: local broker notes (register, page-referenced); DSE "
                "daily prices (library, split-adjusted); this study, register v1.3. Evidence class: Derived.")


NOTE = """# Local analyst register (prompts 19-20, D88)

Generated by `tzbank.chapters.ch13_analysts` (stage `market_ext`).

## Research question
How do local brokers value NMB and CRDB, and do their targets follow the price (Chapter 14, price formation)?

## Variable definitions
Targets and prices on the post-split basis (NMB 1:10, 24 Aug 2026); price at a note's date = last DSE close on or
before it (library, split-adjusted), not the price printed in the note.

## Sources
data/raw/analyst_reports/register.csv (merged register), analyst_notes_extracted.csv (fields with pages), DSE library.

## Validation test
`tests/test_analysts.py`.

## Results
{results}

## Evidence class
Observed (register), Derived (ratios, revisions), Scenario (our model at brokers' inputs).

## Limitations
- Few notes (exploratory only); some notes do not disclose the method, COE or horizon.
- Our model at a broker's COE holds every other input at our base case; it isolates the COE level, not the method.
"""


def write_analysts(ctx) -> list[Path]:
    root = ctx.root
    if not (root / REGISTER).exists():
        return []
    tab, figd = path("tables", root), path("figures", root)
    out = path("processed", root) / "analyst_reports"
    out.mkdir(parents=True, exist_ok=True)
    r = processed(root)
    files = [write_csv(r.drop(columns=["date_ts", "price_date_ts"]).assign(evidence_class=r.get("evidence_class", "Observed")),
                       out / "register.csv", require_evidence=True)]
    t = targets(r, root)
    nf = path("notes", root) / "ch13_analysts.md"
    tables = [(methods(r), "ch13_analyst_methods.csv", "D1. Methods summary", "Derived"),
              (t, "ch13_analyst_targets.csv", "D2. Targets against price, with our values", "Derived"),
              (accuracy(r, root), "ch13_analyst_accuracy.csv", "D3. Forecast accuracy", "Derived"),
              (anchoring(r), "ch13_analyst_anchoring.csv", "D4. Anchoring", "Derived"),
              (our_model(r, ctx), "ch13_analyst_our_model.csv", "D5. Our model at the brokers' COE and g", "Scenario"),
              (revisions(r), "ch13_analyst_revisions.csv", "D6. Target revisions against the price move", "Derived"),
              (issues(root), "ch13_analyst_issues.csv", "D7. Register disagreements and data issues", "Observed")]
    man_t = []
    for df, name, desc, ev in tables:
        f = write_csv(ctx.stamp(df), tab / name, require_evidence=True)
        files.append(f)
        man_t.append({"path": f.relative_to(root).as_posix(), "description": desc, "evidence_class": ev,
                      "note": nf.relative_to(root).as_posix()})
    p = chart(t, figd)
    dd = write_csv(t, p.with_name(f"{p.stem}_data.csv"), require_evidence=True)
    files += [p, dd]
    rv = tables[5][0]
    res = [f"- {len(r)} register rows; {int(r['bank'].isin(list(FOCUS)).sum())} on NMB or CRDB."]
    res += [f"- {x.broker} {x.bank} {x['from']} -> {x['to']}: target {x.target_change:+.1%}, price {x.price_change:+.1%}"
            for _, x in rv.iterrows()] if len(rv) else []
    files.append(write_text(NOTE.format(results="\n".join(res)), nf))
    m = {"chapter": "13_analysts", "title": "Local analyst register (prompts 19-20)", "status": "complete",
         "hypotheses": [], "tables": man_t, "figures": [{"path": p.relative_to(root).as_posix(),
                                                         "data": dd.relative_to(root).as_posix(),
                                                         "description": "D2. Targets against price",
                                                         "evidence_class": "Derived",
                                                         "note": nf.relative_to(root).as_posix()}],
         "notes": [nf.relative_to(root).as_posix()], "inputs": [str(REGISTER), str(EXTRACT)],
         "module": "tzbank.chapters.ch13_analysts", "run_log_stage": "market_ext"}
    files.append(write_json(m, path("manifests", root) / "ch13_analysts.json"))
    return files
