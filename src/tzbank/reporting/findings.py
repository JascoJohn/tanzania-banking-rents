"""Findings pack (prompt 12a, analysis baseline): every figure and verdict the dashboard shows, with a stable id.

``outputs/findings/findings_pack_v<version>.csv`` (+ ``.md`` view of the headline tier). One row per figure: id, value,
unit, bank, date, chapter, source output file, the column it came from, register version, input hash, the dashboard
page that shows it and whether it is shown there (``dashboard_shown``), a ``tier`` (headline / detail) and the
evidence class of its source row. Sources: every table listed in ``outputs/manifests/*.json`` (the dashboard renders
each manifest table in full on its chapter page, so a figure from a manifest table is shown on that page), except the
bulk surfaces in ``BULK`` (sensitivity grids, iso-value surface), plus every hypothesis verdict. The headline tier is
the curated set in ``HEADLINE`` and all verdicts; the detail tier lets the figure checker match every number a note
quotes. Ad-hoc analyses go through the pipeline (config or a named sensitivity), so they land here (D66).

Ids: ``CH<nn>.<TABLE>.<COLUMN>.<key>[.<key>...]`` with the table name upper-cased without its chapter prefix, the key
from the row's identifying columns (bank, valuation date, year, scenario, ...), e.g. ``CH06.DEPOSITS.COST_OF_DEPOSITS.
NMB.FY2025``. A duplicate id gets a ``.r<row>`` suffix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv, write_text

PACK_VERSION = "1.17"  # prompt 25 (D96); 1.16 prompt 24b (D94); earlier versions: CHANGELOG (v1.0 = analysis baseline)
BULK = {"ch09_sensitivity_grids.csv", "ch10_iso_surface.csv"}
KEYS = ["valuation", "bank", "bank_id", "ticker", "date", "fiscal_year", "year", "period", "scenario", "variant",
        "measure", "category", "comparison", "book", "position_usd", "participation", "step", "series", "provider",
        "sector_label", "sector", "component", "metric", "variable", "window", "method", "item", "hypothesis", "case",
        "lambda", "er_star", "basis", "segment", "quarter", "date_label", "line"]
TICK = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB", "B07": "MUCOBA"}
DATES = {"HIST": "2025-12-31", "CURR": "2026-10-31"}
PAGES = {"01": "pages/01_concentration.py", "02": "pages/02_credit_allocation.py",
         "03": "pages/03_systemic_importance.py", "04": "pages/04_banking_economics/04_0_overview.py",
         "04_4_0": "pages/04_banking_economics/04_0a_dupont_attribution.py",
         "04_4_1": "pages/04_banking_economics/04_1_revenue.py", "04_4_2": "pages/04_banking_economics/04_2_funding.py",
         "04_4_3": "pages/04_banking_economics/04_3_deployment.py",
         "04_4_4": "pages/04_banking_economics/04_4_transactions.py",
         "04_4_5": "pages/04_banking_economics/04_5_operating_leverage.py",
         "04_4_6": "pages/04_banking_economics/04_6_risk.py",
         "04_4_7": "pages/04_banking_economics/04_7_business_model_map.py", "05": "pages/05_persistence.py",
         "06": "pages/06_funding_architecture.py", "07": "pages/07_capital_allocation.py",
         "08": "pages/08_credit_risk.py", "09": "pages/09_comparative_valuation.py", "10": "pages/10_market_implied.py",
         "11": "pages/11_strategic_scenarios.py", "12": "pages/12_usd_implementation.py",
         "13": "pages/13_report_support.py", "14": "pages/14_market_extensions.py", "part2": "app.py",
         "hypotheses": "app.py"}
# (table, columns, row filter) curated as headline figures
HEADLINE: list[tuple[str, list[str], str | None]] = [
    ("ch01_shares.csv", ["cr2", "nmb_share", "crdb_share"], "year == year.max()"),
    ("ch01_profit_vs_assets.csv", ["profit_minus_asset_share_pretax", "profit_minus_asset_share_posttax"], None),
    ("ch01_dse_shares.csv", ["share"], None),
    ("ch01_hhi_bound.csv", ["hhi_lower_bound"], "year == year.max()"),
    ("ch02_dissimilarity.csv", ["dissimilarity"], None),
    ("ch03_deposit_cost.csv", ["advantage"], None),
    ("ch05_hypotheses.csv", [], None),
    ("ch06_franchise.csv", ["net_franchise", "gdfa"], None),
    ("ch09_valuations.csv", ["value_per_share", "price", "value_to_price", "target_price_12m", "expected_total_return",
                             "bear_value_per_share", "bull_value_per_share", "coe", "rating"], None),
    ("ch09_coe_ab.csv", ["coe_a", "coe_b"], None),
    ("ch10_implied.csv", ["implied_coe", "implied_roe_star", "implied_er_star_at_ch5_lambda"], None),
    ("ch10_rerating.csv", None, None),
    ("ch11_monte_carlo.csv", ["p10", "p50", "p90", "prob_value_above_price"], "variant.str.startswith('base')"),
    ("ch11_tested_call.csv", ["hist_value_per_share", "hist_expected_return_12m", "actual_total_return",
                              "hist_rating", "call_borne_out"], None),
    ("ch12_returns_spans.csv", ["usd_cagr", "tzs_cagr"], None),
    ("ch12_expected_usd.csv", ["expected_usd_return", "usd_hurdle", "breakeven_depreciation"],
     "base_scenario & (position_usd == 20000000)"),
    ("ch12_capacity.csv", ["days_to_build_or_exit", "round_trip_cost"], "participation == 0.2"),
]
TEXT_COLS = {"rating", "hist_rating", "call_borne_out", "verdict", "verdict_by_bank", "verdict_by_bank_date"}
COUNT_HINT = re.compile(r"(^n_|_n$|^n$|count|days|years|draws|valid|rows|quarters|shares$|number|^year$|fiscal_year|"
                        r"position_usd|pages?$|^t$|months)", re.I)
RATIO_HINT = re.compile(r"(return|share|ratio|rate$|_rate|cagr|margin|yield|growth|gap|^roe|_roe|roa|coe|prob|"
                        r"cost|change|beta|dissimilarity|pct|depreciation|hurdle|headroom|payout|density|spread|"
                        r"premium|er_star|lambda)", re.I)
MONEY_HINT = re.compile(r"(_tzs$|tzs_|equity|value$|^value_|per_share|price|target|dps|capacity|turnover|adv|"
                        r"bear_value|bull_value|^p\d+$|^mean$)", re.I)


def _ver(root: Path) -> tuple[str, str]:
    from tzbank.valuation.approve import input_hash

    reg = pd.read_csv(root / "config" / "assumption_register.csv", usecols=["version"])
    return str(reg["version"].iloc[0]), input_hash(root)


def _unit(col: str, v: pd.Series) -> str:
    if col in TEXT_COLS or v.dtype == object:
        return "text"
    if COUNT_HINT.search(col):
        return "count"
    if RATIO_HINT.search(col) and not re.search(r"(_tzs$|per_share|^price|target|_value$|^value)", col):
        return "ratio"
    if "per_share" in col or (col in {"price", "target_price_12m", "p5", "p10", "p50", "p90", "p95", "mean"}
                              and v.abs().max() > 5):
        return "TZS per share"
    if MONEY_HINT.search(col) and v.abs().max() > 1000:
        return "TZS"
    if v.abs().max() <= 5:
        return "ratio"
    return "number"


def _key(r: dict, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        x = r.get(c)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        if c == "valuation":
            out.append(str(x))
        elif c == "bank_id":
            out.append(TICK.get(str(x), str(x)))
        elif c == "fiscal_year":
            out.append(str(x) if str(x).startswith("FY") else f"FY{int(x)}")
        elif c in ("year", "position_usd") and isinstance(x, (int, float, np.integer, np.floating)):
            out.append(f"{x:.0f}" if c == "year" else f"USD{x / 1e6:.0f}M")
        elif c == "participation":
            out.append(f"P{float(x) * 100:.0f}")
        else:
            out.append(re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")[:30])
    return out


def _date(r: dict) -> str:
    if r.get("valuation") in DATES:
        return DATES[r["valuation"]]
    for c in ("date", "fiscal_year", "year", "period"):
        x = r.get(c)
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            if c == "fiscal_year":
                return str(x) if str(x).startswith("FY") else f"FY{int(x)}"
            return str(x)[:10] if c == "date" else str(x).replace(".0", "")
    return ""


def _bank(r: dict) -> str:
    for c in ("bank", "ticker"):
        if isinstance(r.get(c), str):
            return r[c]
    return TICK.get(str(r.get("bank_id", "")), "") if r.get("bank_id") is not None else ""


def _headline(table: str, df: pd.DataFrame) -> dict[str, set[int]]:
    """Columns -> row positions marked headline for one table."""
    out: dict[str, set[int]] = {}
    for t, cols, flt in HEADLINE:
        if t != table:
            continue
        rows = df.index if flt is None else df.query(flt, engine="python").index
        for c in (cols if cols is not None else list(df.columns)):
            out.setdefault(c, set()).update(rows)
    return out


def _version(p: Path) -> tuple[int, ...]:
    """findings_pack_v1.10.csv -> (1, 10): versions sort as numbers, not text (v1.10 after v1.9)."""
    m = re.search(r"_v(\d+(?:\.\d+)*)", p.stem)
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def pack_files(folder: Path) -> list[Path]:
    """All findings packs in a folder, oldest first by version number."""
    return sorted(Path(folder).glob("findings_pack_v*.csv"), key=_version)


def latest_pack(root: Path | None = None) -> Path:
    """The highest-numbered findings pack in outputs/findings."""
    files = pack_files(path("outputs", root or find_root()) / "findings")
    if not files:
        raise FileNotFoundError("no findings pack in outputs/findings: run `python -m tzbank run findings`")
    return files[-1]


def build(root: Path | None = None) -> pd.DataFrame:
    root = root or find_root()
    version, ihash = _ver(root)
    rows: list[dict] = []
    for mf in sorted(path("manifests", root).glob("*.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        chap = mf.stem[2:] if mf.stem.startswith("ch") else mf.stem
        ch_label = f"CH{chap[:2]}" if chap[:2].isdigit() else chap.upper()
        page = PAGES.get(chap, PAGES.get(chap[:2], ""))
        for t in m.get("tables", []):
            p = root / t["path"]
            name = p.name
            if name in BULK or not p.exists():
                continue
            df = pd.read_csv(p, low_memory=False)
            if df.empty:
                continue
            tname = re.sub(r"^ch\d{2}_(4_\d_)?", "", p.stem).upper()
            hl = _headline(name, df)
            keys = [k for k in KEYS if k in df.columns]
            hyp = "hypothesis" in df.columns and "verdict" in df.columns
            vcols = ([c for c in ("verdict", "verdict_by_bank", "verdict_by_bank_date") if c in df.columns] if hyp else
                     [c for c in df.columns if c not in keys and (pd.api.types.is_numeric_dtype(df[c]) or c in TEXT_COLS)
                      and c not in ("register_version", "input_hash")])
            units = {c: _unit(c, df[c]) for c in vcols}
            for k in keys:  # numeric identifying values (e.g. position sizes) are figures too
                if k in ("year", "fiscal_year", "hypothesis") or not pd.api.types.is_numeric_dtype(df[k]):
                    continue
                for x in sorted(df[k].dropna().unique()):
                    rows.append({"id": f"{ch_label}.{tname}.{k.upper()}.{x:g}", "value": float(x),
                                 "unit": _unit(k, df[k]), "bank": "", "date": "", "chapter": chap,
                                 "source_file": t["path"], "source_column": k, "source_row": -1,
                                 "register_version": version, "input_hash": ihash,
                                 "dashboard_page": f"dashboard/{page}" if page else "", "dashboard_shown": bool(page),
                                 "tier": "detail", "evidence_class": t.get("evidence_class", ""),
                                 "pack_version": PACK_VERSION})
            for i, r in enumerate(df.to_dict("records")):
                key = _key(r, keys)
                if hyp and isinstance(r.get("evidence"), str):  # figures printed in the verdict's evidence
                    from tzbank.reporting.check_figures import numbers

                    for j, e in enumerate(numbers(r["evidence"]).to_dict("records")):
                        if e["status"] == "ignored":
                            continue
                        rows.append({"id": f"{ch_label}.{str(r['hypothesis']).replace('.', '_')}.EVIDENCE.{j}",
                                     "value": e["value"], "unit": "ratio" if e["kind"] == "ratio" else "number",
                                     "bank": _bank(r), "date": "", "chapter": chap, "source_file": t["path"],
                                     "source_column": "evidence (as printed)", "source_row": i,
                                     "register_version": r.get("register_version", version)
                                     if "register_version" in df.columns else version,
                                     "input_hash": r.get("input_hash", ihash) if "input_hash" in df.columns else ihash,
                                     "dashboard_page": f"dashboard/{page}" if page else "",
                                     "dashboard_shown": bool(page), "tier": "detail",
                                     "evidence_class": r.get("evidence_class", ""), "pack_version": PACK_VERSION})
                for c in vcols:
                    v = r[c]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    rid = ".".join([ch_label, tname, c.upper(), *key]) if not hyp else ".".join(
                        [ch_label, str(r["hypothesis"]).replace(".", "_"), c.upper()])
                    rows.append({"id": rid, "value": v, "unit": "verdict" if hyp else units[c], "bank": _bank(r),
                                 "date": _date(r), "chapter": chap, "source_file": t["path"], "source_column": c,
                                 "source_row": i, "register_version": r.get("register_version", version)
                                 if "register_version" in df.columns else version,
                                 "input_hash": r.get("input_hash", ihash) if "input_hash" in df.columns else ihash,
                                 "dashboard_page": f"dashboard/{page}" if page else "", "dashboard_shown": bool(page),
                                 "tier": "headline" if hyp or i in hl.get(c, set()) else "detail",
                                 "evidence_class": r.get("evidence_class", t.get("evidence_class", "")),
                                 "pack_version": PACK_VERSION})
    pack = pd.DataFrame(rows)
    pack = pack.drop_duplicates(["id", "value", "source_file"])  # repeated key values within a table
    dup = pack.duplicated("id", keep=False)
    pack.loc[dup, "id"] = pack.loc[dup, "id"] + ".r" + pack.loc[dup, "source_row"].astype(str)
    dup = pack.duplicated("id", keep=False)
    pack.loc[dup, "id"] = pack.loc[dup, "id"] + "." + pack[dup].groupby("id").cumcount().astype(str)
    assert pack["id"].is_unique
    return pack


def _md(pack: pd.DataFrame, version: str) -> str:
    h = pack[pack["tier"] == "headline"].copy()

    def fmt(r) -> str:
        v = r["value"]
        if r["unit"] in ("text", "verdict") or isinstance(v, str):
            return str(v)
        v = float(v)
        return f"{v:.1%}" if r["unit"] == "ratio" else f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"

    h["shown"] = h.apply(fmt, axis=1)
    lines = [f"# Findings pack v{version}", "",
             f"Generated by `tzbank.reporting.findings` from the outputs listed in `outputs/manifests/`; register "
             f"{h['register_version'].iloc[0]}, input hash `{h['input_hash'].iloc[0]}`. Headline tier: {len(h)} of "
             f"{len(pack)} figures (the full pack is the CSV). Values are rounded here; the CSV holds them unrounded.",
             ""]
    for ch, g in h.groupby("chapter", sort=True):
        lines += [f"## {ch}", "", "| id | value | unit | bank | date | dashboard page |", "| --- | --- | --- | --- | --- "
                  "| --- |"]
        lines += [f"| `{r.id}` | {r.shown} | {r.unit} | {r.bank} | {r.date} | {r.dashboard_page} |" for r in
                  g.itertuples()]
        lines.append("")
    return "\n".join(lines)


def write_findings(root: Path | None = None, version: str = PACK_VERSION) -> list[Path]:
    root = root or find_root()
    pack = build(root)
    out = path("outputs", root) / "findings"
    out.mkdir(parents=True, exist_ok=True)
    f = write_csv(pack, out / f"findings_pack_v{version}.csv", require_evidence=True)
    files = [f, write_text(_md(pack, version), out / f"findings_pack_v{version}.md")]
    older = [p for p in pack_files(out) if p != f]
    if older:  # changes against the previous pack (12a amendment), for the owner's drafts
        d = diff_pack(older[-1], f).assign(evidence_class="Derived")
        prev = older[-1].stem.replace("findings_pack_", "")
        files.append(write_csv(d, out / f"diff_{prev}_v{version}.csv", require_evidence=True))
    if any((root / "drafts").glob("*.docx")):  # prompt 13 item 15: fill list for the owner's drafts (not edited)
        from tzbank.reporting.draft_check import fill_list

        files.append(write_text(fill_list(root), out / "fill_list.md"))
    return files


def diff_pack(old: str | Path, new: str | Path) -> pd.DataFrame:
    """Figures and verdicts that changed between two pack versions (owner, 12a amendment): added, removed, changed."""
    a = pd.read_csv(old, low_memory=False).set_index("id")
    b = pd.read_csv(new, low_memory=False).set_index("id")
    rows = []
    for i in sorted(set(a.index) | set(b.index)):
        if i not in b.index:
            rows.append({"id": i, "change": "removed", "old": a.at[i, "value"], "new": None})
            continue
        if i not in a.index:
            rows.append({"id": i, "change": "added", "old": None, "new": b.at[i, "value"]})
            continue
        x, y = a.at[i, "value"], b.at[i, "value"]
        try:
            same = np.isclose(float(x), float(y), rtol=1e-9, atol=1e-12)
        except (TypeError, ValueError):
            same = str(x) == str(y)
        if not same:
            rows.append({"id": i, "change": "changed", "old": x, "new": y})
    d = pd.DataFrame(rows, columns=["id", "change", "old", "new"])
    if len(d):
        meta = b.reindex(d["id"])[["unit", "bank", "date", "chapter", "tier", "dashboard_page"]].reset_index(drop=True)
        meta = meta.fillna(a.reindex(d["id"])[["unit", "bank", "date", "chapter", "tier", "dashboard_page"]]
                           .reset_index(drop=True))
        d = pd.concat([d.reset_index(drop=True), meta], axis=1)
    return d


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.diff_pack",
                                 description="List figures and verdicts that changed between two findings packs.")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", help="write the differences to this CSV")
    a = ap.parse_args(argv)
    d = diff_pack(a.old, a.new)
    if a.out:
        d.to_csv(a.out, index=False)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{len(d)} change(s): " + ", ".join(f"{k} {v}" for k, v in d["change"].value_counts().items()))
    if len(d):
        print(d.to_string(index=False, max_rows=200))
    return 0
