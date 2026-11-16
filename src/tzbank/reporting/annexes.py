"""Annex and pipeline tables for the designed reports (prompt 16; D65, D77).

Every annex is laid out from pipeline tables in ``outputs/tables`` (and the generated variable dictionary for the
glossary). Values are formatted, never recomputed; the only operations are selection, ordering and (price history)
monthly grouping of the observed daily series. Each returns a ``Grid``: title, header, rows, source line and the
evidence classes of the rows shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tzbank.config import path


@dataclass
class Grid:
    title: str
    header: list
    rows: list
    source: str
    evidence: list = field(default_factory=list)
    bold_rows: set = field(default_factory=set)
    dropped: list = field(default_factory=list)


def report_config(root: Path) -> dict:
    """config/reports.yaml: layout constants of the report build (kept out of settings.yaml so that they do not move
    the analytical input hash)."""
    f = root / "config" / "reports.yaml"
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f.exists() else {}


# --- number formats (cols="name:Label:format") ---------------------------------------------------------------------
def fmt_value(v, f: str = "") -> str:
    """Format one value: pct0/pct1/pct2, x1/x2, int, num1/num2, bn (TZS bn, negatives in brackets), text. A negative
    number prints with a true minus sign (U+2212), never a hyphen (prompt 25, 7.2)."""
    out = _fmt_value(v, f)
    return "−" + out[1:] if out.startswith("-") and len(out) > 1 and (out[1].isdigit() or out[1] == ".") else out


def _fmt_value(v, f: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return "–"
    if isinstance(v, str) or not f and not isinstance(v, (int, float, np.number)):
        return str(v)
    x = float(v)
    if f.startswith("pct"):
        return f"{x * 100:.{int(f[3:] or 1)}f}%"
    if f.startswith("x"):
        return f"{x:.{int(f[1:] or 2)}f}×"
    if f == "bn":
        s = f"{abs(x) / 1e9:,.0f}"
        return f"({s})" if x < 0 else s
    if f == "int":
        return f"{x:,.0f}"
    if f.startswith("num"):
        return f"{x:,.{int(f[3:] or 1)}f}"
    if f == "text":
        return str(v)
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def _table(root: Path, name: str) -> pd.DataFrame:
    f = path("tables", root) / (name if name.endswith(".csv") else f"{name}.csv")
    if not f.exists():
        raise FileNotFoundError(f"pipeline table {f.name} not found in outputs/tables")
    return pd.read_csv(f, low_memory=False)


def _ev(d: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(d["evidence_class"].dropna().astype(str))) if "evidence_class" in d else []


def _src(text: str, d: pd.DataFrame) -> str:
    """Source line of a generated table: a plain description and the findings pack as the traceable reference
    ([[pack]] is resolved by the build). The inputs-register version is stated once, in the methods annex, not in
    every caption (prompt 23)."""
    return f"Source: {text}; findings pack ([[pack]]). Evidence class: {', '.join(_ev(d)) or 'n/a'}."


# plain descriptions of the pipeline tables a report cites without its own source line (prompt 23: no table names
# in the PDF); any other table is described as the author's calculations
TABLE_SOURCE = {
    "ch13_kpi_table": "Banks' annual reports; DSE prices",
    "ch13_dec_interp_weights": "Author's scenarios: value at each weight on the exceptional reading of 2015-18",
    "ch13_dec_bank_coe": "Author's cost-of-equity variants",
    "ch13_dec_implied_growth": "Author's calculations: growth implied by the price",
    "ch09_coe_ab": "Author's cost-of-equity build-up (Approaches A and B)",
    "ch13_dec_monitoring": "Author's thesis monitor and FY2025 ratios",
    "ch13_p21_forecast_kpis": "Banks' annual reports and the forecast model",
}


def _value_filter(d: pd.DataFrame, token: str) -> pd.DataFrame:
    """Rows where any column holds the value (case-sensitive text match); ``FY2025`` also matches a fiscal year of
    2025 and labels such as ``FY2025A``."""
    keep = pd.Series(False, index=d.index)
    for c in d.columns:
        keep |= d[c].astype(str).str.strip().eq(token)
    m = re.fullmatch(r"FY(\d{4})", token)
    if m:
        for c in ("fiscal_year", "year", "anchor_fy"):
            if c in d:
                v = d[c].astype(str).str.replace(r"\.0$", "", regex=True)
                keep |= v.eq(m.group(1)) | v.str.fullmatch(rf"FY{m.group(1)}[AE]?")
    if not keep.any():
        raise ValueError(f"filter {token!r}: no row holds this value")
    return d[keep]


MULTIPLE = re.compile(r"(p/?b|p/?e|(market|model|justified)_p[be]|value_to_price|pb_.*|pe_.*)")


def _default_fmt(col: str, v: pd.Series) -> str:
    """Format for a column without one in ``cols``: the findings pack's unit rule (ratios in percent, money and
    counts as whole numbers)."""
    from tzbank.reporting.findings import _unit

    if not pd.api.types.is_numeric_dtype(v):
        return ""
    if MULTIPLE.fullmatch(col.lower()):
        return "x2"
    u = _unit(col, v)
    return {"ratio": "pct1", "TZS": "int", "TZS per share": "int", "count": "int"}.get(u, "")


def pipeline_table(root: Path, src: str, filt: str | None, cols: str | None, source: str | None,
                   group: str | None = None) -> Grid:
    """{{table src filter cols}}: rows by a pandas query, columns as ``name[:Label[:format]]`` (comma-separated).
    ``group``: a column whose values become bold group rows (prompt 23); it is then not printed as a column."""
    d = _table(root, src)
    if filt and re.fullmatch(r"[\w.\-/ ]+", filt.strip()) and not re.search(r"\b(and|or|in|not)\b", filt):
        d = _value_filter(d, filt.strip())  # shorthand: a bare value such as CURR or FY2025
    elif filt:
        try:
            d = d.query(filt, engine="python")
        except Exception as e:  # noqa: BLE001 - report the author's filter error plainly
            raise ValueError(f"table {src}: filter {filt!r} failed: {e}") from None
    spec = [c.strip() for c in (cols.split(",") if cols else [c for c in d.columns if c not in
            ("evidence_class", "register_version", "input_hash")])]
    names, labels, fmts = [], [], []
    for c in spec:
        parts = c.split(":")
        if parts[0] not in d.columns:
            raise ValueError(f"table {src}: no column {parts[0]!r} (has: {', '.join(d.columns[:30])})")
        names.append(parts[0])
        labels.append(parts[1] if len(parts) > 1 and parts[1] else parts[0])
        fmts.append(parts[2] if len(parts) > 2 else _default_fmt(parts[0], d[parts[0]]))
    rows = [[fmt_value(r[n], f) for n, f in zip(names, fmts, strict=True)] for _, r in d.iterrows()]
    if not rows:
        raise ValueError(f"table {src}: the filter leaves no rows")
    bold: set = set()
    if group:  # a bold group row (the group value, capitalised) above each group; the column itself is dropped
        if group not in d.columns:
            raise ValueError(f"table {src}: no group column {group!r}")
        gv = [str(r[group]) for _, r in d.iterrows()]
        keep = [k for k, n in enumerate(names) if n != group]
        labels = [labels[k] for k in keep]
        grouped, last = [], None
        for g, row in zip(gv, rows, strict=True):
            if g != last:
                bold.add(len(grouped))
                grouped.append([g[:1].upper() + g[1:]] + [""] * (len(keep) - 1))
                last = g
            grouped.append([row[k] for k in keep])
        rows = grouped
    desc = TABLE_SOURCE.get(Path(src).stem, "Author's calculations")
    return Grid("", labels, rows, source or _src(desc, d), _ev(d), bold)


# --- annexes ---------------------------------------------------------------------------------------------------------
FIN_ROWS = [("NII", "Net interest income", "bn"), ("NonII", "Non-interest income", "bn"),
            ("OpEx", "Operating expenses", "bn"), ("Imp", "Impairment", "bn"), ("PBT", "Profit before tax", "bn"),
            ("NP_attr", "Profit attributable", "bn"), ("net_loans", "Net loans", "bn"),
            ("deposits", "Customer deposits", "bn"), ("total_assets", "Total assets", "bn"),
            ("equity", "Equity attributable", "bn")]
RATIO_ROWS = [("ROE", "ROE (average equity)", "pct1"), ("NIM", "Net interest margin", "pct1"),
              ("CIR", "Cost-to-income", "pct1"), ("CoR", "Cost of risk", "pct2"), ("NPL", "NPL ratio", "pct1"),
              ("NPL_coverage", "NPL coverage", "pct0"), ("Tier1", "Tier 1 ratio", "pct1"),
              ("LDR", "Loans to deposits", "pct1"), ("EPS", "EPS (TZS)", "int"), ("DPS", "DPS (TZS)", "int"),
              ("BVPS", "Book value per share (TZS)", "int"), ("payout", "Payout", "pct0"), ("P/E", "P/E", "x1"),
              ("P/B", "P/B", "x2"), ("dividend_yield", "Dividend yield", "pct1")]


def _by_year(root: Path, bank: str, rows: list, title: str, unit: str) -> Grid:
    d = _table(root, "ch13_financial_table")
    d = d[d["bank"] == bank]
    if d.empty:
        raise ValueError(f"ch13_financial_table has no rows for {bank}")
    years = list(d["fiscal_year"])
    head = [unit] + years
    out = [[lab] + [fmt_value(v, f) for v in d[c]] for c, lab, f in rows if c in d]
    basis = d["per_share_basis"].dropna().iloc[0] if "per_share_basis" in d and d["per_share_basis"].notna().any() \
        else ""
    src = _src(f"{bank} annual reports (actuals, A) and the forecast model (E)"
               + (f"; per share: {basis}" if basis else ""), d)
    bold = {i for i, r in enumerate(out) if r[0] == "Profit attributable"}
    return Grid(f"{title}, {years[0]}–{years[-1]}", head, out, src, _ev(d), bold)


def financials(root: Path, bank: str) -> Grid:
    return _by_year(root, bank, FIN_ROWS, f"{bank} financial summary", "TZS bn")


def ratios(root: Path, bank: str) -> Grid:
    return _by_year(root, bank, RATIO_ROWS, f"{bank} ratios and per-share data", "Ratio")


VAL_ROWS = [("date", "Valuation date", "text"), ("coe", "Cost of equity", "pct1"),
            ("rim_per_share", "Residual income value per share", "int"),
            ("ddm_per_share", "Dividend discount value per share", "int"),
            ("value_per_share", "Value per share (blend)", "int"), ("price", "Price", "int"),
            ("value_to_price", "Value / price", "x2"), ("target_price_12m", "12-month target", "int"),
            ("expected_dps_12m", "Expected DPS, 12 months", "int"),
            ("expected_total_return", "Expected total return", "pct1"),
            ("bear_value_per_share", "Bear-case value", "int"), ("bull_value_per_share", "Bull-case value", "int"),
            ("bear_downside", "Bear-case downside", "pct1"), ("rating", "Rating", "text"),
            ("market_pb", "Market P/B", "x2"), ("justified_pb", "Justified P/B on sustainable ROE (model)", "x2")]
PER_SHARE = {"rim_per_share", "ddm_per_share", "value_per_share", "price", "target_price_12m", "expected_dps_12m",
             "bear_value_per_share", "bull_value_per_share", "p5", "p10", "p50", "p90", "p95", "mean"}


def split(root: Path, bank: str) -> tuple[float, pd.Timestamp | None, str]:
    """(ratio, effective date, note) of a share split for the bank (config/corporate_actions.csv); (1, None, '')."""
    f = root / "config" / "corporate_actions.csv"
    if not f.exists():
        return 1.0, None, ""
    ca = pd.read_csv(f)
    x = ca[(ca["ticker"] == bank) & (ca["action"] == "split")]
    if x.empty:
        return 1.0, None, ""
    r = x.iloc[0]
    eff = pd.Timestamp(r["effective_date"])
    return float(r["ratio"]), eff, f"Restated for the 1:{int(r['ratio'])} split of {eff:%d %b %Y}"


def post_split(root: Path, d: pd.DataFrame, bank: str, cols: set[str], date_col: str | None = None) -> tuple[
        pd.DataFrame, str]:
    """Per-share columns of rows dated before the bank's split restated to the post-split basis (divided by the
    ratio; CLAUDE.md 3.4, prompt 17 A5). Rows are dated by ``date_col`` or by their valuation (HIST = 31 Dec 2025).
    Returns the frame and the footnote ('' when nothing was restated)."""
    ratio, eff, note = split(root, bank)
    if ratio == 1 or d.empty:
        return d, ""
    if date_col:
        when = pd.to_datetime(d[date_col])
    else:
        from tzbank.config import settings

        vd = {k: pd.Timestamp(v) for k, v in (settings(root).get("valuation_dates") or {}).items()}
        hist = vd.get("HIST", pd.Timestamp("2025-12-31"))
        curr = vd.get("CURR", pd.Timestamp("2026-10-31"))
        when = d["valuation"].map({"HIST": hist, "CURR": curr})
    pre = when < eff
    if not pre.any():
        return d, ""
    d = d.copy()
    for c in cols & set(d.columns):
        d.loc[pre, c] = d.loc[pre, c] / ratio
    return d, f"{note}: per-share figures dated before it are divided by {int(ratio)}"


def valuation(root: Path, bank: str) -> Grid:
    d = _table(root, "ch09_valuations")
    d = d[d["bank"] == bank]
    d, foot = post_split(root, d, bank, PER_SHARE, "date")
    head = ["Item"] + [f"{v} ({dt})" for v, dt in zip(d["valuation"], d["date"], strict=True)]
    out = [[lab] + [fmt_value(v, f) for v in d[c]] for c, lab, f in VAL_ROWS if c in d]
    basis = "post-split basis" if foot else "per share as reported"
    return Grid(f"{bank} valuation summary ({basis})", head, out,
                _src("Author's valuation (blend of residual income and dividend discount models)"
                     + (f". {foot}" if foot else ""), d), _ev(d), {4})


def scenarios(root: Path, bank: str) -> Grid:
    d = _table(root, "ch11_scenarios")
    d = d[d["bank"] == bank]
    eps = {c for c in d.columns if c.startswith("eps_")}
    d, foot = post_split(root, d, bank, {"value_per_share", "price"} | eps)
    cols = [("valuation", "Valuation", "text"), ("scenario", "Scenario", "text"),
            ("value_per_share", "Value/share", "int"), ("price", "Price", "int")]
    for y in (2026, 2027, 2028):
        cols += [(f"roe_{y}", f"ROE {y}", "pct1")] if d.get(f"roe_{y}", pd.Series(dtype=float)).notna().any() else []
    for y in (2026, 2027, 2028):
        cols += [(f"eps_{y}", f"EPS {y}", "int")] if d.get(f"eps_{y}", pd.Series(dtype=float)).notna().any() else []
    out = [[fmt_value(r[c], f) for c, _, f in cols] for _, r in d.iterrows()]
    return Grid(f"{bank} scenarios", [lab for _, lab, _ in cols], out,
                _src("Author's scenarios (bear, base and bull on the forecast model)" + (f". {foot}" if foot else ""),
                     d), _ev(d))


def montecarlo(root: Path, bank: str) -> Grid:
    d = _table(root, "ch11_monte_carlo")
    d = d[d["bank"] == bank]
    d, foot = post_split(root, d, bank, PER_SHARE)
    cols = [("valuation", "Valuation", "text"), ("variant", "Variant", "text"), ("valid", "Draws", "int"),
            ("price", "Price", "int"), ("p5", "P5", "int"), ("p10", "P10", "int"), ("p50", "Median", "int"),
            ("p90", "P90", "int"), ("p95", "P95", "int"), ("prob_value_above_price", "P(value > price)", "pct1")]
    out = [[fmt_value(r[c], f) for c, _, f in cols] for _, r in d.iterrows()]
    seed = d["seed"].dropna().astype(str).iloc[0] if "seed" in d and d["seed"].notna().any() else ""
    return Grid(f"{bank} Monte Carlo value distribution", [lab for _, lab, _ in cols], out,
                _src(f"Author's Monte Carlo simulation; value per share in TZS; seed {seed}"
                     + (f". {foot}" if foot else ""), d),
                _ev(d))


def price_history(root: Path, bank: str) -> Grid:
    """Monthly price history from the pipeline table ch13_price_monthly (close, range, turnover, days traded)."""
    d = _table(root, "ch13_price_monthly")
    d = d[d["bank"] == bank]
    if d.empty:
        raise ValueError(f"ch13_price_monthly has no rows for {bank}")
    out = [[r["month"], fmt_value(r["close"], "int"), fmt_value(r["high"], "int"), fmt_value(r["low"], "int"),
            fmt_value(r["turnover_tzs"] / 1e6, "int"), fmt_value(r["days_traded"], "int")] for _, r in d.iterrows()]
    return Grid(f"{bank} share price history, {out[0][0]}–{out[-1][0]}",
                ["Month", "Close (TZS)", "High (TZS)", "Low (TZS)", "Turnover (TZS m)", "Days traded"], out,
                _src("DSE prices: daily closes and turnover, monthly; post-split basis", d), _ev(d))


def peers(root: Path, bank: str | None = None) -> Grid:
    """Peer set at 31 Oct 2026 on the prompt 19 figures (``ch13_peers_annex``: NMB, CRDB and the seven African peers;
    P/B on reported book). Other DSE-listed Tanzanian banks are not in the set and are named in the source line
    (prompt 17 A6; prompt 23)."""
    d = _table(root, "ch13_peers_annex")
    k = _table(root, "ch13_kpi_table")
    rows, bold = [], set()
    for _, r in d.iterrows():
        if r["bank"] == bank:
            bold.add(len(rows))
        rows.append([r["bank"], r["country"], str(r["fy_basis"]), fmt_value(r["roe"], "pct1"), fmt_value(r["pb"], "x2"),
                     fmt_value(r["pe"], "x1"), fmt_value(r["dividend_yield"], "pct1"),
                     fmt_value(r["npl_ratio"], "pct1"), fmt_value(r["cost_to_income"], "pct1")])
    dropped = sorted(set(k["bank"]) - set(d["bank"]))
    notes = []  # footnotes for the blanks and the derived cells (prompt 24)
    nopx = list(d[d["pb_note"].fillna("").ne("")]["bank"]) if "pb_note" in d else []
    if nopx:
        notes.append(f"{' and '.join(nopx)}: no traded price at 31 Oct 2026, so no P/B or P/E")
    if "cir_basis" in d:
        for r in d[d["cir_basis"].fillna("").str.startswith("staff")].itertuples():
            notes.append(f"{r.bank} cost-to-income: {r.cir_basis}")
        miss = list(d[d["cost_to_income"].isna()]["bank"])
        if miss:
            notes.append(f"{' and '.join(miss)} cost-to-income: not reported on a comparable basis")
    note = ("Peer annual reports and exchange prices at 30-31 October 2026 (NMB and CRDB: DSE price over FY2025 book); "
            "P/B on reported book value; NPL and cost-to-income from the banks' annual reports"
            + "".join(f". {x}" for x in notes)
            + (f". Not in the peer set: {', '.join(dropped)}" if dropped else ""))
    g = Grid("Peer comparison", ["Bank", "Country", "FY", "ROE", "P/B", "P/E", "Dividend yield", "NPL",
                                 "Cost-to-income"], rows, _src(note, d), _ev(d), bold)
    g.dropped = dropped
    return g


GLOSSARY_SECTION = "glossary"


def reader_glossary(root: Path) -> list[tuple[str, str]]:
    """(term, meaning) rows of the reader glossary: the table under the main report's '## Annex X. Glossary'
    heading in reports/source (owner-authored; prompt 17 A7)."""
    from tzbank.reporting.report_source import parse_parts, source_files

    files = source_files(root / report_config(root).get("source_dir", "reports/source"), "main")
    if not files:
        raise ValueError("glossary: no main-report source in reports/source to take the reader glossary from")
    nodes = parse_parts(files).nodes
    for i, n in enumerate(nodes):
        if n.kind == "chapter" and n.text.strip().lower() == GLOSSARY_SECTION:
            tab = next((x for x in nodes[i + 1:] if x.kind in ("table", "chapter", "part")), None)
            if tab is not None and tab.kind == "table":
                return [(r[0], r[1]) for r in tab.items[1:]]
    raise ValueError("glossary: the main report has no '## Annex X. Glossary' section with a Term | Meaning table")


def _aliases(term: str) -> list[str]:
    """'ROE / ROA' -> ROE, ROA; 'Excess return (ER)' -> Excess return, ER; 'P/B, P/E' -> P/B, P/E."""
    out = []
    for part in re.split(r"\s*,\s*|\s+/\s+", re.sub(r"\(([^)]*)\)", r", \1", term)):
        part = part.strip(" ,")
        if part:
            out.append(part)
    return out


def glossary(root: Path, report_text: str | None = None) -> Grid:
    """Reader glossary filtered to the terms this report uses (any alias found in its text)."""
    rows = reader_glossary(root)
    if report_text is not None:
        def used(term: str) -> bool:
            for a in _aliases(term):
                flags = 0 if a.isupper() or len(a) <= 4 else re.I
                if re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", report_text, flags):
                    return True
            return False
        rows = [r for r in rows if used(r[0])]
    if not rows:
        raise ValueError("glossary: none of the reader-glossary terms appears in this report")
    return Grid("Glossary", ["Term", "Meaning"], [list(r) for r in rows],
                "Source: reader glossary of the main report (Annex D, reports/source), terms used in this report. "
                "Evidence class: Interpretation.", ["Interpretation"])


ANNEXES = {"price_history": price_history, "financials": financials, "ratios": ratios, "valuation": valuation,
           "scenarios": scenarios, "montecarlo": montecarlo, "peers": peers, "glossary": glossary}


def annex(root: Path, kind: str, bank: str | None, report_text: str | None = None) -> Grid:
    """Build one annex grid from pipeline outputs ({{annex type bank}}); the glossary filters to the report's text."""
    return ANNEXES[kind](root, report_text) if kind == "glossary" else ANNEXES[kind](root, bank)
