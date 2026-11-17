"""Report figures in the companion workbook (D104 amended): every exhibit of the four reports, on its chapter sheet,
numbered and titled as in the report.

- ``native``: an Excel chart (line, column, bar, scatter) drawn from a data block on the chapter sheet. A data cell
  whose value equals exactly one live workbook cell's Python value is a link to that cell (green), so the chart moves
  with the inputs; other cells are the plotted values from the figure's data file (grey, static).
- ``heatmap``: a colour scale on a grid (the live sensitivity grids and iso-value surfaces carry their own).
- ``image``: the report PNG, captioned "Static image from report", where a native chart would not be faithful
  (diagrams, waterfalls, distributions, heavily annotated charts).

The map of figures is written to ``outputs/valuation/excel_model_figures.csv``.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pandas as pd
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule

from tzbank.excel.book import Sheet, a1

BANK = {"B01": "CRDB", "B02": "NMB", "B03": "DCB", "B04": "MBP", "B05": "MKCB", "B06": "MCB", "B07": "MUCOBA"}
COLOR = {"CRDB": "2A78D6", "NMB": "EB6834", "DCB": "1BAF7A", "MBP": "EDA100", "MKCB": "E87BA4", "MCB": "008300",
         "MUCOBA": "6250D6"}
PALETTE = ["006B3F", "E6B323", "66A88A", "06452E", "8A6A00", "6B736F", "2A78D6", "EB6834", "1BAF7A", "6250D6"]
REPORT = {"main": "Main report", "nmb": "NMB report", "crdb": "CRDB report", "foreign": "Investor note"}
CAPTION = "Static image from report"
HEAT = ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50,
                      mid_color="FFEB84", end_type="max", end_color="63BE7B")

# kind, x, group, values (y for scatter), optional row filter (column, values)
SPECS: dict[str, dict] = {
    "ch01_cr2": dict(kind="line", x="year", group="measure", values=["cr2"]),
    "main/02": dict(kind="line", x="year", group="series", values=["cr2_asset_share", "cr2_pbt_share"]),
    "main/03": dict(kind="column", x="measure", values=["share"]),
    "foreign/02": dict(kind="column", x="measure", values=["share"]),
    "ch02_loan_mix": dict(kind="bar", x="category", group="book", values=["share"]),
    "main/05": dict(kind="column", x="bank", values=["binding_headroom_mean"]),
    "ch03_pb_roe": dict(kind="scatter", x="roe", group="bank_id", values=["pb"]),
    "ch04_roe_by_bank": dict(kind="line", x="fiscal_year", group="bank_id", values=["roe"]),
    "ch13_p21_dupont": dict(kind="column", x="effect", values=["nmb_value", "crdb_value"]),
    "ch04_4_5_cost_to_income": dict(kind="line", x="fiscal_year", group="bank_id", values=["cost_to_income"]),
    "ch04_4_1_nonii_share": dict(kind="line", x="fiscal_year", group="bank_id", values=["nonii_share"]),
    "ch04_4_4_fees_fx": dict(kind="line", x="fiscal_year", group="bank_id", values=["fee_to_avg_assets",
                                                                                     "fx_to_avg_assets"]),
    "ch04_4_3_yields": dict(kind="line", x="fiscal_year", group="bank_id", values=["loan_yield", "iea_yield"]),
    "ch04_4_6_risk": dict(kind="line", x="fiscal_year", group="bank_id", values=["cost_of_risk",
                                                                                  "risk_adjusted_nim"]),
    "ch05_excess_returns": dict(kind="line", x="fiscal_year", group="bank_id", values=["er"]),
    "main/08": dict(kind="column", x="bank", values=["mean_er"]),
    "ch05_coe_annual": dict(kind="line", x="year", values=["coe", "coe_minus_200bp", "coe_plus_200bp", "y_long"]),
    "ch06_cost_of_deposits": dict(kind="line", x="fiscal_year", group="bank_id", values=["cost_of_deposits"]),
    "main/09": dict(kind="line", x="fiscal_year", group="bank_id", values=["gdfa_roe_pts",
                                                                           "net_franchise_central_roe_pts"]),
    "main/10": dict(kind="column", x=["bank", "casa_repriced"], values=["roe_change_pts"]),
    "ch07_leverage": dict(kind="line", x="fiscal_year", group="bank_id", values=["tier1_headroom",
                                                                                  "total_headroom"]),
    "main/11": dict(kind="column", x=["bank", "management_buffer"], values=["payout_max_steady_state",
                                                                            "payout_actual"]),
    "ch13_p21_surplus": dict(kind="column", x=["bank", "hurdle"], values=["drag_share_of_profit"]),
    "ch07_rore": dict(kind="line", x="fiscal_year", group="bank_id", values=["rore_primary", "rore_adjusted"]),
    "main/12": dict(kind="line", x="fiscal_year", group="bank_id", values=["cost_of_risk"]),
    "ch08_asset_quality": dict(kind="line", x="fiscal_year", group="bank_id", values=["npl_ratio"]),
    "main/13": dict(kind="line", x="fiscal_year", group="bank_id", values=["return_on_rwa", "risk_adjusted_rorwa"]),
    "ch09_valuation_range": dict(kind="bar", x=["bank", "item"], values=["value_per_share"]),
    "ch13_dec_interp_weights": dict(kind="line", x="weight_on_B", group=["valuation", "bank"],
                                    values=["value_per_share"]),
    "ch14_roe_lambda_grid": dict(kind="heatmap", x="lambda", group="roe_star_shift", values=["value_to_price"],
                                 panels=["valuation", "bank"]),
    "ch10_iso_value": dict(kind="line", x="lambda", group=["valuation", "bank"], values=["er_star_at_price"]),
    "ch13_peers_scatter": dict(kind="scatter", x="roe", group="valuation", values=["pb"]),
    "main/18": dict(kind="column", x="bank", values=["market_total_return_log", "earnings_and_time_log",
                                                     "discount_rate_log", "residual_log"]),
    "main/19": dict(kind="line", x="stress_weight", group=["valuation", "bank"], values=["prob_value_above_price"]),
    "main/21": dict(kind="column", x="bank", values=["tzs_return", "usd_return"]),
    "foreign/07": dict(kind="column", x="bank", values=["tzs_return", "usd_return"]),
    "main/22": dict(kind="line", x="depreciation", group="bank", values=["expected_usd_return", "usd_hurdle"]),
    "foreign/09": dict(kind="line", x="depreciation", group="bank", values=["expected_usd_return", "usd_hurdle"]),
    "ch12_capacity": dict(kind="column", x=["bank", "position_usd"], values=["days_to_build_or_exit"]),
    "ch13_dec_income_vs_return": dict(kind="column", x=["bank", "holding_period"], values=["tsr_annualised",
                                                                                          "tbill_annualised"]),
    "ch14_split_study": dict(kind="line", x="trading_date", group="ticker", values=["turnover"]),
    "ch14_seasonality": dict(kind="column", x="month", group="series", values=["mean_return"]),
    "ch14_rolling_correlation": dict(kind="line", x="trading_date", group="window", values=["correlation"]),
    "nmb/03": dict(kind="bar", x="category", values=["BoT commercial banks", "NMB"]),
    "crdb/03": dict(kind="bar", x="category", values=["BoT commercial banks", "CRDB"]),
    "nmb/05": dict(kind="scatter", x="roe", group="group", values=["pb"]),
    "crdb/05": dict(kind="scatter", x="roe", group="group", values=["pb"]),
    "foreign/05": dict(kind="scatter", x="roe", group="group", values=["pb"]),
    "nmb/09": dict(kind="line", x="fiscal_year", group="series", values=["pb"]),
    "crdb/09": dict(kind="line", x="fiscal_year", group="series", values=["pb"]),
}
# not faithful as native charts: diagrams, waterfalls, distributions, annotated price and broker charts
IMAGES = {"ch13_p21_thesis", "ch04_4_0_change_waterfall_NMB", "ch04_4_0_change_waterfall_CRDB", "ch04_4_1_fee_mix",
          "main/15", "ch13_p21_roe_bridge", "ch13_dec_mc_inputs", "ch11_monte_carlo", "ch13_p21_rating_map",
          "ch13_p21_transmission", "main/20", "ch13_dec_analyst_compare", "ch14_return_attribution",
          "ch14_event_windows", "ch13_price_lines", "nmb/06", "crdb/06", "nmb/10", "crdb/10"}


def inventory(root: Path) -> list[dict]:
    """Every report exhibit (id, title, report numbers, chapter sheet), from the report sources and build logs."""
    from tzbank.reporting.report_source import parse_parts, source_files

    src_dir = root / "reports" / "source"
    out: dict[str, dict] = {}
    for rep in ("main", "nmb", "crdb", "foreign"):
        files = source_files(src_dir, rep) if src_dir.exists() else []
        logf = root / "outputs" / "reports" / f"{rep}_build.json"
        if not files or not logf.exists():
            continue
        numbers = [e["number"] for e in json.loads(logf.read_text(encoding="utf-8"))["exhibits"]]
        chap = ""
        seq: list[tuple[str, str, str]] = []

        def walk(ns, seq=seq) -> None:
            nonlocal chap
            for n in ns:
                if n.kind in ("part", "chapter"):
                    chap = n.label
                if n.kind == "exhibit":
                    seq.append((n.attrs["id"], n.attrs["title"], chap))
                walk(n.children)
        walk(parse_parts(files).nodes)
        for i, (fid, title, ch) in enumerate(seq):
            no = numbers[i] if i < len(numbers) else None
            label = f"{REPORT[rep]}, Exhibit {no}" if no else f"{REPORT[rep]}, page-1 chart"
            sheet = "Ch13"
            if rep == "main" and ch.startswith("Chapter "):
                sheet = f"Ch{int(ch.split()[1]):02d}"
            if fid in out:
                if label not in out[fid]["labels"]:
                    out[fid]["labels"].append(label)
                continue
            out[fid] = {"id": fid, "title": title, "labels": [label], "sheet": sheet}
    return list(out.values())


def _key(v: float) -> str:
    return f"{v:.12g}"


def live_index(items: list[dict]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for it in items:
        v = it["python"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        if v == round(v) or abs(v) < 1e-12:
            continue
        sheet = "'" + it["sheet"].replace("'", "''") + "'"
        idx.setdefault(_key(v), []).append(f"{sheet}!{it['cell'].replace('$', '')}")
    return idx


def _label(row, cols: list[str]) -> str:
    vals = []
    for c in cols:
        v = row[c]
        v = BANK.get(v, v)
        if isinstance(v, float) and v == round(v):
            v = int(v)
        vals.append(str(v))
    return " ".join(vals)


def _wide(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    x = spec["x"] if isinstance(spec["x"], list) else [spec["x"]]
    vals = spec["values"]
    d = df.copy()
    d["_x"] = d.apply(lambda r: _label(r, x), axis=1) if len(x) > 1 else d[x[0]]
    if len(x) == 1 and d["_x"].dtype == object:
        d["_x"] = d["_x"].map(lambda v: BANK.get(v, v))
    grp = spec.get("group")
    if not grp:
        w = d[["_x", *vals]].drop_duplicates("_x").set_index("_x")
    else:
        g = grp if isinstance(grp, list) else [grp]
        d["_g"] = d.apply(lambda r: _label(r, g), axis=1)
        w = d.pivot_table(index="_x", columns="_g", values=vals, aggfunc="first")
        w.columns = [c[1] if len(vals) == 1 else f"{c[1]} {c[0]}" for c in w.columns]
    w = w.dropna(axis=1, how="all")
    try:
        w = w.sort_index()
    except TypeError:
        pass
    w.index.name = "x" if len(x) > 1 else x[0]
    return w


def _put_block(sh: Sheet, w: pd.DataFrame, idx: dict, r0: int, c0: int) -> tuple[int, int]:
    """Write a wide data block; returns (rows written incl. header, cells linked to live values)."""
    sh.put(r0, c0, w.index.name, "label", bold=True)
    for j, c in enumerate(w.columns, 1):
        sh.put(r0, c0 + j, str(c), "label", bold=True)
    links = 0
    for i, (x, row) in enumerate(w.iterrows(), 1):
        xv = x.item() if hasattr(x, "item") else x
        if isinstance(xv, pd.Timestamp):
            xv = xv.to_pydatetime()
        sh.put(r0 + i, c0, xv, "value")
        for j, v in enumerate(row, 1):
            v = v.item() if hasattr(v, "item") else v
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            hit = idx.get(_key(float(v))) if isinstance(v, (int, float)) else None
            if hit and len(hit) == 1:
                sh.put(r0 + i, c0 + j, f"={hit[0]}", "link", "num6")
                links += 1
            else:
                sh.put(r0 + i, c0 + j, v, "static", "num6" if isinstance(v, float) else None)
    return len(w) + 1, links


def _colour(name: str, j: int) -> str:
    for k, v in COLOR.items():
        if name.split(" ")[0] == k or name.startswith(k + " "):
            return v
    return PALETTE[j % len(PALETTE)]


def _chart(sh: Sheet, spec: dict, w: pd.DataFrame, r0: int, c0: int, title: str, anchor: str):
    n = len(w)
    kind = spec["kind"]
    if kind == "scatter":
        ch = ScatterChart()
        ch.style = 13
        xs = Reference(sh.ws, min_col=c0, min_row=r0 + 1, max_row=r0 + n)
        for j, c in enumerate(w.columns, 1):
            ys = Reference(sh.ws, min_col=c0 + j, min_row=r0, max_row=r0 + n)
            s = Series(ys, xs, title_from_data=True)
            s.marker.symbol = "circle"
            s.graphicalProperties.line.noFill = True
            col_ = _colour(str(c), j - 1)
            s.marker.graphicalProperties.solidFill = col_
            s.marker.graphicalProperties.line.solidFill = col_
            ch.series.append(s)
    else:
        ch = LineChart() if kind == "line" else BarChart()
        if kind in ("bar", "column"):
            ch.type = "bar" if kind == "bar" else "col"
            ch.grouping = "clustered"
        data = Reference(sh.ws, min_col=c0 + 1, max_col=c0 + len(w.columns), min_row=r0, max_row=r0 + n)
        ch.add_data(data, titles_from_data=True)
        ch.set_categories(Reference(sh.ws, min_col=c0, min_row=r0 + 1, max_row=r0 + n))
        for j, s in enumerate(ch.series):
            col_ = _colour(str(w.columns[j]), j)
            if kind == "line":
                s.graphicalProperties.line.solidFill = col_
                s.graphicalProperties.line.width = 19050
                s.smooth = False
            else:
                s.graphicalProperties.solidFill = col_
                s.graphicalProperties.line.solidFill = col_
    ch.title = title[:250]
    ch.height, ch.width = 8.5, 17.0
    ch.legend.position = "b"
    sh.ws.add_chart(ch, anchor)
    return ch


def _image(sh: Sheet, png: Path, anchor: str, max_w: int = 1000) -> tuple[int, int]:
    from PIL import Image as PILImage

    with PILImage.open(png) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        w_, h_ = im.width, im.height
    buf.seek(0)
    img = XLImage(buf)
    scale = 640 / w_
    img.width, img.height = int(w_ * scale), int(h_ * scale)
    sh.ws.add_image(img, anchor)
    return img.width, img.height


def add_figures(root: Path, chs: dict[str, Sheet], items: list[dict]) -> list[dict]:
    """Add every report figure to its chapter sheet; returns the figure map rows."""
    from tzbank.reporting.build_reports import exhibit_index, resolve

    inv = inventory(root)
    if not inv:
        return []
    eidx = exhibit_index(root)
    idx = live_index(items)
    rows: list[dict] = []
    started: set[str] = set()
    for f in inv:
        sh = chs.get(f["sheet"]) or chs["Ch13"]
        if sh.title not in started:
            sh.row += 1
            sh.heading("Report figures (numbered and titled as in the reports)", 12)
            sh.note("Native Excel charts draw from the data block beside them: green cells link live workbook values, "
                    "grey cells are the plotted values from the figure's data file. Images are static copies of the "
                    "report figure.")
            started.add(sh.title)
        try:
            r = resolve(root, f["id"], eidx)
        except Exception:  # noqa: BLE001 - an unresolvable exhibit is reported in the map, not fatal
            r = {}
        data = Path(root / r["data"]) if r.get("data") else None
        png = Path(root / r["file"]) if r.get("file") else None
        spec = SPECS.get(f["id"])
        kind = "image" if f["id"] in IMAGES or spec is None else ("heatmap" if spec["kind"] == "heatmap" else "native")
        title = f"{'; '.join(f['labels'])}: {f['title']}"
        r0 = sh.row
        sh.put(r0, 1, title, "label", bold=True)
        row = {"figure": "; ".join(f["labels"]), "exhibit_id": f["id"], "title": f["title"], "sheet": sh.title,
               "kind": kind, "chart_type": "", "source_range": "", "live_links": 0,
               "data_file": str(r.get("data") or ""), "evidence_class": "Derived"}
        if kind == "native" and data is not None and data.exists():
            df = pd.read_csv(data, low_memory=False)
            w = _wide(df, spec)
            if spec["kind"] == "scatter":
                n_rows, links = _scatter_block(sh, df, spec, idx, r0 + 1)
                c_last = 1 + 2 * df[spec["group"]].nunique() if spec.get("group") else 2
                w_plot = None
            else:
                n_rows, links = _put_block(sh, w, idx, r0 + 1, 1)
                c_last = 1 + len(w.columns)
                w_plot = w
            anchor = a1(r0 + 1, c_last + 2)
            if w_plot is not None:
                _chart(sh, spec, w_plot, r0 + 1, 1, title, anchor)
            else:
                _scatter_chart(sh, df, spec, r0 + 1, title, anchor)
            row.update(chart_type=spec["kind"], source_range=f"{sh.title}!{a1(r0 + 1, 1)}:{a1(r0 + n_rows, c_last)}",
                       live_links=links)
            sh.row = r0 + 1 + max(n_rows, 19) + 2
        elif kind == "heatmap" and data is not None and data.exists():
            df = pd.read_csv(data, low_memory=False)
            rr = r0 + 1
            first = rr
            c_last = 1
            for key, g in df.groupby(spec["panels"]):
                sh.put(rr, 1, " ".join(BANK.get(k, str(k)) for k in key), "label", bold=True)
                w = _wide(g, {k: v for k, v in spec.items() if k != "kind"})
                n_rows, links = _put_block(sh, w, idx, rr + 1, 1)
                rng = f"{a1(rr + 2, 2)}:{a1(rr + n_rows, 1 + len(w.columns))}"
                sh.ws.conditional_formatting.add(rng, HEAT)
                row["live_links"] += links
                c_last = max(c_last, 1 + len(w.columns))
                rr += n_rows + 2
            row.update(chart_type="colour scale", source_range=f"{sh.title}!{a1(first, 1)}:{a1(rr - 2, c_last)}")
            sh.row = rr + 1
        else:
            row["kind"] = "image"
            if png is not None and png.exists():
                sh.put(r0 + 1, 1, CAPTION, "note")
                _, h = _image(sh, png, a1(r0 + 2, 1))
                row.update(chart_type="image", source_range=str(r.get("file")))
                sh.row = r0 + 2 + int(h / 20) + 3
            else:
                sh.put(r0 + 1, 1, "Figure file not available in this checkout.", "note")
                sh.row = r0 + 3
        rows.append(row)
    return rows


def _scatter_block(sh: Sheet, df: pd.DataFrame, spec: dict, idx: dict, r0: int) -> tuple[int, int]:
    """Scatter data: one (x, y) column pair per group."""
    grp, x, y = spec.get("group"), spec["x"], spec["values"][0]
    groups = [(None, df)] if not grp else list(df.groupby(grp))
    links, n = 0, 0
    for j, (g, d) in enumerate(groups):
        c = 1 + 2 * j
        name = BANK.get(g, str(g)) if g is not None else y
        sh.put(r0, c, f"{name} {x}", "label", bold=True)
        sh.put(r0, c + 1, f"{name}", "label", bold=True)
        d = d[[x, y]].dropna()
        for i, (xv, yv) in enumerate(d.itertuples(index=False), 1):
            for k, v in ((c, xv), (c + 1, yv)):
                v = float(v)
                hit = idx.get(_key(v))
                if hit and len(hit) == 1:
                    sh.put(r0 + i, k, f"={hit[0]}", "link", "num6")
                    links += 1
                else:
                    sh.put(r0 + i, k, v, "static", "num6")
        n = max(n, len(d))
    return n + 1, links


def _scatter_chart(sh: Sheet, df: pd.DataFrame, spec: dict, r0: int, title: str, anchor: str) -> None:
    grp, x, y = spec.get("group"), spec["x"], spec["values"][0]
    groups = [(None, df)] if not grp else list(df.groupby(grp))
    ch = ScatterChart()
    ch.style = 13
    for j, (g, d) in enumerate(groups):
        c = 1 + 2 * j
        m = len(d[[x, y]].dropna())
        xs = Reference(sh.ws, min_col=c, min_row=r0 + 1, max_row=r0 + m)
        ys = Reference(sh.ws, min_col=c + 1, min_row=r0, max_row=r0 + m)
        s = Series(ys, xs, title_from_data=True)
        s.marker.symbol = "circle"
        s.graphicalProperties.line.noFill = True
        col_ = _colour(BANK.get(g, str(g)) if g is not None else "", j)
        s.marker.graphicalProperties.solidFill = col_
        s.marker.graphicalProperties.line.solidFill = col_
        ch.series.append(s)
    ch.title = title[:250]
    ch.height, ch.width = 8.5, 17.0
    ch.x_axis.title = x
    ch.y_axis.title = y
    ch.legend.position = "b"
    sh.ws.add_chart(ch, anchor)


def heat(sh: Sheet, r1: int, c1: int, r2: int, c2: int) -> str:
    """Colour scale on a live grid; returns its address."""
    rng = f"{a1(r1, c1)}:{a1(r2, c2)}"
    sh.ws.conditional_formatting.add(rng, HEAT)
    return f"{sh.title}!{rng}"
