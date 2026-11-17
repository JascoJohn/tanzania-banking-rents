"""Excel companion model (D104): structure, tolerance rule, package normalisation, and live input flow vs Python."""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from tzbank.excel import check as CK
from tzbank.excel.recalc import RecalcUnavailable, normalise, recalc

VOLATILE = re.compile(r"\b(INDIRECT|OFFSET|TODAY|NOW|RAND|RANDBETWEEN|CELL|INFO)\(", re.I)


@pytest.fixture(scope="module")
def built():
    from tzbank.excel.model import build_model

    return build_model()


def test_workbook_structure(built):
    wb, checks, live, figs = built
    names = wb.sheetnames
    assert names[:5] == ["README", "Summary", "Inputs", "Data_Panel", "Data_Chapters"]
    assert [n for n in names if n.startswith("Ch") and len(n) == 4] == [f"Ch{i:02d}" for i in range(1, 15)]
    assert sum(n.startswith("Run ") for n in names) == 14  # 12 cases + 2 bridge runs
    assert "R_HIST_NMB_ER_star_terminal_base" in wb.defined_names and "S_CURR_CRDB_value_per_share" in \
        wb.defined_names
    assert not wb._external_links
    n = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.data_type == "f":
                    n += 1
                    assert not VOLATILE.search(str(c.value)), (ws.title, c.coordinate)
                    assert "[" not in str(c.value).split("!")[0] or "'" in str(c.value), (ws.title, c.coordinate)
    assert n > 50_000 and len(checks.items) > 6000
    assert live.live.get("Ch09", 0) > 500 and live.static.get("Ch13", 0) > 1000


def test_tolerance_rule():
    assert CK._ok("ps", 1000.0, 1000.009)[0] and not CK._ok("ps", 10.0, 10.02)[0]
    assert CK._ok("ratio", 0.1, 0.100009)[0] and not CK._ok("ratio", 0.1, 0.10002)[0]
    assert CK._ok("abs", 1e-16, 3e-12)[0] and not CK._ok("abs", 0.0, 1e-6)[0]
    assert CK._ok("text", "Sell", "Sell")[0] and not CK._ok("text", "Sell", "Hold")[0]
    assert not CK._ok("ratio", 0.1, "#VALUE!")[0]


def test_normalise_strips_paths_and_fixes_dates(tmp_path: Path):
    wb = Workbook()
    wb.active["A1"] = 1
    f = tmp_path / "x.xlsx"
    wb.save(f)
    # inject an Excel-style absolute path element and a user name
    data = {}
    with zipfile.ZipFile(f) as z:
        for n in z.namelist():
            data[n] = z.read(n)
    data["xl/workbook.xml"] = data["xl/workbook.xml"].replace(
        b"<workbookPr", b'<x15ac:absPath xmlns:x15ac="x" url="C:\\Users\\someone\\"/><workbookPr', 1)
    data["docProps/core.xml"] = data["docProps/core.xml"].replace(b">openpyxl</dc:creator>", b">someone</dc:creator>")
    assert b"someone" in data["docProps/core.xml"]
    with zipfile.ZipFile(f, "w") as z:
        for n, b in data.items():
            z.writestr(n, b)
    normalise(f, "T", "Jasco John")
    b1 = f.read_bytes()
    normalise(f, "T", "Jasco John")
    assert f.read_bytes() == b1  # idempotent and deterministic
    with zipfile.ZipFile(f) as z:
        wbx, core = z.read("xl/workbook.xml"), z.read("docProps/core.xml")
        assert b"absPath" not in wbx and b"someone" not in core and b"Jasco John" in core
        import xml.etree.ElementTree as ET

        for part in ("docProps/core.xml", "docProps/app.xml", "xl/workbook.xml"):
            ET.fromstring(z.read(part))  # well-formed (an undeclared prefix made Excel refuse the file)
        assert all(i.date_time == (2026, 10, 31, 0, 0, 0) for i in z.infolist())


@pytest.mark.skipif(sys.platform != "win32" and not (shutil.which("soffice") or shutil.which("libreoffice")),
                    reason="no spreadsheet engine to recalculate the workbook")
def test_input_change_flows_to_summary_like_python(built, tmp_path: Path):
    from tzbank.valuation import uncertainty as un
    from tzbank.valuation.context import build_ctx

    wb, _, _, _ = built
    inp = wb["Inputs"]

    def cell(name: str):
        dest = wb.defined_names[name].attr_text.split("!")[1].replace("$", "")
        return inp[dest]

    er = cell("R_CURR_NMB_ER_star_terminal_base")
    er.value = float(er.value) + 0.01
    nim = cell("R_CURR_CRDB_NIM_FY2027_base")
    nim.value = float(nim.value) + 0.005
    src, dst = tmp_path / "in.xlsx", tmp_path / "out.xlsx"
    wb.save(src)
    try:
        recalc(src, dst)
    except RecalcUnavailable:
        pytest.skip("no spreadsheet engine")
    v = load_workbook(dst, data_only=True)["Summary"]
    got = {}
    for k in ("S_CURR_NMB_value_per_share", "S_CURR_CRDB_value_per_share"):
        got[k] = v[wb.defined_names[k].attr_text.split("!")[1].replace("$", "")].value
    ctx = build_ctx()
    c = ctx.cases[("CURR", "B02", "base")]
    roe = c.dr.roe_star + 0.01
    nmb = un.value(c, replace(c.dr, roe_star=roe, payout_star=1 - c.dr.g / roe)) / c.shares
    c2 = ctx.cases[("CURR", "B01", "base")]
    nims = list(c2.dr.NIM)
    nims[1] += 0.005  # FY2027 = second explicit year at 31 Oct 2026
    crdb = un.value(c2, replace(c2.dr, NIM=nims)) / c2.shares
    assert abs(got["S_CURR_NMB_value_per_share"] - nmb) / nmb < 1e-6
    assert abs(got["S_CURR_CRDB_value_per_share"] - crdb) / crdb < 1e-6
    base = load_workbook(Path("outputs/valuation") / CK.OUT_NAME, data_only=True)["Summary"] \
        if (Path("outputs/valuation") / CK.OUT_NAME).exists() else None
    if base is not None:  # the change moved the value
        ref = wb.defined_names["S_CURR_NMB_value_per_share"].attr_text.split("!")[1].replace("$", "")
        assert abs(base[ref].value - got["S_CURR_NMB_value_per_share"]) > 1.0


PUB = Path("outputs/valuation") / CK.OUT_NAME


def _drawings(xlsx: Path) -> dict[str, dict[str, int]]:
    """Charts and pictures per sheet, read from the package (openpyxl does not load charts back)."""
    import posixpath

    out: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(xlsx) as z:
        wbx = z.read("xl/workbook.xml").decode()
        rels = z.read("xl/_rels/workbook.xml.rels").decode()
        target = dict(re.findall(r'<Relationship[^>]*Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
        target.update({a: b for b, a in re.findall(r'<Relationship[^>]*Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)})
        for name, rid in re.findall(r'<sheet [^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wbx):
            sp = posixpath.normpath(posixpath.join("xl", target[rid].lstrip("/").replace("xl/", "")))
            srel = posixpath.join(posixpath.dirname(sp), "_rels", posixpath.basename(sp) + ".rels")
            n = {"chart": 0, "picture": 0}
            if srel in z.namelist():
                for d in re.findall(r'Target="([^"]*drawing[^"]*\.xml)"', z.read(srel).decode()):
                    dp = posixpath.normpath(posixpath.join(posixpath.dirname(sp), d))
                    xml = z.read(dp).decode()
                    n["chart"] += len(re.findall(r"<c:chart ", xml))
                    n["picture"] += len(re.findall(r"<xdr:pic>", xml))
            out[name.replace("&amp;", "&")] = n
    return out


@pytest.mark.skipif(not PUB.exists(), reason="step 20 has not been run")
def test_published_workbook_has_cached_values_and_auto_calc():
    wb = load_workbook(PUB, data_only=True)
    s = wb["Summary"]
    for name in ("S_HIST_NMB_value_per_share", "S_CURR_CRDB_value_per_share", "S_CURR_NMB_target_price_12m",
                 "S_HIST_CRDB_expected_total_return", "S_CURR_NMB_rating", "S_all_runs_converged"):
        v = s[wb.defined_names[name].attr_text.split("!")[1].replace("$", "")].value
        assert v is not None and v != "", name
    assert s[wb.defined_names["S_all_runs_converged"].attr_text.split("!")[1].replace("$", "")].value == "PASS"
    with zipfile.ZipFile(PUB) as z:
        assert 'calcMode="manual"' not in z.read("xl/workbook.xml").decode()
    # every Summary difference is zero with the default inputs
    for row in s.iter_rows(min_row=4):
        for c in row:
            if isinstance(c.value, (int, float)) and s.cell(c.row, 1).value and c.column in (4, 7):
                assert c.value == 0, (c.coordinate, c.value)


@pytest.mark.skipif(not PUB.exists(), reason="step 20 has not been run")
def test_chart_count_per_sheet_matches_the_figure_map():
    import pandas as pd

    fm = pd.read_csv(Path("outputs/valuation/excel_model_figures.csv"))
    got = _drawings(PUB)
    for sheet, g in fm.groupby("sheet"):
        native = int((g["kind"] == "native").sum())
        images = int(((g["kind"] == "image") & g["chart_type"].eq("image")).sum())
        assert got.get(sheet, {}).get("chart", 0) == native, (sheet, got.get(sheet), native)
        assert got.get(sheet, {}).get("picture", 0) == images, (sheet, got.get(sheet), images)
    assert (fm["kind"] == "native").sum() > 30

