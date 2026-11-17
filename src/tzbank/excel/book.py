"""Workbook primitives for the Excel companion model (D104): cell addresses, house formats, named inputs, static blocks.

Colour legend (README sheet): blue font = input; black = formula on the same sheet; green = formula or link that reads
another sheet; grey fill = static value computed in Python (statistics, simulations, data-heavy steps).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName

BLUE, BLACK, GREEN, GREY = "0000FF", "000000", "008000", "595959"
FILL_STATIC = PatternFill("solid", fgColor="E7E6E6")
FILL_HEAD = PatternFill("solid", fgColor="E5F1E8")
FILL_TITLE = PatternFill("solid", fgColor="006B3F")
FILL_INPUT = PatternFill("solid", fgColor="FFF7DF")
THIN = Side(style="thin", color="D3E3D8")

FMT = {"tzs": "#,##0", "pct": "0.00%", "pct4": "0.0000%", "ps": "#,##0.00", "num": "0.0000", "x": "0.00x",
       "int": "0", "date": "yyyy-mm-dd", "num6": "0.000000", "text": "@", "gen": "General"}


def col(c: int) -> str:
    return get_column_letter(c)


def a1(r: int, c: int, absolute: bool = False) -> str:
    return f"${col(c)}${r}" if absolute else f"{col(c)}{r}"


def quote(title: str) -> str:
    return "'" + title.replace("'", "''") + "'"


@dataclass
class Sheet:
    """One worksheet with a row cursor and helpers that apply the house conventions."""

    wb: Workbook
    title: str
    ws: object = None
    row: int = 1
    widths: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ws = self.wb.create_sheet(self.title)
        self.ws.sheet_view.showGridLines = False

    # addresses
    def ref(self, r: int, c: int, other: Sheet | None = None, absolute: bool = True) -> str:
        """Address of (r, c) on this sheet as seen from ``other`` (cross-sheet when different)."""
        s = a1(r, c, absolute)
        return s if other is self else f"{quote(self.title)}!{s}"

    def rng(self, r1: int, c1: int, r2: int, c2: int, other: Sheet | None = None) -> str:
        s = f"{a1(r1, c1, True)}:{a1(r2, c2, True)}"
        return s if other is self else f"{quote(self.title)}!{s}"

    # writing
    def put(self, r: int, c: int, v, kind: str = "auto", fmt: str | None = None, bold: bool = False,
            wrap: bool = False):
        cell = self.ws.cell(row=r, column=c)
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        cell.value = v
        is_f = isinstance(v, str) and v.startswith("=")
        if kind == "auto":
            kind = ("link" if "!" in v else "formula") if is_f else "value"
        if is_f and kind not in ("formula", "link"):
            cell.data_type = "s"  # text that happens to start with '=' (static values, notes) stays text
        color = {"input": BLUE, "formula": BLACK, "link": GREEN, "static": GREY, "value": BLACK, "label": BLACK,
                 "note": GREY, "head": "1F3A2E", "title": "FFFFFF"}[kind]
        cell.font = Font(name="Calibri", size=10 if kind != "title" else 12, color=color,
                         bold=bold or kind in ("head", "title"), italic=kind == "note")
        if kind == "static":
            cell.fill = FILL_STATIC
        elif kind == "head":
            cell.fill = FILL_HEAD
        elif kind == "title":
            cell.fill = FILL_TITLE
        elif kind == "input":
            cell.fill = FILL_INPUT
        if fmt:
            cell.number_format = FMT.get(fmt, fmt)
        if wrap:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        return cell

    def title_block(self, title: str, subtitle: str = "", width: int = 12) -> None:
        for c in range(1, width + 1):
            self.put(1, c, None, "title")
        self.put(1, 1, title, "title")
        if subtitle:
            self.put(2, 1, subtitle, "note")
        self.row = 4

    def heading(self, text: str, width: int = 12, r: int | None = None) -> int:
        r = r or self.row
        for c in range(1, width + 1):
            self.put(r, c, None, "head")
        self.put(r, 1, text, "head")
        self.row = r + 1
        return r

    def note(self, text: str, r: int | None = None, c: int = 1) -> int:
        r = r or self.row
        self.put(r, c, text, "note")
        self.row = r + 1
        return r

    def width(self, c: int, w: float) -> None:
        self.ws.column_dimensions[col(c)].width = w


def name_ok(s: str) -> str:
    """A legal Excel defined name (letters, digits, underscore; not a cell address)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if re.match(r"[A-Za-z_]", s) else f"n_{s}"


def define(wb: Workbook, name: str, sheet: Sheet, r: int, c: int) -> str:
    name = name_ok(name)
    wb.defined_names[name] = DefinedName(name=name, attr_text=f"{quote(sheet.title)}!{a1(r, c, True)}")
    return name


def static_block(sh: Sheet, df, title: str, source: str, script: str, max_rows: int = 2000, width: int = 12) -> int:
    """A grey block of values computed in Python, labelled with the script and the source table."""
    sh.heading(f"Static – computed in Python ({script}): {title}", width)
    sh.note(f"Source: {source}. Values, not formulas: statistics, simulations or data-heavy steps run in Python.")
    cols = [c for c in df.columns if c not in ("register_version", "input_hash")]
    r0 = sh.row
    for j, c in enumerate(cols, 1):
        sh.put(r0, j, c, "label", bold=True)
    rows = df.head(max_rows)
    for i, rec in enumerate(rows[cols].itertuples(index=False, name=None), 1):
        for j, v in enumerate(rec, 1):
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            if isinstance(v, (str,)) and len(v) > 250:
                v = v[:247] + "..."
            sh.put(r0 + i, j, v, "static")
    end = r0 + len(rows)
    if len(df) > max_rows:
        sh.note(f"{len(df) - max_rows} further rows in {source}.", end + 1)
        end += 1
    sh.row = end + 2
    return len(rows)
