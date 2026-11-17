"""Chapter sheets of the companion workbook (D104): live formula blocks where the logic is deterministic, grey static
blocks for statistics and simulations, and the check items that compare each live cell with the Python table.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from tzbank.excel.book import Sheet, a1, static_block

TOL_REL = 1e-4  # 0.01% (task tolerance); per-share TZS items also pass within TZS 0.01


@dataclass
class Checks:
    items: list = field(default_factory=list)

    def add(self, item: str, sheet: Sheet, r: int, c: int, python, kind: str = "ratio") -> None:
        if python is None or (isinstance(python, float) and not math.isfinite(python)):
            return
        if hasattr(python, "item"):
            python = python.item()
        self.items.append({"item": item, "sheet": sheet.title, "cell": a1(r, c), "python": python, "kind": kind})


@dataclass
class Live:
    """Counts of live (formula) and static cells per chapter sheet, for the summary."""

    live: dict = field(default_factory=dict)
    static: dict = field(default_factory=dict)

    def add(self, sheet: str, n_live: int = 0, n_static: int = 0) -> None:
        self.live[sheet] = self.live.get(sheet, 0) + n_live
        self.static[sheet] = self.static.get(sheet, 0) + n_static


def data_block(ds: Sheet, df: pd.DataFrame, cols: list[str], title: str, source: str, width: int = 14) -> dict:
    """Write input columns to the data sheet; returns {"rows": [excel row per df row], "cols": {name: column}}."""
    ds.heading(f"{title}", width)
    ds.note(f"Source: {source} (inputs to the live formulas on the chapter sheet; values as computed in Python).")
    r0 = ds.row
    for j, c in enumerate(cols, 1):
        ds.put(r0, j, c, "label", bold=True)
    rows = []
    for i, rec in enumerate(df[cols].itertuples(index=False), 1):
        r = r0 + i
        for j, v in enumerate(rec, 1):
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            ds.put(r, j, v, "value", "num6" if isinstance(v, float) and abs(v) < 10 else None)
        rows.append(r)
    ds.row = r0 + len(df) + 2
    return {"rows": rows, "cols": {c: j for j, c in enumerate(cols, 1)}}


def live_table(ch: Sheet, ds: Sheet, df: pd.DataFrame, keys: list[str], inputs: list[str], derived: dict[str, str],
               title: str, source: str, checks: Checks, live: Live, fmt: dict | None = None,
               python_cols: dict | None = None) -> dict:
    """A chapter table: key and input columns linked from the data sheet, derived columns as formulas from the
    template (``{col}`` = the same row's cell). Every derived cell is compared with the Python column."""
    fmt = fmt or {}
    blk = data_block(ds, df, keys + inputs, title, source)
    ch.heading(f"Live: {title}", max(len(keys) + len(inputs) + len(derived), 8))
    ch.note(f"Formulas reproduce {source}. Inputs link to {ds.title} (green); derived columns are formulas (black).")
    cols = keys + inputs + list(derived)
    r0 = ch.row
    for j, c in enumerate(cols, 1):
        ch.put(r0, j, c, "label", bold=True)
    pos = {c: j for j, c in enumerate(cols, 1)}
    out_rows = []
    for i, (_idx, rec) in enumerate(df.iterrows()):
        r = r0 + 1 + i
        dr = blk["rows"][i]
        for c in keys + inputs:
            src = ds.ref(dr, blk["cols"][c], ch)
            if c in keys:
                ch.put(r, pos[c], f"={src}", "link")
            else:
                ch.put(r, pos[c], f"=IF(ISBLANK({src}),\"\",{src})", "link", fmt.get(c, "num6"))
        for c, tpl in derived.items():
            used = re.findall(r"\{(\w+)\}", tpl)
            expr = re.sub(r"\{(\w+)\}", lambda m, r=r: a1(r, pos[m.group(1)]), tpl)
            ins = [a1(r, pos[u]) for u in dict.fromkeys(used)]
            f = f"=IF(COUNT({','.join(ins)})<{len(ins)},\"\",IFERROR({expr},\"\"))"
            ch.put(r, pos[c], f, "formula", fmt.get(c, "num6"))
            pyc = (python_cols or {}).get(c, c)
            if pyc in df.columns:
                checks.add(f"{ch.title} {title}: {c} [{', '.join(str(rec[k]) for k in keys)}]", ch, r, pos[c],
                           rec[pyc], "ratio")
        out_rows.append(r)
    live.add(ch.title, n_live=len(df) * len(derived))
    ch.row = r0 + len(df) + 2
    return {"rows": out_rows, "pos": pos}


def statics(ch: Sheet, tables: dict[str, pd.DataFrame], names: list[str], script: str, live: Live,
            max_rows: int = 400) -> None:
    for n in names:
        if n in tables:
            k = static_block(ch, tables[n], n, f"outputs/tables/{n}.csv", script, max_rows)
            live.add(ch.title, n_static=k * len([c for c in tables[n].columns
                                                 if c not in ("register_version", "input_hash")]))
