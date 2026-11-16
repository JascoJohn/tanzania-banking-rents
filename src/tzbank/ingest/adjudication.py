"""Inclusion and NPS value adjudications (prompt 02d section 7; D30, D31).

The owner's panel values are never changed. Four values that the PDF cross-check of prompt 02c could not confirm are
listed with both values and the pages in ``reports/inclusion_nps_adjudication.csv``. The owner's decisions live in
``config/inclusion_nps_decisions.csv`` (``reports/`` is regenerated and never edited by hand, CLAUDE.md 3.2); when a
decision is filled, the pipeline applies it to a separate ``adjudicated_value`` column of the processed panel, next to
the untouched ``value``.

Decision codes: blank (open: ``adjudicated_value`` = panel value), ``panel`` (keep the panel value), ``report`` (use
the report value), ``exclude`` (no value), or a number (use it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

# item_id, panel, series key, year, report value, report evidence. Panel values are read from the panels at run time.
ITEMS = [
    {"item_id": "ADJ-01", "panel": "inclusion", "series": "SAV_SACCO_ZNZ_AMT", "year": 2024,
     "item": "SACCO Zanzibar savings 2024 (TZS bn)", "obs": "O0631 (2024 report vintage)", "report_value": None,
     "report_evidence": "2024 Financial Inclusion Report Table 2.6 (p.28) prints no 2024 column and the value 21.76 "
                        "appears nowhere in the 2024 report; the 2025 report prints 21.8 (Table 2.1, p.23, obs "
                        "O0632), which is the latest vintage the panel already uses (D31)"},
    {"item_id": "ADJ-02", "panel": "nps", "series": "13_Table1_Indicators|PANEL: KEY PAYMENT SYSTEM INDICATORS (Body "
                                                     "Table 1)|Licensed non-banks", "year": 2025,
     "item": "Licensed non-banks 2025", "obs": "", "report_value": 72.0,
     "report_evidence": "2025 NPS Annual Report prints 72 (pp.3, 14, 16, 17 Table 1 '59 72', 62); 89 is not printed "
                        "as this indicator in any NPS report"},
    {"item_id": "ADJ-03", "panel": "nps", "series": "03_TACH|Table C3|Volume of USD Cheque processed", "year": 2025,
     "item": "USD cheques processed 2025 (volume)", "obs": "", "report_value": 45_785.0,
     "report_evidence": "2025 NPS Annual Report prints 45,785 on pp.17, 33 and 69; 45,792 is printed nowhere"},
    {"item_id": "ADJ-04", "panel": "nps", "series": "03_TACH|Table C1|Volume of transactions", "year": 2024,
     "item": "EFT volume 2024", "obs": "", "report_value": 20_595_123.0,
     "report_evidence": "2024 NPS Annual Report p.60 prints 21,075,755 (the panel value); the 2025 report restates "
                        "FY2024 as 20,595,123 (p.69). Both vintages kept; the latest is used per D31 (owner, "
                        "prompt 02d 7)"},
]
DECISION_COLUMNS = ["item_id", "owner_decision", "decided_by", "decided_on", "note"]


def decisions_file(root: Path | None = None) -> Path:
    return path("config", root or find_root()) / "inclusion_nps_decisions.csv"


def read_decisions(root: Path | None = None) -> pd.DataFrame:
    f = decisions_file(root)
    if not f.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)
    return pd.read_csv(f, dtype=str).fillna("").reindex(columns=DECISION_COLUMNS, fill_value="")


def resolve(decision: str, panel_value: float, report_value: float | None) -> tuple[float, str]:
    """(adjudicated value, status) for one decision code."""
    d = str(decision or "").strip()
    if d == "":
        return panel_value, "open: panel value used until the owner decides"
    if d.lower() == "panel":
        return panel_value, "decided: panel value"
    if d.lower() == "report":
        if report_value is None:
            raise ValueError("decision 'report' for an item without a printed report value")
        return report_value, "decided: report value"
    if d.lower() == "exclude":
        return np.nan, "decided: excluded"
    try:
        return float(d.replace(",", "")), "decided: owner value"
    except ValueError as e:
        raise ValueError(f"unknown adjudication decision {d!r}") from e


def _key(panel: str) -> tuple[str, str, str]:
    return ("Series_ID", "Year", "value") if panel == "inclusion" else ("series_id", "year", "value")


def adjudication_table(panels: dict[str, pd.DataFrame], long_incl: pd.DataFrame | None,
                       decisions: pd.DataFrame) -> pd.DataFrame:
    dec = decisions.set_index("item_id") if len(decisions) else pd.DataFrame(columns=DECISION_COLUMNS[1:])
    rows = []
    for it in ITEMS:
        if it["panel"] not in panels:
            continue
        sc, yc, vc = _key(it["panel"])
        p = panels[it["panel"]]
        hit = p[(p[sc] == it["series"]) & (p[yc] == it["year"])]
        used = float(hit[vc].iloc[0]) if len(hit) else np.nan
        if it["item_id"] == "ADJ-01" and long_incl is not None:  # the questioned observation is an older vintage
            o = long_incl[long_incl["Obs_ID"] == "O0631"]
            questioned = float(o["Std_Value"].iloc[0]) if len(o) else np.nan
        else:
            questioned = used
        d = dec["owner_decision"].get(it["item_id"], "") if len(dec) else ""
        adj, status = resolve(d, used, it["report_value"])
        rows.append({"item_id": it["item_id"], "panel": it["panel"], "item": it["item"], "series": it["series"],
                     "year": it["year"], "questioned_panel_value": questioned, "panel_value_used": used,
                     "report_value": it["report_value"], "observation": it["obs"],
                     "report_evidence": it["report_evidence"], "owner_decision": d,
                     "decided_by": dec["decided_by"].get(it["item_id"], "") if len(dec) else "",
                     "adjudicated_value": adj, "status": status, "evidence_class": "Observed"})
    return pd.DataFrame(rows)


def apply(panel: pd.DataFrame, name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Add ``adjudicated_value`` and ``adjudication`` to a processed panel; ``value`` is never changed."""
    sc, yc, vc = _key(name)
    out = panel.copy()
    out["adjudicated_value"] = out[vc]
    out["adjudication"] = ""
    for _, t in table[table["panel"] == name].iterrows():
        m = (out[sc] == t["series"]) & (out[yc] == t["year"])
        out.loc[m, "adjudicated_value"] = t["adjudicated_value"]
        out.loc[m, "adjudication"] = f"{t['item_id']}: {t['status']}"
    return out


def adjudicate(panel: pd.DataFrame, name: str, root: Path | None = None,
               long_incl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the owner's decisions to one processed panel (called when the panel is written)."""
    return apply(panel, name, adjudication_table({name: panel}, long_incl, read_decisions(root)))


def write_adjudication(root: Path | None = None) -> list[Path]:
    """reports/inclusion_nps_adjudication.csv from the processed panels and the owner's decisions."""
    root = root or find_root()
    p = path("processed", root)
    panels = {"inclusion": pd.read_csv(p / "inclusion_panel.csv"), "nps": pd.read_csv(p / "nps_panel.csv")}
    li = path("interim", root) / "inclusion_long.csv"
    t = adjudication_table(panels, pd.read_csv(li) if li.exists() else None, read_decisions(root))
    return [write_csv(t, path("reports", root) / "inclusion_nps_adjudication.csv", require_evidence=True)]
