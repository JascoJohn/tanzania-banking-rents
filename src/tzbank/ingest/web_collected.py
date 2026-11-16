"""Web-collected values (owner data-collection pass, 2026-11-12; D64): macro forecasts (IMF WEO / REO) and 2026
T-bill auction yields, read by a person from cited web sources.

Path (mirrors D29 / D39): ``data/raw/manual_entries/web_collected.csv`` (raw, MANIFEST-hashed; every row with source
title, URL, page, publication date, access date, vintage, confidence) -> validated here -> ``data/interim/
web_collected.csv`` and ``reports/validation/web_collected_issues.csv``. Gates: required fields and an http(s) URL;
publication date on or before the as-at date (point in time); ratio values within [-0.5, 1]; where the source PDF is
held under ``data/raw/documents``, the value (in percent, one decimal) must be printed on the cited page. A row that
fails a gate is listed and never used. Precedence: lower than the inputs workbook: a value fills only a macro row the
workbook marks TO COLLECT (``fill_macro``), or a series the workbook does not hold.

Derived: nominal GDP growth = (1 + real GDP growth) (1 + CPI inflation) - 1 for the same as-at date and vintage (CPI
as the deflator proxy; labelled Derived).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

REQUIRED = ["item_id", "canonical_name", "as_at", "value", "unit", "source_title", "source_url", "publication_date",
            "access_date", "confidence"]
NOMINAL = "macro_tanzania_nominal_gdp_growth_forecast"


def raw_file(root: Path) -> Path:
    return path("raw", root) / "manual_entries" / "web_collected.csv"


def _in_pdf(root: Path, doc: str, page: str, value: float, unit: str = "ratio") -> bool | None:
    """The value as printed on the cited page: a ratio in percent with one decimal; a level with thousands commas."""
    f = path("documents", root) / doc
    if not doc or not f.exists() or not str(page).isdigit():
        return None
    from tzbank.extract.pdftext import _doc

    t = re.sub(r"\s+", " ", _doc(str(f))[int(page) - 1].get_text())
    return (f"{value * 100:.1f}" if unit == "ratio" else f"{value:,.0f}") in t


def validate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = raw_file(root)
    if not f.exists():
        return pd.DataFrame(), pd.DataFrame(columns=["item_id", "issue", "evidence_class"])
    d = pd.read_csv(f, dtype=str).fillna("")
    issues, ok = [], []
    for r in d.to_dict("records"):
        why = [c for c in REQUIRED if not str(r[c]).strip()]
        if not str(r["source_url"]).startswith("http"):
            why.append("source_url not http(s)")
        try:
            v = float(r["value"])
            pub, asat = pd.Timestamp(r["publication_date"]), pd.Timestamp(r["as_at"])
            pd.Timestamp(r["access_date"])
        except (ValueError, TypeError):
            why.append("value or a date does not parse")
            v, pub, asat = float("nan"), None, None
        if pub is not None and pub > asat:
            why.append(f"published {pub.date()} after the as-at date {asat.date()} (point in time)")
        if r["unit"] == "ratio" and not -0.5 <= v <= 1:
            why.append("ratio outside [-0.5, 1]")
        found = _in_pdf(root, r["source_doc"], r["page"], v, r["unit"]) if not why else None
        if found is False:
            why.append(f"value {v * 100:.1f} not printed on {r['source_doc']} p.{r['page']}")
        r["pdf_check"] = {True: "value found on the cited page", False: "NOT FOUND", None: "PDF not held / not a page"}[
            found]
        if why:
            issues.append({"item_id": r["item_id"], "issue": "; ".join(why), "evidence_class": "Observed"})
        else:
            r["value"] = v
            ok.append(r)
    good = pd.DataFrame(ok)
    if len(good):
        der = []
        for _, g in good.groupby(["as_at", "vintage"]):
            g = g.set_index("canonical_name")
            if {"macro_tanzania_real_gdp_growth_forecast", "macro_tanzania_inflation_forecast_medium_term"} <= set(
                    g.index):
                real = float(g.loc["macro_tanzania_real_gdp_growth_forecast", "value"])
                cpi = float(g.loc["macro_tanzania_inflation_forecast_medium_term", "value"])
                base = g.loc["macro_tanzania_real_gdp_growth_forecast"].to_dict()
                der.append({**base, "item_id": base["item_id"] + "-NOM", "canonical_name": NOMINAL,
                            "value": (1 + real) * (1 + cpi) - 1, "evidence_class": "Derived",
                            "notes": f"(1 + real {real:.3f}) (1 + CPI {cpi:.3f}) - 1; CPI as the GDP deflator proxy",
                            "pdf_check": "derived"})
        good = pd.concat([good, pd.DataFrame(der)], ignore_index=True)
    return good, pd.DataFrame(issues, columns=["item_id", "issue", "evidence_class"])


def load(root: Path | None = None) -> pd.DataFrame:
    f = path("interim", root or find_root()) / "web_collected.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["canonical_name", "as_at", "value"])


def fill_macro(m: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Fill macro rows the workbook marks TO COLLECT (never overwrite a workbook value); append series it lacks."""
    w = load(root)
    w = w[w["canonical_name"].astype(str).str.startswith("macro_")]
    if w.empty:
        return m
    m = m.copy()
    asat = pd.to_datetime(m["As At (Valuation Date)"])
    extra = []
    for r in w.itertuples():
        hit = (m["canonical_name"] == r.canonical_name) & (asat == pd.Timestamp(r.as_at))
        note = (f"{r.source_title} ({r.vintage}; period {r.period}); {r.source_url}; accessed {r.access_date}; "
                f"web_collected {r.item_id} (D64)")
        if hit.any():
            i = m.index[hit][0]
            if pd.notna(m.at[i, "value_numeric"]):
                continue  # the workbook value has precedence
            m.loc[i, ["value_numeric", "Status", "Doc ID", "Confidence", "Notes", "publication_date",
                      "Observation / Auction Date", "evidence_class"]] = [
                float(r.value), "COLLECTED (web, D64)", r.item_id, r.confidence, note, pd.Timestamp(r.publication_date),
                pd.Timestamp(r.as_at), r.evidence_class]
            m.loc[i, ["eligible_historical", "eligible_current"]] = "Y"
        else:
            extra.append({"Row ID": r.item_id, "Category": "Collected", "canonical_name": r.canonical_name,
                          "Variable": r.canonical_name.replace("macro_", "").replace("_", " "),
                          "As At (Valuation Date)": pd.Timestamp(r.as_at), "Unit": r.unit,
                          "Observation / Auction Date": pd.Timestamp(r.as_at),
                          "Status": "COLLECTED (web, D64)", "Doc ID": r.item_id, "Confidence": r.confidence,
                          "Notes": note, "value_raw": r.value, "value_numeric": float(r.value),
                          "publication_date": pd.Timestamp(r.publication_date), "eligible_historical": "Y",
                          "eligible_current": "Y", "evidence_class": r.evidence_class})
    return pd.concat([m, pd.DataFrame(extra)], ignore_index=True) if extra else m


def write_web_collected(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    good, issues = validate(root)
    if good.empty and not raw_file(root).exists():
        return []
    return [write_csv(good, path("interim", root) / "web_collected.csv", require_evidence=True),
            write_csv(issues, path("validation", root) / "web_collected_issues.csv", require_evidence=True)]
