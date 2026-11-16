"""BoT research database intake (decision D40; prompt 02f section 1; blueprint §6 data audit: sector totals, rates,
credit by activity).

The owner's panel comes in two raw, immutable workbooks built from the BoT Annual Reports:

- ``Tanzania_Banking_Research_Database.xlsx`` (v1): reports FY2002/03-FY2014/15, observations 1993 to June 2015;
- ``Tanzania_Banking_Research_Database_v2.xlsx`` (v2): reports FY2015/16-FY2024/25, observations 2006 to June 2025.

Both are read from their ``Research_Database`` sheet (one cleaned value per observation, latest valid vintage).
Each row is mapped to a harmonised key: topic (the BoT appendix table, e.g. A3.13 interest-rate structure), a
normalised series label (currency prefix, plural and punctuation differences removed), period end and frequency
class. Where both workbooks hold the same key, v2 (the later report vintage) supersedes v1 (D40, consistent with D31);
every value conflict (difference > 0.5%) is listed in ``reports/validation/bot_v1_v2_diff.csv``. v1-only rows are
kept. The merged rows are written in the BoT template schema to ``data/raw``-independent
``data/interim/bot_research_database.csv`` and stacked into ``data/interim/bot_long.csv`` by ``tzbank.ingest.bot``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import write_csv

V1 = "Tanzania_Banking_Research_Database.xlsx"
V2 = "Tanzania_Banking_Research_Database_v2.xlsx"
TOL = 0.005

TOPICS = [  # (regex on v1 Indicator_Group or v2 Dataset, topic, template category)
    (r"interest rate structure|A3\.13", "interest_rates", "rates"),
    (r"yields at treasury bill auctions|A3\.14", "tbill_auction_yields", "rates"),
    (r"discount rate and minimum reserve|A3\.15", "discount_and_reserve_ratio", "reserve_requirements"),
    (r"lending by (borrowing sector|economic activity)|A3\.11", "credit_by_activity", "credit_by_activity"),
    (r"lending and (holding of )?securities|A3\.10", "lending_and_securities", "sector_totals"),
    (r"commercial banks? -? ?assets \(A3\.6\)|^Commercial Banks Assets$", "bank_assets", "sector_totals"),
    (r"liabilities|A3\.7", "bank_liabilities", "sector_totals"),
    (r"domestic assets|A3\.9", "bank_domestic_assets", "sector_totals"),
    (r"deposits \(A3\.12\)|^Commercial Banks Deposits$", "bank_deposits", "sector_totals"),
    (r"liquidity|A3\.8", "bank_liquidity", "sector_totals"),
    (r"financial soundness", "financial_soundness", "sector_totals"),
    (r"banking sector structure", "sector_structure", "sector_totals"),
    (r"treasury bills auction|A2\.2", "tbill_auction_sales", "other"),
    (r"treasury bills outstanding", "tbill_holdings", "other"),
    (r"stocks, bonds|government securities", "government_securities", "other"),
    (r"monetary survey|A3\.1\b", "monetary_survey", "other"),
    (r"GDP", "gdp", "other"),
    (r"consumer price|CPI|inflation", "cpi", "other"),
    (r"exchange rate|A4\.7", "exchange_rates", "other"),
    (r"reserve", "foreign_reserves", "other"),
    (r"agent", "agent_banking", "other"),
    (r"credit reference", "credit_reference", "other"),
    (r"bureau|foreign exchange retail|A7\.4", "fx_market", "other"),
]
DIFF_COLUMNS = ["topic", "series_key", "frequency_class", "period_end", "v1_value", "v2_value", "rel_diff", "v1_label",
                "v2_label", "unit_v1", "unit_v2", "v1_source", "v2_source", "status", "evidence_class"]


def topic_of(group: str) -> tuple[str, str]:
    g = str(group or "")
    if re.search(r"narrative", g, re.I):
        return "narrative", "sector_totals"
    for rx, t, cat in TOPICS:
        if re.search(rx, g, re.I):
            if t == "gdp":  # GDP tables differ by price base and measure: never matched across them
                t = "gdp_" + re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_")[:50]
            return t, cat
    return re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_")[:40] or "other", "other"


def norm_label(text: str) -> str:
    """Harmonised series label: currency prefix, plurals, hyphens and punctuation removed."""
    t = str(text or "").lower().replace("domestic currency", "dom").replace("foreign currency", "fx")
    t = re.sub(r"(\d)\s*-?\s*(years?|months?|days?)", r"\1 \2", t)
    t = t.replace("1year", "1 year").replace("days and above", "and above").replace("reverse repo", "reverse_repo")
    words = re.findall(r"[a-z0-9]+", t)
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    return "_".join(w for w in words if w not in {"of", "the", "and", "rate", "s"})


def freq_class(freq: str) -> str:
    f = str(freq or "").lower()
    if "month" in f:
        return "monthly"
    if "quarter" in f:
        return "quarterly"
    if "auction" in f:
        return "event"
    if "june" in f or "financial-year" in f or "financial year" in f:
        return "fy_june"
    return "annual"


UNIT_MAP = {"tzs million": ("TZS million", 1.0), "millions of tzs": ("TZS million", 1.0),
            "tzs billion": ("TZS billion", 1.0), "percent": ("ratio", 0.01), "% per annum": ("ratio", 0.01),
            "%": ("ratio", 0.01), "per cent": ("ratio", 0.01), "count": ("count", 1.0)}


def unit_of(unit: str) -> tuple[str, float]:
    u = str(unit or "").strip().lower()
    if u in UNIT_MAP:
        return UNIT_MAP[u]
    if "percent" in u or u.startswith("%") or "per annum" in u:
        return "ratio", 0.01
    if "million" in u and "usd" in u:
        return "USD", 1e6
    if "number" in u or "branches" in u or "institution" in u or "count" in u:
        return "count", 1.0
    if "index" in u:
        return "index", 1.0
    if "tzs per" in u:
        return "number", 1.0
    return "number", 1.0


def read_v1(file: Path) -> pd.DataFrame:
    d = pd.read_excel(file, sheet_name="Research_Database")
    lab = np.where(d["Dimension_Value"].notna() & (d["Dimension_Value"].astype(str) != "NA"),
                   d["Variable"].astype(str).str.extract(r"^(dom|fx)_", expand=False).fillna("") + " "
                   + d["Dimension_Value"].astype(str), d["Original_Source_Label"].fillna(d["Variable"]).astype(str))
    t = d["Indicator_Group"].map(topic_of)
    return pd.DataFrame({
        "workbook": "v1", "topic": [x[0] for x in t], "category": [x[1] for x in t], "label": lab,
        "source_label": d["Original_Source_Label"], "period_end": pd.to_datetime(d["Obs_Date"], errors="coerce"),
        "frequency": d["Frequency"], "value": pd.to_numeric(d["Value"], errors="coerce"), "unit": d["Unit"],
        "source_report": "BoT Annual Report " + d["Source_Report_FY"].astype(str),
        "file": d["Source_Document_File"], "table": d["Table_No"].astype(str), "page": d["PDF_Page"],
        "flag": d["Validation_Flag"], "group": d["Indicator_Group"]})


def read_v2(file: Path) -> pd.DataFrame:
    d = pd.read_excel(file, sheet_name="Research_Database")
    t = d["Dataset"].map(topic_of)
    return pd.DataFrame({
        "workbook": "v2", "topic": [x[0] for x in t], "category": [x[1] for x in t], "label": d["Indicator"],
        "source_label": d["Indicator"], "period_end": pd.to_datetime(d["Period end date"], errors="coerce"),
        "frequency": d["Frequency"], "value": pd.to_numeric(d["Value"], errors="coerce"), "unit": d["Unit"],
        "source_report": d["Source document"], "file": d["Original file name"], "table": d["Table number"].astype(str),
        "page": d["Page number"], "flag": d["Selection status"], "group": d["Dataset"]})


def load_crosswalk(root: Path) -> dict[tuple[str, str], str]:
    """config/bot_series_crosswalk.csv (prompt 10 task 1): (topic, normalised label variant) -> canonical label."""
    f = root / "config" / "bot_series_crosswalk.csv"
    if not f.exists():  # an input, never generated: a missing file would silently un-harmonise the merge
        raise FileNotFoundError(f"{f} is missing (prompt 10 task 1 input; restore it from git)")
    x = pd.read_csv(f)
    x = x[~x["status"].astype(str).str.upper().str.startswith("REJECTED")]
    return {(r.topic, r.label_variant): r.canonical_label for r in x.itertuples()}


def collisions(df: pd.DataFrame, xw: dict) -> pd.DataFrame:
    """Within one workbook, two labels mapped to the same canonical key for the same period (a crosswalk check)."""
    h = harmonise(df, xw)
    h = h[h["series_key"].isin({f"{t}|{c}" for (t, _), c in xw.items()})]  # keys the crosswalk creates
    key = ["workbook", "series_key", "frequency_class", "period_end"]
    d = h[h.duplicated(key, keep=False)].sort_values(key)
    rows = []
    for k, g in d.groupby(key):
        v = g["value"].to_numpy()
        rows.append({"workbook": k[0], "series_key": k[1], "frequency_class": k[2], "period_end": k[3].date(),
                     "labels": " | ".join(g["label"].astype(str).unique()), "values": " | ".join(f"{x:,.1f}" for x in v),
                     "agree": bool(np.allclose(v, v[0], rtol=TOL)), "evidence_class": "Observed"})
    return pd.DataFrame(rows, columns=["workbook", "series_key", "frequency_class", "period_end", "labels", "values",
                                       "agree", "evidence_class"])


def harmonise(df: pd.DataFrame, xw: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    out = df.copy()
    pct = out["unit"].map(lambda u: unit_of(u)[0] == "ratio")
    lab = out["label"].map(norm_label)
    if xw:
        lab = pd.Series([xw.get((t, x), x) for t, x in zip(out["topic"], lab, strict=True)], index=out.index)
    out["series_key"] = (out["topic"] + "|" + lab
                         + np.where(pct & ~out["topic"].isin(["interest_rates", "tbill_auction_yields",
                                                              "discount_and_reserve_ratio", "financial_soundness"]),
                                    "|pct", ""))
    out["frequency_class"] = out["frequency"].map(freq_class)
    return out


def merge(v1: pd.DataFrame, v2: pd.DataFrame, xw: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(merged rows, diff) - v2 supersedes v1 on the same series key, frequency class and period end (D40); with the
    crosswalk, label variants share one canonical key before matching."""
    key = ["series_key", "frequency_class", "period_end"]
    a = harmonise(v1, xw).drop_duplicates(key, keep="last")
    b = harmonise(v2, xw).drop_duplicates(key, keep="last")
    both = a.merge(b, on=key, suffixes=("_v1", "_v2"))
    ua, ub = both["unit_v1"].map(unit_of), both["unit_v2"].map(unit_of)
    x1 = both["value_v1"] * [u[1] for u in ua]
    x2 = both["value_v2"] * [u[1] for u in ub]
    rel = (x2 - x1) / x1.abs().where(x1 != 0)
    same = np.isclose(x1, x2, rtol=TOL, atol=1e-9) | (x1.isna() & x2.isna())
    diff = pd.DataFrame({
        "topic": both["topic_v2"], "series_key": both["series_key"], "frequency_class": both["frequency_class"],
        "period_end": both["period_end"].dt.date, "v1_value": both["value_v1"], "v2_value": both["value_v2"],
        "rel_diff": rel, "v1_label": both["label_v1"], "v2_label": both["label_v2"], "unit_v1": both["unit_v1"],
        "unit_v2": both["unit_v2"], "v1_source": both["source_report_v1"] + " p." + both["page_v1"].astype(str),
        "v2_source": both["source_report_v2"] + " p." + both["page_v2"].astype(str),
        "status": np.where(same, "same value", "CONFLICT: v2 supersedes (later vintage, D40)"),
        "evidence_class": "Observed"})
    keys_b = set(map(tuple, b[key].astype(str).to_numpy()))
    a_only = a[[tuple(r) not in keys_b for r in a[key].astype(str).to_numpy()]]
    merged = pd.concat([b.assign(precedence="v2"), a_only.assign(precedence="v1 only")], ignore_index=True)
    return merged, diff.sort_values(["status", "topic", "series_key", "period_end"]).reset_index(drop=True)


def to_template(m: pd.DataFrame) -> pd.DataFrame:
    """BoT template schema (``tzbank.ingest.bot.TEMPLATE_COLUMNS``); percentages become ratios."""
    from tzbank.ingest.bot import TEMPLATE_COLUMNS

    m = m[m["value"].notna() & m["period_end"].notna()].copy()
    u = m["unit"].map(unit_of)
    m["unit_t"] = [x[0] for x in u]
    m["value_t"] = m["value"] * [x[1] for x in u]
    freq = m["frequency_class"].map({"monthly": "monthly", "quarterly": "quarterly", "event": "event",
                                     "fy_june": "annual", "annual": "annual"})
    sid = ("BOTDB_" + m["series_key"].str.upper().str.replace(r"[^A-Z0-9|]+", "_", regex=True)
           .str.replace("|", "__", regex=False) + "_" + m["frequency_class"].str.upper())
    out = pd.DataFrame({
        "series_id": sid, "series_name": m["group"].astype(str) + ": " + m["label"].astype(str),
        "category": m["category"], "scope": "sector", "bank_name": None, "frequency": freq,
        "period_end": m["period_end"].dt.strftime("%Y-%m-%d"), "value": m["value_t"], "unit": m["unit_t"],
        "source_document": m["source_report"].astype(str) + " (" + m["file"].astype(str) + ")",
        "page_table": "table " + m["table"].astype(str) + ", p." + m["page"].astype(str),
        "publication_date": None, "confidence": np.where(m["workbook"] == "v2", "HIGH", "MEDIUM"),
        "notes": ("owner research database " + m["workbook"] + " (" + m["precedence"] + "); flag "
                  + m["flag"].astype(str))})[TEMPLATE_COLUMNS]
    return out.drop_duplicates(["series_id", "period_end"], keep="first")


def write_bot_database(root: Path | None = None) -> list[Path]:
    """data/interim/bot_research_database.csv (template schema) and reports/validation/bot_v1_v2_diff.csv."""
    root = root or find_root()
    raw = path("raw", root)
    if not (raw / V1).exists() or not (raw / V2).exists():
        return []
    v1, v2 = read_v1(raw / V1), read_v2(raw / V2)
    xw = load_crosswalk(root)
    _, diff_raw = merge(v1, v2)
    merged, diff = merge(v1, v2, xw)
    rc, xc = recheck(diff_raw, diff, harmonise(v2, xw), xw)
    t = to_template(merged)
    t["evidence_class"] = "Observed"
    return [write_csv(t, path("interim", root) / "bot_research_database.csv", require_evidence=True),
            write_csv(diff, path("validation", root) / "bot_v1_v2_diff.csv", require_evidence=True),
            write_csv(rc, path("validation", root) / "bot_v1_v2_recheck.csv", require_evidence=True),
            write_csv(xc, path("validation", root) / "bot_crosswalk_check.csv", require_evidence=True),
            write_csv(pd.concat([collisions(v1, xw), collisions(v2, xw)], ignore_index=True),
                      path("validation", root) / "bot_crosswalk_collisions.csv", require_evidence=True)]


def recheck(diff_raw: pd.DataFrame, diff: pd.DataFrame, v2h: pd.DataFrame, xw: dict) -> tuple[pd.DataFrame,
                                                                                             pd.DataFrame]:
    """Prompt 10 task 1: (a) every pre-harmonisation v1/v2 conflict, re-classified after the crosswalk merge, with a
    label-swap search (the v1 value equals another v2 series of the same table and period); (b) per crosswalk entry,
    the newly matched overlap and how many values agree."""
    conf = diff_raw[diff_raw["status"].str.startswith("CONFLICT")].copy()
    v2h = v2h.assign(pe=v2h["period_end"].dt.date)
    rows = []
    for r in conf.itertuples():
        cand = v2h[(v2h["topic"] == r.topic) & (v2h["frequency_class"] == r.frequency_class) & (v2h["pe"] == r.period_end)
                   & (v2h["series_key"] != r.series_key)]
        hit = cand[np.isclose(cand["value"], r.v1_value, rtol=TOL)]
        after = diff[(diff["series_key"] == r.series_key) & (diff["frequency_class"] == r.frequency_class)
                     & (diff["period_end"] == r.period_end)]
        status_after = after["status"].iloc[0] if len(after) else "not matched"
        if len(hit):
            cls = f"label swap: the v1 value equals v2 '{hit['label'].iloc[0]}' for the same table and period"
        elif status_after == "same value":
            cls = "resolved by harmonisation"
        else:
            cls = "revision: later vintage differs (v2 supersedes, D40)"
        rows.append({"topic": r.topic, "series_key": r.series_key, "frequency_class": r.frequency_class,
                     "period_end": r.period_end, "v1_value": r.v1_value, "v2_value": r.v2_value, "rel_diff": r.rel_diff,
                     "status_after_harmonisation": status_after, "classification": cls, "evidence_class": "Observed"})
    rc = pd.DataFrame(rows)
    xrows = []
    for (topic, var), canon in xw.items():
        k = f"{topic}|{canon}"
        d = diff[diff["series_key"] == k]
        new = d[~d.set_index(["series_key", "frequency_class", "period_end"]).index.isin(
            diff_raw.set_index(["series_key", "frequency_class", "period_end"]).index)]
        xrows.append({"topic": topic, "label_variant": var, "canonical_label": canon,
                      "newly_matched_periods": len(new), "agree": int((new["status"] == "same value").sum()),
                      "conflict": int(new["status"].str.startswith("CONFLICT").sum()),
                      "check": "no overlap (series joined end to end)" if new.empty else
                      ("all overlapping values agree" if (new["status"] == "same value").all() else
                       "some overlapping values differ (v2 supersedes, listed in bot_v1_v2_diff.csv)"),
                      "evidence_class": "Observed"})
    return rc, pd.DataFrame(xrows)
