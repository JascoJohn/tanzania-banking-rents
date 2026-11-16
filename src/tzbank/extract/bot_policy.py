"""Statutory minimum reserve (SMR) history and deposit-insurance lines, by text search (D42; prompt 05 task 1).

- SMR: BoT Annual Reports in ``documents/reports/``. A statement is kept when the sentence names the statutory
  minimum reserve (SMR) and gives the new ratio ("from X percent to Y percent", "by N basis points to Y percent",
  "set at Y percent"); the effective date comes from the same passage (chronology prefix "May 2020:", "effective from
  8th June 2020", "in April 2017", "end of May 2015", "from July 2019", "towards the end of 2014"). Statements are
  grouped into events (same new ratio, same month); the gate is that the chain is consistent (each event's "from"
  equals the previous event's ratio). Government-deposit requirements are kept separately (scope ``government``).
- Deposit insurance: NMB and CRDB annual reports, the "Deposit Insurance Fund" (or premium/contribution) line in the
  operating-expense note, current and prior year, Group column, both printed on the page.

Output: ``data/interim/bot_policy_text.csv``. ``merge_into_bot_long`` appends the rows to ``data/interim/bot_long.csv``
(categories ``reserve_requirements`` and ``deposit_insurance``) after each ingest; gaps are collection tasks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.extract.bot_reports import bot_report_files, fy_of, report_fy
from tzbank.extract.pdftext import _doc
from tzbank.io import write_csv

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], start=1)}
MON = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
NUM = r"(\d+(?:\.\d+)?)"
KEY = r"(statutory minimum reserves?(?: requirement)?(?: ratio| rate)?(?: \(SMR\))?(?: requirement ratio)?|\bSMR\b" \
      r"(?: requirement)?(?: ratio)?|minimum reserve requirement)"
CHANGE = re.compile(KEY + r"(?P<mid>[^%]{0,140}?)(?:from\s+" + NUM + r"\s*(?:percent|%)\s+)?to\s+" + NUM
                    + r"\s*(?:percent|%)", re.I)
BPS = re.compile(KEY + r"[^%]{0,60}?by\s+(\d+)\s+basis points\s+to\s+" + NUM + r"\s*(?:percent|%)", re.I)
SET = re.compile(KEY + r"\s+was\s+set\s+at\s+" + NUM + r"\s*(?:percent|%)\s+of\s+([^.(]{0,60})", re.I)
BLOCK = re.compile(r"discount rate|haircut|averaging band|Lombard|policy rate|CBR", re.I)
COLUMNS = ["kind", "scope", "event_date", "date_precision", "from_value", "value", "unit", "report_fy", "file",
           "pdf_page", "quote", "gate", "gate_result", "evidence_class"]


def _date(before: str, after: str) -> tuple[str | None, str]:
    """Most precise effective date in the passage: day > month > year."""
    m = re.search(r"effective (?:from )?(\d{1,2})(?:st|nd|rd|th)? " + MON + r",? (\d{4})", after, re.I)
    if m:
        return f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}", "day"
    m = re.search(r"[Ee]ffective " + MON + r" (\d{1,2}),? (\d{4})", before + " " + after)
    if m:
        return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}", "day"
    m = re.search(MON + r",? (\d{4}):\s*[^:]{0,240}$", before) or re.search(r"(\d{4}) " + MON + r":\s*[^:]{0,240}$",
                                                                             before)
    if m:
        a, b = m.group(1), m.group(2)
        mon, yr = (a, b) if a[0].isalpha() else (b, a)
        return f"{yr}-{MONTHS[mon.lower()]:02d}-15", "month (chronology)"
    m = re.search(r"(?:in|end of|from) " + MON + r" (\d{4})", after[:230], re.I)
    if m:
        return f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}-15", "month"
    m = re.search(r"towards the end of (\d{4})", after[:120], re.I)
    if m:
        return f"{m.group(1)}-12-15", "approximate (end of year)"
    return None, "none"


def _fy_date(fy: str) -> tuple[str | None, str]:
    """Fallback: the statement only says 'during the period' of the report's financial year (July-June)."""
    m = re.match(r"(\d{4})/\d\d", fy)
    return (f"{m.group(1)}-12-31", "approximate (report financial year; dated to its December)") if m else (None,
                                                                                                          "none")


def smr_statements(root: Path) -> pd.DataFrame:
    fys = report_fy(root)
    rows = []
    for f in bot_report_files(root):
        d = _doc(str(f))
        fy = fy_of(f, fys)
        for i in range(d.page_count):
            t = re.sub(r"\s+", " ", d[i].get_text())
            hits = []
            for m in CHANGE.finditer(t):
                if BLOCK.search(m.group("mid") or ""):
                    continue
                hits.append((m, m.group(3), m.group(4)))
            for m in BPS.finditer(t):
                hits.append((m, f"{float(m.group(3)) + int(m.group(2)) / 100:.1f}", m.group(3)))
            for m in SET.finditer(t):
                hits.append((m, None, m.group(2)))
            for m, frm, to in hits:
                before, after = t[max(0, m.start() - 260): m.start()], t[m.start(): m.end() + 220]
                scope = "government" if re.search(r"government deposits", t[m.start() - 80: m.end() + 40], re.I) \
                    else "private deposit liabilities"
                date, prec = _date(before, after)
                if date is None:
                    date, prec = _fy_date(fy)
                rows.append({"kind": "SMR", "scope": scope, "event_date": date, "date_precision": prec,
                             "from_value": float(frm) / 100 if frm else None, "value": float(to) / 100,
                             "unit": "ratio", "report_fy": fy, "file": f.name, "pdf_page": i + 1,
                             "quote": t[max(0, m.start() - 60): m.end() + 120].strip(),
                             "gate": "sentence quoted from the page; effective date from the same passage"})
    return pd.DataFrame(rows)


def smr_events(st: pd.DataFrame) -> pd.DataFrame:
    """Group statements into events (scope, new ratio, effective month); keep the most precise date; chain gate."""
    s = st.dropna(subset=["event_date"]).copy()
    s["month"] = s["event_date"].str[:7]
    rank = {"day": 0, "month (chronology)": 1, "month": 2, "approximate (end of year)": 3,
            "approximate (report financial year; dated to its December)": 4}
    s["prec_rank"] = s["date_precision"].map(rank)
    ev = []
    for _key, g in s.groupby(["scope", "value", "month"]):
        g = g.sort_values("prec_rank")
        r = g.iloc[0].to_dict()
        r["supporting_statements"] = len(g)
        r["sources"] = "; ".join(sorted({f"{x.report_fy} p.{x.pdf_page}" for x in g.itertuples()}))
        r["from_value"] = g["from_value"].dropna().iloc[0] if g["from_value"].notna().any() else None
        ev.append(r)
    e = pd.DataFrame(ev)
    # one event stated with a precise date and, in another report, only with the report year: keep the precise one
    e = e.sort_values(["scope", "event_date"]).reset_index(drop=True)
    d = pd.to_datetime(e["event_date"])
    drop = []
    for i, r in e.iterrows():
        if r["prec_rank"] >= 4:
            twin = ((e["scope"] == r["scope"]) & (e["value"] == r["value"]) & (e["prec_rank"] < 4)
                    & ((d - d[i]).abs().dt.days <= 400))
            if twin.any():
                drop.append(i)
    e = e.drop(index=drop).reset_index(drop=True)
    res = []
    for _scope, g in e.groupby("scope"):
        prev = None
        for _, r in g.iterrows():
            if prev is None or pd.isna(r["from_value"]):
                res.append("PASS (first event or base level)" if prev is None else "PASS (no from-value stated)")
            else:
                res.append("PASS (chain consistent)" if abs(r["from_value"] - prev) < 1e-9
                           else f"FLAG: from {r['from_value']:.3f} but previous level {prev:.3f}")
            prev = r["value"]
    e["gate_result"] = res
    e["evidence_class"] = "Observed"
    return e


def di_lines(root: Path) -> pd.DataFrame:
    docs = path("documents", root)
    rows = []
    rx = re.compile(r"(Deposit Insurance (?:Fund|Board|premium|contribution)s?[^0-9(]{0,30})\s*\(?([\d,]{3,})\)?\s+"
                    r"\(?([\d,]{3,})\)?", re.I)
    for b, bank in (("NMB", "B02"), ("CRDB", "B01")):
        for f in sorted((docs / b).glob("*.pdf")):
            d = _doc(str(f))
            for i in range(d.page_count):
                t = re.sub(r"\s+", " ", d[i].get_text())
                for m in rx.finditer(t):
                    yr = re.search(r"(20\d\d)", f.name)
                    rows.append({"kind": "deposit_insurance_expense", "scope": bank, "event_date": None,
                                 "date_precision": "fiscal year", "from_value": None,
                                 "value": float(m.group(2).replace(",", "")) * 1e6, "unit": "TZS",
                                 "report_fy": yr.group(1) if yr else "unknown", "file": f"{b}/{f.name}",
                                 "pdf_page": i + 1, "quote": m.group(0)[:160],
                                 "gate": "line and prior-year comparative printed on the page (TZS million, Group "
                                         "column first)", "gate_result": "PASS", "evidence_class": "Observed",
                                 "prior_value": float(m.group(3).replace(",", "")) * 1e6})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["scope", "report_fy", "value"]) if len(df) else df


def write_bot_policy(root: Path | None = None) -> list[Path]:
    root = root or find_root()
    st = smr_statements(root)
    ev = smr_events(st)
    di = di_lines(root)
    cap = capital_requirements(root)
    out = pd.concat([ev.reindex(columns=COLUMNS + ["supporting_statements", "sources"]),
                     di.reindex(columns=COLUMNS + ["prior_value"]), cap.reindex(columns=COLUMNS)], ignore_index=True)
    out["pdf_page"] = out["pdf_page"].astype("Int64")  # search-record rows have no page; keep pages integer
    return [write_csv(out, path("interim", root) / "bot_policy_text.csv", require_evidence=True),
            write_csv(st.assign(evidence_class="Observed"), path("interim", root) / "bot_policy_statements.csv",
                      require_evidence=True)]


def merge_into_bot_long(root: Path | None = None) -> list[Path]:
    """Append the SMR events and deposit-insurance lines to bot_long.csv (idempotent; after each ingest)."""
    root = root or find_root()
    src = path("interim", root) / "bot_policy_text.csv"
    bl = path("interim", root) / "bot_long.csv"
    if not src.exists() or not bl.exists():
        return []
    p = pd.read_csv(src)
    b = pd.read_csv(bl, low_memory=False)
    b = b[b["file"].astype(str) != "data/interim/bot_policy_text.csv"]  # replace only rows added here
    rows = []
    for r in p[p["kind"].isin(["SMR", "deposit_insurance_expense"])].itertuples():
        smr = r.kind == "SMR"
        rows.append({"series_id": f"BOT_SMR_{'GOVT' if r.scope == 'government' else 'PRIVATE'}" if smr else
                     f"DEPOSIT_INSURANCE_EXPENSE_{r.scope}",
                     "series_name": f"Statutory minimum reserve ratio ({r.scope}), new level from the effective date"
                     if smr else f"Deposit insurance expense ({r.scope}, annual report operating-expense note)",
                     "category": "reserve_requirements" if smr else "deposit_insurance",
                     "scope": "sector" if smr else "bank", "bank_name": None if smr else r.scope,
                     "frequency": "event" if smr else "annual", "period_end": r.event_date if smr else
                     f"{r.report_fy}-12-31", "value": r.value, "unit": r.unit,
                     "source_document": f"BoT Annual Report {r.report_fy} ({r.file})" if smr else r.file,
                     "page_table": f"text, p.{int(r.pdf_page)}" + (f" (also {r.sources})" if smr and
                                                              isinstance(r.sources, str) else ""),
                     "publication_date": None, "confidence": "MEDIUM" if smr else "HIGH",
                     "notes": f"{r.gate_result}; {r.date_precision}; quote: {r.quote}"[:500],
                     "file": "data/interim/bot_policy_text.csv", "value_numeric": r.value,
                     "evidence_class": "Observed"})
    out = pd.concat([b, pd.DataFrame(rows, columns=b.columns)], ignore_index=True)
    return [write_csv(out, bl)]


# --- capital requirements stated in the banks' capital-management notes (prompt 06, chapter 7) -----------------

CAP_INLINE = re.compile(r"Tier 1 capital \(required ratio (\d+(?:\.\d+)?)\s*%\)", re.I)
CAP_MIN = re.compile(r"core capital to (?:the )?risk-? ?weighted assets[^.]{0,160}?required minimum of (\d+(?:\.\d+)?)\s*%",
                     re.I)
CAP_TABLE = re.compile(r"Tier 1 capital\s+(\d+(?:\.\d+)?)\s*%?\s+(\d+(?:\.\d+)?)\s*%", re.I)
CCB = re.compile(r"capital conservation buffer of (\d+(?:\.\d+)?)\s*%[^.]{0,160}?from " + MON + r" (\d{4})", re.I)
PLAUSIBLE_T1 = {10.0, 12.0, 12.5, 14.0, 14.5}
# prompt 07b: stated total-capital requirement, the context of CRDB's 12.5% / 14.5% and the D-SIB search
CAP_TOTAL = re.compile(r"total capital of not less than (\d+(?:\.\d+)?)\s*%", re.I)
CAP_TOTAL_INLINE = re.compile(r"Tier 1 \+ Tier 2 capital \(required ratio (\d+(?:\.\d+)?)\s*%\)", re.I)
CAP_TOTAL_TABLE = re.compile(r"Tier 1 \+ Tier 2 capital\s+(\d+(?:\.\d+)?)\s*%?\s+(\d+(?:\.\d+)?)\s*%", re.I)
PLAUSIBLE_TOTAL = {12.0, 14.5, 15.0, 17.0}
CONTEXT = re.compile(r"minimum capital ratio of [^.]{0,40}?has increased to [^.]{0,80}"
                     r"|required buffer of \d+(?:\.\d+)?\s*%[^.]{0,80}"
                     r"|capital conservation buffer in addition to the minimum regulatory capital requirements[^.]{0,80}"
                     r"|In August 2014, the Central Bank of Tanzania issued a regulation[^.]{0,120}", re.I)
DSIB = re.compile(r"domestic systemically important|systemically important (?:banks?|financial institutions?)"
                  r"|\bD-?SIBs?\b|capital surcharge|systemic (?:risk )?(?:buffer|surcharge)", re.I)


def capital_requirements(root: Path) -> pd.DataFrame:
    """Tier 1 requirement and capital conservation buffer as stated in each NMB / CRDB annual report (report year from
    the text, ``detail_lines.report_year``). Table form: of the two ratios after 'Tier 1 capital' on a page headed
    'Required ratio', the requirement is the smaller one and must be a plausible regulatory level."""
    from tzbank.extract.detail_lines import report_year

    docs = path("documents", root)
    rows = []
    for b, bank in (("NMB", "B02"), ("CRDB", "B01")):
        for f in sorted((docs / b).glob("*.pdf")):
            fy = report_year(f)
            if not fy:
                continue
            d = _doc(str(f))
            for i in range(d.page_count):
                t = re.sub(r"\s+", " ", d[i].get_text())
                hits = [(m, float(m.group(1)), "inline '(required ratio x%)'") for m in CAP_INLINE.finditer(t)]
                hits += [(m, float(m.group(1)), "'required minimum of x%' (core capital)") for m in CAP_MIN.finditer(t)]
                if re.search(r"Required ratio", t, re.I):
                    for m in CAP_TABLE.finditer(t):
                        v = min(float(m.group(1)), float(m.group(2)))
                        hits.append((m, v, "Required-ratio table (smaller of the two ratios)"))
                for m, v, how in hits:
                    rows.append({"kind": "tier1_requirement", "scope": bank, "event_date": f"{fy}-12-31",
                                 "date_precision": "fiscal year (report)", "from_value": None, "value": v / 100,
                                 "unit": "ratio", "report_fy": str(fy), "file": f"{b}/{f.name}", "pdf_page": i + 1,
                                 "quote": t[max(0, m.start() - 80): m.end() + 60].strip()[:300],
                                 "gate": f"{how}; plausible regulatory level",
                                 "gate_result": "PASS" if v in PLAUSIBLE_T1 else "FAIL: implausible level",
                                 "evidence_class": "Observed"})
                tot = [(m, float(m.group(1)), "'total capital of not less than x%'") for m in CAP_TOTAL.finditer(t)]
                tot += [(m, float(m.group(1)), "inline 'Tier 1 + Tier 2 capital (required ratio x%)'")
                        for m in CAP_TOTAL_INLINE.finditer(t)]
                if re.search(r"Required ratio", t, re.I):
                    tot += [(m, min(float(m.group(1)), float(m.group(2))),
                             "Required-ratio table, Tier 1 + Tier 2 (smaller of the two ratios)")
                            for m in CAP_TOTAL_TABLE.finditer(t)]
                for m, v, how in tot:
                    rows.append({"kind": "total_capital_requirement", "scope": bank, "event_date": f"{fy}-12-31",
                                 "date_precision": "fiscal year (report)", "from_value": None, "value": v / 100,
                                 "unit": "ratio", "report_fy": str(fy), "file": f"{b}/{f.name}", "pdf_page": i + 1,
                                 "quote": t[max(0, m.start() - 160): m.end() + 60].strip()[:300],
                                 "gate": f"{how}; plausible regulatory level",
                                 "gate_result": "PASS" if v in PLAUSIBLE_TOTAL else "FAIL: implausible level",
                                 "evidence_class": "Observed"})
                for m in CONTEXT.finditer(t):
                    rows.append({"kind": "capital_requirement_context", "scope": bank, "event_date": f"{fy}-12-31",
                                 "date_precision": "fiscal year (report)", "from_value": None, "value": None,
                                 "unit": "text", "report_fy": str(fy), "file": f"{b}/{f.name}", "pdf_page": i + 1,
                                 "quote": t[m.start(): m.end() + 120].strip()[:400],
                                 "gate": "sentence on the origin or nature of the stated requirement",
                                 "gate_result": "PASS", "evidence_class": "Observed"})
                for m in DSIB.finditer(t):
                    rows.append(_dsib_row(bank, str(fy), f"{b}/{f.name}", i + 1, t, m))
                for m in CCB.finditer(t):
                    rows.append({"kind": "capital_conservation_buffer", "scope": bank,
                                 "event_date": f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-01",
                                 "date_precision": "month (effective date stated)", "from_value": None,
                                 "value": float(m.group(1)) / 100, "unit": "ratio", "report_fy": str(fy),
                                 "file": f"{b}/{f.name}", "pdf_page": i + 1, "quote": m.group(0)[:300],
                                 "gate": "buffer and effective month stated in one sentence", "gate_result": "PASS",
                                 "evidence_class": "Observed"})
    rows += dsib_reports(root)
    nd = {b: len(list((docs / b).glob("*.pdf"))) for b in ("NMB", "CRDB")}
    nd["BoT"] = len(bot_report_files(root))
    for scope, n in nd.items():
        sc = {"NMB": "B02", "CRDB": "B01"}.get(scope, "BoT")
        hits = sum(1 for r in rows if r["kind"] == "dsib_statement" and r["scope"] == sc)
        rows.append({"kind": "dsib_search", "scope": sc, "event_date": None, "date_precision": "search record",
                     "from_value": None, "value": float(hits), "unit": "count of matching passages",
                     "report_fy": "all", "file": f"{'reports' if scope == 'BoT' else scope}/*.pdf ({n} files)",
                     "pdf_page": None, "quote": f"pattern: {DSIB.pattern}",
                     "gate": "every page of every PDF searched", "gate_result": "PASS", "evidence_class": "Observed"})
    df = pd.DataFrame(rows)
    keep = df[df["kind"].isin(["tier1_requirement", "capital_conservation_buffer", "total_capital_requirement"])]
    rest = df[~df.index.isin(keep.index)]
    keep = keep.drop_duplicates(["kind", "scope", "report_fy", "value", "event_date"])
    rest = rest.drop_duplicates(["kind", "scope", "file", "pdf_page", "quote"])
    return pd.concat([keep, rest]).sort_index() if len(df) else df


def _dsib_row(scope: str, fy: str, file: str, page: int, t: str, m: re.Match) -> dict:
    return {"kind": "dsib_statement", "scope": scope, "event_date": None, "date_precision": "report",
            "from_value": None, "value": None, "unit": "text", "report_fy": fy, "file": file, "pdf_page": page,
            "quote": t[max(0, m.start() - 200): m.end() + 200].strip()[:400],
            "gate": "D-SIB designation, framework or capital surcharge named in the text",
            "gate_result": "PASS", "evidence_class": "Observed"}


def dsib_reports(root: Path) -> list[dict]:
    """D-SIB / capital-surcharge passages in the BoT Annual Reports (prompt 07b task 2; blueprint chapter 3)."""
    rows, fys = [], report_fy(root)
    for f in bot_report_files(root):
        d = _doc(str(f))
        fy = fy_of(f, fys)
        for i in range(d.page_count):
            t = re.sub(r"\s+", " ", d[i].get_text())
            for m in DSIB.finditer(t):
                rows.append(_dsib_row("BoT", str(fy or "unknown"), f"reports/{f.name}", i + 1, t, m))
    return rows
