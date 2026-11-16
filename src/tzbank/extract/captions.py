"""Caption inventory and coverage audit of the inclusion and NPS panels (prompt 02c section 3; D30-D33).

Every table, chart, appendix and annex caption is read from the report text layers with its number, printed page
and PDF page, and whether it prints data (a table, or a chart with numeric data labels; heuristic: numeric text
spans in the area below the caption). Each caption is matched against the panel's source references and tagged with
blueprint relevance. Output: reports/inclusion_nps_coverage.md and .csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.io import write_csv, write_text

CAPTION = re.compile(r"^\s*(Table|Chart|Figure|Graph|Appendix|Annex)\s+([A-Z]?\d+(?:\.\s?\d+|\s\d+(?=\s*:))*|[A-Z])\b\s*[:.\-–]?\s*(.*)$",
                     re.I)
KIND = {"table": "table", "chart": "chart", "figure": "chart", "graph": "chart", "appendix": "appendix",
        "annex": "appendix"}
NUMERIC = re.compile(r"^\(?-?[\d.,]+%?\)?$")
INCLUSION_FILES = {2023: "FinInc - 2024120408412609.pdf", 2024: "FinInc - 2025082509110517.pdf",
                   2025: "FinInc - 2026072316300213.pdf"}
NPS_FILES = {2022: "Annual Reports - BANK OF TANZANIA NATIONAL PAYMENT SYSTEMS ANNUAL REPORT 2022.pdf",
             2023: "Annual Reports - National Payment System Annual Report for 2023.pdf",
             2024: "Annual Reports - National Payment System Annual Report for 2024.pdf",
             2025: "Annual Reports - National Payment System Annual Report for 2025.pdf"}

# relevance tags (prompt 02c section 3.3), checked in order
TAGS = [
    ("low", r"insurance|pension|social security|nhif|literacy|social media|complaint|consumer protection|tasaf|"
            r"policyholder|contribution|capital markets? awareness|education"),
    ("ch6", r"saving|deposit|interest rate|lending rate|financial depth|\bdepth"),
    ("ch1", r"access point|branch|\batm|geograph|region|distance|within 5 ?km|coverage"),
    ("ch2", r"loan|credit|borrow|guarantee|\blga\b|local government|women|youth|microfinance|sacco|neec|sido"),
    ("ch4.4/ch4.7", r"mobile|digital|wallet|\bpos\b|point of sale|payment|transaction|tips|tiss|\beft\b|tach|"
                    r"\bcards?\b|remittance|cross[- ]border|merchant|internet banking|ussd|e-?money|fintech|cheque|"
                    r"eaps|sadc|\bqr\b|agent"),
    ("ch1", r"access|agent"),
]


def tag(caption: str) -> str:
    c = caption.lower()
    for t, pat in TAGS:
        if re.search(pat, c):
            return t
    return "context"


def parse_caption(line: str) -> tuple[str, str, str] | None:
    """('chart', '2.1', 'title') for a caption line; None for references inside sentences and TOC lines."""
    if re.search(r"\.{4,}|…{2,}", line):
        return None  # table-of-contents leader dots
    m = CAPTION.match(line)
    if not m:
        return None
    kind, num, title = KIND[m.group(1).lower()], re.sub(r"\.?\s+", ".", m.group(2)), m.group(3).strip()
    if title and title[0].islower():
        return None  # 'Table 2 shows ...' style sentence
    return kind, num, title


def printed_page(page: pymupdf.Page) -> int | None:
    """A standalone page number in the top or bottom band of the page."""
    h = page.rect.height
    for b in page.get_text("words"):
        if (b[1] > 0.9 * h or b[3] < 0.08 * h) and re.fullmatch(r"\d{1,3}", b[4]):
            return int(b[4])
    return None


def has_printed_data(page: pymupdf.Page, y: float, kind: str) -> str:
    """'table' for tables/appendices; for charts, 'labels' when >= 4 numeric spans lie in the 380 pt below."""
    if kind in {"table", "appendix"}:
        return "table"
    nums = [w for w in page.get_text("words") if y < w[1] < y + 380 and NUMERIC.fullmatch(w[4])
            and len(w[4]) <= 9]
    return "labels" if len(nums) >= 4 else "no labels"


def inventory(pdf: Path, report: int, panel: str) -> pd.DataFrame:
    """Captions of one report (TOC pages skipped: pages with >= 6 caption lines)."""
    rows = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            lines = []
            for b in page.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    text = "".join(s["text"] for s in ln["spans"]).strip()
                    lines.append((text, ln["bbox"][1]))
            caps = [(parse_caption(t), y) for t, y in lines]
            caps = [(c, y) for c, y in caps if c]
            if len(caps) >= 6 and sum(bool(re.search(r"\d+\s*$", c[2])) for c, _ in caps) >= 4:
                continue  # list of tables/charts
            pp = printed_page(page)
            for (kind, num, title), y in caps:
                rows.append({"panel": panel, "report": report, "kind": kind, "number": num, "caption": title[:150],
                             "pdf_page": i, "printed_page": pp, "printed_data": has_printed_data(page, y, kind)})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["report", "kind", "number"]) if len(df) else df


def _refs(text: str) -> list[tuple[str, str]]:
    """('chart', '3.1'), ('chart', '3.2') from 'Charts 3.1-3.2'; ('appendix', '1') from 'Appendix 1 (p48)'."""
    out = []
    for m in re.finditer(r"(Tables?|Charts?|Figures?|Appendix|Appendices|Annex(?:es)?)\s+([A-Z]?\d+(?:\.\d+)*)"
                         r"(?:\s*[–-]\s*([A-Z]?\d+(?:\.\d+)*))?", str(text), re.I):
        k = KIND.get(re.sub(r"(e?s|ices)$", "", m.group(1).lower()).replace("appendic", "appendix"), None)
        k = k or KIND.get(m.group(1).lower().rstrip("s"), "table")
        a, b = m.group(2), m.group(3)
        out.append((k, a))
        if b and "." in a and "." in b and a.split(".")[0] == b.split(".")[0]:
            for n in range(int(a.split(".")[1]) + 1, int(b.split(".")[1]) + 1):
                out.append((k, f"{a.split('.')[0]}.{n}"))
    return out


def inclusion_refs(root: Path) -> tuple[set, dict, set]:
    """(captured refs {(report, kind, number)}, pages {(report, pdf_page): [refs]}, narrative pages)."""
    i = path("interim", root)
    long = pd.read_csv(i / "inclusion_long.csv")
    refs, pages = set(), {}
    for rep, ref, pg in zip(long["Source_Report"], long["Source_Table_or_Section"], long["PDF_Page"], strict=False):
        for k, n in _refs(ref):
            refs.add((int(rep), k, n))
        if pd.notna(pg):
            pages.setdefault((int(rep), int(pg)), set()).add(str(ref))
    for f in i.glob("inclusion_*.csv"):
        if f.name in {"inclusion_long.csv", "inclusion_text_facts.csv", "inclusion_data_issues.csv",
                      "inclusion_source_register.csv", "inclusion_revision_check.csv"}:
            continue
        d = pd.read_csv(f)
        texts = " ".join(map(str, d.get("source_note", pd.Series(dtype=str)).dropna().unique()))
        texts += " " + " ".join(map(str, d.get("Source", pd.Series(dtype=str)).dropna().unique()))
        texts += " " + " ".join(f"{r} report {t}" for r, t in zip(d.get("Source report", []), d.get("Source table", []),
                                                                   strict=False))
        for m in re.finditer(r"(20\d\d) report,? ((?:Tables?|Charts?|Appendix|Annex)[^;,()]*)", texts):
            for k, n in _refs(m.group(2)):
                refs.add((int(m.group(1)), k, n))
    tf = pd.read_csv(i / "inclusion_text_facts.csv")
    narrative = {(int(r), int(p)) for r, p in zip(tf["Source report"], tf["PDF page"], strict=False)
                 if pd.notna(r) and pd.notna(p) and str(p).isdigit()}
    return refs, pages, narrative


def nps_refs(root: Path) -> set:
    """NPS workbook table references {(kind, number)} (annex tables A1.., body Table 1)."""
    p = pd.read_csv(path("interim", root) / "nps_long.csv")
    refs = set()
    for t in p["table_title"].unique():
        refs.update(_refs(t))
    for s in p["sheet"].unique():
        m = re.search(r"Annex ([A-Z])", " ".join(p.loc[p["sheet"] == s, "table_title"].unique()))
        if m:
            refs.add(("appendix", m.group(1)))
    refs.add(("table", "1"))  # 13_Table1_Indicators
    for kind, num in list(refs):
        if kind == "table" and re.fullmatch(r"[A-Z]\d+", num):
            refs.add(("appendix", num[0]))  # an annex is captured when its tables are
    return refs


# NPS annex letter per workbook sheet whose tables are titled by topic, not by annex number
NPS_SHEET_ANNEX = {"06_Remittance": "F", "07_CrossBorder": "G", "11_Merchants": "K"}
# body-chart topic -> annex letter holding the series in the NPS workbook (topic match, not a table reference)
NPS_TOPIC_ANNEX = [(r"\btiss\b", "A"), (r"\btips\b", "B"), (r"\beft\b|tach|cheque|electronic fund", "C"),
                   (r"eaps|sadc|regional", "D"), (r"\bcard|\batm", "E"), (r"remittance|remmitance", "F"),
                   (r"cross.?border", "G"), (r"mobile (money|payment)|\bp2[pbg]\b|wallet|cash.in|cash.out", "H"),
                   (r"point of sale|\bpos\b", "I"), (r"internet banking", "J"), (r"merchant", "K"),
                   (r"mobile banking", "L")]


def nps_annex_of(caption: str) -> str | None:
    """Annex letter whose NPS sheet holds the series a body chart or a 2022 table shows (topic match)."""
    low = caption.lower()
    if re.search(r"digital (saving|credit)|digital financial", low):
        return None  # Table H12 (digital financial services) is not in the NPS workbook
    for pat, letter in NPS_TOPIC_ANNEX:
        if re.search(pat, low):
            return letter
    return None


def _words(t: str) -> set[str]:
    stop = {"the", "of", "in", "and", "to", "for", "number", "by", "a", "an", "tzs", "billions", "amount"}
    return {w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in stop and len(w) > 2}


def _similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def classify(inv: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Coverage status per caption (prompt 02c section 3.2)."""
    inc_refs, inc_pages, narrative = inclusion_refs(root)
    nps_r = nps_refs(root)
    nps_reports = {2023, 2024, 2025}
    out = []
    for _, c in inv.iterrows():
        key = (c["kind"], c["number"])
        match = ""
        twin = None
        if c["panel"] == "inclusion":
            same = inv[(inv["panel"] == "inclusion") & (inv["report"] == c["report"]) & (inv.index != c.name)]
            cands = [(r, _similar(c["caption"], r["caption"])) for _, r in same.iterrows()
                     if (r["report"], r["kind"], r["number"]) in inc_refs and r["kind"] != c["kind"]]
            cands = [x for x in cands if x[1] >= 0.75]
            twin = max(cands, key=lambda x: x[1])[0] if cands else None
            if (c["report"], *key) in inc_refs:
                status = "captured"
            elif twin is not None:
                status = "captured under a different reference"
                match = f"same content as {twin['kind']} {twin['number']} ('{twin['caption'][:60]}', captured)"
            elif (c["report"], c["pdf_page"]) in inc_pages:
                same_refs = sorted(inc_pages[(c["report"], c["pdf_page"])])
                # other charts on the page do not hold this chart's series; only a table or appendix would
                textual = all(re.search(r"text|section|box|chart", r, re.I) for r in same_refs)
                if textual and c["kind"] == "chart":
                    status = "narrative only"
                    match = "partial: values from the narrative on the same page (" + "; ".join(same_refs)[:150] + ")"
                else:
                    status = "captured under a different reference"
                    match = "same page: " + "; ".join(same_refs)[:180]
            elif (c["report"], c["pdf_page"]) in narrative:
                status = "narrative only"
            else:
                status = None
        else:
            status = "captured" if key in nps_r and c["report"] in nps_reports else None
            letter = c["number"][0] if re.fullmatch(r"[A-Z]\d*", c["number"]) else None
            annex_whole = {("appendix", x) for x in "BEIJL"}  # annexes held as one PANEL sheet
            topic = nps_annex_of(c["caption"])
            if status is None and c["report"] in nps_reports and letter and (
                    ("appendix", letter) in annex_whole or letter in NPS_SHEET_ANNEX.values()):
                status = "captured under a different reference"
                sheet = [k for k, v in NPS_SHEET_ANNEX.items() if v == letter]
                match = (f"annex {letter} held as sheet {sheet[0]} (titled by topic)" if sheet else
                         f"annex {letter} held as one PANEL sheet")
            elif status is None and c["kind"] in {"chart", "table"} and topic and (
                    c["report"] == 2022 or not re.fullmatch(r"[A-Z]\d*", c["number"])):
                status = "captured under a different reference"
                match = (f"topic match: series held in the annex {topic} sheet of the NPS workbook" +
                         (" (2022 report: its 2018 values are the only years not in the 2023-2025 annexes)"
                          if c["report"] == 2022 else " (body chart plotted from the annex table)"))
        if status is None and re.fullmatch(r"(references?|acronyms|abbreviations|glossary)\W*", c["caption"].lower()):
            status = "narrative only"
        if status is None:
            status = ("not captured, chart without data labels" if c["printed_data"] == "no labels"
                      else "not captured, with printed data")
        out.append(c.to_dict() | {"status": status, "matched_reference": match, "relevance": tag(c["caption"])})
    df = pd.DataFrame(out)
    relevant = df["relevance"].isin(["ch1", "ch2", "ch4.4/ch4.7", "ch6"])
    partial = (df["status"] == "narrative only") & (df["printed_data"] == "labels") & df["matched_reference"].str.startswith(
        "partial")
    df["priority_gap"] = relevant & ((df["status"] == "not captured, with printed data") | partial)
    df["gap_type"] = ""
    df.loc[relevant & (df["status"] == "not captured, with printed data"), "gap_type"] = "not captured"
    df.loc[relevant & partial, "gap_type"] = "partial (narrative endpoints only)"
    df["evidence_class"] = "Observed"
    return df


def write_coverage(root: Path | None = None) -> list[Path]:
    """reports/inclusion_nps_coverage.csv and .md."""
    from tzbank.validate.run import _md_table

    root = root or find_root()
    docs = path("documents", root)
    parts = [inventory(docs / "inclusion" / f, r, "inclusion") for r, f in INCLUSION_FILES.items()]
    parts += [inventory(docs / "nps" / f, r, "nps") for r, f in NPS_FILES.items()]
    cov = classify(pd.concat(parts, ignore_index=True), root)
    rdir = path("reports", root)
    lines = ["# Inclusion and NPS panels: coverage of report tables and charts", "",
             "Generated by `tzbank.extract.captions` (prompt 02c section 3). Captions read from each report's text "
             "layer; 'printed data' is a heuristic (tables, appendices, or >= 4 numeric labels below a chart caption). "
             "Matching uses the panels' source references; NPS matches are by annex table number because the NPS "
             "workbook combines reports. Unlabelled charts are not digitised (D33).", ""]
    for panel in ["inclusion", "nps"]:
        d = cov[cov["panel"] == panel]
        lines += [f"## {panel.upper()} panel", "", f"Captions inventoried: {len(d)}.", "", "### By status", "",
                  _md_table(d["status"].value_counts().rename_axis("status").reset_index(name="captions")), "",
                  "### By status and relevance", "",
                  _md_table(pd.crosstab(d["status"], d["relevance"]).reset_index()), "",
                  "### Priority extraction gaps (not captured, printed data, tagged ch1-ch6)", "",
                  _md_table(d[d["priority_gap"]][["report", "kind", "number", "caption", "printed_page", "pdf_page",
                                                  "relevance", "gap_type"]]), ""]
    tasks = collection_tasks(cov)
    lines += ["## Collection tasks", "",
              "Priority gaps plus chapter-relevant charts without data labels (D33: never digitised; the underlying "
              "BoT series is collected instead). Written to `reports/inclusion_nps_collection_tasks.csv`.", "",
              _md_table(tasks.drop(columns=["evidence_class"])), ""]
    return [write_csv(cov, rdir / "inclusion_nps_coverage.csv", require_evidence=True),
            write_csv(tasks, rdir / "inclusion_nps_collection_tasks.csv", require_evidence=True),
            write_text("\n".join(lines), rdir / "inclusion_nps_coverage.md")]


def collection_tasks(cov: pd.DataFrame) -> pd.DataFrame:
    """One task per priority gap or chapter-relevant unlabelled chart (prompt 02c section 3.4; D33)."""
    relevant = cov["relevance"].isin(["ch1", "ch2", "ch4.4/ch4.7", "ch6"])
    unlabelled = relevant & (cov["status"] == "not captured, chart without data labels")
    t = cov[cov["priority_gap"] | unlabelled].copy()

    def action(r: pd.Series) -> str:
        if r["status"] == "not captured, chart without data labels" or str(r["gap_type"]).startswith("partial"):
            return "D33: collect the underlying BoT series (do not digitise the chart)"
        return f"transcribe the printed {r['kind']} into the owner's panel (PDF p.{r['pdf_page']})"

    t["task"] = t.apply(action, axis=1)
    t["evidence_class"] = "Observed"
    cols = ["panel", "report", "kind", "number", "caption", "pdf_page", "relevance", "status", "gap_type", "task",
            "evidence_class"]
    # prompt 02d section 6 (D35): the NPS reports are dated only by a lower bound (PDF last-modified date)
    dates = pd.DataFrame([{"panel": "nps", "report": y, "kind": "publication date", "number": "",
                           "caption": f"BoT National Payment Systems Annual Report {y} (BOT-NPSAR-{y})",
                           "pdf_page": "", "relevance": "point-in-time (D35)", "status": "lower_bound",
                           "gap_type": "posting date not confirmed",
                           "task": "owner: confirm the posting date on the BoT website (Publications > NPS annual "
                                   "reports) and record it as Actual Publication Date in Sources",
                           "evidence_class": "Observed"} for y in sorted(NPS_FILES)])
    return pd.concat([t[cols], dates], ignore_index=True)
