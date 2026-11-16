"""Raw-data MANIFEST (prompt 01 section 2; CLAUDE.md section 3.1: data/raw is immutable).

``write_manifest`` registers every file under data/raw with its sha256. ``verify_manifest`` is run by the ingest
stage: a raw file whose hash no longer matches its MANIFEST row is a critical failure.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.io import rel, sha256, write_csv

MANIFEST_COLUMNS = ["file", "sha256", "bytes", "received_date", "description", "source", "committed (Y/N)"]

# (path prefix or exact relative path under data/raw, description, source)
_RULES: list[tuple[str, str, str]] = [
    ("Tanzania_DSE_Commercial_Banks_Phase1_v2_disclosure.xlsx",
     "Seven-bank Phase 1 panel v2.0-disclosure (RAW_* sheets, VERSION_LOG, AUDIT_SUMMARY); read-only source",
     "Owner (Phase 1 manuscript workbook, audited 2026-09-06)"),
    ("NMB_CRDB_Valuation_Inputs.xlsx",
     "Working NMB/CRDB inputs workbook (D18; version and change history on its Read Me sheet). Started from v0.6",
     "Owner (compiled from audited annual reports, interims, DSE, BoT, Damodaran)"),
    ("Treasury_Bonds_Auction_Results_-_2026-11-06.xlsx",
     "BoT Treasury bond auction calendar Sep 2012 - Nov 2026 (number, title, date; no yields). Received as "
     "'Treasury Bonds Auction Results - 2026-11-06.xlsx'", "Bank of Tanzania website, auction results list"),
    ("NPS_MultiYear_Panel_Dataset.xlsx", "BoT National Payment Systems multi-year panel 2019-2025",
     "Compiled by owner from BoT NPS Annual Reports 2022-2025"),
    ("Tanzania_Financial_Inclusion_Panel_2018-2025.xlsx",
     "Financial inclusion panel 2018-2025 (Panel_Long observations, supporting and standalone sheets; D30). Read-only "
     "raw input; formula views are recomputed in Python (D31)",
     "Compiled by owner from BoT Annual Financial Inclusion Reports 2023, 2024 and 2025"),
    ("NMB_CRDB_52wk_and_trading_value.xlsx", "Original daily DSE trading file for NMB and CRDB (52-week and traded value)",
     "Owner, from DSE daily trading reports"),
    ("finscope/FinScope.csv", "FinScope Tanzania microdata (9,915 respondents, 721 variables)",
     "FSD Tanzania / FinScope Tanzania survey microdata"),
    ("finscope/FinScope_Datamap.xlsx", "FinScope Tanzania data map (variables and value labels). Received as "
     "'FinScope Datamap.xlsx'", "FSD Tanzania / FinScope Tanzania"),
    ("superseded/", "Superseded version of the inputs workbook, kept for audit trail only (not read by the pipeline)",
     "Owner"),
    ("bot/", "Further BoT data file", "Bank of Tanzania"),
    ("documents/inclusion/", "BoT Annual Financial Inclusion Report (source of the inclusion panel; prompt 02c)",
     "Bank of Tanzania website"),
    ("documents/nps/", "BoT National Payment Systems Annual Report (source of the NPS panel; prompt 02c)",
     "Bank of Tanzania website"),
    ("documents/regulation/tz-government-gazette-dated-2008-12-12-no-50.pdf",
     "Government Gazette No. 50 of 12 Dec 2008, notice No. 632: GN 368 of 2008 (BFI Publication of Financial "
     "Statements Regulations) published in Subsidiary Legislation Supplement No. 41 (evidence for D36)",
     "Government of Tanzania (Gazette), supplied by the owner"),
    ("documents/regulation/", "Banking regulation text (publication deadlines, D16/D25)", "Government of Tanzania"),
    ("manual_entries/manual_check_list_filled_", "Owner's filled manual checklist (D38); converted and gated by "
     "tzbank.ingest.manual_checklist. Received as 'manual_check_list_updated (1).csv'", "Owner (manual PDF reading)"),
    ("manual_entries/benchmarks_collected.csv", "Benchmark figures (prompt 13 item 12): one value per row with "
     "document, URL, page, publication and access dates, confidence; shortfalls listed as NOT OBTAINED rows",
     "Collected by Claude for the owner, 2026-11-12 (cited documents in documents/benchmarks)"),
    ("manual_entries/benchmarks_verified.csv", "Benchmark figures confirmed on the source page (prompt 24, TO_VERIFY A14): "
     "McKinsey Global Banking Annual Review 2025 ROE and P/B, with PDF and printed page and a short quote; applied over "
     "benchmarks_collected.csv", "Checked by Claude for the owner, 2026-11-15 (PDF supplied by the owner)"),
    ("manual_entries/events_collected.csv", "Dated corporate events, NMB and CRDB (prompt 14 item 2), with sources and "
     "confidence", "Collected by Claude for the owner, 2026-11-13 (cited web sources)"),
    ("manual_entries/annual_report_items.csv", "Annual-report items FY2023-25 (prompt 14 item 6), value as printed and "
     "page; checked on the page in the pipeline", "Transcribed by Claude for the owner, 2026-11-13 (bank annual reports "
                                                  "held)"),
    ("manual_entries/imf_cpi_tza.csv", "IMF WEO Tanzania average CPI inflation 2008-2025 (prompt 14 item 5)",
     "Collected by Claude for the owner, 2026-11-13 (IMF DataMapper API)"),
    ("manual_entries/repricing_profile.csv", "IFRS 7 interest rate repricing profiles, NMB and CRDB FY2025 (prompt 13b "
     "item 1); each value checked against the cited annual-report page", "Transcribed by Claude for the owner, "
                                                                         "2026-11-12 (bank annual reports held)"),
    ("manual_entries/peers_collected.csv", "African peer panel (prompt 13 item 11): KPIs, market data and country "
     "COE inputs with dated sources; shortfalls per bank", "Collected by Claude for the owner, 2026-11-12 (cited "
                                                           "web sources)"),
    ("documents/benchmarks/", "Benchmark report (prompt 13 item 12; McKinsey, BCG, Wall Street Africa, Kadoo, Solomon, "
     "CardinalStone), downloaded 2026-11-12 at the owner's request; URL in manual_entries/benchmarks_collected.csv",
     "Publisher website (free download)"),
    ("manual_entries/web_collected.csv", "Web-collected values (D64): IMF WEO / REO macro forecasts and 2026 BoT "
     "T-bill auction yields, each row with source URL, page, publication and access dates; validated by "
     "tzbank.ingest.web_collected", "Collected by Claude for the owner, 2026-11-12 (cited web sources)"),
    ("manual_entries/", "Manual entries read by a person (D29)", "Owner"),
    ("dse/site/", "DSE site page (all-equities securities list or indices page), saved unchanged (gzip) with metadata "
     "(D72)", "Dar es Salaam Stock Exchange website (dse.co.tz), collected by tzbank.ingest.dse_library"),
    ("dse/profile/", "DSE company-profile page (overview, financial-reports, disclosures or dividends) for one "
     "security, saved unchanged (gzip) with metadata (D72)", "Dar es Salaam Stock Exchange website (dse.co.tz), "
                                                              "collected by tzbank.ingest.dse_library"),
    ("dse/documents/", "DSE-hosted company document (PDF) listed on the company's financial-reports or disclosures "
     "page, saved unchanged with a metadata JSON (D72; git-ignored, third-party copyright)",
     "Dar es Salaam Stock Exchange website (dse.co.tz), collected by tzbank.ingest.dse_library"),
    ("dse/trading_history/", "DSE company-profile trading-history page for one security, saved unchanged (gzip) "
     "with a metadata JSON (URL, UTC collection time, HTTP status, body sha256); one immutable snapshot per "
     "collection (D71)", "Dar es Salaam Stock Exchange website (dse.co.tz), collected by "
                         "tzbank.ingest.dse_trading_history"),
    ("documents/CRDB/crdb-ir-2025-8th-may-2026.pdf", "CRDB Bank Integrated Report and Consolidated Financial "
     "Statements FY2025 (333 pp; Sources CRDB-AR-FY2025)", "CRDB Bank website (file name as posted)"),
    ("documents/NMB/NMB 2023 Integrated.pdf", "NMB Integrated Annual Report FY2023, re-saved copy (same 218 pages as "
     "'NMB Integrated Annual Report 2023.pdf'; Sources NMB-AR-FY2023)", "NMB Bank website"),
    ("Tanzania_Banking_Research_Database.xlsx", "Owner's BoT research database v1 (BoT Annual Reports "
     "FY2002/03-FY2014/15; observations 1993 to June 2015). Raw, immutable; merged with v2 under D40",
     "Owner (compiled from BoT Annual Reports)"),
    ("Tanzania_Banking_Research_Database_v2.xlsx", "Owner's BoT research database v2 (BoT Annual Reports "
     "FY2015/16-FY2024/25; observations to June 2025). Raw, immutable; supersedes v1 on shared keys (D40)",
     "Owner (compiled from BoT Annual Reports)"),
    # non-BoT documents filed in documents/reports (prompts 24 and 24b): described by file before the BoT catch-all
    ("documents/reports/1tzaea2024001-print-pdf.pdf", "IMF Country Report No. 24/187 (June 2024): Tanzania, Third "
     "Review under the ECF; Annex VII para. 3 (printed p. 78) restructured-loan sensitivity analysis (prompt 24b)",
     "International Monetary Fund website"),
    ("documents/reports/002-article-A003-en.pdf", "IMF: Tanzania 2023 Article IV Consultation and First Review under "
     "the ECF, debt sustainability analysis (prompt 24 context)", "International Monetary Fund website"),
    ("documents/reports/121436-", "World Bank: Political Economy of Public Financial Management Reforms (Tanzania); "
     "Ch5 context box, printed p. 63 (prompt 24)", "World Bank website"),
    ("documents/reports/P177386", "World Bank: Privatizing Growth, Tanzania Country Economic Memorandum; Ch5 context "
     "box, printed pp. 16, 63 (prompt 24)", "World Bank website"),
    ("documents/reports/MPS ", "Bank of Tanzania Monetary Policy Statement (June issue of the year in the file name); "
     "Ch5 context box (prompt 24)", "Bank of Tanzania website"),
    ("documents/reports/2020073013503630.pdf", "Bank of Tanzania Annual Report 2015/16, a second copy of the BoT file "
     "(not read by the BoT extractors); Ch5 context box, printed pp. viii, 15 (prompt 24)", "Bank of Tanzania website"),
    ("documents/reports/", "Bank of Tanzania Annual Report (source of the BoT research database, D40; code-extracted "
     "tables, D39)", "Bank of Tanzania website"),
    ("documents/damodaran/", "Damodaran archived dataset (ctryprem, betaemerg from "
     "https://pages.stern.nyu.edu/~adamodar/pc/archives/; histimpl from .../pc/datasets/histimpl.xls), downloaded "
     "2026-11-08 at the owner's request (prompt 05 pre-flight 3); file suffix yy = data at end-yy, published January "
     "yy+1", "Aswath Damodaran, NYU Stern (public website)"),
    ("documents/dse/", "DSE market information document", "Dar es Salaam Stock Exchange"),
    ("documents/bot/MER - ", "BoT Monthly Economic Review, December 2024 (end-of-period TZS/USD table; D64)",
     "Bank of Tanzania website (https://www.bot.go.tz/Publications/Regular/Monthly%20Economic%20Review/en/"
     "2025011009411334.pdf), downloaded 2026-11-12"),
    ("documents/bot/FSR - ", "BoT Financial Stability Report, December 2023 (D-SIB capital buffer statement, p.44; D64)",
     "Bank of Tanzania website (https://www.bot.go.tz/Publications/Regular/Financial%20Stability/sw/"
     "2024062419342761.pdf), downloaded 2026-11-12"),
    ("documents/bot/BSR - ", "BoT Banking Supervision Annual Report 2025 (FSI appendix; ROA with profit before tax, "
     "Figure 3.3; D64)", "Bank of Tanzania website (https://www.bot.go.tz/Publications/Other/Banking%20Supervision%20"
     "Annual%20Reports/sw/2026070216351589.pdf), downloaded 2026-11-12"),
    ("documents/imf/", "IMF Regional Economic Outlook: Sub-Saharan Africa, October 2025 (Statistical Appendix SA1; "
     "D64)", "IMF (https://www.imf.org/-/media/files/publications/reo/afr/2025/october/english/text.pdf), downloaded "
     "2026-11-12"),
    ("documents/bot/auction_screenshot", "Screenshot of a BoT Treasury bond auction result page", "Bank of Tanzania website"),
    ("documents/bot/", "BoT National Payment System annual report", "Bank of Tanzania"),
    ("documents/", "Bank annual report, interim statement or Phase 1 extraction-kit file", "Bank publication / owner"),
]


_GENERIC = {_RULES[-1][1], "Raw input"} | {  # fallback descriptions, replaced when a specific rule exists; the
    d for pre, d, _ in _RULES if pre == "documents/reports/"}  # BoT catch-all too (non-BoT files there, prompt 24b)

# Raw files kept out of git for licence reasons (D15); their hash stays in MANIFEST.
UNCOMMITTED = {"finscope/FinScope.csv"}


def _describe(relpath: str) -> tuple[str, str]:
    for prefix, desc, src in _RULES:
        if relpath == prefix or relpath.startswith(prefix):
            return desc, src
    return "Raw input", "Owner"


def build_manifest(root: Path | None = None) -> pd.DataFrame:
    """Hash every file under data/raw (except MANIFEST.csv itself)."""
    root = root or find_root()
    raw = path("raw", root)
    rows = []
    for f in sorted(p for p in raw.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"):
        r = rel(f, raw)
        desc, src = _describe(r)
        # third-party PDFs are never committed (CLAUDE.md section 9): source documents, peer reports, broker notes
        third_party = r.startswith(("documents/", "peers/")) or (
            r.startswith(("analyst_reports/", "quarterly/")) and r.endswith(".pdf"))
        committed = "N" if third_party or r in UNCOMMITTED else "Y"
        received = dt.datetime.fromtimestamp(f.stat().st_mtime).date().isoformat()
        rows.append({"file": r, "sha256": sha256(f), "bytes": f.stat().st_size, "received_date": received,
                     "description": desc, "source": src, "committed (Y/N)": committed})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(root: Path | None = None, notes: dict[str, str] | None = None) -> Path:
    """Write data/raw/MANIFEST.csv.

    Unchanged files keep their row. A file whose hash changed gets today's date as received_date and, if given,
    ``notes[file]`` as its description (D18: description of the change). Files no longer present keep their row
    only if they are uncommitted (documents, FinScope microdata), so a clone without them still verifies.
    """
    root = root or find_root()
    out = path("manifest", root)
    new = build_manifest(root)
    notes = notes or {}
    if out.exists():
        old = pd.read_csv(out, dtype=str).set_index("file")
        rows = []
        for _, r in new.iterrows():
            f = r["file"]
            if f in old.index and old.at[f, "sha256"] == r["sha256"]:
                keep = old.loc[f].to_dict() | {"file": f, "committed (Y/N)": r["committed (Y/N)"]}
                if keep["description"] in _GENERIC and r["description"] not in _GENERIC:
                    keep |= {"description": r["description"], "source": r["source"]}  # a rule added later
                rows.append(keep)
                continue
            if f in old.index:
                r["received_date"] = pd.Timestamp.today().date().isoformat()
                r["description"] = notes.get(f, old.at[f, "description"])
            elif f in notes:
                r["description"] = notes[f]
            rows.append(r.to_dict())
        present = set(new["file"])
        for f, r in old.iterrows():
            if f not in present and r["committed (Y/N)"] == "N":
                rows.append(r.to_dict() | {"file": f})
        new = pd.DataFrame(rows)[MANIFEST_COLUMNS].sort_values("file")
    return write_csv(new.reset_index(drop=True), out)


def verify_manifest(root: Path | None = None, include_documents: bool = False) -> pd.DataFrame:
    """Compare current hashes with MANIFEST.csv. Returns rows with status OK / CHANGED / MISSING / UNREGISTERED."""
    root = root or find_root()
    raw = path("raw", root)
    man = pd.read_csv(path("manifest", root), dtype=str)
    rows = []
    for _, r in man.iterrows():
        if r["file"].startswith("documents/") and not include_documents:
            continue
        f = raw / r["file"]
        if not f.exists():
            status = "MISSING" if r["committed (Y/N)"] == "Y" else "NOT_PRESENT_UNCOMMITTED"
        else:
            status = "OK" if sha256(f) == r["sha256"] else "CHANGED"
        rows.append({"file": r["file"], "status": status})
    registered = set(man["file"])
    for f in sorted(p for p in raw.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"):
        r = rel(f, raw)
        if r.startswith("documents/") and not include_documents:
            continue
        if r not in registered:
            rows.append({"file": r, "status": "UNREGISTERED"})
    return pd.DataFrame(rows, columns=["file", "status"])


def input_hashes(root: Path | None = None) -> dict[str, str]:
    """sha256 of every committed raw input (for run logs), read from MANIFEST.csv."""
    root = root or find_root()
    man = pd.read_csv(path("manifest", root), dtype=str)
    man = man[man["committed (Y/N)"] == "Y"]
    return dict(zip(man["file"], man["sha256"], strict=False))
