"""The ``pdf`` stage: work that reads the git-ignored source PDFs (prompt 02f section 5).

Jobs, each cached by an input fingerprint (``outputs/run_logs/cache_<job>.json``):

- ``bot_reports``: BoT Annual Report table inventory and code-extracted supplement (D39, D40);
- ``inclusion_nps_supplement``: code-extracted inclusion and NPS tables (D39);
- ``inspection_manual_list``: PDF inspection, collection tasks and the manual check list (slow, ~10 minutes).

The fingerprint is a sha256 over the job's small input files (content), the PDFs it reads (path, size and
modification time) and its own source modules. ``python -m tzbank run all --skip-slow`` reuses a job's outputs when
the fingerprint matches the cached one and every output exists; without the flag every job runs. Without the PDFs
(a fresh clone) the stage skips cleanly and the committed outputs stay as they are.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from tzbank.config import path
from tzbank.io import sha256

log = logging.getLogger("tzbank")


def fingerprint(files: list[Path], pdfs: list[Path], modules: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f"{f.name}:{sha256(f) if f.exists() else 'missing'}".encode())
    for f in sorted(pdfs):
        st = f.stat()
        h.update(f"{f.as_posix()}:{st.st_size}:{st.st_mtime_ns}".encode())
    for f in sorted(modules):
        h.update(f"{f.name}:{sha256(f)}".encode())
    return h.hexdigest()


def _jobs(root: Path) -> dict[str, tuple[Callable[[Path], list[Path]], list[Path], list[Path], list[Path],
                                           list[str]]]:
    """name -> (function, input files, PDFs, modules, expected outputs)."""
    from tzbank.extract.bot_reports import bot_report_files, write_bot_reports
    from tzbank.extract.inclusion_nps_supplement import write_supplements

    docs, interim, src = path("documents", root), path("interim", root), root / "src" / "tzbank"
    ext = root / "data" / "extraction"

    def _bot_policy(r: Path) -> list[Path]:
        from tzbank.extract.bot_policy import write_bot_policy

        return write_bot_policy(r)

    def _sector_loans(r: Path) -> list[Path]:
        from tzbank.extract.sector_loans import write_sector_candidates

        return write_sector_candidates(r)

    def _bot_fx(r: Path) -> list[Path]:
        from tzbank.extract.bot_fx import write_bot_fx

        return write_bot_fx(r)

    def inspection(r: Path) -> list[Path]:
        from tzbank.extract.inspection import write_inspection
        from tzbank.extract.manual_list import write_manual_list

        return write_inspection(r) + write_manual_list(r)

    bank_pdfs = [f for d in ("CRDB", "NMB", "DCB", "Maendeleo", "MKCB", "MCB", "MUCOBA")
                 for f in (docs / d).glob("*.pdf")]
    return {
        "bot_reports": (write_bot_reports, [interim / "bot_research_database.csv"],
                        bot_report_files(root),
                        [src / "extract" / "bot_reports.py", src / "extract" / "pdftext.py"],
                        ["reports/bot_report_inventory.csv", "data/interim/bot_code_extracted.csv"]),
        "bot_policy": (_bot_policy, [], bot_report_files(root) + list((docs / "NMB").glob("*.pdf"))
                       + list((docs / "CRDB").glob("*.pdf")),
                       [src / "extract" / "bot_policy.py", src / "extract" / "pdftext.py"],
                       ["data/interim/bot_policy_text.csv", "data/interim/bot_policy_statements.csv"]),
        "inclusion_nps_supplement": (write_supplements, [],
                                     list((docs / "inclusion").glob("*.pdf")) + list((docs / "nps").glob("*.pdf")),
                                     [src / "extract" / "inclusion_nps_supplement.py"],
                                     ["data/interim/inclusion_code_extracted.csv",
                                      "data/interim/nps_code_extracted.csv",
                                      "reports/inclusion_nps_code_extraction_review.csv"]),
        "sector_loans": (_sector_loans, [],
                         list((docs / "NMB").glob("*.pdf")) + list((docs / "CRDB").glob("*.pdf")),
                         [src / "extract" / "sector_loans.py", src / "extract" / "pdftext.py"],
                         ["data/interim/loans_by_sector_candidates.csv"]),
        "bot_fx": (_bot_fx, [], bot_report_files(root) + list((docs / "bot").glob("MER - *.pdf")),
                   [src / "extract" / "bot_fx.py", src / "extract" / "pdftext.py"],
                   ["data/interim/bot_fx_eop.csv", "reports/validation/bot_fx_check.csv"]),
        "inspection_manual_list": (inspection, [ext / "ledger.csv", ext / "candidates_all.csv",
                                                ext / "detail_lines.csv", ext / "detail_lines_review.csv",
                                                interim / "inputs_historical_financials.csv",
                                                interim / "inputs_regulatory_capital.csv"], bank_pdfs,
                                   [src / "extract" / "inspection.py", src / "extract" / "manual_list.py"],
                                   ["reports/pdf_inspection.csv", "reports/collection_tasks.csv",
                                    "reports/manual_check_list.csv", "reports/manual_check_summary.csv"]),
    }


def run_pdf_stage(root: Path, skip_slow: bool = False) -> list[Path]:
    out: list[Path] = []
    for name, (fn, files, pdfs, modules, outputs) in _jobs(root).items():
        if not pdfs:
            log.info("pdf stage: %s skipped (source PDFs not held)", name)
            continue
        fp = fingerprint(files, pdfs, modules)
        cache = path("run_logs", root) / f"cache_{name}.json"
        old = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        if skip_slow and old.get("fingerprint") == fp and all((root / o).exists() for o in outputs):
            log.info("pdf stage: %s reused (inputs unchanged, fingerprint %s)", name, fp[:12])
            out += [root / o for o in outputs]
            continue
        log.info("pdf stage: %s running", name)
        out += fn(root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"fingerprint": fp, "outputs": outputs}, indent=2), encoding="utf-8")
    from tzbank.extract.bot_policy import merge_into_bot_long

    return out + merge_into_bot_long(root)  # not cached: re-applied after every ingest (prompt 05 task 1)
