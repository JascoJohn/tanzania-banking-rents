"""Leakage check (prompt 23, D92): no internal name reaches a report PDF.

Fails the build when a PDF page prints a pipeline table name (``chNN_...``), an unresolved token (``[[`` or ``{{``),
a backtick, or a task reference ("prompt NN"). The owner's reports cite the findings pack ("Findings pack vX.Y") as
the traceable reference; table names, tokens and prompt numbers belong in the repository, not the PDF.

Usage: ``python -m tzbank.reporting.leak_check [--report main|nmb|crdb|foreign|all]``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PATTERNS = {
    "table name": re.compile(r"\bch\d{2}_\w+", re.I),
    "unresolved token": re.compile(r"\[\[|\]\]|\{\{|\}\}"),
    "backtick": re.compile("`"),
    "prompt reference": re.compile(r"\bprompts? \d+", re.I),
    # a raw column name such as cost_to_income (prompt 24): two or more lower-case words joined by underscores
    "snake_case name": re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
}
COLUMNS = ["report", "page", "kind", "match", "context"]


def scan_text(text: str) -> list[tuple[str, str, str]]:
    """(kind, match, context) for every leak in one page of text. Line breaks are joined first, so a name that
    wraps across lines is still found (Blueprint data layer: report output checks)."""
    flat = re.sub(r"-\s*\n\s*", "", text)  # a name hyphenated at a line end
    flat = re.sub(r"_\s*\n\s*", "_", flat)  # a name broken after an underscore
    flat = re.sub(r"\s+", " ", flat)
    out = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(flat):
            if kind == "snake_case name" and max(len(x) for x in m.group(0).split("_")) < 3:
                continue  # maths notation such as k_e, not a column name
            out.append((kind, m.group(0), flat[max(0, m.start() - 60):m.end() + 60].strip()))
    return out


def scan_pdf(pdf: Path, report: str = "") -> pd.DataFrame:
    """Every leak in a PDF, one row per match (empty frame when the PDF is clean)."""
    import pymupdf

    rows = []
    with pymupdf.open(pdf) as d:
        for i, page in enumerate(d):
            rows += [{"report": report or pdf.stem, "page": i + 1, "kind": k, "match": m, "context": c}
                     for k, m, c in scan_text(page.get_text())]
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> int:
    from tzbank.config import find_root

    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.leak_check", description=__doc__.split("\n")[0])
    ap.add_argument("--report", default="all", choices=["main", "nmb", "crdb", "foreign", "all"])
    ap.add_argument("--dir", help="folder holding the PDFs (default outputs/reports)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    d = Path(a.dir) if a.dir else find_root() / "outputs" / "reports"
    reps = ["main", "nmb", "crdb", "foreign"] if a.report == "all" else [a.report]
    hits = pd.concat([scan_pdf(d / f"{r}.pdf", r) for r in reps if (d / f"{r}.pdf").exists()], ignore_index=True)
    for r in reps:
        print(f"{r}: {int((hits['report'] == r).sum())} leak(s)")
    for h in hits.itertuples():
        print(f"  {h.report} p.{h.page} [{h.kind}] {h.match}: ...{h.context}...")
    return 1 if len(hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
