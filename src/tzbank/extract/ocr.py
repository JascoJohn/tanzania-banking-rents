"""OCR for scanned bank PDFs (prompt 02 OCR task; data layer).

Scanned PDFs (no usable text layer) are passed through OCRmyPDF + Tesseract. The original under
data/raw/documents is never modified: the OCR copy (same pages, same page numbers, invisible text layer added) is
written to data/ocr/<bank>/<file>.pdf with a plain-text sidecar, and every attempt is logged in
data/ocr/ocr_log.csv. data/ocr/ is git-ignored (derived from uncommitted documents).

Extraction then runs the unchanged text-layer pipeline on the OCR copy. Because OCR can misread digits, values
from an OCR copy are accepted only through an arithmetic gate (see ``tzbank.extract.gated``); the rest are flagged
for manual verification.

System dependencies: Tesseract OCR 5.x with the 'eng' and 'osd' language data. OCRmyPDF (Python) rasterises with
pypdfium2, so Ghostscript and Poppler are not required. Set TZBANK_TESSERACT to the folder holding tesseract(.exe)
if it is not on PATH; the Windows default install folder is detected automatically.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pymupdf

from tzbank.config import find_root, path
from tzbank.io import write_csv

WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR")
MIN_TEXT_SHARE = 0.5  # a PDF needs OCR when fewer than half its pages carry a text layer
LOG_COLUMNS = ["file", "pages", "text_pages_before", "text_pages_after", "status", "seconds", "ocr_file", "sidecar",
               "ocrmypdf_version", "tesseract_version", "run_at", "message"]


def ensure_tesseract() -> str | None:
    """Put Tesseract on PATH if it is installed in a known place; return its version string or None."""
    folders = [os.environ.get("TZBANK_TESSERACT", ""), str(WINDOWS_TESSERACT)]
    if not shutil.which("tesseract"):
        for f in folders:
            if f and (Path(f) / ("tesseract.exe" if sys.platform == "win32" else "tesseract")).exists():
                os.environ["PATH"] = f + os.pathsep + os.environ.get("PATH", "")
                break
    exe = shutil.which("tesseract")
    if not exe:
        return None
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def text_page_count(pdf: Path) -> tuple[int, int]:
    """(pages, pages with > 200 characters of text)."""
    d = pymupdf.open(pdf)
    n = d.page_count
    t = sum(1 for p in d if len(p.get_text()) > 200)
    d.close()
    return n, t


def needs_ocr(pdf: Path) -> bool:
    """True when fewer than MIN_TEXT_SHARE of the pages have a text layer."""
    n, t = text_page_count(pdf)
    return n > 0 and t / n < MIN_TEXT_SHARE


def ocr_dir(root: Path | None = None) -> Path:
    return find_root() / "data" / "ocr" if root is None else root / "data" / "ocr"


def ocr_target(pdf: Path, root: Path | None = None) -> Path:
    """data/ocr/<bank folder>/<file name> for a document under data/raw/documents."""
    root = root or find_root()
    try:
        rel = pdf.resolve().relative_to(path("documents", root).resolve())
    except ValueError:
        dse = (path("raw", root) / "dse" / "documents").resolve()
        if pdf.resolve().is_relative_to(dse):  # DSE library documents (data/raw/dse/documents/<ticker>/; prompt 15)
            rel = Path("dse") / pdf.resolve().relative_to(dse)
        else:  # any other raw folder (quarterly statements, broker notes; prompts 19-20): data/ocr/<path under raw>
            rel = pdf.resolve().relative_to(path("raw", root).resolve())
    return ocr_dir(root) / rel


def extraction_source(pdf: Path, root: Path | None = None) -> Path:
    """The file the text pipeline should read: the OCR copy for a scanned PDF when it exists, else the original."""
    if needs_ocr(pdf):
        t = ocr_target(pdf, root)
        if t.exists():
            return t
    return pdf


def run_ocr(pdf: Path, root: Path | None = None, jobs: int = 10, force: bool = False) -> dict[str, object]:
    """OCR one scanned PDF to data/ocr; never touches the original. Returns a log row."""
    import ocrmypdf

    root = root or find_root()
    target = ocr_target(pdf, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix(".txt")
    n, before = text_page_count(pdf)
    row: dict[str, object] = {"file": pdf.resolve().relative_to(path("raw", root).resolve()).as_posix(),
                              "pages": n, "text_pages_before": before, "ocr_file": target.relative_to(root).as_posix(),
                              "sidecar": sidecar.relative_to(root).as_posix(), "ocrmypdf_version": ocrmypdf.__version__,
                              "tesseract_version": ensure_tesseract(), "run_at": dt.datetime.now().isoformat(timespec="seconds")}
    if target.exists() and not force:
        _, after = text_page_count(target)
        return row | {"text_pages_after": after, "status": "OK (cached)", "seconds": 0, "message": ""}
    t0 = time.time()
    try:
        ocrmypdf.ocr(pdf, target, language=["eng"], output_type="pdf", rasterizer="pypdfium", rotate_pages=True,
                     deskew=True, oversample=300, jobs=jobs, sidecar=sidecar, progress_bar=False,
                     skip_text=True, optimize=0)
        _, after = text_page_count(target)
        msg = ""
        if after < max(1, 0.5 * n):
            # pages carrying a few stray characters are skipped by skip_text; rasterise them all and retry
            ocrmypdf.ocr(pdf, target, language=["eng"], output_type="pdf", rasterizer="pypdfium",
                         rotate_pages=True, deskew=True, oversample=300, jobs=jobs, sidecar=sidecar,
                         progress_bar=False, force_ocr=True, optimize=0)
            _, after = text_page_count(target)
            msg = "retried with force_ocr (stray text layer)"
        status = "OK" if after >= max(1, 0.5 * n) else "PARTIAL (few pages recognised)"
    except Exception as exc:  # noqa: BLE001 - one failed file must not stop the batch
        after, status, msg = 0, "FAILED", f"{type(exc).__name__}: {exc}"[:300]
    return row | {"text_pages_after": after, "status": status, "seconds": round(time.time() - t0, 1), "message": msg}


def scanned_documents(root: Path | None = None, folders: list[str] | None = None) -> list[Path]:
    """Every PDF under data/raw/documents/<bank>/ that needs OCR (regulation and DSE folders excluded)."""
    root = root or find_root()
    base = path("documents", root)
    folders = folders or ["DCB", "Maendeleo", "MKCB", "MCB", "MUCOBA", "NMB", "CRDB"]
    return [p for f in folders for p in sorted((base / f).glob("*.pdf")) if needs_ocr(p)]


def run_all(root: Path | None = None, folders: list[str] | None = None, force: bool = False) -> Path:
    """OCR every scanned document and write data/ocr/ocr_log.csv (merged with earlier runs)."""
    root = root or find_root()
    if ensure_tesseract() is None:
        raise RuntimeError("Tesseract not found. Install it (README 'OCR') or set TZBANK_TESSERACT.")
    log_file = ocr_dir(root) / "ocr_log.csv"
    old = pd.read_csv(log_file) if log_file.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    rows = []
    for pdf in scanned_documents(root, folders):
        r = run_ocr(pdf, root, force=force)
        print(f"{r['status']:10s} {r['file']} ({r['text_pages_after']}/{r['pages']} pages, {r['seconds']}s)")
        rows.append(r)
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    merged = pd.concat([old[~old["file"].isin(new["file"])], new], ignore_index=True).sort_values("file")
    return write_csv(merged, log_file)


if __name__ == "__main__":
    print(run_all(folders=sys.argv[1:] or None))
