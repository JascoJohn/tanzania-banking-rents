"""Recalculate the companion workbook and make the saved file deterministic and free of local paths (D104).

Engines: Microsoft Excel through COM on Windows (``excel``) or LibreOffice headless (``libreoffice``); ``auto`` tries
Excel, then LibreOffice. The recalculated file keeps cached values, so viewers without a calculation engine show the
numbers. After saving, the package is normalised: fixed timestamps in the zip and in docProps, author and company
removed, and Excel's absolute-path element dropped (a local path must never be published).
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXCEL_PS = r"""
$ErrorActionPreference = 'Stop'
$x = New-Object -ComObject Excel.Application
$x.Visible = $false
$x.DisplayAlerts = $false
$x.AskToUpdateLinks = $false
try {
  $b = $x.Workbooks.Open('%SRC%', 0, $false)
  $x.CalculateFullRebuild()
  $b.SaveAs('%DST%', 51)
  $b.Close($false)
} finally { $x.Quit() }
"""

FIXED = "2026-10-31T00:00:00Z"


class RecalcUnavailable(RuntimeError):
    """No spreadsheet engine is available to recalculate the workbook."""


def _excel(src: Path, dst: Path, timeout: int, attempts: int = 3) -> None:
    """Excel through COM; a failed start of PowerShell or Excel (transient, e.g. under memory pressure) is retried."""
    import time

    ps = EXCEL_PS.replace("%SRC%", str(src.resolve())).replace("%DST%", str(dst.resolve()))
    err = ""
    for k in range(attempts):
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode == 0 and dst.exists():
            return
        err = r.stderr.strip()[-600:]
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"Excel recalculation failed after {attempts} attempts: {err}")


def _libreoffice(src: Path, dst: Path, timeout: int) -> None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        raise RecalcUnavailable("LibreOffice not found")
    with tempfile.TemporaryDirectory() as td:
        # recalculate on load, then convert (xlsx -> xlsx keeps formulas and writes the cached values)
        prof = Path(td) / "profile"
        cmd = [exe, f"-env:UserInstallation=file:///{prof.as_posix().lstrip('/')}", "--headless", "--calc",
               "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", td, str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = Path(td) / src.name
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"LibreOffice recalculation failed: {r.stderr.strip()[-600:]}")
        shutil.copyfile(out, dst)


def recalc(src: Path, dst: Path, engine: str = "auto", timeout: int = 1800) -> str:
    """Recalculate ``src`` into ``dst``; returns the engine used. Raises RecalcUnavailable when none is available."""
    errors = []
    if engine in ("auto", "excel") and sys.platform == "win32":
        try:
            _excel(src, dst, timeout)
            return "excel"
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            errors.append(str(e))
            if engine == "excel":
                raise
    if engine in ("auto", "libreoffice"):
        try:
            _libreoffice(src, dst, timeout)
            return "libreoffice"
        except RecalcUnavailable as e:
            errors.append(str(e))
    raise RecalcUnavailable("; ".join(errors) or f"engine {engine!r} not available")


def normalise(path: Path, title: str, author: str) -> None:
    """Deterministic package: fixed zip timestamps and document properties; no absolute paths, user or company."""
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in sorted(zin.infolist(), key=lambda i: i.filename):
            data = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                t = data.decode("utf-8")
                t = re.sub(r"(<dc:creator[^>]*>).*?(</dc:creator>)", rf"\g<1>{author}\g<2>", t)
                t = re.sub(r"(<cp:lastModifiedBy[^>]*>).*?(</cp:lastModifiedBy>)", r"\g<1>tzbank\g<2>", t)
                t = re.sub(r"(<dcterms:(?:created|modified)[^>]*>).*?(</dcterms:(?:created|modified)>)",
                           rf"\g<1>{FIXED}\g<2>", t)
                if "<dc:title" not in t:
                    t = t.replace("</cp:coreProperties>", f'<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">{title}'
                                  "</dc:title></cp:coreProperties>")  # dc may be declared locally only
                data = t.encode("utf-8")
            elif info.filename == "docProps/app.xml":
                t = data.decode("utf-8")
                t = re.sub(r"<Company>.*?</Company>", "<Company></Company>", t)
                t = re.sub(r"<Manager>.*?</Manager>", "", t)
                data = t.encode("utf-8")
            elif info.filename == "xl/workbook.xml":
                t = data.decode("utf-8")
                t = re.sub(r"<mc:AlternateContent[^>]*>\s*<mc:Choice[^>]*>\s*<x15ac:absPath[^>]*/>\s*</mc:Choice>\s*"
                           r"</mc:AlternateContent>", "", t)
                t = re.sub(r"<x15ac:absPath[^>]*/>", "", t)
                data = t.encode("utf-8")
            zi = zipfile.ZipInfo(info.filename, date_time=(2026, 10, 31, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zout.writestr(zi, data)
    path.write_bytes(buf.getvalue())
