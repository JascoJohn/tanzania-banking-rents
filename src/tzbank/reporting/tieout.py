"""Full tie-out of a built report against the findings pack (prompt 17 Part C; D65).

``python -m tzbank.reporting.tieout [--report main|nmb|crdb|foreign|all]``

Every number in the built ``outputs/reports/<report>.docx`` (all numeric tokens the figure checker reads, tables
included) is listed with its page in the PDF, its chapter, the sentence fragment, the value as printed, the matched
pack id and value, and a status:

- ``match``: equals a pack value at the printed precision. ``match_strength``: ``confirmed`` when a matching pack id
  shares a topic word with the sentence or table row, ``unconfirmed`` when none does (possibly a coincidental match,
  e.g. an outside figure that happens to equal an unrelated pack value), ``weak`` for a small whole number matching
  many unrelated pack values;
- ``rounding``: misses at the printed precision but agrees within one coarser digit or 0.5%;
- ``mismatch``: a pack value lies within 5% but the figure does not agree (the nearest pack id and value are given);
- ``outside source``: not a pack figure, and the sentence cites its source;
- ``unmatched``: not a pack figure and no citation in the sentence (including external-rule figures without one);
- ``not a figure``: a year, a date, a document or section reference, a phone number, ORCID or URL.

Outputs (git-ignored, like the reports): ``outputs/reports/<report>_tieout.csv``, ``tieout_summary.csv`` (per report
and chapter), ``tieout_mismatches.csv`` (with the pack value to use) and ``tieout_cross_report.csv`` (the same pack
figure printed differently in two reports).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from tzbank.config import find_root, path
from tzbank.reporting.check_figures import check, read_text
from tzbank.reporting.report_source import parse_parts, source_files, split_heading

MONTHS = r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)"
DATE_DAY = re.compile(r"^\d{1,2}$")
CITATION = re.compile(r"\(source|\[ext|according to|\bBoT\b|Bank of Tanzania|\bIMF\b|World Bank|McKinsey|FinScope|"
                      r"Damodaran|DSE market report|annual report|\bTRA\b|\bCMSA\b|\bNBS\b|survey|\(\w[^()]*\d{4}\)",
                      re.I)
REPORTS = ("main", "nmb", "crdb", "foreign")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("|", " ").replace(" ", " ").replace(" ", " ")).strip()


def _pages(pdf: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(pdf) as d:
        return [_norm(p.get_text()) for p in d]


def _chapters(root: Path, report: str) -> list[str]:
    files = source_files(root / "reports" / "source", report)
    if not files:
        return []
    out = []
    for n in parse_parts(files).nodes:
        if n.kind in ("part", "chapter"):
            out.append(_norm(n.text))
    return out


def _pack(root: Path) -> pd.DataFrame:
    f = sorted((path("outputs", root) / "findings").glob("findings_pack_v*.csv"),
               key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem)])[-1]
    p = pd.read_csv(f, low_memory=False)
    return p.set_index("id")


SENT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\[])")


def _sentence(line: str, context: str, text: str) -> str:
    """The sentence of ``line`` that holds the number (located through its checker context)."""
    if not line:
        return context
    core = context.strip()[:40]
    pos = line.find(core) if core else -1
    pos = pos + context.strip().find(text) if pos >= 0 else line.find(text)
    start = 0
    for m in SENT.finditer(line):
        if m.start() > pos >= 0:
            return line[start:m.start()]
        start = m.end()
    return line[start:]


GENERIC = {"value", "values", "share", "ratio", "table", "nmb", "crdb", "curr", "hist", "base", "the", "and", "per",
           "tzs", "usd", "data", "text", "row", "fy", "series", "bank", "banks"}


def _id_words(pid: str) -> set[str]:
    """Topic words of a pack id (CH13.PRICE_LINES.PRICE.NMB.Sell_above_... -> price, lines, sell, above, ...)."""
    toks = re.split(r"[._\s]+", pid.lower())
    return {t for t in toks if len(t) >= 3 and not re.match(r"^(ch\d+|fy\d+|r\d+|\d+)$", t) and t not in GENERIC}


def _same_topic(pid: str, sentence: str) -> bool:
    words = set(re.findall(r"[a-z]{3,}", sentence.lower()))
    stems = {w[:5] for w in words}
    return any(t in words or t[:5] in stems for t in _id_words(pid))


def classify(r: dict, pack: pd.DataFrame) -> tuple[str, str, object, str]:
    """(status, pack id, pack value, strength) for one checker row (statuses: module docstring)."""
    ctx = str(r.get("context", ""))
    sent = str(r.get("sentence") or ctx)
    st = r["status"]
    txt = str(r["text"])
    if st in ("ignored", "quotation") or r.get("references"):
        return "not a figure", "", None, ""
    if DATE_DAY.match(txt.strip()) and re.search(rf"\b{re.escape(txt.strip())}\s+{MONTHS}\b", ctx):
        return "not a figure", "", None, ""
    if re.search(r"doi\.org|\b\d+\(\d+[^)]*\),\s*\d+", sent):  # a bibliographic entry: volume, pages, DOI
        return "not a figure", "", None, ""
    ids = str(r.get("pack_ids") or "")
    cited = CITATION.search(sent)
    if st == "matched":
        cands = [x.strip() for x in ids.replace(" ...", "").split(";") if x.strip()]
        topical = [c for c in cands if _same_topic(c, sent)]
        first = (topical or cands or [""])[0]
        val = pack["value"].get(first) if first in pack.index else None
        if float(r["value"]).is_integer() and abs(float(r["value"])) <= 31 and len(cands) > 1 and not topical:
            strength = "weak"
        else:
            strength = "confirmed" if topical else "unconfirmed"
        return "match", first, val, strength
    if st == "mismatch":
        m = re.match(r"nearest (\S+) = (\S+)", ids)
        pid, pv = (m.group(1), float(m.group(2))) if m else ("", None)
        if pid in pack.index:
            try:
                pv = float(pack["value"][pid])
            except (TypeError, ValueError):
                pass
        if pv is not None and _same_topic(pid, sent):  # a near value of the same quantity
            printed, tol = float(r["value"]), float(r["tolerance"])
            about = re.search(r"\b(about|around|roughly|nearly|almost|some)\b", ctx, re.I)
            k = 10 if about else 3
            rnd = any(abs(c - printed) <= k * tol + 1e-12 for c in (pv, pv * 100))
            return ("rounding" if rnd else "mismatch"), pid, pv, ""
        return ("outside source" if cited else "unmatched"), pid, pv, "nearest pack value unrelated"
    if st in ("external", "unmatched") and cited:
        return "outside source", "", None, ""
    return "unmatched", "", None, ""


def tieout(root: Path, report: str) -> pd.DataFrame:
    out = path("outputs", root) / "reports"
    docx, pdf = out / f"{report}.docx", out / f"{report}.pdf"
    if not docx.exists():
        raise FileNotFoundError(f"{docx} not built: run python -m tzbank.reporting.build_reports --report {report}")
    pack = _pack(root)
    res = check(docx, root=root)
    lines = read_text(docx).splitlines()
    titles = _chapters(root, report)
    chap_of_line, cur = {}, ""
    for i, ln in enumerate(lines, 1):
        n = _norm(ln)
        hit = next((t for t in titles if n == t or (len(t) > 12 and n.endswith(t))), None)
        if hit:
            cur = hit
        chap_of_line[i] = cur
    pages = _pages(pdf) if pdf.exists() else []
    page_i = 0
    rows = []
    for r in res.to_dict("records"):
        r["sentence"] = _sentence(lines[int(r["line"]) - 1] if 0 < int(r["line"]) <= len(lines) else "",
                                  str(r["context"]), str(r["text"]))
        r["references"] = "reference" in chap_of_line.get(int(r["line"]), "").lower()
        status, pid, pv, strength = classify(r, pack)
        frag = _norm(str(r["context"]))
        page, approx = None, False
        if pages:
            words = frag.split()
            probes = [frag] + [" ".join(words[k:k + 5]) for k in range(0, max(1, len(words) - 4), 2)] + [
                _norm(str(r["text"]))]
            for pr in probes:
                if not pr:
                    continue
                idx = next((k for k in range(page_i, len(pages)) if pr in pages[k]), None)
                if idx is None:
                    idx = next((k for k in range(0, page_i) if pr in pages[k]), None)
                if idx is not None:
                    page, approx = idx + 1, pr == probes[-1]
                    if not approx:  # reading order: later figures are searched from this page on
                        page_i = idx
                    break
        chap = chap_of_line.get(int(r["line"]), "")
        lab, title = split_heading(chap) if chap else ("", "")
        rows.append({"report": report, "page": page, "page_approx": approx, "chapter": chap,
                     "chapter_label": lab, "fragment": frag, "printed": r["text"], "value": r["value"],
                     "status": status, "match_strength": strength, "pack_id": pid, "pack_value": pv,
                     "checker_status": r["status"], "n_pack_ids": len([x for x in str(r.get("pack_ids") or "").split(";")
                                                                       if x.strip() and x.strip() != "..."]),
                     "not_on_dashboard": bool(r.get("not_on_dashboard", False))})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"{report}_tieout.csv", index=False)
    return t


def summary(all_: pd.DataFrame) -> pd.DataFrame:
    order = ["match", "rounding", "mismatch", "outside source", "unmatched", "not a figure"]
    a = all_.assign(chapter=all_["chapter"].fillna("").replace("", "(front matter)"))
    a["unconfirmed_match"] = (a["status"] == "match") & a["match_strength"].fillna("").isin(["unconfirmed", "weak"])

    def piv(keys):
        t = a.pivot_table(index=keys, columns="status", values="value", aggfunc="size", fill_value=0)
        t = t.reindex(columns=order, fill_value=0)
        t["of_which_unconfirmed_match"] = a.groupby(keys)["unconfirmed_match"].sum()
        return t.reset_index()

    by_rep = piv(["report"]).assign(chapter="(all)")
    by_ch = piv(["report", "chapter"])
    return pd.concat([by_rep, by_ch], ignore_index=True)[["report", "chapter", *order, "of_which_unconfirmed_match"]]


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", str(t).lower())}


def cross_report(all_: pd.DataFrame) -> pd.DataFrame:
    """The same quantity printed differently in two or more reports: rows that match exactly one pack id, whose
    sentences share wording (Jaccard >= 0.25 on words of 4+ letters) across reports. ``kind``: 'different value'
    when the printed values differ by more than 1%, else 'precision only' (26.9% vs 27%)."""
    x = all_[all_["status"].isin(["match", "rounding", "mismatch"]) & (all_["pack_id"].fillna("") != "")
             & (all_["n_pack_ids"] == 1) & (all_["match_strength"].fillna("") != "weak")]
    rows = []
    for pid, g in x.groupby("pack_id"):
        if g["report"].nunique() < 2:
            continue
        recs = g.to_dict("records")
        keep = set()
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if a["report"] == b["report"] or str(a["printed"]).strip() == str(b["printed"]).strip():
                    continue
                wa, wb = _words(a["fragment"]), _words(b["fragment"])
                if wa and wb and len(wa & wb) / len(wa | wb) >= 0.25:
                    keep |= {i, recs.index(b)}
        if not keep:
            continue
        vals = [float(recs[i]["value"]) for i in keep]
        spread = (max(vals) - min(vals)) / max(abs(max(vals)), 1e-12)
        kind = "different value" if spread > 0.01 else "precision only"
        for i in sorted(keep):
            r = recs[i]
            rows.append({"pack_id": pid, "pack_value": r["pack_value"], "kind": kind, "report": r["report"],
                         "page": r["page"], "printed": r["printed"], "fragment": r["fragment"]})
    return pd.DataFrame(rows, columns=["pack_id", "pack_value", "kind", "report", "page", "printed", "fragment"])


def run(root: Path, reports: list[str]) -> dict:
    out = path("outputs", root) / "reports"
    tabs = [tieout(root, r) for r in reports]
    a = pd.concat(tabs, ignore_index=True)
    s = summary(a)
    s.to_csv(out / "tieout_summary.csv", index=False)
    mm = a[a["checker_status"] == "mismatch"][["report", "page", "chapter", "fragment", "printed", "status",
                                                "pack_id", "pack_value"]]
    mm.to_csv(out / "tieout_mismatches.csv", index=False)
    cr = cross_report(a)
    cr.to_csv(out / "tieout_cross_report.csv", index=False)
    return {"rows": len(a), "summary": s, "mismatches": mm, "cross": cr}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.tieout")
    ap.add_argument("--report", default="all", choices=[*REPORTS, "all"])
    a = ap.parse_args(argv)
    root = find_root()
    sys.stdout.reconfigure(encoding="utf-8")
    reps = list(REPORTS) if a.report == "all" else [a.report]
    reps = [r for r in reps if (path("outputs", root) / "reports" / f"{r}.docx").exists()]
    res = run(root, reps)
    print(res["summary"][res["summary"]["chapter"] == "(all)"].to_string(index=False))
    cr = res["cross"]
    print(f"{len(res['mismatches'])} checker mismatches; across reports: "
          f"{cr.loc[cr['kind'] == 'different value', 'pack_id'].nunique()} figures with different values, "
          f"{cr.loc[cr['kind'] == 'precision only', 'pack_id'].nunique()} at different precision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
