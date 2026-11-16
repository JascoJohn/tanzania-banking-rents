"""Owner's Word review -> edit list matched to the report sources (prompt 24, D93).

Reads a reviewed report (``reports/review/<report>_JJ.docx``, Track Changes on) and lists every tracked insertion,
deletion, move and formatting change (``w:ins``, ``w:del``, ``w:moveFrom`` / ``w:moveTo``, ``w:rPrChange`` /
``w:pPrChange``) - in body paragraphs, tables and text boxes, and in the headers, footers, footnotes and endnotes
(column ``part``; prompt 25) - and every comment with its anchor text. Each edit gets its page (rendered page breaks
counted, as Word saved them; body only), its paragraph before and after the change, and the best-matching line of ``reports/source`` by the
surrounding text (never by position). Writes ``reports/review/edits_<report>.csv``; the status column starts as
``to apply`` / ``generated content`` / ``formatting (ignored)`` / ``comment`` and is updated as edits are applied.

Usage: ``python -m tzbank.reporting.review_edits [--report main|nmb|crdb|foreign|all]``.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
COLUMNS = ["id", "report", "part", "page", "paragraph", "kind", "old_text", "new_text", "comment_text", "anchor_text",
           "paragraph_old", "paragraph_new", "in_table", "source_file", "source_line", "match_score", "status"]


def _q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def _text(el, deleted: bool) -> str:
    """Text of an element; ``deleted``: the 'before' view (w:delText and moved-from text, without inserted or
    moved-to text); otherwise the 'after' view (without moved-from text)."""
    out = []
    for n in el.iter(_q("t"), _q("delText"), _q("tab"), _q("br")):
        if n.tag == _q("tab"):
            out.append("\t")
        elif n.tag == _q("br"):
            out.append(" ")
        elif n.tag == _q("delText"):
            if deleted:
                out.append(n.text or "")
        else:
            anc = {a.tag for a in n.iterancestors()}
            if deleted and (_q("ins") in anc or _q("moveTo") in anc):
                continue
            if not deleted and _q("moveFrom") in anc:
                continue
            out.append(n.text or "")
    return "".join(out)


def _norm(s: str) -> str:
    s = re.sub(r"\*\*|__|`|<!--.*?-->|\{\{[^}]*\}\}|\[\[[^\]]*\]\]", " ", s)
    s = s.replace("−", "-").replace("–", "-").replace("’", "'").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def source_lines(src_dir: Path, report: str) -> list[tuple[str, int, str]]:
    """(file, line number, normalised text) for every non-empty line of the report's sources."""
    from tzbank.reporting.report_source import source_files

    out = []
    for f in source_files(src_dir, report):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            n = _norm(line)
            if len(n) > 3:
                out.append((f.name, i, n))
    return out


def match(paragraph: str, lines: list[tuple[str, int, str]]) -> tuple[str, int, float]:
    """Best source line for a paragraph's 'before' text: an exact substring first, else the highest difflib ratio of
    the paragraph against each line (surrounding text, not position)."""
    p = _norm(paragraph)
    if not p:
        return "", 0, 0.0
    probe = p[:120]
    for f, i, n in lines:
        if probe in n:
            return f, i, 1.0
    best = max(((difflib.SequenceMatcher(None, p, n, autojunk=False).ratio(), f, i) for f, i, n in lines),
               default=(0.0, "", 0))
    return best[1], best[2], round(best[0], 3)


def _comments(z: zipfile.ZipFile) -> dict[str, str]:
    if "word/comments.xml" not in z.namelist():
        return {}
    x = etree.fromstring(z.read("word/comments.xml"))
    return {c.get(_q("id")): " ".join(_text(p, False) for p in c.iter(_q("p"))).strip()
            for c in x.iter(_q("comment"))}


def extract(docx: Path, report: str, src_dir: Path | None = None) -> pd.DataFrame:
    """Every tracked change and comment of a reviewed report, matched to its source line."""
    with zipfile.ZipFile(docx) as z:
        parts = [("document", etree.fromstring(z.read("word/document.xml")))] + [
            (re.sub(r"^word/|\.xml$", "", n), etree.fromstring(z.read(n))) for n in sorted(z.namelist())
            if re.fullmatch(r"word/(header|footer)\d*\.xml|word/(footnotes|endnotes)\.xml", n)]
        comments = _comments(z)
    lines = source_lines(src_dir, report) if src_dir else []
    rows: list[dict] = []
    page, k = 1, 0
    last_text = ""  # the last non-empty 'after' paragraph not moved here: the anchor of a moved-to paragraph
    open_comments: dict[str, list[str]] = {}
    for part, doc in parts:  # body, then headers, footers, footnotes, endnotes (prompt 25)
        for para in doc.iter(_q("p")):
            if any(a.tag == _q("p") for a in para.iterancestors()):
                continue  # nested paragraph of a text box: visited as its own paragraph below
            for sub in [para, *para.iter(_q("txbxContent"))]:
                ps = [sub] if sub is para else list(sub.iter(_q("p")))
                for p in ps:
                    k += 1
                    before, after = _text(p, True), _text(p, False)
                    in_table = any(a.tag == _q("tc") for a in p.iterancestors())
                    edited = any(p.find(f".//w:{t}", NS) is not None for t in (
                        "ins", "del", "moveFrom", "moveTo", "rPrChange", "pPrChange", "commentRangeEnd", "commentReference"))
                    f, ln, sc = match(before or after, lines) if lines and edited else ("", 0, 0.0)  # only edited paragraphs
                    base = {"report": report, "page": page, "part": part, "paragraph": k, "paragraph_old": before, "paragraph_new": after,
                            "in_table": in_table, "source_file": f, "source_line": ln, "match_score": sc}
                    els = list(p.iter(_q("ins"), _q("del"), _q("moveFrom"), _q("moveTo")))
                    i = 0
                    while i < len(els):
                        e = els[i]
                        if e.tag == _q("del") and i + 1 < len(els) and els[i + 1].tag == _q("ins") \
                                and e.getnext() is els[i + 1]:
                            rows.append(base | {"kind": "replace", "old_text": _text(e, True),
                                                "new_text": _text(els[i + 1], False)})
                            i += 2
                            continue
                        if e.tag in (_q("moveFrom"), _q("moveTo")):  # a tracked move (drag and drop)
                            frm = e.tag == _q("moveFrom")
                            rows.append(base | {"kind": "move from" if frm else "move to",
                                                "old_text": _text(e, True) if frm else "",
                                                "new_text": "" if frm else _text(e, False),
                                                "anchor_text": "" if frm else last_text[-200:]})
                        elif e.tag == _q("del"):
                            rows.append(base | {"kind": "delete", "old_text": _text(e, True), "new_text": ""})
                        else:
                            rows.append(base | {"kind": "insert", "old_text": "", "new_text": _text(e, False)})
                        i += 1
                    if p.find(".//w:rPrChange", NS) is not None or p.find(".//w:pPrChange", NS) is not None:
                        rows.append(base | {"kind": "format", "old_text": "", "new_text": ""})
                    for cs in p.iter(_q("commentRangeStart")):
                        open_comments[cs.get(_q("id"))] = []
                    for cid in list(open_comments):
                        open_comments[cid].append(after)
                    for ce in p.iter(_q("commentRangeEnd")):
                        cid = ce.get(_q("id"))
                        anchor = " ".join(open_comments.pop(cid, [after]))
                        rows.append(base | {"kind": "comment", "old_text": "", "new_text": "",
                                            "comment_text": comments.get(cid, ""), "anchor_text": anchor[:400]})
                    if after.strip() and p.find(".//w:moveTo", NS) is None:
                        last_text = after.strip()
                    for cr in p.iter(_q("commentReference")):  # a comment without a range: anchor = the paragraph
                        cid = cr.get(_q("id"))
                        if not any(r.get("kind") == "comment" and r.get("comment_text") == comments.get(cid)
                                   and r.get("paragraph") == k for r in rows):
                            rows.append(base | {"kind": "comment", "old_text": "", "new_text": "",
                                                "comment_text": comments.get(cid, ""), "anchor_text": after[:400]})
            page += len(para.findall(".//w:lastRenderedPageBreak", NS)) + len(
                [b for b in para.iter(_q("br")) if b.get(_q("type")) == "page"])
    d = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("id", "status")])
    d.insert(0, "id", [f"{report}-{i + 1:03d}" for i in range(len(d))])
    d["status"] = ["formatting (ignored)" if r.kind == "format" else "comment" if r.kind == "comment"
                   else "move" if r.kind.startswith("move")
                   else "generated content" if r.part != "document"  # running header / footer: the build makes it
                   else "generated content" if r.in_table or (r.source_file and r.match_score < 0.6)
                   else "to apply" for r in d.itertuples()]
    return d[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    from tzbank.config import find_root

    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.review_edits", description=__doc__.split("\n")[0])
    ap.add_argument("--report", default="all", choices=["main", "nmb", "crdb", "foreign", "all"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    root = find_root()
    rev = root / "reports" / "review"
    reps = ["main", "nmb", "crdb", "foreign"] if a.report == "all" else [a.report]
    rc = 0
    for r in reps:
        f = rev / f"{r}_JJ.docx"
        if not f.exists():
            print(f"{r}: {f.relative_to(root)} not found")
            rc = 1
            continue
        d = extract(f, r, root / "reports" / "source")
        out = rev / f"edits_{r}.csv"
        d.to_csv(out, index=False)
        print(f"{r}: {len(d)} edits ({d['kind'].value_counts().to_dict()}) -> {out.relative_to(root)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
