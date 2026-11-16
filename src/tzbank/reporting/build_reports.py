"""Report build stage (prompt 16; D65, D77): owner-authored Markdown + pipeline outputs -> designed .docx and PDF.

``python -m tzbank.reporting.build_reports [--report main|nmb|crdb|foreign|all] [--source FILE] [--no-pdf]``

- Reads ``reports/source/<report>.md`` (grammar: ``reports/source/README.md``; ``tzbank.reporting.report_source``).
- Resolves every exhibit id (figure stem such as ``ch01_cr2`` or export key such as ``main/02``) through
  ``outputs/exhibits/index.csv`` and re-exports the chart in the house theme (``tzbank.reporting.chart_theme``) to
  ``outputs/exhibits/report/<stem>_<preset>.png``.
- Builds annex and pipeline tables from ``outputs/tables`` (``tzbank.reporting.annexes``).
- Lays the report out in the house style (``tzbank.reporting.house_docx``) -> ``outputs/reports/<report>.docx``.
- Exports the PDF (Word on Windows, else LibreOffice headless), updating the table of contents; checks that every
  font in the PDF is embedded; renders the first pages to ``outputs/reports/preview/``.
- Runs the figure checker on the built .docx -> ``outputs/reports/<report>_figures.csv``.

The stage lays out text; it never writes or edits report prose (D65). Any unknown marker, unresolved exhibit id or
annex that cannot be built fails the build with a message naming the file and line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tzbank.config import find_root
from tzbank.reporting import annexes as ax
from tzbank.reporting import chart_theme, house_art
from tzbank.reporting import equations as EQ
from tzbank.reporting import house_docx as H
from tzbank.reporting import layout_fill as LF
from tzbank.reporting.report_source import INVESTMENT, RATINGS, Node, Source, SourceError, inline, parse_parts, source_files, walk

PRESET = {"full": "full", "half": "half", "side": "side", "landscape": "landscape",
          "page1": "full"}  # page1: investment-report page 1, full width under the opening text (prompt 24b)
PAGE1_ASPECT = 0.40  # the page-1 price chart in a landscape aspect (owner, prompt 24b)
JUSTIFY_MIN = 5103  # twips (9 cm): narrower text keeps left alignment (justification opens wide gaps; prompt 21)
LINKS = ("report_doi", "zenodo_doi", "dashboard_url", "github_url", "code_doi")
TITLES = {"inbrief": "In brief", "finding": "Key finding", "limitation": "Limitation", "thesis": "Investment thesis",
          "method": "Method note", "case": "Case"}


FLOW = ("nmb", "crdb", "foreign")


def _mark(n: Node) -> str:
    """A short plain-text snippet of a node, to find it in the PDF (page-1 spill check)."""
    import re as _re

    if n.kind == "exhibit":
        t = "Source: " + n.attrs.get("source", "")
    elif n.kind == "block":
        return _mark(n.children[-1]) if n.children else ""
    elif n.kind == "table":
        t = " ".join(n.items[-1])
    elif n.kind == "list":
        t = n.items[-1]
    else:
        t = n.text
    t = _re.sub(r"\*|\{\{[^}]*\}\}", "", t).strip()
    t = _re.split(r"\[\[[^\]]*\]\]", t)[-1]  # text after the last token: a token prints resolved (prompt 23)
    words = [w for w in t.split() if _re.search(r"\w", w)]
    return " ".join(words[-4:])  # words only: punctuation next to a token may print differently (prompt 24)


def _mark_first(n: Node) -> str:
    """The first words of a node (to find where an intended page break lands)."""
    t = re.sub(r"\*|\{\{[^}]*\}\}", "", n.text or (" ".join(n.items[0]) if n.items else "")).split()
    return " ".join(t[:4])


def _box_title(n: Node) -> str:
    """The bold lead of a house box ('**International experience: ...**'), for the side-block list."""
    first = n.children[0].text if n.children and n.children[0].kind == "para" else ""
    m = re.match(r"\*\*(.+?)\*\*", first)
    return m.group(1).rstrip(".") if m else n.label


def _all_bold(text: str) -> bool:
    t = text.strip()
    return t.startswith("**") and t.endswith("**") and t.count("**") == 2 and len(t) > 4


class BuildError(RuntimeError):
    """A report that cannot be built (unresolved exhibit, failed annex, missing source)."""


def cfg(root: Path) -> dict:
    return ax.report_config(root)


# --- exhibit resolution ----------------------------------------------------------------------------------------------
def exhibit_index(root: Path) -> pd.DataFrame:
    f = root / "outputs" / "exhibits" / "index.csv"
    if not f.exists():
        raise BuildError("outputs/exhibits/index.csv not found: run `python -m tzbank run support` first")
    i = pd.read_csv(f, dtype=str).fillna("")
    for c in ("export_key", "figure_stem"):
        if c not in i:
            raise BuildError(f"outputs/exhibits/index.csv has no {c} column: re-run `python -m tzbank run support`")
    return i


def resolve(root: Path, ex_id: str, idx: pd.DataFrame) -> dict:
    """Exhibit id -> index row (export key such as main/02, or figure stem such as ch01_cr2)."""
    hit = idx[idx["export_key"] == ex_id]
    if hit.empty:
        hit = idx[(idx["figure_stem"] == ex_id) & (idx["report"] == "figure")]
    if hit.empty:
        hit = idx[idx["figure_stem"] == ex_id]
    if hit.empty:
        raise BuildError(f"exhibit id {ex_id!r} is neither an export key nor a figure stem in outputs/exhibits/index.csv")
    r = hit.iloc[0].to_dict()
    r["key"] = r["figure_stem"]
    return r


def themed(root: Path, key: str, preset: str, panel: str | None = None, max_aspect: float | None = None) -> Path:
    c = cfg(root)
    out = root / c.get("exhibits_dir", "outputs/exhibits/report")
    try:
        return chart_theme.render(key, preset, out / "_src", out, panel, max_aspect)
    except (FileNotFoundError, ValueError) as e:  # ValueError: the text-survival / minimum-size check (prompt 22)
        raise BuildError(str(e)) from None


# --- the builder -----------------------------------------------------------------------------------------------------
@dataclass
class Built:
    report: str
    docx: Path
    pdf: Path | None = None
    exhibits: list = field(default_factory=list)
    annexes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    previews: list = field(default_factory=list)
    fonts: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    side_blocks: list = field(default_factory=list)  # each side-by-side block and what fills its text column
    page_breaks: list = field(default_factory=list)  # first words after each {{pagebreak}} (layout check)
    remedies: dict = field(default_factory=dict)  # half-empty page remedies applied (prompt 26)
    links: int = 0  # URLs turned into hyperlinks (D99)
    plain_blocks: list = field(default_factory=list)  # :::plain blocks: section, first and last words (D100)


class Builder:
    def __init__(self, root: Path, src: Source, assets: dict, compact: int = 0, remedies: dict | None = None):
        self.root, self.src, self.meta = root, src, src.meta
        # half-empty page remedies per exhibit id (prompt 26): "defer", "side" or ("shrink", max aspect)
        self.remedies = remedies or {}
        self.compact = compact  # page-1 fitting level (0 none, 1 smaller sidebar chart, 2 also tighter body type)
        self.rep = src.meta["report"]
        self.c = cfg(root)
        self.assets = assets
        self.idx = exhibit_index(root)
        self.doc = H.new_document()
        H.hyphenation(self.doc)
        self.numbers: dict[str, int] = {}
        self.annex_letter: dict[int, str] = {}
        self.chapter_no = 0
        self.after_part = False
        self.page_has_content = False
        self.built = Built(self.rep, Path())
        self.flow = self.rep in FLOW  # investment reports: ## sections flow; breaks only before annexes and disclosures
        self.annex_heading: Node | None = None  # the "## Annex X. title" section being laid out
        self.annexes_started = False
        self.break_next = False
        self.section = ""  # the current # or ## heading, lower case (":::plain" sizes differ in the disclosures)
        self.page_one_marks: list[str] = []
        self.has_disclosures = any(n.kind in ("part", "chapter") and n.text.strip().lower() == "disclosures"
                                   for n in src.nodes)
        # an investment report uses its own bank's panel of a per-bank figure (e.g. ch13_price_lines in nmb.md)
        self.panel = self.meta.get("ticker") if self.rep in INVESTMENT else None
        self._number()
        self.preflight()

    # numbering in reading order (sidebar exhibits are not numbered)
    def _number(self) -> None:
        n = 0
        for node in self.src.nodes:
            side = node.kind == "block" and node.label == "sidebar"
            for x in [node, *walk(node.children)]:
                if x.kind == "exhibit" and not side and x.attrs.get("layout") != "page1" \
                        and x.attrs["id"] not in self.numbers:  # page-1 charts are not numbered, as in the sidebar
                    n += 1
                    self.numbers[x.attrs["id"]] = n
        letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for x in walk(self.src.nodes):
            if x.kind == "annex" and not x.attrs.get("inline"):  # body tables take no annex letter
                self.annex_letter[id(x)] = next(letters)
        for x in walk(self.src.nodes):
            for text in [x.text, *x.items] if x.kind != "table" else [c for r in x.items for c in r]:
                for _, f in inline(text, x.where) if text else []:
                    if "ref" in f and f["ref"] not in self.numbers:
                        raise BuildError(f"{x.where}: {{{{ref id=\"{f['ref']}\"}}}} points to an exhibit "
                                         "that is not shown (numbered) in this report")

    def preflight(self) -> None:
        """Check every exhibit (index entry and chart snapshot), pipeline table and annex before laying anything out;
        fail once with the full list of what is missing (no silent gaps)."""
        problems: list[str] = []
        for x in walk(self.src.nodes):  # formulas must be $$...$$ equations, never typed with _ or ^ (prompt 24b)
            for text in ([x.text] if x.kind in ("para", "h3") else x.items if x.kind == "list" else []):
                typed = EQ.typed_formulas(text or "")
                if typed:
                    problems.append(f"{x.where}: typed formula {', '.join(typed)}: write it as a $$...$$ equation")
        src_dir = self.root / self.c.get("exhibits_dir", "outputs/exhibits/report") / "_src"
        for x in walk(self.src.nodes):
            if x.kind == "exhibit":
                try:
                    r = resolve(self.root, x.attrs["id"], self.idx)
                except BuildError as e:
                    problems.append(f"{x.where}: {e}")
                    continue
                if not (src_dir / f"{r['key']}.pickle").exists():
                    problems.append(f"{x.where}: exhibit {x.attrs['id']}: no chart snapshot {r['key']}.pickle "
                                    "(re-run the stage that draws it)")
            elif x.kind == "ptable":
                a = x.attrs
                try:
                    ax.pipeline_table(self.root, a["src"], a.get("filter"), a.get("cols"), a.get("source"),
                                      a.get("group"))
                except (ValueError, FileNotFoundError, KeyError) as e:
                    problems.append(f"{x.where}: table {a['src']}: {e}")
            elif x.kind == "annex":
                a = x.attrs
                try:
                    ax.annex(self.root, a["type"], a.get("bank"), report_text=self.text())
                except (ValueError, FileNotFoundError, KeyError) as e:
                    problems.append(f"{x.where}: annex {a['type']}{' ' + a['bank'] if a.get('bank') else ''}: {e}")
        if problems:
            sep = "\n  - "
            raise BuildError(f"{len(problems)} missing or failing input(s):{sep}" + sep.join(problems))

    def exhibit_id(self, ex_id: str, sidebar: bool = False) -> str:
        """Report-level substitution from config/reports.yaml (sidebar_exhibits), e.g. ch13_price_lines -> nmb/15."""
        if sidebar:
            return ((self.c.get("sidebar_exhibits") or {}).get(self.rep) or {}).get(ex_id, ex_id)
        return ex_id

    def layout_of(self, n: Node) -> str:
        """Exhibit layout, with the per-report width overrides of config/reports.yaml (exhibit_layout)."""
        over = (self.c.get("exhibit_layout") or {}).get(self.rep) or {}
        if self.remedies.get(n.attrs["id"]) == "side" and n.attrs.get("layout") != "landscape":
            return "half"  # remedy 2 (prompt 26): a side block with the following text
        return over.get(n.attrs["id"], n.attrs.get("layout", "full"))

    # inline text
    def tokens(self, text: str) -> str:
        """[[zenodo_doi]], [[dashboard_url]], [[github_url]] from config/reports.yaml ('[to be added]' until set);
        [[pack]] = the findings pack the build reads (prompt 21)."""
        if "[[" not in text:
            return text
        from tzbank.reporting.findings import latest_pack

        for k in LINKS:  # [[k:short]] is the address without https:// and a closing slash (footer display, D100)
            v = str(self.c.get(k) or "")
            text = text.replace(f"[[{k}:short]]", re.sub(r"^https?://", "", v).rstrip("/") or "[to be added]")
            text = text.replace(f"[[{k}]]", v or "[to be added]")
        for k in ("code_title", "data_title"):  # the Zenodo record titles (D100)
            text = text.replace(f"[[{k}]]", str(self.c.get(k) or "[to be added]"))
        pack = latest_pack(self.root)
        if "[[register]]" in text:  # the inputs-register version the pack was built on (prompt 23: stated once)
            reg = pd.read_csv(pack, usecols=["register_version"], nrows=50)["register_version"].dropna()
            text = text.replace("[[register]]", str(reg.iloc[0]) if len(reg) else "[to be added]")
        ver = pack.stem.replace("findings_pack_", "")
        # "Findings pack ([[pack]])" prints as "Findings pack v1.14", not "findings pack (findings pack v1.14)"
        text = re.sub(r"\b([Ff]indings pack) \(\[\[pack\]\]\)", rf"\1 {ver}", text)
        return text.replace("[[pack]]", f"findings pack {ver}")

    def runs(self, p, text: str, size: float = 10, color: str = H.C["text"], bold: bool = False,
             italic: bool = False) -> None:
        text = self.tokens(text)
        for t, f in inline(text):
            if f.get("link"):  # [display text](url): the text prints, the address is linked (D100)
                H.link_run(p, t, f["link"], size, H.C["green"], bold or f.get("bold", False),
                           italic or f.get("italic", False))
            elif "ref" in f:
                H.run(p, f"Exhibit {self.numbers[f['ref']]}", size, color, bold or f.get("bold", False),
                      italic or f.get("italic", False))
            else:
                H.run(p, t, size, color, bold or f.get("bold", False), italic or f.get("italic", False))

    def table_runs(self, p, text, size, color, bold) -> None:
        self.runs(p, text, size, color, bold)

    # elements
    def paragraph(self, cont, n: Node, size: float = 10, color: str = H.C["text"], after: float = 6,
                  justify: bool = False) -> None:
        p = H.para(cont, 0, after, align="justify" if justify else None)
        p.paragraph_format.keep_together = True  # no paragraph split across pages (prompt 25)
        self.runs(p, n.text, size, color)

    def bullets(self, cont, n: Node, size: float = 10) -> None:
        num = H.list_num(self.doc, n.ordered)
        for it in n.items:
            p = H.list_para(cont, num, size=size)
            p.paragraph_format.alignment = 3  # justified (prompt 25)
            p.paragraph_format.keep_together = True
            self.runs(p, it, size)

    def exhibit(self, cont, n: Node, width: int, preset: str, numbered: bool = True,
                image_width: int | None = None, max_aspect: float | None = None) -> int:
        """Exhibit panel: 'Exhibit n' label, takeaway title, chart, italic source. Returns its height (twips)."""
        a = n.attrs
        r = resolve(self.root, a["id"], self.idx)
        rem = self.remedies.get(a["id"])
        if max_aspect is None and isinstance(rem, tuple) and rem[0] == "shrink":
            max_aspect = rem[1]  # remedy 3 (prompt 26): a shorter chart; the text check keeps text at 7 pt or more
        png = themed(self.root, r["key"], preset, self.panel, max_aspect)
        self.built.exhibits.append({"id": a["id"], "key": r["key"], "preset": preset, "png": png,
                                    "number": self.numbers.get(a["id"]) if numbered else None})
        cell = H.box(cont, width, H.C["white"], border=H.C["rule"])
        H.cell_style(cell, H.C["white"], (140, 120, 180, 180),
                     {s: ("single", 4, H.C["rule"]) for s in ("top", "left", "bottom", "right")})
        H.row_props(cell._parent.rows[0])
        if numbered:
            p = H.para(cell, 0, 1, keep_next=True)
            H.run(p, f"Exhibit {self.numbers[a['id']]}", 8, H.C["green"], bold=True)
        p = H.para(cell, 0, 4, keep_next=True)
        self.runs(p, a["title"], 9.5, H.C["text"], bold=True)
        p = H.para(cell, 0, 3, align="center", line=1.0, keep_next=True)
        _, h = H.image(p, png, min(width - 360, image_width or width))
        src = self.tokens(a["source"].strip())
        p = H.para(cell, 0, 0)
        self.runs(p, src if src.lower().startswith("source") else f"Source: {src}", 7, H.C["grey"], italic=True)
        H.finish_cell(cell)
        return h + 1300

    def house_table(self, cont, header, rows, width, align=None, size=8, no_header=False, bold_rows=None,
                    widths=None) -> None:
        widths = widths or H.col_widths(header, rows, width)
        H.house_table(cont, header, rows, widths, self.table_runs, align, size, no_header, bold_rows)

    def source_line(self, cont, text: str) -> None:
        p = H.para(cont, 3, 8)
        H.run(p, self.tokens(text), 7, H.C["grey"], italic=True)  # [[pack]] in table sources too (prompt 23)

    def md_table(self, cont, n: Node, width: int, size: float = 8) -> None:
        header, rows = n.items[0], n.items[1:]
        no_header = all(not h.strip() for h in header)
        self.house_table(cont, header, rows, width, n.align, size, no_header)
        H.spacer(cont, 6)

    def pipeline_table(self, cont, n: Node, width: int) -> None:
        a = n.attrs
        try:
            g = ax.pipeline_table(self.root, a["src"], a.get("filter"), a.get("cols"), a.get("source"), a.get("group"))
        except (ValueError, FileNotFoundError) as e:
            raise BuildError(f"{n.where}: {e}") from None
        self.house_table(cont, g.header, g.rows, width, bold_rows=g.bold_rows)
        self.source_line(cont, g.source)

    def annex(self, cont, n: Node, width: int) -> None:
        a = n.attrs
        try:
            g = ax.annex(self.root, a["type"], a.get("bank"), report_text=self.text())
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise BuildError(f"{n.where}: annex {a['type']} cannot be built: {e}") from None
        if a.get("inline"):  # an annex-built table placed in the body: a plain title, no "Annex X" (prompt 24b)
            letter = ""
            p = H.para(cont, 6, 3, keep_next=True)
            H.run(p, g.title, 9.5, H.C["bottle"], bold=True)
        elif self.annex_heading is not None:  # the source's "## Annex X. title" is the banner: add none of our own
            letter = self.annex_heading.label.replace("Annex", "").strip() or self.annex_heading.label
            p = H.para(cont, 6, 3, keep_next=True)
            H.run(p, g.title, 9.5, H.C["bottle"], bold=True)
        else:  # no source heading: banner with an automatic letter (fallback)
            letter = self.annex_letter[id(n)]
            H.spacer(cont, 10)
            H.banner(cont, f"Annex {letter}", g.title, width, toc=(f"Annex {letter}. {g.title}", 2))
            H.spacer(cont, 6, keep_next=True)
        size = 8 if len(g.header) <= 7 else 7
        self.house_table(cont, g.header, g.rows, width, size=size, bold_rows=g.bold_rows)
        self.source_line(cont, g.source)
        self.built.annexes.append({"letter": letter, "type": a["type"], "bank": a.get("bank"), "rows": len(g.rows),
                                   "evidence": g.evidence})

    def text(self) -> str:
        """All prose of this report (for the glossary filter)."""
        if not hasattr(self, "_text"):
            parts = []
            for x in walk(self.src.nodes):
                parts += [x.text, *[i if isinstance(i, str) else " ".join(i) for i in x.items]]
                parts += [x.attrs.get("title", "")] if x.kind == "exhibit" else []
            self._text = "\n".join(p for p in parts if p)
        return self._text

    def plain(self, cont, n: Node) -> None:
        """:::plain (D100): contact, materials and citation blocks, and the disclosures. No box; left-aligned; a
        paragraph that is entirely bold and does not end with ':' is a sub-heading; every source line of a paragraph
        is its own printed line; lists are left-aligned bullets; the whole block is kept on one page. Size and
        spacing come from ``plain_blocks`` in config/reports.yaml (per report and per section)."""
        pb = self.c.get("plain_blocks") or {}
        sec = "disclosures" if self.section == "disclosures" else "body"
        over = (pb.get(self.rep) or {}).get(sec) or {}
        size = float(over.get("size", pb.get("size", 10)))
        gap = float(over.get("after", pb.get("after", 3)))
        kids = list(n.children)
        start = len(self.doc.paragraphs)
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            if k.kind == "para" and _all_bold(k.text) and not k.text.strip().strip("*").rstrip().endswith(":"):
                p = H.para(cont, 0 if i == 0 else gap + 2, 2, keep_next=True)
                self.runs(p, k.text, size + 0.5, H.C["bottle"], bold=True)
            elif k.kind == "para":
                p = H.para(cont, 0, gap, keep_next=not last)
                p.paragraph_format.keep_together = True
                H._no_hyphens(p._p)  # left-aligned text: no automatic hyphenation
                for j, line in enumerate(k.raw or [k.text]):
                    if j:
                        p.add_run().add_break()
                    self.runs(p, line, size)
            elif k.kind == "list":
                num = H.list_num(self.doc, k.ordered)
                for j, it in enumerate(k.items):
                    p = H.list_para(cont, num, after=1 if j < len(k.items) - 1 else gap, size=size)
                    p.paragraph_format.keep_together = True
                    p.paragraph_format.keep_with_next = not (last and j == len(k.items) - 1)
                    H._no_hyphens(p._p)
                    self.runs(p, it, size)
            else:
                raise BuildError(f"{k.where}: only paragraphs and lists are allowed in :::plain")
        # marks from the printed text (links show their display text), leaving out addresses, which can wrap
        runs_: list[list[str]] = [[]]  # runs of consecutive plain words (an address or e-mail ends a run)
        for w in (w for p in self.doc.paragraphs[start:] for w in p.text.split() if re.search(r"\w", w)):
            if "/" in w or "@" in w:
                runs_.append([])
            else:
                runs_[-1].append(w)
        runs_ = [r for r in runs_ if r] or [[]]
        self.built.plain_blocks.append({"section": self.section, "first": " ".join(runs_[0][:5]),
                                        "last": " ".join(runs_[-1][-4:]), "size": size,
                                        "paragraphs": len(self.doc.paragraphs) - start})
        H.spacer(cont, gap + 3)

    def block(self, cont, n: Node, width: int) -> None:
        kind = n.label
        if kind == "plain":
            return self.plain(cont, n)
        if kind == "sidebar":
            raise BuildError(f"{n.where}: :::sidebar belongs before the first ## heading (page 1)")
        fill, edge, border, ink = {
            "inbrief": (H.C["soft"], None, None, H.C["green"]), "thesis": (H.C["soft"], None, None, H.C["green"]),
            "finding": (H.C["pale_gold"], H.C["gold"], None, H.C["gold_ink"]),
            "limitation": (H.C["pale_gold"], H.C["gold"], None, H.C["gold_ink"]),
            "method": (H.C["white"], H.C["teal"], H.C["rule"], H.C["green"]),
            "case": (H.C["pale_gold"], H.C["teal"], None, H.C["gold_ink"]),
        }[kind]
        cell = H.box(cont, width, fill, edge, border)
        H.row_props(cell._parent.rows[0])  # a box stays on one page (prompt 25)
        kids = list(n.children)
        title = TITLES[kind]
        first = kids[0] if kids and kids[0].kind == "para" else None
        lead = inline(first.text) if first else []
        if lead and lead[0][1].get("bold"):  # a bold lead ("**Key finding.** ...") becomes the box title
            title = lead[0][0].strip().rstrip(".:").strip()
            rest = first.text.split("**", 2)[2].strip() if first.text.startswith("**") else ""
            kids = ([Node("para", first.line, text=rest)] if rest else []) + kids[1:]
        p = H.para(cell, 0, 3, keep_next=True)
        H.run(p, title, 8.5 if kind in ("finding", "limitation", "case") else 9, ink, bold=True)
        size = 9 if kind == "method" else 9.5
        inner = width - 440
        for k in kids:
            self.render(cell, k, inner, size=size, in_box=True)
        H.finish_cell(cell)
        H.spacer(cont, 8)

    def heading3(self, cont, n: Node, size: float = 10.5) -> None:
        p = H.para(cont, 6, 4, keep_next=True)
        self.runs(p, n.text, size, H.C["bottle"], bold=True)

    # dispatch
    def render(self, cont, n: Node, width: int, size: float = 10, in_box: bool = False) -> None:
        if n.kind == "para" and EQ.DISPLAY.match(n.text.strip()):  # $$ LaTeX $$: a Word equation (prompt 24b)
            p = H.para(cont, 4, 6, align="center")
            try:
                EQ.add_display(p, EQ.DISPLAY.match(n.text.strip()).group(1))
            except (ValueError, FileNotFoundError) as e:
                raise BuildError(f"{n.where}: equation cannot be built: {e}") from None
        elif n.kind == "para":  # all body text justified, in columns and boxes too (owner, prompt 25)
            self.paragraph(cont, n, size, after=4 if in_box else 6, justify=True)
        elif n.kind == "pagebreak":  # {{pagebreak}} (prompt 24b)
            self.new_page()
        elif n.kind == "list":
            self.bullets(cont, n, size)
        elif n.kind == "h3":
            self.heading3(cont, n)
        elif n.kind == "table":
            self.md_table(cont, n, width)
        elif n.kind == "ptable":
            self.pipeline_table(cont, n, width)
        elif n.kind == "annex":
            self.annex(cont, n, width)
        elif n.kind == "exhibit" and self.layout_of(n) == "landscape":
            self.landscape(n)
        elif n.kind == "exhibit":
            preset = "full" if width > 6000 else "half" if width > 4000 else "side"
            if self.layout_of(n) == "full" and width > 6000:
                preset = "full"
            self.exhibit(cont, n, width, preset)
            H.spacer(cont, 6)
        elif n.kind == "block":
            if in_box:
                raise BuildError(f"{n.where}: blocks cannot be nested")
            self.block(cont, n, width)
        else:
            raise BuildError(f"{n.where}: {n.kind} is not allowed here")

    def landscape(self, n: Node) -> None:
        """A full-page landscape section holding one exhibit, then back to portrait (prompt 21, E1)."""
        from docx.enum.section import WD_ORIENT

        sec = H.section(self.doc)
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = H.Twips(H.PAGE_H), H.Twips(H.PAGE_W)
        width = H.PAGE_H - H.MARGIN["left"] - H.MARGIN["right"]
        H.header_footer(sec, *self.hf, None, width)  # date and page number at the landscape margin (prompt 23)
        self.exhibit(self.doc, n, width, "landscape")
        back = H.section(self.doc)  # portrait again (house margins), with the portrait header and page art
        H.header_footer(back, *self.hf, self.assets["page"])
        self.page_has_content = False  # the section break opened a fresh page: no page break before what follows
        # (otherwise an empty page followed the landscape exhibit; owner comment main-012, prompt 24b)

    def columns(self, cont, n: Node, follow: list[Node], layout: str | None = None) -> None:
        """Side (two-thirds / one-third) or half (half / half) exhibit beside the text that follows it."""
        layout = layout or self.layout_of(n)
        txt = H.SIDE_TXT if layout == "side" else (H.W_FULL - H.GAP) // 2
        ex = H.W_FULL - txt - H.GAP
        t = H.table(cont, [txt, H.GAP, ex])
        H.row_props(t.rows[0])  # the block stays on one page (prompt 25)
        left, right = t.cell(0, 0), t.cell(0, 2)
        for k in follow:
            self.render(left, k, txt)
        self.exhibit(right, n, ex, PRESET[layout])
        for c in (left, t.cell(0, 1), right):
            H.finish_cell(c)
        H.spacer(cont, 8)

    def _fill(self, nodes: list[Node], i: int, layout: str) -> tuple[dict, float, str]:
        """Side-block fill rule (prompt 25, D96): the run of following elements (an optional ### heading, then
        paragraphs, lists, owner boxes, small tables) set beside the exhibit so that the two columns end within about
        1.5 cm (``layout_fill.choose``). A one-third ('side') exhibit may be drawn at half width when that balances
        better (wider, never stretched); a half exhibit is never narrowed (exhibit_layout widened some for legibility).
        The opening paragraph may print full width above the block when the rest balances better
        (``layout_fill.choose_lead``). Returns (lead end, end index, text height, exhibit height, layout), heights
        in twips; nodes[i+1:lead end] print above the block, nodes[lead end:end] beside the exhibit."""
        a = nodes[i].attrs
        r = resolve(self.root, a["id"], self.idx)
        best = None
        widened = a["id"] in ((self.c.get("exhibit_layout") or {}).get(self.rep) or {})  # set wide for legibility
        options = [layout] + (["half"] if layout == "side" else ["side"] if layout == "half" and not widened else [])
        for lay in options:
            txt = H.SIDE_TXT if lay == "side" else (H.W_FULL - H.GAP) // 2
            ex = H.W_FULL - txt - H.GAP
            mark = len(chart_theme.TEXT_LOG)
            try:
                png = themed(self.root, r["key"], PRESET[lay], self.panel)
            except BuildError:
                if lay == layout:
                    raise
                del chart_theme.TEXT_LOG[mark:]  # a narrower trial that fails the text check is simply not used
                continue
            target = LF.exhibit_h(a["title"], self.tokens(a["source"]), png, ex)
            pl = LF.plan(nodes, i, target, txt, H.W_FULL)
            score = abs(target - pl["h"]) if pl["column"] else float("inf")
            if best is None or score < best[0] - 200:  # the exhibit's own width wins a near tie
                best = (score, pl, target, lay)
        return best[1], best[2], best[3]

    def _take_following(self, nodes: list[Node], i: int, layout: str) -> int:
        """How many following paragraphs/lists sit beside a side or half exhibit (about the exhibit's height)."""
        ex_w = H.W_FULL - (H.SIDE_TXT if layout == "side" else (H.W_FULL - H.GAP) // 2) - H.GAP - 360
        r = resolve(self.root, nodes[i].attrs["id"], self.idx)
        png = themed(self.root, r["key"], PRESET[layout], self.panel)
        from PIL import Image  # noqa: PLC0415

        with Image.open(png) as im:
            target = ex_w * im.size[1] / im.size[0] + 1300
        txt_cm = (H.SIDE_TXT if layout == "side" else (H.W_FULL - H.GAP) // 2) / 567
        used, j = 0.0, i + 1
        def fits(k: int) -> bool:
            return nodes[k].kind in ("para", "list") or (nodes[k].kind == "h3" and k == i + 1)

        while j < len(nodes) and fits(j) and (used < target or j <= i + 2 and nodes[i + 1].kind == "h3"
                                              or j == i + 1):
            chars = len(nodes[j].text) if nodes[j].kind == "para" else sum(len(x) + 30 for x in nodes[j].items)
            lines = chars / (txt_cm * 5.2) + 1
            used += lines * 276 + 120
            j += 1
        return j

    # pages
    def new_page(self) -> None:
        if self.page_has_content:
            H.page_break(self.doc)
        self.page_has_content = False

    def body(self, nodes: list[Node]) -> None:
        nodes = defer_exhibits(nodes, [k for k, v in self.remedies.items() if v == "defer"])
        i = 0
        toc_pending = self.rep == "main"
        exec_level = 0  # 1: the executive summary is a # part, 2: a ## chapter; the contents follow its section
        while i < len(nodes):
            n = nodes[i]
            ends_exec = n.kind == "part" or (n.kind == "chapter" and exec_level == 2)
            if toc_pending and exec_level and ends_exec:
                self.new_page()
                H.toc(self.doc)
                self.page_has_content = True
                toc_pending = False
            if n.kind in ("part", "chapter") and not exec_level and n.text.lower().startswith("executive summary"):
                exec_level = 1 if n.kind == "part" else 2
            if n.kind in ("part", "chapter"):
                self.section = n.text.strip().lower()
                is_annex = n.label.lower().startswith("annex") or n.text.lower().startswith("annex")
                self.annex_heading = n if n.kind == "chapter" and n.label.lower().startswith("annex") else None
                if self.flow:  # investment reports: forced breaks only before the first annex and the disclosures
                    disclosures = n.text.strip().lower() == "disclosures"
                    if self.break_next or (is_annex and not self.annexes_started) or disclosures:
                        self.new_page()
                    elif self.page_has_content:
                        H.spacer(self.doc, 10)
                    self.break_next = False
                    self.annexes_started = self.annexes_started or is_annex
            if n.kind == "part":
                if not self.flow:
                    self.new_page()
                H.part_label(self.doc, n.label, n.text)
                self.after_part = True
                self.page_has_content = True
            elif n.kind == "chapter":
                self.chapter_no += 1
                if not self.after_part and not self.flow:
                    self.new_page()
                plain = self.rep == "main" or self.annexes_started or n.text.strip().lower() == "disclosures"
                label = n.label or ("" if plain else f"Section {self.chapter_no}")
                H.banner(self.doc, label, n.text, toc=(f"{n.label + '. ' if n.label else ''}{n.text}", 2))
                H.spacer(self.doc, 8, keep_next=True)
                self.after_part = False
                self.page_has_content = True
            elif n.kind == "block" and n.label == "sidebar":
                # page 1 holds the sidebar and everything from the title (# heading) down to the first ## heading
                j = i + 1
                if j < len(nodes) and nodes[j].kind == "part":
                    j += 1
                j = next((k for k in range(j, len(nodes)) if nodes[k].kind in ("part", "chapter")), len(nodes))
                self.new_page()
                self.page_one(n, nodes[i + 1:j])
                self.after_part = False
                self.break_next = True  # the sections start on the next page
                i = j
                continue
            elif n.kind == "exhibit" and self.layout_of(n) in ("side", "half"):
                pl, h_ex, lay = self._fill(nodes, i, self.layout_of(n))
                h_txt = pl["h"]
                body = [x for x in pl["column"] if x.kind != "h3"]
                # nothing suitable, or nothing that comes near the exhibit's height: full width (rule 1, prompt 25)
                if not body or h_txt < h_ex - 1.5 * LF.TOL:
                    self.exhibit(self.doc, n, H.W_FULL, "full")
                    H.spacer(self.doc, 6)
                    self.built.side_blocks.append({"exhibit": n.attrs["id"], "title": n.attrs["title"],
                                                   "layout": "full (fallback)", "column": [], "est_text": h_txt,
                                                   "est_exhibit": h_ex})
                    self.page_has_content = True
                    i += 1
                    continue
                for x in pl["above"]:  # the opening paragraph (or its first sentences) above the block
                    self.render(self.doc, x, H.W_FULL)
                self.columns(self.doc, n, pl["column"], lay)
                for x in pl["below"]:  # the rest of a paragraph split at a sentence continues full width below
                    self.render(self.doc, x, H.W_FULL)
                j = pl["next"]
                self.built.side_blocks.append({"exhibit": n.attrs["id"], "title": n.attrs["title"],
                                               "layout": lay,
                                               "above": [_mark(x) for x in pl["above"]],
                                               "column": [_mark(x) for x in pl["column"]],
                                               "below": [_mark(x) for x in pl["below"]],
                                               "boxes": [_box_title(x) for x in pl["column"] if x.kind == "block"],
                                               "est_text": h_txt, "est_exhibit": h_ex})
                self.page_has_content = True
                i = j
                continue
            elif n.kind == "pagebreak":  # the page is fresh after it (prompt 24b)
                self.new_page()
                if i + 1 < len(nodes):  # the layout check treats an intended break like a section start
                    self.built.page_breaks.append(_mark_first(nodes[i + 1]))
            else:
                self.render(self.doc, n, H.W_FULL)
                # a landscape exhibit ends on a fresh portrait page (no empty page after it; prompt 24b)
                self.page_has_content = not (n.kind == "exhibit" and self.layout_of(n) == "landscape")
            i += 1
        if toc_pending and self.rep == "main":
            self.built.warnings.append("no 'Executive summary' part or chapter: the table of contents was not placed")

    def page_one(self, sidebar: Node, main: list[Node]) -> None:
        """Investment-report page 1: sidebar (rating box, key data, price chart, holders) beside title and thesis."""
        main_w = H.W_FULL - H.SIDEBAR - H.GAP
        t = H.table(self.doc, [H.SIDEBAR, H.GAP, main_w])
        side, gap, col = t.cell(0, 0), t.cell(0, 1), t.cell(0, 2)
        if not self.meta.get("rating"):
            raise BuildError(f"{sidebar.where}: a :::sidebar needs a rating in the front matter (and ticker, sector "
                             "in the nmb and crdb reports)")
        # the rating box already states the rating: drop a bold "RATING: ..." line in the sidebar (prompt 17 A1)
        kids = [k for k in sidebar.children if not (k.kind == "para" and _all_bold(k.text)
                                                     and k.text.strip("* ").lower().startswith(("rating", "verdict")))]
        lead = None  # the sidebar's leading plain paragraph: the lines under the rating (each bulleted, prompt 24b)
        if kids and kids[0].kind == "para" and not _all_bold(kids[0].text):
            lead = kids.pop(0)
        self.rating_box(side, lead)
        for k in kids:
            if k.kind == "para" and _all_bold(k.text):
                p = H.para(side, 4, 2, keep_next=True)
                H.run(p, k.text.strip("*"), 9, H.C["green"], bold=True)
            elif k.kind == "exhibit":
                a = k.attrs
                ex_id = self.exhibit_id(a["id"], sidebar=True)
                r = resolve(self.root, ex_id, self.idx)
                aspect = {0: None, 1: 0.62, 2: 0.5}[min(self.compact, 2)]
                png = themed(self.root, r["key"], "side", self.panel, aspect)
                self.built.exhibits.append({"id": ex_id, "key": r["key"], "preset": "side", "png": png,
                                            "number": None, "sidebar": True})
                p = H.para(side, 4, 2, keep_next=True)
                self.runs(p, a["title"], 9, H.C["green"], bold=True)
                p = H.para(side, 0, 2, align="center", line=1.0)
                H.image(p, png, H.SIDEBAR)
                p = H.para(side, 0, 4)
                src = self.tokens(a["source"].strip())
                self.runs(p, src if src.lower().startswith("source") else f"Source: {src}", 6.5, H.C["grey"],
                          italic=True)
            elif k.kind == "table":
                hdr, rows = k.items[0], k.items[1:]
                if hdr[0].strip() and not any(h.strip() for h in hdr[1:]):  # "| Key data (31 Oct 2026) | |": a heading
                    p = H.para(side, 4, 2, keep_next=True)
                    H.run(p, hdr[0].strip("* "), 9, H.C["green"], bold=True)
                size = 7.5 if self.compact < 2 else 7
                lab = max(len(r[0].replace("*", "")) for r in rows) if rows else 10
                cap = H.SIDEBAR - 1100 if len(hdr) <= 2 else int(H.SIDEBAR * 0.45)  # leave the values room
                w0 = min(int(lab * size * 9.0) + 200, cap)  # label column: no label wraps where it fits
                widths = [w0] + [round((H.SIDEBAR - w0) / (len(hdr) - 1))] * (len(hdr) - 1)
                widths[-1] += H.SIDEBAR - sum(widths)
                self.house_table(side, hdr, rows, H.SIDEBAR, k.align, size, True, widths=widths)
                H.spacer(side, 4)
            else:
                self.render(side, k, H.SIDEBAR, size=8.5)
        if self.meta.get("ticker") and "," in self.meta["ticker"]:  # several tickers (foreign report)
            eyebrow = f"{self.meta['title']}  ·  DSE: {self.meta['ticker']}"
        elif self.meta.get("ticker"):
            eyebrow = f"{self.company()}  ·  DSE: {self.meta['ticker']}"
        else:
            eyebrow = self.meta["title"]
        if self.meta.get("sector"):
            eyebrow += f"  ·  {self.meta['sector']}"
        p = H.para(col, 0, 2)
        H.run(p, eyebrow, 8.5, H.C["teal"], bold=True)
        title = self.meta["subtitle"]
        if main and main[0].kind in ("part", "h3"):  # the # (or ###) heading is the page-1 title, not a part label
            first = main[0]
            title = f"{first.label}: {first.text}" if first.kind == "part" and first.label else first.text
            main = main[1:]
        p = H.para(col, 0, 8, line=None, exact=19)
        self.runs(p, title, 17, H.C["bottle"], bold=True)
        size = 10 if self.compact < 2 else 9.5
        # a layout="page1" exhibit and everything after it go under the sidebar table at full text width (prompt 24b)
        cut = next((i for i, n in enumerate(main) if n.kind == "exhibit" and n.attrs.get("layout") == "page1"),
                   len(main))
        main, below = main[:cut], main[cut:]
        for n in main:
            if n.kind == "block" and n.label == "sidebar":
                raise BuildError(f"{n.where}: only one :::sidebar per page")
            self.render(col, n, main_w, size=size)
        for c in (side, gap, col):
            H.finish_cell(c)
        for n in below:
            H.spacer(self.doc, 4)
            if n.kind == "exhibit" and n.attrs.get("layout") == "page1":
                wide = Node("exhibit", n.line, attrs=dict(n.attrs, id=self.exhibit_id(n.attrs["id"], sidebar=True)))
                aspect = {0: PAGE1_ASPECT, 1: 0.34, 2: 0.30}[min(self.compact, 2)]
                self.exhibit(self.doc, wide, H.W_FULL, "full", numbered=False, max_aspect=aspect)
            else:
                self.render(self.doc, n, H.W_FULL, size=size)
        last = below[-1] if below else (main[-1] if main else None)
        words = title.replace("*", "").split()
        self.page_one_marks = [" ".join(words[:4]), _mark(last) if last else "",
                               _mark(sidebar.children[-1]) if sidebar.children else ""]
        self.page_has_content = True

    def company(self) -> str:
        """Legal name for the front-matter ticker (config/banks.yaml)."""
        from tzbank.config import banks

        names = {b["ticker"]: b.get("legal_name") or b["name"] for b in banks(self.root)}
        if self.meta["ticker"] not in names:
            raise BuildError(f"{self.src.path}: ticker {self.meta['ticker']} is not in config/banks.yaml")
        return names[self.meta["ticker"]]

    def check_rating(self) -> None:
        """The front-matter rating against the pipeline's rating at the current valuation date (report within
        dashboard, D65): a difference is a build warning."""
        f = self.root / "outputs" / "tables" / "ch09_valuations.csv"
        if not f.exists():
            return
        v = pd.read_csv(f)
        cur = v[(v["bank"] == self.meta["ticker"]) & (v["valuation"] == "CURR")]
        if len(cur) and str(cur["rating"].iloc[0]).capitalize() != self.meta["rating"]:
            self.built.warnings.append(f"front-matter rating {self.meta['rating']} differs from the pipeline rating "
                                       f"{cur['rating'].iloc[0]} ({self.meta['ticker']}, ch09_valuations, CURR)")

    def rating_box(self, cont, n: Node | None) -> None:
        """Green rating box: the front-matter rating, then the lines of the sidebar's leading paragraph (a line in
        **bold** prints in gold)."""
        cell = H.box(cont, H.SIDEBAR, H.C["green"])
        H.cell_style(cell, H.C["green"], (200, 180, 260, 200))
        lines = (n.raw or [n.text]) if n else []
        if lines and lines[0].lower().startswith("rating:"):
            raise BuildError(f"{n.where}: the rating comes from the front matter (rating: {self.meta['rating']}); "
                             "remove the 'Rating:' line")
        p = H.para(cell, 0, 3)
        H.run(p, "VERDICT" if self.meta["rating"] not in RATINGS else "RATING", 7.5, H.C["label"], bold=True)
        verdict = self.meta["rating"] not in RATINGS  # the foreign report: a verdict string, printed as written
        if verdict:
            p = H.para(cell, 0, 3 if lines else 0, line=1.0)
            H.run(p, self.meta["rating"], 14 if len(self.meta["rating"]) <= 28 else 12, H.C["white"], bold=True)
        else:
            p = H.para(cell, 0, 3 if lines else 0, line=None, exact=30)
            H.run(p, self.meta["rating"].upper(), 26, H.C["white"], bold=True)
        for ln in lines:
            gold = _all_bold(ln)
            p = H.para(cell, 0, 0)
            self.runs(p, ln.strip("*") if gold else ln, 8.5, H.C["gold"] if gold else H.C["white"], bold=gold)
        H.finish_cell(cell)
        H.spacer(cont, 8)

    def disclosures(self) -> None:
        f = self.root / self.c.get("source_dir", "reports/source") / self.c.get("disclosures", "disclosures.md")
        self.new_page()
        H.banner(self.doc, "", "Disclosures", toc=("Disclosures", 2))
        H.spacer(self.doc, 8)
        if not f.exists():
            self.built.warnings.append(f"{f.relative_to(self.root).as_posix()} not found: the disclosures page is empty")
            cell = H.box(self.doc, H.W_FULL, H.C["pale_gold"], H.C["gold"])
            p = H.para(cell, 0, 0)
            H.run(p, f"[Owner text required: {f.name}]", 9.5, H.C["gold_ink"], bold=True)
            H.finish_cell(cell)
            return
        frag = parse_fragment(f)
        for n in frag:
            if n.kind in ("part", "chapter"):
                raise BuildError(f"{f}:{n.line}: the disclosures page takes no # or ## headings")
            self.render(self.doc, n, H.W_FULL)

    def build(self, out: Path) -> Built:
        kind = (self.c.get("kind") or {}).get(self.rep, "Research report")
        line = f"{kind}  |  Prices at {self.meta['date']}  |  Financials to {self.c.get('financials_to', 'FY2025')}"
        H.cover(self.doc, self.assets["cover"], self.meta, self.c.get("cover_eyebrow", ""), line)
        sec = H.section(self.doc)
        label = self.meta["title"].upper()
        if self.rep in INVESTMENT:
            label = f"{self.company().upper()}  |  EQUITY RESEARCH"
        if self.meta.get("rating") in RATINGS and self.meta.get("ticker"):
            self.check_rating()
        foot = dict(self.c["footer"])
        if foot.get("line2"):
            foot["line2"] = self.tokens(foot["line2"])  # [[github_url]] etc. (prompt 25)
        self.hf = (label, self.meta["date"], foot)
        H.header_footer(sec, *self.hf, self.assets["page"])
        nodes = list(self.src.nodes)
        self.body(nodes)
        if not self.has_disclosures:  # the source's own Disclosures section replaces the builder's page
            self.disclosures()
        out.parent.mkdir(parents=True, exist_ok=True)
        core = self.doc.core_properties
        core.title, core.author, core.subject = self.meta["title"], self.meta["author"], self.meta["subtitle"]
        core.comments = "Built by tzbank.reporting.build_reports (prompt 16); text from " + \
            self.src.path.name
        self.built.links = H.link_urls(self.doc)  # every printed URL becomes a hyperlink (D99)
        self.doc.save(out)
        self.built.docx = out
        return self.built


def parse_fragment(f: Path) -> list[Node]:
    """A body-only source (no front matter), e.g. the disclosures page."""
    from tzbank.reporting.report_source import _body, _strip_comments

    lines = _strip_comments(f.read_text(encoding="utf-8").splitlines())
    return _body([(k + 1, ln) for k, ln in enumerate(lines)], f)


# --- PDF, fonts, previews, checks --------------------------------------------------------------------------------------
WORD_PS = r"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open('%DOCX%', $false, $false, $false)
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  $d.Save()
  $d.ExportAsFixedFormat('%PDF%', 17, $false, 0, 0, 0, 0, 0, $true, $true, 0, $true, $true, $false)
  $d.Close($false)
} finally { $w.Quit() }
"""


def to_pdf(docx: Path, engine: str = "auto") -> Path | None:
    """docx -> PDF: Word via COM (Windows; also updates the TOC and saves the .docx), else LibreOffice headless."""
    pdf = docx.with_suffix(".pdf")
    if engine == "none":
        return None
    word_ok = platform.system() == "Windows" and engine in ("auto", "word")
    if word_ok:
        ps = WORD_PS.replace("%DOCX%", str(docx.resolve())).replace("%PDF%", str(pdf.resolve()))
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=600)
        if r.returncode == 0 and pdf.exists():
            return pdf
        if engine == "word":
            raise BuildError(f"Word PDF export failed: {r.stderr.strip()[:500]}")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice and engine in ("auto", "libreoffice"):
        r = subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and pdf.exists():
            return pdf
        raise BuildError(f"LibreOffice PDF export failed: {r.stderr.strip()[:500]}")
    raise BuildError("no PDF engine: install Microsoft Word (Windows) or LibreOffice, or pass --no-pdf")


def pdf_fonts(pdf: Path) -> dict:
    """{font name: embedded?} over all pages."""
    import pymupdf

    out: dict[str, bool] = {}
    with pymupdf.open(pdf) as d:
        for i in range(d.page_count):
            for f in d.get_page_fonts(i):
                name = f[3].split("+", 1)[-1]
                out[name] = out.get(name, True) and f[1] not in ("n/a", "")
    return out


def previews(pdf: Path, out_dir: Path, report: str, pages: int, dpi: int) -> list[Path]:
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    with pymupdf.open(pdf) as d:
        for i in range(min(pages, d.page_count)):
            f = out_dir / f"{report}_p{i + 1}.png"
            d[i].get_pixmap(dpi=dpi).save(f)
            files.append(f)
    return files


def check(root: Path, built: Built) -> dict:
    """Figure checker on the built .docx (D65): per-number CSV next to the report and a status count."""
    from tzbank.reporting.check_figures import check as cf
    from tzbank.reporting.check_figures import summary

    res = cf(built.docx, root=root)
    res.to_csv(built.docx.with_name(f"{built.report}_figures.csv"), index=False)
    counts = res["status"].value_counts().to_dict() if len(res) else {}
    return {"summary": summary(res) if len(res) else "no numbers", "counts": counts,
            "not_on_dashboard": int(res["not_on_dashboard"].sum()) if len(res) else 0}


REMEDY_ORDER = ("defer", "side", "shrink")


def defer_exhibits(nodes: list[Node], ids: list[str]) -> list[Node]:
    """Remedy 1 (prompt 26): the paragraphs and lists that follow an exhibit print before it, so the text continues
    on the page and the exhibit opens the next one."""
    nodes = list(nodes)
    for ex in ids:
        i = next((k for k, n in enumerate(nodes) if n.kind == "exhibit" and n.attrs["id"] == ex), None)
        if i is None:
            continue
        j = i + 1
        while j < len(nodes) and _movable(nodes[j]):
            j += 1
        if j > i + 1:
            nodes[i:j] = nodes[i + 1:j] + [nodes[i]]
    return nodes


def _movable(n: Node) -> bool:
    """Text that may print before the exhibit it follows (remedy 1): paragraphs, lists and finding or method boxes."""
    return n.kind in ("para", "list") or (n.kind == "block" and n.label != "sidebar")


def _remedy_round(b: Built, nodes: list[Node], stage: dict, remedies: dict) -> list[str]:
    """One pass of the half-empty page rule (prompt 26, D97): for each page that ends with more than 6 cm blank
    because the next exhibit does not fit, the next remedy in order - (1) move the following text of the same chapter
    up, (2) a side block with the following text, (3) a shorter chart, down to 60% of its height (the text check keeps
    chart text at 7 pt or more). An exhibit that closes a chapter has no following text, so only (3) applies. A shrink
    that would need more than that is re-tried in the next pass, whose pages may differ. Returns the exhibits changed;
    a page with no remedy left stays flagged by the layout check."""
    import pymupdf

    from tzbank.reporting.layout_check import exhibit_gaps

    with pymupdf.open(b.pdf) as d:
        gaps = exhibit_gaps(d, {"exhibits": b.exhibits, "page_breaks": b.page_breaks})
    flat = list(walk(nodes))
    changed = []
    for g in gaps:
        ex = g["exhibit"]
        i = next((k for k, n in enumerate(flat) if n.kind == "exhibit" and n.attrs["id"] == ex), None)
        if i is None:
            continue
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        ok = {"defer": nxt is not None and _movable(nxt),
              "side": nxt is not None and nxt.kind == "para" and flat[i].attrs.get("layout") != "landscape",
              "shrink": bool(g["img_w_pt"])}
        k = stage.get(ex, -1)
        if k >= 0 and REMEDY_ORDER[k] != "shrink":
            k += 1  # the last remedy left the gap: escalate
        k = max(k, 0)
        while k < len(REMEDY_ORDER) and not ok[REMEDY_ORDER[k]]:
            k += 1
        if k >= len(REMEDY_ORDER):
            if remedies.pop(ex, None) is not None:  # back to the original layout; flagged by the layout check
                changed.append(ex)
            stage[ex] = len(REMEDY_ORDER) - 1
            continue
        r = REMEDY_ORDER[k]
        if r == "shrink":
            prev = remedies.get(ex)
            orig_h = prev[2] if isinstance(prev, tuple) else g["img_h_pt"]
            new_h = g["img_h_pt"] - (g["panel_pt"] - (g["free_pt"] - 0.8 * 72 / 2.54))
            if new_h < 0.6 * orig_h:  # too short: not this pass (a defer or side tried before stays undone)
                if not isinstance(prev, tuple) and remedies.pop(ex, None) is not None:
                    changed.append(ex)
                stage[ex] = k
                continue
            remedies[ex] = ("shrink", new_h / g["img_w_pt"], orig_h)
        else:
            remedies[ex] = r
        stage[ex] = k
        changed.append(ex)
    return changed


def page_one_fit(pdf: Path, marks: list[str]) -> dict:
    """Whether the page-1 column ends on the page that holds its title (marks[0]), and the room left there (cm)."""
    import pymupdf

    with pymupdf.open(pdf) as d:
        k = next((i for i in range(1, d.page_count) if marks and marks[0] and d[i].search_for(marks[0])), 1)
        p = d[k]
        flat = re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", p.get_text()))  # a word hyphenated at a line end
        found = [bool(m) and (bool(p.search_for(m)) or re.sub(r"\s+", " ", m) in flat) for m in marks[1:]]
        h = p.rect.height
        footer_top = h - (H.MARGIN["footer"] + 560) / 20  # points (two footer lines, prompt 25)
        bottoms = [b[3] for b in p.get_text("blocks") if 60 < b[1] < footer_top - 2]
        room_cm = (h - H.MARGIN["bottom"] / 20 - max(bottoms)) / 72 * 2.54 if bottoms else None
    return {"fits": all(found) if marks else True, "page": k + 1, "marks_on_page": found,
            "room_left_cm": round(room_cm, 1) if room_cm is not None else None}


def assets(root: Path) -> dict:
    out = root / cfg(root).get("out_dir", "outputs/reports") / "assets"
    m = {k: v / 567 for k, v in H.MARGIN.items()}
    return {"cover": house_art.cover(out / "cover.png"), "page": house_art.page(out / "page.png", m)}


def build(root: Path, report: str, source: Path | None = None, pdf: bool = True) -> Built:
    c = cfg(root)
    src_dir = root / c.get("source_dir", "reports/source")
    try:
        files = [Path(source)] if source else source_files(src_dir, report)
        if not files or not files[0].exists():
            raise BuildError(f"no source for {report or source}: {src_dir / (report + '.md')} or "
                             f"{report}_part<N>.md (owner-authored; D65)")
        src = parse_parts(files)
    except SourceError as e:
        raise BuildError(str(e)) from None
    src_path = files[0]
    name = src_path.stem if source else report
    out = root / c.get("out_dir", "outputs/reports") / f"{name}.docx"
    art = assets(root)
    sidebar = any(n.kind == "block" and n.label == "sidebar" for n in src.nodes)
    page_one: dict = {}
    from tzbank.reporting import chart_theme

    for level in range(3):  # page 1 must hold its column: shrink the sidebar chart, then the body type
        chart_theme.LEGEND_LOG.clear()
        chart_theme.TEXT_LOG.clear()
        bld = Builder(root, src, art, compact=level)
        b = bld.build(out)
        b.report = name
        if not pdf:
            break
        b.pdf = to_pdf(out, c.get("pdf_engine", "auto"))
        if not (sidebar and b.pdf):
            break
        page_one = page_one_fit(b.pdf, bld.page_one_marks)
        page_one["compact_level"] = level
        if page_one["fits"]:
            break
    if page_one and not page_one["fits"]:
        b.warnings.append(f"page 1 does not fit beside the sidebar even at the tightest setting: {page_one}")
    remedies: dict = {}
    if pdf and b.pdf:  # prompt 26: half-empty pages before a too-tall exhibit, fixed in up to four passes
        stage: dict[str, int] = {}
        for _ in range(5):
            fixes = _remedy_round(b, src.nodes, stage, remedies)
            if not fixes:
                break
            chart_theme.LEGEND_LOG.clear()
            chart_theme.TEXT_LOG.clear()
            level = page_one.get("compact_level", 0) if page_one else 0
            try:
                b = Builder(root, src, art, compact=level, remedies=remedies).build(out)
            except BuildError:  # a shorter chart would print text under 7 pt: that remedy is not available
                for ex in fixes:
                    if isinstance(remedies.get(ex), tuple):
                        remedies.pop(ex)
                        stage[ex] = len(REMEDY_ORDER)
                b = Builder(root, src, art, compact=level, remedies=remedies).build(out)
            b.report = name
            b.pdf = to_pdf(out, c.get("pdf_engine", "auto"))
            if not b.pdf:
                break
    b.remedies = {k: (v if isinstance(v, str) else f"shrink to aspect {v[1]:.2f}") for k, v in remedies.items()}
    if pdf:
        if b.pdf:
            b.fonts = pdf_fonts(b.pdf)
            if not all(b.fonts.values()):
                b.warnings.append("fonts not embedded in the PDF: " + ", ".join(k for k, v in b.fonts.items() if not v))
            b.previews = previews(b.pdf, out.parent / "preview", name, int(c.get("preview_pages", 4)),
                                  int(c.get("preview_dpi", 80)))
    b.figures = check(root, b)
    rel = lambda f: f.resolve().relative_to(root).as_posix() if f.resolve().is_relative_to(root) else str(f)  # noqa: E731
    log = {"report": name, "sources": [rel(f) for f in src.parts], "docx": out.relative_to(root).as_posix(),
           "pdf": b.pdf.relative_to(root).as_posix() if b.pdf else None,
           "exhibits": [{k: (str(v.relative_to(root).as_posix()) if isinstance(v, Path) else v) for k, v in e.items()}
                        for e in b.exhibits],
           "annexes": b.annexes, "page_one": page_one, "fonts_embedded": b.fonts, "figure_check": b.figures,
           "side_blocks": b.side_blocks, "page_breaks": b.page_breaks, "page_remedies": b.remedies,
           "plain_blocks": b.plain_blocks, "one_page_sections": cfg(root).get("one_page_sections", {}).get(name, {}),
           "warnings": b.warnings,
           "previews": [p.relative_to(root).as_posix() for p in b.previews]}
    out.with_name(f"{name}_build.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    lg = pd.DataFrame(chart_theme.LEGEND_LOG, columns=["key", "preset", "overlaps_found", "points_covered",
                                                       "overlaps_after_fix"]).drop_duplicates()
    lg.to_csv(out.with_name(f"{name}_legend_check.csv"), index=False)
    pd.DataFrame(chart_theme.TEXT_LOG).drop_duplicates().to_csv(out.with_name(f"{name}_text_check.csv"), index=False)
    b.warnings += [f"legend still covers data after the fix: {r.key} ({r.preset})" for r in lg.itertuples()
                   if r.overlaps_after_fix]
    if b.pdf:  # prompt 23 (D92): no table name, token, backtick or prompt number may reach the PDF
        from tzbank.reporting.leak_check import scan_pdf

        leaks = scan_pdf(b.pdf, name)
        leaks.to_csv(out.with_name(f"{name}_leak_check.csv"), index=False)
        if len(leaks):
            raise BuildError(f"{len(leaks)} internal name(s) printed in {b.pdf.name}: "
                             + "; ".join(f"p.{r.page} {r.match}" for r in leaks.head(8).itertuples())
                             + f" (all in {name}_leak_check.csv)")
    return b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tzbank.reporting.build_reports", description=__doc__.split("\n")[0])
    ap.add_argument("--report", default="all", choices=["main", "nmb", "crdb", "foreign", "all"])
    ap.add_argument("--source", help="build one source file instead (e.g. reports/source/sample.md)")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    root = find_root()
    sys.stdout.reconfigure(encoding="utf-8")
    c = cfg(root)
    if a.source:
        jobs = [(None, Path(a.source).resolve())]
    elif a.report == "all":
        jobs = [(r, None) for r in ("main", "nmb", "crdb", "foreign")
                if source_files(root / c.get("source_dir", "reports/source"), r)]
        if not jobs:
            print("no report sources in reports/source (<report>.md or <report>_part<N>.md)")
            return 1
    else:
        jobs = [(a.report, None)]
    rc = 0
    for rep, srcf in jobs:
        try:
            b = build(root, rep or "", srcf, pdf=not a.no_pdf)
        except BuildError as e:
            print(f"BUILD FAILED ({rep or srcf}): {e}")
            rc = 1
            continue
        print(f"{b.report}: {b.docx.relative_to(root)}" + (f", {b.pdf.relative_to(root)}" if b.pdf else "")
              + f"; {len(b.exhibits)} exhibits, {len(b.annexes)} annexes; figures: {b.figures['summary']}")
        for w in b.warnings:
            print(f"  warning: {w}")
    return rc


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
