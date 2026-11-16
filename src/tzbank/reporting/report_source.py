"""Parser for owner-authored report sources in ``reports/source/*.md`` (prompt 16; D65, D77).

The grammar is documented in ``reports/source/README.md`` and parsed exactly; anything unknown fails with the file and
line. The parser only structures the text: it never changes a word of it.

- YAML front matter with exactly: report (main|nmb|crdb|foreign), title, subtitle, question, date, author.
- ``# Part I. ...`` part label; ``## Chapter 1. ...`` banner; ``### ...`` subsection heading (``####`` is an error).
- Fenced blocks ``:::inbrief``, ``:::finding``, ``:::limitation``, ``:::thesis``, ``:::sidebar``, ``:::method``,
  ``:::case``, each closed by ``:::`` (no nesting).
- ``{{exhibit id=".." title=".." source=".." layout="full|side|half"}}``, ``{{table src=".." filter=".." cols=".."}}``
  and ``{{annex type=".." bank=".."}}`` on a line of their own; ``{{ref id=".."}}`` inline.
- Markdown tables, ``-``/``*`` bullets, ``1.`` numbered items, paragraphs, ``**bold**``, ``*italic*``; ``<!-- -->``
  comments are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPORTS = ("main", "nmb", "crdb", "foreign")
FRONT = ("report", "title", "subtitle", "question", "date", "author")
INVESTMENT = ("nmb", "crdb")  # investment reports: front matter also carries ticker, sector and rating (page 1)
INVEST_FRONT = ("ticker", "sector", "rating")
RATINGS = ("Buy", "Hold", "Sell")
BLOCKS = ("inbrief", "finding", "limitation", "thesis", "sidebar", "method", "case")
LAYOUTS = ("full", "side", "half", "landscape", "page1")  # landscape: full-page section (prompt 21); page1: 24b
ANNEX_TYPES = ("price_history", "financials", "ratios", "valuation", "scenarios", "montecarlo", "peers", "glossary")
BANK_ANNEXES = ("price_history", "financials", "ratios", "valuation", "scenarios", "montecarlo")
MARKERS = {  # marker: (required attributes, optional attributes)
    "exhibit": (("id", "title", "source"), ("layout",)),
    "table": (("src",), ("filter", "cols", "source", "group")),
    "annex": (("type",), ("bank", "inline")),  # inline: an annex-built table in the body (prompt 24b)
    "ref": (("id",), ()),
    "pagebreak": ((), ()),  # start what follows on a fresh page (owner layout comments, prompt 24b)
}
ATTR = re.compile(r'(\w+)="([^"]*)"')
MARK = re.compile(r"\{\{\s*(\w+)((?:\s+\w+=\"[^\"]*\")*)\s*\}\}")
STAR_ESC, STAR_HOLD = "\\*", "\u2063"  # escaped asterisk and its placeholder (invisible separator)
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|\{\{.*?\}\})")


class SourceError(ValueError):
    """A report source that does not follow the documented grammar."""


@dataclass
class Node:
    kind: str  # part, chapter, h3, para, list, table, block, exhibit, ptable, annex
    line: int
    text: str = ""
    label: str = ""
    attrs: dict = field(default_factory=dict)
    items: list = field(default_factory=list)  # list items, or table rows (first row = header)
    ordered: bool = False
    align: list = field(default_factory=list)
    children: list = field(default_factory=list)
    raw: list = field(default_factory=list)  # raw lines of a paragraph (the sidebar rating box keeps line breaks)
    where: str = ""  # "<file>:<line>" for messages (multi-part sources)


@dataclass
class Source:
    path: Path
    meta: dict
    nodes: list
    parts: list = field(default_factory=list)  # the source files in reading order (one, or <report>_part<N>.md)


def attrs(marker: str, body: str, where: str) -> dict:
    """Parse and validate the attributes of one ``{{marker ...}}``."""
    if marker not in MARKERS:
        raise SourceError(f"{where}: unknown marker {{{{{marker}}}}}; allowed: {', '.join(MARKERS)}")
    a = dict(ATTR.findall(body))
    req, opt = MARKERS[marker]
    if leftover := ATTR.sub("", body).strip():
        raise SourceError(f"{where}: cannot read attributes {leftover!r} in {{{{{marker}}}}} (use name=\"value\")")
    if missing := [k for k in req if not a.get(k)]:
        raise SourceError(f"{where}: {{{{{marker}}}}} needs {', '.join(missing)}")
    if extra := [k for k in a if k not in req + opt]:
        raise SourceError(f"{where}: {{{{{marker}}}}} does not take {', '.join(extra)}")
    if marker == "exhibit":
        a.setdefault("layout", "full")
        if a["layout"] not in LAYOUTS:
            raise SourceError(f"{where}: exhibit layout {a['layout']!r} is not one of {', '.join(LAYOUTS)}")
    if marker == "annex":
        if a["type"] not in ANNEX_TYPES:
            raise SourceError(f"{where}: annex type {a['type']!r} is not one of {', '.join(ANNEX_TYPES)}")
        if a["type"] in BANK_ANNEXES and a.get("bank") not in ("NMB", "CRDB"):
            raise SourceError(f"{where}: annex {a['type']} needs bank=\"NMB\" or bank=\"CRDB\"")
        if a.get("bank") and a["bank"] not in ("NMB", "CRDB"):
            raise SourceError(f"{where}: annex bank must be NMB or CRDB")
    return a


def inline(text: str, where: str = "") -> list[tuple[str, dict]]:
    """Split text into runs: (text, {"bold", "italic"}) or ("", {"ref": id}). Unknown inline markers fail."""
    runs: list[tuple[str, dict]] = []
    if re.fullmatch(r"\$\$.+\$\$", text.strip(), re.S):  # a display equation (LaTeX): no inline markup inside
        return [(text.strip(), {"equation": True})]
    if STAR_ESC in text:  # \* is a literal asterisk (ROE\*, ER\*), never italics (prompt 24b)
        return [(t.replace(STAR_HOLD, "*"), f) for t, f in inline(text.replace(STAR_ESC, STAR_HOLD), where)]
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("{{"):
            m = MARK.fullmatch(part)
            if not m:
                raise SourceError(f"{where}: unreadable marker {part!r}")
            if m.group(1) != "ref":
                raise SourceError(f"{where}: {{{{{m.group(1)}}}}} must stand on its own line; only {{{{ref}}}} is inline")
            runs.append(("", {"ref": attrs("ref", m.group(2), where)["id"]}))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            runs += [(t, f | {"bold": True}) for t, f in inline(part[2:-2], where)]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs += [(t, f | {"italic": True}) for t, f in inline(part[1:-1], where)]
        else:
            if "{{" in part or "}}" in part:
                raise SourceError(f"{where}: unbalanced marker braces in {part[:60]!r}")
            runs.append((part, {}))
    return runs


def split_heading(text: str) -> tuple[str, str]:
    """'Chapter 1. Central to ...' -> ('Chapter 1', 'Central to ...'); no short label -> ('', text)."""
    m = re.match(r"^([^.]{1,40}?)\.\s+(.+)$", text)
    if m and len(m.group(1).split()) <= 4:
        return m.group(1), m.group(2)
    return "", text


def _front(lines: list[str], path: Path) -> tuple[dict, int]:
    if not lines or lines[0].strip() != "---":
        raise SourceError(f"{path}:1: the file must start with YAML front matter (---)")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise SourceError(f"{path}: front matter is not closed with ---") from None
    meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    if missing := [k for k in FRONT if not meta.get(k)]:
        raise SourceError(f"{path}: front matter needs {', '.join(missing)}")
    if extra := [k for k in meta if k not in FRONT + INVEST_FRONT]:
        raise SourceError(f"{path}: front matter does not take {', '.join(extra)} "
                          f"(allowed: {', '.join(FRONT + INVEST_FRONT)})")
    if meta["report"] not in REPORTS:
        raise SourceError(f"{path}: report must be one of {', '.join(REPORTS)}")
    has = [k for k in INVEST_FRONT if meta.get(k)]
    rep = meta["report"]
    if rep in INVESTMENT and len(has) < 3:
        raise SourceError(f"{path}: an investment report ({rep}) needs "
                          f"{', '.join(k for k in INVEST_FRONT if k not in has)} in the front matter")
    if rep == "foreign" and "rating" not in has:
        raise SourceError(f"{path}: the foreign report needs rating (its verdict string) in the front matter")
    if rep == "main" and has and len(has) < 3:  # optional, but all three together (a page-1 sidebar needs them)
        raise SourceError(f"{path}: ticker, sector and rating go together; missing "
                          f"{', '.join(k for k in INVEST_FRONT if k not in has)}")
    meta = {k: str(v).strip() for k, v in meta.items()}
    if "ticker" in has:  # "NMB", "DSE: NMB" or, in the foreign report, "DSE: NMB, CRDB"
        meta["ticker"] = re.sub(r"^DSE:\s*", "", meta["ticker"], flags=re.I).upper()
    if rep != "foreign" and has:  # a rating; the foreign report's rating is a verdict string, kept as written
        r = meta["rating"].capitalize()
        if r not in RATINGS:
            raise SourceError(f"{path}: rating must be one of {', '.join(RATINGS)} (not {meta['rating']!r})")
        meta["rating"] = r
        if rep in INVESTMENT and meta["ticker"] != rep.upper():
            raise SourceError(f"{path}: ticker {meta['ticker']} does not match report {rep}")
    return meta, end + 1


def _strip_comments(lines: list[str]) -> list[str]:
    """Blank out <!-- --> comments (line numbers are kept)."""
    out, inside = [], False
    for ln in lines:
        s = ln
        if inside:
            if "-->" in s:
                s, inside = s.split("-->", 1)[1], False
            else:
                out.append("")
                continue
        while "<!--" in s:
            a, b = s.split("<!--", 1)
            if "-->" in b:
                s = a + b.split("-->", 1)[1]
            else:
                s, inside = a, True
        out.append(s)
    if inside:
        raise SourceError("an HTML comment <!-- is not closed")
    return out


def _cells(row: str) -> list[str]:
    r = row.strip()
    r = r[1:] if r.startswith("|") else r
    r = r[:-1] if r.endswith("|") else r
    return [c.strip() for c in r.split("|")]


def _standalone(s: str) -> bool:
    """A line that is a block marker ({{exhibit}}, {{table}}, {{annex}} or anything unknown), not inline {{ref}}."""
    return s.startswith("{{") and not re.match(r"^\{\{\s*ref\b", s)


def _body(lines: list[tuple[int, str]], path: Path, in_block: bool = False) -> list[Node]:
    nodes: list[Node] = []
    i = 0
    while i < len(lines):
        no, ln = lines[i]
        s = ln.strip()
        where = f"{path}:{no}"
        if not s:
            i += 1
            continue
        if s.startswith(":::"):
            kind = s[3:].strip()
            if not kind:
                raise SourceError(f"{where}: ':::' closes a block that was not opened")
            if kind not in BLOCKS:
                raise SourceError(f"{where}: unknown block :::{kind}; allowed: {', '.join(BLOCKS)}")
            if in_block:
                raise SourceError(f"{where}: blocks cannot be nested (:::{kind})")
            j = i + 1
            while j < len(lines) and lines[j][1].strip() != ":::":
                if lines[j][1].strip().startswith(":::") and lines[j][1].strip()[3:].strip():
                    raise SourceError(f"{path}:{lines[j][0]}: block :::{kind} (line {no}) is not closed before "
                                      f"{lines[j][1].strip()}")
                j += 1
            if j == len(lines):
                raise SourceError(f"{where}: block :::{kind} is not closed with :::")
            nodes.append(Node("block", no, label=kind, children=_body(lines[i + 1:j], path, True)))
            i = j + 1
            continue
        if s.startswith("#"):
            m = re.match(r"^(#+)\s+(.+)$", s)
            if not m or len(m.group(1)) > 3:
                raise SourceError(f"{where}: only #, ## and ### headings are allowed")
            if in_block:
                raise SourceError(f"{where}: headings are not allowed inside a block")
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 3:
                nodes.append(Node("h3", no, text=txt))
            else:
                lab, title = split_heading(txt)
                nodes.append(Node("part" if lvl == 1 else "chapter", no, text=title, label=lab))
            i += 1
            continue
        if _standalone(s):
            m = MARK.fullmatch(s)
            if not m:
                raise SourceError(f"{where}: unreadable marker {s[:80]!r}")
            name = m.group(1)
            a = attrs(name, m.group(2), where)
            nodes.append(Node({"exhibit": "exhibit", "table": "ptable", "annex": "annex", "pagebreak": "pagebreak"}[name], no, attrs=a))
            if name == "exhibit":
                inline(a["title"], where), inline(a["source"], where)
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i][1].strip().startswith("|"):
                rows.append(_cells(lines[i][1]))
                i += 1
            if len(rows) < 2 or not all(re.fullmatch(r":?-{3,}:?", c) for c in rows[1]):
                raise SourceError(f"{where}: a table needs a header row and a |---| separator row")
            if any(len(r) != len(rows[0]) for r in rows):
                raise SourceError(f"{where}: table rows have different numbers of cells")
            align = ["center" if c.startswith(":") and c.endswith(":") else "right" if c.endswith(":") else
                     "left" if c.startswith(":") else "" for c in rows[1]]
            for r in rows:
                for c in r:
                    inline(c, where)
            nodes.append(Node("table", no, items=[rows[0], *rows[2:]], align=align))
            continue
        if re.match(r"^([-*]|\d+\.)\s+", s):
            ordered = bool(re.match(r"^\d+\.", s))
            items: list[str] = []
            while i < len(lines) and lines[i][1].strip():
                t = lines[i][1].strip()
                m = re.match(r"^([-*]|\d+\.)\s+(.*)$", t)
                if m:
                    items.append(m.group(2))
                elif items:
                    items[-1] += " " + t
                i += 1
                if i < len(lines) and lines[i][1].strip().startswith((":::", "{{", "|", "#")):
                    break
            for it in items:
                inline(it, where)
            nodes.append(Node("list", no, items=items, ordered=ordered))
            continue
        raw = []
        while i < len(lines):
            t = lines[i][1].strip()
            if not t or t.startswith((":::", "|", "#")) or _standalone(t) or (raw and re.match(r"^([-*]|\d+\.)\s+", t)):
                break
            raw.append(t)
            i += 1
        text = " ".join(raw)
        inline(text, where)
        nodes.append(Node("para", no, text=text, raw=raw))
    return nodes


def parse(path: Path) -> Source:
    """Read one report source; raises SourceError with file:line on anything outside the grammar."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    meta, start = _front(lines, path)
    try:
        body = _strip_comments(lines[start:])
    except SourceError as e:
        raise SourceError(f"{path}: {e}") from None
    nodes = _body([(start + k + 1, ln) for k, ln in enumerate(body)], path)
    _locate(nodes, path)
    return Source(path, meta, nodes, [path])


def _locate(nodes: list[Node], path: Path) -> None:
    for n in walk(nodes):
        n.where = f"{path}:{n.line}"


PART = re.compile(r"^(?P<report>.+)_part(?P<n>\d+)\.md$")


def source_files(src_dir: Path, report: str) -> list[Path]:
    """The files of one report: ``<report>.md`` if it exists, else ``<report>_part<N>.md`` in numeric order of N
    (``main_part0.md`` ... ``main_part3.md``); empty if neither exists."""
    single = Path(src_dir) / f"{report}.md"
    if single.exists():
        return [single]
    parts: dict[int, Path] = {}
    for f in sorted(Path(src_dir).glob(f"{report}_part*.md")):
        m = PART.match(f.name)
        if not m or m.group("report") != report:
            continue
        n = int(m.group("n"))
        if n in parts:
            raise SourceError(f"{f}: part number {n} is used twice ({parts[n].name} and {f.name})")
        parts[n] = f
    return [parts[n] for n in sorted(parts)]


def parse_parts(files: list[Path]) -> Source:
    """Parse a report split into parts: front matter only in the first (lowest-numbered) file; the bodies are joined
    in order. Blocks and comments cannot run across files; messages name the part file and its line."""
    if not files:
        raise SourceError("no source files")
    src = parse(files[0])
    for f in files[1:]:
        lines = Path(f).read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip() == "---":
            raise SourceError(f"{f}:1: front matter belongs only in the lowest-numbered part ({Path(files[0]).name})")
        try:
            body = _strip_comments(lines)
        except SourceError as e:
            raise SourceError(f"{f}: {e}") from None
        nodes = _body([(k + 1, ln) for k, ln in enumerate(body)], Path(f))
        _locate(nodes, Path(f))
        src.nodes += nodes
    src.parts = [Path(f) for f in files]
    return src


def walk(nodes: list[Node]):
    """All nodes in reading order, block children included."""
    for n in nodes:
        yield n
        yield from walk(n.children)
