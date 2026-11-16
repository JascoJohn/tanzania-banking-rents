"""House-style Word primitives (prompt 16; D77): python-docx plus direct OOXML.

Matches the approved sample (``reports/design/House_Style_Sample.pdf``, built by ``house_style_build.js``): A4, page
colour #F8FAF9, Calibri, chapter banner (green, gold left edge), In brief box (soft green), Key finding / limitation
callout (pale gold, gold left edge), exhibit panel (white, thin rule border), house table (green header, light
banding, figures right-aligned, green closing rule), running header and footer, cover. Sizes follow the sample's
docx-js values (half-points / twips) so the two render alike. Layout tables carry ``tblCaption = layout`` so the
figure checker reads their text as prose, not as data tables.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Emu, Pt, RGBColor, Twips

C = {"green": "006B3F", "bottle": "06452E", "soft": "E5F1E8", "teal": "66A88A", "gold": "E6B323", "page": "F8FAF9",
     "white": "FFFFFF", "text": "252B28", "grey": "6B736F", "rule": "D3E3D8", "band": "F2F7F3", "pale_gold": "FFF7DF",
     "gold_ink": "8A6A00", "label": "CDE6D6"}
FONT = "Calibri"
PAGE_W, PAGE_H = 11906, 16838  # A4, twips
MARGIN = {"top": 1250, "bottom": 1000, "left": 1134, "right": 1134, "header": 560, "footer": 480}
W_FULL = PAGE_W - MARGIN["left"] - MARGIN["right"]  # 9638
GAP = 240
SIDE_TXT = 5700  # two-thirds / one-third (sample page 4)
SIDEBAR = 3400  # investment page 1 sidebar (sample page 3)
EMU_PER_TWIP = 635


# --- low level ---------------------------------------------------------------------------------------------------------
def el(tag: str, **attrs) -> OxmlElement:
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k.replace("_", ":", 1)) if "_" in k else qn(f"w:{k}"), str(v))
    return e


def _rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def run(p, text: str, size: float = 10, color: str = C["text"], bold: bool = False, italic: bool = False):
    r = p.add_run(text)
    f = r.font
    f.name, f.size, f.bold, f.italic, f.color.rgb = FONT, Pt(size), bold, italic, _rgb(color)
    rpr = r._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), FONT)
    return r


URL = re.compile(r"https?://[^\s<>\"'·]+")
MD_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")  # [display text](url), as in the report sources (D100)


def _link_style(r_el, color: str) -> None:
    """House link style on a run element: house green, dotted underline (owner's reference image, D100)."""
    from docx.enum.text import WD_UNDERLINE
    from docx.text.run import Run

    f = Run(r_el, None).font
    f.color.rgb = _rgb(color)
    f.underline = WD_UNDERLINE.DOTTED


def _wrap_link(r_el, part, url: str):
    """Wrap a run element in w:hyperlink pointing at ``url`` (external relationship of ``part``)."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), part.relate_to(url, RT.HYPERLINK, is_external=True))
    h.set(qn("w:history"), "1")
    r_el.addprevious(h)
    h.append(r_el)
    return h


def link_run(p, text: str, url: str, size: float = 10, color: str = C["green"], bold: bool = False,
             italic: bool = False):
    """A run printing ``text`` and linking ``url`` (https: or mailto:), in the house link style (D100)."""
    r = run(p, text, size, color, bold, italic)
    _link_style(r._r, color)
    _wrap_link(r._r, p.part, url)
    _no_hyphens(p._p)
    return r


def runs_with_links(p, text: str, size: float, color: str) -> None:
    """Plain text with [display](url) links (the footer's second line)."""
    pos = 0
    for m in MD_LINK.finditer(text):
        if m.start() > pos:
            run(p, text[pos:m.start()], size, color)
        link_run(p, m.group(1), m.group(2), size)
        pos = m.end()
    if pos < len(text):
        run(p, text[pos:], size, color)


def _url_end(u: str) -> str:
    """A URL without the sentence punctuation that follows it ('…/zenodo.1.' -> '…/zenodo.1')."""
    while u and u[-1] in ".,;:)]":
        u = u[:-1]
    return u


def link_urls(doc, color: str = C["green"]) -> int:
    """Turn every printed URL into a Word hyperlink (D99), in the body, tables, text boxes, headers and footers.

    A run holding a URL is split into the text before, the URL and the text after; the URL run keeps the run's
    font and size, takes the house green, and is wrapped in ``w:hyperlink`` with an external relationship, which
    Word's PDF export writes as a link annotation. Returns the number of links made."""
    import copy

    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    parts = {id(doc.part): doc.part}
    for s in doc.sections:
        for hf in (s.header, s.footer, s.first_page_header, s.first_page_footer, s.even_page_header,
                   s.even_page_footer):
            if not hf.is_linked_to_previous:
                parts[id(hf.part)] = hf.part
    n = 0
    for part in parts.values():
        for r in list(part.element.iter(qn("w:r"))):
            if r.getparent() is None or r.getparent().tag == qn("w:hyperlink"):
                continue
            ts = r.findall(qn("w:t"))
            if len(ts) != 1 or not ts[0].text or "http" not in ts[0].text:
                continue
            text, pieces, pos = ts[0].text, [], 0
            for m in URL.finditer(text):
                a, b = m.start(), m.start() + len(_url_end(m.group()))
                if a > pos:
                    pieces.append((text[pos:a], None))
                pieces.append((text[a:b], text[a:b]))
                pos = b
            if not any(u for _, u in pieces):
                continue
            if pos < len(text):
                pieces.append((text[pos:], None))
            parent = r.getparent()
            idx = parent.index(r)
            for txt, url in pieces:
                nr = copy.deepcopy(r)
                t = nr.find(qn("w:t"))
                t.text = txt
                t.set(qn("xml:space"), "preserve")
                node = nr
                if url:
                    _link_style(nr, color)
                    node = OxmlElement("w:hyperlink")
                    node.set(qn("r:id"), part.relate_to(url, RT.HYPERLINK, is_external=True))
                    node.set(qn("w:history"), "1")
                    node.append(nr)
                    n += 1
                parent.insert(idx, node)
                idx += 1
            parent.remove(r)
            _no_hyphens(parent)
    return n


# successors of w:suppressAutoHyphens in CT_PPr (ECMA-376 order), so the element lands where Word expects it
_PPR_AFTER_HYPHENS = ("w:kinsoku", "w:wordWrap", "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
                      "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
                      "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                      "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
                      "w:pPrChange")


def _no_hyphens(node) -> None:
    """No automatic hyphenation in the paragraph holding a link, so a wrapped URL never gains a hyphen (D99)."""
    while node is not None and node.tag != qn("w:p"):
        node = node.getparent()
    if node is None:
        return
    ppr = node.get_or_add_pPr()
    if ppr.find(qn("w:suppressAutoHyphens")) is None:
        ppr.insert_element_before(el("w:suppressAutoHyphens"), *_PPR_AFTER_HYPHENS)


def fmt(p, before: float = 0, after: float = 6, line: float | None = 1.15, align=None, indent_left: int = 0,
        indent_right: int = 0, keep_next: bool = False, exact: float | None = None):
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if exact:
        pf.line_spacing = Pt(exact)
        pf.line_spacing_rule = 4  # WD_LINE_SPACING.EXACTLY
    elif line:
        pf.line_spacing = line
    if align:
        pf.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT,
                        "center": WD_ALIGN_PARAGRAPH.CENTER, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    if indent_left:
        pf.left_indent = Twips(indent_left)
    if indent_right:
        pf.right_indent = Twips(indent_right)
    if keep_next:
        pf.keep_with_next = True
    return p


def para(container, before=0, after=6, **kw):
    """New paragraph in a document body or a table cell (reuses a cell's untouched first paragraph)."""
    if hasattr(container, "_tc"):
        kids = [k for k in container._tc if k.tag != qn("w:tcPr")]
        if len(kids) == 1 and kids[0].tag == qn("w:p") and len(kids[0]) == 0:  # untouched first paragraph
            return fmt(container.paragraphs[0], before, after, **kw)
    return fmt(container.add_paragraph(), before, after, **kw)


def field(p, instr: str, result: str = "", size: float = 10, color: str = C["text"], bold: bool = False) -> None:
    """Complex field (PAGE, TOC, TC) in paragraph p."""
    def fc(kind):
        r = run(p, "", size, color, bold)
        r._r.append(el("w:fldChar", fldCharType=kind))
    fc("begin")
    r = run(p, "", size, color, bold)
    it = el("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r._r.append(it)
    if result or instr.strip().startswith(("PAGE", "TOC")):
        fc("separate")
        run(p, result, size, color, bold)
    fc("end")


def page_break(container) -> None:
    """Start what follows on a new page: a 1-pt paragraph with 'page break before'. Unlike a page-break character,
    it adds no page when the previous page is already full, so no empty page appears (prompt 25)."""
    p = para(container, 0, 0, line=None)
    fmt(p, 0, 0, line=None, exact=1)
    p.paragraph_format.page_break_before = True
    run(p, "", 1)


def spacer(container, pts: float = 6, keep_next: bool = False) -> None:
    p = para(container, 0, 0, line=None)
    fmt(p, 0, 0, line=None, exact=max(pts, 1), keep_next=keep_next)
    run(p, "", 1)


# --- tables ------------------------------------------------------------------------------------------------------------
def _border(tag: str, val="single", sz=4, color=C["rule"]):
    return el(f"w:{tag}", val=val, sz=sz, space=0, color=color)


def table(container, widths: list[int], borders: dict | None = None, layout: bool = True, rows: int = 1):
    """Fixed-width table; borders: {top,left,bottom,right,insideH,insideV: (val, sz, color)}; default none."""
    t = container.add_table(rows=rows, cols=len(widths))
    t.autofit = False
    tbl = t._tbl
    pr = tbl.tblPr
    for tag in ("w:tblStyle", "w:tblW", "w:tblLayout", "w:tblBorders", "w:tblLook", "w:tblCellMar"):
        for x in pr.findall(qn(tag)):
            pr.remove(x)
    pr.append(el("w:tblW", w=sum(widths), type="dxa"))
    b = el("w:tblBorders")
    spec = borders or {}
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        v = spec.get(side)
        b.append(_border(side, *v) if v else el(f"w:{side}", val="nil"))
    pr.append(b)
    pr.append(el("w:tblLayout", type="fixed"))
    mar = el("w:tblCellMar")
    for side in ("left", "right"):
        mar.append(el(f"w:{side}", w=0, type="dxa"))
    pr.append(mar)
    pr.append(el("w:tblLook", val="0000", firstRow=0, lastRow=0, firstColumn=0, lastColumn=0, noHBand=1, noVBand=1))
    if layout:
        pr.append(el("w:tblCaption", val="layout"))
    grid = tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths, strict=True):
        gc.set(qn("w:w"), str(w))
    for row in t.rows:
        for c, w in zip(row.cells, widths, strict=True):
            c.width = Twips(w)
    return t


def cell_style(cell, fill: str | None = None, margins: tuple[int, int, int, int] | None = None,
               borders: dict | None = None, valign: str | None = None) -> None:
    """Shading, margins (top, bottom, left, right), borders {side: (val, sz, color)}, vertical alignment."""
    tcpr = cell._tc.get_or_add_tcPr()
    for tag in ("w:tcBorders", "w:shd", "w:tcMar", "w:vAlign"):
        for x in tcpr.findall(qn(tag)):
            tcpr.remove(x)
    if borders:
        b = el("w:tcBorders")
        for side in ("top", "left", "bottom", "right"):
            v = borders.get(side)
            b.append(_border(side, *v) if v else el(f"w:{side}", val="nil"))
        tcpr.append(b)
    if fill:
        tcpr.append(el("w:shd", val="clear", color="auto", fill=fill))
    if margins:
        m = el("w:tcMar")
        for side, v in zip(("top", "left", "bottom", "right"), (margins[0], margins[2], margins[1], margins[3]),
                           strict=True):
            m.append(el(f"w:{side}", w=v, type="dxa"))
        tcpr.append(m)
    if valign:
        tcpr.append(el("w:vAlign", val=valign))


def row_props(row, header: bool = False, cant_split: bool = True, height: int | None = None,
              exact: bool = False) -> None:
    trpr = row._tr.get_or_add_trPr()
    if cant_split:
        trpr.append(el("w:cantSplit"))
    if header:
        trpr.append(el("w:tblHeader"))
    if height:
        trpr.append(el("w:trHeight", val=height, hRule="exact" if exact else "atLeast"))


def finish_cell(cell) -> None:
    """A cell must end with a paragraph; drop an untouched leading empty paragraph if the cell has other content."""
    tc = cell._tc
    kids = [k for k in tc if k.tag != qn("w:tcPr")]
    if len(kids) > 1 and kids[0].tag == qn("w:p") and not kids[0].xpath(".//w:t|.//w:drawing|.//w:fldChar") and \
            kids[0].find(qn("w:pPr")) is None:
        tc.remove(kids[0])
    kids = [k for k in tc if k.tag != qn("w:tcPr")]
    if not kids or kids[-1].tag == qn("w:tbl"):
        p = tc.add_p()
        ppr = p.get_or_add_pPr()
        sp = el("w:spacing", before=0, after=0, line=20, lineRule="exact")
        ppr.append(sp)


# --- images ------------------------------------------------------------------------------------------------------------
def image(p, png: Path, width_twips: int):
    """Inline picture scaled to a width; returns (width, height) in twips."""
    from PIL import Image

    with Image.open(png) as im:
        w, h = im.size
    height = round(width_twips * h / w)
    p.add_run().add_picture(str(png), width=Emu(width_twips * EMU_PER_TWIP), height=Emu(height * EMU_PER_TWIP))
    return width_twips, height


def float_image(p, png: Path, width_twips: int, height_twips: int, x: int = 0, y: int = 0) -> None:
    """Picture anchored to the page at (x, y) twips, behind text, no wrapping (cover art, page pattern)."""
    r = p.add_run()
    r.add_picture(str(png), width=Emu(width_twips * EMU_PER_TWIP), height=Emu(height_twips * EMU_PER_TWIP))
    inline = r._r.xpath(".//wp:inline")[0]
    cx, cy = width_twips * EMU_PER_TWIP, height_twips * EMU_PER_TWIP
    anchor = parse_xml(
        f'<wp:anchor {nsdecls("wp", "a")} distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="0" '
        f'behindDoc="1" locked="1" layoutInCell="1" allowOverlap="1"><wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{x * EMU_PER_TWIP}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{y * EMU_PER_TWIP}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/></wp:anchor>')
    anchor.append(inline.find(qn("wp:docPr")))
    anchor.append(parse_xml(f'<wp:cNvGraphicFramePr {nsdecls("wp")}/>'))
    anchor.append(inline.find(qn("a:graphic")))
    inline.getparent().replace(inline, anchor)


# --- document ----------------------------------------------------------------------------------------------------------
def hyphenation(doc, limit: int = 2, zone: int = 340) -> None:
    """Automatic hyphenation (owner, prompt 25): at most ``limit`` consecutive hyphenated lines, hyphenation zone
    ``zone`` twips (340 = 0.6 cm), so justified narrow columns do not open wide gaps."""
    st = doc.settings.element
    for tag in ("w:autoHyphenation", "w:consecutiveHyphenLimit", "w:hyphenationZone"):
        for x in st.findall(qn(tag)):
            st.remove(x)
    new = [el("w:autoHyphenation", val="true"), el("w:consecutiveHyphenLimit", val=limit),
           el("w:hyphenationZone", val=zone)]
    anchor = st.find(qn("w:defaultTabStop"))  # schema order: right after defaultTabStop
    if anchor is not None:
        idx = list(st).index(anchor) + 1
        for k, e in enumerate(new):
            st.insert(idx + k, e)
    else:
        for e in new:
            st.append(e)


def new_document() -> Document:
    """A4 document with the house defaults (Calibri 10 pt, text colour, 1.15 lines, page colour, TOC styles,
    bullet and number lists)."""
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size, st.font.color.rgb = FONT, Pt(10), _rgb(C["text"])
    rpr = st.element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), FONT)
    rpr.rFonts.set(qn("w:cs"), FONT)
    pf = st.paragraph_format
    pf.space_after, pf.line_spacing = Pt(6), 1.15
    styles = doc.styles.element
    for x in styles.findall(qn("w:docDefaults")):
        for f in x.iter(qn("w:rFonts")):
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                f.set(qn(a), FONT)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                f.attrib.pop(qn(a), None)
    for lvl, (ind, bold, color, before) in {1: (0, True, C["green"], 8), 2: (240, False, C["text"], 2)}.items():
        styles.append(parse_xml(
            f'<w:style {nsdecls("w")} w:type="paragraph" w:styleId="TOC{lvl}"><w:name w:val="toc {lvl}"/>'
            f'<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/>'
            f'<w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{W_FULL}"/></w:tabs>'
            f'<w:spacing w:before="{before * 20}" w:after="60"/><w:ind w:left="{ind}"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/>{"<w:b/>" if bold else ""}'
            f'<w:color w:val="{color}"/><w:sz w:val="{21 if lvl == 1 else 20}"/></w:rPr></w:style>'))
    # page colour (also painted by the page art, so every renderer shows it)
    body = doc.element
    body.insert(0, el("w:background", color=C["page"]))
    settings = doc.settings.element
    settings.insert(1, el("w:displayBackgroundShape"))  # after w:zoom (schema order)
    # automatic hyphenation for the justified body text (prompt 21): after w:defaultTabStop (schema order)
    tab = settings.find(qn("w:defaultTabStop"))
    hyph = el("w:autoHyphenation")
    if tab is not None:
        tab.addnext(hyph)
    else:
        settings.append(hyph)
    for cs in settings.iter(qn("w:compatSetting")):  # Word 2013+ layout: table edges sit on the text margin
        if cs.get(qn("w:name")) == "compatibilityMode":
            cs.set(qn("w:val"), "15")
    _numbering(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Twips(PAGE_W), Twips(PAGE_H)
    return doc


def _numbering(doc) -> None:
    num = doc.part.numbering_part.element
    bullet = parse_xml(
        f'<w:abstractNum {nsdecls("w")} w:abstractNumId="90"><w:multiLevelType w:val="singleLevel"/>'
        f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val=""/>'
        f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="300" w:hanging="220"/></w:pPr>'
        f'<w:rPr><w:rFonts w:ascii="Wingdings" w:hAnsi="Wingdings" w:cs="Wingdings" w:hint="default"/>'
        f'<w:color w:val="{C["green"]}"/><w:sz w:val="18"/></w:rPr></w:lvl></w:abstractNum>')
    decimal = parse_xml(
        f'<w:abstractNum {nsdecls("w")} w:abstractNumId="91"><w:multiLevelType w:val="singleLevel"/>'
        f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
        f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="340" w:hanging="300"/></w:pPr>'
        f'<w:rPr><w:b/><w:color w:val="{C["green"]}"/></w:rPr></w:lvl></w:abstractNum>')
    first_num = num.find(qn("w:num"))
    for a in (bullet, decimal):
        if first_num is not None:
            first_num.addprevious(a)
        else:
            num.append(a)
    num.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="90"><w:abstractNumId w:val="90"/></w:num>'))
    doc._house_next_num = 91  # noqa: SLF001


def list_num(doc, ordered: bool) -> int:
    """numId for a list: one shared bullet list; each ordered list restarts at 1."""
    if not ordered:
        return 90
    doc._house_next_num += 1  # noqa: SLF001
    n = doc._house_next_num  # noqa: SLF001
    doc.part.numbering_part.element.append(parse_xml(
        f'<w:num {nsdecls("w")} w:numId="{n}"><w:abstractNumId w:val="91"/><w:lvlOverride w:ilvl="0">'
        f'<w:startOverride w:val="1"/></w:lvlOverride></w:num>'))
    return n


def list_para(container, num_id: int, after: float = 4, size: float = 10):
    """List item. Bullets are drawn as a green Wingdings square and a Calibri tab (Word's own bullet suffix falls back
    to a non-embedded Arial space in the PDF); numbered items use Word numbering (restarting per list)."""
    p = para(container, 0, after)
    if num_id == 90:
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Twips(300), Twips(-220)
        pf.tab_stops.add_tab_stop(Twips(300))
        r = run(p, "", size * 0.9, C["green"])
        fonts = r._r.rPr.find(qn("w:rFonts"))
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(a), "Wingdings")
        run(p, "	", size)
        return p
    ppr = p._p.get_or_add_pPr()
    numpr = el("w:numPr")
    numpr.append(el("w:ilvl", val=0))
    numpr.append(el("w:numId", val=num_id))
    ppr.insert(0, numpr)
    return p


def section(doc, cover: bool = False):
    """A new page section with the house margins (cover: no margins)."""
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    set_margins(sec, cover)
    return sec


def set_margins(sec, cover: bool = False) -> None:
    sec.page_width, sec.page_height = Twips(PAGE_W), Twips(PAGE_H)
    m = {k: 0 for k in MARGIN} if cover else MARGIN
    sec.top_margin, sec.bottom_margin = Twips(m["top"]), Twips(m["bottom"])
    sec.left_margin, sec.right_margin = Twips(m["left"]), Twips(m["right"])
    sec.header_distance, sec.footer_distance = Twips(m["header"]), Twips(m["footer"])


def header_footer(sec, label: str, date: str, footer: dict, pattern: Path | None, width: int = W_FULL) -> None:
    """Running header (label left, date right, thin rule; page art anchored behind) and footer (contact, page).
    ``width``: the section's text width, so the date and page number sit at its right margin (a landscape section
    passes its own; prompt 23). ``pattern=None``: no page art (the portrait art does not fit a landscape page)."""
    sec.header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False
    h = sec.header
    p0 = h.paragraphs[0]
    fmt(p0, 0, 0, line=None, exact=1)
    if pattern is not None:
        float_image(p0, pattern, PAGE_W, PAGE_H)
    p = h.add_paragraph()
    fmt(p, 0, 0, line=None)
    p.paragraph_format.tab_stops.add_tab_stop(Twips(width), alignment=2)  # right
    ppr = p._p.get_or_add_pPr()
    bdr = el("w:pBdr")
    bdr.append(el("w:bottom", val="single", sz=4, space=4, color=C["rule"]))
    ppr.append(bdr)
    run(p, label, 7.5, C["green"], bold=True)
    run(p, f"\t{date}", 7.5, C["grey"])
    f = sec.footer.paragraphs[0]
    fmt(f, 0, 0, line=None)
    for pos in (4680, 9360):  # Word's Footer style tabs (centre, right) would catch the page number first (prompt 23)
        f.paragraph_format.tab_stops.add_tab_stop(Twips(pos), alignment=WD_TAB_ALIGNMENT.CLEAR)
    # centred footer (owner, after prompt 25): the contact line on a centre tab, the page number at the right margin
    f.paragraph_format.tab_stops.add_tab_stop(Twips(width // 2), alignment=WD_TAB_ALIGNMENT.CENTER)
    f.paragraph_format.tab_stops.add_tab_stop(Twips(width), alignment=2)
    run(f, "\t", 7)
    run(f, footer["name"], 7, C["green"], bold=True)
    run(f, "  |  " + "  |  ".join(footer["details"]), 7, C["grey"])
    run(f, "\t", 8)
    field(f, "PAGE", "1", 8, C["green"], bold=True)
    if footer.get("line2"):  # a proper second line (owner, prompt 25), not spaces pushing text over
        f2 = sec.footer.add_paragraph()
        fmt(f2, 1, 0, line=None, align="center")
        # short display text linked to the full address (D100), so the line reads at 7 pt
        runs_with_links(f2, footer["line2"], float(footer.get("line2_size", 7)), C["grey"])


def blank_header_footer(sec) -> None:
    sec.header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False


# --- house elements ----------------------------------------------------------------------------------------------------
def banner(container, label: str, title: str, width: int = W_FULL, toc: tuple[str, int] | None = None) -> None:
    """Chapter banner: green panel, gold left edge, small label line and title."""
    t = table(container, [width])
    c = t.cell(0, 0)
    cell_style(c, C["green"], (160, 160, 280, 200), {"left": ("single", 48, C["gold"])})
    row_props(t.rows[0])
    if label:
        p = para(c, 0, 2, keep_next=True)
        run(p, label.upper(), 9, C["label"], bold=True)
    p = para(c, 0, 0, keep_next=True, line=None, exact=19)
    run(p, title, 17, C["white"], bold=True)
    if toc:
        field(p, f'TC "{toc[0].replace(chr(34), chr(39))}" \\l {toc[1]}')
    finish_cell(c)


def part_label(container, label: str, title: str) -> None:
    p = para(container, 0, 4, keep_next=True)
    run(p, f"{label}  ·  {title}".upper() if label else title.upper(), 8.5, C["teal"], bold=True)
    entry = ((label + ". ") if label else "") + title
    field(p, f'TC "{entry.replace(chr(34), chr(39))}" \\l 1')


def box(container, width: int, fill: str, edge: str | None = None, border: str | None = None):
    """A one-cell panel; returns the cell to fill (finish_cell after filling)."""
    t = table(container, [width])
    c = t.cell(0, 0)
    b = {s: ("single", 4, border) for s in ("top", "left", "bottom", "right")} if border else {}
    if edge:
        b["left"] = ("single", 24, edge)
    cell_style(c, fill, (140, 120, 220, 220), b or None)
    return c


def toc(container) -> None:
    """Contents page: banner and a TOC field over the TC entries (updated when the PDF is exported)."""
    banner(container, "", "Contents")
    spacer(container, 8)
    p = para(container, 0, 0)
    field(p, 'TOC \\f \\l "1-2" \\h \\z', "Update this field (F9) to list the contents.", 10, C["grey"])


KEEP_TOGETHER_ROWS = 15  # prompt 17 A3: tables of up to 15 rows are kept on one page

NUMERIC = re.compile(r"^[\s(+\-−–]*[A-Z]{0,3}\s?[\d.,]+\s*(%|×|x|pp|bn|tn|m|pts?)?\)?\s*$")


def is_numeric(s: str) -> bool:
    return bool(NUMERIC.match(s.replace("*", ""))) or s.strip() in ("–", "-", "n/a", "")


def house_table(container, header: list, rows: list[list], widths: list[int], add_runs, align: list[str] | None = None,
                size: float = 8, no_header: bool = False, bold_rows: set | None = None) -> None:
    """House data table: green header, banding, rules, green closing rule; ``add_runs(p, text, size, color, bold)``
    writes inline text. Columns right-align when most body cells are figures (or as ``align`` says)."""
    ncol = len(widths)
    if not align or not any(align):
        align = ["left"] + ["right" if sum(is_numeric(str(r[j])) for r in rows) >= 0.6 * max(len(rows), 1) else "left"
                            for j in range(1, ncol)]
    align = [a or ("left" if j == 0 else "right") for j, a in enumerate(align)]
    body = [r for r in rows]
    t = table(container, widths, {"bottom": ("single", 8, C["green"]), "insideH": ("single", 2, C["rule"])},
              layout=False, rows=len(body) + (0 if no_header else 1))
    k = 0
    if not no_header:
        row_props(t.rows[0], header=True)
        for j, h in enumerate(header):
            c = t.cell(0, j)
            cell_style(c, C["green"], (50, 50, 90, 90), valign="center")
            p = para(c, 0, 0, align=align[j], line=1.0)
            add_runs(p, str(h), size, C["white"], True)
        k = 1
    together = len(body) <= KEEP_TOGETHER_ROWS  # short tables stay on one page; longer ones repeat the header row
    for i, r in enumerate(body):
        row_props(t.rows[k + i])
        bold = bool(bold_rows and i in bold_rows)
        for j, v in enumerate(r):
            c = t.cell(k + i, j)
            cell_style(c, C["band"] if i % 2 == 1 else C["white"], (50, 50, 90, 90), valign="center")
            p = para(c, 0, 0, align=align[j], line=1.0, keep_next=together and i < len(body) - 1)
            add_runs(p, str(v), size, C["text"], bold)
    if not no_header:
        for c in t.rows[0].cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True


def col_widths(header: list, rows: list[list], total: int) -> list[int]:
    """Column widths in proportion to content (first column wider; each at least 700 twips and wide enough for its
    longest single word, so a header such as 'Threshold' or 'performance' is never hyphenated; prompt 23)."""
    n = len(header)
    lens, floor = [], []
    for j in range(n):
        cells = [re.sub(r"\*", "", str(header[j]))] + [re.sub(r"\*", "", str(r[j])) for r in rows]
        lens.append(min(max(len(c) for c in cells), 48) + 2)
        word = max((len(w) for c in cells for w in c.split()), default=0)
        floor.append(max(700, word * 115 + 260))  # about 115 twips per character at 8 pt bold, plus cell padding
    lens[0] = max(lens[0], 14)
    if sum(floor) > total * 0.9:  # many columns: scale the word floors so every column keeps a share (prompt 24)
        floor = [max(600, round(f * total * 0.9 / sum(floor))) for f in floor]
    w = [max(f, round(total * x / sum(lens))) for x, f in zip(lens, floor, strict=True)]
    extra = sum(w) - total
    if extra > 0:  # take the excess from the columns with room above their floor, widest first
        for j in sorted(range(n), key=lambda j: w[j] - floor[j], reverse=True):
            cut = min(extra, w[j] - floor[j])
            w[j] -= cut
            extra -= cut
    w[0] += total - sum(w)
    return w


def cover(doc, art: Path, meta: dict, eyebrow: str, line: str) -> None:
    """Cover page (first section, no header or footer): art behind, then the text in fixed-height rows so the
    cover never spills onto a second page."""
    sec = doc.sections[0]
    set_margins(sec, cover=True)
    p0 = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
    fmt(p0, 0, 0, line=None, exact=1)
    float_image(p0, art, PAGE_W, PAGE_H)
    ind = 1300
    t = table(doc, [PAGE_W], rows=3)
    for r, h in zip(t.rows, (3000, 8800, 2200), strict=True):
        row_props(r, height=h, exact=True)
    a, b, c = (t.cell(i, 0) for i in range(3))
    for cell in (a, b, c):
        cell_style(cell, margins=(0, 0, ind, 2200))
    p = para(a, 90, 0)
    run(p, eyebrow, 9, C["label"], bold=True)
    title = meta["title"]
    lines = [title[: title.index(",") + 1], title[title.index(",") + 1:].strip()] if "," in title else [title]
    for i, ln in enumerate(lines):
        p = para(b, 60 if i == 0 else 0, 18 if i == len(lines) - 1 else 0, line=None, exact=36)
        run(p, ln, 32, C["white"], bold=True)
    p = para(b, 0, 45, indent_right=1200)
    run(p, meta["subtitle"], 15, C["soft"])
    q = meta["question"].strip()
    m = re.match(r"^(.+?[.?!])\s+(.*)$", q)
    first, rest = (m.group(1), m.group(2)) if m else (q, "")
    p = para(b, 0, 3)
    run(p, first, 12, C["gold"], bold=True)
    if rest:
        p = para(b, 0, 0, indent_right=1100)
        run(p, rest, 12, C["white"])
    p = para(c, 0, 2)
    run(p, line, 9, C["soft"])
    p = para(c, 0, 0)
    run(p, meta["author"], 10, C["white"], bold=True)
    for cell in (a, b, c):
        finish_cell(cell)
