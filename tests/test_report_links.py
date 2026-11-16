"""Printed URLs become Word hyperlinks, in the body and in footers, and their paragraphs never hyphenate (D99)."""

from __future__ import annotations

from docx.oxml.ns import qn

from tzbank.reporting import house_docx as H


def test_link_urls_splits_runs_and_relates_targets():
    d = H.new_document()
    p = d.add_paragraph()
    H.fmt(p, 0, 6, align="justify", keep_next=True)
    H.run(p, "See https://doi.org/10.5281/zenodo.24280749. Code (https://github.com/JascoJohn/tanzania-banking-rents)", 9)
    f = d.sections[0].footer.paragraphs[0]
    H.run(f, "Dashboard: https://tanzania-banking-rents.streamlit.app/", 6.5)
    assert H.link_urls(d) == 3
    links = p._p.findall(qn("w:hyperlink"))
    texts = ["".join(t.text for t in h.iter(qn("w:t"))) for h in links]
    assert texts == ["https://doi.org/10.5281/zenodo.24280749", "https://github.com/JascoJohn/tanzania-banking-rents"]
    assert p.text.endswith("tanzania-banking-rents)") and "24280749. Code (" in p.text  # punctuation kept outside
    targets = {r.target_ref for r in d.part.rels.values() if r.reltype.endswith("/hyperlink")}
    assert targets == {"https://doi.org/10.5281/zenodo.24280749", "https://github.com/JascoJohn/tanzania-banking-rents"}
    foot = {r.target_ref for r in d.sections[0].footer.part.rels.values() if r.reltype.endswith("/hyperlink")}
    assert foot == {"https://tanzania-banking-rents.streamlit.app/"}
    order = [c.tag.split("}")[1] for c in p._p.pPr]
    assert order.index("keepNext") < order.index("suppressAutoHyphens") < order.index("spacing")
    assert H.link_urls(d) == 0  # idempotent: runs already inside a hyperlink are left alone
