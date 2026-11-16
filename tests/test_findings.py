"""Prompt 12a: findings pack, figure checker, pack diff (report subset of dashboard)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tzbank.config import path
from tzbank.reporting.check_figures import check, numbers, read_text
from tzbank.reporting.findings import diff_pack, latest_pack


def _pack(root) -> Path:
    return latest_pack(root)  # the latest pack


def test_pack_ids_unique_and_complete(root):
    p = pd.read_csv(_pack(root), low_memory=False)
    assert p["id"].is_unique and len(p) > 1000
    for c in ("value", "unit", "chapter", "source_file", "register_version", "input_hash", "dashboard_page",
              "evidence_class"):
        assert p[c].notna().all(), c
    assert p["dashboard_shown"].all()
    assert (p["register_version"] == pd.read_csv(root / "config" / "assumption_register.csv")["version"].iloc[0]).all()


def test_every_verdict_is_in_the_pack(root):
    p = pd.read_csv(_pack(root), low_memory=False)
    v = p[p["unit"] == "verdict"]
    for f in sorted(path("tables", root).glob("ch*_hypotheses.csv")):
        h = pd.read_csv(f)
        for r in h.itertuples():
            hit = v[v["id"].str.endswith(f"{str(r.hypothesis).replace('.', '_')}.VERDICT")
                    & (v["source_file"].str.endswith(f.name))]
            assert len(hit) == 1 and hit["value"].iloc[0] == r.verdict, (f.name, r.hypothesis)


def test_headline_tier_covers_the_valuations(root):
    p = pd.read_csv(_pack(root), low_memory=False)
    h = p[p["tier"] == "headline"]
    assert h["id"].str.startswith("CH09.VALUATIONS.VALUE_PER_SHARE.").sum() == 4
    assert h["id"].str.startswith("CH11.MONTE_CARLO.P50.").sum() == 4
    assert _pack(root).with_suffix(".md").exists()


def test_checker_reads_numbers_with_precision():
    n = numbers("Value 11,062 a share; return 57.3%; gap +2.6 pp; TZS 16,517.6bn; FY2025 D59 H9.1 in 2024; 1.8e-16")
    got = dict(zip(n["text"], zip(n["value"], n["status"], strict=True), strict=True))
    assert got["11,062"][0] == 11062 and got["57.3%"][0] == pytest.approx(0.573)
    assert got["+2.6 pp"][0] == pytest.approx(0.026)
    assert got["TZS 16,517.6bn"][0] == pytest.approx(16517.6e9)
    assert got["1.8e-16"][0] == pytest.approx(1.8e-16)
    assert all(s == "ignored" for t, (_, s) in got.items() if t in ("2025", "59", "9.1", "2024"))


def test_checker_matches_flags_and_reads_office_files(root, tmp_path):
    p = pd.read_csv(_pack(root), low_memory=False)
    r = p[p["id"].str.startswith("CH09.VALUATIONS.VALUE_PER_SHARE.HIST.NMB")].iloc[0]
    v = float(r["value"])
    md = tmp_path / "draft.md"
    md.write_text(f"NMB is worth {v:,.0f} a share; a stale figure {v * 1.03:,.0f}; an invented 98,765,4.\n",
                  encoding="utf-8")
    res = check(md, _pack(root), root)
    st = dict(zip(res["text"], res["status"], strict=True))
    assert st[f"{v:,.0f}"] == "matched" and st[f"{v * 1.03:,.0f}"] == "mismatch"
    # report subset of dashboard: a matched figure whose pack rows are not on the dashboard is flagged
    q = p.copy()
    num = pd.to_numeric(q["value"], errors="coerce")
    q.loc[(num - v).abs() <= 0.5, "dashboard_shown"] = False  # every pack row the printed figure matches
    alt = tmp_path / "pack.csv"
    q.to_csv(alt, index=False)
    res2 = check(md, alt, root)
    assert res2.loc[res2["text"] == f"{v:,.0f}", "not_on_dashboard"].all()
    # .docx and .pptx are read as zipped XML
    for name, member in (("d.docx", "word/document.xml"), ("s.pptx", "ppt/slides/slide1.xml")):
        f = tmp_path / name
        with zipfile.ZipFile(f, "w") as z:
            z.writestr(member, f"<x><w:p><w:t>Value {v:,.0f}</w:t></w:p></x>")
        assert f"{v:,.0f}" in read_text(f)


def test_chapter_notes_match_the_pack(root):
    """Checker test on the chapter notes: no matched figure off the dashboard; >= 97% of checked figures matched."""
    tot = ok = 0
    for ch in ("01", "06", "08", "09", "10", "12"):
        res = check(path("notes", root) / f"ch{ch}.md", _pack(root), root)
        assert not res["not_on_dashboard"].any(), ch
        c = res[res["status"] != "ignored"]
        tot, ok = tot + len(c), ok + int((c["status"] == "matched").sum())
    assert ok / tot >= 0.97, ok / tot


def test_diff_pack_lists_changes(root, tmp_path):
    p = pd.read_csv(_pack(root), low_memory=False)
    q = p.copy()
    i = q.index[q["id"].str.startswith("CH09.VALUATIONS.RATING.HIST.NMB")][0]
    q.loc[i, "value"] = "Buy"
    j = q.index[q["unit"] == "ratio"][0]
    q.loc[j, "value"] = str(float(q.loc[j, "value"]) + 0.01)
    q = q.drop(q.index[-1])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    p.to_csv(a, index=False)
    q.to_csv(b, index=False)
    d = diff_pack(a, b)
    assert set(d["change"]) == {"changed", "removed"} and (d["change"] == "changed").sum() == 2
    assert d.loc[d["id"] == p.loc[i, "id"], "new"].iloc[0] == "Buy"


def test_checker_external_figures_are_not_matched(root):
    """Prompt 13b pre-flight A: [ext] tags, cited sources and context rules mark external figures."""
    from tzbank.reporting.check_figures import external_rules, numbers

    t = ("The BoT Herfindahl index was 1,216.6 for assets and 1,261.6 for deposits, below the 1,800 reference.\n"
         "The Deposit Insurance Fund holds TZS 1.591tn, 3.2% of deposits. NMB ROE was 26.9%.\n"
         "Banks earned KES 83.5bn, up 13.6% [ext: CBK]. Next 26.9%.\n"
         "ROE 26.9% (source: CBK). NMB had 73,592 agents and 9.9 million accounts.\n"
         "CRDB's 2015\u201318 average; the 364-day yield; gap -0.74 points [-1.53, -0.11]\n"
         "Source: research report Chapters 1\u20133 and 6.")
    n = numbers(t, external_rules(root))
    st = dict(zip(n["line"].astype(str) + ":" + n["text"], n["status"], strict=True))
    for k in ("1:1,216.6", "1:1,261.6", "1:1,800", "2:TZS 1.591tn", "2:3.2%", "3:83.5bn", "3:13.6%", "4:26.9%",
              "4:73,592", "4:9.9 million"):
        assert st[k] == "external", k
    assert st["2:26.9%"] == "" and st["3:26.9%"] == ""  # outside the tagged sentence
    assert st["5:18"] == "ignored" and st["6:3"] == "ignored" and st["5:364"] == "ignored"  # span; caption; tenor
    pts = n[n["line"] == 5].set_index("text")
    assert pts.loc["-0.74 points", "value"] == pytest.approx(-0.0074) and bool(pts.loc["-0.11", "pct_context"])
    assert n.loc[n["text"] == "TZS 1.591tn", "value"].iloc[0] == 1.591e12


def test_checker_header_units_and_minus(tmp_path):
    """Prompt 15b item 4: table header units and the Unicode minus / en dash used as a minus."""
    import zipfile

    from tzbank.reporting.check_figures import header_unit, numbers, read_text

    assert header_unit("Deposits (TZS bn)") == " bn" and header_unit("ROE, %") == "%" and header_unit("P/B (×)") == ""
    t = numbers("returns: −2.3% and –3.4%; band 6.3–18.9")
    assert list(t["value"][:3]) == pytest.approx([-0.023, -0.034, 6.3])

    def cell(x):
        return f"<w:tc><w:p><w:r><w:t>{x}</w:t></w:r></w:p></w:tc>"

    rows = [["Year", "Deposits (TZS bn)", "ROE (%)", "P/B (×)"], ["2025", "16,517.6", "−3.2", "1.4"]]
    body = "<w:tbl>" + "".join("<w:tr>" + "".join(cell(c) for c in r) + "</w:tr>" for r in rows) + "</w:tbl>"
    f = tmp_path / "t.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml", f"<w:document><w:body>{body}</w:body></w:document>")
    n = numbers(read_text(f))
    assert n.loc[n["text"].str.startswith("16,517.6"), "value"].iloc[0] == pytest.approx(16_517.6e9)
    assert n.loc[n["text"].str.startswith("−3.2"), "value"].iloc[0] == pytest.approx(-0.032)
    assert n.loc[n["text"] == "1.4", "kind"].iloc[0] == "plain"
