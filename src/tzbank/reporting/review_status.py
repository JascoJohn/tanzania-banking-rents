"""Status of every edit in the owner's Word review (prompt 24b, D94): ``reports/review/edits_<report>.csv`` gains the
final ``status`` and a ``note`` saying what was done and why.

The decisions are the ones recorded in prompt 24b (the analyst's corrections A, layout comments B, other edits C). Rules,
in order: an explicit decision matched on the edit's text and paragraph; a typographic quote change (A10, ignored); a formatting change
(ignored, house style decides layout); an edit with no visible text (an empty paragraph or a space: layout, ignored).

Usage: ``python -m tzbank.reporting.review_status``.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

APPLIED, CORRECTED, NOT_APPLIED, LAYOUT = "applied", "applied with correction", "not applied", "layout (ignored)"

# (report, regex on the edit's paragraph text, old/new text and comment, extra condition, status, note). First match
# wins. Matching is on text, not on edit ids, so the decisions survive a re-read of the review files.
TABLE_CELL = r"^(FY202[45]A|ROE \(average equity\)|Net interest margin|Cost-to-income|Cost of risk|NPL ratio / coverage|" \
             r"CET1 ratio.*|Core capital ratio.*|Total capital ratio.*|Capital headroom.*|[0-9.]+%( / [0-9]+%)?|" \
             r"[0-9.]+ pts)$"
DECISIONS: list[tuple[str, str, str, str, str]] = [
    ("main", r"Jasco John is a finance professional", "", APPLIED, "About the author: owner's biography (C)"),
    ("main", r"capital ratios are Bank-", "", APPLIED, "'Bank-only' for 'Bank-solo', applied to every report (C)"),
    ("main", r"computationally reproducible|All figures come from a reproducible pipeline", "", APPLIED,
     "reproducibility paragraph; GitHub, Zenodo and Streamlit linked to the url tokens (C)"),
    ("main", r"empty page", "", APPLIED, "empty page after the landscape Exhibit 1 removed in the build (B)"),
    ("main", r"What the literature adds|The theory is divided", "", APPLIED, "moved up in Ch1 (owner move)"),
    ("main", r"International experience demonstrates|^International Experien|^ce$", "page<28", CORRECTED,
     "resolution box kept once in Ch3: house box style, sentence case, 'wiped out', SRB (2017) and IMF (2013) "
     "references (A1, A2, A8)"),
    ("main", r"International experience demonstrates|^International Experience$", "page>=28", NOT_APPLIED,
     "duplicate of the Ch3 resolution box (A2)"),
    ("main", r"empty spaces", "", APPLIED, "box applied in Ch3; spacing set by the build (B)"),
    ("main", r"Evolution of Bank Revenue|digital transformation of banking", "", CORRECTED,
     "BIS box: Schmieder, Corrias and Wünsch (2026, BIS WP 1391), full stop, the paper's quotation, Annex E (A3)"),
    ("main", r"International experience highlights|looking beyond reported lending|restructured loans|ecognize|"
             r"These experiences raise|absorb credit risk|Tanzania's own 2015|^International Experience$", "", CORRECTED,
     "credit-loss box rewritten as instructed (A4); IMF 50%/75% sentence from IMF Country Report 24/187, Annex VII "
     "para. 3 (printed p. 78), supplied by the owner and checked, in Annex E"),
    ("main", r"Ongoing Saga|growing investment in government securities", "", CORRECTED,
     "retitled 'Banks and government securities', rewritten with The Citizen, 24 Nov 2025 (A5)"),
    ("main", r"Banking Sector Model|banking-sector profitability|^ and its $|^Outlook $|^not$|^ then$|"
             r"the implications extend", "", CORRECTED,
     "'Banking sector model and outlook', supported by Ch5: all five smaller panel banks below their cost of equity "
     "on average (A8, A9)"),
    ("main", r"Depositor|Concentration|depositor confidentiality|^Side Note: $", "", CORRECTED,
     "side note: depositor concentration, house box style (A8)"),
    ("main", r"Dividends and Competitive|competitive value|among Tanzanian banks|dividend distributions are consistently",
     "", CORRECTED, "side note: dividends and competitive advantage, placed after the surplus table (A8)"),
    ("main", r"Related Party|related-party exposures during FY2025|^T$|^he Timing", "", CORRECTED,
     "retitled 'Related-party exposures and leadership succession'; tenure rule corrected and cited (Corporate "
     "Governance Regulations 2021); no individuals named (A6)"),
    ("main", r"A more particular|reprice banks|fundamental differen", "", APPLIED, "closing question added in Ch10"),
    ("main", r"nnualized both franchises|^recognises$", "", NOT_APPLIED, "find-and-replace error: 'recognises' kept (A1)"),
    ("main", r"^annualised$|9\.5% nnualized", "", NOT_APPLIED, "find-and-replace error: 'annualised' kept (A1)"),
    ("main", r"Currency Translation|Cross Border|Currency depreciation does not impose|^Currency $|^ranslation and $",
     "", CORRECTED, "currency box: currencies named from the pipeline's FX data (ch12_cross_rates, from end-2020, the "
                    "earliest held), claim softened (A7, A8)"),
    ("main", r"formulae|k_e = ", "", APPLIED, "typed formula replaced; Annex A formulas as Word equations (OMML) (B)"),
    ("main", r"Context: what caused|Public deposits\.|Liquidity\. The 2017/18|The deposit rule|Policy uncertainty and "
             r"arrears|On these sources the stress", "", APPLIED, "Ch5 context box moved up after the IFRS 9 bullets"),
    ("main", r"^Key finding$|Lower bill yields erode", "", APPLIED, "Ch6 key finding moved up (owner move)"),
    ("main", r"Remove this empty space", "", APPLIED, "the moved Ch5 box starts at the top of a page (B)"),
    ("nmb|crdb", r"made horizontal|more horizontal|moves up to the first page|^What would change our view$|"
                 r"A cost of equity 1\.5 points lower; a larger", "", APPLIED,
     "page 1: price chart wide under the opening text, 'What would change our view' beneath it (B)"),
    ("nmb|crdb", r"^· |^Value TZS|^Price TZS|^12-month target|^Expected 12-month return|^Hold if the cost", "page<3",
     APPLIED, "sidebar items one per line, each bulleted"),
    ("nmb", r"make this paragraph fit", "", APPLIED, "household paragraph first, loan-book chart after it, wide"),
    ("nmb|crdb", TABLE_CELL + r"|ensure consistency|added it here", "", APPLIED,
     "key-financials table under the persistence heading, the same in NMB and CRDB, fully visible (B)"),
    ("nmb", r"Rating lines and what would justify|On the October 2026 value path|^Exhibit 4$|P/B is near the top of its "
            r"own history|Year-end P/B with the 31 October", "", APPLIED, "rating-lines text before Exhibit 4 (B)"),
    ("nmb", r"below the paragraph", "", APPLIED, "ROE bridge below its paragraph, wide"),
    ("crdb", r"^Exhibit 5$|Sustainable ROE falls short|Steps: cost of risk|figure horizontal", "", APPLIED,
     "ROE bridge: CRDB panel only, wide, after its paragraph (B)"),
    ("crdb", r"page above", "", APPLIED, "profit bridge now sits on the page above (B)"),
    ("crdb", r"no need for empty space", "", APPLIED, "'What we monitor' starts on a fresh page (B; same in NMB)"),
    ("foreign", r"Expected 12-month USD return|^depreciation $|^· ", "page<3", APPLIED,
     "sidebar items one per line, bulleted; '2026 depreciation pace' wording"),
    ("foreign", r"graph should be made horizontal", "", APPLIED,
     "page 1: chart wide, 'When the answer would change' beneath it (B; same in NMB and CRDB)"),
    ("foreign", r"annex a", "", APPLIED, "body peer table no longer headed 'Annex A' (B)"),
    ("foreign", r"How prices form|Block trades were|^Diversification$|two stocks barely move|^Costs$|"
                r"At the DSE fee cap", "", APPLIED, "section order as the owner moved it (the source already matched)"),
    ("foreign", r"make this horizontal", "", APPLIED, "rolling-correlation chart wide (B)"),
]
CURLY = re.compile(r"^['‘’\"“”]$")


def status(row) -> tuple[str, str]:
    rep = row.report
    old, new = str(row.old_text or ""), str(row.new_text or "")
    if row.kind == "replace" and CURLY.match(old.strip() or "x") and CURLY.match(new.strip() or "x"):
        return "formatting (ignored)", "curly quote from Word; house convention is straight quotes (A10)"
    vis = old.strip() or new.strip() or str(row.comment_text or "").strip()
    part = str(getattr(row, "part", "document") or "document")
    if part.startswith("footer") and vis:
        return APPLIED, ("footer: two lines on every page of all four reports, line 2 'Code / Data / Dashboard' from "
                         "the url tokens, built as a real second line (prompt 25)")
    if part != "document" and vis:
        return "needs owner", f"edit in {part}: not mapped"
    for reps, pat, cond, st, note in DECISIONS:
        if not re.fullmatch(reps, rep) or not vis:
            continue
        if cond and not eval(cond, {}, {"page": int(row.page)}):  # noqa: S307 - a fixed condition from this file
            continue
        if any(re.search(pat, t.strip(), re.I) for t in (old, new, str(row.comment_text or ""))) or \
                re.search(pat, str(row.paragraph_new or "") + " " + str(row.paragraph_old or ""), re.I):
            return st, note
    if row.kind == "comment" and not str(row.comment_text or "").strip():
        return "no action", "comment with no text"
    if row.kind == "format":
        return "formatting (ignored)", "formatting only; the house style decides layout"
    if not (old.strip() or new.strip()) and row.kind != "comment":
        return LAYOUT, "empty paragraph or space (Word layout); spacing is set by the build"
    return "needs owner", "not matched to a decision"


def main() -> int:
    from tzbank.config import find_root

    sys.stdout.reconfigure(encoding="utf-8")
    rev = find_root() / "reports" / "review"
    for r in ("main", "nmb", "crdb", "foreign"):
        f = rev / f"edits_{r}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, keep_default_na=False)
        res = [status(x) for x in d.itertuples()]
        d["status"], d["note"] = [s for s, _ in res], [n for _, n in res]
        d.to_csv(f, index=False)
        print(r, d["status"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
