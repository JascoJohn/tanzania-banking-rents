"""House chart theme for the designed reports (prompt 16; D77).

Every chart figure is drawn once, by its chapter module, for the dashboard. ``snapshot`` keeps a pickled copy of the
figure as drawn, before its footnote is added (called by ``tzbank.chapters.ch04.common.save``), in
``outputs/exhibits/report/_src/<key>.pickle`` (git-ignored). ``render`` re-exports a snapshot in the house theme for
one of three size presets:

- no embedded figure title or footnote (the exhibit panel carries them); panel titles of multi-panel charts stay;
- a Calibri-like sans font, white background, light grid, integer year ticks;
- house data colours: NMB orange, CRDB blue, peers grey, market teal (by series label);
- ``full`` (16.5 cm), ``half`` (8 cm) and ``side`` (6 cm); half and side use larger type, stack side-by-side panels
  vertically and put a simplified legend (no parenthetical detail) below the plot.

Output: ``outputs/exhibits/report/<stem>_<preset>.png``. Dashboard PNGs are never touched. Nothing is calculated here:
the theme changes styling and layout of an already drawn figure only (CLAUDE.md 7).
"""

from __future__ import annotations

import io
import itertools
import pickle
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.category import StrCategoryLocator  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402

CM = 1 / 2.54
PRESETS = {  # width (cm), aspect clamp (h/w), tick, label, legend, other-text clamp (pt)
    "full": {"width": 16.5, "aspect": (0.38, 0.62), "tick": 7.0, "label": 7.5, "legend": 7.0, "text": (7.0, 7.5)},
    "half": {"width": 8.0, "aspect": (0.68, 0.95), "tick": 7.5, "label": 8.0, "legend": 7.5, "text": (7.0, 8.0)},
    "side": {"width": 6.0, "aspect": (0.72, 1.0), "tick": 7.5, "label": 8.0, "legend": 7.5, "text": (7.0, 8.0)},
    "landscape": {"width": 25.0, "aspect": (0.45, 0.62), "tick": 7.0, "label": 7.5, "legend": 7.0, "text": (7.0, 8.0)},
}
MIN_PT = {"chart": 7.0, "diagram": 8.0}  # smallest text at the placed width (prompt 22); Calibri below ~4.7 pt draws no ink
TEXT_LOG: list[dict] = []  # text-survival check (prompt 22): one row per rendered exhibit
LEGEND_LOG: list[dict] = []  # legend-overlap check (prompt 21): one row per rendered exhibit with a legend
FONTS = ["Calibri", "Carlito", "DejaVu Sans"]
HOUSE = {"NMB": "#eb6834", "CRDB": "#2a78d6", "market": "#66A88A"}
PEER_GREYS = ["#8c8c86", "#6b736f", "#a8aca9", "#7d8581", "#b9bdba"]
PEERS = ("DCB", "MBP", "MKCB", "MCB", "MUCOBA", "peer", "Peer", "KCB", "Equity", "Co-operative", "Stanbic", "panel")
MARKET = ("DSEI", "market", "Market", "index")
GRID, SPINE, TICK = "#E3ECE6", "#B9C9BE", "#6B736F"


def font_family() -> str:
    """First installed family of Calibri, Carlito (metric-compatible), DejaVu Sans."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS if f in have), "DejaVu Sans")


def _key(png: Path) -> tuple[str, Path] | None:
    """Snapshot key and folder for a chart PNG: ``<stem>`` for outputs/figures, ``<report>_<nn>`` for exhibits."""
    png = Path(png)
    if png.parent.name == "figures":
        return png.stem, png.parent.parent / "exhibits" / "report" / "_src"
    if png.parent.parent.name == "exhibits" and png.parent.name != "report":
        return f"{png.parent.name}_{png.stem}", png.parent.parent / "report" / "_src"
    return None


class _Pickler(pickle.Pickler):
    """Python 3.14 no longer pickles itertools.count, which matplotlib keeps in string-category axes
    (``category.UnitData``): reduce a counter to its next value."""

    def reducer_override(self, obj):
        if isinstance(obj, itertools.count):
            return itertools.count, (int(re.search(r"count\((-?\d+)", repr(obj)).group(1)),)
        return NotImplemented


def dumps(fig) -> bytes:
    buf = io.BytesIO()
    _Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(fig)
    return buf.getvalue()


def snapshot(fig, png: Path) -> Path | None:
    """Pickle the figure as drawn (before footnote and tick formatting) for the report theme. Never fails the chart:
    a figure that cannot be pickled is skipped (the report build then names it as unresolved)."""
    k = _key(png)
    if k is None:
        return None
    key, folder = k
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{key}.pickle"
    try:
        out.write_bytes(dumps(fig))
    except Exception:  # noqa: BLE001 - styling copy only
        out.unlink(missing_ok=True)
        return None
    return out


def _label_colour(label: str, peer_i: dict) -> str | None:
    """House colour for a series label, or None to keep the drawn colour."""
    if not label or label.startswith("_"):
        return None
    if re.match(r"(DSEI|market|Market|index)", label):  # 'DSEI, rebased to NMB ...' is the market, not the bank
        return HOUSE["market"]
    has_n, has_c = re.search(r"\bNMB\b", label), re.search(r"\bCRDB\b", label)
    if has_n and not has_c:
        return HOUSE["NMB"]
    if has_c and not has_n:
        return HOUSE["CRDB"]
    if has_n or has_c:
        return None
    if any(m in label for m in MARKET):
        return HOUSE["market"]
    if any(re.search(rf"\b{re.escape(p)}\b", label) for p in PEERS):
        peer_i.setdefault(label, PEER_GREYS[len(peer_i) % len(PEER_GREYS)])
        return peer_i[label]
    return None


def _paint(artist, c: str) -> None:
    """Set an artist's colour (line, marker, patch or collection), keeping hollow markers hollow."""
    if hasattr(artist, "set_markeredgecolor"):
        artist.set_color(c)
        mfc = artist.get_markerfacecolor()
        if not (isinstance(mfc, str) and mfc == "none"):
            artist.set_markerfacecolor(c)
        artist.set_markeredgecolor(c)
    elif isinstance(artist, Collection):
        fc = artist.get_facecolor()
        if len(fc) and fc[0][3] > 0:
            artist.set_facecolor(c)
        artist.set_edgecolor(c)
    elif hasattr(artist, "set_facecolor"):
        artist.set_facecolor(c)


def _recolour(fig, peer_i: dict) -> None:
    """Apply house colours to labelled series (lines, scatter/collections, bar containers) and NMB/CRDB text labels."""
    for ax in fig.axes:
        for ln in ax.get_lines():
            if c := _label_colour(ln.get_label(), peer_i):
                _paint(ln, c)
        for col in ax.collections:
            if c := _label_colour(col.get_label(), peer_i):
                _paint(col, c)
        for cont in ax.containers:
            if c := _label_colour(cont.get_label() or "", peer_i):
                for patch in getattr(cont, "patches", []):
                    patch.set_facecolor(c)
        for t in ax.texts:
            s = t.get_text().strip()
            if s in ("NMB", "CRDB"):
                t.set_color(HOUSE[s])


def _simplify(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", label).strip() or label


def _legends(fig, p: dict, preset: str, peer_i: dict) -> None:
    """Full preset: restyle legends in place (the author's placement stays). Half and side: rebuild them below the
    plot with short labels. Legend keys take the house colours of their labels."""
    def keys(leg):
        labels = [t.get_text() for t in leg.get_texts()]
        for h, lab in zip(leg.legend_handles, labels, strict=False):
            if h is not None and (c := _label_colour(lab, peer_i)):
                _paint(h, c)
        return leg.legend_handles, labels

    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        handles, labels = keys(leg)
        if preset == "full":
            leg.set_frame_on(False)
            for t in leg.get_texts():
                t.set_fontsize(p["legend"])
            continue
        leg.remove()
        short = [_simplify(s) for s in labels]
        wide = max(map(len, short), default=0) > (24 if preset == "half" else 16)
        ax.legend(handles, short, fontsize=p["legend"], frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=1 if wide or len(short) == 1 else 2, handlelength=1.4,
                  columnspacing=1.0)
    for leg in list(fig.legends):
        handles, labels = keys(leg)
        leg.remove()
        lab = labels if preset == "full" else [_simplify(s) for s in labels]
        longest = max(map(len, lab), default=0)
        wide = preset != "full" and longest > (24 if preset == "half" else 16)
        # full width: as many columns as fit the placed width (about 0.13 cm a character at 7 pt plus the key),
        # so long labels never run off the page (prompt 23)
        fit = max(1, int(p["width"] / (1.4 + longest * 0.13)))
        fig.legend(handles, lab, fontsize=p["legend"], frameon=False, loc="lower center",
                   ncol=1 if wide else min(len(lab), min(4, fit) if preset == "full" else 2))


def _stack(fig, axes: list) -> int:
    """Stack side-by-side panels vertically (half and side presets). Returns the number of rows."""
    specs = [a.get_subplotspec() for a in axes]
    rows = {s.rowspan.start for s in specs}
    if len(axes) < 2 or len(rows) != 1:
        return len(rows) or 1
    gs = GridSpec(len(axes), 1, figure=fig)
    for i, a in enumerate(axes):
        a.set_subplotspec(gs[i])
        a.yaxis.set_tick_params(labelleft=True)
    return len(axes)


def _place_legends(fig, bottom: float = 0.0) -> None:
    """Put each below-plot legend just under its axis (tick labels and axis label included), then re-fit."""
    r = fig.canvas.get_renderer()
    for _ in range(2):
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            bb = ax.xaxis.get_tightbbox(r)
            if bb is None:
                continue
            y = ax.transAxes.inverted().transform((0, bb.y0))[1]
            leg.set_bbox_to_anchor((0.5, y - 0.02), transform=ax.transAxes)
        fig.tight_layout(pad=0.4, rect=(0, bottom, 1, 1))
        r = fig.canvas.get_renderer()


def _categorical(axis) -> bool:
    """Ticks placed at fixed positions with text labels (set_xticks with labels, or string categories)."""
    return isinstance(axis.get_major_locator(), (FixedLocator, StrCategoryLocator))


def _fit_labels(fig, p: dict) -> None:
    """Half and side presets: category tick labels that do not fit side by side are wrapped and turned upright;
    long axis labels wrap onto two lines; line labels anchored outside the plot (right margin) move inside it."""
    for ax in fig.axes:
        for t in list(ax.texts):  # e.g. 'Sell above: 1,469' at axes fraction x = 1.0: no margin room at this width
            xy = getattr(t, "xy", None)
            coords = getattr(t, "xycoords", None)
            if xy is not None and isinstance(coords, tuple) and coords[0] == "axes fraction" and xy[0] >= 0.99:
                ax.annotate(t.get_text(), xy=(0.99, xy[1]), xycoords=coords, xytext=(0, 1.5),
                            textcoords="offset points", ha="right", va="bottom", color=t.get_color(),
                            fontsize=max(p["text"][0] - 0.5, 6))
                t.remove()
        if _categorical(ax.xaxis):
            labs = [t.get_text() for t in ax.get_xticklabels()]
            if labs and any(re.search(r"[A-Za-z]", lab) for lab in labs):
                width_in = ax.get_position().width * fig.get_figwidth()
                need = sum(min(len(lab), 14) for lab in labs) * p["tick"] * 0.5 / 72
                if need > width_in * 0.9:
                    short = [textwrap.shorten(lab.replace("\n", " "), 24, placeholder="...") for lab in labs]
                    ax.set_xticks(ax.get_xticks(), short, rotation=90, ha="center")
                    grow = max(map(len, short)) * p["tick"] * 0.5 / 72  # upright labels take height, not width
                    fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() + grow)
        if _categorical(ax.yaxis):  # horizontal bars: wrap long category labels
            labs = [t.get_text() for t in ax.get_yticklabels()]
            width = 14 if fig.get_figwidth() < 3 else 20
            if labs and any(len(lab) > width for lab in labs):
                wrapped = [textwrap.fill(lab.replace("\n", " "), width) for lab in labs]
                ax.set_yticks(ax.get_yticks(), wrapped)
                lines = sum(w.count("\n") + 1 for w in wrapped)  # each category needs its lines of height
                need = lines * p["tick"] * 1.3 / 72 / max(ax.get_position().height, 0.3)
                if need > fig.get_figheight():
                    fig.set_size_inches(fig.get_figwidth(), need)
        for lab in (ax.xaxis.label, ax.yaxis.label):
            t = lab.get_text()
            if len(t) > 22 and "\n" not in t:
                lab.set_text(textwrap.fill(_simplify(t), 22))


def _points_in(ax, box) -> int:
    """Plotted data points (line vertices, scatter offsets, bar rectangles) whose display position falls in ``box``."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    def inside(pts) -> int:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        ok = np.isfinite(pts).all(axis=1)
        x, y = pts[ok, 0], pts[ok, 1]
        return int(((x >= box.x0) & (x <= box.x1) & (y >= box.y0) & (y <= box.y1)).sum())

    n = 0
    for a in ax.get_children():
        if isinstance(a, Line2D) and a.get_visible() and len(a.get_xydata()):
            if a.get_label().startswith("_") and len(a.get_xydata()) <= 2:  # reference lines (axhline) are fine
                continue
            xy = a.get_transform().transform(a.get_xydata())
            if a.get_linestyle() not in ("None", "", " ") and len(xy) > 1:  # densify: a segment under the box counts
                t = np.linspace(0, 1, 25)[:, None]
                xy = np.vstack([xy[i] + t * (xy[i + 1] - xy[i]) for i in range(len(xy) - 1)])
            n += inside(xy)
        elif isinstance(a, Collection) and a.get_visible() and len(a.get_offsets()):
            n += inside(a.get_offset_transform().transform(a.get_offsets()))
        elif isinstance(a, Rectangle) and a.get_visible() and a is not ax.patch and a.get_height() and a.get_width():
            n += int(a.get_window_extent().overlaps(box))
    return n


def text_overlaps(fig) -> list[tuple]:
    """(axes, text, points under it) for notes placed in axes coordinates that sit over plotted data."""
    fig.canvas.draw()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_transform() is ax.transAxes and t.get_visible() and t.get_text().strip():
                k = _points_in(ax, t.get_window_extent())
                if k:
                    out.append((ax, t, k))
    return out


def legend_overlaps(fig) -> list[tuple]:
    """(axes, legend, points under it) for every axes legend that sits over plotted data."""
    fig.canvas.draw()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        k = _points_in(ax, leg.get_window_extent())
        if k:
            out.append((ax, leg, k))
    return out


def fix_legends(fig, key: str, preset: str) -> None:
    """Move any legend that covers plotted data below its plot (ncol up to 3); log what was found and fixed."""
    found = legend_overlaps(fig)
    for ax, _leg, _k in found:
        handles, labels = ax.get_legend_handles_labels()
        longest = max((len(x) for x in labels), default=0)
        ncol = 1 if longest > 38 else 2 if longest > 18 else 3
        ncol = min(ncol, len(labels))
        rows = -(-len(labels) // ncol)
        # grow the figure by the legend's height so the plot keeps its size, then hang the legend below the axes
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() + rows * PRESETS[preset]["legend"] * 1.5 / 72)
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol, frameon=False,
                  prop={"family": font_family(), "size": PRESETS[preset]["legend"]})
    notes = text_overlaps(fig)
    for _ax, t, _k in notes:  # an axes note over the data moves just above the plot (titles are removed)
        t.set_position((0.0, 1.02))
        t.set_ha("left")
        t.set_va("bottom")
    if found or notes:
        fig.tight_layout(pad=0.4)
    left = legend_overlaps(fig) + text_overlaps(fig) if (found or notes) else []
    if any(ax.get_legend() is not None for ax in fig.axes) or notes:
        LEGEND_LOG.append({"key": key, "preset": preset, "overlaps_found": len(found) + len(notes),
                           "points_covered": sum(k for *_, k in found), "overlaps_after_fix": len(left)})


def theme(fig, preset: str, max_aspect: float | None = None) -> None:
    """Restyle a drawn figure in place for a preset (see module docstring)."""
    from tzbank.chapters.ch04.common import year_ticks

    p = PRESETS[preset]
    if getattr(fig, "_house_diagram", False):  # diagrams: drawn at their placed size; only the font family changes
        w0, h0 = fig.get_size_inches()
        w = p["width"] * CM
        if abs(w / w0 - 1) > 0.02:  # placed at another width: scale the drawing, never the text below the minimum
            fig.set_size_inches(w, w * h0 / w0)
            for t in fig.findobj(Text):
                t.set_fontsize(max(t.get_fontsize() * w / w0, MIN_PT["diagram"]))
        for t in fig.findobj(Text):
            t.set_fontfamily(font_family())
        return
    axes = [a for a in fig.axes if a.get_subplotspec() is not None and a.get_label() != "<colorbar>"]
    w0, h0 = fig.get_size_inches()
    nrows = len({a.get_subplotspec().rowspan.start for a in axes}) or 1
    if preset != "full":
        nrows = _stack(fig, axes)
    lo, hi = p["aspect"]
    if max_aspect:  # a shorter chart (sidebar fitting, prompt 17 A1)
        hi = min(hi, max_aspect)
        lo = min(lo, hi)
    per_row = min(max(h0 / w0 * (1 if preset == "full" else max(1, len(axes)) / max(1, nrows)), lo), hi)
    if preset != "full" and nrows > 1:
        per_row = lo * 0.8
    w = p["width"] * CM
    fig.set_size_inches(w, w * per_row * (1 if preset == "full" else nrows))
    if fig._suptitle is not None:  # noqa: SLF001
        fig._suptitle.set_text("")  # noqa: SLF001
    for t in list(fig.texts):
        t.remove()
    multi = len(axes) > 1
    for ax in fig.axes:
        if not multi:
            ax.set_title("")
            ax.set_title("", loc="left")
        else:
            for loc in ("left", "center"):
                s = ax.get_title(loc=loc)
                if s:
                    ax.set_title(_simplify(s) if preset != "full" else s, loc=loc, fontsize=p["label"],
                                 fontweight="bold", color="#252B28")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(SPINE)
        ax.tick_params(labelsize=p["tick"], colors=TICK, length=2.5)
        ax.xaxis.label.set_size(p["label"])
        ax.yaxis.label.set_size(p["label"])
        for t in ax.texts:
            t.set_fontsize(min(max(t.get_fontsize(), p["text"][0]), p["text"][1]))
            if preset != "full" and len(t.get_text()) > 40 and chr(10) not in t.get_text():
                t.set_text(textwrap.fill(t.get_text(), 38 if preset == "half" else 30))
    peer_i: dict = {}
    _recolour(fig, peer_i)
    _legends(fig, p, preset, peer_i)
    if preset != "full":
        _fit_labels(fig, p)
    fam = font_family()
    for t in fig.findobj(Text):
        t.set_fontfamily(fam)
    fig.patch.set_facecolor("white")
    year_ticks(fig)
    if preset != "full":
        for ax in fig.axes:  # fewer year ticks at small widths
            if isinstance(ax.xaxis.get_major_locator(), MaxNLocator):
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4 if preset == "side" else 5, integer=True,
                                                       steps=[1, 2, 5, 10]))
        # room for the legends below each plot: grow the figure by their height
        extra = sum(len(leg.get_texts()) / max(leg._ncols, 1) + 0.5 for ax in fig.axes  # noqa: SLF001
                    if (leg := ax.get_legend()) is not None)
        extra += sum(len(leg.get_texts()) / max(leg._ncols, 1) + 0.5 for leg in fig.legends)  # noqa: SLF001
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() + extra * p["legend"] * 1.35 / 72)
    rows = 0
    if fig.legends:
        leg = fig.legends[0]
        rows = -(-len(leg.get_texts()) // max(leg._ncols, 1))  # noqa: SLF001
    bottom = rows * p["legend"] * 1.45 / 72 / fig.get_figheight() + (0.03 if rows else 0)
    fig.tight_layout(pad=0.4, rect=(0, bottom, 1, 1))
    if preset != "full":
        _place_legends(fig, bottom)


def bank_panels(fig) -> dict:
    """{ticker: axes} when the figure has one panel per bank (each panel title names exactly one of NMB, CRDB)."""
    out = {}
    axes = [a for a in fig.axes if a.get_subplotspec() is not None]
    for ax in axes:
        t = " ".join(ax.get_title(loc=loc) for loc in ("left", "center"))
        named = [b for b in ("NMB", "CRDB") if re.search(rf"\b{b}\b", t)]
        if len(named) == 1:
            out.setdefault(named[0], ax)
    return out if len(out) >= 2 else {}


def select_panel(fig, ticker: str) -> bool:
    """Keep only the bank's own panel of a per-bank figure (resized to one panel's width). False if not per-bank."""
    panels = bank_panels(fig)
    if ticker not in panels:
        return False
    keep = panels[ticker]
    axes = [a for a in fig.axes if a.get_subplotspec() is not None]
    ncols = keep.get_subplotspec().get_gridspec().ncols
    for a in axes:
        if a is not keep:
            fig.delaxes(a)
    keep.set_subplotspec(GridSpec(1, 1, figure=fig)[0])
    keep.yaxis.set_tick_params(labelleft=True)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w / max(ncols, 1), h)
    return True


def content_texts(fig) -> list:
    """Visible text that carries content: axes texts and annotations, legend entries (titles and footnotes are
    removed by the theme by design; the report caption replaces them)."""
    out = []
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend() is not None:
            out += [t for t in ax.get_legend().get_texts() if t.get_text().strip()]
    for leg in fig.legends:
        out += [t for t in leg.get_texts() if t.get_text().strip()]
    return out


def check_text(fig, key: str, preset: str, n0: int) -> None:
    """Prompt 22 build check: every content text of the snapshot survives, and no visible text is below the minimum
    size at the placed width (8 pt in diagrams, 7 pt elsewhere). Raises ValueError (the build fails)."""
    kind = "diagram" if getattr(fig, "_house_diagram", False) else "chart"
    vis = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    small = min((t.get_fontsize() for t in vis), default=99.0)
    n1 = len(content_texts(fig))
    ok = n1 == n0 and small >= MIN_PT[kind] - 1e-9
    TEXT_LOG.append({"key": key, "preset": preset, "kind": kind, "snapshot_texts": n0, "report_texts": n1,
                     "min_pt": round(small, 2), "min_required": MIN_PT[kind], "ok": ok})
    if not ok:
        raise ValueError(f"{key} ({preset}): {n0 - n1} content text(s) lost or text below {MIN_PT[kind]} pt "
                         f"(smallest {small:.1f} pt)")


def render(key: str, preset: str, src_dir: Path, out_dir: Path, panel: str | None = None,
           max_aspect: float | None = None) -> Path:
    """Theme one snapshot for one preset; returns ``out_dir/<key>_<preset>.png`` (300 dpi). With ``panel`` (a
    ticker) a per-bank figure keeps only that bank's panel and is written as ``<key>_<ticker>_<preset>.png``."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    src = Path(src_dir) / f"{key}.pickle"
    if not src.exists():
        raise FileNotFoundError(f"no chart snapshot for {key!r} ({src}); re-run the stage that draws it")
    fig = pickle.loads(src.read_bytes())  # noqa: S301 - our own pipeline output
    picked = bool(panel) and select_panel(fig, panel)
    n0 = len(content_texts(fig))
    theme(fig, preset, max_aspect)
    if not getattr(fig, "_house_diagram", False):
        fix_legends(fig, key, preset)
    check_text(fig, key, preset, n0)
    stem = f"{key}_{panel}_{preset}" if picked else f"{key}_{preset}"
    out = Path(out_dir) / (f"{stem}_a{round(max_aspect * 100)}.png" if max_aspect else f"{stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return out
