"""Shared helpers for the chapter 4 subsections (blueprint §4; CLAUDE.md §5-§7): panel loading, averages, figure
style, manifests and notes. No analysis logic lives here beyond average balances."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter  # noqa: E402

from tzbank.config import banks, find_root, path  # noqa: E402
from tzbank.io import write_json, write_text  # noqa: E402

FOCUS = ["B01", "B02"]
COLORS = {"B01": "#2a78d6", "B02": "#eb6834", "B03": "#1baf7a", "B04": "#eda100", "B05": "#e87ba4",
          "B06": "#008300", "B07": "#6250d6"}
INK, MUTED, GRID = "#1f1f1d", "#5a5a55", "#e4e3dd"


def tickers(root: Path | None = None) -> dict[str, str]:
    return {b["bank_id"]: b["ticker"] for b in banks(root or find_root())}


def wide(root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path("processed", root or find_root()) / "bank_year_wide.csv", dtype={"bank_id": str},
                       low_memory=False).sort_values(["bank_id", "fiscal_year"]).reset_index(drop=True)


def col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)


def lag(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Prior-year value for the same bank, only when the prior year is exactly t-1."""
    prev = s.groupby(df["bank_id"]).shift(1)
    gap = df["fiscal_year"] - df.groupby("bank_id")["fiscal_year"].shift(1)
    return prev.where(gap == 1)


def avg(df: pd.DataFrame, c: str) -> pd.Series:
    """(opening + closing) / 2; NaN unless both exist (blueprint §2; ch04 starter convention)."""
    s = col(df, c)
    return (s + lag(df, s)) / 2


def div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)


def _year_label(v, _pos) -> str:
    return f"{v:.0f}"


def year_ticks(fig) -> None:
    """Integer year ticks on every axis whose x-range is a span of fiscal years (1990-2045); date axes (matplotlib
    day numbers) and categorical axes are left alone (figure fix, prompt 15c)."""
    for ax in fig.axes:
        lo, hi = ax.get_xlim()
        if ax.has_data() and 1990 <= lo and hi <= 2045 and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(FuncFormatter(_year_label))


def save(fig, png: Path, footnote: str) -> Path:
    from tzbank.reporting.chart_theme import snapshot

    snapshot(fig, png)  # untitled-footnote copy for the report theme (prompt 16); the dashboard PNG is unchanged
    year_ticks(fig)
    width = int(fig.get_figwidth() * 20)  # characters at 6.5 pt that fit the figure width (with margin)
    lines = textwrap.wrap(footnote, width) if len(footnote) > width else [footnote]
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=6.5, color=MUTED)
    extra = 0.1 / fig.get_figheight() * (len(lines) - 1)  # ~0.1 in per extra footnote line
    fig.tight_layout(rect=(0, (0.12 if fig.legends else 0.04) + extra, 1, 1))  # room for a figure-level legend
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, metadata={"Software": None})
    plt.close(fig)
    return png


def manifest(root: Path, sub: str, title: str, tables: list[dict], figures: list[dict], note: Path,
             module: str, inputs: list[str], status: str) -> Path:
    """outputs/manifests/ch04_<sub>.json (CLAUDE.md §7: the dashboard reads only this)."""
    rel = lambda p: p.relative_to(root).as_posix() if isinstance(p, Path) else p  # noqa: E731
    m = {"chapter": "04", "subsection": sub, "title": title, "hypotheses": ["H4.1", "H4.2"], "status": status,
         "tables": [t | {"path": rel(t["path"]), "note": rel(note)} for t in tables],
         "figures": [f | {"path": rel(f["path"]), "data": rel(f["data"]), "note": rel(note)} for f in figures],
         "notes": [rel(note)], "inputs": inputs, "module": module, "run_log_stage": "profitability"}
    return write_json(m, path("manifests", root) / f"ch04_{sub.replace('.', '_')}.json")


def note(root: Path, sub: str, text: str) -> Path:
    return write_text(text, path("notes", root) / f"ch04_{sub.replace('.', '_')}.md")


def plt_subplots(*a, **k):
    return plt.subplots(*a, **k)


def pct(v, digits: int = 1, sign: bool = False, reason: str = "not held") -> str:
    """Percent for notes; a missing value prints as 'n/a (reason)', never 'nan%' (prompt 03b fix 2)."""
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isinf(v))) or pd.isna(v):
        return f"n/a ({reason})"
    return f"{v:{'+' if sign else ''}.{digits}%}"


def num(v, fmt: str = ",.0f", reason: str = "not held") -> str:
    """Number for notes with the same missing-value rule as ``pct``."""
    if v is None or pd.isna(v):
        return f"n/a ({reason})"
    return format(v, fmt)
