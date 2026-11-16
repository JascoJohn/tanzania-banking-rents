"""Dashboard (Output C) checks: every page runs for every bank selection; every number drawn is in the frame the
download button writes; every number in that frame is a value of a published table (CLAUDE.md section 7)."""

from __future__ import annotations

import base64
import io
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tzbank.config import find_root  # noqa: E402

ROOT = find_root()
DASH = ROOT / "dashboard"
sys.path.insert(0, str(DASH))
PAGES = sorted(p.relative_to(DASH).as_posix() for p in (DASH / "pages").rglob("*.py"))
SELECTIONS = ("Both", "NMB", "CRDB")


def _run(page: str, pick: str | None = None) -> AppTest:
    at = AppTest.from_file(str(DASH / page), default_timeout=120)
    at.session_state["_record"] = True
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    if pick:
        for sc in [w for w in at.get("button_group") if str(w.key or "").startswith("bank2_")]:
            sc.set_value(pick)
        at.run()
        assert not at.exception, [e.message for e in at.exception]
    return at


def _numbers(v) -> list[float]:
    """Numeric values in a plotly JSON field (lists, numpy arrays or plotly's base64 typed arrays)."""
    if isinstance(v, dict) and "bdata" in v:
        arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        return [float(x) for x in arr.ravel() if np.isfinite(x)]  # gaps are drawn as gaps
    if isinstance(v, np.ndarray):
        v = v.ravel().tolist()
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in _numbers(item)]
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and np.isfinite(v):
        return [float(v)]
    return []


def _in(values, pool: np.ndarray) -> list[float]:
    """Values not found in ``pool`` (relative tolerance 1e-9)."""
    pool = np.sort(pool)
    out = []
    for v in values:
        i = np.searchsorted(pool, v)
        near = pool[max(i - 1, 0): i + 1]
        if not len(near) or not np.isclose(near, v, rtol=1e-9, atol=1e-12).any():
            out.append(v)
    return out


def _table_numbers(paths: list[str]) -> np.ndarray:
    vals = []
    for p in paths:
        d = pd.read_csv(ROOT / p, low_memory=False)
        vals.append(d.select_dtypes("number").to_numpy(dtype=float).ravel())
    a = np.concatenate(vals) if vals else np.array([])
    return a[np.isfinite(a)]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("pick", SELECTIONS)
def test_page_runs_for_every_bank_selection(page: str, pick: str) -> None:
    _run(page, pick)


@pytest.mark.parametrize("page", PAGES)
def test_numbers_trace_to_published_tables(page: str) -> None:
    at = _run(page)
    shown = at.session_state["_shown"] if "_shown" in at.session_state else {}
    figs = at.session_state["_figs"] if "_figs" in at.session_state else {}
    for key, (payload, sources) in shown.items():
        frame = pd.read_csv(io.BytesIO(payload))
        nums = frame.select_dtypes("number")
        nums = nums.drop(columns=[c for c in nums if c in ("year", "fiscal_year", "fy")]).to_numpy(float).ravel()
        nums = nums[np.isfinite(nums)]
        missing = _in(nums, _table_numbers(sources))
        assert not missing, f"{page} / {key}: {missing[:5]} not in {sources}"
        # the chart draws only numbers that are in the downloaded frame (0 stands in for a missing bar)
        if key in figs:
            pool = np.concatenate([frame.select_dtypes("number").to_numpy(float).ravel(), [0.0]])
            drawn = [x for tr in figs[key] for f in ("y", "z", "q1", "median", "q3", "lowerfence", "upperfence",
                                                      "mean") for x in _numbers(tr.get(f))]
            if tr_x := [x for tr in figs[key] if tr.get("type") in ("heatmap", "contour") for x in _numbers(tr.get("x"))]:
                drawn += tr_x
            missing = _in(drawn, pool[np.isfinite(pool)])
            assert not missing, f"{page} / {key}: drawn {missing[:5]} not in the download"


def test_download_matches_display_for_a_panel_chart() -> None:
    """Spot check: the chapter 4 chart for NMB and CRDB equals the published ROE rows, and the download equals it."""
    at = _run("pages/04_banking_economics/04_0_overview.py")
    payload, sources = at.session_state["_shown"]["ch04"]
    frame = pd.read_csv(io.BytesIO(payload))
    pub = pd.read_csv(ROOT / "outputs/tables/ch04_dupont_panel.csv")
    pub = pub[pub["bank_id"].isin(["B01", "B02"]) & pub["roe"].notna()]
    assert sources == ["outputs/tables/ch04_dupont_panel.csv"]
    assert sorted(frame["roe"].dropna().round(12)) == sorted(pub["roe"].round(12))
    drawn = sorted(x for tr in at.session_state["_figs"]["ch04"] for x in _numbers(tr["y"]))
    assert np.allclose(drawn, sorted(frame["roe"].dropna()))
