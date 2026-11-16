"""Persistence estimators (blueprint §2 "Persistence model"; CLAUDE.md §8 / D12 guardrails).

ER_t = alpha + lambda ER_{t-1} + e_t,  ER* = alpha / (1 - lambda),  half-life = ln 0.5 / ln lambda.

The AR(1) is a descriptive benchmark. Uncertainty comes from a moving-block bootstrap of the (ER_t, ER_{t-1}) pairs
(blocks of consecutive pairs, percentile intervals, fixed seed). A half-life is quoted only when 0 < lambda < 1 and the
interval excludes 1. Break tests: known-date Chow F tests (intercept and slope shift) and a sup-F test whose p-value
comes from a parametric bootstrap of the fitted AR(1) under the null of no break (small samples; no asymptotic
tables). Pooled fixed effects are secondary (Nickell bias).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def pairs(er: pd.Series, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """(year, y = ER_t, x = ER_{t-1}) for consecutive fiscal years only; ``er`` is indexed by fiscal year."""
    s = er.dropna().sort_index()
    prev = s.reindex(s.index - 1)
    d = pd.DataFrame({"year": s.index.astype(int), "y": s.values, "x": prev.values}).dropna().reset_index(drop=True)
    if extra is not None:
        d = d.join(extra, on="year")
    return d


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficients and residual sum of squares (NaN coefficients if the design is rank-deficient)."""
    if X.shape[0] <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return np.full(X.shape[1], np.nan), float("nan")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, float(e @ e)


def design(d: pd.DataFrame, dummies: list[str] | None = None) -> np.ndarray:
    cols = [np.ones(len(d)), d["x"].to_numpy()] + [d[c].to_numpy(float) for c in (dummies or [])]
    return np.column_stack(cols)


def ar1(d: pd.DataFrame, dummies: list[str] | None = None, at: list[float] | None = None) -> dict[str, float]:
    """AR(1) by OLS; with intercept dummies, ER* is evaluated in the regime given by ``at`` (default all zero)."""
    b, rss = ols(design(d, dummies), d["y"].to_numpy())
    alpha = _alpha(b, at)
    return {"alpha": alpha, "lambda": b[1], "er_star": er_star(alpha, b[1]), "rss": rss, "n": len(d)}


def _alpha(b: np.ndarray, at: list[float] | None) -> float:
    """Intercept of the chosen regime: alpha + sum(delta_k * at_k)."""
    return float(b[0] + (np.dot(b[2:], at) if at is not None and len(b) > 2 else 0.0))


def er_star(alpha: float, lam: float) -> float:
    return alpha / (1 - lam) if np.isfinite(lam) and lam < 1 else float("nan")


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index: ceil(n / block) blocks of consecutive rows, truncated to n."""
    block = max(1, min(block, n))
    k = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=k)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap(d: pd.DataFrame, reps: int, block: int, seed: int | list[int],
              dummies: list[str] | None = None, at: list[float] | None = None) -> pd.DataFrame:
    """Replicates of (lambda, ER*) from the moving-block bootstrap of the pairs (reproducible with the seed)."""
    rng = np.random.default_rng(seed)
    y, X = d["y"].to_numpy(), design(d, dummies)
    out = np.full((reps, 2), np.nan)
    for i in range(reps):
        ix = block_indices(len(d), block, rng)
        b, _ = ols(X[ix], y[ix])
        out[i] = b[1], er_star(_alpha(b, at), b[1])
    return pd.DataFrame(out, columns=["lambda", "er_star"])


def interval(v: pd.Series, level: float) -> tuple[float, float]:
    v = v.dropna()
    if len(v) < 20:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(v.quantile(a)), float(v.quantile(1 - a))


def half_life(lam: float, hi: float) -> tuple[float, str]:
    """Blueprint §2 validity rule: quoted only for 0 < lambda < 1 with an interval that excludes 1."""
    if not np.isfinite(lam):
        return float("nan"), "not estimable"
    if lam <= 0:
        return float("nan"), "not quoted: lambda <= 0 (no positive persistence)"
    if lam >= 1:
        return float("nan"), "not quoted: lambda >= 1 (mean reversion not established)"
    if not np.isfinite(hi) or hi >= 1:
        return float("nan"), "not quoted: interval includes 1 (mean reversion not established)"
    return math.log(0.5) / math.log(lam), "valid (0 < lambda < 1, interval excludes 1)"


def estimate(d: pd.DataFrame, reps: int, block: int, level: float, seed, dummies: list[str] | None = None,
             min_pairs: int = 6, at: list[float] | None = None) -> dict:
    """Point estimates, bootstrap intervals and the half-life with its validity status."""
    if len(d) < min_pairs:
        return {"n": len(d), "status": f"not estimated: {len(d)} pairs < {min_pairs}"}
    p = ar1(d, dummies, at)
    bs = bootstrap(d, reps, block, seed, dummies, at)
    lo, hi = interval(bs["lambda"], level)
    elo, ehi = interval(bs["er_star"], level)
    hl, hl_status = half_life(p["lambda"], hi)
    return {"n": p["n"], "years": f"{int(d['year'].min())}-{int(d['year'].max())}", "alpha": p["alpha"],
            "lambda": p["lambda"], "lambda_lo": lo, "lambda_hi": hi, "er_star": p["er_star"], "er_star_lo": elo,
            "er_star_hi": ehi, "er_star_valid_share": float(bs["er_star"].notna().mean()), "half_life": hl,
            "half_life_status": hl_status, "status": "estimated"}


def chow(d: pd.DataFrame, regime: np.ndarray) -> dict[str, float]:
    """F test that intercept and slope differ in the ``regime`` rows (k = 2 restrictions)."""
    regime = np.asarray(regime, bool)
    n = len(d)
    if regime.sum() < 2 or (~regime).sum() < 2 or n - 4 < 1:
        return {"F": float("nan"), "p_value": float("nan"), "df1": 2, "df2": n - 4,
                "status": "not computable: fewer than 2 pairs in a regime"}
    y = d["y"].to_numpy()
    _, rss_r = ols(design(d), y)
    X_u = np.column_stack([design(d), regime.astype(float), regime * d["x"].to_numpy()])
    _, rss_u = ols(X_u, y)
    if not np.isfinite(rss_u) or rss_u <= 0:
        return {"F": float("nan"), "p_value": float("nan"), "df1": 2, "df2": n - 4,
                "status": "not computable: unrestricted model rank-deficient or exact fit"}
    f = ((rss_r - rss_u) / 2) / (rss_u / (n - 4))
    return {"F": f, "p_value": float(stats.f.sf(f, 2, n - 4)), "df1": 2, "df2": n - 4, "status": "computed"}


def sup_f_stat(d: pd.DataFrame, min_regime: int) -> tuple[float, int | None]:
    best, at = float("nan"), None
    yrs = d["year"].to_numpy()
    for k in range(min_regime, len(d) - min_regime + 1):
        f = chow(d, yrs >= yrs[k])["F"]
        if np.isfinite(f) and (not np.isfinite(best) or f > best):
            best, at = f, int(yrs[k])
    return best, at


def sup_f(d: pd.DataFrame, reps: int, min_regime: int, seed) -> dict:
    """sup-F over break dates with ``min_regime`` pairs per side; p-value by parametric bootstrap under no break."""
    stat, at = sup_f_stat(d, min_regime)
    if not np.isfinite(stat):
        return {"F": float("nan"), "break_year": None, "p_value": float("nan")}
    p = ar1(d)
    b = np.array([p["alpha"], p["lambda"]])
    e = d["y"].to_numpy() - design(d) @ b
    e = e - e.mean()
    rng = np.random.default_rng(seed)
    exceed, done = 0, 0
    for _ in range(reps):
        y0 = d["x"].iloc[0]
        sim = []
        for es in rng.choice(e, size=len(d)):
            y1 = b[0] + b[1] * y0 + es
            sim.append((y1, y0))
            y0 = y1
        ds = pd.DataFrame(sim, columns=["y", "x"]).assign(year=d["year"].to_numpy())
        s, _ = sup_f_stat(ds, min_regime)
        if np.isfinite(s):
            done += 1
            exceed += s >= stat
    return {"F": stat, "break_year": at, "p_value": (exceed + 1) / (done + 1) if done else float("nan")}


def pooled_fe(panel: pd.DataFrame, level: float) -> dict:
    """Within (fixed-effects) AR(1) over bank pairs: secondary, biased towards zero in short panels (Nickell)."""
    g = panel.groupby("bank_id")
    yd = panel["y"] - g["y"].transform("mean")
    xd = panel["x"] - g["x"].transform("mean")
    lam = float((xd * yd).sum() / (xd * xd).sum())
    e = yd - lam * xd
    n, k = len(panel), panel["bank_id"].nunique()
    se = math.sqrt(float(e @ e) / (n - k - 1) / float(xd @ xd))
    t = stats.t.ppf(1 - (1 - level) / 2, n - k - 1)
    tbar = n / k
    return {"n": n, "banks": k, "lambda": lam, "lambda_lo": lam - t * se, "lambda_hi": lam + t * se,
            "nickell_bias_approx": -(1 + lam) / (tbar - 1)}


def sign_persistence(er: pd.Series) -> dict[str, float]:
    s = er.dropna().sort_index()
    pos = s > 0
    prev = pos.reindex(s.index - 1)
    both = prev.notna().to_numpy()
    stay = (pos.to_numpy()[both] & prev.to_numpy()[both].astype(bool))
    n_prev_pos = int(prev.to_numpy()[both].astype(bool).sum())
    return {"n_years": len(s), "share_positive": float(pos.mean()) if len(s) else float("nan"),
            "p_positive_after_positive": float(stay.sum() / n_prev_pos) if n_prev_pos else float("nan"),
            "n_transitions_from_positive": n_prev_pos}


def rank_persistence(er: pd.DataFrame, lag: int = 1, min_banks: int = 4) -> pd.DataFrame:
    """Spearman correlation of bank ER across years t and t - lag (banks present in both years)."""
    w = er.pivot(index="fiscal_year", columns="bank_id", values="er")
    rows = []
    for t in w.index:
        if t - lag not in w.index:
            continue
        a, b = w.loc[t], w.loc[t - lag]
        ok = a.notna() & b.notna()
        if ok.sum() >= min_banks:
            rho = stats.spearmanr(a[ok], b[ok]).statistic
            rows.append({"year": int(t), "lag": lag, "n_banks": int(ok.sum()), "spearman": float(rho)})
    return pd.DataFrame(rows)


def simulated_coverage(lam: float, n_pairs: int, sims: int, reps: int, block: int, level: float, seed: int) -> float:
    """Share of simulated AR(1) samples (``n_pairs`` pairs, true ``lam``) whose bootstrap interval contains ``lam``:
    a check of the interval's small-sample coverage, reported in the chapter 5 limitations."""
    rng = np.random.default_rng([seed, 777])
    hits = 0
    for s in range(sims):
        y = [0.0]
        for _ in range(n_pairs + 50):
            y.append(0.01 + lam * y[-1] + rng.normal(0, 0.03))
        d = pairs(pd.Series(y[-(n_pairs + 1):], index=range(2000, 2001 + n_pairs)))
        r = estimate(d, reps, block, level, [seed, 778, s])
        hits += r["lambda_lo"] < lam < r["lambda_hi"]
    return hits / sims


# --- small-sample inference: Kilian (1998) and Andrews (1993) (prompt 05 pre-flight 2) ---------------------------


def _batch_ols(Y: np.ndarray, X: np.ndarray, D: np.ndarray | None) -> np.ndarray:
    """OLS of each row of Y on [1, X_row, D] (D shared across rows); returns (B, k) coefficients [a, lambda, d...]."""
    B, n = Y.shape
    cols = [np.ones((B, n)), X]
    if D is not None and D.shape[1]:
        cols += [np.broadcast_to(D[:, j], (B, n)) for j in range(D.shape[1])]
    Z = np.stack(cols, axis=2)  # (B, n, k)
    ZtZ = np.einsum("bnk,bnl->bkl", Z, Z)
    Zty = np.einsum("bnk,bn->bk", Z, Y)
    out = np.full((B, Z.shape[2]), np.nan)
    ok = np.abs(np.linalg.det(ZtZ)) > 1e-14
    if ok.any():
        out[ok] = np.linalg.solve(ZtZ[ok], Zty[ok][..., None])[..., 0]
    return out


def _simulate(c: np.ndarray, lam: float, e: np.ndarray, y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y_t = c_t + lam y_{t-1} + e_t for B rows; returns (Y, X = lagged Y)."""
    B, n = e.shape
    Y = np.empty((B, n))
    prev = y0
    for t in range(n):
        prev = c[t] + lam * prev + e[:, t]
        Y[:, t] = prev
    X = np.column_stack([y0, Y[:, :-1]])
    return Y, X


def _dummy_matrix(d: pd.DataFrame, dummies: list[str] | None) -> np.ndarray | None:
    return d[dummies].to_numpy(float) if dummies else None


def _intercepts(y: np.ndarray, x: np.ndarray, lam: float, D: np.ndarray | None) -> np.ndarray:
    """Intercept (and dummy shifts) given lambda: OLS of y - lam x on [1, D]."""
    Z = np.column_stack([np.ones(len(y))] + ([D] if D is not None else []))
    b, *_ = np.linalg.lstsq(Z, y - lam * x, rcond=None)
    return b


def _star(b: np.ndarray, lam: float, at: list[float] | None) -> float:
    a = b[0] + (float(np.dot(b[1:], at)) if at is not None and len(b) > 1 else 0.0)
    return a / (1 - lam) if lam < 1 else float("nan")


def _bias_adjust(lam_hat: float, bias: float) -> float:
    """Kilian (1998): no correction if lam_hat >= 1; shrink the correction until the corrected value is below 1."""
    if not np.isfinite(lam_hat) or lam_hat >= 1:
        return lam_hat
    b = bias
    for _ in range(1000):
        if lam_hat - b < 1:
            break
        b *= 0.99
    return lam_hat - b


def _kilian_core(d: pd.DataFrame, b1: int, b2: int, seed, dummies: list[str] | None = None,
                 at: list[float] | None = None) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Kilian (1998) stages; returns (lam_bc, bias, intercepts at lam_bc, corrected second-stage lambdas, their ER*)."""
    rng = np.random.default_rng(seed)
    y, x = d["y"].to_numpy(), d["x"].to_numpy()
    D = _dummy_matrix(d, dummies)
    n = len(y)
    p = ols(design(d, dummies), y)[0]
    lam = p[1]
    c = p[0] + (D @ p[2:] if D is not None else np.zeros(n))
    e = y - (c + lam * x)
    e = e - e.mean()
    Y, X = _simulate(c, lam, rng.choice(e, size=(b1, n)), np.full(b1, x[0]))
    bias = float(np.nanmean(_batch_ols(Y, X, D)[:, 1]) - lam)
    lam_bc = _bias_adjust(lam, bias)
    bb = _intercepts(y, x, lam_bc, D)
    c2 = bb[0] + (D @ bb[1:] if D is not None else np.zeros(n))
    e2 = y - (c2 + lam_bc * x)
    e2 = e2 - e2.mean()
    Y2, X2 = _simulate(c2, lam_bc, rng.choice(e2, size=(b2, n)), np.full(b2, x[0]))
    coef = _batch_ols(Y2, X2, D)
    lams = np.array([_bias_adjust(v, bias) for v in coef[:, 1]])
    stars = np.full(b2, np.nan)
    for i in range(b2):
        if np.isfinite(lams[i]) and lams[i] < 1:
            stars[i] = _star(_intercepts(Y2[i], X2[i], lams[i], D), lams[i], at)
    return lam_bc, bias, bb, lams, stars


def kilian(d: pd.DataFrame, b1: int, b2: int, level: float, seed, dummies: list[str] | None = None,
           at: list[float] | None = None) -> dict:
    """Bootstrap-after-bootstrap (Kilian 1998): a residual bootstrap from the OLS fit gives the bias; the bias-corrected
    model generates the second-stage samples, each bias-corrected; percentile interval of the corrected lambdas."""
    lam_bc, bias, bb, lams, stars = _kilian_core(d, b1, b2, seed, dummies, at)
    lo, hi = interval(pd.Series(lams), level)
    elo, ehi = interval(pd.Series(stars), level)
    return {"lambda": lam_bc, "lambda_lo": lo, "lambda_hi": hi, "er_star": _star(bb, lam_bc, at), "er_star_lo": elo,
            "er_star_hi": ehi, "bias": bias}


def kilian_draws(d: pd.DataFrame, b1: int, b2: int, seed, dummies: list[str] | None = None,
                 at: list[float] | None = None) -> pd.DataFrame:
    """The Kilian (1998) second-stage replicates as joint (lambda, ER*) draws (D58 amended, owner issue 25): each row
    is one bias-corrected bootstrap sample's lambda and the ER* implied by its intercepts."""
    _, _, _, lams, stars = _kilian_core(d, b1, b2, seed, dummies, at)
    return pd.DataFrame({"lambda": lams, "er_star": stars})


_ANDREWS_CACHE: dict = {}
GRID = np.round(np.arange(-0.5, 1.0001, 0.01), 2)


def andrews_quantiles(n: int, shift: np.ndarray | None, sims: int, seed) -> np.ndarray:
    """Quantiles (5%, 50%, 95%) of the OLS lambda for each grid lambda; Gaussian errors (sigma = 1), stationary start
    (lambda < 1) or y0 = 0 (lambda = 1). ``shift`` is the dummy column times delta/sigma (plug-in), or None."""
    key = (n, None if shift is None else tuple(np.round(shift, 6)), sims, str(seed))
    if key in _ANDREWS_CACHE:
        return _ANDREWS_CACHE[key]
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((sims, n))
    Z0 = rng.standard_normal(sims)
    D = None if shift is None else (shift != 0).astype(float)[:, None]
    q = np.empty((len(GRID), 3))
    for i, lam in enumerate(GRID):
        c = np.zeros(n) if shift is None else shift
        y0 = Z0 / np.sqrt(1 - lam ** 2) if lam < 1 else np.zeros(sims)
        Y, X = _simulate(c, lam, E, y0)
        q[i] = np.nanquantile(_batch_ols(Y, X, D)[:, 1], [0.05, 0.5, 0.95])
    q = np.maximum.accumulate(q, axis=0)  # quantile functions are non-decreasing in lambda
    _ANDREWS_CACHE[key] = q
    return q


def _invert(f: np.ndarray, v: float) -> float:
    if v <= f[0]:
        return float(GRID[0])
    if v >= f[-1]:
        return float(GRID[-1])
    return float(np.interp(v, f, GRID))


def andrews(d: pd.DataFrame, level: float, sims: int, seed, dummies: list[str] | None = None,
            at: list[float] | None = None) -> dict:
    """Median-unbiased AR(1) (Andrews 1993) by inverting simulated quantile functions of the OLS estimator on a grid
    lambda in [-0.5, 1]; with one intercept dummy the delta/sigma ratio is plugged in (approximately median-unbiased).
    ER* uses the median-unbiased lambda; the method defines no ER* interval."""
    if abs(level - 0.90) > 1e-9:
        raise ValueError("Andrews intervals here use the 5% and 95% quantile functions (90% level)")
    y, x = d["y"].to_numpy(), d["x"].to_numpy()
    D = _dummy_matrix(d, dummies)
    Zd = design(d, dummies)
    p, rss = ols(Zd, y)
    shift = None
    if D is not None:
        sig = np.sqrt(rss / max(len(y) - Zd.shape[1], 1))
        shift = D[:, 0] * (p[2] / sig if sig > 0 else 0.0)
    q = andrews_quantiles(len(y), shift, sims, seed)
    mu = _invert(q[:, 1], p[1])
    lo, hi = _invert(q[:, 2], p[1]), _invert(q[:, 0], p[1])
    return {"lambda": mu, "lambda_lo": lo, "lambda_hi": hi, "er_star": _star(_intercepts(y, x, mu, D), mu, at),
            "er_star_lo": float("nan"), "er_star_hi": float("nan")}


METHODS = ("standard (moving-block)", "Kilian (1998)", "Andrews (1993)")


def coverage_study(lams: tuple[float, ...], n_pairs: int, sims: int, reps: int, block: int, level: float, seed: int,
                   andrews_sims: int) -> pd.DataFrame:
    """Coverage of the 90% interval for lambda by method (prompt 05 pre-flight 2): AR(1) with Gaussian errors and a
    stationary start, ``n_pairs`` pairs."""
    rows = []
    for li, lam in enumerate(lams):
        rng = np.random.default_rng([seed, 900, li])
        hits = dict.fromkeys(METHODS, 0)
        width = dict.fromkeys(METHODS, 0.0)
        for s in range(sims):
            y = [0.01 / (1 - lam) + rng.normal(0, 0.03) / np.sqrt(1 - lam ** 2)]
            for _ in range(n_pairs):
                y.append(0.01 + lam * y[-1] + rng.normal(0, 0.03))
            d = pairs(pd.Series(y, index=range(2000, 2001 + n_pairs)))
            r = {METHODS[0]: estimate(d, reps, block, level, [seed, 901, li, s]),
                 METHODS[1]: kilian(d, reps, reps, level, [seed, 902, li, s]),
                 METHODS[2]: andrews(d, level, andrews_sims, [seed, 903])}
            for k, v in r.items():
                hits[k] += v["lambda_lo"] < lam < v["lambda_hi"]
                width[k] += v["lambda_hi"] - v["lambda_lo"]
        for k in METHODS:
            rows.append({"true_lambda": lam, "n_pairs": n_pairs, "method": k, "simulations": sims,
                         "coverage": hits[k] / sims, "nominal": level, "mean_width": width[k] / sims})
    return pd.DataFrame(rows)


def primary_method(cov: pd.DataFrame) -> str:
    """The method whose coverage is closest to nominal on average across the true lambdas."""
    g = (cov["coverage"] - cov["nominal"]).abs().groupby(cov["method"]).mean()
    return str(g.idxmin())


def mean_interval(er: pd.Series, reps: int, block: int, level: float, seed) -> tuple[float, float, float]:
    """Sample mean of ER with a moving-block bootstrap percentile interval (through-the-cycle level, no model)."""
    v = er.dropna().sort_index().to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([v[block_indices(len(v), block, rng)].mean() for _ in range(reps)])
    lo, hi = interval(pd.Series(means), level)
    return float(v.mean()), lo, hi
