"""Point-in-time filter (prompt 01 section 5; CLAUDE.md section 3.3; decision D1; blueprint §7 limitation L9)."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

log = logging.getLogger(__name__)


class MissingPublicationDateWarning(UserWarning):
    """Rows without a publication date were excluded by :func:`eligible`."""


def eligible(df: pd.DataFrame, valuation_date: str | pd.Timestamp, pub_col: str = "publication_date",
             period_col: str = "period_end", warn: bool = True,
             type_col: str = "publication_date_type") -> pd.DataFrame:
    """Keep rows whose publication date AND period end are on or before ``valuation_date``.

    Rows with no publication date are excluded and counted in a :class:`MissingPublicationDateWarning`.
    Rows whose date is only a lower bound (``type_col`` == 'lower_bound', D35) are excluded too: the true posting date
    may be later than the valuation date. Rows with no period end are judged on the publication date alone.
    """
    d = pd.Timestamp(valuation_date)
    pub = pd.to_datetime(df[pub_col], errors="coerce")
    lower = df[type_col].eq("lower_bound") if type_col in df.columns else pd.Series(False, index=df.index)
    missing = pub.isna() | lower
    ok = ~missing & (pub <= d)
    if period_col in df.columns:
        per = pd.to_datetime(df[period_col], errors="coerce")
        ok &= per.isna() | (per <= d)
    n_missing = int(missing.sum())
    if n_missing and warn:
        msg = (f"pit.eligible({d.date()}): {n_missing} of {len(df)} rows have no publication date (or only a "
               "lower bound, D35) and were excluded")
        warnings.warn(msg, MissingPublicationDateWarning, stacklevel=2)
        log.warning(msg)
    return df[ok]
