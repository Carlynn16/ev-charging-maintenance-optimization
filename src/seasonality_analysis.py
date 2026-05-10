"""
src/seasonality_analysis.py

Month-level analysis of whether per-visit incident rate varies across the six
calendar months in the study window (Jan–Jun 2024).

Design notes
------------
The unit of analysis here is the (location, month) row from the panel produced
by apply_missing_data_assumption(). Each row's incident_rate = incidents /
tasks is treated as a single observation for that station-month. We test whether
the distribution of that rate differs across months.

Why Kruskal-Wallis first?
  Incident rate is right-skewed and zero-inflated; one-way ANOVA would require
  normality within each month group, which will not hold. KW tests only whether
  the distributions are identical without assuming a specific shape.

Why Dunn's post-hoc (only if KW is significant)?
  KW tells us 'at least one month differs' but not which. Dunn's test uses the
  same rank-sum logic as KW and applies Bonferroni correction to control the
  family-wise error rate across the 15 pairwise comparisons.

Why not run Dunn's unconditionally?
  Running post-hoc comparisons when the omnibus test is non-significant inflates
  type-I error and produces results that are hard to interpret. We gate on KW
  significance (p < 0.05) to avoid this.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal as _kruskal
import scikit_posthocs as sp


def monthly_rate_summary(
    panel: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-month descriptive statistics with bootstrapped confidence intervals.

    Each row summarises the distribution of incident_rate values for one
    calendar month. The 95 % CI on the mean is computed via the percentile
    bootstrap (1 000 resamples) — preferred over the normal approximation
    because the within-month distributions are right-skewed.

    The 'month' column is formatted as 'YYYY-MM' for readability in tables and
    plots. The DataFrame is sorted chronologically (Jan → Jun).

    Parameters
    ----------
    panel : pd.DataFrame
        Panel as returned by apply_missing_data_assumption(); must contain
        'month' (datetime) and 'incident_rate' columns.
    n_bootstrap : int
        Number of bootstrap resamples for the CI. Default 1 000.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        One row per calendar month with columns:
          month           — 'YYYY-MM' string label
          n_observations  — number of (location, month) rows in that month
          mean_rate       — mean incident_rate
          median_rate     — median incident_rate
          std             — standard deviation
          ci_low          — 2.5th percentile of bootstrap distribution of mean
          ci_high         — 97.5th percentile
    """
    required = {"month", "incident_rate"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"panel is missing required columns: {missing}. "
            "Did you call apply_missing_data_assumption() first?"
        )

    rng = np.random.default_rng(seed)
    rows = []

    for month_dt, grp in panel.groupby(panel["month"].dt.to_period("M")):
        vals = grp["incident_rate"].values
        boots = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "month": str(month_dt),
            "n_observations": len(vals),
            "mean_rate": float(vals.mean()),
            "median_rate": float(np.median(vals)),
            "std": float(vals.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        })

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def kruskal_seasonality(panel: pd.DataFrame) -> dict:
    """Kruskal-Wallis H-test for differences in incident rate across months.

    Tests whether the six monthly distributions of incident_rate are identical.
    A significant result (p < 0.05) indicates that at least one month's rate
    distribution differs from the others, warranting post-hoc testing with
    dunns_posthoc(). A non-significant result supports the conclusion that
    seasonality is not a meaningful driver of incident rate in this dataset.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with 'month' (datetime) and 'incident_rate' columns.

    Returns
    -------
    dict
        H_statistic  : float — Kruskal-Wallis H statistic.
        p_value      : float — p-value.
        n_per_month  : dict  — {'YYYY-MM': n_observations} for each month.
    """
    groups = []
    n_per_month = {}
    for month_dt, grp in panel.groupby(panel["month"].dt.to_period("M")):
        vals = grp["incident_rate"].values
        groups.append(vals)
        n_per_month[str(month_dt)] = len(vals)

    h_stat, p_val = _kruskal(*groups)
    return {
        "H_statistic": float(h_stat),
        "p_value": float(p_val),
        "n_per_month": n_per_month,
    }


def dunns_posthoc(
    panel: pd.DataFrame,
    p_adjust: str = "bonferroni",
) -> Optional[pd.DataFrame]:
    """Dunn's pairwise post-hoc test across months, gated on KW significance.

    Runs Dunn's test only if Kruskal-Wallis is significant (p < 0.05); returns
    None otherwise. This gate prevents inflated type-I error from unconditional
    post-hoc testing.

    Dunn's test uses the same rank-sum logic as Kruskal-Wallis and is therefore
    the natural follow-up. Bonferroni correction controls the family-wise error
    rate across all 15 pairwise comparisons (C(6,2) = 15).

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with 'month' (datetime) and 'incident_rate' columns.
    p_adjust : str
        Multiple-comparison adjustment method passed to posthoc_dunn().
        Default 'bonferroni'. Other options: 'holm', 'fdr_bh', etc.

    Returns
    -------
    pd.DataFrame or None
        6×6 symmetric matrix of adjusted p-values indexed and columned by
        'YYYY-MM' month labels, or None if KW is not significant.
    """
    kw = kruskal_seasonality(panel)
    if kw["p_value"] >= 0.05:
        print(
            f"Kruskal-Wallis not significant (p = {kw['p_value']:.4f}). "
            "Dunn's post-hoc not run."
        )
        return None

    # Build a flat list of (rate, month_label) pairs for posthoc_dunn.
    labels = panel["month"].dt.to_period("M").astype(str)
    result = sp.posthoc_dunn(
        panel.assign(_month=labels),
        val_col="incident_rate",
        group_col="_month",
        p_adjust=p_adjust,
    )
    # Sort rows and columns chronologically.
    order = sorted(result.index.tolist())
    return result.loc[order, order]
