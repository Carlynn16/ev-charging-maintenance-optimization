"""
src/tier_analysis.py

Tier-based analysis of whether station utilization predicts incident rate.

Design notes
------------
The key analytical choice here is the unit of analysis: we aggregate to the
*station* level (per-station mean utilization and mean incident rate across all
observed months) before assigning tiers. This avoids the confound of the same
physical station sitting in different tiers month-to-month — per TASK.md §4.

The test stack is:
  1. Kruskal-Wallis — non-parametric test for any rate difference across tiers.
     Used first because incident_rate is skewed and zero-inflated; ANOVA
     normality assumptions would not hold.
  2. Spearman rank correlation — tests the monotone (ordered) hypothesis that
     higher utilization → higher rate, one station at a time.
Both are descriptive complements to the joint GEE model in Phase 4.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal as _kruskal
from scipy.stats import spearmanr as _spearmanr


def compute_station_aggregates(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (location, month) panel to one row per station.

    Computes the mean utilization and mean incident_rate across all months
    for which a station has a task record. Using the mean (rather than, say,
    the first observed value) smooths out month-to-month variability and gives
    a stable signal for tier assignment.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel as returned by apply_missing_data_assumption(); must already
        contain the 'incident_rate' and 'utilization' columns.

    Returns
    -------
    pd.DataFrame
        One row per station with columns:
          charging_location_id  — station identifier
          mean_utilization      — mean sessions/EVSE across observed months
          mean_incident_rate    — mean incidents/tasks across observed months
          n_months_observed     — number of (location, month) rows contributed
    """
    required = {"charging_location_id", "utilization", "incident_rate"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"panel is missing required columns: {missing}. "
            "Did you call apply_missing_data_assumption() first?"
        )

    agg = (
        panel.groupby("charging_location_id")
        .agg(
            mean_utilization=("utilization", "mean"),
            mean_incident_rate=("incident_rate", "mean"),
            n_months_observed=("incident_rate", "count"),
        )
        .reset_index()
    )
    return agg


def assign_tertiles(
    station_df: pd.DataFrame,
    col: str = "mean_utilization",
    labels: Sequence[str] = ("Low", "Medium", "High"),
) -> pd.DataFrame:
    """Add a 'tier' column by splitting stations into tertiles of `col`.

    Uses pd.qcut with q=3, which produces (roughly) equal-count groups. When
    ties in the cut column cause overlapping bin edges, duplicates='drop'
    silently merges the affected bins; the resulting tiers may not be perfectly
    equal-sized but the assignment remains deterministic. The actual group sizes
    are visible in tier_summary().

    Parameters
    ----------
    station_df : pd.DataFrame
        Per-station DataFrame as returned by compute_station_aggregates().
    col : str
        Column to cut on. Default 'mean_utilization'.
    labels : sequence of str
        Tier names in ascending order. Must have the same length as q=3.

    Returns
    -------
    pd.DataFrame
        Copy of station_df with a 'tier' column added (ordered Categorical).
    """
    if len(labels) != 3:
        raise ValueError(f"labels must have length 3, got {len(labels)}")

    df = station_df.copy()
    df["tier"] = pd.qcut(df[col], q=3, labels=list(labels), duplicates="drop")
    return df


def tier_summary(
    station_df: pd.DataFrame,
    value_col: str = "mean_incident_rate",
    tier_col: str = "tier",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-tier descriptive statistics with bootstrapped confidence intervals.

    The 95% CI is computed via the percentile bootstrap (1 000 resamples,
    reproducible via `seed`). The percentile method is preferred over the
    normal approximation because the within-tier distributions are skewed.

    Also includes the utilization range of each tier for interpretability —
    knowing that 'Low' spans [5, 85] sessions/EVSE is more useful to the
    operator than a tertile label alone.

    Parameters
    ----------
    station_df : pd.DataFrame
        Per-station DataFrame with tier column (from assign_tertiles()).
    value_col : str
        Column to summarise. Default 'mean_incident_rate'.
    tier_col : str
        Column containing tier labels. Default 'tier'.
    n_bootstrap : int
        Number of bootstrap resamples for the CI. Default 1 000.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        Indexed by tier label (Low → Medium → High) with columns:
          n          — number of stations
          mean       — mean of value_col
          median     — median of value_col
          std        — standard deviation of value_col
          ci_low     — 2.5th percentile of bootstrap distribution of the mean
          ci_high    — 97.5th percentile
          util_min   — minimum mean_utilization in the tier
          util_max   — maximum mean_utilization in the tier
    """
    rng = np.random.default_rng(seed)
    rows = []

    categories = station_df[tier_col].cat.categories
    for tier in categories:
        mask = station_df[tier_col] == tier
        vals = station_df.loc[mask, value_col].values
        utils = station_df.loc[mask, "mean_utilization"].values

        boots = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

        rows.append({
            "tier": tier,
            "n": len(vals),
            "mean": vals.mean(),
            "median": float(np.median(vals)),
            "std": vals.std(),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "util_min": utils.min(),
            "util_max": utils.max(),
        })

    return pd.DataFrame(rows).set_index("tier")


def kruskal_test(
    station_df: pd.DataFrame,
    value_col: str = "mean_incident_rate",
    tier_col: str = "tier",
) -> dict:
    """Kruskal-Wallis H-test for differences in incident rate across tiers.

    Kruskal-Wallis is a non-parametric alternative to one-way ANOVA. It tests
    whether the population distributions of `value_col` are identical across
    tier groups. It is appropriate here because incident_rate is skewed and
    zero-inflated — ANOVA's normality assumption would not hold.

    A non-significant result (p > 0.05) means we cannot reject the null that
    all three tiers have the same distribution. Combined with the GEE result,
    it supports recommending against utilization-tier-based scheduling.

    Parameters
    ----------
    station_df : pd.DataFrame
        Per-station DataFrame with tier column.
    value_col : str
        Column to test. Default 'mean_incident_rate'.
    tier_col : str
        Tier column. Default 'tier'.

    Returns
    -------
    dict
        H_statistic : float   — Kruskal-Wallis H statistic.
        p_value     : float   — p-value.
        n_per_tier  : dict    — {tier_label: n_stations} for each tier.
    """
    groups = [
        station_df.loc[station_df[tier_col] == t, value_col].values
        for t in station_df[tier_col].cat.categories
    ]
    h_stat, p_val = _kruskal(*groups)
    n_per_tier = {
        str(t): int((station_df[tier_col] == t).sum())
        for t in station_df[tier_col].cat.categories
    }
    return {"H_statistic": h_stat, "p_value": p_val, "n_per_tier": n_per_tier}


def spearman_test(
    station_df: pd.DataFrame,
    x_col: str = "mean_utilization",
    y_col: str = "mean_incident_rate",
) -> dict:
    """Spearman rank correlation between utilization and incident rate.

    Tests the monotone version of the client's hypothesis: do stations with
    *higher* utilization consistently have *higher* incident rates? Spearman
    is preferred over Pearson because both variables are right-skewed and
    Spearman is robust to outliers.

    This is computed at the station level (one observation per station), which
    avoids the pseudo-replication problem of using the raw (location, month)
    panel where each station appears up to six times.

    Parameters
    ----------
    station_df : pd.DataFrame
        Per-station DataFrame.
    x_col : str
        Utilization column. Default 'mean_utilization'.
    y_col : str
        Incident rate column. Default 'mean_incident_rate'.

    Returns
    -------
    dict
        rho     : float — Spearman correlation coefficient.
        p_value : float — two-tailed p-value.
        n       : int   — number of stations.
    """
    result = _spearmanr(station_df[x_col], station_df[y_col])
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": len(station_df),
    }
