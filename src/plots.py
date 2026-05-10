"""
src/plots.py

Reusable plotting helpers for the EV charging maintenance analysis.

All functions accept an optional `ax` argument so they can be embedded into a
multi-panel figure layout. Call `setup_style()` once at the start of each
notebook or script to apply consistent defaults.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess

# Restrained palette: one muted blue, one warm accent, neutral grey.
_PALETTE: dict[str, str] = {
    "primary":   "#4C72B0",  # muted blue  — main data series
    "accent":    "#DD8452",  # warm orange — smoothers, highlights
    "secondary": "#8C8C8C",  # neutral grey — scatter points, secondary lines
    "light":     "#C7D7EE",  # light blue  — fills, bands
}


def setup_style() -> None:
    """Apply consistent matplotlib/seaborn defaults for all project plots.

    Sets a white-grid theme, removes top/right spines, and loads the project
    colour cycle (primary → accent → secondary). Call once per notebook before
    any plotting. Keeps savefig at 200 DPI with tight bounding box so all saved
    figures are report-ready without extra arguments at the call site.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(
            "color",
            [_PALETTE["primary"], _PALETTE["accent"], _PALETTE["secondary"]],
        ),
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def plot_distribution(
    series: pd.Series,
    ax: Optional[plt.Axes] = None,
    log_x: bool = False,
    title: str = "",
    xlabel: str = "",
) -> plt.Axes:
    """Histogram + KDE for one numeric series.

    Intended for distributional profiling of analysis variables before modelling.
    When `log_x=True` the histogram bins are log-spaced, which handles heavily
    right-skewed data (e.g. utilization ranging from 1 to 734) more faithfully
    than a linear axis that compresses most observations at the left edge. Non-
    positive values are silently dropped when log_x is True.

    Parameters
    ----------
    series : pd.Series
        Numeric values to plot; NaN rows are dropped.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    log_x : bool
        If True, use a log-scale x-axis and skip non-positive values.
    title : str
        Axes title.
    xlabel : str
        X-axis label; falls back to series.name.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    data = series.dropna()
    if log_x:
        data = data[data > 0]
        sns.histplot(data, kde=True, ax=ax, color=_PALETTE["primary"],
                     alpha=0.7, log_scale=True)
    else:
        sns.histplot(data, kde=True, ax=ax, color=_PALETTE["primary"], alpha=0.7)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel or (series.name or ""))
    ax.set_ylabel("Count")
    return ax


def plot_scatter_with_smoother(
    x: pd.Series,
    y: pd.Series,
    ax: Optional[plt.Axes] = None,
    log_x: bool = False,
    ylim: Optional[tuple] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> plt.Axes:
    """Scatter plot with a LOWESS smoother overlaid.

    Semi-transparent points (alpha=0.3) reveal density in overplotted regions.
    The LOWESS curve (statsmodels, frac=0.3) provides a non-parametric trend
    without assuming a functional form, making it useful for deciding whether
    the utilization–rate relationship is monotone before modelling it with GEE.

    When `log_x=True` the scatter and LOWESS are computed in log10 space and
    the x-axis is relabelled in original units — use this for utilization, which
    spans nearly three orders of magnitude.

    When `ylim` is set, the visible y-range is cropped to that window *after*
    the LOWESS is computed on all data. This prevents a handful of outliers from
    compressing the main data mass without biasing the smoother.

    Parameters
    ----------
    x, y : pd.Series
        Aligned series; rows where either is NaN are dropped together.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    log_x : bool
        If True, plot x on a log10 scale with original-unit tick labels.
    ylim : tuple (ymin, ymax), optional
        Crop the visible y-axis to this range. LOWESS is computed on all data
        regardless of this setting.
    title : str
        Axes title.
    xlabel, ylabel : str
        Axis labels; fall back to series names.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    mask = x.notna() & y.notna()
    xv = x[mask].values.astype(float)
    yv = y[mask].values.astype(float)

    if log_x:
        xv_plot = np.log10(np.clip(xv, 1e-9, None))
        ax.scatter(xv_plot, yv, alpha=0.3, color=_PALETTE["secondary"],
                   s=12, linewidths=0)
        # LOWESS computed on all data before any ylim crop.
        smoothed = _sm_lowess(yv, xv_plot, frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1],
                color=_PALETTE["accent"], linewidth=2.5, label="LOWESS")
        # Relabel x-axis in original (non-log) units for readability.
        candidates = [1, 5, 10, 50, 100, 500, 1000]
        log_min, log_max = xv_plot.min(), xv_plot.max()
        ticks = [(v, np.log10(v)) for v in candidates
                 if log_min <= np.log10(v) <= log_max]
        if ticks:
            ax.set_xticks([lt for _, lt in ticks])
            ax.set_xticklabels([str(v) for v, _ in ticks])
    else:
        ax.scatter(xv, yv, alpha=0.3, color=_PALETTE["secondary"],
                   s=12, linewidths=0)
        # LOWESS computed on all data before any ylim crop.
        smoothed = _sm_lowess(yv, xv, frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1],
                color=_PALETTE["accent"], linewidth=2.5, label="LOWESS")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel or (x.name or ""))
    ax.set_ylabel(ylabel or (y.name or ""))
    ax.legend(frameon=False)
    return ax


def plot_tier_means(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    value_col: str = "mean",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    title: str = "Mean incident rate by utilization tier",
    ylabel: str = "Mean incident rate (incidents / tasks)",
) -> plt.Axes:
    """Bar chart of per-tier mean incident rate with 95 % bootstrap CI error bars.

    Parameters
    ----------
    summary_df : pd.DataFrame
        As returned by tier_summary(), indexed by tier label (Low→Medium→High).
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    value_col, ci_low_col, ci_high_col : str
        Column names for the bar heights and confidence-interval bounds.
    title, ylabel : str
        Plot labels.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    tiers = summary_df.index.tolist()
    means = summary_df[value_col].values
    err_low = means - summary_df[ci_low_col].values
    err_high = summary_df[ci_high_col].values - means

    ax.bar(
        tiers,
        means,
        color=_PALETTE["primary"],
        alpha=0.85,
        width=0.5,
        zorder=2,
    )
    ax.errorbar(
        tiers,
        means,
        yerr=[err_low, err_high],
        fmt="none",
        color=_PALETTE["secondary"],
        capsize=5,
        linewidth=1.5,
        zorder=3,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Utilization tier")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    return ax


def plot_monthly_rate(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Mean incident rate by month",
    ylabel: str = "Mean incident rate (incidents / tasks)",
) -> plt.Axes:
    """Line + marker plot of per-month mean incident rate with 95 % bootstrap CI band.

    Connects months chronologically with a line so the temporal trend (or lack
    thereof) is immediately visible. The shaded CI band uses the percentile
    bootstrap intervals from monthly_rate_summary().

    Parameters
    ----------
    summary_df : pd.DataFrame
        As returned by monthly_rate_summary(), with columns 'month',
        'mean_rate', 'ci_low', 'ci_high'.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    title, ylabel : str
        Plot labels.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    months = summary_df["month"].tolist()
    means = summary_df["mean_rate"].values
    ci_low = summary_df["ci_low"].values
    ci_high = summary_df["ci_high"].values

    ax.plot(months, means, color=_PALETTE["primary"], linewidth=2, marker="o",
            markersize=7, zorder=3)
    ax.fill_between(months, ci_low, ci_high,
                    color=_PALETTE["light"], alpha=0.7,
                    label="95 % bootstrap CI", zorder=2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    return ax


def plot_monthly_rate_comparison(
    untrimmed: pd.DataFrame,
    trimmed_5pct: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "March elevation collapses when the top 5% of rates are trimmed",
    ylabel: str = "Mean incident rate (incidents / tasks)",
) -> plt.Axes:
    """Two-line comparison of untrimmed vs 5%-trimmed monthly means.

    Designed to accompany the tail sensitivity check: shows visually that
    March's apparent elevation in the untrimmed series collapses to the same
    level as other months once the top 5 % of rates are removed from each
    month. The gap between the two lines at March is the visual punchline.

    Parameters
    ----------
    untrimmed : pd.DataFrame
        Must have columns 'month', 'untrimmed_mean', and optionally 'ci_low' /
        'ci_high' for the shaded CI band.
    trimmed_5pct : pd.DataFrame
        Must have columns 'month' and 'trimmed_mean_p95'.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    title, ylabel : str
        Plot labels.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    months = untrimmed["month"].tolist()
    means_u = untrimmed["untrimmed_mean"].values
    means_t = trimmed_5pct["trimmed_mean_p95"].values

    # Untrimmed — solid line with CI band if available.
    ax.plot(months, means_u, color=_PALETTE["primary"], linewidth=2,
            marker="o", markersize=7, label="Including all observations", zorder=3)
    if "ci_low" in untrimmed.columns and "ci_high" in untrimmed.columns:
        ax.fill_between(months,
                        untrimmed["ci_low"].values,
                        untrimmed["ci_high"].values,
                        color=_PALETTE["light"], alpha=0.6, zorder=2)

    # Trimmed — dashed grey line, no CI band.
    ax.plot(months, means_t, color=_PALETTE["secondary"], linewidth=1.8,
            linestyle="--", marker="o", markersize=6,
            label="After 5% trim from each month", zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)
    return ax


def plot_forest(
    rate_ratios_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Rate ratios with 95 % CI (GEE Poisson)",
    xlabel: str = "Rate ratio (95 % CI)",
) -> plt.Axes:
    """Horizontal forest plot of GEE rate ratios.

    Terms are displayed top-to-bottom in the order they appear in the DataFrame
    (log_utilization first, then months chronologically). The x-axis is on a
    log scale because rate ratios are multiplicative. A vertical reference line
    at RR = 1 (no effect) is drawn in light grey. Significant terms (p < 0.05)
    are plotted in the primary blue; non-significant terms in secondary grey.

    Parameters
    ----------
    rate_ratios_df : pd.DataFrame
        As returned by extract_rate_ratios(). Must have columns:
        'term', 'rate_ratio', 'ci_low', 'ci_high', 'p_value'.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    title, xlabel : str
        Plot labels.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    df = rate_ratios_df.iloc[::-1].reset_index(drop=True)  # top-to-bottom display
    y_pos = np.arange(len(df))

    ax.axvline(1.0, color="#CCCCCC", linewidth=1.2, zorder=1)

    for i, row in df.iterrows():
        color = _PALETTE["primary"] if row["p_value"] < 0.05 else _PALETTE["secondary"]
        err_low  = row["rate_ratio"] - row["ci_low"]
        err_high = row["ci_high"] - row["rate_ratio"]
        ax.errorbar(
            row["rate_ratio"], i,
            xerr=[[err_low], [err_high]],
            fmt="o", color=color, markersize=7,
            elinewidth=1.5, capsize=4, zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["term"].tolist())
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    # Significance legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_PALETTE["primary"],
               markersize=8, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_PALETTE["secondary"],
               markersize=8, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    return ax


def plot_forest_comparison(
    rr_zero: pd.DataFrame,
    rr_unknown: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Rate ratios: missing_is_zero vs missing_is_unknown",
    xlabel: str = "Rate ratio (95 % CI)",
    labels: tuple[str, str] = ("missing_is_zero", "missing_is_unknown"),
) -> plt.Axes:
    """Side-by-side forest plot comparing two sets of rate ratios.

    Plots both sets of rate ratios on the same axes, with the two series
    slightly offset vertically so CIs do not overlap. Primary blue for the
    first series; warm orange for the second.

    Parameters
    ----------
    rr_zero, rr_unknown : pd.DataFrame
        As returned by extract_rate_ratios() for each model. Both must have
        the same terms in the same order.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    title, xlabel : str
        Plot labels.
    labels : tuple of two str
        Legend labels for the first (blue) and second (orange) series.
        Default: ('missing_is_zero', 'missing_is_unknown').

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    n = len(rr_zero)
    y_base = np.arange(n)
    offset = 0.18

    ax.axvline(1.0, color="#CCCCCC", linewidth=1.2, zorder=1)

    for df, y_off, color, label in [
        (rr_zero.iloc[::-1].reset_index(drop=True),    +offset, _PALETTE["primary"], labels[0]),
        (rr_unknown.iloc[::-1].reset_index(drop=True), -offset, _PALETTE["accent"],  labels[1]),
    ]:
        for i, row in df.iterrows():
            err_low  = row["rate_ratio"] - row["ci_low"]
            err_high = row["ci_high"] - row["rate_ratio"]
            ax.errorbar(
                row["rate_ratio"], y_base[i] + y_off,
                xerr=[[err_low], [err_high]],
                fmt="o", color=color, markersize=6,
                elinewidth=1.4, capsize=3, zorder=3,
                label=label if i == 0 else None,
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(rr_zero.iloc[::-1]["term"].tolist())
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    method: str = "spearman",
) -> plt.Axes:
    """Annotated correlation heatmap showing the lower triangle plus diagonal.

    Defaults to Spearman rank correlation because the analysis variables are
    all right-skewed; Spearman is less sensitive to extreme values than Pearson.
    The upper triangle is masked to remove redundancy while the diagonal
    (ρ = 1.0) is retained as a visual reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be correlated pairwise.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    method : str
        Correlation method passed to pd.DataFrame.corr(). Default 'spearman'.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    corr = df.corr(method=method)
    # k=1: mask upper triangle but keep diagonal (ρ=1.0 visible as reference).
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title(f"{method.capitalize()} correlation matrix",
                 fontsize=12, fontweight="bold")
    return ax
