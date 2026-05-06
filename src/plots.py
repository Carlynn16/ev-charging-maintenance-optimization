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

    Parameters
    ----------
    x, y : pd.Series
        Aligned series; rows where either is NaN are dropped together.
    ax : plt.Axes, optional
        Target axes. A new figure is created if not supplied.
    log_x : bool
        If True, plot x on a log10 scale with original-unit tick labels.
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
        smoothed = _sm_lowess(yv, xv, frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1],
                color=_PALETTE["accent"], linewidth=2.5, label="LOWESS")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel or (x.name or ""))
    ax.set_ylabel(ylabel or (y.name or ""))
    ax.legend(frameon=False)
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
