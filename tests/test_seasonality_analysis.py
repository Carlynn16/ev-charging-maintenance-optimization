"""
tests/test_seasonality_analysis.py

Pytest tests for src/seasonality_analysis.py public functions.
Runs against the real data file at data/session_stats.xlsx.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import apply_missing_data_assumption, build_panel, load_raw
from plots import plot_monthly_rate_comparison
from seasonality_analysis import dunns_posthoc, kruskal_seasonality, monthly_rate_summary

DATA_PATH = Path(__file__).parent.parent / "data" / "session_stats.xlsx"


# ── Module-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panel_zero():
    sessions, tasks, incidents = load_raw(DATA_PATH)
    panel = build_panel(sessions, tasks, incidents)
    return apply_missing_data_assumption(panel, "missing_is_zero")


@pytest.fixture(scope="module")
def summary(panel_zero):
    return monthly_rate_summary(panel_zero)


@pytest.fixture(scope="module")
def kw_result(panel_zero):
    return kruskal_seasonality(panel_zero)


# ── monthly_rate_summary ──────────────────────────────────────────────────────

def test_summary_returns_six_months(summary):
    assert len(summary) == 6


def test_summary_expected_columns(summary):
    for col in ("month", "n_observations", "mean_rate", "median_rate", "std", "ci_low", "ci_high"):
        assert col in summary.columns, f"Missing column: {col}"


def test_summary_months_are_strings(summary):
    # pandas 3.x uses StringDtype rather than object for string columns;
    # check values directly instead of the dtype sentinel.
    assert all(isinstance(v, str) for v in summary["month"])


def test_summary_months_sorted_chronologically(summary):
    months = summary["month"].tolist()
    assert months == sorted(months)


def test_summary_ci_low_le_mean_le_ci_high(summary):
    assert (summary["ci_low"] <= summary["mean_rate"]).all()
    assert (summary["mean_rate"] <= summary["ci_high"]).all()


def test_summary_n_observations_sum_matches_panel(panel_zero, summary):
    assert summary["n_observations"].sum() == len(panel_zero)


def test_summary_no_nan(summary):
    assert not summary.isnull().any().any()


def test_summary_missing_column_raises(panel_zero):
    bad = panel_zero.drop(columns=["incident_rate"])
    with pytest.raises(ValueError, match="missing required columns"):
        monthly_rate_summary(bad)


# ── kruskal_seasonality ───────────────────────────────────────────────────────

def test_kruskal_returns_expected_keys(kw_result):
    for key in ("H_statistic", "p_value", "n_per_month"):
        assert key in kw_result, f"Missing key: {key}"


def test_kruskal_h_is_non_negative_float(kw_result):
    assert isinstance(kw_result["H_statistic"], float)
    assert kw_result["H_statistic"] >= 0


def test_kruskal_p_value_in_range(kw_result):
    assert 0.0 <= kw_result["p_value"] <= 1.0
    assert not math.isnan(kw_result["p_value"])


def test_kruskal_n_per_month_has_six_entries(kw_result):
    assert len(kw_result["n_per_month"]) == 6


def test_kruskal_n_per_month_sums_to_panel_length(panel_zero, kw_result):
    assert sum(kw_result["n_per_month"].values()) == len(panel_zero)


# ── dunns_posthoc ─────────────────────────────────────────────────────────────

def test_dunns_returns_none_or_dataframe(panel_zero):
    result = dunns_posthoc(panel_zero)
    assert result is None or isinstance(result, pd.DataFrame)


def test_dunns_dataframe_is_6x6_when_returned(panel_zero):
    result = dunns_posthoc(panel_zero)
    if result is not None:
        assert result.shape == (6, 6)


# ── plot_monthly_rate_comparison ──────────────────────────────────────────────

def test_plot_monthly_rate_comparison_returns_axes(summary):
    import numpy as np
    # Build the trimmed DataFrame the notebook constructs.
    months = summary["month"].tolist()
    trimmed = pd.DataFrame({
        "month": months,
        "trimmed_mean_p95": summary["mean_rate"].values * 0.7,  # synthetic trim
    })
    ax = plot_monthly_rate_comparison(summary.rename(columns={"mean_rate": "untrimmed_mean"}),
                                      trimmed)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 2
    plt.close("all")
