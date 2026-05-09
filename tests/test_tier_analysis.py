"""
tests/test_tier_analysis.py

Pytest tests for src/tier_analysis.py public functions.
Runs against the real data file at data/session_stats.xlsx.
"""
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import apply_missing_data_assumption, build_panel, load_raw
from tier_analysis import (
    assign_tertiles,
    compute_station_aggregates,
    kruskal_test,
    spearman_test,
    tier_summary,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "session_stats.xlsx"


# ── Module-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panel_zero():
    sessions, tasks, incidents = load_raw(DATA_PATH)
    panel = build_panel(sessions, tasks, incidents)
    return apply_missing_data_assumption(panel, "missing_is_zero")


@pytest.fixture(scope="module")
def station_df(panel_zero):
    agg = compute_station_aggregates(panel_zero)
    return assign_tertiles(agg)


# ── compute_station_aggregates ────────────────────────────────────────────────

def test_aggregates_one_row_per_station(panel_zero):
    agg = compute_station_aggregates(panel_zero)
    n_unique = panel_zero["charging_location_id"].nunique()
    assert len(agg) == n_unique


def test_aggregates_expected_columns(panel_zero):
    agg = compute_station_aggregates(panel_zero)
    for col in ("charging_location_id", "mean_utilization", "mean_incident_rate", "n_months_observed"):
        assert col in agg.columns, f"Missing column: {col}"


def test_aggregates_missing_column_raises(panel_zero):
    bad = panel_zero.drop(columns=["incident_rate"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_station_aggregates(bad)


# ── assign_tertiles ───────────────────────────────────────────────────────────

def test_tertiles_adds_tier_column(station_df):
    assert "tier" in station_df.columns


def test_tertiles_three_categories(station_df):
    assert len(station_df["tier"].cat.categories) == 3


def test_tertiles_category_order(station_df):
    cats = list(station_df["tier"].cat.categories)
    assert cats == ["Low", "Medium", "High"]


# ── kruskal_test ──────────────────────────────────────────────────────────────

def test_kruskal_returns_expected_keys(station_df):
    result = kruskal_test(station_df)
    for key in ("H_statistic", "p_value", "n_per_tier"):
        assert key in result, f"Missing key: {key}"


def test_kruskal_p_value_range(station_df):
    result = kruskal_test(station_df)
    assert 0.0 <= result["p_value"] <= 1.0


def test_kruskal_n_per_tier_sums_to_total(station_df):
    result = kruskal_test(station_df)
    assert sum(result["n_per_tier"].values()) == len(station_df)


# ── spearman_test ─────────────────────────────────────────────────────────────

def test_spearman_returns_expected_keys(station_df):
    result = spearman_test(station_df)
    for key in ("rho", "p_value", "n"):
        assert key in result, f"Missing key: {key}"


def test_spearman_rho_in_range(station_df):
    result = spearman_test(station_df)
    assert -1.0 <= result["rho"] <= 1.0


def test_spearman_n_equals_station_count(station_df):
    result = spearman_test(station_df)
    assert result["n"] == len(station_df)
