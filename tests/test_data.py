"""
tests/test_data.py

Pytest tests for src/data.py public functions.
Runs against the real data file at data/session_stats.xlsx.
"""
import numpy as np
import numpy.testing as npt
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import (
    apply_missing_data_assumption,
    build_panel,
    coverage_summary,
    load_raw,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "session_stats.xlsx"


# ── Module-scoped fixtures (Excel file read once per test run) ─────────────────

@pytest.fixture(scope="module")
def raw_sheets():
    sessions, tasks, incidents = load_raw(DATA_PATH)
    return sessions, tasks, incidents


@pytest.fixture(scope="module")
def panel(raw_sheets):
    sessions, tasks, incidents = raw_sheets
    return build_panel(sessions, tasks, incidents)


@pytest.fixture(scope="module")
def panel_zero(panel):
    return apply_missing_data_assumption(panel, "missing_is_zero")


@pytest.fixture(scope="module")
def panel_unknown(panel):
    return apply_missing_data_assumption(panel, "missing_is_unknown")


# ── load_raw ──────────────────────────────────────────────────────────────────

def test_load_raw_sessions_has_charging_location_id_not_location_id(raw_sheets):
    sessions, _, _ = raw_sheets
    assert "charging_location_id" in sessions.columns
    assert "location_id" not in sessions.columns


def test_load_raw_tasks_has_charging_location_id(raw_sheets):
    _, tasks, _ = raw_sheets
    assert "charging_location_id" in tasks.columns


def test_load_raw_incidents_has_charging_location_id(raw_sheets):
    _, _, incidents = raw_sheets
    assert "charging_location_id" in incidents.columns


def test_load_raw_no_trailing_nan_in_sessions(raw_sheets):
    sessions, _, _ = raw_sheets
    assert sessions["charging_location_id"].notna().all()


def test_load_raw_no_trailing_nan_in_tasks(raw_sheets):
    _, tasks, _ = raw_sheets
    assert tasks["charging_location_id"].notna().all()


def test_load_raw_no_trailing_nan_in_incidents(raw_sheets):
    _, _, incidents = raw_sheets
    assert incidents["charging_location_id"].notna().all()


def test_load_raw_sessions_row_count(raw_sheets):
    sessions, _, _ = raw_sheets
    assert len(sessions) == 3661


def test_load_raw_tasks_row_count(raw_sheets):
    _, tasks, _ = raw_sheets
    assert len(tasks) == 2407


def test_load_raw_incidents_row_count(raw_sheets):
    _, _, incidents = raw_sheets
    assert len(incidents) == 1040


def test_load_raw_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw("data/does_not_exist.xlsx")


# ── build_panel ───────────────────────────────────────────────────────────────

def test_build_panel_row_count(panel):
    assert len(panel) == 3661


def test_build_panel_expected_columns(panel):
    expected = {
        "charging_location_id", "month", "evse_count", "session_count",
        "sessions_per_evse", "avg_duration_sec",
        "tasks_solved", "incidents_created", "utilization",
    }
    assert expected.issubset(set(panel.columns))


def test_build_panel_utilization_non_negative(panel):
    assert (panel["utilization"] >= 0).all()


def test_build_panel_utilization_not_nan(panel):
    # evse_count and session_count are always present (from Sessions) — no NaN expected.
    assert panel["utilization"].notna().all()


def test_build_panel_utilization_matches_recomputed(panel):
    recomputed = panel["session_count"] / panel["evse_count"]
    npt.assert_allclose(panel["utilization"].values, recomputed.values, rtol=1e-9)


def test_build_panel_tasks_solved_nan_fraction(panel):
    nan_pct = panel["tasks_solved"].isna().mean() * 100
    assert abs(nan_pct - 35.0) <= 2.0, f"tasks_solved NaN% = {nan_pct:.1f}, expected ~35%"


def test_build_panel_incidents_created_nan_fraction(panel):
    nan_pct = panel["incidents_created"].isna().mean() * 100
    assert abs(nan_pct - 72.0) <= 2.0, f"incidents_created NaN% = {nan_pct:.1f}, expected ~72%"


# ── apply_missing_data_assumption ─────────────────────────────────────────────

def test_missing_is_zero_row_count(panel_zero):
    assert len(panel_zero) == 2396


def test_missing_is_zero_no_nan_in_incidents(panel_zero):
    assert panel_zero["incidents_created"].notna().all()


def test_missing_is_zero_incident_rate_computed(panel_zero):
    assert "incident_rate" in panel_zero.columns
    assert panel_zero["incident_rate"].notna().all()


def test_missing_is_zero_mean_rate(panel_zero):
    mean_rate = panel_zero["incident_rate"].mean()
    assert abs(mean_rate - 0.10) <= 0.01, f"mean incident_rate = {mean_rate:.4f}, expected ~0.10"


def test_missing_is_unknown_row_count(panel_unknown):
    assert len(panel_unknown) == 1030


def test_missing_is_unknown_no_nan_in_incidents(panel_unknown):
    assert panel_unknown["incidents_created"].notna().all()


def test_missing_is_unknown_no_nan_in_rate(panel_unknown):
    assert panel_unknown["incident_rate"].notna().all()


def test_missing_is_unknown_mean_rate(panel_unknown):
    mean_rate = panel_unknown["incident_rate"].mean()
    assert abs(mean_rate - 0.236) <= 0.01, f"mean incident_rate = {mean_rate:.4f}, expected ~0.236"


def test_invalid_mode_raises_value_error(panel):
    with pytest.raises(ValueError, match="missing_is_purple"):
        apply_missing_data_assumption(panel, "missing_is_purple")


def test_incident_rate_equals_incidents_over_tasks_zero(panel_zero):
    expected = panel_zero["incidents_created"] / panel_zero["tasks_solved"]
    npt.assert_allclose(panel_zero["incident_rate"].values, expected.values, rtol=1e-9)


def test_incident_rate_equals_incidents_over_tasks_unknown(panel_unknown):
    expected = panel_unknown["incidents_created"] / panel_unknown["tasks_solved"]
    npt.assert_allclose(panel_unknown["incident_rate"].values, expected.values, rtol=1e-9)


# ── coverage_summary ──────────────────────────────────────────────────────────

def test_coverage_summary_has_expected_keys(raw_sheets):
    sessions, tasks, incidents = raw_sheets
    result = coverage_summary(sessions, tasks, incidents)
    for key in ("row_counts", "unique_locations", "missing_pct", "monthly_counts", "panel_coverage"):
        assert key in result, f"Missing key: {key}"


def test_coverage_summary_row_counts(raw_sheets):
    sessions, tasks, incidents = raw_sheets
    result = coverage_summary(sessions, tasks, incidents)
    assert result["row_counts"]["sessions"] == 3661
    assert result["row_counts"]["tasks"] == 2407
    assert result["row_counts"]["incidents"] == 1040


def test_coverage_summary_unique_locations(raw_sheets):
    sessions, tasks, incidents = raw_sheets
    result = coverage_summary(sessions, tasks, incidents)
    assert result["unique_locations"]["sessions"] == 638
    assert result["unique_locations"]["tasks"] == 572
    assert result["unique_locations"]["incidents"] == 454
