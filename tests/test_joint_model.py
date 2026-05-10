"""
tests/test_joint_model.py

Pytest tests for src/joint_model.py public functions.
Runs against the real data file at data/session_stats.xlsx.
Module-scoped fixtures so the GEE model is fitted only once per test run.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import apply_missing_data_assumption, build_panel, load_raw
from joint_model import (
    dispersion_check,
    extract_rate_ratios,
    fit_gee_poisson,
    fit_under_unknown,
    prepare_model_data,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "session_stats.xlsx"


# ── Module-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_panel():
    sessions, tasks, incidents = load_raw(DATA_PATH)
    return build_panel(sessions, tasks, incidents)


@pytest.fixture(scope="module")
def panel_zero(raw_panel):
    return apply_missing_data_assumption(raw_panel, "missing_is_zero")


@pytest.fixture(scope="module")
def model_df(panel_zero):
    return prepare_model_data(panel_zero)


@pytest.fixture(scope="module")
def gee_result(model_df):
    return fit_gee_poisson(model_df)


@pytest.fixture(scope="module")
def rate_ratios(gee_result):
    return extract_rate_ratios(gee_result)


# ── prepare_model_data ────────────────────────────────────────────────────────

def test_prepare_model_data_expected_columns(model_df):
    for col in ("charging_location_id", "incidents_created", "tasks_solved",
                "log_utilization", "month_str"):
        assert col in model_df.columns, f"Missing column: {col}"


def test_prepare_model_data_no_zero_tasks(model_df):
    assert (model_df["tasks_solved"] > 0).all()


def test_prepare_model_data_no_nonpositive_utilization(panel_zero):
    # Inject a bad row and verify it is dropped.
    bad = panel_zero.copy()
    bad.loc[bad.index[0], "utilization"] = 0.0
    result = prepare_model_data(bad)
    assert (result["log_utilization"].notna()).all()
    assert not np.isinf(result["log_utilization"]).any()


def test_prepare_model_data_missing_column_raises(panel_zero):
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_model_data(panel_zero.drop(columns=["utilization"]))


# ── fit_gee_poisson ───────────────────────────────────────────────────────────

def test_fit_gee_has_seven_params(gee_result):
    # Intercept + log_utilization + 5 month dummies = 7
    assert len(gee_result.params) == 7


def test_fit_gee_params_are_finite(gee_result):
    assert np.isfinite(gee_result.params.values).all()


# ── extract_rate_ratios ───────────────────────────────────────────────────────

def test_extract_rate_ratios_has_six_rows(rate_ratios):
    # Intercept excluded: log_utilization + 5 month dummies = 6
    assert len(rate_ratios) == 6


def test_extract_rate_ratios_expected_columns(rate_ratios):
    for col in ("term", "rate_ratio", "ci_low", "ci_high", "p_value"):
        assert col in rate_ratios.columns, f"Missing column: {col}"


def test_extract_rate_ratios_ci_bounds_ordered(rate_ratios):
    assert (rate_ratios["ci_low"] <= rate_ratios["rate_ratio"]).all()
    assert (rate_ratios["rate_ratio"] <= rate_ratios["ci_high"]).all()


def test_extract_rate_ratios_first_term_is_utilization(rate_ratios):
    assert rate_ratios.iloc[0]["term"] == "log(utilization)"


# ── dispersion_check ──────────────────────────────────────────────────────────

def test_dispersion_check_returns_expected_keys(gee_result):
    result = dispersion_check(gee_result)
    for key in ("dispersion_ratio", "n", "n_params", "interpretation"):
        assert key in result, f"Missing key: {key}"


def test_dispersion_check_ratio_is_positive(gee_result):
    result = dispersion_check(gee_result)
    assert result["dispersion_ratio"] > 0


def test_dispersion_check_interpretation_is_string(gee_result):
    result = dispersion_check(gee_result)
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


# ── fit_under_unknown ─────────────────────────────────────────────────────────

def test_fit_under_unknown_smaller_n(raw_panel, model_df):
    result_unknown = fit_under_unknown(raw_panel)
    n_unknown = result_unknown.model.nobs
    assert n_unknown < len(model_df)
