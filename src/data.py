"""
src/data.py

Loading and panel construction for the EV charging station maintenance dataset.

Design notes
------------
The Excel file has three sheets:
  - Sessions   : (location, month) charging activity. 100 % coverage.
  - tasks      : (location, month) completed technician visits. ~65 % coverage.
  - Incidents  : (location, month) defects found during visits. ~28 % coverage.

Each sheet has a title row in cell A1 that must be skipped (skiprows=1). Each of
tasks and Incidents also has a trailing row whose charging_location_id is NaN —
an Excel artefact — which is dropped by dropna on the identifier column.

The Sessions sheet uses 'location_id' as the location key; the other two use
'charging_location_id'. All three are standardized to 'charging_location_id' here
so that downstream code never has to remember the inconsistency.

Incidents can only be discovered during a technician visit. The absence of an
incident record therefore has two defensible meanings:
  (a) The technician visited and found nothing (true zero).
  (b) We cannot know (the record is simply missing).
This ambiguity is surfaced as an explicit parameter in apply_missing_data_assumption()
rather than silently resolved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Anchored to this module's location so it resolves correctly regardless of cwd.
_DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "session_stats.xlsx"


def load_raw(
    path: Path | str = _DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw sheets and apply only the minimal cleaning needed to merge them.

    'Minimal' means: skip the title row, drop the trailing identifier-NaN row,
    standardize the location key name, cast identifiers to int64, and normalize
    month to datetime. No imputation, no filtering, no derived columns — those
    decisions belong to the caller.

    Parameters
    ----------
    path:
        Path to session_stats.xlsx. Defaults to data/session_stats.xlsx relative
        to the project root.

    Returns
    -------
    sessions, tasks, incidents : tuple[DataFrame, DataFrame, DataFrame]
        sessions  : 3 661 rows × 6 cols — (location, month) charging activity.
        tasks     : 2 407 rows × 3 cols — completed technician visits.
        incidents : 1 040 rows × 3 cols — defects found during visits.

    Raises
    ------
    FileNotFoundError
        If the Excel file does not exist at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    sessions = pd.read_excel(path, sheet_name="Sessions", skiprows=1)
    tasks = pd.read_excel(path, sheet_name="tasks", skiprows=1)
    incidents = pd.read_excel(path, sheet_name="Incidents", skiprows=1)

    # Drop the trailing row whose location identifier is NaN — an Excel artefact
    # present in both tasks and Incidents (one row each). Using the identifier
    # column rather than positional slicing is more robust to future file changes.
    sessions = sessions.dropna(subset=["location_id"]).copy()
    tasks = tasks.dropna(subset=["charging_location_id"]).copy()
    incidents = incidents.dropna(subset=["charging_location_id"]).copy()

    # Standardize the location key. Sessions calls it 'location_id'; rename so
    # all three sheets share 'charging_location_id' before any merge.
    sessions = sessions.rename(columns={"location_id": "charging_location_id"})

    # The trailing NaN row forced tasks and Incidents identifiers to float64.
    # Cast back to int64 after the drop so join keys are type-consistent.
    for df in (sessions, tasks, incidents):
        df["charging_location_id"] = df["charging_location_id"].astype("int64")

    # Normalize month to datetime so merge keys are type-consistent and the
    # column sorts chronologically without string-sort artefacts.
    for df in (sessions, tasks, incidents):
        df["month"] = pd.to_datetime(df["month"])

    return sessions, tasks, incidents


def build_panel(
    sessions: pd.DataFrame,
    tasks: pd.DataFrame,
    incidents: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three raw sheets into a single (location, month) panel DataFrame.

    Sessions is the join anchor because it covers 100 % of the universe (638
    locations × 6 months = 3 661 rows). A left merge preserves every session-month
    even when no task or incident record exists, keeping missingness explicit rather
    than silently discarding observations. This matters: a NaN in tasks_solved means
    'no visit recorded', which is analytically different from 'zero visits'.

    A floating-point 'utilization' column (session_count / evse_count) is derived
    and cross-checked against the pre-computed integer 'sessions_per_evse' column
    to catch any accidental data-loading error early.

    Parameters
    ----------
    sessions, tasks, incidents:
        DataFrames as returned by load_raw().

    Returns
    -------
    pd.DataFrame
        3 661 rows. All Sessions columns plus:
          tasks_solved       — NaN where no task record exists (~35 % of rows).
          incidents_created  — NaN where no incident record exists (~72 % of rows).
          utilization        — session_count / evse_count (floating-point).

    Raises
    ------
    ValueError
        If derived utilization deviates from sessions_per_evse by more than 1.0
        sessions per EVSE. (sessions_per_evse is a rounded integer in the source
        data, so differences up to 1.0 are expected and acceptable.)
    """
    panel = (
        sessions
        .merge(
            tasks[["charging_location_id", "month", "tasks_solved"]],
            on=["charging_location_id", "month"],
            how="left",
        )
        .merge(
            incidents[["charging_location_id", "month", "incidents_created"]],
            on=["charging_location_id", "month"],
            how="left",
        )
    )

    panel["utilization"] = panel["session_count"] / panel["evse_count"]

    # sessions_per_evse is stored as a rounded integer (e.g. 505/6 → 84, not 84.17),
    # so we allow up to 1.0 absolute difference rather than a strict float comparison.
    if not np.all(
        np.isclose(
            panel["utilization"].values,
            panel["sessions_per_evse"].values,
            atol=1.0,
            rtol=0.0,
            equal_nan=True,
        )
    ):
        diffs = (panel["utilization"] - panel["sessions_per_evse"]).abs()
        worst_idx = diffs.idxmax()
        raise ValueError(
            f"Derived utilization deviates from sessions_per_evse by up to "
            f"{diffs[worst_idx]:.4f} at index {worst_idx}. "
            "Check that the Sessions sheet was read with skiprows=1."
        )

    return panel


def apply_missing_data_assumption(
    panel: pd.DataFrame,
    mode: Literal["missing_is_zero", "missing_is_unknown"],
) -> pd.DataFrame:
    """Apply an explicit missing-data assumption and compute the incident_rate column.

    About 28 % of panel rows lack an incident record. Because incidents can only
    be found during a visit, this missingness has two defensible interpretations:

    'missing_is_zero'
        The technician visited and found nothing — the expected outcome for a
        well-maintained station. Fills missing incidents_created with 0 and retains
        all rows that have a valid task record. Under this assumption the mean
        incident rate is approximately 10 %.

    'missing_is_unknown'
        We cannot infer what would have been found; the row is dropped entirely.
        This is the more conservative assumption. Under it the mean incident rate
        is approximately 24 %. Use for sensitivity analysis.

    Both modes drop rows where tasks_solved is missing or zero because the rate
    denominator must be a positive integer for the result to be meaningful. (In
    practice tasks_solved is never zero in the raw data — min is 1 — but rows with
    no task record at all appear as NaN in the merged panel.)

    Parameters
    ----------
    panel:
        DataFrame as returned by build_panel().
    mode:
        'missing_is_zero' or 'missing_is_unknown'.

    Returns
    -------
    pd.DataFrame
        Filtered copy of panel with an 'incident_rate' column
        (incidents_created / tasks_solved) appended.

    Raises
    ------
    ValueError
        If mode is not one of the two valid strings.
    """
    valid = {"missing_is_zero", "missing_is_unknown"}
    if mode not in valid:
        raise ValueError(
            f"mode must be 'missing_is_zero' or 'missing_is_unknown', got {mode!r}. "
            f"Valid options: {sorted(valid)}"
        )

    df = panel.copy()

    if mode == "missing_is_zero":
        # Technician visited, found nothing: treat as zero incidents.
        df["incidents_created"] = df["incidents_created"].fillna(0.0)
        # Retain only rows where a visit occurred (tasks_solved is a valid positive number).
        df = df[df["tasks_solved"].notna() & (df["tasks_solved"] > 0)].copy()
    else:  # missing_is_unknown
        # Drop any row where either the numerator or denominator is uncertain.
        df = df[
            df["tasks_solved"].notna()
            & (df["tasks_solved"] > 0)
            & df["incidents_created"].notna()
        ].copy()

    df["incident_rate"] = df["incidents_created"] / df["tasks_solved"]
    return df


def coverage_summary(
    sessions: pd.DataFrame,
    tasks: pd.DataFrame,
    incidents: pd.DataFrame,
) -> dict:
    """Summarize coverage and missingness across the three raw sheets.

    Intended to be called immediately after load_raw() to give an at-a-glance
    picture of the data's structure before any panel construction or modelling.
    The 'panel_coverage' section is the most analytically important: it shows
    what fraction of the Sessions universe has a matching record in each of the
    other two sheets, which determines how much data is available under each
    missing-data assumption.

    Parameters
    ----------
    sessions, tasks, incidents:
        DataFrames as returned by load_raw().

    Returns
    -------
    dict with keys:
        'row_counts'       : {sheet_name: int}
        'unique_locations' : {sheet_name: int}
        'missing_pct'      : {sheet_name: {column: float}} — % missing per column.
        'monthly_counts'   : DataFrame indexed by month with one column per sheet.
        'panel_coverage'   : stats on how many Sessions rows match in tasks/incidents.
    """
    sheets = {"sessions": sessions, "tasks": tasks, "incidents": incidents}

    row_counts = {name: len(df) for name, df in sheets.items()}

    unique_locations = {
        name: int(df["charging_location_id"].nunique()) for name, df in sheets.items()
    }

    missing_pct = {
        name: (df.isnull().sum() / len(df) * 100).round(1).to_dict()
        for name, df in sheets.items()
    }

    # Per-month row counts for each sheet, formatted as "YYYY-MM" for readability.
    monthly_series = []
    for name, df in sheets.items():
        s = (
            df.groupby(df["month"].dt.strftime("%Y-%m"))
            .size()
            .rename(name)
        )
        monthly_series.append(s)
    monthly_counts = pd.concat(monthly_series, axis=1).sort_index()

    # Coverage: what fraction of Sessions (location, month) cells have a match.
    # Using merge with indicator is explicit and avoids MultiIndex isin() subtleties.
    universe = sessions[["charging_location_id", "month"]]
    total = len(universe)

    def _count_matched(other: pd.DataFrame) -> int:
        merged = universe.merge(
            other[["charging_location_id", "month"]],
            on=["charging_location_id", "month"],
            how="left",
            indicator=True,
        )
        return int((merged["_merge"] == "both").sum())

    tasks_matched = _count_matched(tasks)
    incidents_matched = _count_matched(incidents)

    panel_coverage = {
        "sessions_total": total,
        "tasks_matched": tasks_matched,
        "tasks_pct": round(tasks_matched / total * 100, 1),
        "incidents_matched": incidents_matched,
        "incidents_pct": round(incidents_matched / total * 100, 1),
    }

    return {
        "row_counts": row_counts,
        "unique_locations": unique_locations,
        "missing_pct": missing_pct,
        "monthly_counts": monthly_counts,
        "panel_coverage": panel_coverage,
    }
