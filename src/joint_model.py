"""
src/joint_model.py

Joint GEE Poisson model estimating the effects of utilization and month on the
per-visit incident rate, using log(tasks_solved) as an exposure offset and
accounting for repeated observations within station via GEE clustering.

Design notes
------------
Why GEE?
  Each station appears up to six times (one row per month). Treating these as
  independent observations would underestimate standard errors and inflate
  significance. GEE handles within-cluster correlation without requiring
  distributional assumptions about the latent random effect.

Why Poisson + offset instead of modelling the rate directly?
  Statsmodels GEE requires an integer-valued response. We model the *count*
  incidents_created with log(tasks_solved) as an offset; the offset shifts the
  linear predictor so the model is implicitly targeting incidents / tasks.

Why log-utilization?
  Utilization spans nearly three orders of magnitude (3–695 sessions/EVSE). A
  linear term would weight high-utilization stations implausibly. Log-utilization
  gives a proportional interpretation: a coefficient of β means a 2.7× increase
  in utilization multiplies the incident rate by exp(β).

Why exchangeable working correlation?
  Exchangeable assumes all within-station month pairs are equally correlated —
  a reasonable default when the correlation structure is unknown. GEE inference
  is robust to misspecification of the working correlation; exchangeable trades
  efficiency for robustness.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable as _Exchangeable

_MONTH_NAMES: dict[str, str] = {
    "2024-01": "January", "2024-02": "February", "2024-03": "March",
    "2024-04": "April",   "2024-05": "May",       "2024-06": "June",
}


def prepare_model_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a clean modelling DataFrame from a panel with incident_rate computed.

    Derives log_utilization and a string month_str column, then drops the small
    number of rows that would cause arithmetic problems:
      - tasks_solved == 0: would make log(tasks_solved) undefined (offset).
      - utilization <= 0: would make log(utilization) undefined.

    In practice these filters remove very few rows; the panel produced by
    apply_missing_data_assumption(..., 'missing_is_zero') already requires
    tasks_solved > 0, so only the utilization filter is active there.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel as returned by apply_missing_data_assumption(). Must contain
        'incidents_created', 'tasks_solved', 'utilization', 'month', and
        'charging_location_id'.

    Returns
    -------
    pd.DataFrame
        Modelling-ready DataFrame with columns:
          charging_location_id — cluster identifier
          incidents_created    — integer count (response)
          tasks_solved         — positive integer (offset denominator)
          log_utilization      — log(utilization)
          month_str            — 'YYYY-MM' string (patsy categorical)
    """
    required = {"incidents_created", "tasks_solved", "utilization",
                "month", "charging_location_id"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    df = panel.copy()
    df = df[df["tasks_solved"] > 0].copy()
    df = df[df["utilization"] > 0].copy()

    df["log_utilization"] = np.log(df["utilization"])
    df["month_str"] = df["month"].dt.to_period("M").astype(str)

    return df[["charging_location_id", "incidents_created",
               "tasks_solved", "log_utilization", "month_str"]].reset_index(drop=True)


def fit_gee_poisson(
    model_df: pd.DataFrame,
    ref_month: str = "2024-01",
):
    """Fit a GEE Poisson model with log(tasks_solved) as exposure offset.

    The working correlation structure is exchangeable (all within-station pairs
    equally correlated). GEE estimates are consistent even if the working
    correlation is misspecified; the exchangeable structure improves efficiency
    when moderate within-station correlation exists.

    The formula includes log_utilization and month dummies (reference = ref_month),
    so the model simultaneously estimates the utilization slope and the seasonal
    pattern while controlling for each.

    Parameters
    ----------
    model_df : pd.DataFrame
        As returned by prepare_model_data().
    ref_month : str
        Reference category for the month dummies. Default '2024-01' (January).

    Returns
    -------
    statsmodels GEEResults
        Fitted result with .params, .pvalues, .conf_int(), .fittedvalues, etc.
    """
    formula = (
        f'incidents_created ~ log_utilization + '
        f'C(month_str, Treatment("{ref_month}"))'
    )
    model = sm.GEE.from_formula(
        formula=formula,
        groups="charging_location_id",
        data=model_df,
        family=sm.families.Poisson(),
        cov_struct=_Exchangeable(),
        offset=np.log(model_df["tasks_solved"]),
    )
    return model.fit()


def _clean_term(name: str, ref_month: str = "2024-01") -> str:
    """Convert a raw statsmodels parameter name to an interpretable label."""
    if name == "log_utilization":
        return "log(utilization)"
    m = re.search(r"\[T\.(\d{4}-\d{2})\]", name)
    if m:
        ref_name = _MONTH_NAMES.get(ref_month, ref_month)
        month_name = _MONTH_NAMES.get(m.group(1), m.group(1))
        return f"{month_name} vs {ref_name}"
    return name


def extract_rate_ratios(result) -> pd.DataFrame:
    """Convert GEE coefficient estimates to rate ratios with 95 % CIs.

    Exponentiates the log-scale coefficients from the Poisson GEE. The
    intercept is excluded because its exponentiated value is the baseline rate
    for the reference month at log_utilization = 0, which is not meaningfully
    interpretable without further context.

    Parameters
    ----------
    result : statsmodels GEEResults
        Fitted result as returned by fit_gee_poisson().

    Returns
    -------
    pd.DataFrame
        One row per non-intercept coefficient with columns:
          term        — interpretable name (e.g. 'log(utilization)', 'March vs January')
          rate_ratio  — exp(coef), rounded to 3 d.p.
          ci_low      — exp(lower 95 % CI), rounded to 3 d.p.
          ci_high     — exp(upper 95 % CI), rounded to 3 d.p.
          p_value     — two-tailed p-value, rounded to 3 d.p.
    """
    params = result.params
    ci = result.conf_int()
    pvals = result.pvalues

    # Detect the reference month from the formula string.
    ref_match = re.search(r'Treatment\("(\d{4}-\d{2})"\)', result.model.formula)
    ref_month = ref_match.group(1) if ref_match else "2024-01"

    rows = []
    for term in params.index:
        if term in ("Intercept", "alpha"):  # alpha is NB dispersion param
            continue
        rows.append({
            "term":       _clean_term(term, ref_month),
            "rate_ratio": round(float(np.exp(params[term])),    3),
            "ci_low":     round(float(np.exp(ci.loc[term, 0])), 3),
            "ci_high":    round(float(np.exp(ci.loc[term, 1])), 3),
            "p_value":    round(float(pvals[term]),             3),
        })
    df = pd.DataFrame(rows)
    # Ensure log(utilization) is first, then months chronologically.
    is_util = df["term"] == "log(utilization)"
    return pd.concat([df[is_util], df[~is_util]], ignore_index=True)


def fit_nb_glm(
    model_df: pd.DataFrame,
    ref_month: str = "2024-01",
):
    """Negative Binomial GLM with cluster-robust standard errors.

    Uses statsmodels' MLE-based NegativeBinomial (NB2 parametrisation), which
    jointly estimates the dispersion parameter alpha alongside the regression
    coefficients. alpha = 0 collapses to Poisson; a positive alpha models the
    extra-Poisson variance directly rather than absorbing it post-hoc via
    sandwich standard errors as GEE does.

    Why smf.negativebinomial instead of sm.GLM(NegativeBinomial)?
        sm.GLM's NegativeBinomial family requires alpha to be fixed in advance,
        producing a quasi-NB model. smf.negativebinomial estimates alpha jointly
        via MLE, giving a proper NB fit and a reported alpha we can inspect.

    Cluster-robust SEs are requested via cov_type='cluster' clustered on
    charging_location_id. This makes the comparison with GEE Poisson
    apples-to-apples: both account for within-station repeated observations,
    one through marginal modelling (GEE) and one through a parametric mixture
    with robust SEs (NB-cluster).

    Parameters
    ----------
    model_df : pd.DataFrame
        As returned by prepare_model_data().
    ref_month : str
        Reference category for the month dummies. Default '2024-01'.

    Returns
    -------
    statsmodels NegativeBinomialResults
        Fitted result with .params (including 'alpha'), .pvalues, .conf_int(),
        .fittedvalues. Use extract_rate_ratios() to convert to rate ratios;
        the 'alpha' parameter is automatically excluded there.
    """
    import statsmodels.formula.api as smf

    formula = (
        f'incidents_created ~ log_utilization + '
        f'C(month_str, Treatment("{ref_month}"))'
    )
    model = smf.negativebinomial(
        formula,
        data=model_df,
        offset=np.log(model_df["tasks_solved"]),
    )
    return model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": model_df["charging_location_id"]},
    )


def dispersion_check(result) -> dict:
    """Pearson-residual-based dispersion diagnostic for the fitted GEE Poisson.

    Computes sum(pearson_residuals²) / (n − p) where n is the number of
    observations and p is the number of model parameters. For a well-specified
    Poisson model this ratio should be close to 1. Substantial overdispersion
    (ratio >> 1) suggests the Poisson variance assumption is violated and a
    negative binomial model may be more appropriate.

    Note: GEE inference is robust to misspecification of the variance function;
    this diagnostic informs whether a more flexible marginal model is warranted,
    not whether the GEE coefficient estimates are biased.

    Parameters
    ----------
    result : statsmodels GEEResults
        Fitted result as returned by fit_gee_poisson().

    Returns
    -------
    dict
        dispersion_ratio : float  — Pearson statistic / (n − p).
        n                : int    — number of observations.
        n_params         : int    — number of model parameters.
        interpretation   : str   — plain-English verdict.
    """
    mu = result.fittedvalues
    y = result.model.endog
    n = len(y)
    p = len(result.params)

    pearson_chi2 = float(np.sum((y - mu) ** 2 / mu))
    ratio = pearson_chi2 / (n - p)

    if ratio < 0.5:
        interp = "substantial underdispersion"
    elif ratio < 0.8:
        interp = "mild underdispersion"
    elif ratio <= 1.2:
        interp = "Poisson assumption holds (~1)"
    elif ratio <= 2.0:
        interp = "mild overdispersion"
    else:
        interp = "substantial overdispersion — consider negative binomial"

    return {
        "dispersion_ratio": round(ratio, 3),
        "n":        n,
        "n_params": p,
        "interpretation": interp,
    }


def fit_under_unknown(panel: pd.DataFrame):
    """Sensitivity check: refit the GEE model under the conservative missing-data assumption.

    Re-applies the 'missing_is_unknown' assumption to the raw panel (keeping only
    rows where both tasks_solved and incidents_created are observed), then runs
    prepare_model_data and fit_gee_poisson. Because 'missing_is_unknown' drops
    ~57 % of the missing_is_zero rows, the resulting model uses a smaller, more
    complete dataset with a higher mean incident rate (~24 % vs ~10 %).

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel as returned by build_panel() — *before* any missing-data
        assumption has been applied.

    Returns
    -------
    statsmodels GEEResults
        Fitted GEE result on the 'missing_is_unknown' subset.
    """
    from data import apply_missing_data_assumption
    panel_unknown = apply_missing_data_assumption(panel, "missing_is_unknown")
    model_df = prepare_model_data(panel_unknown)
    return fit_gee_poisson(model_df)
