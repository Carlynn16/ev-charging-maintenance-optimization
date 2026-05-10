# EV Charging Station Maintenance — Optimizing Technician Dispatch

A statistical analysis of incident rates across 638 EV charging stations to evaluate whether a uniform 2–4 week inspection cycle should be modulated by station utilization or by season.

📄 **[Read the full report (PDF)](EV_charging_maintenance_report.pdf)**

---

## Business question

An EV charging network operator runs 638 stations under a uniform technician dispatch policy: every station is inspected every 2–4 weeks, regardless of how heavily it is used or what time of year it is. The operator asked whether this fixed cycle should be adapted under one of two hypotheses:

- **H1.** Heavy-traffic stations may wear out faster → inspect them more often.
- **H2.** Failure rates may rise in spring/summer → tighten the cycle in those months.

The analysis evaluates both hypotheses on six months of operational data (Jan–Jun 2024) and translates the findings into a concrete operational recommendation.

---

## Headline findings

- **Utilization does not predict the per-visit incident rate.** Joint model rate ratio = 0.96 (95% CI [0.85, 1.08], p = 0.47). Heavily-used stations are not more failure-prone than quiet ones — in fact, the High-utilization tier is slightly more *consistent* across stations.
- **A spring elevation exists but is concentrated, not network-wide.** March/April/May show statistically significant rate ratios (1.58, 1.53, 1.35 vs. January). However, a tail-sensitivity check shows the March elevation is driven by ~19 specific stations: removing them collapses March's mean to the level of January and June.
- **The actionable signal is local, not seasonal.** Recommendation: do not modulate the dispatch cycle by traffic or by season; instead, identify the ~19 stations driving the March tail and follow up on them individually.
- **Findings are robust** to (a) the choice of missing-data assumption, (b) re-fit with a Negative Binomial family that explicitly models the overdispersion observed in the data (α = 1.50), and (c) tail-trimming.

---

## Methodological framing

The most important methodological step was reframing the unit of analysis. The intuitive approach — regressing raw incident counts on utilization — has a confound: incidents can only be recorded during technician visits, so `incidents = tasks × (incidents per task)`. A model fitted on raw counts cannot separate "busy stations fail more" (the question of interest) from "busy stations are inspected more thoroughly" (a mechanical artefact).

The fix is to model the **per-visit incident rate** directly. This is implemented as a Poisson GEE with `log(tasks_solved)` as offset, giving exponentiated coefficients that read as multiplicative effects on the rate.

**Statistical methods used:** Kruskal-Wallis and Spearman rank correlations (descriptive), bootstrap percentile confidence intervals, GEE Poisson with exchangeable working correlation and cluster-robust standard errors (joint model), Dunn's post-hoc with Bonferroni correction (pairwise seasonality), Negative Binomial GLM as a robustness extension.

---

## Repository structure

```
ev-charging-maintenance-optimization/
├── src/                                    # Reusable analysis modules
│   ├── data.py                             # Load, panel construction, missing-data assumptions
│   ├── plots.py                            # Shared plotting helpers (consistent style)
│   ├── tier_analysis.py                    # Phase 3: utilization tier analysis
│   ├── seasonality_analysis.py             # Phase 4: monthly comparisons
│   └── joint_model.py                      # Phase 5: GEE Poisson + Negative Binomial
├── notebooks/                              # Narrative analysis, end-to-end
│   ├── 01_data_exploration.ipynb           # Coverage and data-quality checks
│   ├── 02_eda.ipynb                        # Descriptive statistics and key visualizations
│   ├── 03_tier_analysis.ipynb              # Hypothesis 1: utilization-based scheduling
│   ├── 04_seasonality_analysis.ipynb       # Hypothesis 2: seasonal scheduling
│   ├── 05_joint_model.ipynb                # GEE Poisson with sensitivity check
│   └── 05b_negbin_robustness.ipynb         # Negative Binomial robustness
├── tests/                                  # 76 pytest tests across all modules
├── figures/                                # 9 PNG figures used in the report
├── EV_charging_maintenance_report.pdf  # Final deliverable (21 pages)
└── requirements.txt
```
---

## Reproducibility

```bash
# Clone the repository
git clone https://github.com/Carlynn16/ev-charging-maintenance-optimization.git
cd ev-charging-maintenance-optimization

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the test suite (should report 76 passing)
pytest tests/ -v

# Open the notebooks in order to reproduce the analysis
jupyter lab notebooks/
```

The raw data file (`data/session_stats.xlsx`) is excluded from the repository for client confidentiality. The pipeline expects this file at the path indicated; results in the report can be reproduced by running the notebooks once the file is in place.

---

## Limitations

The analysis is restricted to a single six-month window, so seasonal claims are about Jan–Jun 2024 specifically rather than an established annual pattern. Coverage of the data ramps up over the period (302 → 482 recorded visits/month from January to June), most plausibly reflecting an operational data-recording ramp-up; this affects the interpretation of January in particular. The Methods Appendix and Section 8 of the report discuss these and other limitations in detail.

---
