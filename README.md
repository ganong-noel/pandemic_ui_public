---
editor_options: 
  markdown: 
    wrap: 72
---

# Partial Replication Kit for "Spending and Job Finding Impacts of Expanded Unemployment Benefits: Evidence from Administrative Micro Data"

By Peter Ganong, Fiona Greig, Pascal Noel, Daniel Sullivan, and Joseph
Vavra

Please send feedback and questions to
[ganong\@uchicago.edu](mailto:ganong@uchicago.edu){.email}.

[DOI will go here]

# Model

*Requirements: MATLAB R2021b*

## Directory Structure

### Inputs

1.  `analysis/input/disclose/` - Model inputs and targets from JPMorgan
    Chase Institute (JPMCI) data.
2.  `analysis/input/public_data/` - Model inputs and targets from
    publicly available data. See below for description of the datasets
    and their sources.

### Model codes

`analysis/source/joint_spend_search_model/` (including subdirectory
`robustness/`)

### Output (exhibits)

`analysis/release/joint_spend_search_model/paper_figures`

## Model codes details

All in `analysis/source/joint_spend_search_model/`

### Driver Script

The driver script `shell.m` runs the code. The shell file comments also
describes exactly which subroutines produce each model figure and table
from the paper. \* The code producing most main text results is in lines
14-50 and only takes a few minutes to run. \* `solve_best_fit_params.m`
is much more time consuming than the rest of the script but we provide
intermediate files so this step can be skipped if so desired. \*
Similarly, the `stimulus_check_size` related code and robustness are
also more time consuming and can be skipped if not specifically
interested in those results.

### Setup Script

The script `prelim.m` defines data inputs to the model (points to
`analysis/input/disclose/` and `analysis/input/public_data/`), sets
parameters, and specifies plotting options.

### Model Scripts

-   `solve_best_fit_params.m` - Calibrate the search parameters of
    different expectations assumptions and MPC targets and save them in
    various intermediate .mat files like
    `bestfit_target_waiting_MPC.mat`. This file takes several hours to
    run, but we provide the .mat files, so shell.m can be run without
    this time consuming step by commenting out this
    solve_best_fit_params line. This code relies on functions
    `pre_pandemic_fit_het_inf_horizon.m`,
    `sse_fit_het_inf_horizon_full.m`, `sse_fit_het_inf_horizon.m`, and
    `sse_fit_het_inf_horizon_onset.m` which solve a prepandemic
    calibration and the pandemic model for expiration and onset.
-   `inf_horizon_het_results.m`, `inf_horizon_het_results_onset.m` -
    Given search parameters calibrated above, this solves and simulates
    the main model for expiration and onset, respectively.
    -   Plots in these scripts include results from other calibrations:
        `prepandemic_results_target500MPC.m`,
        `inf_horizon_het_results_target500MPC.m`,
        `inf_horizon_het_results_nodiscountfactorshock.m`,
        `prepandemic_results_onset_target500MPC.m`,
        `inf_horizon_het_results_onset_target500MPC`.
-   `inf_horizon_het_results_stimulus_check_size.m`,
    `inf_horizon_het_results_stimulus_check_size_onetenth.m`,
    `inf_horizon_het_counterfactuals.m` - This code constructs the
    spending responses for stimulus checks vs. severance of various
    sizes. It is somewhat time consuming since not written very
    efficiently (about an hour), and results are only used for Figure
    13, so could be skipped if not interested in these counterfactuals.
-   `liquidity_effects_prepandemic.m` - Computes baseline effects of
    liquidity on job search to compare to Card, Chetty, Weber 2006
-   `make_table_agg_effects.m`, `make_table_mpc_for_paper.m`,
    `make_table_supplement_effects.m`,
    `make_table_alt_job_find_specs.m` - Format model outputs for the
    paper.
-   `plot_duration_elasticities.m` - Plot literature estimates of
    duration elasticities.
-   `pandemic_hazard_vs_duration_elasticity_constanteffects_v2.m` -
    Decomposes role of different channels in low elasticity (paper
    figure 11)
-   `liquidity_effects_on_mpcs.m` - some statistics related to liquidity
    changes that are briefly mentioned, otherwise this file is mostly
    deprecated
-   `inf_horizon_het_results_by_liquidity.m` - This redoes the main
    results but splitting separately by high and low liquidity
    households constructed in various ways

### Robustness Scripts

-   `inf_horizon_het_results_timeaggregation_target500MPC.m`,
    `inf_horizon_het_results_timeaggregation.m` - Code for constructing
    Figure A-18
-   Subdirectory `/robustness/beta_delta_revision_v2/` - Running the
    shell file in this folder creates robustness figure A-28. Note that
    the shell file provides the intermediate files and comments out the
    running of the model for a large set of parameters, which was run on
    a cluster with an array job. If you want to re-run this many hour
    grid search, see `grid_search.sh` and `grid_search_append.sh`
-   `test_homogeneity.m` - Homogeneity results which are briefly
    mentioned in Appendix C.2 (not included in shell since there are no
    specific numbers reported/saved from this code)

### Functions Written for this Project and Called by Routines Above

-   `average_duration.m` - This computes the average duration of
    unemployment given an exit rate
-   `elasticity_distortions_and_aggregates.m` - This computes duration
    elasticities as well as measures of aggregate distortions in
    response to the supplements in the pandemic
-   `search_elasticity_implications.m` - This computes a duration
    elasticity to a small change in benefits (i.e. a normal benefit
    elasticity)
-   `share_remaining_survival.m` - Computes share of unemployed
    remaining over time given exit hazard
-   `week_to_month_exit.m` - Converts weekly exit rates in data to
    monthly exit rates used in model

### Other Functions

-   `fminsearchbnd.m` - Bounded optimization function. Used for
    optimization in `solve_best_fit_params.m`.
-   `cab.m` - Close a subset of figures.
-   `hex2rgb.m` - Convert hexadecimal color code to RGB values.
-   `table2latex_numbers_only.m` - Convert MATLAB table to tex Table.

# JPMCI data

Some of the data used for this paper were prepared in JPMorganChase
Insitute's (JPMCI) secure computing facilities. Due to JPMCI's rules on
access and confidentiality, the programming code and analysis files
cannot be made available publicly. The analysis files and programming
code created by the authors will be available within JPMCI's secure
computing facilities until 2028, and can be requested by researchers
with approved projects (email `institute@jpmchase.com`). We grant any
researchers with appropriate approval to conduct research on JPMCI's
secure computing facilities access to these files. Below, we describe
the three key tables needed to replicate the analysis

## tables

-   weekly file with receipt of UI benefits and labor income including
    surrogate id for employer
-   monthly file with UI benefits, other income, several measures of
    spending, and checking account assets
-   file with demographics such as age, gender, states of residence, and
    Economic Impact Payment amount

# Public data

`analysis/input/public_data/` captures both inputs to the model and
inputs to benchmarking

[Rupsha will add additional information here]

*Requirements: Stata*

## Code in this directory

Stata code `decompose_pua.do` converts the data in `ap902.csv` to model
input `ui_receipt_benchmarks.xlsx`. `monthly_exit_rates.dta` is an
intermediate dataset saved by `decompose_pua.do`.

## Data from the Department of Labor (DOL)

ETA datasets can be found under the following link:
<https://oui.doleta.gov/unemploy/DataDownloads.asp> 
- `ap902.csv`: ETA902P 
- `elig_ui_reg_pua.csv`: this adds up the number of 'regular' UI
payments (ETA 5159) and PUA payments (ETA 902P) 
- `ui_receipt_benchmarks.xlsx`: sheets `month_final_payments` and
`month_initial_claims` are from ETA 5159, and sheet
`week_continued_claims` is from
<https://oui.doleta.gov/unemploy/claims.asp>

## Data from FRED

-   `PAYEMS.xls`: <https://fred.stlouisfed.org/series/payems>
-   `PCE.xls`: <https://fred.stlouisfed.org/series/PCE>

## Data from the Bureau of Labor Statistics (BLS)

-   `bls_payroll_emp_nonfarm_no_adj.xlsx`: select "Multi Screen" under
    "Employment, Hours, and Earnings - National" in section "Employment"
    of the following link <https://www.bls.gov/data/home.htm>
    (alternatively, the series can be accessed here:
    <https://beta.bls.gov/dataViewer/view/timeseries/CEU0000000001>)

## Data from other papers

-   `literature_elasticities.csv` is a set of elasticities from
    Schmieder and Von Wachter 2016

# Plots benchmarking JPMCI series to public data

The R script `driver.R` in `analysis/source/` runs the script
`diagnostic_benchmarking_plots.R`, also in `analysis/source/`, to
produce plots benchmarking JPMCI numbers to public data. It produces
four figures which are in `analysis/release/ui_benchmarking`:

-   Figure A-1: `hexmap_jpmci_sample.png`

-   Figure A-2: `diagnostic_levels_norm.png`;
    `state_hetero_inc_scatter.png`;
    `weekly_benefits_median_2019_mthly.png`
