# Model

_Requirements: MATLAB R2021b_

## Directory Structure
### Inputs
1. `analysis/input/disclose/` - Model inputs and targets from JPMorgan Chase Institute (JPMCI) data.
2. `analysis/input/public_data/` - Model inputs and targets from publicly available data. See below for description of the datasets and their sources.
3. `analysis/input/csvs_key_plots/literature_elasticities.csv` - Input for a plot overviewing literature estimates of duration elasticities.

### Model codes
`analysis/source/joint_spend_search_model/` (including subdirectory `robustness/`)

### Output (exhibits)
`analysis/release/joint_spend_search_model/paper_figures` 

## Model codes details 
All in `analysis/source/joint_spend_search_model/`

### Driver Script
The driver script `shell.m` runs the code. The shell file comments also describes exactly which subroutines produce each model figure and table from the paper.
* The code producing most main text results is in lines 14-50 and only takes a few minutes to run. 
* `solve_best_fit_params.m` is much more time consuming than the rest of the script but we provide intermediate files so this step can be skipped if so desired.
* Similarly, the `stimulus_check_size` related code and robustness are also more time consuming and can be skipped if not specifically interested in those results. 

### Setup Script
The script `prelim.m` defines data inputs to the model (points to `analysis/input/disclose/` and `analysis/input/public_data/`), sets parameters, and specifies plotting options.

### Model Scripts
- `solve_best_fit_params.m` - Calibrate the search parameters of different expectations assumptions and MPC targets and save them in various intermediate .mat files like `bestfit_target_waiting_MPC.mat`. This file takes several hours to run, but we provide the .mat files, so shell.m can be run without this time consuming step by commenting out this solve_best_fit_params line. This code relies on functions `pre_pandemic_fit_het_inf_horizon.m`, `sse_fit_het_inf_horizon_full.m`, `sse_fit_het_inf_horizon.m`, and `sse_fit_het_inf_horizon_onset.m` which solve a prepandemic calibration and the pandemic model for expiration and onset. 
- `inf_horizon_het_results.m`, `inf_horizon_het_results_onset.m` - Given search parameters calibrated above, this solves and simulates the main model for expiration and onset, respectively.
    - Plots in these scripts include results from other calibrations: `prepandemic_results_target500MPC.m`, `inf_horizon_het_results_target500MPC.m`, `inf_horizon_het_results_nodiscountfactorshock.m`, `prepandemic_results_onset_target500MPC.m`, `inf_horizon_het_results_onset_target500MPC`.
- `inf_horizon_het_results_stimulus_check_size.m`, `inf_horizon_het_results_stimulus_check_size_onetenth.m`, `inf_horizon_het_counterfactuals.m` - This code constructs the spending responses for stimulus checks vs. severance of various sizes. It is somewhat time consuming since not written very efficiently (about an hour), and results are only used for Figure 13, so could be skipped if not interested in these counterfactuals.
- `liquidity_effects_prepandemic.m` - Computes baseline effects of liquidity on job search to compare to Card, Chetty, Weber 2006
- `make_table_agg_effects.m`, `make_table_mpc_for_paper.m`, `make_table_supplement_effects.m`, `make_table_alt_job_find_specs.m` - Format model outputs for the paper.
- `plot_duration_elasticities.m` - Plot literature estimates of duration elasticities.
- `pandemic_hazard_vs_duration_elasticity_constanteffects_v2.m` - Decomposes role of different channels in low elasticity (paper figure 11)
- `liquidity_effects_on_mpcs.m` - some statistics related to liquidity changes that are briefly mentioned, otherwise this file is mostly deprecated
- `inf_horizon_het_results_by_liquidity.m` - This redoes the main results but splitting separately by high and low liquidity households constructed in various ways

### Robustness Scripts
- `inf_horizon_het_results_timeaggregation_target500MPC.m`, `inf_horizon_het_results_timeaggregation.m` - Code for constructing Figure A-18
- Subdirectory `/robustness/beta_delta_revision_v2/` - Running the shell file in this folder creates robustness figure A-28. Note that the shell file provides the intermediate files and comments out the running of the model for a large set of parameters, which was run on a cluster with an array job. If you want to re-run this many hour grid search, see `grid_search.sh` and `grid_search_append.sh`
- `test_homogeneity.m` - Homogeneity results which are briefly mentioned in Appendix C.2 (not included in shell since there are no specific numbers reported/saved from this code)

### Functions Written for this Project and Called by Routines Above
- `average_duration.m` - This computes the average duration of unemployment given an exit rate
- `elasticity_distortions_and_aggregates.m` - This computes duration elasticities as well as measures of aggregate distortions in response to the supplements in the pandemic
- `search_elasticity_implications.m` - This computes a duration elasticity to a small change in benefits (i.e. a normal benefit elasticity)
- `share_remaining_survival.m` - Computes share of unemployed remaining over time given exit hazard
- `week_to_month_exit.m` - Converts weekly exit rates in data to monthly exit rates used in model

### Other Functions
- `fminsearchbnd.m` - Bounded optimization function. Used for optimization in `solve_best_fit_params.m`. 
- `cab.m` - Close a subset of figures.
- `hex2rgb.m` - Convert hexadecimal color code to RGB values.
- `table2latex_numbers_only.m` - Convert MATLAB table to tex Table.

# Sources and code for the public datasets in `analysis/input/public_data/`

_Requirements: Stata_

## Code in this directory
Stata code `decompose_pua.do` converts the data in `ap902.csv` to model input `ui_receipt_benchmarks.xlsx`. `monthly_exit_rates.dta` is an intermediate dataset saved by `decompose_pua.do`.

## Data from the Department of Labor (DOL)
ETA datasets can be found under the following link: [https://oui.doleta.gov/unemploy/DataDownloads.asp](https://oui.doleta.gov/unemploy/DataDownloads.asp)
- `ap902.csv`: ETA 902P 
- `elig_ui_reg_pua.csv`: this adds up the number of 'regular' UI payments (ETA 5159) and PUA payments (ETA 902P)
- `ui_receipt_benchmarks.xlsx`: sheets `month_final_payments` and `month_initial_claims` are from ETA 5159, and sheet `week_continued_claims` is from [https://oui.doleta.gov/unemploy/claims.asp](https://oui.doleta.gov/unemploy/claims.asp)

## Data from FRED
- `PAYEMS.xls`: [https://fred.stlouisfed.org/series/payems](https://fred.stlouisfed.org/series/payems)
- `PCE.xls`: [https://fred.stlouisfed.org/series/PCE](https://fred.stlouisfed.org/series/PCE)

## Data from the Bureau of Labor Statistics (BLS)
- `bls_payroll_emp_nonfarm_no_adj.xlsx`: select "Multi Screen" under "Employment, Hours, and Earnings - National" in section "Employment" of the following link [https://www.bls.gov/data/home.htm](https://www.bls.gov/data/home.htm) (alternatively, the series can be accessed here: [https://beta.bls.gov/dataViewer/view/timeseries/CEU0000000001](https://beta.bls.gov/dataViewer/view/timeseries/CEU0000000001))

## Data from other papers
- `literature_elasticities.csv` is a set of elasticities from Schmieder and Von Wachter 2016 
