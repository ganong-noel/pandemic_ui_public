% This script defines data inputs (JPMCI data and public sources), defines 
% household income groups, specifies output paths, and sets graph options.

% 1. Chase data inputs ====================================================
% Input: job find ---------------------------------------------------------
% Input workbook
jobfind_input_path = '../../input/disclose/2023-06-23_disclosure_packet/jobfind/2023-06-18_ui_jobfind_for_export.xls';
eval(['jobfind_input_directory' '=jobfind_input_path;']);
save('jobfind_input_directory.mat','jobfind_input_directory');

% Input sheets
fig1_df = 'weekly_summary_natl_export';
expiry_sample = 'expiry_sample'; 
onset_sample = 'onset_sample'; 
fig_a12b_onset = 'weekly_beta_new_job_onset_df';
fig_a12b_expiry = 'weekly_beta_new_job_expire_df';
per_change_overall = 'per_change_overall';
sheet_names = {'fig1_df','expiry_sample','onset_sample', 'fig_a12b_onset', ...
    'fig_a12b_expiry', 'per_change_overall'};
save('jobfind_input_sheets.mat',sheet_names{:});

marginal_effects_hazard_calc_inputs=readtable('../../input/disclose/2023-06-23_disclosure_packet/jobfind/tables/marginal_effects_hazard_calc_inputs.csv');
save('marginal_effects_hazard_calc_inputs.mat','marginal_effects_hazard_calc_inputs')
% Input: spend ------------------------------------------------------------
spending_input_path = '../../input/disclose/2023-06-18_disclosure_packet/spend/ui_spend_for_export.xls';
eval(['spending_input_directory' '=spending_input_path;']);
save('spending_input_directory.mat','spending_input_directory');
model_data = 'matched_model_data_table';
save('spending_input_sheets.mat', 'model_data');

% Input: interupted time series shift 'inter_time_series_...' -------------
inter_time_series_input_path = '../../input/disclose/2023-06-23_disclosure_packet/jobfind/tables/table_effects_summary.csv';
eval(['inter_time_series_input_directory' '=inter_time_series_input_path;']);
save('inter_time_series_input_directory.mat', 'inter_time_series_input_directory');

% Input: spending by liquidity_...' -------------
data_high_liq = readtable('../../input/disclose/2023-06-23_disclosure_packet/spend/tables/df_monthly_collapsed_high_liquidity_for_plot.csv');
data_low_liq = readtable('../../input/disclose/2023-06-23_disclosure_packet/spend/tables/df_monthly_collapsed_low_liquidity_for_plot.csv');
save('liquidity', "data_low_liq", "data_high_liq");

% Input: duration elasticity (UIEIP estimate) -----------------------------
% (This is a model result that is read in for the plot comparing the UIEIP
% duration elasticity estimate with literature values.)
elasticity_uieip_input_path = '..\..\release\joint_spend_search_model\paper_figures\table_supplement_effects.csv';
eval(['elasticity_uieip_input_directory' '=elasticity_uieip_input_path;']);
save('elasticity_uieip_input_directory.mat', 'elasticity_uieip_input_directory');


% 2. Public data inputs ===================================================
% Input: duration elasticities from literature ----------------------------
% This csv has been created for this project (typed up numbers from a review
% paper), i.e., it is public information (but not a public dataset).
literature_elasticities_input_path = '..\..\input\csvs_key_plots\literature_elasticities.csv';
eval(['literature_elasticities_input_directory' '=literature_elasticities_input_path;']);
save('literature_elasticities_input_directory.mat', 'literature_elasticities_input_directory');

% Public data input: BLS employment ---------------------------------------
bls_employment_input_path = '../../input/public_data/bls_payroll_emp_nonfarm_no_adj.xlsx';
eval(['bls_employment_input_directory' '=bls_employment_input_path;']);
save('bls_employment_input_directory.mat', 'bls_employment_input_directory');

% Public data input: Initial loss per state (elig_ui_reg_pua) -------------
initial_loss_input_path = '../../input/public_data/elig_ui_reg_pua.csv';
eval(['initial_loss_input_directory' '=initial_loss_input_path;']);
save('initial_loss_input_directory.mat', 'initial_loss_input_directory');

% Public data input: PUA claims -------------------------------------------
pua_claims_input_path = '../../input/public_data/decompose_pua.csv';
eval(['pua_claims_input_directory' '=pua_claims_input_path;']);
save('pua_claims_input_directory.mat', 'pua_claims_input_directory');

% Public data input: PAYEMS -----------------------------------------------
payems_input_path = '../../input/public_data/PAYEMS.xls';
eval(['payems_input_directory' '=payems_input_path;']);
save('payems_input_directory.mat', 'payems_input_directory');

% Public data input: PCE --------------------------------------------------
pce_input_path = '../../input/public_data/pce.xls';
eval(['pce_input_directory' '=pce_input_path;']);
save('pce_input_directory.mat', 'pce_input_directory');

% Public data input: UI benchmark -----------------------------------------
ui_benchmark_input_path = '../../input/public_data/ui_receipt_benchmarks.xlsx';
eval(['ui_benchmark_input_directory' '=ui_benchmark_input_path;'])
save('ui_benchmark_input_directory.mat', 'ui_benchmark_input_directory');


% 3. Calculate HH income quintiles ========================================
expiry_data_2019 = readtable(jobfind_input_directory, 'Sheet', expiry_sample);
idx = datenum(expiry_data_2019.week_start_date) <= datenum('2020-01-01');
expiry_data_2019 = expiry_data_2019(idx, :);
idx = (string(expiry_data_2019.type) == 'By rep rate quintile');
expiry_data_2019= expiry_data_2019(idx, :);
hh_income_data = grpstats(expiry_data_2019, 'cut', 'mean', 'DataVars', {'income_2019'});
hh_income_data = hh_income_data.mean_income_2019;
hh_income_data = sort(hh_income_data, 'Descend');
%note that 2 and 4 are chosen to make distribution symmetric
w(1) = hh_income_data(5) / hh_income_data(3);
w(2) = (hh_income_data(5) / hh_income_data(3) + 1) / 2;
w(3) = hh_income_data(3) / hh_income_data(3);
w(4) = (hh_income_data(1) / hh_income_data(3) + 1) / 2;
w(5) = hh_income_data(1) / hh_income_data(3);
save('hh_wage_groups', 'w');


% 4. Define release paths (for saving figures etc.) =======================
release_path = "../../release/joint_spend_search_model/";
release_path_paper = "../../release/joint_spend_search_model/paper_figures";
release_path_slides = "../../release/joint_spend_search_model/slides_figures";
save('release_paths.mat', 'release_path', 'release_path_paper', 'release_path_slides');


% 5. Graph options ========================================================
% Colors ------------------------------------------------------------------
% navy blue 004577, purple 8a89c4, green bbd976, light orange ffae5f, teal a2dadb, yellow e8c815
global qual_blue qual_purple qual_green qual_orange qual_yellow matlab_red_orange %qual_teal
qual_blue = hex2rgb('004577') / 256; % often used for data
qual_purple = hex2rgb('8a89c4') / 256;
qual_green = hex2rgb('bbd976') / 256; % often used for full model (surprise, with new mpc, all the other extra stuff etc.)
qual_orange = hex2rgb('ffae5f') /256; % often used for surprise (or expect $600 to continue/not expire)
qual_yellow = hex2rgb('e8c815') / 256;
matlab_red_orange = [0.8500, 0.3250, 0.0980]; % often used for expect (expect expiration of $600)
save('matlab_qual_colors.mat', 'qual_yellow', 'qual_orange', 'qual_green', 'qual_purple', 'qual_blue', 'matlab_red_orange')

% Define labels for time axes ---------------------------------------------
label_months_jan20_nov20 = {'Jan 20', 'Feb 20', 'Mar 20', 'Apr 20' 'May 20', 'Jun 20', 'Jul 20', 'Aug 20', 'Sep 20', 'Oct 20', 'Nov 20'};
label_months_apr20_nov20 = {'Apr 20', 'May 20', 'Jun 20', 'Jul 20', 'Aug 20', 'Sep 20', 'Oct 20', 'Nov 20'};
label_months_nov20_feb21 = {'Nov 20', 'Dec 20', 'Jan 21', 'Feb 21'};

save('graph_axis_labels_timeseries', 'label_months_jan20_nov20', 'label_months_apr20_nov20', 'label_months_nov20_feb21')


% 6. Model parameters =====================================================
% Model fundamentals ------------------------------------------------------
dt = 1/12; %model period 1 month
r = exp(.04 * dt) - 1; %interest rate
mu = 2; % risk aversion coefficient

% Aprime grid -------------------------------------------------------------
initial_a = .7; %A value of initial assets from the data, only relevant if use_initial_a is set to 1
aprimemin = 0;
aprimemax = 2000;
n_aprime = 100; %number of asset states

% Labor market parameters -------------------------------------------------
sep_rate = .028; %exogenous separation rate
% exog_find_rate = .3;
repshare = 0.21; %Replacement rate as share of wages (fit to match observed data on household income in different states, note this is quite different from what one would get from doing a worker level rep rate)
n_b = 13; %length of benefit profile (12 months of UI benefits in the pandemic + last period absorbing state which is no benefits)
FPUC_expiration = .345; %The fixed supplement added to benefits in the pandemic (note matched to observed data on household income time series, again produces results which differ from worker level replacement rates)
FPUC_onset = .1725;
LWAsize = 1300 * FPUC_expiration / (4.5 * 600);

data_series_jan_feb = readtable(jobfind_input_directory, 'Sheet', fig1_df);
data_series_jan_feb.exit_ui_rate = data_series_jan_feb.ExitRateToRecall + data_series_jan_feb.ExitRateNotToRecall;
data_series_jan_feb = data_series_jan_feb(datenum(data_series_jan_feb.week_start_date) >= datenum('2020-01-12') & datenum(data_series_jan_feb.week_start_date) < datenum('2020-03-01'), :); % Make the targets of Jan/Feb 2020 ui exit rates
preperiod_target_weekly = mean(data_series_jan_feb.exit_ui_rate);
preperiod_target = week_to_month_exit(preperiod_target_weekly);
exog_find_rate = preperiod_target;

save('model_parameters.mat', 'dt', 'r', 'mu', 'initial_a', 'aprimemin', 'aprimemax', 'n_aprime', 'sep_rate', 'exog_find_rate', 'repshare', 'n_b', 'FPUC_expiration', 'FPUC_onset', 'LWAsize')

% Discount factors --------------------------------------------------------
%these discount factors are chosen so the model results eventually produce
%a quarterly MPC of 0.25 or a waiting design MPC of 0.43. This was done
%basically by guess and check on the end results
beta_target500MPC = .99005;
beta_targetwaiting = .98107;
beta_oneperiodshock = 1.375;
save('discountfactors.mat', 'beta_target500MPC', 'beta_targetwaiting', 'beta_oneperiodshock') %saving these choices so that we make sure the discount factors used in later results are consistent with those used in solving for best fit
