% This script defines data inputs (JPMCI data and public sources), defines 
% household income groups, specifies output paths, and sets graph options.

% 0. Add main folder to path ==============================================
oldpath = path;
addpath(oldpath, extractBefore(string(pwd), '\robustness\'));


% 1. Chase data inputs ====================================================
% Input: job find ---------------------------------------------------------
% Input workbook
jobfind_input_path = '../../../../input/disclose/2023-06-23_disclosure_packet/jobfind/2023-06-18_ui_jobfind_for_export.xls';
eval(['jobfind_input_directory' '=jobfind_input_path;']);
save('jobfind_input_directory.mat','jobfind_input_directory');

% Input sheets
% Defined in main model folder.

% Input: spend ------------------------------------------------------------
spending_input_path = '../../../../input/disclose/2023-06-23_disclosure_packet/spend/ui_spend_for_export.xls';
eval(['spending_input_directory' '=spending_input_path;']);
save('spending_input_directory.mat','spending_input_directory');
model_data = 'matched_model_data_table';
save('spending_input_sheets.mat', 'model_data');

% Input: interupted time series shift 'inter_time_series_...' -------------
inter_time_series_input_path = '../../../../input/disclose/2023-06-23_disclosure_packet/jobfind/tables/table_effects_summary.csv';
eval(['inter_time_series_input_directory' '=inter_time_series_input_path;']);
save('inter_time_series_input_directory.mat', 'inter_time_series_input_directory');

% Input: duration elasticity (UIEIP estimate) -----------------------------
% (This is a model result that is read in for the plot comparing the UIEIP
% duration elasticity estimate with literature values.)
elasticity_uieip_input_path = '..\..\..\..\release\joint_spend_search_model\paper_figures\table_supplement_effects.csv';
eval(['elasticity_uieip_input_directory' '=elasticity_uieip_input_path;']);
save('elasticity_uieip_input_directory.mat', 'elasticity_uieip_input_directory');


% 2. Public data inputs ===================================================
% Input: duration elasticities from literature ----------------------------
% This csv has been created for this project (typed up numbers from a review
% paper), i.e., it is public information (but not a public dataset).
literature_elasticities_input_path = '..\..\..\..\input\public_data\literature_elasticities.csv';
eval(['literature_elasticities_input_directory' '=literature_elasticities_input_path;']);
save('literature_elasticities_input_directory.mat', 'literature_elasticities_input_directory');

% Public data input: BLS employment ---------------------------------------
bls_employment_input_path = '../../../../input/public_data/bls_payroll_emp_nonfarm_no_adj.xlsx';
eval(['bls_employment_input_directory' '=bls_employment_input_path;']);
save('bls_employment_input_directory.mat', 'bls_employment_input_directory');

% Public data input: Initial loss per state (elig_ui_reg_pua) -------------
initial_loss_input_path = '../../../../input/public_data/elig_ui_reg_pua.csv';
eval(['initial_loss_input_directory' '=initial_loss_input_path;']);
save('initial_loss_input_directory.mat', 'initial_loss_input_directory');

% Public data input: PUA claims -------------------------------------------
pua_claims_input_path = '../../../../input/public_data/decompose_pua.csv';
eval(['pua_claims_input_directory' '=pua_claims_input_path;']);
save('pua_claims_input_directory.mat', 'pua_claims_input_directory');

% Public data input: PAYEMS -----------------------------------------------
payems_input_path = '../../../../input/public_data/PAYEMS.xls';
eval(['payems_input_directory' '=payems_input_path;']);
save('payems_input_directory.mat', 'payems_input_directory');

% Public data input: PCE --------------------------------------------------
pce_input_path = '../../../../input/public_data/pce.xls';
eval(['pce_input_directory' '=pce_input_path;']);
save('pce_input_directory.mat', 'pce_input_directory');

% Public data input: UI benchmark -----------------------------------------
ui_benchmark_input_path = '../../../../input/public_data/ui_receipt_benchmarks.xlsx';
eval(['ui_benchmark_input_directory' '=ui_benchmark_input_path;'])
save('ui_benchmark_input_directory.mat', 'ui_benchmark_input_directory');


% 3. Calculate HH income quintiles ========================================
% Defined in main model folder.

% 4. Define release paths (for saving figures etc.) =======================
release_path = "../../../../release/joint_spend_search_model/";
release_path_paper = "../../../../release/joint_spend_search_model/paper_figures";
release_path_slides = "../../../../release/joint_spend_search_model/slides_figures";
save('release_paths.mat', 'release_path', 'release_path_paper', 'release_path_slides');


% 5. Graph options ========================================================
% Defined in main model folder.

% 6. Model parameters =====================================================
% Model fundamentals ------------------------------------------------------
% Defined in main model folder.

% Discount factors --------------------------------------------------------
%these discount factors are chosen so the model results eventually produce
%a quarterly MPC of 0.25 or a waiting design MPC of 0.43. This was done
%basically by guess and check on the end results
beta_target500MPC = .9896;
beta_targetwaiting = .978;
beta_oneperiodshock = 1.35;
delta = 0.845;
save('discountfactors.mat', 'beta_target500MPC', 'beta_targetwaiting', 'beta_oneperiodshock', 'delta') %saving these choices so that we make sure the discount factors used in later results are consistent with those used in solving for best fit
