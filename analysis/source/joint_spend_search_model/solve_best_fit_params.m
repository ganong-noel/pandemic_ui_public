%This code solves for the best fit search cost parameters under a number of
%different versions of the model. Various best fit parameters are solved 
%and saved, and these are then loaded and used in later *results* files.

%The first specifications target a quarterly MPC of 0.25 out of $500 while 
%the later specifications target a waiting MPC of 0.43 (which will be the 
%"best fit" model). 

%Given the discount factor (again summarized by which MPC is targeted), 
%there are several versions of the model solved: a prepandemic calibration 
%which targets a prepandemic job-finding rate and duration elasticity, and
%models which target job-finding time-series during the pandemic under
%different expectations: an expect expiration model and a surprise 
%expiration model (here expect vs. surprise refers to the end of the $600).
 
%Search costs are picked to jointly match the job-finding rate for both the
%"expiration" exercise (Apr-Oct 2020 time series) and "onset" (Nov20-Feb21
%time series). The two simulations are done separately and then stitched 
%together.

%The *het* label is a holdover from earlier code when some versions of 
%model had no wage heterogeneity. Now we always include wage heterogeneity.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load hh_wage_groups.mat

rng('default')
display('target 500 MPC')

% On/off switches for sections of this script -----------------------------
%If any of these are set to 0, then will just load the best fit rather than
%re-running the optimization. If this whole solve_best_fit_params.m file is
%commented out in shell.m then the results will instead just be computed
%for the last saved results, so has same effect as setting all these to 0.
run_prepandemic = 1;
run_job_find_rate_with_500MPC = 1;
run_job_find_rate_target_waiting_MPC = 1;

%various parameters which are used in the model solution/objective function
%without having to pass their values explicitly through the function when
%using the matlab solver
global permLWA monthly_search_data infinite_dur dt initial_a mu r sep_rate repshare w FPUC_expiration FPUC_onset n_aprime n_b n_ben_profiles_allowed aprimemin aprimemax y exog_find_rate beta_normal beta_high use_initial_a
load model_parameters.mat

% Further parameters specific to this script ------------------------------
n_ben_profiles_allowed = 2; %This captures the surprise vs. expected expiration scenarios (note when moving to results files we simulate for several additional possible profiles but they aren't needed
%when simulating the series that are relevant for best fit, so they are dropped here to make the solutions and sims faster

% Discount factors --------------------------------------------------------
%The model allows for a one period long shock to the discount factor to
%match the decline in spend observed in the pandemic for employed workers.
%This beta_high for one period generates that (note the model would not
%converge if this lasted permanently since beta>>1/(1+r) but it only lasts
%for one month). This ends up playing a limited role in the analysis
load discountfactors.mat

% On/off switches for parts of the model ----------------------------------
permLWA = 0; %Some robustness checks treat LWA like it's permanent even though it's temporary in practice. If this is set to 1 then it solves that model instead
infinite_dur = 0; %Some versions of code solved model where benefits were expected to last forever instead of 6-12 months. Mostly don't use this anymore
include_recalls = true; %some versions of code solved things without recalls
use_initial_a = 0; %This allows for the possibility of an initial asset value in sims which might differ from steady state (the most obvious alternative being something from the data)
y = 1; %Place holder for the fixed average wage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-pandemic best-fit model: Target quarterly $500 MPC of .25
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if run_prepandemic == 0
    load bestfit_prepandemic.mat
else
    % Target $500 MPC of .25
    beta_normal = beta_target500MPC;
    beta_high = beta_oneperiodshock;

    disp('Pre-pandemic')

    % Initial parameters: for pre-pandemic we impose that intercept c in 
    % search cost is zero, so only two parameters
    pars_init(1) = 18.4629;
    pars_init(2) = 3.2123;

    % Preperiod target, need to convert old weekly to a monthly value
    data_series_jan_feb = readtable(jobfind_input_directory, 'Sheet', fig1_df);
    % Make an exit_ui_rate variable
    data_series_jan_feb.exit_ui_rate = data_series_jan_feb.ExitRateToRecall + data_series_jan_feb.ExitRateNotToRecall;
    % Make the targets of Jan/Feb 2020 ui exit rates
    % Restrict to only data past January 2020
    data_series_jan_feb = data_series_jan_feb(datenum(data_series_jan_feb.week_start_date) >= datenum('2020-01-12') & datenum(data_series_jan_feb.week_start_date) < datenum('2020-03-01'), :);
    preperiod_target_weekly = mean(data_series_jan_feb.exit_ui_rate);
    preperiod_target = week_to_month_exit(preperiod_target_weekly);
    
    %Set various tolerances for the numerical optimization
    no_max_pre = optimset('MaxIter', Inf, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .075, 'TolX', .075);
    
    %Defining the objective function for the pre pandemic model:
    fun = @(pars)pre_pandemic_fit_het_inf_horizon(pars, preperiod_target, infinite_dur, include_recalls);
    
    %Solve for the best fit parameters given that objective function:
    [pre_pandemic_fit(1, :), pre_pandemic_fit_val(1, 1)] = fminsearch(fun, pars_init, no_max_pre);
    
    %Sometimes try solving for multiple initial conditions in solver, 
    %this step would just solve for best fit amongst those:
    [val index] = min(pre_pandemic_fit_val);
    pre_pandemic_fit_match500MPC = pre_pandemic_fit(index, :);
    pre_pandemic_fit_val_match500MPC = val;

    save('bestfit_prepandemic.mat', 'pre_pandemic_fit_match500MPC'); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target pandemic job-find rates & quarterly $500 MPC of 0.25
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_job_find_rate_with_500MPC == 0
    load bestfit_target500mpcs
else
    % Target $500 MPC of .25
    beta_normal = beta_target500MPC;
    beta_high = beta_oneperiodshock;

    disp('Het Surprise: target $500 MPC')
    surprise = 1;
    pars_init = [115.5754, 1.6137, -.3377];
    no_max = optimset('MaxIter', Inf, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .1, 'TolX', .25);
    fun = @(pars)sse_fit_het_inf_horizon_full(pars, surprise);
    [sse_surprise_fit_het_full_match500MPC(1, :), sse_surprise_fit_het_val_full_match500MPC(1, 1)] = fminsearchbnd(fun, pars_init, [-inf -inf -inf], [inf inf 0], no_max);

    disp('Het Expect: target $500 MPC')
    surprise = 0;
    pars_init = [25.8473, .3964, -1.1319];
    no_max = optimset('MaxIter', Inf, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .25, 'TolX', .25);
    fun = @(pars)sse_fit_het_inf_horizon_full(pars, surprise);
    [sse_expect_fit_het_full_match500MPC(1, :), sse_expect_fit_het_val_full_match500MPC(1, 1)] = fminsearchbnd(fun, pars_init, [-inf -inf -inf], [inf inf 0], no_max);

    save('bestfit_target500mpcs.mat', 'sse_expect_fit_het_full_match500MPC', 'sse_surprise_fit_het_full_match500MPC')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Best fit model: Target pandemic job-find rates & waiting MPC of 0.43
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_job_find_rate_target_waiting_MPC == 0
    load bestfit_target_waiting_MPC
else
    % Target waiting MPC of .43
    beta_normal = beta_targetwaiting;
    beta_high = beta_oneperiodshock;

    disp('Het Surprise: target waiting')
    surprise = 1;
    pars_init = [115.5754, 1.6137, -.3377];
    no_max = optimset('MaxIter', Inf, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .05, 'TolX', .05);
    fun = @(pars)sse_fit_het_inf_horizon_full(pars, surprise);
    %Some earlier versions of code had problems if fed parameters where
    %c>0, but that has since fixed. But still using this fminsearchbnd as a
    %holdover from that
    [sse_surprise_fit_het_full(1, :), sse_surprise_fit_het_val_full(1, 1)] = fminsearchbnd(fun, pars_init, [-inf -inf -inf], [inf inf 10], no_max);

    disp('Het Expect: target waiting')
    pars_init = [25.8473, .3964, -1.1319];
    surprise = 0;
    no_max = optimset('MaxIter', Inf, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .25, 'TolX', .25);
    fun = @(pars)sse_fit_het_inf_horizon_full(pars, surprise);
    [sse_expect_fit_het_full(1, :), sse_expect_fit_het_val_full(1, 1)] = fminsearchbnd(fun, pars_init, [-inf -inf -inf], [inf inf 10], no_max);

    save('bestfit_target_waiting_MPC.mat', 'sse_expect_fit_het_full', 'sse_surprise_fit_het_full')
end


% Comparing implications for hazard elasticities --------------------------
load bestfit_prepandemic.mat
load bestfit_target_waiting_MPC
pars = [pre_pandemic_fit_match500MPC 0];
stat_for_text_hazard_prepan = search_elasticity_implications(pars, infinite_dur, include_recalls)
pars = sse_surprise_fit_het_full;
stat_for_text_hazard_pan = search_elasticity_implications(pars, infinite_dur, include_recalls)

save('stats_for_text_model_miscellaneous', 'stat_for_text_hazard_prepan', 'stat_for_text_hazard_pan')
