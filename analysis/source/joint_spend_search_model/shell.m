%This file runs all the code necessary to produce model results from the
%paper. The figures presented in the paper (incl. appendix) remain open
%after running the codes, figures presented in slides etc. will be closed.

clear

%Defines data inputs and parameters 
prelim;

%solve for the search parameters that best fit different targets

solve_best_fit_params;

clc
prepandemic_results_target500MPC;
inf_horizon_het_results_target500MPC; 
inf_horizon_het_results_nodiscountfactorshock;
inf_horizon_het_results; %paper figures A26 A27 9b 9a A20
prepandemic_results_onset_target500MPC;
inf_horizon_het_results_onset_target500MPC;
inf_horizon_het_results_onset; %paper figures 10a 10b A19

inf_horizon_het_results_stimulus_check_size; %note these next two subroutines are not very efficient. They re-rerun inf_horizon_het_results for various size transfers but much of this code isn't actually needed
inf_horizon_het_results_stimulus_check_size_onetenth;
inf_horizon_het_counterfactuals; %paper figure 13
liquidity_effects_prepandemic; %table_stats_for_text_liquidity


% Create tex tables for paper
make_table_agg_effects; %table_agg_effects
make_table_mpc_for_paper; %table_mpc_for_paper
make_table_supplement_effects; %mpc_supplement_effects
make_table_alt_job_find_specs; %table_alt_job_find_specs

% Plot comparing duration elasticity estimates with previous results in the literature
plot_duration_elasticities; %paper figure 8

pandemic_hazard_vs_duration_elasticity_constanteffects_v2; %paper figure 11
liquidity_effects_on_mpcs;

%% Results by liquidity ==================================================
inf_horizon_het_results_by_liquidity; % paper figure 12 (takes a little longer than most of the other subroutines, 10-15 min)


%% Collect stats for text =================================================
% Collect statistics which are reported in the text of the paper but do
% not appear in tables/figures.
clearvars -except -regexp fig_paper_*
load release_paths.mat
stats = struct2table(load('stats_for_text_model_miscellaneous.mat'));
writetable(stats, fullfile(release_path_paper, 'table_stats_for_text_model_miscellaneous.csv'))

%% Run robustness checks ==================================================

inf_horizon_het_results_timeaggregation_target500MPC; %execute this script first and load the results for the graph in the second script
inf_horizon_het_results_timeaggregation; %paper figure A18

% Hyperbolic model, allowing for multiple types
run("robustness\beta_delta_revision_v2\shell.m"); %paper figures A28a A28b A28c A28d A28e 

%test_homogeneity.m this file computes results which are briefly mentioned but not shown in appendix C.2




%% Keep figures open that are in the paper
cab fig_paper_A26 fig_paper_A27 fig_paper_9b fig_paper_9a fig_paper_A20 ...
    fig_paper_10a fig_paper_10b fig_paper_A19 fig_paper_13 fig_paper_8 ...
    fig_paper_11 fig_paper_12 fig_paper_A18 fig_paper_A28a fig_paper_A28b ...
    fig_paper_A28c fig_paper_A28d fig_paper_A28e

