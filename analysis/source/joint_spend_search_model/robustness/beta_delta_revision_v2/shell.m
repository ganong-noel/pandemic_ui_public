% shell script for many-types version on RCC

%Define data inputs and parameters 
prelim;

% Solve many types (run on the RCC)
% This code solves the hyperbolic model for a 5-dimensional grid of
% present-bias parameter, discount factor, and three search parameters. The
% code is set up for an array job on the RCC. One run of the code is for a
% given combination of present-bias parameter and discount factor but
% solves for the grid of search parameters. The array job is run with a
% total number of tasks that equals the number of combinations of
% present-bias parameter and discount factor that the grid has. The results
% are saved for each type individually.
% inf_horizon_het_results_search_cost_search;

% Script to append the individual types (run on the RCC)
% append_results_grid_search;

% Analysis: find optimal weights between types & plots
analyze_results;
