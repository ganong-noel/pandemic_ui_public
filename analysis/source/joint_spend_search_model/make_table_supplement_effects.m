
load release_paths.mat

table_supplement_effects = readtable(fullfile(release_path_paper,'table_supplement_effects.csv'));

% Round everything to two decimal places and get in correct format with trailing zeroes
table_supplement_effects.supplement600 = round(table_supplement_effects.supplement600, 2);
table_supplement_effects.supplement600 = num2str(table_supplement_effects.supplement600, '%.2f    ');
table_supplement_effects.supplement300 = round(table_supplement_effects.supplement300, 2);
table_supplement_effects.supplement300 = num2str(table_supplement_effects.supplement300, '%.2f    ');

% Break off elasticity part
elasticity_supplement_effects = table_supplement_effects(1:3, :);

% Rename columns
elasticity_supplement_effects.Row = [{'Best fit model'}; {'Statistical model: time-series'}; {'Statistical model: cross-section'}];

% Save elasticity part
%table2latex_numbers_only(elasticity_supplement_effects , char(strcat(release_path_paper, '/elasticity_supplement_effects.tex')));

% Break off MPC part
mpc_supplement_effects = table_supplement_effects(4:7, :);

% Rename columns
mpc_supplement_effects.Row = [{'1-month MPC out of 1st month of supplements'}; {'3-month MPC out of 1st 3 months of supplements'}; {'Total MPC through month supplement ends'}; {'Total MPC through 3 months after supplement ends'}];

% Save MPC part
table2latex_numbers_only(mpc_supplement_effects , char(strcat(release_path_paper, '/mpc_supplement_effects.tex')));
