
load release_paths.mat

table_agg_effects = readtable(fullfile(release_path_paper,'table_agg_effects.csv'));

%Round everything to one decimal place and get in correct format
table_agg_effects.Supp600_Apr_2020_Through_July_2020 = round(table_agg_effects.Supp600_Apr_2020_Through_July_2020, 1);
table_agg_effects.Supp600_Apr_2020_Through_July_2020 = num2str(table_agg_effects.Supp600_Apr_2020_Through_July_2020, '%.1f    ');
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021 = round(table_agg_effects.Supp300_Jan_2021_Through_Aug_2021, 1);
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021 = num2str(table_agg_effects.Supp300_Jan_2021_Through_Aug_2021, '%.1f    ');

% Add percent signs
table_agg_effects.Supp600_Apr_2020_Through_July_2020 = strcat(table_agg_effects.Supp600_Apr_2020_Through_July_2020, '\%');
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021 = strcat(table_agg_effects.Supp300_Jan_2021_Through_Aug_2021, '\%');

% Rename columns
table_agg_effects.Row = [{'Average Change in Aggregate Employment'}; {'Share of Aggregate Employment Gap Explained'}; {'Average Change in Aggregate Spending'}; {'Share of Aggregate Spending Gap Explained'}];

% Save LaTex
table2latex_numbers_only(table_agg_effects , char(strcat(release_path_paper, '/table_agg_effects.tex')));
