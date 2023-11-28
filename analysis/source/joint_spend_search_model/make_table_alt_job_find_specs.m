
load release_paths.mat

table_alt_job_find_specs = readtable(fullfile(release_path_paper,'table_alt_job_find_specs.csv'));

% Round everything to two decimal places and get in correct format with trailing zeroes
table_alt_job_find_specs.Hazard_600 = round(table_alt_job_find_specs.Hazard_600, 2);
table_alt_job_find_specs.Hazard_600 = num2str(table_alt_job_find_specs.Hazard_600, '%.2f');
table_alt_job_find_specs.Hazard_300 = round(table_alt_job_find_specs.Hazard_300, 2);
table_alt_job_find_specs.Hazard_300 = num2str(table_alt_job_find_specs.Hazard_300, '%.2f');
table_alt_job_find_specs.Duration_elasticity_600 = round(table_alt_job_find_specs.Duration_elasticity_600, 2);
table_alt_job_find_specs.Duration_elasticity_600 = num2str(table_alt_job_find_specs.Duration_elasticity_600, '%.2f');
table_alt_job_find_specs.Duration_elasticity_300 = round(table_alt_job_find_specs.Duration_elasticity_300, 2);
table_alt_job_find_specs.Duration_elasticity_300 = num2str(table_alt_job_find_specs.Duration_elasticity_300, '%.2f');

%I include the two first columns with text

table_alt_job_find_specs.Row = [{'Baseline: Absolute pp'}; {'Relative percent change'}; {'Logit'}];
Aggregation = [{'Worker-level \raggedright'}; {'Benefit change deciles \raggedright'}; {'Worker-level \raggedright'}];
table_alt_job_find_specs = addvars(table_alt_job_find_specs, Aggregation, 'After', 'Row');


% Save to Latex

table2latex_numbers_only(table_alt_job_find_specs, char(strcat(release_path_paper, '/table_alt_job_find_specs.tex')));

