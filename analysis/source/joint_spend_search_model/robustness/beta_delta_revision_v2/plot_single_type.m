close all
clear

% Identify the types among those solved on the RCC which best-fit the four
% targets defined below. Plot their spend and job-find time series.

% Load results for all types from RCC job
load('release/array_job/full_rcc_2023_05_12d.mat')
search_model = full_search(:, [4 4:10]); %model timing

% Data series for job finding (from exit rates)
load jobfind_input_directory.mat
load jobfind_input_sheets.mat
exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-11-20');
exit_rates_data = exit_rates_data(idx, :);
exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
monthly_search_data = exit_rates_data_week_to_month.exit_not_to_recall';
monthly_search_data = monthly_search_data(4:end);

% Define targets
target1 = monthly_search_data;
target2 = [0 0 0 0 .4 .4 .4 .4];
target3 = [.5 * monthly_search_data(1:4), 2 * monthly_search_data(5:8)];

% Difference from targets
diff1 = sum((search_model - target1).^2, 2);
diff2 = sum((search_model - target2).^2, 2);
diff3 = sum((search_model - target3).^2, 2);

% Index of minimum
[val1, ind1] = min(diff1);
[val2, ind2] = min(diff2);
[val3, ind3] = min(diff3);

% Target 4: minimize anticipation, maximize jump
search_data_fd = diff(monthly_search_data);
search_model_fd = diff(search_model, 1, 2); %first difference in the correct dimension
ind_small_anticipation = search_model_fd(:, 3) < .01 & search_model(:, 8) < .92;
[val4, ind_tmp] = max(search_model_fd(ind_small_anticipation, 4));
ind4 = find(val4 == search_model_fd(:,4));

% %confirming that the index is correct
% list_index = cumsum(ones(80987, 1));
% subset_list = list_index(ind_small_anticipation);
% final_index = subset_list(ind_tmp);
% ind4 == final_index

% Plot job finding
figure
hold on
p = patch([4 4 7 7], [0 1 1 0], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
plot(4:11, target1, '--o', 'Color', 'blue')
plot(4:11, search_model(ind1,:), '-v', 'Color', 'blue')
plot(4:11, target2, '--o', 'Color', 'red')
plot(4:11, search_model(ind2,:), '-v', 'Color', 'red')
plot(4:11, target3, '--o', 'Color', 'black')
plot(4:11, search_model(ind3,:), '-v', 'Color', 'black')
plot(4:11, search_model(ind4,:), '-v', 'Color', 'green')
legend('Target 1: data', 'Best-fit 1', 'Target 2: 0 to 40%', 'Best-fit 2', 'Target 3: .5*data to 2*data', 'Best-fit 3', 'Best-fit 4', 'location', 'West')

% Data series for spending plot
load spending_input_directory.mat
load spending_input_sheets.mat
data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
total_spend_e = data_update_e.value;
total_spend_u = data_update_u.value;
total_spend_e = total_spend_e(13:end);
total_spend_u = total_spend_u(13:end);

us_v1=total_spend_u./total_spend_e-1;
us_v1=us_v1-us_v1(1);
spend_dollars_u_vs_e = us_v1 * mean(total_spend_u(1:2));

% Convert model simulations to dollar deviations in U vs. E space
mean_c_sim_pandemic_expect_dollars = full_spend_u ./ full_spend_e(:, 1:18) * total_spend_u(1) - total_spend_u(1);

% Plot spending
figure
hold on
p = patch([4 4 7 7], [-10 25 25 -10], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
plot(1:11, spend_dollars_u_vs_e(1:11)/total_spend_u(1)*100, '--o', 'Color', 'blue')
plot(1:11, mean_c_sim_pandemic_expect_dollars(ind1, 1:11)/total_spend_u(1)*100, 'Color', 'blue')
plot(1:11, mean_c_sim_pandemic_expect_dollars(ind2, 1:11)/total_spend_u(1)*100, 'Color', 'red')
plot(1:11, mean_c_sim_pandemic_expect_dollars(ind3, 1:11)/total_spend_u(1)*100, 'Color', 'black')
plot(1:11, mean_c_sim_pandemic_expect_dollars(ind4, 1:11)/total_spend_u(1)*100, 'Color', 'green')
legend('Data', 'Best-fit 1', 'Best-fit 2', 'Best-fit 3', 'Best-fit 4')

% Parameters
table(full_beta([ind1 ind2 ind3]), ...
    full_delta([ind1 ind2 ind3]), ...
    full_k([ind1 ind2 ind3]), full_gamma([ind1 ind2 ind3]), ...
    full_c_param([ind1 ind2 ind3]), ...
    'VariableNames', {'beta', 'delta', 'k', 'gamma', 'c_param'})
