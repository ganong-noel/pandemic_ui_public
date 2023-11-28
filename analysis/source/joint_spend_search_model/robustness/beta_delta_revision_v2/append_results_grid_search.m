clear
close all

% Go to folder with results
try
    cd release/array_job/
catch
    cd release\array_job\
end

% Specify job name & number of tasks/parameter values
jobname = 'rcc_2023_05_12d';
num_tasks = 91;
N_k = 10;
N_gamma = 10;
N_c_param = 10;

% Create empty objects (for appending the individual results in loops)
full_beta       = [];
full_delta      = [];
full_k          = [];
full_gamma      = [];
full_c_param    = [];
full_spend_e    = [];
full_spend_u    = [];
full_search     = [];
full_mpc        = [];
full_run_time   = [];

% Append results of all tasks
for taskid = 1:num_tasks
for i_k = 1:N_k
for i_gamma = 1:N_gamma
for i_c_param = 1:N_c_param

    disp(['Task ID ', num2str(taskid), '; index_k=', num2str(i_k), '; index_gamma=', num2str(i_gamma)])
    
    res = ['type_', jobname, '_task', num2str(taskid),'_k', num2str(i_k), '_g', num2str(i_gamma), '_c', num2str(i_c_param), '.mat'];
    eval(['load ', res])

    full_beta       = [full_beta;       beta_normal];
    full_delta      = [full_delta;      delta];
    full_k          = [full_k;          k];
    full_gamma      = [full_gamma;      gamma];
    full_c_param    = [full_c_param;    c_param];
    full_spend_e    = [full_spend_e;    mean_c_sim_e];
    full_spend_u    = [full_spend_u;    mean_c_sim_pandemic_expect];
    full_search     = [full_search;     mean_search_sim_pandemic_expect];
    full_mpc        = [full_mpc;        mpc_expect_waiting];
    full_run_time   = [full_run_time;   runtime];

end
end
end
end

num_simulations = size(full_beta, 1);

% Save appended results
save(['full_', jobname, '.mat'], ...
    'full_beta', 'full_delta', 'full_k', 'full_gamma', 'full_c_param', ...
    'full_spend_e', 'full_spend_u', 'full_search', 'full_mpc', 'full_run_time');

% Go back to robustness folder
try
    cd ../..
catch
    cd ..\..
end

% % Create directory for saving plots for this job
% try
%     path_plots = ['release\array_job\plots_', jobname];
%     eval(['mkdir ', path_plots]) %prints a warning if the directory already exists
% catch
%     path_plots = ['release/array_job/plots_', jobname];
%     eval(['mkdir ', path_plots]) %prints a warning if the directory already exists
% end
% 
% % Job find TS plot --------------------------------------------------------
% % Data series for job finding plot (from exit rates)
% load jobfind_input_directory.mat
% load jobfind_input_sheets.mat
% exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
% exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
% idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-11-20');
% exit_rates_data = exit_rates_data(idx, :);
% exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
% exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
% exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
% monthly_search_data = exit_rates_data_week_to_month.exit_not_to_recall';
% monthly_search_data = monthly_search_data(4:end);
% 
% % Plot
% load matlab_qual_colors.mat
% load graph_axis_labels_timeseries.mat
% figure
% p = patch([4 4 7 7], [0 1 1 0], [0.92 0.92 0.92], 'EdgeColor', 'none');
% set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
% hold on
% plot(4:11, monthly_search_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
% plot(4:11, full_search(:, [4 4:10]), 'Color', qual_orange, 'LineWidth', .2)
% legend('Data', ['By type (', num2str(num_simulations), ')'], 'Location', 'NorthEast', 'FontSize', 9)
% ylim([0 1])
% xticks([4 5 6 7 8 9 10 11])
% xticklabels(label_months_apr20_nov20)
% title('Job finding')
% set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
% set(gca, 'fontsize', 11);
% set(gca, 'Layer', 'top');
% set(gcf, 'PaperSize', [2.5 2]); %Keep the same paper size
% pbaspect([2.5 2 1])
% fig = gcf;
% saveas(fig, [path_plots, '/jobfind_', jobname , '.png'])
% 
% 
% % Spend TS plot -----------------------------------------------------------
% % Data series for spending plot
% load spending_input_directory.mat
% load spending_input_sheets.mat
% data_update = readtable(spending_input_directory, 'Sheet', model_data);
% idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
% idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
% data_update_e = data_update(idx_emp, :);
% data_update_u = data_update(idx_u, :);
% total_spend_e = data_update_e.value;
% total_spend_u = data_update_u.value;
% total_spend_e = total_spend_e(13:end);
% total_spend_u = total_spend_u(13:end);
% 
% us_v1=total_spend_u./total_spend_e-1;
% us_v1=us_v1-us_v1(1);
% spend_dollars_u_vs_e = us_v1 * mean(total_spend_u(1:2));
% 
% idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
% data_update_e = data_update(idx_emp, :);
% income_e = data_update_e.value;
% income_e = income_e(13:end);
% 
% % Convert model simulations to dollar deviations in U vs. E space
% mean_c_sim_pandemic_expect_dollars = full_spend_u ./ full_spend_e(:, 1:18) * total_spend_u(1) - total_spend_u(1);
% 
% % Plot
% figure
% p = patch([4 4 7 7], [-20 40 40 -20], [0.9 0.9 0.9], 'EdgeColor', 'none');
% set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
% hold on
% plot(1:11, spend_dollars_u_vs_e(1:11)/total_spend_u(1)*100, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
% plot(1:11, mean_c_sim_pandemic_expect_dollars(:, 1:11)/total_spend_u(1)*100, 'Color', qual_orange, 'LineWidth', .2)
% legend('Data', ['By type (', num2str(num_simulations), ')'], 'Location', 'NorthEast')
% ylim([-10 30])
% xlim([1 11])
% xticks([1 2 3 4 5 6 7 8 9 10 11])
% xticklabels(label_months_jan20_nov20)
% title('Spending')
% fig = gcf;
% saveas(fig, [path_plots, '/spend_', jobname , '.png'])
