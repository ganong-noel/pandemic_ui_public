function fit_total = sse_fit_het_inf_horizon_full(pars, surprise)

    global monthly_search_data n_ben_profiles_allowed;
    
    load jobfind_input_directory.mat
    load jobfind_input_sheets.mat

    % Suppress variable name change warning
    warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')

    exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
    exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
    idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-11-20');
    exit_rates_data = exit_rates_data(idx, :);

    exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
    % For the exit variables we want the average exit probability at a monthly level
    exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
    exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
    monthly_search_data = exit_rates_data_week_to_month.exit_not_to_recall';
    monthly_search_data = monthly_search_data(4:end);

    n_ben_profiles_allowed = 2;
    fit1 = sse_fit_het_inf_horizon(pars, surprise);

    % Suppress variable name change warning
    warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')

    exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
    exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
    idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-11-01') & datenum(exit_rates_data.week_start_date) < datenum('2021-05-21');
    exit_rates_data = exit_rates_data(idx, :);

    % Smoothing the January spike
    %exit_rates_data = sortrows(exit_rates_data, {'type', 'cut', 'week_start_date'});
    % Set bad weeks to NaN
    weeks_to_fix_idx = datenum(exit_rates_data.week_start_date) >= datenum('2021-01-01') & datenum(exit_rates_data.week_start_date) <= datenum('2021-01-14');
    weeks_to_use_idx = datenum(exit_rates_data.week_start_date) > datenum('2021-01-14') & datenum(exit_rates_data.week_start_date) < datenum('2021-02-01');
    exit_rates_data.ExitRateToRecall(weeks_to_fix_idx) = mean(exit_rates_data.ExitRateToRecall(weeks_to_use_idx));
    exit_rates_data.ExitRateNotToRecall(weeks_to_fix_idx) = mean(exit_rates_data.ExitRateNotToRecall(weeks_to_use_idx));

    % Fill missing values
    exit_rates_data = fillmissing(exit_rates_data, 'next', 'DataVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'});

    exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
    % For the exit variables we want the average exit probability at a monthly level
    exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
    exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);

    monthly_search_data = exit_rates_data_week_to_month.exit_not_to_recall';

    n_ben_profiles_allowed = 3;
    fit2 = sse_fit_het_inf_horizon_onset(pars, 0);
    fit_total = fit1 + fit2;

end
