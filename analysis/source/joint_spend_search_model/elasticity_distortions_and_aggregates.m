% Computes duration elasticity and employment distortion
function [elasticity, employment_distortion, ave_diff_employment, share_unemployment_reduced, employment_FPUC, employment_noFPUC, monthly_spend_pce, monthly_spend_no_FPUC, total_hazard_elasticity,newjob_hazard_elasticity,newjob_duration_elasticity, total_percent_change,elasticity_prepanlevel_newjob,elasticity_prepanlevel_both,elasticity_prepanlevel_recall,table_elasticity_comparisons] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed)

    load ui_benchmark_input_directory.mat
    load bls_employment_input_directory.mat
    load initial_loss_input_directory.mat
    load pua_claims_input_directory.mat
    load payems_input_directory.mat
    load pce_input_directory.mat

    % Read in reference data for the months considered
    reference_months = readtable(ui_benchmark_input_directory);
    reference_months = unique(reference_months(:, {'month'}));
    % Collapse to the monthly level
    % Special version of reference month for counting initial losses that includes the last two weeks in March and hence March
    reference_months_with_march = reference_months(datenum(reference_months.month) >= datenum('2020-03-01'), :);
    % Standard version starts at date_sim_start
    reference_months = reference_months(datenum(reference_months.month) >= datenum(date_sim_start), :);
    % Renaming to match the rest of the code
    reference_months = renamevars(reference_months, 'month', 'year_month');

    % Create index for weeks or months
    reference_months_with_march.index = [1:height(reference_months_with_march)]';
    reference_months.index = [1:height(reference_months)]';

    % Add recall_probs to new job exits to get total exit rates
    total_exit_rate_FPUC = newjob_exit_rate_FPUC + recall_probs;
    total_exit_rate_no_FPUC = newjob_exit_rate_no_FPUC + recall_probs;

%{
    exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
    exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
    %idx = datenum(exit_rates_data.week_start_date) >= datenum('2019-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-01-01');
    %exit_rates_data = exit_rates_data(idx, :);
    exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
    % For the exit variables we want the average exit probability at a monthly level
    exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
    exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);

    recall_2019=exit_rates_data_week_to_month.exit_to_recall;
    newjob_2019=exit_rates_data_week_to_month.exit_not_to_recall;

    recall_2019_aprstart=[recall_2019(4:end); recall_2019(1:3)];
    recall_2019_aprstart(end:1000)=mean(recall_2019);
%}

    load jobfind_input_directory.mat 
    load jobfind_input_sheets.mat
    % Preperiod target, need to convert old weekly to a monthly value
    data_series_jan_feb = readtable(jobfind_input_directory, 'Sheet', fig1_df);
    % Make an exit_ui_rate variable
    data_series_jan_feb.exit_ui_rate = data_series_jan_feb.ExitRateToRecall + data_series_jan_feb.ExitRateNotToRecall;
    % Make the targets of Jan/Feb 2020 ui exit rates
    % Restrict to only data past january 2020
    data_series_jan_feb = data_series_jan_feb(datenum(data_series_jan_feb.week_start_date) >= datenum('2020-01-01') & datenum(data_series_jan_feb.week_start_date) < datenum('2020-03-01'), :);
    %data_series_jan_feb = data_series_jan_feb(datenum(data_series_jan_feb.week_start_date) < datenum('2020-03-01'), :);
    preperiod_target_weekly = mean(data_series_jan_feb.exit_ui_rate);
    preperiod_target = week_to_month_exit(preperiod_target_weekly);
    preperiod_target_weekly2 = mean(data_series_jan_feb.ExitRateToRecall);
    preperiod_target2 = week_to_month_exit(preperiod_target_weekly2);

    

    total_exit_rate_FPUC_prepanlevel=newjob_exit_rate_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) +recall_probs;
    total_exit_rate_no_FPUC_prepanlevel = newjob_exit_rate_no_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) + recall_probs;

    newjob_exit_rate_FPUC_prepanlevel=newjob_exit_rate_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1)));
    newjob_exit_rate_no_FPUC_prepanlevel=newjob_exit_rate_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1)));

    total_exit_rate_FPUC_prepanlevel_both=newjob_exit_rate_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) +recall_probs*preperiod_target2/mean(recall_probs(3:5));
    total_exit_rate_no_FPUC_prepanlevel_both = newjob_exit_rate_no_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) + recall_probs*preperiod_target2/mean(recall_probs(3:5));

    %total_exit_rate_FPUC_prepanlevel_both=newjob_exit_rate_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) +.08;
    %total_exit_rate_no_FPUC_prepanlevel_both = newjob_exit_rate_no_FPUC*(preperiod_target/mean(newjob_exit_rate_no_FPUC(1))) + .08;

    total_exit_rate_FPUC_prepanlevel_recall=newjob_exit_rate_FPUC +recall_probs*preperiod_target2/mean(recall_probs(3:5));
    total_exit_rate_no_FPUC_prepanlevel_recall = newjob_exit_rate_no_FPUC + recall_probs*preperiod_target2/mean(recall_probs(3:5));

    %total_exit_rate_FPUC_prepanlevel_recall=newjob_exit_rate_FPUC +.08;
    %total_exit_rate_no_FPUC_prepanlevel_recall = newjob_exit_rate_no_FPUC + .08;

    % Produce elasticity calculated over the period of interest
    elasticity = (average_duration(total_exit_rate_FPUC(t_distortion_start-1:end)) / average_duration(total_exit_rate_no_FPUC(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;
    elasticity_prepanlevel_newjob=(average_duration(total_exit_rate_FPUC_prepanlevel(t_distortion_start-1:end)) / average_duration(total_exit_rate_no_FPUC_prepanlevel(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;
    elasticity_prepanlevel_both=(average_duration(total_exit_rate_FPUC_prepanlevel_both(t_distortion_start-1:end)) / average_duration(total_exit_rate_no_FPUC_prepanlevel_both(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;
    elasticity_prepanlevel_recall=(average_duration(total_exit_rate_FPUC_prepanlevel_recall(t_distortion_start-1:end)) / average_duration(total_exit_rate_no_FPUC_prepanlevel_recall(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;



    newjob_duration_elasticity = (average_duration(newjob_exit_rate_FPUC(t_distortion_start-1:end)) / average_duration(newjob_exit_rate_no_FPUC(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;
    newjob_duration_elasticity_prepanlevel = (average_duration(newjob_exit_rate_FPUC_prepanlevel(t_distortion_start-1:end)) / average_duration(newjob_exit_rate_no_FPUC_prepanlevel(t_distortion_start-1:end)) - 1) / perc_change_benefits_data;

    %total_hazard_elasticity=mean((total_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1)./total_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1)-1)/ perc_change_benefits_data);
    %newjob_hazard_elasticity=mean((newjob_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1)./newjob_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1)-1)/ perc_change_benefits_data);
   
    total_hazard_elasticity=(mean(total_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))./mean(total_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
    total_hazard_elasticity_prepanlevel_newjob=(mean(total_exit_rate_no_FPUC_prepanlevel(t_distortion_start-1:t_distortion_end-1))./mean(total_exit_rate_FPUC_prepanlevel(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
    total_hazard_elasticity_prepanlevel_recall=(mean(total_exit_rate_no_FPUC_prepanlevel_recall(t_distortion_start-1:t_distortion_end-1))./mean(total_exit_rate_FPUC_prepanlevel_recall(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
    total_hazard_elasticity_prepanlevel_both=(mean(total_exit_rate_no_FPUC_prepanlevel_both(t_distortion_start-1:t_distortion_end-1))./mean(total_exit_rate_FPUC_prepanlevel_both(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
    
    

    newjob_hazard_elasticity=(mean(newjob_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))./mean(newjob_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
    newjob_hazard_elasticity_prepanlevel_newjob=(mean(newjob_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))./mean(newjob_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1))-1)/ perc_change_benefits_data;
   


    elasticity_arc=(average_duration(total_exit_rate_FPUC(t_distortion_start-1:end)) - average_duration(total_exit_rate_no_FPUC(t_distortion_start-1:end))) /(.5*((average_duration(total_exit_rate_FPUC(t_distortion_start-1:end)) + average_duration(total_exit_rate_no_FPUC(t_distortion_start-1:end)))))/perc_change_benefits_data;
    newjob_duration_elasticity_arc=(average_duration(newjob_exit_rate_FPUC(t_distortion_start-1:end)) - average_duration(newjob_exit_rate_no_FPUC(t_distortion_start-1:end))) /(.5*((average_duration(newjob_exit_rate_FPUC(t_distortion_start-1:end)) + average_duration(newjob_exit_rate_no_FPUC(t_distortion_start-1:end)))))/perc_change_benefits_data;
    
    total_hazard_elasticity_arc=(mean(total_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))-mean(total_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1)))/(.5*(mean(total_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))+mean(total_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1))))/ perc_change_benefits_data;
    newjob_hazard_elasticity_arc=(mean(newjob_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))-mean(newjob_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1)))/(.5*(mean(newjob_exit_rate_no_FPUC(t_distortion_start-1:t_distortion_end-1))+mean(newjob_exit_rate_FPUC(t_distortion_start-1:t_distortion_end-1))))/perc_change_benefits_data;


    table_elasticity_comparisons=table();
    table_elasticity_comparisons.Duration_Elasticity('Baseline')=elasticity;
    table_elasticity_comparisons.Duration_Elasticity('PrePandemicNewJobRate')=elasticity_prepanlevel_newjob;
    table_elasticity_comparisons.Duration_Elasticity('PrePandemicRecallRate')=elasticity_prepanlevel_recall;
    table_elasticity_comparisons.Duration_Elasticity('PrePandemicBoth')=elasticity_prepanlevel_both;
    table_elasticity_comparisons.Hazard_Elasticity('Baseline')=total_hazard_elasticity;
    table_elasticity_comparisons.Hazard_Elasticity('PrePandemicNewJobRate')=total_hazard_elasticity_prepanlevel_newjob;
    table_elasticity_comparisons.Hazard_Elasticity('PrePandemicRecallRate')=total_hazard_elasticity_prepanlevel_recall;
    table_elasticity_comparisons.Hazard_Elasticity('PrePandemicBoth')=total_hazard_elasticity_prepanlevel_both;

    % Produce distortions

    % Read in BLS Employment Data
    % Data is the employed, not seasonally adj option here: https://www.bls.gov/webapps/legacy/cpsatab1.htm
    bls_employment = readtable(bls_employment_input_directory);
    bls_employment.Properties.VariableNames = {'year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
    bls_employment = stack(bls_employment, 2:13);
    bls_employment.Properties.VariableNames = {'year', 'month', 'emp'};
    bls_employment.year = char(int2str(bls_employment.year));
    bls_employment.year_month = strcat(bls_employment.year, repmat('-', height(bls_employment), 1), char(bls_employment.month));
    bls_employment.year_month = datetime(bls_employment.year_month, 'InputFormat', 'yyyy-MMM');
    bls_employment = rmmissing(bls_employment);
    bls_employment = bls_employment(:, {'year_month', 'emp'});
    % Select relevant data
    bls_employment = innerjoin(bls_employment, reference_months);
    bls_employment = bls_employment.emp * 1000;

    % Initial loss of employment from the data, starter pool of unemployed
    % There is a value of this for each month
    initial_loss = readtable(initial_loss_input_directory);
    initial_loss = renamevars(initial_loss, 'month', 'year_month');
 
    % Group by month and summarize as a sum across states
    initial_loss = rowfun(@sum, initial_loss, 'InputVariables', 'all_first_payments', 'GroupingVariables', 'year_month', 'OutputVariableName', 'all_first_payments');

    % Select weeks of interest
    initial_loss = innerjoin(initial_loss, unique(reference_months(:, 'year_month')));

    % Sorting for convenience
    initial_loss = sortrows(initial_loss, 'year_month');

    % Read in PUA claims and remove them from the counts
    pua_claims = readtable(pua_claims_input_directory);
    pua_claims.year = str2double(regexprep(pua_claims.month, '([0-9]+)(m)([0-9]+)', '$1'));
    pua_claims.month = str2double(regexprep(pua_claims.month, '([0-9]+)(m)([0-9]+)', '$3'));
    pua_claims.day = ones(height(pua_claims), 1);
    pua_claims.year_month = datetime(pua_claims.year, pua_claims.month, pua_claims.day);
    initial_loss = outerjoin(initial_loss, pua_claims(:, {'year_month', 'entry_self_employed', 'continuing_claims_self_employed'}), 'MergeKeys', true, 'Type', 'left');

    % In the onset and other cases where we start later we will need to add continued claims from the week before date_sim_start to the initial loss
    if (datenum(date_sim_start) ~= datenum('2020-04-01'))
        cc = readtable(ui_benchmark_input_directory, 'sheet', 'week_continued_claims');
        % Sum up claims
        cc = rowfun(@sum, cc, 'InputVariables', 'total_cc', 'GroupingVariables', 'week', 'OutputVariableName', 'cc');
        % Keep only dates before the sim start and pick the closest date
        cc = cc(datenum(cc.week) < datenum(date_sim_start), :);
        [~, closest_ind] = min(abs(datenum(cc.week) - datenum(date_sim_start)));
        % Add to initial loss
        initial_loss.all_first_payments(1) = initial_loss.all_first_payments(1) + cc.cc(closest_ind);
    end

    % If we want to include self-employed we zero out PUA material from the appropriate tables
    if include_self_employed == 1
        initial_loss.entry_self_employed = zeros(height(initial_loss.entry_self_employed), 1);
        pua_claims.continuing_claims_self_employed = zeros(height(pua_claims.continuing_claims_self_employed), 1);
    end

    % Correcting for PUA
    % Standard expiry case
    if (datenum(date_sim_start) == datenum('2020-04-01'))
        % Remove entry into self employment for every month
        initial_loss.all_first_payments = initial_loss.all_first_payments - initial_loss.entry_self_employed;
    % In the onset and other cases for the first period we actually want to not remove entry to self employment, but instead remove continued claims from self-employment from the period before
    % For the other periods we remove entry into self employment from the month before
    else
        initial_loss.all_first_payments(1) = initial_loss.all_first_payments(1) - pua_claims.continuing_claims_self_employed(pua_claims.year_month == (date_sim_start - calmonths(1)));
        % There's a weird date-shift for the following line, where PUA entry subtracted is from the previous month
        initial_loss.all_first_payments(2:end) = initial_loss.all_first_payments(2:end) - initial_loss.entry_self_employed(1:end - 1);
    end 

    % Finish conversion to vector
    initial_loss = initial_loss.all_first_payments;
    initial_loss=max(initial_loss,0);

    % Vectors to store those remaining on UI
    number_remaining_FPUC = zeros(1000, 1);
    number_remaining_no_FPUC = zeros(1000, 1);

    % Calculate the employment distortion for each monthly cohort
    for i = 1:length(bls_employment)

        % Make shifted series based on the month starts
        search_series_shifted_FPUC = total_exit_rate_FPUC(i:1000);
        search_series_shifted_no_FPUC = total_exit_rate_no_FPUC(i:1000);

        % Initial periods dropped above, so fill in the end of vector w/ constant hazard to make length 1000
        search_series_shifted_FPUC(1000 - (i - 1):1000) = search_series_shifted_FPUC(1000 - (i - 1));

        % Calculate the number remaining as the initial level of unemployed multiplied by survival
        number_remaining_FPUC(i:1000, i) = initial_loss(i) * share_remaining_survival(search_series_shifted_FPUC(1:1000 - (i - 1)));

        % Do these steps for the no supplement series also
        search_series_shifted_no_FPUC(1000 - (i - 1):1000) = search_series_shifted_no_FPUC(1000 - (i - 1));

        number_remaining_no_FPUC(i:1000, i) = initial_loss(i) * share_remaining_survival(search_series_shifted_no_FPUC(1:1000 - (i - 1)));

    end

    % Changes in unemployment are relative to the counterfactual of no supplement; these should be positive
    changes_in_unemploy = number_remaining_FPUC - number_remaining_no_FPUC;

    % The impact of the supplement on person-weeks is the sum of the changes in unemp across all month cohorts from the start to end of the desired period of calculation
    % Distortion is difference in person-weeks due to the supplement divided by calculated total employment without the supplement

    d_total_person_weeks_or_months = sum(sum(changes_in_unemploy(t_distortion_start:t_distortion_end, :)));
    %d_monthly_person_weeks_or_months = sum(changes_in_unemploy(t_distortion_start:t_distortion_end, :),2);
    %monthly_change=zeros(length(d_monthly_person_weeks_or_months),1);
    %monthly_change(1)=d_monthly_person_weeks_or_months(1);
    %monthly_change(2:end)=d_monthly_person_weeks_or_months(2:end)-d_monthly_person_weeks_or_months(1:end-1);
    %mean_monthly_change=(d_total_person_weeks_or_months/1000000)/(t_distortion_end - t_distortion_start+1);
    employment_distortion = 100 * (d_total_person_weeks_or_months) ./ (sum(bls_employment(t_distortion_start-1:t_distortion_end-1)) + d_total_person_weeks_or_months);
    ave_diff_employment = (d_total_person_weeks_or_months) / (1000000*(t_distortion_end - t_distortion_start+1));
    total_diff_employment = sum(changes_in_unemploy(t_distortion_end, :)) / 1000000;
    %ave_diff_employment=total_diff_employment/(t_distortion_end - t_distortion_start+1);
    %ave_diff_employment=sum(changes_in_unemploy(t_distortion_end, :))/(bls_employment(t_distortion_end)-bls_employment(t_distortion_start));
    share_unemployment_reduced = sum(changes_in_unemploy(t_distortion_end, :)) / sum(number_remaining_FPUC(t_distortion_end, :));
    total_percent_change=total_diff_employment/(bls_employment(t_distortion_end-1)/1000000);
    
    for i = 1:1000
        total_number_remaining_unemployed_FPUC(i) = sum(number_remaining_FPUC(i, :));
        total_number_remaining_unemployed_no_FPUC(i) = sum(number_remaining_no_FPUC(i, :));
    end

    % Load in employment data for aggregates calculations
    bls_emp_long = readtable(payems_input_directory);
    bls_emp_long.Properties.VariableNames([1 2]) = {'year_month', 'Employment'};
    bls_emp_long.Employment = bls_emp_long.Employment * 1000;
    % No FPUC counterfactual- start with actual
    bls_emp_long.Employment_noFPUC = bls_emp_long.Employment;

    % Sum up changes in unemploy
    for i = 1:1000
        summed_changes_in_unemploy(i) = sum(changes_in_unemploy(i, :));
    end

    % Need to extend year_month to length 1000
    extra_year_month = dateshift(reference_months.year_month(end), 'start', 'month', 1:1000 - length(reference_months.year_month))';
    extended_year_month = [reference_months.year_month; extra_year_month];
    changes_in_unemploy = table(extended_year_month, summed_changes_in_unemploy');
    changes_in_unemploy.Properties.VariableNames = {'year_month', 'changes_in_unemploy'};
    % Right-join the employment data to the distortions data
    bls_emp_long = outerjoin(bls_emp_long, changes_in_unemploy);
    % Add the change in unemploy distortion to no FPUC column
    bls_emp_long.Employment_noFPUC = bls_emp_long.Employment_noFPUC + bls_emp_long.changes_in_unemploy;

    if include_self_employed == 0

        % Output the employment data
        employment_FPUC = bls_emp_long.Employment;
        employment_noFPUC = bls_emp_long.Employment_noFPUC;

    end

    if include_self_employed == 1

        % Use unemployment shares and spending of unemployed to get counterfactual spending
        unemployed_share_FPUC = total_number_remaining_unemployed_FPUC(1:61 - 44)' ./ (bls_emp_long.Employment(45:61) + total_number_remaining_unemployed_FPUC(1:61 - 44)');
        unemployed_share_no_FPUC = total_number_remaining_unemployed_no_FPUC(1:61 - 44)' ./ (bls_emp_long.Employment(45:61) + total_number_remaining_unemployed_no_FPUC(1:61 - 44)');
        spend_increase = mean_c_sim_pandemic_surprise_overall_FPUC(1:17)' .* unemployed_share_FPUC - mean_c_sim_pandemic_surprise_overall_noFPUC(1:17)' .* unemployed_share_no_FPUC;

        pce = readtable(pce_input_directory);
        monthly_spend_pce = pce.PCE / 12; %adjusting the raw data from an annual rate
        monthly_spend_no_FPUC = monthly_spend_pce;
        monthly_spend_no_FPUC(45:61) = monthly_spend_no_FPUC(45:61) .* (1 - spend_increase);

    end

    % Return settings- return NaN for items we don't want
    if include_self_employed == 0
        monthly_spend_pce = NaN;
        monthly_spend_no_FPUC = NaN;
    end

    if include_self_employed == 1
        elasticity = NaN; 
        employment_distortion = NaN;
        total_diff_employment = NaN;
        share_unemployment_reduced = NaN;
        employment_FPUC = NaN;
        employment_noFPUC = NaN;
    end

end
