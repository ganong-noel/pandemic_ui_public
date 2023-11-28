display('Simulating Full Model Effects of $600')
clearvars -except -regexp fig_paper_*
tic

load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load bls_employment_input_directory.mat
load inter_time_series_input_directory.mat
load hh_wage_groups.mat
load release_paths.mat
load matlab_qual_colors.mat

load bestfit_prepandemic.mat
load bestfit_target_waiting_MPC.mat
load discountfactors.mat

benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
table_effects_summary = readtable(inter_time_series_input_directory);
inter_time_series_expiration=1-(1+table_effects_summary.ts_exit(1))^4;
cross_section_expiration=1-(1+table_effects_summary.cs_exit(1))^4;
inter_time_series_onset=1-(1+table_effects_summary.ts_exit(2))^4;
cross_section_onset=1-(1+table_effects_summary.cs_exit(2))^4;
perc_change_benefits_data = benefit_change_data.non_sym_per_change(1);


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
recall_probs_pandemic_actual = exit_rates_data_week_to_month.exit_to_recall';
recall_probs_pandemic_actual = recall_probs_pandemic_actual(4:end);



data_series = readtable(jobfind_input_directory, 'Sheet', fig1_df);
% Make an exit_ui_rate variable
data_series.exit_ui_rate = data_series.ExitRateToRecall + data_series.ExitRateNotToRecall;
% Make the targets of Jan/Feb 2020 ui exit rates
% Restrict to only data past january 2020
data_series_jan_feb = data_series(datenum(data_series.week_start_date) >= datenum('2020-01-01') & datenum(data_series.week_start_date) < datenum('2020-03-01'), :);
data_series_aug_dec = data_series(datenum(data_series.week_start_date) >= datenum('2020-08-01') & datenum(data_series.week_start_date) < datenum('2020-12-31'), :);
data_series_apr_july = data_series(datenum(data_series.week_start_date) >= datenum('2020-04-01') & datenum(data_series.week_start_date) < datenum('2020-07-31'), :);

%data_series_jan_feb = data_series_jan_feb(datenum(data_series_jan_feb.week_start_date) < datenum('2020-03-01'), :);
preperiod_target_weekly = mean(data_series_jan_feb.exit_ui_rate);
preperiod_target = week_to_month_exit(preperiod_target_weekly);
preperiod_target_weekly2 = mean(data_series_jan_feb.ExitRateToRecall);
preperiod_target2 = week_to_month_exit(preperiod_target_weekly2);
preperiod_target_weekly3 = mean(data_series_jan_feb.ExitRateNotToRecall);
preperiod_target3 = week_to_month_exit(preperiod_target_weekly3);
preperiod_target=preperiod_target+preperiod_target2;

pan_target_weekly = mean(data_series_apr_july.exit_ui_rate);
pan_target = week_to_month_exit(pan_target_weekly);
pan_target_weekly2 = mean(data_series_apr_july.ExitRateToRecall);
pan_target2 = week_to_month_exit(pan_target_weekly2);
pan_target_weekly3 = mean(data_series_apr_july.ExitRateNotToRecall);
pan_target3 = week_to_month_exit(pan_target_weekly3);

%pan_target2=mean(recall_probs_pandemic_actual);
pan_target=pan_target2+pan_target3;

normal_jf=preperiod_target;
normal_recall_share=preperiod_target2/normal_jf;

perc_change_benefits_small=.001;

newjf_ratio=pan_target3/preperiod_target3; %this will be the decline in job finding we use for "depressed labor market scenario"
recall_ratio=pan_target2/preperiod_target2;
newjob_hazard_elasticity_pan=(inter_time_series_expiration/(pan_target3))/perc_change_benefits_data
total_hazard_elasticity_pan=(inter_time_series_expiration/(pan_target))/perc_change_benefits_data

totalhazard_elasticity_normal=0.5;
inter_time_series_normal=totalhazard_elasticity_normal*perc_change_benefits_data*preperiod_target
newjobhazard_elasticity_normal=(inter_time_series_normal/preperiod_target3)/perc_change_benefits_data

pan_recall_share=(1-pan_target3/pan_target);
total_hazard_elasticity_prepanhazard_panrecallshare=newjobhazard_elasticity_normal*(1-pan_recall_share);

total_hazard_elasticity_usepanhazard=newjob_hazard_elasticity_pan*(1-normal_recall_share);


exit_no_supp_normaljf=preperiod_target*ones(100,1);
exit_no_supp_lowjf=pan_target*ones(100,1);

exit_with_supp_normaljf_prepanhazard=exit_no_supp_normaljf;
exit_with_supp_lowjf_prepanhazard=exit_no_supp_lowjf;

exit_with_supp_normaljf_panhazard=exit_no_supp_normaljf;
exit_with_supp_lowjf_panhazard=exit_no_supp_lowjf;

exit_with_supp_normaljf_panhazard_panrecallshare=exit_no_supp_normaljf;
exit_with_supp_lowjf_panhazard_panrecallshare=exit_no_supp_lowjf;

exit_with_supp_normaljf_prepanhazard_panrecallshare=exit_no_supp_normaljf;
exit_with_supp_lowjf_prepanhazard_panrecallshare=exit_no_supp_lowjf;

max_supp_length=18;
for i=1:max_supp_length
    exit_with_supp_normaljf_prepanhazard(i)=exit_with_supp_normaljf_prepanhazard(i)*(1-totalhazard_elasticity_normal*perc_change_benefits_small);
    exit_with_supp_lowjf_prepanhazard(i)=exit_with_supp_lowjf_prepanhazard(i)*(1-totalhazard_elasticity_normal*perc_change_benefits_small);

    exit_with_supp_normaljf_panhazard(i)=exit_with_supp_normaljf_panhazard(i)*(1-total_hazard_elasticity_usepanhazard*perc_change_benefits_small);
    exit_with_supp_lowjf_panhazard(i)=exit_with_supp_lowjf_panhazard(i)*(1-total_hazard_elasticity_usepanhazard*perc_change_benefits_small);

    exit_with_supp_normaljf_panhazard_panrecallshare(i)=exit_with_supp_normaljf_panhazard_panrecallshare(i)*(1-total_hazard_elasticity_pan*perc_change_benefits_small);
    exit_with_supp_lowjf_panhazard_panrecallshare(i)=exit_with_supp_lowjf_panhazard_panrecallshare(i)*(1-total_hazard_elasticity_pan*perc_change_benefits_small);

    exit_with_supp_normaljf_prepanhazard_panrecallshare(i)=exit_with_supp_normaljf_prepanhazard_panrecallshare(i)*(1-total_hazard_elasticity_prepanhazard_panrecallshare*perc_change_benefits_small);
    exit_with_supp_lowjf_prepanhazard_panrecallshare(i)=exit_with_supp_lowjf_prepanhazard_panrecallshare(i)*(1-total_hazard_elasticity_prepanhazard_panrecallshare*perc_change_benefits_small);

    %exit_with_supp_normaljf_panhazard_panrecallshare(i)=exp(log(exit_with_supp_normaljf_panhazard_panrecallshare(i))-log(perc_change_benefits_data)-log(total_hazard_elasticity_pan));
    %exit_with_supp_lowjf_panhazard_panrecallshare(i)=exp(log(exit_with_supp_lowjf_panhazard_panrecallshare(i))-log(perc_change_benefits_data)-log(total_hazard_elasticity_pan));

    elasticity_normaljf_prepanhazard(i) = (average_duration(exit_with_supp_normaljf_prepanhazard) / average_duration(exit_no_supp_normaljf) - 1) / perc_change_benefits_small;
    elasticity_lowjf_prepanhazard(i) = (average_duration(exit_with_supp_lowjf_prepanhazard) / average_duration(exit_no_supp_lowjf) - 1) / perc_change_benefits_small;

    elasticity_normaljf_panhazard(i) = (average_duration(exit_with_supp_normaljf_panhazard) / average_duration(exit_no_supp_normaljf) - 1) / perc_change_benefits_small;
    elasticity_lowjf_panhazard(i) = (average_duration(exit_with_supp_lowjf_panhazard) / average_duration(exit_no_supp_lowjf) - 1) / perc_change_benefits_small;

    elasticity_normaljf_panhazard_panrecallshare(i) = (average_duration(exit_with_supp_normaljf_panhazard_panrecallshare) / average_duration(exit_no_supp_normaljf) - 1) / perc_change_benefits_small;
    elasticity_lowjf_panhazard_panrecallshare(i) = (average_duration(exit_with_supp_lowjf_panhazard_panrecallshare) / average_duration(exit_no_supp_lowjf) - 1) / perc_change_benefits_small;

    elasticity_normaljf_prepanhazard_panrecallshare(i) = (average_duration(exit_with_supp_normaljf_prepanhazard_panrecallshare) / average_duration(exit_no_supp_normaljf) - 1) / perc_change_benefits_small;
    elasticity_lowjf_prepanhazard_panrecallshare(i) = (average_duration(exit_with_supp_lowjf_prepanhazard_panrecallshare) / average_duration(exit_no_supp_lowjf) - 1) / perc_change_benefits_small;
end

figure
plot(1:18,totalhazard_elasticity_normal*ones(18,1),1:18,elasticity_normaljf_prepanhazard,1:18,elasticity_lowjf_prepanhazard,'LineWidth',2)
xlabel('Supplement Length (Months)')
legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title('Target Pre Pandemic Hazard')
ylim([0,0.55])
figure
plot(1:18,total_hazard_elasticity_usepanhazard*ones(18,1),1:18,elasticity_normaljf_panhazard,1:18,elasticity_lowjf_panhazard,'LineWidth',2)
xlabel('Supplement Length (Months)')
legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title('Target Pandemic Search Hazard')
ylim([0,0.55])
figure
plot(1:18,total_hazard_elasticity_pan*ones(18,1),1:18,elasticity_normaljf_panhazard_panrecallshare,1:18,elasticity_lowjf_panhazard_panrecallshare,'LineWidth',2)
xlabel('Supplement Length (Months)')
legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title('Target Pandemic Search Hazard and Recall Composition')
ylim([0,0.55])

%load matlab_qual_colors.mat

figure
tiledlayout(1,3)
nexttile
plot(1:18,totalhazard_elasticity_normal*ones(18,1),1:18,elasticity_normaljf_prepanhazard,1:18,elasticity_lowjf_prepanhazard,'LineWidth',2)
xlabel('Supplement length (months)')
%legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title({'Recall share: normal','New job hazard elasticity: normal'})
ylim([0,0.55])
xlim([0 12])
set(gca,'fontsize', 12);
hold on
scatter(12,totalhazard_elasticity_normal,150,'filled','MarkerEdgeColor',qual_orange,'MarkerFaceColor',qual_orange)
scatter(4,elasticity_normaljf_prepanhazard(4),150,'filled','s')
scatter(4,elasticity_lowjf_prepanhazard(4),150,'filled','^')
nexttile
h=plot(1:18,total_hazard_elasticity_prepanhazard_panrecallshare*ones(18,1),1:18,elasticity_normaljf_prepanhazard_panrecallshare,1:18,elasticity_lowjf_prepanhazard_panrecallshare,'LineWidth',2);
xlabel('Supplement length (months)')
%legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title({'Recall share: pandemic','New job hazard elasticity: normal'})
ylim([0,0.55])
xlim([0 12])
set(gca,'fontsize', 12);
hold on
scatter(4,elasticity_lowjf_prepanhazard_panrecallshare(4),150,'filled','+','MarkerEdgeColor', qual_purple, 'MarkerFaceColor',qual_purple,'LineWidth',2)
nexttile
plot(1:18,total_hazard_elasticity_pan*ones(18,1),1:18,elasticity_normaljf_panhazard_panrecallshare,1:18,elasticity_lowjf_panhazard_panrecallshare,'LineWidth',2)
xlabel('Supplement length (months)')
%legend('Total Hazard Elasticity','Duration Elasticity (Normal Labor Market)','Duration Elasticity (Depressed Labor Market)','Location','SouthEast')
title({'Recall share: pandemic','New job hazard elasticity: pandemic'})
ylim([0,0.55])
xlim([0 12])
lg  = legend(h,'Total hazard elasticity','Duration elasticity (normal job-finding rate)','Duration elasticity (depressed job-finding rate)');
lg.Layout.Tile = 'South';
lg.FontSize = 14;
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
hold on
scatter(4,elasticity_lowjf_panhazard_panrecallshare(4),150,'d','MarkerEdgeColor', 'k', 'MarkerFaceColor','k')
set(gcf, 'PaperPosition', [0 0 11 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [11 5]); %Keep the same paper size
fig_paper_11 = gcf;
saveas(fig_paper_11, fullfile(release_path_paper, 'hazard_vs_duration_elasticities.png'))
saveas(fig_paper_11, fullfile(release_path_slides, 'hazard_vs_duration_elasticities.png'))

