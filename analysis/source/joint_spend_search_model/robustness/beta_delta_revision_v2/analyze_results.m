
% Load results for all types from RCC job
load('release/array_job/full_rcc_2023_05_12d.mat')

beta_to_use_low=.94;
beta_to_use_high=.985;
unique(full_beta)
unique(full_delta)
unique(full_c_param)
unique(full_k)

full_c_param=full_c_param(full_beta==beta_to_use_low | full_beta==beta_to_use_high);
full_delta=full_delta(full_beta==beta_to_use_low | full_beta==beta_to_use_high);
full_gamma=full_gamma(full_beta==beta_to_use_low | full_beta==beta_to_use_high);
full_k=full_k(full_beta==beta_to_use_low | full_beta==beta_to_use_high);
full_mpc=full_mpc(full_beta==beta_to_use_low | full_beta==beta_to_use_high,:);
full_search=full_search(full_beta==beta_to_use_low | full_beta==beta_to_use_high,:);
full_spend_e=full_spend_e(full_beta==beta_to_use_low | full_beta==beta_to_use_high,:);
full_spend_u=full_spend_u(full_beta==beta_to_use_low | full_beta==beta_to_use_high,:);
full_beta=full_beta(full_beta==beta_to_use_low | full_beta==beta_to_use_high);

full_c_param=full_c_param(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high);
full_gamma=full_gamma(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high);
full_k=full_k(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high);
full_mpc=full_mpc(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high,:);
full_search=full_search(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high,:);
full_spend_e=full_spend_e(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high,:);
full_spend_u=full_spend_u(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high,:);
full_beta=full_beta(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high);
full_delta=full_delta(full_delta==beta_to_use_low | full_delta==.88 | full_delta==beta_to_use_high);

full_c_param=full_c_param(full_k==1 | full_k==10 | full_k==3| full_k==130);
full_gamma=full_gamma(full_k==1 | full_k==10 |full_k==3| full_k==130);
full_mpc=full_mpc(full_k==1 | full_k==10 |full_k==3| full_k==130,:);
full_search=full_search(full_k==1 | full_k==10 |full_k==3| full_k==130,:);
full_spend_e=full_spend_e(full_k==1 | full_k==10 |full_k==3| full_k==130,:);
full_spend_u=full_spend_u(full_k==1 | full_k==10 |full_k==3| full_k==130,:);
full_beta=full_beta(full_k==1 | full_k==10 |full_k==3| full_k==130);
full_delta=full_delta(full_k==1 | full_k==10 |full_k==3| full_k==130);
full_k=full_k(full_k==1 | full_k==10 |full_k==3| full_k==130);

full_gamma=full_gamma(full_c_param==-0.5);
full_mpc=full_mpc(full_c_param==-0.5,:);
full_search=full_search(full_c_param==-0.5,:);
full_spend_e=full_spend_e(full_c_param==-0.5,:);
full_spend_u=full_spend_u(full_c_param==-0.5,:);
full_beta=full_beta(full_c_param==-0.5);
full_delta=full_delta(full_c_param==-0.5);
full_k=full_k(full_c_param==-0.5);
full_c_param=full_c_param(full_c_param==-0.5);

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

search_model_shift=[full_search(:,4) full_search(:,4:10)];

expected_recall=.079;

% Data series for spending
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

% Preallocate objects for optimal-weights loop
indices=1:size(full_search,1);

% Loop to find optimal weights
for i=1:10
    disp(['optimal weights loop iteration: ', num2str(i)])
    
    if i==2
        indices=indices(params>.0025);
    elseif i>2 && i<=4
        indices=indices(params>.006);
    elseif i>4 && i <=6
        indices=indices(params>.0115);
    elseif i>6 && i<=8
        indices=indices(params>.02);
    elseif i>8
        indices=indices(params>.05);
    end
    
    length(indices)
    search_model_shift_subset=search_model_shift(indices,:);
    
    % Convert model simulations to dollar deviations in U vs. E space
    mean_c_sim_pandemic_expect_dollars = full_spend_u ./ full_spend_e(:, 1:18) * total_spend_u(1) - total_spend_u(1);
    spend_model=[mean_c_sim_pandemic_expect_dollars(indices,1:11)]; 
    
    spend_data=spend_dollars_u_vs_e(1:11);
    
    A1=[];
    b1=[];
    Aeq1=ones(1,length(indices));
    beq1=1;
    lb1=zeros(length(indices),1);
    ub1=ones(length(indices),1);
    
    A2=[];
    b2=[];
    Aeq2=ones(1,length(indices));
    beq2=1;
    lb2=zeros(length(indices),1);
    ub2=ones(length(indices),1);
    
    A=[A1;A2];
    b=[b1;b2;];
    Aeq=[Aeq1;Aeq2];
    beq=[beq1;beq2;];
    lb=lb1;
    ub=ub1;
    %x = lsqlin(C,d,A,b,Aeq,beq,lb,ub);
    
    
    pars_init = ones(length(indices),1);
    pars_init=pars_init/sum(pars_init);
    no_max = optimset('MaxIter', 500, 'MaxFunEvals', Inf, 'Display', 'iter', 'TolFun', .00006, 'TolX', .00006);
    fun = @(pars)model_data_diff(pars,search_model_shift_subset,monthly_search_data,expected_recall,spend_model,spend_data);
    
    [params, fit] = fmincon(fun, pars_init, A2,b2,Aeq2,beq2,lb2,ub2,[],no_max);
    
    share_dynamic=params/sum(params);
    for t=2:8
        share_dynamic(:,t)=share_dynamic(:,t-1).*(1-search_model_shift_subset(:,t-1)-expected_recall);
        share_dynamic(:,t)=share_dynamic(:,t)/sum(share_dynamic(:,t));
    end
    
    mean_search=sum(search_model_shift_subset.*share_dynamic,1);
    diff=mean((mean_search./monthly_search_data-1).^2)
    
    s=sort(params);
    
    mean_spend_model=sum(share_dynamic(:,1).*spend_model);
    
end

results_types=[params full_beta(indices) full_delta(indices) full_c_param(indices) full_gamma(indices) full_k(indices)]
save('results_types.mat', 'results_types')



% Plots
load matlab_qual_colors.mat
load graph_axis_labels_timeseries.mat
load release_paths.mat

figure
p = patch([1 1 4 4], [0 1 1 0], [0.92 0.92 0.92], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(1:8,share_dynamic(1,:), '-o', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(1:8,share_dynamic(2,:), '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(1:8,share_dynamic(3,:), '-s', 'Color', qual_yellow, 'MarkerFaceColor', qual_yellow, 'LineWidth', 2)
legend('Hyperbolic type with \phi=0.1', 'Hyperbolic type with \phi=1', 'Exponential type')
%ylim([-10 30])
xticks([1 2 3 4 5 6 7 8])
xticklabels(label_months_apr20_nov20)
fig_paper_A28e = gcf;
saveas(fig_paper_A28e, [char(release_path_paper), '/shares_by_by_type.png'])

% Plot spending (by type)
figure
p = patch([4 4 7 7], [-20 40 40 -20], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(1:11,spend_data/total_spend_u(1)*100, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
plot(1:11,spend_model(1,1:11)'/total_spend_u(1)*100, '-o', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(1:11,spend_model(2,1:11)'/total_spend_u(1)*100, '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(1:11,spend_model(3,1:11)'/total_spend_u(1)*100, '-s', 'Color', qual_yellow, 'MarkerFaceColor', qual_yellow, 'LineWidth', 2)
legend('Data', 'Hyperbolic type with \phi=0.1', 'Hyperbolic type with \phi=1', 'Exponential type')
ylim([-10 30])
xticklabels(label_months_jan20_nov20)
yticks([-20 -10 0 10 20])
yticklabels({'-20%','-10%', '0%','10%', '20%'})
%title('Spending by Type vs. Data')
fig_A28d = gcf;
saveas(fig_A28d, [char(release_path_paper), '/spend_many_types_by_type.png'])

% Plot search (by type)
figure
p = patch([4 4 7 7], [0 1 1 0], [0.92 0.92 0.92], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(4:11,monthly_search_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
plot(4:11,search_model_shift_subset(1, :), '-o', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(4:11,search_model_shift_subset(2, :), '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(4:11,search_model_shift_subset(3, :), '-s', 'Color', qual_yellow, 'MarkerFaceColor', qual_yellow, 'LineWidth', 2)
legend('Data', 'Hyperbolic type with \phi=0.1', 'Hyperbolic type with \phi=1', 'Exponential type', 'location', 'NorthWest')
ylim([0 .5])
xticklabels(label_months_apr20_nov20)
xtickangle(30)
%title('Job Finding by Type vs. Data')
fig_paper_A28c = gcf;
saveas(fig_paper_A28c, [char(release_path_paper), '/jobfind_many_types_by_type.png'])

% Plot spending (weighted average)
figure
hold on
p = patch([4 4 7 7], [-20 40 40 -20], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(1:11, spend_data/total_spend_u(1)*100, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
plot(1:11, mean_spend_model(1:11)/total_spend_u(1)*100, '-v', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
legend('Data', 'Model with multiple types', 'location', 'SouthWest')
ylim([-10 15])
yticks([-20 -10 0 10 20])
yticklabels({'-20%','-10%', '0%','10%', '20%'})
xticklabels(label_months_jan20_nov20)
%title('Spending Multiple Types vs. Data')
fig_paper_A28b = gcf;
saveas(fig_paper_A28b, [char(release_path_paper), '/spend_many_types_averaged.png'])

% Plot job finding (weighted average)
figure
p = patch([1 1 4 4], [0 1 1 0], [0.92 0.92 0.92], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(1:8, monthly_search_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
plot(1:8, mean_search, '-v', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
legend('Data', 'Model with multiple types', 'location', 'NorthWest')
ylim([.05 .13])
xticks([1 2 3 4 5 6 7 8])
xticklabels(label_months_apr20_nov20)
xtickangle(30)
%title('Job Finding Multiple Types vs. Data')
fig_paper_A28a = gcf;
saveas(fig_paper_A28a, [char(release_path_paper), '/jobfind_many_types_averaged.png'])



function diff=model_data_diff(pars,search_model_shift_subset,monthly_search_data,expected_recall,spend_model,spend_data)
    share_dynamic=pars/sum(pars);
    for t=2:8
        share_dynamic(:,t)=share_dynamic(:,t-1).*(1-search_model_shift_subset(:,t-1)-expected_recall);
        share_dynamic(:,t)=share_dynamic(:,t)/sum(share_dynamic(:,t));
    end
    
    mean_search=sum(search_model_shift_subset.*share_dynamic);
    diff=mean((mean_search(1:8)./monthly_search_data(1:8)-1).^2);
    

    mean_spend_model=sum(share_dynamic(:,1).*spend_model);
    diff=diff+.00001*mean((mean_spend_model(4:11)'-spend_data(4:11)).^2);
end

