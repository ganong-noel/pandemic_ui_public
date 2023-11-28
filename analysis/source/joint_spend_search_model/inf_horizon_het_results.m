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

load bestfit_prepandemic.mat
load bestfit_target_waiting_MPC.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%loading various empirical results from jpmci for later comparisons:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
table_effects_summary = readtable(inter_time_series_input_directory);
inter_time_series_expiration=1-(1+table_effects_summary.ts_exit(1))^4;
cross_section_expiration=1-(1+table_effects_summary.cs_exit(1))^4;
inter_time_series_onset=1-(1+table_effects_summary.ts_exit(2))^4;
cross_section_onset=1-(1+table_effects_summary.cs_exit(2))^4;
cross_section_expiration_logit=1-(1+table_effects_summary.cs_exit(1)*.01415247/.0163)^4;
cross_section_onset_logit=1-(1+table_effects_summary.cs_exit(2)*0.01723101/.0204)^4;

% Plot settings
load matlab_qual_colors.mat
global qual_blue qual_purple qual_green qual_orange matlab_red_orange qual_yellow
load graph_axis_labels_timeseries.mat


exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');

% For the exit variables we want the average exit probability at a monthly level
exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
new = exit_rates_data_week_to_month.exit_not_to_recall';
new_pre = new(13:14);
recall = exit_rates_data_week_to_month.exit_to_recall';
recall_pre = recall(13:14);
mean(recall_pre)
mean(new_pre)



data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_w = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
data_update_w = data_update(idx_w, :);
total_spend_e = data_update_e.value;
total_spend_u = data_update_u.value;
total_spend_w = data_update_w.value;
total_spend_e_yoy = total_spend_e(13:end) ./ total_spend_e(1:end - 12);
total_spend_u_yoy = total_spend_u(13:end) ./ total_spend_u(1:end - 12);
total_spend_w_yoy = total_spend_w(13:end) ./ total_spend_w(1:end - 12);
total_spend_e = total_spend_e(13:end);
total_spend_u = total_spend_u(13:end);
total_spend_w = total_spend_w(13:end);
perc_spend_e = data_update_e.percent_change;
perc_spend_u = data_update_u.percent_change;
perc_spend_w = data_update_w.percent_change;

perc_spend_u_vs_e=total_spend_u_yoy-total_spend_e_yoy;
perc_spend_w_vs_e=total_spend_w_yoy-total_spend_e_yoy;

perc_spend_u_vs_e=perc_spend_u_vs_e-mean(perc_spend_u_vs_e(1:2));
perc_spend_w_vs_e=perc_spend_w_vs_e-mean(perc_spend_w_vs_e(1:2));
spend_dollars_u_vs_e = perc_spend_u_vs_e * mean(total_spend_u(1:2));
spend_dollars_w_vs_e = perc_spend_w_vs_e * mean(total_spend_w(1:2));

idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_w = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
data_update_w = data_update(idx_w, :);
income_e = data_update_e.value;
income_u = data_update_u.value;
income_w = data_update_w.value;
income_e_yoy = income_e(13:end) ./ income_e(1:end - 12) ;
income_u_yoy = income_u(13:end) ./ income_u(1:end - 12);
income_w_yoy = income_w(13:end) ./ income_w(1:end - 12);
income_e = income_e(13:end);
income_u = income_u(13:end);
income_w = income_w(13:end);

perc_income_u_vs_e=income_u_yoy-income_e_yoy;
perc_income_w_vs_e=income_w_yoy-income_e_yoy;

perc_income_e = data_update_e.percent_change;
perc_income_u = data_update_u.percent_change;
perc_income_w = data_update_w.percent_change;
perc_income_u_vs_e=perc_income_u_vs_e-mean(perc_income_u_vs_e(1:3));
perc_income_w_vs_e=perc_income_w_vs_e-mean(perc_income_w_vs_e(1:3));
income_dollars_u_vs_e = perc_income_u_vs_e * mean(income_u(1:3));
income_dollars_w_vs_e = perc_income_w_vs_e * mean(income_w(1:3));

idx_emp = (string(data_update.category) == 'Checking account balance') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_w = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
data_update_w = data_update(idx_w, :);
checking_e = data_update_e.value;
checking_u = data_update_u.value;
checking_w = data_update_w.value;
checking_e = checking_e(13:end);
checking_u = checking_u(13:end);
checking_w = checking_w(13:end);

stat_for_text_change_checking_u_vs_e=(checking_u(7)-checking_u(3))-(checking_e(7)-checking_e(3))
save('stats_for_text_model_miscellaneous.mat', 'stat_for_text_change_checking_u_vs_e', '-append')

u_v1=income_u./income_e-1;
u_v1=u_v1-u_v1(1);

w_v1=income_w./income_e-1;
w_v1=w_v1-w_v1(1);

us_v1=total_spend_u./total_spend_e-1;
us_v1=us_v1-us_v1(1);

ws_v1=total_spend_w./total_spend_e-1;
ws_v1=ws_v1-ws_v1(1);

income_dollars_u_vs_e = u_v1 * mean(income_u(1:3));
income_dollars_w_vs_e = w_v1 * mean(income_w(1:3));

spend_dollars_u_vs_e = us_v1 * mean(total_spend_u(1:2));
spend_dollars_w_vs_e = ws_v1 * mean(total_spend_w(1:2));


mpc_waiting_data=((total_spend_u(5)-total_spend_u(3))-(total_spend_w(5)-total_spend_w(3)))/((income_u(5)-income_u(3))-(income_w(5)-income_w(3)))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Solve model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

shock_500 = 500 / income_e(1);

k_prepandemic = pre_pandemic_fit_match500MPC(1);
gamma_prepandemic = pre_pandemic_fit_match500MPC(2);
c_param_prepandemic = 0;

% Assign parameter values
load discountfactors.mat
beta_normal = beta_targetwaiting;
beta_high = beta_oneperiodshock;

load model_parameters.mat
initial_a = initial_a - aprimemin;

n_ben_profiles_allowed = 8; %This captures the surprise vs. expected expiration scenarios w/ wait or no delay (and 2 extra benefit profiles for liquidity decomposition)

% Set on/off switches
infinite_dur = 0;
use_initial_a = 0;



% Start solving the model with EGM
for iy = 1:5

    y = w(iy);
    h = 0.7 * y;
    b = repshare * y;
    
    % Transfer profile: expect 12 months of FPUC
    transfer_profile(1:12, 1) = FPUC_expiration; 
    transfer_profile(13, 1) = 0; 

    % Transfer profile: expect 4 months of FPUC
    transfer_profile(1:4, 2) = FPUC_expiration; 
    transfer_profile(5:13, 2) = 0; 


    for surprise = 0:1

        rng('default')

        % Set search cost parameters
        if surprise == 1
            k = sse_surprise_fit_het_full(1);
            gamma = sse_surprise_fit_het_full(2);
            c_param = sse_surprise_fit_het_full(3);
        else
            k = sse_expect_fit_het_full(1);
            gamma = sse_expect_fit_het_full(2);
            c_param = sse_expect_fit_het_full(3);
        end

      

        % Aprime grid
        aprimemax = 2000;
        Aprime = linspace(0, aprimemax, n_aprime);
        Aprime = Aprime';

        Aprime = exp(linspace(0, log(aprimemax), n_aprime)) - 1;
        Aprime = Aprime';

        %regular benefits profile
        benefit_profile(1:6, 1) = h + b;

        if infinite_dur == 1
            benefit_profile(7:13, 1) = h + b;
        else
            benefit_profile(7:13, 1) = h;
        end

        %expect $600 for 4 months
        benefit_profile_pandemic(1:4, 1) = b + h + FPUC_expiration;
        benefit_profile_pandemic(5:12, 1) = b + h;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 1) = b + h;
        else
            benefit_profile_pandemic(13, 1) = h;
        end

        %expect $600 for 12 months
        benefit_profile_pandemic(1:12, 2) = b + h + FPUC_expiration;
        if infinite_dur == 1
            benefit_profile_pandemic(13, 2) = b + h + FPUC_expiration;
        else
            benefit_profile_pandemic(13, 2) = h;
        end

        %Matching actual income profile for waiting group
        %expect $600 for 4 months, but w/ 2 month wait
        benefit_profile_pandemic(1, 3) = 1.19*h; %note this is chosen to match the actual income decline for waiting group in the data which is a little more gradual
        benefit_profile_pandemic(2, 3) = h;
        benefit_profile_pandemic(3, 3) = h + 2.35 * (b + FPUC_expiration);
        benefit_profile_pandemic(4, 3) = h + b + FPUC_expiration;
        benefit_profile_pandemic(5:12, 3) = b + h;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 3) = b + h;
        else
            benefit_profile_pandemic(13, 3) = h;
        end

        %expect $600 for 12 months, but w/ 2 month wait
        benefit_profile_pandemic(1, 4) = 1.19*h;
        benefit_profile_pandemic(2, 4) = h;
        benefit_profile_pandemic(3, 4) = h + 2.35 * (b + FPUC_expiration);
        benefit_profile_pandemic(4:12, 4) = b + h + FPUC_expiration;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 4) = b + h + FPUC_expiration;
        else
            benefit_profile_pandemic(13, 4) = h;
        end

        %No FPUC
        benefit_profile_pandemic(1:12, 5) = h + b;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 5) = h + b;
        else
            benefit_profile_pandemic(13, 5) = h;
        end

        %expect LWA supplement to go from month 6+ (months before 6 irrelevant
        %since won't be used until LWA turned on)
        benefit_profile_pandemic(1:5, 6) = b + h;
        benefit_profile_pandemic(6:12, 6) = b + h + LWAsize;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 6) = b + h + LWAsize;
        else
            benefit_profile_pandemic(13, 6) = h;
        end

        %FPUC unconditional: Benefit profile 7 is identical to 2. I duplicate it to loop over
        %this case where employed households also receive FPUC.
        %expect $600 for 12 months
        benefit_profile_pandemic(1:12, 7) = b + h + FPUC_expiration;
        if infinite_dur == 1
            benefit_profile_pandemic(13, 7) = b + h + FPUC_expiration;
        else
            benefit_profile_pandemic(13, 7) = h;
        end

        %FPUC unconditional: Benefit profile 8 is identical to 1. I duplicate it to loop over
        %this case where employed households also receive FPUC.
        %expect $600 for 4 months
        benefit_profile_pandemic(1:4, 8) = b + h + FPUC_expiration;
       benefit_profile_pandemic(5:12, 8) = b + h;

        if infinite_dur == 1
            benefit_profile_pandemic(13, 8) = b + h;
        else
            benefit_profile_pandemic(13, 8) = h;
        end

        recall_probs_pandemic(1:13, 1) = 0.00;
        recall_probs_regular = recall_probs_pandemic;


        recall_probs_pandemic(1:13) = .08;
        recall_probs_regular = recall_probs_pandemic;

        n_transfer_profiles_allowed = 2;

        %initialization of variables for speed
        c_pol_e = zeros(n_aprime, 1);
        c_pol_e_with_transfer = zeros(n_aprime, n_b, n_transfer_profiles_allowed);
        c_pol_u = zeros(n_aprime, n_b, 1);
        c_pol_u_pandemic = zeros(n_aprime, n_b, n_ben_profiles_allowed);
        v_e = c_pol_e;
        v_e_with_transfer = c_pol_e_with_transfer;
        v_u = c_pol_u;
        v_u_pandemic = c_pol_u_pandemic;

        rhs_e = zeros(n_aprime, 1);
        rhs_u = zeros(n_aprime, n_b);
        rhs_u_pandemic = zeros(n_aprime, n_b, n_ben_profiles_allowed);

        for beta_loop = 1:2

            if beta_loop == 1
                beta = beta_normal;
            elseif beta_loop == 2
                beta = beta_high;
            end

            %Iteration counter
            iter = 0;
            % Set tolerance for convergence
            tol = 1e-4;
            tol_percent = 0.0001;
            % Initialize difference in consumption from guess and new
            diffC = tol + 1;
            diffC_percent = tol_percent + 1;

            tol_s = 1e-3;
            tol_c_percent = .05;

            ave_change_in_C_percent = 100;
            ave_change_in_S = 100;

            % Initial guesses
            c_pol_e_guess(:) = y(1) + Aprime(:) * (1 + r) + r * aprimemin;
            v_e_guess(:) = ((c_pol_e_guess(:)).^(1 - mu) - 1) / (1 - mu);

            %FPUC unconditional: employed guess
            for i_transfer_profile = 1:n_transfer_profiles_allowed
                for ib = 1:n_b
                    c_pol_e_with_transfer_guess(:, ib, i_transfer_profile) = transfer_profile(ib, i_transfer_profile) + y(1) + Aprime(:) * (1 + r) + r * aprimemin;
                    v_e_with_transfer_guess(:, ib, i_transfer_profile) = ((c_pol_e_with_transfer_guess(:, ib, i_transfer_profile)).^(1 - mu) - 1) / (1 - mu);
                end 
            end

            optimal_search_guess = zeros(n_aprime, n_b);
            optimal_search_pandemic_guess = zeros(n_aprime, n_b, n_ben_profiles_allowed);

            for ib = 1:n_b
                c_pol_u_guess(:, ib) = benefit_profile(ib) + Aprime(:) * (1 + r) + r * aprimemin;
                v_u_guess(:, ib) = ((c_pol_u_guess(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k_prepandemic * 0^(1 + gamma_prepandemic)) / (1 + gamma_prepandemic) + c_param_prepandemic;
            end

            for i_ben_profile = 1:n_ben_profiles_allowed

                for ib = 1:n_b
                    c_pol_u_pandemic_guess(:, ib, i_ben_profile) = benefit_profile_pandemic(ib, i_ben_profile) + Aprime(:) * (1 + r) + r * aprimemin;
                    v_u_pandemic_guess(:, ib, i_ben_profile) = ((c_pol_u_pandemic_guess(:, ib, i_ben_profile)).^(1 - mu) - 1) / (1 - mu) - (k * 0^(1 + gamma)) / (1 + gamma) + c_param;
                end

            end

            %c_pol is c(a,y)
            %c_tilde is c(a',y)

            if beta_loop == 2
                maxiter = 1; %this effectively governs how many periods households will think the high discount factor will last, setting maxiter=1 essentially runs one backward induction step from the beta_normal solutions
                %note that the code must be structured so that it solves the
                %beta_normal part first
                c_pol_e_guess = c_pol_e_betanormal;
                c_pol_e_with_transfer_guess = c_pol_e_with_transfer_betanormal;
                c_pol_u_guess = c_pol_u_betanormal;
                c_pol_u_pandemic_guess = c_pol_u_pandemic_betanormal;

                v_e_guess = v_e_betanormal;
                v_e_with_transfer_guess = v_e_with_transfer_betanormal;
                v_u_guess = v_u_betanormal;
                v_u_pandemic_guess = v_u_pandemic_betanormal;

            else
                maxiter = 1000;
            end

            % Iterate to convergence
            while ((ave_change_in_C_percent > tol_c_percent) || (ave_change_in_S > tol_s)) && iter < maxiter


                %employed

                rhs_e(:) = beta * (1 + r) * ((1 - sep_rate) * c_pol_e_guess(:).^(-mu) + sep_rate * c_pol_u_guess(:, 1).^(-mu));
                c_tilde_e(:) = (rhs_e(:)).^(-1 / mu); %unconstrained
                a_star_e(:) = (Aprime(:) + c_tilde_e(:) - y(1)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                a_star1_e = (c_tilde_e(1) - y(1)) / (1 + r);
                %time1(t)=toc;
                %tic;
                for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both

                    if Aprime(ia) > a_star1_e
                        c_pol_e(ia:end) = interp1(a_star_e(:), c_tilde_e(:), Aprime(ia:end), 'linear', 'extrap');
                        break
                    else
                        c_pol_e(ia) = (1 + r) * Aprime(ia) + y + r * aprimemin;
                    end

                end

                a_prime_holder = (1 + r) * Aprime + y(1) - c_pol_e(:);
                v_e(:) = ((c_pol_e(:)).^(1 - mu) - 1) / (1 - mu) + beta * ((1 - sep_rate) .* interp1(Aprime, v_e_guess(:), a_prime_holder, 'linear', 'extrap') + sep_rate * interp1(Aprime, v_u_guess(:, 1), a_prime_holder, 'linear', 'extrap'));
                %time2(t)=toc;

                %FPUC unconditional: employed
                for i_transfer_profile = 1:n_transfer_profiles_allowed
                    for ib = 1:n_b
                        rhs_e_with_transfer(:, ib, i_transfer_profile) = beta * (1 + r) * ((1 - sep_rate) * c_pol_e_with_transfer_guess(:, min(ib + 1, n_b), i_transfer_profile).^(-mu) + sep_rate * c_pol_u_guess(:, 1).^(-mu)); 
                        c_tilde_e_with_transfer(:, ib, i_transfer_profile) = (rhs_e_with_transfer(:, ib, i_transfer_profile)).^(-1 / mu); %unconstrained
                        a_star_e_with_transfer(:, ib, i_transfer_profile) = (Aprime(:) + c_tilde_e_with_transfer(:, ib, i_transfer_profile) - y(1) - transfer_profile(ib, i_transfer_profile)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                        a_star1_e_with_transfer(ib, i_transfer_profile) = (c_tilde_e_with_transfer(1, ib, i_transfer_profile) - y(1) - transfer_profile(ib, i_transfer_profile)) / (1 + r);
                        %a_star1_e_with_transfer(:, ib) = (c_tilde_e(1) - y(1)) / (1 + r);
                    end 
        
                    for ib = 1:n_b
                        for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
                            if Aprime(ia) > a_star1_e_with_transfer(ib, i_transfer_profile)
                                c_pol_e_with_transfer(ia:end, ib, i_transfer_profile) = interp1(a_star_e_with_transfer(:, ib, i_transfer_profile), c_tilde_e_with_transfer(:, ib, i_transfer_profile), Aprime(ia:end), 'linear', 'extrap');
                                break
                            else
                                c_pol_e_with_transfer(ia, ib, i_transfer_profile) = (1 + r) * Aprime(ia) + y + transfer_profile(ib, i_transfer_profile) + r * aprimemin;
                            end
                        end
                    end
        
                    for ib = 1:n_b
                        a_prime_holder_with_transfer(:, ib, i_transfer_profile) = (1 + r) * Aprime + y(1) + transfer_profile(ib, i_transfer_profile) - c_pol_e_with_transfer(:, ib, i_transfer_profile);
                        v_e_with_transfer(:, ib, i_transfer_profile) = ((c_pol_e_with_transfer(:, ib, i_transfer_profile)).^(1 - mu) - 1) / (1 - mu) + beta * ((1 - sep_rate) .* interp1(Aprime, v_e_with_transfer_guess(:, min(ib + 1, n_b), i_transfer_profile), a_prime_holder_with_transfer(:, ib, i_transfer_profile), 'linear', 'extrap') + sep_rate * interp1(Aprime, v_u_guess(:, 1), a_prime_holder_with_transfer(:, ib, i_transfer_profile), 'linear', 'extrap'));
                    end 
                end                

                %unemployed

                %tic;
                for ib = 1:n_b

                    tmp = min(1-recall_probs_regular(ib), max(0, (beta * (v_e_guess(:) - v_u_guess(:, min(ib + 1, n_b))) / k_prepandemic).^(1 / gamma_prepandemic)));
                    tmp(imag(tmp) ~= 0) = 0;
                    optimal_search(:, ib) = tmp;

                    rhs_u(:, ib) = beta * (1 + r) * ((recall_probs_regular(ib) + optimal_search(:, ib)) .* c_pol_e_guess(:).^(-mu) + (1 - optimal_search(:, ib) - recall_probs_regular(ib)) .* c_pol_u_guess(:, min(ib + 1, n_b)).^(-mu));
                    c_tilde_u(:, ib) = (rhs_u(:, ib)).^(-1 / mu); %unconstrained
                    a_star_u(:, ib) = (Aprime(:) + c_tilde_u(:, ib) - benefit_profile(ib)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                    a_star1_u(ib) = (c_tilde_u(1, ib) - benefit_profile(ib)) / (1 + r);
                end

                %time3(t)=toc;
                %tic;
                for ib = 1:n_b

                    for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both

                        if Aprime(ia) > a_star1_u(ib)
                            c_pol_u(ia:end, ib) = interp1(a_star_u(:, ib), c_tilde_u(:, ib), Aprime(ia:end), 'linear', 'extrap');
                            break
                        else
                            c_pol_u(ia, ib) = (1 + r) * Aprime(ia) + benefit_profile(ib) + r * aprimemin;
                        end

                    end

                end

                for ib = 1:n_b
                    a_prime_holder_u(:, ib) = (1 + r) * Aprime + benefit_profile(ib) - c_pol_u(:, ib);
                    v_u(:, ib) = ((c_pol_u(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k_prepandemic * optimal_search(:, ib).^(1 + gamma_prepandemic)) / (1 + gamma_prepandemic) + c_param_prepandemic + beta * ((optimal_search(:, ib) + recall_probs_regular(ib)) .* interp1(Aprime, v_e_guess(:), a_prime_holder_u(:, ib), 'linear', 'extrap') + (1 - optimal_search(:, ib) - recall_probs_regular(ib)) .* interp1(Aprime, v_u_guess(:, min(ib + 1, n_b)), a_prime_holder_u(:, ib), 'linear', 'extrap'));
                end

                %pandemic unemployed
                for i_ben_profile = 1:n_ben_profiles_allowed
                    if i_ben_profile <= 6
                    %tic;
                        for ib = 1:n_b
    
                            tmp = min(1-recall_probs_pandemic(ib), max(0, (beta * (v_e_guess(:) - v_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile)) / k).^(1 / gamma)));
                            tmp(imag(tmp) ~= 0) = 0;
                            optimal_search_pandemic(:, ib, i_ben_profile) = tmp;
    
                            rhs_u_pandemic(:, ib, i_ben_profile) = beta * (1 + r) * ((recall_probs_pandemic(ib) + optimal_search_pandemic(:, ib, i_ben_profile)) .* c_pol_e_guess(:).^(-mu) + (1 - optimal_search_pandemic(:, ib, i_ben_profile) - recall_probs_pandemic(ib)) .* c_pol_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile).^(-mu));
                            c_tilde_u_pandemic(:, ib, i_ben_profile) = (rhs_u_pandemic(:, ib, i_ben_profile)).^(-1 / mu); %unconstrained
                            a_star_u_pandemic(:, ib, i_ben_profile) = (Aprime(:) + c_tilde_u_pandemic(:, ib, i_ben_profile) - benefit_profile_pandemic(ib, i_ben_profile)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                            a_star1_u_pandemic(ib, i_ben_profile) = (c_tilde_u_pandemic(1, ib, i_ben_profile) - benefit_profile_pandemic(ib, i_ben_profile)) / (1 + r);
                        end

                    %time3(t)=toc;
                        %tic;
                        for ib = 1:n_b
    
                            for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
    
                                if Aprime(ia) > a_star1_u_pandemic(ib, i_ben_profile)
                                    c_pol_u_pandemic(ia:end, ib, i_ben_profile) = interp1(a_star_u_pandemic(:, ib, i_ben_profile), c_tilde_u_pandemic(:, ib, i_ben_profile), Aprime(ia:end), 'linear', 'extrap');
                                    break
                                    %constrained_u(ia,ib,t)=0;
                                else
                                    c_pol_u_pandemic(ia, ib, i_ben_profile) = (1 + r) * Aprime(ia) + benefit_profile_pandemic(ib, i_ben_profile) + r * aprimemin;
                                    %constrained_u(ia,ib,t)=1;
                                end
    
                            end
    
                        end
    
                        for ib = 1:n_b
                            a_prime_holder_u_pandemic(:, ib, i_ben_profile) = (1 + r) * Aprime + benefit_profile_pandemic(ib, i_ben_profile) - c_pol_u_pandemic(:, ib, i_ben_profile);
                            v_u_pandemic(:, ib, i_ben_profile) = ((c_pol_u_pandemic(:, ib, i_ben_profile)).^(1 - mu) - 1) / (1 - mu) - (k * optimal_search_pandemic(:, ib, i_ben_profile).^(1 + gamma)) / (1 + gamma) + c_param +beta * ((recall_probs_pandemic(ib) + optimal_search_pandemic(:, ib, i_ben_profile)) .* interp1(Aprime, v_e_guess(:), a_prime_holder_u_pandemic(:, ib, i_ben_profile), 'linear', 'extrap') + (1 - optimal_search_pandemic(:, ib, i_ben_profile) - recall_probs_pandemic(ib)) .* interp1(Aprime, v_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile), a_prime_holder_u_pandemic(:, ib, i_ben_profile), 'linear', 'extrap'));
                        end

                    % FPUC unconditional: pandemic unemployed
                    elseif i_ben_profile >= 7
                    %use transfer_profile(:,1) with benefit_profile_pandemic(:,7);  transfer_profile(:,2) with benefit_profile_pandemic(:,8); 
                        i_transfer_profile = i_ben_profile - 6;
                        for ib = 1:n_b
    
                            tmp = min(1-recall_probs_pandemic(ib), max(0, (beta * (v_e_with_transfer_guess(:, ib, i_transfer_profile) - v_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile)) / k).^(1 / gamma)));
                            tmp(imag(tmp) ~= 0) = 0;
                            optimal_search_pandemic(:, ib, i_ben_profile) = tmp;
    
                            rhs_u_pandemic(:, ib, i_ben_profile) = beta * (1 + r) * ((recall_probs_pandemic(ib) + optimal_search_pandemic(:, ib, i_ben_profile)) .* c_pol_e_with_transfer_guess(:, min(ib + 1, n_b), i_transfer_profile).^(-mu) + (1 - optimal_search_pandemic(:, ib, i_ben_profile) - recall_probs_pandemic(ib)) .* c_pol_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile).^(-mu));
                            c_tilde_u_pandemic(:, ib, i_ben_profile) = (rhs_u_pandemic(:, ib, i_ben_profile)).^(-1 / mu); %unconstrained
                            a_star_u_pandemic(:, ib, i_ben_profile) = (Aprime(:) + c_tilde_u_pandemic(:, ib, i_ben_profile) - benefit_profile_pandemic(ib, i_ben_profile)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                            a_star1_u_pandemic(ib, i_ben_profile) = (c_tilde_u_pandemic(1, ib, i_ben_profile) - benefit_profile_pandemic(ib, i_ben_profile)) / (1 + r);
                        end

                        for ib = 1:n_b
    
                            for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
    
                                if Aprime(ia) > a_star1_u_pandemic(ib, i_ben_profile)
                                    c_pol_u_pandemic(ia:end, ib, i_ben_profile) = interp1(a_star_u_pandemic(:, ib, i_ben_profile), c_tilde_u_pandemic(:, ib, i_ben_profile), Aprime(ia:end), 'linear', 'extrap');
                                    break
                                    %constrained_u(ia,ib,t)=0;
                                else
                                    c_pol_u_pandemic(ia, ib, i_ben_profile) = (1 + r) * Aprime(ia) + benefit_profile_pandemic(ib, i_ben_profile) + r * aprimemin;
                                    %constrained_u(ia,ib,t)=1;
                                end
    
                            end
    
                        end
                   
                        for ib = 1:n_b
                            a_prime_holder_u_pandemic(:, ib, i_ben_profile) = (1 + r) * Aprime + benefit_profile_pandemic(ib, i_ben_profile) - c_pol_u_pandemic(:, ib, i_ben_profile);
                            v_u_pandemic(:, ib, i_ben_profile) = ((c_pol_u_pandemic(:, ib, i_ben_profile)).^(1 - mu) - 1) / (1 - mu) - (k * optimal_search_pandemic(:, ib, i_ben_profile).^(1 + gamma)) / (1 + gamma) + c_param +beta * ((recall_probs_pandemic(ib) + optimal_search_pandemic(:, ib, i_ben_profile)) .* interp1(Aprime, v_e_with_transfer_guess(:, min(ib + 1, n_b), i_transfer_profile), a_prime_holder_u_pandemic(:, ib, i_ben_profile), 'linear', 'extrap') + (1 - optimal_search_pandemic(:, ib, i_ben_profile) - recall_probs_pandemic(ib)) .* interp1(Aprime, v_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile), a_prime_holder_u_pandemic(:, ib, i_ben_profile), 'linear', 'extrap'));
                        end         
                    end
                end

                % Computing changes in consumption, etc to measure convergence
                diffC = max([max(max(abs(c_pol_e(:) - c_pol_e_guess(:)))), max(max(max(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), max(max(max(max(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :))))))]);

                diffC_percent = 100 * max([max(max(abs((c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess(:)))), max(max(max(abs((c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :))))), max(max(max(max(abs((c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess(:, :, :))))))]);

                % Absolute difference in value to measure convergence
                diffV = max([max(abs(v_e(:) - v_e_guess(:))), max(max(abs(v_u(:, :) - v_u_guess(:, :)))), max(max(max(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

                ave_change_in_C = mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :))))))]);

                ave_change_in_C_percent = 100 * mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
              
                ave_change_in_C_percent_transfer = 100 * mean([mean(mean(mean(abs(c_pol_e_with_transfer(:,:) - c_pol_e_with_transfer_guess(:,:)) ./ c_pol_e_with_transfer_guess(:, :)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
              
                ave_change_in_C_percent = (ave_change_in_C_percent + ave_change_in_C_percent_transfer) ./ 2;

                ave_change_in_V = mean([mean(abs(v_e(:) - v_e_guess(:))), mean(mean(abs(v_u(:, :) - v_u_guess(:, :)))), mean(mean(mean(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

                ave_change_in_S = mean([mean(mean(mean(abs(optimal_search(:, :) - optimal_search_guess(:, :))))), mean(mean(mean(mean(abs(optimal_search_pandemic(:, :, :) - optimal_search_pandemic_guess(:, :, :))))))]);

                % Update guesses, fully for now.
                c_pol_e_guess = c_pol_e;
                c_pol_e_with_transfer_guess = c_pol_e_with_transfer;
                c_pol_u_guess = c_pol_u;
                c_pol_u_pandemic_guess = c_pol_u_pandemic;
                v_e_guess = v_e;
                v_e_with_transfer_guess = v_e_with_transfer;
                v_u_guess = v_u;
                v_u_pandemic_guess = v_u_pandemic;
                optimal_search_guess = optimal_search;
                optimal_search_pandemic_guess = optimal_search_pandemic;

                % Count the iteration
                iter = iter + 1;

            end

            %Solution with and without the temporary high discount rate
            if beta_loop == 1
                c_pol_e_betanormal = c_pol_e;
                c_pol_e_with_transfer_betanormal = c_pol_e_with_transfer;
                c_pol_u_betanormal = c_pol_u;
                c_pol_u_pandemic_betanormal = c_pol_u_pandemic;

                v_e_betanormal = v_e;
                v_e_with_transfer_betanormal = v_e_with_transfer;
                v_u_betanormal = v_u;
                v_u_pandemic_betanormal = v_u_pandemic;
            elseif beta_loop == 2
                c_pol_e_betahigh = c_pol_e;
                c_pol_e_with_transfer_betahigh = c_pol_e_with_transfer;
                c_pol_u_betahigh = c_pol_u;
                c_pol_u_pandemic_betahigh = c_pol_u_pandemic;

                v_e_betahigh = v_e;
                v_e_with_transfer_betahigh = v_e_with_transfer;
                v_u_betahigh = v_u;
                v_u_pandemic_betahigh = v_u_pandemic;
            end

        end

        % Begin simulations using policy functions

        A = Aprime;
    %{
    figure
    plot(A,c_pol_e(:,1),A,c_pol_u(:,:,1))
    title('Consumption functions E vs regular U t=1')

    figure
    plot(A,c_pol_e(:,1),A,c_pol_u_pandemic(:,:,1,1))
    title('Consumption functions E vs pandemic U  t=1')

    figure
    plot(A,v_e(:,1),A,v_u(:,:,1))
    title('Value functions E vs. regular U  t=1')

    figure
    plot(A,v_e(:,1),A,v_u_pandemic(:,:,1))
    title('Value functions E vs. pandemic U  t=1')

    figure
    plot(A,optimal_search(:,:,1))
    title('Optimal search regular u, t=1')

    figure
    plot(A,optimal_search_pandemic(:,:,1))
    title('Optimal search pandemic, t=1')
    %}

        numt_sim = 36;
        a_u_sim = zeros(numt_sim, 1);
        c_u_sim = a_u_sim;
        c_u_pandemic_expect_sim = a_u_sim;
        c_u_pandemic_surprise_sim = a_u_sim;
        c_e_sim = c_u_sim;
        a_u_sim(1) = initial_a;
        a_u_pandemic_expect_sim = a_u_sim;
        a_u_pandemic_surprise_sim = a_u_sim;
        a_e_sim = a_u_sim;

        c_e_with500_sim = c_e_sim;
        a_e_with500_sim = a_e_sim;

        c_u_with500_sim1 = c_u_sim;
        a_u_with500_sim1 = a_u_sim;
        c_u_with500_sim2 = c_u_sim;
        a_u_with500_sim2 = a_u_sim;
        c_u_with500_sim3 = c_u_sim;
        a_u_with500_sim3 = a_u_sim;


        numhh = 1000;
        numsim = 18;
        burnin = 15;
        a_sim = zeros(numhh, burnin + 1);
        c_sim = zeros(numhh, burnin + 1);
        e_sim = zeros(numhh, burnin + 1);
        
        u_dur_sim = zeros(numhh, burnin + 1);
        a_sim(:, 1) = initial_a;
        e_sim(:, 1) = 1;

        c_pol_e = c_pol_e_betanormal;
        c_pol_e_with_transfer = c_pol_e_with_transfer_betanormal;
        c_pol_u = c_pol_u_betanormal;
        c_pol_u_pandemic = c_pol_u_pandemic_betanormal;

        v_e = v_e_betanormal;
        v_e_with_transfer = v_e_with_transfer_betanormal;
        v_u = v_u_betanormal;
        v_u_pandemic = v_u_pandemic_betanormal;

        for t = 1:burnin

            for i = 1:numhh

                if e_sim(i, t) == 1
                    c_sim(i, t) = interp1(A, c_pol_e(:), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = max(y + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);
                else
                    c_sim(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);
                end

                randy = rand(1, 1);

                if e_sim(i, t) == 1

                    if randy < sep_rate
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = 1;
                    else
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    end

                else

                    if randy < exog_find_rate
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    else
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = min(u_dur_sim(i, t) + 1, n_b);
                   end

                end

            end

        end

        tmp_a = a_sim(:, burnin + 1);
        tmp_u = u_dur_sim(:, burnin + 1);
        tmp_e = e_sim(:, burnin + 1);

        a_sim = zeros(numhh, numsim);
        c_sim = zeros(numhh, numsim);
        e_sim = zeros(numhh, numsim);
        a_sim_betahigh = zeros(numhh, burnin + 1);
        c_sim_betahigh = zeros(numhh, burnin + 1);
        e_sim_betahigh = zeros(numhh, burnin + 1);
        
        u_dur_sim = zeros(numhh, numsim);
        a_sim(:, 1) = tmp_a;
        u_dur_sim(:, 1) = tmp_u;
        e_sim(:, 1) = tmp_e;
        u_dur_sim_betahigh = zeros(numhh, numsim);
        a_sim_betahigh(:, 1) = tmp_a;
        u_dur_sim_betahigh(:, 1) = tmp_u;
        e_sim_betahigh(:, 1) = tmp_e;

        c_sim_with_500 = c_sim;
        a_sim_with_500 = a_sim;
        a_sim_with_500(:, 1) = a_sim_with_500(:, 1) + shock_500;
        c_sim_with_500_betahigh = c_sim;
        a_sim_with_500_betahigh = a_sim;
        a_sim_with_500_betahigh(:, 1) = a_sim_with_500_betahigh(:, 1) + shock_500;

        c_sim_with_2400 = c_sim;
        a_sim_with_2400 = a_sim;
        a_sim_with_2400(:, 1) = a_sim_with_500(:, 1) + FPUC_expiration;
        c_sim_with_2400_betahigh = c_sim;
        a_sim_with_2400_betahigh = a_sim;
        a_sim_with_2400_betahigh(:, 1) = a_sim_with_500_betahigh(:, 1) + FPUC_expiration;

        tmp_a_unemployed=tmp_a(tmp_u > 0);
        
        if use_initial_a == 1
            a_sim_pandemic_surprise = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_surprise_extramonth = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_expect = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_surprise_wait = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0) + initial_a;
            a_sim_regular = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0) + initial_a;
            a_sim_e = tmp_a(tmp_u == 0) + initial_a;
            a_sim_pandemic_LWAperm = a_sim_pandemic_surprise;
            a_sim_pandemic_surprise_onlyasseteffect=a_sim_pandemic_noFPUC+FPUC_expiration*12;
            a_sim_pandemic_surprise_FPUC_uncond = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_surprise_noasseteffect=tmp_a(tmp_u > 0) + initial_a;
        else
            a_sim_pandemic_surprise = tmp_a(tmp_u > 0);
            a_sim_pandemic_surprise_extramonth = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect = tmp_a(tmp_u > 0);
            a_sim_pandemic_surprise_wait = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0);
            a_sim_regular = tmp_a(tmp_u > 0);
            a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0);
            a_sim_e = tmp_a(tmp_u == 0);
            a_sim_pandemic_LWAperm = a_sim_pandemic_surprise;
            a_sim_pandemic_surprise_onlyasseteffect=a_sim_pandemic_noFPUC+FPUC_expiration*12;
            a_sim_pandemic_surprise_FPUC_uncond = tmp_a(tmp_u > 0);
            a_sim_pandemic_surprise_noasseteffect=tmp_a(tmp_u > 0);
        end

        num_unemployed_hh = length(a_sim_pandemic_surprise);
        num_employed_hh = length(a_sim_e);
        c_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_LWAperm = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_surprise_extramonth = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_surprise_wait = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect_wait = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_regular = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_surprise_onlyasseteffect = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_surprise_FPUC_uncond = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_surprise_noasseteffect = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_e = zeros(length(a_sim_e), 30);

        search_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_LWAperm = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
        search_sim_pandemic_surprise_wait = zeros(length(a_sim_pandemic_surprise_wait), 30);
        search_sim_pandemic_expect_wait = zeros(length(a_sim_pandemic_expect_wait), 30);
        search_sim_regular = zeros(length(a_sim_regular), 30);
        search_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_surprise_onlyasseteffect = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_surprise_FPUC_uncond = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_surprise_noasseteffect = zeros(length(a_sim_pandemic_surprise), 30);

        %this is looping over all hh after the burnin period, to get the average
        %MPC
        for t = 1:numsim

            for i = 1:numhh

                if e_sim(i, t) == 1
                    c_sim(i, t) = interp1(A, c_pol_e(:), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = max(y + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);

                    c_sim_with_500(i, t) = interp1(A, c_pol_e(:), a_sim_with_500(i, t), 'linear');
                    a_sim_with_500(i, t + 1) = max(y + (1 + r) * a_sim_with_500(i, t) - c_sim_with_500(i, t), 0);

                    c_sim_with_2400(i, t) = interp1(A, c_pol_e(:), a_sim_with_2400(i, t), 'linear');
                    a_sim_with_2400(i, t + 1) = max(y + (1 + r) * a_sim_with_2400(i, t) - c_sim_with_2400(i, t), 0);
                    
                    
                    if t==1
                        c_sim_betahigh(i, t) = interp1(A, c_pol_e_betahigh(:), a_sim_betahigh(i, t), 'linear');
                        a_sim_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_betahigh(i, t) - c_sim_betahigh(i, t), 0);

                        c_sim_with_500_betahigh(i, t) = interp1(A, c_pol_e_betahigh(:), a_sim_with_500_betahigh(i, t), 'linear');
                        a_sim_with_500_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_with_500_betahigh(i, t) - c_sim_with_500_betahigh(i, t), 0);

                        c_sim_with_2400_betahigh(i, t) = interp1(A, c_pol_e_betahigh(:), a_sim_with_2400_betahigh(i, t), 'linear');
                        a_sim_with_2400_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_with_2400_betahigh(i, t) - c_sim_with_2400_betahigh(i, t), 0);
                    else
                        c_sim_betahigh(i, t) = interp1(A, c_pol_e(:), a_sim_betahigh(i, t), 'linear');
                        a_sim_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_betahigh(i, t) - c_sim_betahigh(i, t), 0);

                        c_sim_with_500_betahigh(i, t) = interp1(A, c_pol_e(:), a_sim_with_500_betahigh(i, t), 'linear');
                        a_sim_with_500_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_with_500_betahigh(i, t) - c_sim_with_500_betahigh(i, t), 0);

                        c_sim_with_2400_betahigh(i, t) = interp1(A, c_pol_e(:), a_sim_with_2400_betahigh(i, t), 'linear');
                        a_sim_with_2400_betahigh(i, t + 1) = max(y + (1 + r) * a_sim_with_2400_betahigh(i, t) - c_sim_with_2400_betahigh(i, t), 0);
                    end
                else
                    c_sim(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);

                    c_sim_with_500(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_with_500(i, t), 'linear');
                    a_sim_with_500(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_500(i, t) - c_sim_with_500(i, t), 0);

                    c_sim_with_2400(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_with_2400(i, t), 'linear');
                    a_sim_with_2400(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_2400(i, t) - c_sim_with_2400(i, t), 0);
                    if t==1
                        c_sim_betahigh(i, t) = interp1(A, c_pol_u_betahigh(:, u_dur_sim(i, t)), a_sim_betahigh(i, t), 'linear');
                        a_sim_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_betahigh(i, t) - c_sim_betahigh(i, t), 0);

                        c_sim_with_500_betahigh(i, t) = interp1(A, c_pol_u_betahigh(:, u_dur_sim(i, t)), a_sim_with_500_betahigh(i, t), 'linear');
                        a_sim_with_500_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_500_betahigh(i, t) - c_sim_with_500_betahigh(i, t), 0);

                        c_sim_with_2400_betahigh(i, t) = interp1(A, c_pol_u_betahigh(:, u_dur_sim(i, t)), a_sim_with_2400_betahigh(i, t), 'linear');
                        a_sim_with_2400_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_2400_betahigh(i, t) - c_sim_with_2400_betahigh(i, t), 0);
                    else
                        c_sim_betahigh(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_betahigh(i, t), 'linear');
                        a_sim_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_betahigh(i, t) - c_sim_betahigh(i, t), 0);

                        c_sim_with_500_betahigh(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_with_500_betahigh(i, t), 'linear');
                        a_sim_with_500_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_500_betahigh(i, t) - c_sim_with_500_betahigh(i, t), 0);

                        c_sim_with_2400_betahigh(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_with_2400_betahigh(i, t), 'linear');
                        a_sim_with_2400_betahigh(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_2400_betahigh(i, t) - c_sim_with_2400_betahigh(i, t), 0);
                    end
                end

                randy = rand(1, 1);

                if e_sim(i, t) == 1

                    if randy < sep_rate
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = 1;
                    else
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    end

                else

                    if randy < exog_find_rate
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    else
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = min(u_dur_sim(i, t) + 1, n_b);
                    end

                end

            end

        end

        mean_a_sim = mean(a_sim, 1);
        mean_c_sim = mean(c_sim, 1);

        mean_a_sim_with_500 = mean(a_sim_with_500);
        mean_c_sim_with_500 = mean(c_sim_with_500);
        
        mean_a_sim_betahigh = mean(a_sim_betahigh, 1);
        mean_c_sim_betahigh = mean(c_sim_betahigh, 1);

        mean_a_sim_with_500_betahigh = mean(a_sim_with_500_betahigh);
        mean_c_sim_with_500_betahigh = mean(c_sim_with_500_betahigh);

        for tt = 1:numsim
            mean_c_sim_e_with_500(tt) = mean(c_sim_with_500(e_sim(:, 1) == 1, tt));
            mean_c_sim_e_without_500(tt) = mean(c_sim(e_sim(:, 1) == 1, tt));
            
            mean_c_sim_e_with_500_betahigh(tt) = mean(c_sim_with_500_betahigh(e_sim(:, 1) == 1, tt));
            mean_c_sim_e_without_500_betahigh(tt) = mean(c_sim_betahigh(e_sim(:, 1) == 1, tt));
        end

        mpc_500_e_by_t = (mean_c_sim_e_with_500 - mean_c_sim_e_without_500) / (shock_500);
        mpc_500_e_by_t_betahigh = (mean_c_sim_e_with_500_betahigh - mean_c_sim_e_without_500_betahigh) / (shock_500);

        for t = 1:numsim
            mpc_500_e_cum_dynamic(t) = sum(mpc_500_e_by_t(1:t));
            mpc_500_e_cum_dynamic_betahigh(t) = sum(mpc_500_e_by_t_betahigh(1:t));
        end

        if surprise == 1
            mpc_surprise_500_e_bywage(iy, :) = mpc_500_e_cum_dynamic(:);
            mpc_surprise_500_e_bywage_betahigh(iy, :) = mpc_500_e_cum_dynamic_betahigh(:);
        else
            mpc_expect_500_e_bywage(iy, :) = mpc_500_e_cum_dynamic(:);
            mpc_expect_500_e_bywage_betahigh(iy, :) = mpc_500_e_cum_dynamic_betahigh(:);
        end

        for tt = 1:numsim
            mean_c_sim_u_with_500(tt) = mean(c_sim_with_500(e_sim(:, 1) == 0, tt));
            mean_c_sim_u_without_500(tt) = mean(c_sim(e_sim(:, 1) == 0, tt));
            
            mean_c_sim_u_with_500_betahigh(tt) = mean(c_sim_with_500_betahigh(e_sim(:, 1) == 0, tt));
            mean_c_sim_u_without_500_betahigh(tt) = mean(c_sim_betahigh(e_sim(:, 1) == 0, tt));
        end

        mpc_500_u_by_t = (mean_c_sim_u_with_500 - mean_c_sim_u_without_500) / (shock_500);
        mpc_500_u_by_t_betahigh = (mean_c_sim_u_with_500_betahigh - mean_c_sim_u_without_500_betahigh) / (shock_500);

        for t = 1:numsim
            mpc_500_u_cum_dynamic(t) = sum(mpc_500_u_by_t(1:t));
            mpc_500_u_cum_dynamic_betahigh(t) = sum(mpc_500_u_by_t_betahigh(1:t));
        end

        if surprise == 1
            mpc_surprise_500_u_bywage(iy, :) = mpc_500_u_cum_dynamic(:);
            mpc_surprise_500_u_bywage_betahigh(iy, :) = mpc_500_u_cum_dynamic_betahigh(:);
        else
            mpc_expect_500_u_bywage(iy, :) = mpc_500_u_cum_dynamic(:);
            mpc_expect_500_u_bywage_betahigh(iy, :) = mpc_500_u_cum_dynamic_betahigh(:);
        end

        mpc_500_by_t = (mean_c_sim_with_500 - mean_c_sim) / (shock_500);
        mpc_500_by_t_betahigh = (mean_c_sim_with_500_betahigh - mean_c_sim_betahigh) / (shock_500);

        for t = 1:numsim
            mpc_500_cum_dynamic(t) = sum(mpc_500_by_t(1:t));
            mpc_500_cum_dynamic_betahigh(t) = sum(mpc_500_by_t_betahigh(1:t));
        end

        %mpc_500_cum_dynamic(3)
        if surprise == 1
            mpc_surprise_500_bywage(iy, :) = mpc_500_cum_dynamic(:);
            mpc_surprise_500_bywage_betahigh(iy, :) = mpc_500_cum_dynamic_betahigh(:);
        else
            mpc_expect_500_bywage(iy, :) = mpc_500_cum_dynamic(:);
            mpc_expect_500_bywage_betahigh(iy, :) = mpc_500_cum_dynamic_betahigh(:);
        end

        mean_a_sim_with_2400 = mean(a_sim_with_2400);
        mean_c_sim_with_2400 = mean(c_sim_with_2400);
        
        mean_a_sim_with_2400_betahigh = mean(a_sim_with_2400_betahigh);
        mean_c_sim_with_2400_betahigh = mean(c_sim_with_2400_betahigh);

        for tt = 1:numsim
            mean_c_sim_e_with_2400(tt) = mean(c_sim_with_2400(e_sim(:, 1) == 1, tt));
            mean_c_sim_e_without_2400(tt) = mean(c_sim(e_sim(:, 1) == 1, tt));
            
            mean_c_sim_e_with_2400_betahigh(tt) = mean(c_sim_with_2400_betahigh(e_sim(:, 1) == 1, tt));
            mean_c_sim_e_without_2400_betahigh(tt) = mean(c_sim_betahigh(e_sim(:, 1) == 1, tt));
        end

        mpc_2400_e_by_t = (mean_c_sim_e_with_2400 - mean_c_sim_e_without_2400) / (FPUC_expiration);
        mpc_2400_e_by_t_betahigh = (mean_c_sim_e_with_2400_betahigh - mean_c_sim_e_without_2400_betahigh) / (FPUC_expiration);

        for t = 1:numsim
            mpc_2400_e_cum_dynamic(t) = sum(mpc_2400_e_by_t(1:t));
            mpc_2400_e_cum_dynamic_betahigh(t) = sum(mpc_2400_e_by_t_betahigh(1:t));
        end

        if surprise == 1
            mpc_surprise_2400_e_bywage(iy, :) = mpc_2400_e_cum_dynamic(:);
            mpc_surprise_2400_e_bywage_betahigh(iy, :) = mpc_2400_e_cum_dynamic_betahigh(:);
        else
            mpc_expect_2400_e_bywage(iy, :) = mpc_2400_e_cum_dynamic(:);
            mpc_expect_2400_e_bywage_betahigh(iy, :) = mpc_2400_e_cum_dynamic_betahigh(:);
        end

        for tt = 1:numsim
            mean_c_sim_u_with_2400(tt) = mean(c_sim_with_2400(e_sim(:, 1) == 0, tt));
            mean_c_sim_u_without_2400(tt) = mean(c_sim(e_sim(:, 1) == 0, tt));
            
            mean_c_sim_u_with_2400_betahigh(tt) = mean(c_sim_with_2400_betahigh(e_sim(:, 1) == 0, tt));
            mean_c_sim_u_without_2400_betahigh(tt) = mean(c_sim_betahigh(e_sim(:, 1) == 0, tt));
        end

        mpc_2400_u_by_t = (mean_c_sim_u_with_2400 - mean_c_sim_u_without_2400) / (FPUC_expiration);
        mpc_2400_u_by_t_betahigh = (mean_c_sim_u_with_2400_betahigh - mean_c_sim_u_without_2400_betahigh) / (FPUC_expiration);

        for t = 1:numsim
            mpc_2400_u_cum_dynamic(t) = sum(mpc_2400_u_by_t(1:t));
            mpc_2400_u_cum_dynamic_betahigh(t) = sum(mpc_2400_u_by_t_betahigh(1:t));
        end

        if surprise == 1
            mpc_surprise_2400_u_bywage(iy, :) = mpc_2400_u_cum_dynamic(:);
            mpc_surprise_2400_u_bywage_betahigh(iy, :) = mpc_2400_u_cum_dynamic_betahigh(:);
        else
            mpc_expect_2400_u_bywage(iy, :) = mpc_2400_u_cum_dynamic(:);
            mpc_expect_2400_u_bywage_betahigh(iy, :) = mpc_2400_u_cum_dynamic_betahigh(:);
        end

        mpc_2400_by_t = (mean_c_sim_with_2400 - mean_c_sim) / (FPUC_expiration);
        mpc_2400_by_t_betahigh = (mean_c_sim_with_2400_betahigh - mean_c_sim_betahigh) / (FPUC_expiration);

        for t = 1:numsim
            mpc_2400_cum_dynamic(t) = sum(mpc_2400_by_t(1:t));
            mpc_2400_cum_dynamic_betahigh(t) = sum(mpc_2400_by_t_betahigh(1:t));
        end

        %mpc_2400_cum_dynamic(3)
        if surprise == 1
            mpc_surprise_2400_bywage(iy, :) = mpc_2400_cum_dynamic(:);
            mpc_surprise_2400_bywage_betahigh(iy, :) = mpc_2400_cum_dynamic_betahigh(:);
        else
            mpc_expect_2400_bywage(iy, :) = mpc_2400_cum_dynamic(:);
            mpc_expect_2400_bywage_betahigh(iy, :) = mpc_2400_cum_dynamic_betahigh(:);
        end

        %this is looping over just unemployed households (continuously unemployed)
        %to get u time-series patterns
        length_u = 0;

        for t = 1:numsim
            length_u = min(length_u + 1, n_b);
            c_pol_u = c_pol_u_betanormal;
            if t == 1
                c_pol_u_pandemic = c_pol_u_pandemic_betahigh;
            else
                c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
            end
            v_e=v_e_betanormal;
            v_u=v_u_betanormal;
            v_u_pandemic=v_u_pandemic_betanormal;
            beta=beta_normal;
                
            for i = 1:num_unemployed_hh

                %allow for initial assets, isomorphic to borrowing
                if length_u==1
                    a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t);
                    a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t);
                    a_sim_pandemic_surprise_extramonth(i, t)=a_sim_pandemic_surprise_extramonth(i, t);
                    a_sim_pandemic_expect_wait(i,t)=a_sim_pandemic_expect_wait(i,t);
                    a_sim_pandemic_surprise_wait(i,t)=a_sim_pandemic_surprise_wait(i,t);
                    a_sim_pandemic_surprise_onlyasseteffect(i,t)=a_sim_pandemic_surprise_onlyasseteffect(i,t);
                    a_sim_pandemic_surprise_noasseteffect(i,t)=a_sim_pandemic_surprise_noasseteffect(i,t);
                    a_sim_pandemic_surprise_FPUC_uncond(i, t)=a_sim_pandemic_surprise_FPUC_uncond(i, t);
                    a_sim_regular(i,t)=a_sim_regular(i,1);
                end
                %LWA
                if length_u == 6
                    a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + LWAsize;
                    a_sim_pandemic_surprise(i, t) = a_sim_pandemic_surprise(i, t) + LWAsize;
                    a_sim_pandemic_surprise_extramonth(i, t) = a_sim_pandemic_surprise_extramonth(i, t) + LWAsize;
                    a_sim_pandemic_expect_wait(i, t) = a_sim_pandemic_expect_wait(i, t) + LWAsize;
                    a_sim_pandemic_surprise_wait(i, t) = a_sim_pandemic_surprise_wait(i, t) + LWAsize;
                    a_sim_pandemic_surprise_onlyasseteffect(i, t) = a_sim_pandemic_surprise_onlyasseteffect(i, t) + LWAsize;
                    a_sim_pandemic_surprise_FPUC_uncond(i, t) = a_sim_pandemic_surprise_FPUC_uncond(i, t) + LWAsize;
                    a_sim_pandemic_surprise_noasseteffect(i, t) = a_sim_pandemic_surprise_noasseteffect(i, t) + LWAsize;
                end

                %Jan EIP
                if length_u == 10
                    a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise(i, t) = a_sim_pandemic_surprise(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_LWAperm(i, t) = a_sim_pandemic_LWAperm(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise_extramonth(i, t) = a_sim_pandemic_surprise_extramonth(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_expect_wait(i, t) = a_sim_pandemic_expect_wait(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise_wait(i, t) = a_sim_pandemic_surprise_wait(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_noFPUC(i, t) = a_sim_pandemic_noFPUC(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise_onlyasseteffect(i, t) = a_sim_pandemic_surprise_onlyasseteffect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise_FPUC_uncond(i, t) = a_sim_pandemic_surprise_FPUC_uncond(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    a_sim_pandemic_surprise_noasseteffect(i, t) = a_sim_pandemic_surprise_noasseteffect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                end
                

                c_sim_regular(i, t) = interp1(A, c_pol_u(:, length_u), a_sim_regular(i, t), 'linear');
                a_sim_regular(i, t + 1) = max(benefit_profile(length_u) + (1 + r) * a_sim_regular(i, t) - c_sim_regular(i, t), 0);

                c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect(i, t), 'linear');
                a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);

                c_sim_pandemic_expect_wait(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_expect_wait(i, t), 'linear');
                a_sim_pandemic_expect_wait(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_expect_wait(i, t) - c_sim_pandemic_expect_wait(i, t), 0);

                %Note this will vary with params, but can just save it accordingly
                %when taking means later
                c_sim_pandemic_noFPUC(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 5), a_sim_pandemic_noFPUC(i, t), 'linear');
                a_sim_pandemic_noFPUC(i, t + 1) = max(benefit_profile_pandemic(length_u, 5) + (1 + r) * a_sim_pandemic_noFPUC(i, t) - c_sim_pandemic_noFPUC(i, t), 0);
                
                c_sim_pandemic_surprise_onlyasseteffect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 5), a_sim_pandemic_surprise_onlyasseteffect(i, t), 'linear');
                a_sim_pandemic_surprise_onlyasseteffect(i, t + 1) = max(benefit_profile_pandemic(length_u, 5) + (1 + r) * a_sim_pandemic_surprise_onlyasseteffect(i, t) - c_sim_pandemic_surprise_onlyasseteffect(i, t), 0);
                if length_u==5
                    a_sim_pandemic_surprise_onlyasseteffect(i, t + 1) = a_sim_pandemic_surprise_onlyasseteffect(i, t + 1)-7*FPUC_expiration;
                end


                %FPUC unconditional (12 months of anticipated FPUC, then
                %surprise expiry in August) 
            
                if length_u <= 4
                    c_sim_pandemic_surprise_FPUC_uncond(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 7), a_sim_pandemic_surprise_FPUC_uncond(i, t), 'linear');
                    a_sim_pandemic_surprise_FPUC_uncond(i, t + 1) = max(benefit_profile_pandemic(length_u, 7) + (1 + r) * a_sim_pandemic_surprise_FPUC_uncond(i, t) - c_sim_pandemic_surprise_FPUC_uncond(i, t), 0);
                else
                    c_sim_pandemic_surprise_FPUC_uncond(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 8), a_sim_pandemic_surprise_FPUC_uncond(i, t), 'linear');
                    a_sim_pandemic_surprise_FPUC_uncond(i, t + 1) = max(benefit_profile_pandemic(length_u, 8) + (1 + r) * a_sim_pandemic_surprise_FPUC_uncond(i, t) - c_sim_pandemic_surprise_FPUC_uncond(i, t), 0);
                end
                

                if length_u <= 4
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);

                    c_sim_pandemic_surprise_wait(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 4), a_sim_pandemic_surprise_wait(i, t), 'linear');
                    a_sim_pandemic_surprise_wait(i, t + 1) = max(benefit_profile_pandemic(length_u, 4) + (1 + r) * a_sim_pandemic_surprise_wait(i, t) - c_sim_pandemic_surprise_wait(i, t), 0);

                else
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);

                    c_sim_pandemic_surprise_wait(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_surprise_wait(i, t), 'linear');
                    a_sim_pandemic_surprise_wait(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_surprise_wait(i, t) - c_sim_pandemic_surprise_wait(i, t), 0);
                end

                if length_u <= 4 %same as surprise ben profile
                    c_sim_pandemic_LWAperm(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_LWAperm(i, t), 'linear');
                    a_sim_pandemic_LWAperm(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_LWAperm(i, t) - c_sim_pandemic_LWAperm(i, t), 0);
                elseif length_u == 6 %now with LWA supplement expected to be permanent
                    c_sim_pandemic_LWAperm(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 6), a_sim_pandemic_LWAperm(i, t), 'linear');
                    a_sim_pandemic_LWAperm(i, t + 1) = max(benefit_profile_pandemic(length_u, 6) + (1 + r) * a_sim_pandemic_LWAperm(i, t) - c_sim_pandemic_LWAperm(i, t), 0);
                else %supplement is now off
                    c_sim_pandemic_LWAperm(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_LWAperm(i, t), 'linear');
                    a_sim_pandemic_LWAperm(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_LWAperm(i, t) - c_sim_pandemic_LWAperm(i, t), 0);
                end

                if length_u <= 5
                    c_sim_pandemic_surprise_extramonth(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_surprise_extramonth(i, t), 'linear');
                    a_sim_pandemic_surprise_extramonth(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_surprise_extramonth(i, t) - c_sim_pandemic_surprise_extramonth(i, t), 0);
                else
                    c_sim_pandemic_surprise_extramonth(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise_extramonth(i, t), 'linear');
                    a_sim_pandemic_surprise_extramonth(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise_extramonth(i, t) - c_sim_pandemic_surprise_extramonth(i, t), 0);
                end
                
                if length_u==4
                    a_sim_pandemic_surprise_noasseteffect(i,t)=tmp_a_unemployed(i);
                end
                if length_u <= 4
                    c_sim_pandemic_surprise_noasseteffect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_surprise_noasseteffect(i, t), 'linear');
                    a_sim_pandemic_surprise_noasseteffect(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_surprise_noasseteffect(i, t) - c_sim_pandemic_surprise_noasseteffect(i, t), 0);

                else
                    c_sim_pandemic_surprise_noasseteffect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise_noasseteffect(i, t), 'linear');
                    a_sim_pandemic_surprise_noasseteffect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise_noasseteffect(i, t) - c_sim_pandemic_surprise_noasseteffect(i, t), 0);
                end
   

                diff_v = interp1(A, v_e(:), a_sim_regular(i, t + 1), 'linear') - interp1(A, v_u(:, min(length_u + 1, n_b)), a_sim_regular(i, t + 1), 'linear');
                search_sim_regular(i, t) = min(1 - recall_probs_regular(ib), max(0, (beta * (diff_v) / k_prepandemic).^(1 / gamma_prepandemic)));

                diff_v = interp1(A, v_e(:), a_sim_pandemic_noFPUC(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 5), a_sim_pandemic_noFPUC(i, t + 1), 'linear');
                search_sim_pandemic_noFPUC(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_noFPUC(i, t)) ~= 0
                    search_sim_pandemic_noFPUC(i, t) = 0;
                end
                
                
                diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise_onlyasseteffect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 5), a_sim_pandemic_surprise_onlyasseteffect(i, t + 1), 'linear');
                search_sim_pandemic_surprise_onlyasseteffect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_surprise_onlyasseteffect(i, t)) ~= 0
                    search_sim_pandemic_surprise_onlyasseteffect(i, t) = 0;
                end


              
                %FPUC unconditional (12 months of anticipated FPUC, then
                %surprise expiry in August) 
                if length_u <= 4
                    diff_v = interp1(A, v_e_with_transfer(:, min(length_u + 1, n_b), 1), a_sim_pandemic_surprise_FPUC_uncond(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 7), a_sim_pandemic_surprise_FPUC_uncond(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e_with_transfer(:, min(length_u + 1, n_b), 2), a_sim_pandemic_surprise_FPUC_uncond(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 8), a_sim_pandemic_surprise_FPUC_uncond(i, t + 1), 'linear');
                end
                search_sim_pandemic_surprise_FPUC_uncond(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_surprise_FPUC_uncond(i, t)) ~= 0
                    search_sim_pandemic_surprise_FPUC_uncond(i, t) = 0;
                end


                

                diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_expect(i, t + 1), 'linear');
                search_sim_pandemic_expect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_expect(i, t)) ~= 0
                    search_sim_pandemic_expect(i, t) = 0;
                end

                diff_v = interp1(A, v_e(:), a_sim_pandemic_expect_wait(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_expect_wait(i, t + 1), 'linear');
                search_sim_pandemic_expect_wait(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_expect_wait(i, t)) ~= 0
                    search_sim_pandemic_expect_wait(i, t) = 0;
                end

                if length_u <= 4
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 2), a_sim_pandemic_surprise(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_surprise(i, t + 1), 'linear');
                end
                search_sim_pandemic_surprise(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_surprise(i, t)) ~= 0
                    search_sim_pandemic_surprise(i, t) = 0;
                end

                if length_u <= 4
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise_wait(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 4), a_sim_pandemic_surprise_wait(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise_wait(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_surprise_wait(i, t + 1), 'linear');
                end

                search_sim_pandemic_surprise_wait(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_surprise_wait(i, t)) ~= 0
                    search_sim_pandemic_surprise_wait(i, t) = 0;
                end

                if length_u <= 4 %same as surprise case
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_LWAperm(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 2), a_sim_pandemic_LWAperm(i, t + 1), 'linear');
                elseif length_u == 6 %now expect LWA to last until benefit exhaustion
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_LWAperm(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 6), a_sim_pandemic_LWAperm(i, t + 1), 'linear');
                else %now no supplement
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_LWAperm(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_LWAperm(i, t + 1), 'linear');
                end

                search_sim_pandemic_LWAperm(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_LWAperm(i, t)) ~= 0
                    search_sim_pandemic_LWAperm(i, t) = 0;
                end

                if length_u <= 4
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise_noasseteffect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 2), a_sim_pandemic_surprise_noasseteffect(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise_noasseteffect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_surprise_noasseteffect(i, t + 1), 'linear');
                end
                search_sim_pandemic_surprise_noasseteffect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_surprise_noasseteffect(i, t)) ~= 0
                    search_sim_pandemic_surprise_noasseteffect(i, t) = 0;
                end

            end

        end

        %this is looping over just employed households (continuously employed)
        %to get e time-series patterns
        for t = 1:numsim

            for i = 1:num_employed_hh

                if t == 4
                    c_pol_e = c_pol_e_betahigh;
                else
                    c_pol_e = c_pol_e_betanormal;
                end

                if t == 12
                    a_sim_e(i, t) = a_sim_e(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                end

                %adjust initial assets isomorphic to allowing for borrowing
                if t==1
                    a_sim_e(i,t)=a_sim_e(i,t);
                end

                c_sim_e(i, t) = interp1(A, c_pol_e(:), a_sim_e(i, t), 'linear');
                a_sim_e(i, t + 1) = y + (1 + r) * a_sim_e(i, t) - c_sim_e(i, t);
            end

        end

        if surprise == 1
            mean_a_sim_pandemic_surprise = mean(a_sim_pandemic_surprise, 1);
            mean_c_sim_pandemic_surprise = mean(c_sim_pandemic_surprise, 1);

            mean_a_sim_pandemic_surprise_extramonth = mean(a_sim_pandemic_surprise_extramonth, 1);
            mean_c_sim_pandemic_surprise_extramonth = mean(c_sim_pandemic_surprise_extramonth, 1);

            mean_a_sim_pandemic_surprise_noFPUC = mean(a_sim_pandemic_noFPUC, 1);
            mean_c_sim_pandemic_surprise_noFPUC = mean(c_sim_pandemic_noFPUC, 1);
            
            mean_a_sim_pandemic_surprise_onlyasseteffect = mean(a_sim_pandemic_surprise_onlyasseteffect, 1);
            mean_c_sim_pandemic_surprise_onlyasseteffect = mean(c_sim_pandemic_surprise_onlyasseteffect, 1);

            mean_a_sim_pandemic_surprise_FPUC_uncond = mean(a_sim_pandemic_surprise_FPUC_uncond, 1);
            mean_c_sim_pandemic_surprise_FPUC_uncond = mean(c_sim_pandemic_surprise_FPUC_uncond, 1);

            mean_a_sim_pandemic_surprise_noasseteffect = mean(a_sim_pandemic_surprise_noasseteffect, 1);
            mean_c_sim_pandemic_surprise_noasseteffect = mean(c_sim_pandemic_surprise_noasseteffect, 1);

            mean_a_sim_pandemic_surprise_wait = mean(a_sim_pandemic_surprise_wait, 1);
            mean_c_sim_pandemic_surprise_wait = mean(c_sim_pandemic_surprise_wait, 1);

            mean_a_sim_regular = mean(a_sim_regular, 1);
            mean_c_sim_regular = mean(c_sim_regular, 1);

            mean_a_sim_e = mean(a_sim_e, 1);
            mean_c_sim_e = mean(c_sim_e, 1);

            mean_a_sim_pandemic_LWAperm = mean(a_sim_pandemic_LWAperm, 1);
            mean_c_sim_pandemic_LWAperm = mean(c_sim_pandemic_LWAperm, 1);

            mean_search_sim_pandemic_surprise_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
            mean_search_sim_pandemic_surprise_onlyasseteffect = mean(search_sim_pandemic_surprise_onlyasseteffect, 1);
            mean_search_sim_pandemic_surprise_FPUC_uncond = mean(search_sim_pandemic_surprise_FPUC_uncond, 1);
            mean_search_sim_pandemic_surprise_noasseteffect = mean(search_sim_pandemic_surprise_noasseteffect, 1);
            mean_search_sim_pandemic_surprise = mean(search_sim_pandemic_surprise, 1);
            mean_search_sim_pandemic_LWAperm = mean(search_sim_pandemic_LWAperm, 1);
            mean_search_sim_pandemic_surprise_wait = mean(search_sim_pandemic_surprise_wait, 1);
        else
            mean_a_sim_pandemic_expect = mean(a_sim_pandemic_expect, 1);
            mean_c_sim_pandemic_expect = mean(c_sim_pandemic_expect, 1);

            mean_a_sim_pandemic_expect_noFPUC = mean(a_sim_pandemic_noFPUC, 1);
            mean_c_sim_pandemic_expect_noFPUC = mean(c_sim_pandemic_noFPUC, 1);
            
            mean_a_sim_pandemic_expect_onlyasseteffect = mean(a_sim_pandemic_surprise_onlyasseteffect, 1);
            mean_c_sim_pandemic_expect_onlyasseteffect = mean(c_sim_pandemic_surprise_onlyasseteffect, 1);
            
            mean_a_sim_pandemic_expect_noasseteffect = mean(a_sim_pandemic_surprise_noasseteffect, 1);
            mean_c_sim_pandemic_expect_noasseteffect = mean(c_sim_pandemic_surprise_noasseteffect, 1);

            mean_a_sim_pandemic_expect_wait = mean(a_sim_pandemic_expect_wait, 1);
            mean_c_sim_pandemic_expect_wait = mean(c_sim_pandemic_expect_wait, 1);

            mean_search_sim_regular = mean(search_sim_regular, 1);
            mean_search_sim_pandemic_expect = mean(search_sim_pandemic_expect, 1);
            mean_search_sim_pandemic_expect_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
            mean_search_sim_pandemic_expect_onlyasseteffect = mean(search_sim_pandemic_surprise_onlyasseteffect, 1);
            mean_search_sim_pandemic_expect_noasseteffect = mean(search_sim_pandemic_surprise_noasseteffect, 1);
            mean_search_sim_pandemic_expect_wait = mean(search_sim_pandemic_expect_wait, 1);
        end

    end

    mean_c_sim_e_bywage(iy, :) = mean_c_sim_e;
    mean_a_sim_e_bywage(iy, :) = mean_a_sim_e;

    %paste on initial Jan-March 3 months of employment
    mean_a_sim_pandemic_surprise_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise(1:numsim - 3)];
    mean_a_sim_pandemic_LWAperm_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_LWAperm(1:numsim - 3)];
    mean_c_sim_pandemic_LWAperm_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_LWAperm(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_extramonth_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_extramonth(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_extramonth_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_extramonth(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_wait_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_wait(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_wait_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_wait(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_noFPUC_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_noFPUC(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_noFPUC_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_noFPUC(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_onlyasseteffect_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_onlyasseteffect(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_onlyasseteffect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_onlyasseteffect(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_FPUC_uncond_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_FPUC_uncond(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_FPUC_uncond_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_FPUC_uncond(1:numsim - 3)];
    mean_a_sim_pandemic_surprise_noasseteffect_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_noasseteffect(1:numsim - 3)];
    mean_c_sim_pandemic_surprise_noasseteffect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_noasseteffect(1:numsim - 3)];
    mean_a_sim_pandemic_expect_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_expect(1:numsim - 3)];
    mean_c_sim_pandemic_expect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect(1:numsim - 3)];
    mean_a_sim_pandemic_expect_wait_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_wait(1:numsim - 3)];
    mean_c_sim_pandemic_expect_wait_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_wait(1:numsim - 3)];
    mean_a_sim_pandemic_expect_noFPUC_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_noFPUC(1:numsim - 3)];
    mean_c_sim_pandemic_expect_noFPUC_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_noFPUC(1:numsim - 3)];
    mean_a_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_onlyasseteffect(1:numsim - 3)];
    mean_c_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_onlyasseteffect(1:numsim - 3)];
    mean_a_sim_pandemic_expect_noasseteffect_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_noasseteffect(1:numsim - 3)];
    mean_c_sim_pandemic_expect_noasseteffect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_noasseteffect(1:numsim - 3)];
    mean_a_sim_regular_bywage(iy, :) = [mean_a_sim_e(1:3) mean_a_sim_regular(1:numsim - 3)];
    mean_c_sim_regular_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_regular(1:numsim - 3)];

    mean_y_sim_pandemic_u_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 1)'];
    mean_y_sim_pandemic_u_bywage(iy,4)=mean_y_sim_pandemic_u_bywage(iy,4);
    mean_y_sim_pandemic_u_bywage(iy, 9) = mean_y_sim_pandemic_u_bywage(iy, 9) + LWAsize;
    mean_y_sim_pandemic_u_bywage(iy, 13) = mean_y_sim_pandemic_u_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    mean_y_sim_pandemic_wait_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 3)'];
    mean_y_sim_pandemic_wait_bywage(iy,4)=mean_y_sim_pandemic_wait_bywage(iy,4);
    mean_y_sim_pandemic_wait_bywage(iy, 9) = mean_y_sim_pandemic_wait_bywage(iy, 9) + LWAsize;
    mean_y_sim_pandemic_wait_bywage(iy, 13) = mean_y_sim_pandemic_wait_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    mean_y_sim_pandemic_noFPUC_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 5)'];
    mean_y_sim_pandemic_noFPUC_bywage(iy,4)=mean_y_sim_pandemic_noFPUC_bywage(iy,4);
    mean_y_sim_pandemic_noFPUC_bywage(iy, 13) = mean_y_sim_pandemic_noFPUC_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    
    mean_y_sim_pandemic_onlyasseteffect_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 5)'];
    mean_y_sim_pandemic_onlyasseteffect_bywage(iy,4:7)=mean_y_sim_pandemic_onlyasseteffect_bywage(iy,4:7)+FPUC_expiration;
    mean_y_sim_pandemic_onlyasseteffect_bywage(iy,4)=mean_y_sim_pandemic_noFPUC_bywage(iy,4);
    mean_y_sim_pandemic_onlyasseteffect_bywage(iy, 13) = mean_y_sim_pandemic_noFPUC_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    
    mean_y_sim_pandemic_noasseteffect_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 5)'];
    mean_y_sim_pandemic_noasseteffect_bywage(iy,4:7)=mean_y_sim_pandemic_onlyasseteffect_bywage(iy,4:7)+FPUC_expiration;
    mean_y_sim_pandemic_noasseteffect_bywage(iy,4)=mean_y_sim_pandemic_noFPUC_bywage(iy,4);
    mean_y_sim_pandemic_noasseteffect_bywage(iy, 13) = mean_y_sim_pandemic_noFPUC_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);

    mean_y_sim_regular_bywage(iy, :) = [y y y benefit_profile(:, 1)'];
    mean_y_sim_regular_bywage(iy, 4) = mean_y_sim_regular_bywage(iy, 4);

    
    mean_y_sim_e_bywage(iy,:)=y*ones(16,1);
    mean_y_sim_e_bywage(iy,4)=mean_y_sim_e_bywage(iy,4);

    mean_search_sim_regular_bywage(iy, :) = [NaN NaN NaN mean_search_sim_regular(1:numsim - 3)];
    mean_search_sim_pandemic_expect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect(1:numsim - 3)];
    mean_search_sim_pandemic_expect_wait_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_wait(1:numsim - 3)];
    mean_search_sim_pandemic_expect_noFPUC_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_noFPUC(1:numsim - 3)];
    mean_search_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_onlyasseteffect(1:numsim - 3)];
    mean_search_sim_pandemic_expect_noasseteffect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_noasseteffect(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise(1:numsim - 3)];
    mean_search_sim_pandemic_LWAperm_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_LWAperm(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_wait_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise_wait(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_noFPUC_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise_noFPUC(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_onlyasseteffect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise_onlyasseteffect(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_FPUC_uncond_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise_FPUC_uncond(1:numsim - 3)];
    mean_search_sim_pandemic_surprise_noasseteffect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_surprise_noasseteffect(1:numsim - 3)];

    mean_c_sim_pandemic_surprise_vs_e_bywage(iy, :) = mean_c_sim_pandemic_surprise ./ mean_c_sim_e;
    mean_c_sim_pandemic_LWAperm_vs_e_bywage(iy, :) = mean_c_sim_pandemic_LWAperm ./ mean_c_sim_e;
    mean_c_sim_pandemic_surprise_extramonth_vs_e_bywage(iy, :) = mean_c_sim_pandemic_surprise_extramonth ./ mean_c_sim_e;
    mean_c_sim_pandemic_surprise_wait_vs_e_bywage(iy, :) = mean_c_sim_pandemic_surprise_wait ./ mean_c_sim_e;
    mean_c_sim_pandemic_expect_vs_e_bywage(iy, :) = mean_c_sim_pandemic_expect ./ mean_c_sim_e;
    mean_c_sim_pandemic_expect_wait_vs_e_bywage(iy, :) = mean_c_sim_pandemic_expect_wait ./ mean_c_sim_e;

end

mpc_surprise_500_quarterly = mean(mpc_surprise_500_bywage(:, 3));
mpc_surprise_500_monthly = mean(mpc_surprise_500_bywage(:, 1));
mpc_expect_500_quarterly = mean(mpc_expect_500_bywage(:, 3));
mpc_expect_500_monthly = mean(mpc_expect_500_bywage(:, 1));

mpc_surprise_500_e_quarterly = mean(mpc_surprise_500_e_bywage(:, 3));
mpc_surprise_500_e_monthly = mean(mpc_surprise_500_e_bywage(:, 1));
mpc_expect_500_e_quarterly = mean(mpc_expect_500_e_bywage(:, 3));
mpc_expect_500_e_monthly = mean(mpc_expect_500_e_bywage(:, 1));

mpc_surprise_500_u_quarterly = mean(mpc_surprise_500_u_bywage(:, 3));
mpc_surprise_500_u_monthly = mean(mpc_surprise_500_u_bywage(:, 1));
mpc_expect_500_u_quarterly = mean(mpc_expect_500_u_bywage(:, 3));
mpc_expect_500_u_monthly = mean(mpc_expect_500_u_bywage(:, 1));

mpc_surprise_2400_quarterly = mean(mpc_surprise_2400_bywage(:, 3));
mpc_surprise_2400_monthly = mean(mpc_surprise_2400_bywage(:, 1));
mpc_expect_2400_quarterly = mean(mpc_expect_2400_bywage(:, 3));
mpc_expect_2400_monthly = mean(mpc_expect_2400_bywage(:, 1));

mpc_surprise_2400_e_quarterly = mean(mpc_surprise_2400_e_bywage(:, 3));
mpc_surprise_2400_e_monthly = mean(mpc_surprise_2400_e_bywage(:, 1));
mpc_expect_2400_e_quarterly = mean(mpc_expect_2400_e_bywage(:, 3));
mpc_expect_2400_e_monthly = mean(mpc_expect_2400_e_bywage(:, 1));

mpc_surprise_2400_u_quarterly = mean(mpc_surprise_2400_u_bywage(:, 3));
mpc_surprise_2400_u_monthly = mean(mpc_surprise_2400_u_bywage(:, 1));
mpc_expect_2400_u_quarterly = mean(mpc_expect_2400_u_bywage(:, 3));
mpc_expect_2400_u_monthly = mean(mpc_expect_2400_u_bywage(:, 1));




mpc_surprise_500_quarterly_betahigh = mean(mpc_surprise_500_bywage_betahigh(:, 3));
mpc_surprise_500_monthly_betahigh = mean(mpc_surprise_500_bywage_betahigh(:, 1));
mpc_expect_500_quarterly_betahigh = mean(mpc_expect_500_bywage_betahigh(:, 3));
mpc_expect_500_monthly_betahigh = mean(mpc_expect_500_bywage_betahigh(:, 1));

mpc_surprise_500_e_quarterly_betahigh = mean(mpc_surprise_500_e_bywage_betahigh(:, 3));
mpc_surprise_500_e_monthly_betahigh = mean(mpc_surprise_500_e_bywage_betahigh(:, 1));
mpc_expect_500_e_quarterly_betahigh = mean(mpc_expect_500_e_bywage_betahigh(:, 3));
mpc_expect_500_e_monthly_betahigh = mean(mpc_expect_500_e_bywage_betahigh(:, 1));

mpc_surprise_500_u_quarterly_betahigh = mean(mpc_surprise_500_u_bywage_betahigh(:, 3));
mpc_surprise_500_u_monthly_betahigh = mean(mpc_surprise_500_u_bywage_betahigh(:, 1));
mpc_expect_500_u_quarterly_betahigh = mean(mpc_expect_500_u_bywage_betahigh(:, 3));
mpc_expect_500_u_monthly_betahigh = mean(mpc_expect_500_u_bywage_betahigh(:, 1));

mpc_surprise_2400_quarterly_betahigh = mean(mpc_surprise_2400_bywage_betahigh(:, 3));
mpc_surprise_2400_monthly_betahigh = mean(mpc_surprise_2400_bywage_betahigh(:, 1));
mpc_expect_2400_quarterly_betahigh = mean(mpc_expect_2400_bywage_betahigh(:, 3));
mpc_expect_2400_monthly_betahigh = mean(mpc_expect_2400_bywage_betahigh(:, 1));

mpc_surprise_2400_e_quarterly_betahigh = mean(mpc_surprise_2400_e_bywage_betahigh(:, 3));
mpc_surprise_2400_e_monthly_betahigh = mean(mpc_surprise_2400_e_bywage_betahigh(:, 1));
mpc_expect_2400_e_quarterly_betahigh = mean(mpc_expect_2400_e_bywage_betahigh(:, 3));
mpc_expect_2400_e_monthly_betahigh = mean(mpc_expect_2400_e_bywage_betahigh(:, 1));

mpc_surprise_2400_u_quarterly_betahigh = mean(mpc_surprise_2400_u_bywage_betahigh(:, 3));
mpc_surprise_2400_u_monthly_betahigh = mean(mpc_surprise_2400_u_bywage_betahigh(:, 1));
mpc_expect_2400_u_quarterly_betahigh = mean(mpc_expect_2400_u_bywage_betahigh(:, 3));
mpc_expect_2400_u_monthly_betahigh = mean(mpc_expect_2400_u_bywage_betahigh(:, 1));




mean_a_sim_e = mean(mean_a_sim_e_bywage, 1);
mean_a_sim_pandemic_surprise = mean(mean_a_sim_pandemic_surprise_bywage, 1);
mean_a_sim_pandemic_LWAperm = mean(mean_a_sim_pandemic_LWAperm_bywage, 1);
mean_a_sim_pandemic_surprise_extramonth = mean(mean_a_sim_pandemic_surprise_extramonth_bywage, 1);
mean_a_sim_pandemic_surprise_wait = mean(mean_a_sim_pandemic_surprise_wait_bywage, 1);
mean_a_sim_pandemic_surprise_noFPUC = mean(mean_a_sim_pandemic_surprise_noFPUC_bywage, 1);
mean_a_sim_pandemic_surprise_onlyasseteffect = mean(mean_a_sim_pandemic_surprise_onlyasseteffect_bywage, 1);
mean_a_sim_pandemic_surprise_noasseteffect = mean(mean_a_sim_pandemic_surprise_noasseteffect_bywage, 1);
mean_a_sim_pandemic_expect = mean(mean_a_sim_pandemic_expect_bywage, 1);
mean_a_sim_pandemic_expect_wait = mean(mean_a_sim_pandemic_expect_wait_bywage, 1);
mean_a_sim_pandemic_expect_noFPUC = mean(mean_a_sim_pandemic_expect_noFPUC_bywage, 1);
mean_a_sim_pandemic_expect_onlyasseteffect = mean(mean_a_sim_pandemic_expect_onlyasseteffect_bywage, 1);
mean_a_sim_pandemic_surprise_FPUC_uncond = mean(mean_a_sim_pandemic_surprise_FPUC_uncond_bywage, 1);
mean_a_sim_pandemic_expect_noasseteffect = mean(mean_a_sim_pandemic_expect_noasseteffect_bywage, 1);
mean_a_sim_pandemic_regular = mean(mean_a_sim_regular_bywage, 1);

mean_y_sim_pandemic_u = mean(mean_y_sim_pandemic_u_bywage, 1);
mean_y_sim_pandemic_wait = mean(mean_y_sim_pandemic_wait_bywage, 1);
mean_y_sim_pandemic_noFPUC = mean(mean_y_sim_pandemic_noFPUC_bywage, 1);
mean_y_sim_e = mean(mean_y_sim_e_bywage, 1);

mean_c_sim_e = mean(mean_c_sim_e_bywage, 1);
mean_c_sim_pandemic_surprise = mean(mean_c_sim_pandemic_surprise_bywage, 1);
mean_c_sim_pandemic_LWAperm = mean(mean_c_sim_pandemic_LWAperm_bywage, 1);
mean_c_sim_pandemic_surprise_extramonth = mean(mean_c_sim_pandemic_surprise_extramonth_bywage, 1);
mean_c_sim_pandemic_surprise_wait = mean(mean_c_sim_pandemic_surprise_wait_bywage, 1);
mean_c_sim_pandemic_surprise_noFPUC = mean(mean_c_sim_pandemic_surprise_noFPUC_bywage, 1);
mean_c_sim_pandemic_surprise_onlyasseteffect = mean(mean_c_sim_pandemic_surprise_onlyasseteffect_bywage, 1);
mean_c_sim_pandemic_surprise_FPUC_uncond = mean(mean_c_sim_pandemic_surprise_FPUC_uncond_bywage, 1);
mean_c_sim_pandemic_surprise_noasseteffect = mean(mean_c_sim_pandemic_surprise_noasseteffect_bywage, 1);
mean_c_sim_pandemic_expect = mean(mean_c_sim_pandemic_expect_bywage, 1);
mean_c_sim_pandemic_expect_wait = mean(mean_c_sim_pandemic_expect_wait_bywage, 1);
mean_c_sim_pandemic_expect_noFPUC = mean(mean_c_sim_pandemic_expect_noFPUC_bywage, 1);
mean_c_sim_pandemic_expect_onlyasseteffect = mean(mean_c_sim_pandemic_expect_onlyasseteffect_bywage, 1);
mean_c_sim_pandemic_expect_noasseteffect = mean(mean_c_sim_pandemic_expect_noasseteffect_bywage, 1);
mean_c_sim_regular = mean(mean_c_sim_regular_bywage, 1);

mean_search_sim_pandemic_surprise = mean(mean_search_sim_pandemic_surprise_bywage, 1);
mean_search_sim_pandemic_LWAperm = mean(mean_search_sim_pandemic_LWAperm_bywage, 1);
mean_search_sim_pandemic_surprise_wait = mean(mean_search_sim_pandemic_surprise_wait_bywage, 1);
mean_search_sim_pandemic_surprise_noFPUC = mean(mean_search_sim_pandemic_surprise_noFPUC_bywage, 1);
mean_search_sim_pandemic_surprise_onlyasseteffect = mean(mean_search_sim_pandemic_surprise_onlyasseteffect_bywage, 1);
mean_search_sim_pandemic_surprise_FPUC_uncond= mean(mean_search_sim_pandemic_surprise_FPUC_uncond_bywage, 1);
mean_search_sim_pandemic_surprise_noasseteffect = mean(mean_search_sim_pandemic_surprise_noasseteffect_bywage, 1);
mean_search_sim_pandemic_expect = mean(mean_search_sim_pandemic_expect_bywage, 1);
mean_search_sim_pandemic_expect_wait = mean(mean_search_sim_pandemic_expect_wait_bywage, 1);
mean_search_sim_pandemic_expect_noFPUC = mean(mean_search_sim_pandemic_expect_noFPUC_bywage, 1);
mean_search_sim_pandemic_expect_onlyasseteffect = mean(mean_search_sim_pandemic_expect_onlyasseteffect_bywage, 1);
mean_search_sim_pandemic_expect_noasseteffect = mean(mean_search_sim_pandemic_expect_noasseteffect_bywage, 1);
mean_search_sim_pandemic_regular = mean(mean_search_sim_regular_bywage, 1);
mean_search_sim_regular = mean(mean_search_sim_regular_bywage, 1);

%Convert model simulations to dollar deviations in U vs. E space
mean_c_sim_pandemic_surprise_dollars = mean_c_sim_pandemic_surprise ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_LWAperm_dollars = mean_c_sim_pandemic_LWAperm ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_extramonth_dollars = mean_c_sim_pandemic_surprise_extramonth ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_wait_dollars = mean_c_sim_pandemic_surprise_wait ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_noFPUC_dollars = mean_c_sim_pandemic_surprise_noFPUC ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_onlyasseteffect_dollars = mean_c_sim_pandemic_surprise_onlyasseteffect ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_FPUC_uncond_dollars = mean_c_sim_pandemic_surprise_FPUC_uncond ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_surprise_noasseteffect_dollars = mean_c_sim_pandemic_surprise_noasseteffect ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_regular_dollars = mean_c_sim_regular / mean_c_sim_e(1) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_expect_dollars = mean_c_sim_pandemic_expect ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_expect_wait_dollars = mean_c_sim_pandemic_expect_wait ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_expect_noFPUC_dollars = mean_c_sim_pandemic_expect_noFPUC ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_expect_onlyasseteffect_dollars = mean_c_sim_pandemic_expect_onlyasseteffect ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_pandemic_expect_noasseteffect_dollars = mean_c_sim_pandemic_expect_noasseteffect ./ mean_c_sim_e(1:numsim) * total_spend_u(1) - total_spend_u(1);
mean_c_sim_e_dollars = mean_c_sim_e(1:numsim) ./ mean_c_sim_e(1:numsim) * total_spend_e(1) - total_spend_e(1);

mean_y_sim_pandemic_u_dollars = mean_y_sim_pandemic_u./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_pandemic_u_extramonth_dollars = mean_y_sim_pandemic_u_dollars;
mean_y_sim_pandemic_u_extramonth_dollars(8) = mean_y_sim_pandemic_u_extramonth_dollars(8) + FPUC_expiration * income_u(1);
mean_y_sim_pandemic_wait_dollars = mean_y_sim_pandemic_wait./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_pandemic_noFPUC_dollars = mean_y_sim_pandemic_noFPUC./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_e_dollars = 0;

scale_factor=(total_spend_e(1)/income_e(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));

load prepandemic_andpandemic_results_target500MPC.mat
load nodiscountfactorshock_mpc_targetwaiting
load nodiscountfactorshock_mpc_target500MPC

figure
p = patch([4 4 7 7], [-3000 8000 8000 -30500], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
h(1) = plot(1:11, income_u(1)*u_v1(1:11), '--', 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410])
h(2) = plot(1:11, mean_y_sim_pandemic_u_dollars(1:11), 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410])
h(3) = plot(1:11, income_u(1)*w_v1(1:11), '--', 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980])
h(4) = plot(1:11, mean_y_sim_pandemic_wait_dollars(1:11), 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980])
uistack(h(2), 'top')
%title('Income of Unemployed vs. Employed')
legend('Location', 'NorthWest')
legend([h(1) h(2) h(3) h(4)], {'Receive UI starting April: Data', 'Receive UI starting April: Model', 'UI delayed until June: Data', 'UI delayed until June: Model'})
%ylabel('Differential Change in Monthly Income')
ylim([-3000 8000])
xlim([1 11])
xticks([1 2 3 4 5 6 7 8 9 10 11])
xticklabels(label_months_jan20_nov20)
yticks([-2500 0 2500 5000 7500])
yticklabels({'-$2,500', '$0', '$2,500', '$5,000', '$7,500'})
set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
set(gca, 'fontsize', 12);
set(gca, 'Layer', 'top');
set(gcf, 'PaperPosition', [0 0 10.4 4.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [3.5 2]); %Keep the same paper size
fig_paper_A26 = gcf;
saveas(fig_paper_A26, fullfile(release_path_paper, 'income_expiration.png'))






spend_data=us_v1*100;











%mean_c_sim_pandemic_surprise_baseline=mean_c_sim_pandemic_surprise;
%mean_c_sim_pandemic_surprise_wait_baseline=mean_c_sim_pandemic_surprise_wait;
%spend_series_baseline_calibration = mean_c_sim_pandemic_surprise_dollars(1:11)/total_spend_u(1)*100;
%save('robustness/waiting_expectations_v2/spend_series_baseline_calibration', 'spend_series_baseline_calibration','mean_c_sim_pandemic_surprise_baseline','mean_c_sim_pandemic_surprise_wait_baseline');

ratio1 = mean_c_sim_pandemic_surprise_dollars(1:11)/total_spend_u(1)*100;
ratio2 = mean_c_sim_pandemic_surprise_dollars_match500MPC(1:11)/total_spend_u(1)*100;
stats_for_text_april_ratio = ratio1(4) / ratio2(4);
save('stats_for_text_model_miscellaneous.mat', 'stats_for_text_april_ratio', '-append')


mpc = table();
mpc.surprise('waiting') = ((mean_c_sim_pandemic_surprise(5) - mean_c_sim_pandemic_surprise(3)) - (mean_c_sim_pandemic_surprise_wait(5) - mean_c_sim_pandemic_surprise_wait(3))) / ((mean_y_sim_pandemic_u(5) - mean_y_sim_pandemic_u(3)) - (mean_y_sim_pandemic_wait(5) - mean_y_sim_pandemic_wait(3)));
mpc.expected('waiting') = ((mean_c_sim_pandemic_expect(5) - mean_c_sim_pandemic_expect(3)) - (mean_c_sim_pandemic_expect_wait(5) - mean_c_sim_pandemic_expect_wait(3))) / ((mean_y_sim_pandemic_u(5) - mean_y_sim_pandemic_u(3)) - (mean_y_sim_pandemic_wait(5) - mean_y_sim_pandemic_wait(3)));
mpc.surprise('600 expiration') = ((mean_c_sim_pandemic_surprise(8) - mean_c_sim_pandemic_surprise(7)) - (mean_c_sim_e(8) - mean_c_sim_e(7))) / (mean_y_sim_pandemic_u(8) - mean_y_sim_pandemic_u(7));
mpc.expected('600 expiration') = ((mean_c_sim_pandemic_expect(8) - mean_c_sim_pandemic_expect(7)) - (mean_c_sim_e(8) - mean_c_sim_e(7))) / (mean_y_sim_pandemic_u(8) - mean_y_sim_pandemic_u(7));
mpc.expected('500 quarterly') = mpc_expect_500_quarterly;
mpc.surprise('500 quarterly') = mpc_surprise_500_quarterly;
mpc.expected('500 monthly') = mpc_expect_500_monthly;
mpc.surprise('500 monthly') = mpc_surprise_500_monthly;
mpc.expected('500 quarterly-employed') = mpc_expect_500_e_quarterly;
mpc.surprise('500 quarterly-employed') = mpc_surprise_500_e_quarterly;
mpc.expected('500 monthly-employed') = mpc_expect_500_e_monthly;
mpc.surprise('500 monthly-employed') = mpc_surprise_500_e_monthly;
mpc.expected('500 quarterly-unemployed') = mpc_expect_500_u_quarterly;
mpc.surprise('500 quarterly-unemployed') = mpc_surprise_500_u_quarterly;
mpc.expected('500 monthly-unemployed') = mpc_expect_500_u_monthly;
mpc.surprise('500 monthly-unemployed') = mpc_surprise_500_u_monthly
mpc.expected('2400 quarterly') = mpc_expect_2400_quarterly;
mpc.surprise('2400 quarterly') = mpc_surprise_2400_quarterly;
mpc.expected('2400 monthly') = mpc_expect_2400_monthly;
mpc.surprise('2400 monthly') = mpc_surprise_2400_monthly;
mpc.expected('2400 quarterly-employed') = mpc_expect_2400_e_quarterly;
mpc.surprise('2400 quarterly-employed') = mpc_surprise_2400_e_quarterly;
mpc.expected('2400 monthly-employed') = mpc_expect_2400_e_monthly;
mpc.surprise('2400 monthly-employed') = mpc_surprise_2400_e_monthly;
mpc.expected('2400 quarterly-unemployed') = mpc_expect_2400_u_quarterly;
mpc.surprise('2400 quarterly-unemployed') = mpc_surprise_2400_u_quarterly;
mpc.expected('2400 monthly-unemployed') = mpc_expect_2400_u_monthly;
mpc.surprise('2400 monthly-unemployed') = mpc_surprise_2400_u_monthly;
mpc.Variables=scale_factor*mpc.Variables

mpc_betahigh=table();
mpc_betahigh.expected('500 quarterly') = mpc_expect_500_quarterly_betahigh;
mpc_betahigh.surprise('500 quarterly') = mpc_surprise_500_quarterly_betahigh;
mpc_betahigh.expected('500 monthly') = mpc_expect_500_monthly_betahigh;
mpc_betahigh.surprise('500 monthly') = mpc_surprise_500_monthly_betahigh;
mpc_betahigh.expected('500 quarterly-employed') = mpc_expect_500_e_quarterly_betahigh;
mpc_betahigh.surprise('500 quarterly-employed') = mpc_surprise_500_e_quarterly_betahigh;
mpc_betahigh.expected('500 monthly-employed') = mpc_expect_500_e_monthly_betahigh;
mpc_betahigh.surprise('500 monthly-employed') = mpc_surprise_500_e_monthly_betahigh;
mpc_betahigh.expected('500 quarterly-unemployed') = mpc_expect_500_u_quarterly_betahigh;
mpc_betahigh.surprise('500 quarterly-unemployed') = mpc_surprise_500_u_quarterly_betahigh;
mpc_betahigh.expected('500 monthly-unemployed') = mpc_expect_500_u_monthly_betahigh;
mpc_betahigh.surprise('500 monthly-unemployed') = mpc_surprise_500_u_monthly_betahigh
mpc_betahigh.expected('2400 quarterly') = mpc_expect_2400_quarterly_betahigh;
mpc_betahigh.surprise('2400 quarterly') = mpc_surprise_2400_quarterly_betahigh;
mpc_betahigh.expected('2400 monthly') = mpc_expect_2400_monthly_betahigh;
mpc_betahigh.surprise('2400 monthly') = mpc_surprise_2400_monthly_betahigh;
mpc_betahigh.expected('2400 quarterly-employed') = mpc_expect_2400_e_quarterly_betahigh;
mpc_betahigh.surprise('2400 quarterly-employed') = mpc_surprise_2400_e_quarterly_betahigh;
mpc_betahigh.expected('2400 monthly-employed') = mpc_expect_2400_e_monthly_betahigh;
mpc_betahigh.surprise('2400 monthly-employed') = mpc_surprise_2400_e_monthly_betahigh;
mpc_betahigh.expected('2400 quarterly-unemployed') = mpc_expect_2400_u_quarterly_betahigh;
mpc_betahigh.surprise('2400 quarterly-unemployed') = mpc_surprise_2400_u_quarterly_betahigh;
mpc_betahigh.expected('2400 monthly-unemployed') = mpc_expect_2400_u_monthly_betahigh;
mpc_betahigh.surprise('2400 monthly-unemployed') = mpc_surprise_2400_u_monthly_betahigh;
mpc_betahigh.Variables=scale_factor*mpc_betahigh.Variables;

mpc_pandemic_match500mpc_betahigh
mpc_pandemic_match500mpc
mpc_prepandemic

mpc_supplements = table();
mpc_supplements.surprise('one_month') = (mean_c_sim_pandemic_surprise(4) - mean_c_sim_pandemic_surprise_noFPUC(4)) / (mean_y_sim_pandemic_u(4) - mean_y_sim_pandemic_noFPUC(4));
mpc_supplements.surprise('3_month') = sum(mean_c_sim_pandemic_surprise(4:6) - mean_c_sim_pandemic_surprise_noFPUC(4:6)) / sum(mean_y_sim_pandemic_u(4:6) - mean_y_sim_pandemic_noFPUC(4:6));
mpc_supplements.surprise('6_month') = sum(mean_c_sim_pandemic_surprise(4:9) - mean_c_sim_pandemic_surprise_noFPUC(4:9)) / sum(mean_y_sim_pandemic_u(4:9) - mean_y_sim_pandemic_noFPUC(4:9));
mpc_supplements.surprise('9_month') = sum(mean_c_sim_pandemic_surprise(4:12) - mean_c_sim_pandemic_surprise_noFPUC(4:12)) / sum(mean_y_sim_pandemic_u(4:12) - mean_y_sim_pandemic_noFPUC(4:12));
mpc_supplements.surprise('full') = sum(mean_c_sim_pandemic_surprise(4:7) - mean_c_sim_pandemic_surprise_noFPUC(4:7)) / sum(mean_y_sim_pandemic_u(4:7) - mean_y_sim_pandemic_noFPUC(4:7));
mpc_supplements.surprise('full+3') = sum(mean_c_sim_pandemic_surprise(4:10) - mean_c_sim_pandemic_surprise_noFPUC(4:10)) / sum(mean_y_sim_pandemic_u(4:10) - mean_y_sim_pandemic_noFPUC(4:10));

%mpc_supplements.surprise('expire')=(mean_c_sim_pandemic_surprise_extramonth(8)-mean_c_sim_pandemic_surprise(8))/(mean_y_sim_pandemic_u_extramonth(8)-mean_y_sim_pandemic_u(8));
mpc_supplements.expect('one_month') = (mean_c_sim_pandemic_expect(4) - mean_c_sim_pandemic_expect_noFPUC(4)) / (mean_y_sim_pandemic_u(4) - mean_y_sim_pandemic_noFPUC(4));
mpc_supplements.expect('3_month') = sum(mean_c_sim_pandemic_expect(4:6) - mean_c_sim_pandemic_expect_noFPUC(4:6)) / sum(mean_y_sim_pandemic_u(4:6) - mean_y_sim_pandemic_noFPUC(4:6));
mpc_supplements.expect('6_month') = sum(mean_c_sim_pandemic_expect(4:9) - mean_c_sim_pandemic_expect_noFPUC(4:9)) / sum(mean_y_sim_pandemic_u(4:9) - mean_y_sim_pandemic_noFPUC(4:9));
mpc_supplements.expect('9_month') = sum(mean_c_sim_pandemic_expect(4:12) - mean_c_sim_pandemic_expect_noFPUC(4:12)) / sum(mean_y_sim_pandemic_u(4:12) - mean_y_sim_pandemic_noFPUC(4:12));
mpc_supplements.expect('full') = sum(mean_c_sim_pandemic_expect(4:7) - mean_c_sim_pandemic_expect_noFPUC(4:7)) / sum(mean_y_sim_pandemic_u(4:7) - mean_y_sim_pandemic_noFPUC(4:7));
mpc_supplements.expect('full+3') = sum(mean_c_sim_pandemic_expect(4:10) - mean_c_sim_pandemic_expect_noFPUC(4:10)) / sum(mean_y_sim_pandemic_u(4:10) - mean_y_sim_pandemic_noFPUC(4:10))
mpc_supplements.Variables=scale_factor*mpc_supplements.Variables;


mpc_supplements_expiration=mpc_supplements;
save('mpc_supplements_expiration','mpc_supplements_expiration');

mpc_supplements_pandemic_match500mpc
mpc_supplements_prepandemic

table_mpc_for_paper = table();
table_mpc_for_paper.one_month_mpc('$2400 for 4 months: Unemployed Households, target waiting') = mpc_supplements_nodiscountfactorshock_targetwaiting.surprise('one_month');
table_mpc_for_paper.one_month_mpc('$2400 one time: Unemployed Households, target waiting') = mpc.surprise('2400 monthly-unemployed');
table_mpc_for_paper.one_month_mpc('$500 one time: Unemployed Households, target waiting') = mpc.surprise('500 monthly-unemployed');
table_mpc_for_paper.one_month_mpc('$500 one time: All Households, target waiting') = mpc.surprise('500 monthly');

table_mpc_for_paper.one_month_mpc('$2400 for 4 months: Unemployed Households, target 500 mpc') = mpc_supplements_nodiscountfactorshock_target500mpc.surprise('one_month');
table_mpc_for_paper.one_month_mpc('$2400 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc.surprise('2400 monthly-unemployed');
table_mpc_for_paper.one_month_mpc('$500 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc.surprise('500 monthly-unemployed');
table_mpc_for_paper.one_month_mpc('$500 one time: All Households, target 500 mpc') = mpc_pandemic_match500mpc.surprise('500 monthly')


table_mpc_for_paper_wbetashock = table();
table_mpc_for_paper_wbetashock.one_month_mpc('$2400 for 4 months: Unemployed Households, target waiting') = mpc_supplements.surprise('one_month');
table_mpc_for_paper_wbetashock.one_month_mpc('$2400 one time: Unemployed Households, target waiting') = mpc_betahigh.surprise('2400 monthly-unemployed');
table_mpc_for_paper_wbetashock.one_month_mpc('$500 one time: Unemployed Households, target waiting') = mpc_betahigh.surprise('500 monthly-unemployed');
table_mpc_for_paper_wbetashock.one_month_mpc('$500 one time: All Households, target waiting') = mpc_betahigh.surprise('500 monthly');

table_mpc_for_paper_wbetashock.one_month_mpc('$2400 for 4 months: Unemployed Households, target 500 mpc') = mpc_supplements_pandemic_match500mpc.surprise('one_month');
table_mpc_for_paper_wbetashock.one_month_mpc('$2400 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('2400 monthly-unemployed');
table_mpc_for_paper_wbetashock.one_month_mpc('$500 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('500 monthly-unemployed');
table_mpc_for_paper_wbetashock.one_month_mpc('$500 one time: All Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('500 monthly')

table_both=table();
table_both=table_mpc_for_paper;
table_both.one_month_mpc_wbetashock('$2400 for 4 months: Unemployed Households, target waiting') = mpc_supplements.surprise('one_month');
table_both.one_month_mpc_wbetashock('$2400 one time: Unemployed Households, target waiting') = mpc_betahigh.surprise('2400 monthly-unemployed');
table_both.one_month_mpc_wbetashock('$500 one time: Unemployed Households, target waiting') = mpc_betahigh.surprise('500 monthly-unemployed');
table_both.one_month_mpc_wbetashock('$500 one time: All Households, target waiting') = mpc_betahigh.surprise('500 monthly');

table_both.one_month_mpc_wbetashock('$2400 for 4 months: Unemployed Households, target 500 mpc') = mpc_supplements_pandemic_match500mpc.surprise('one_month');
table_both.one_month_mpc_wbetashock('$2400 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('2400 monthly-unemployed');
table_both.one_month_mpc_wbetashock('$500 one time: Unemployed Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('500 monthly-unemployed');
table_both.one_month_mpc_wbetashock('$500 one time: All Households, target 500 mpc') = mpc_pandemic_match500mpc_betahigh.surprise('500 monthly')


table_mpc_for_paper = table();
table_mpc_for_paper.Transfer('1')="$2400 Persistent+Regular Benefits";
table_mpc_for_paper.Who_Gets_Transfer('1')="Unemployed w/ No Benefits";
table_mpc_for_paper.Discount_Factor_Calibration('1')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('1')="Pandemic";
table_mpc_for_paper.MPC_horizon('1')="Month";
table_mpc_for_paper.MPC_model=mpc.surprise('waiting');
table_mpc_for_paper.MPC_data="0.42";

table_mpc_for_paper.Transfer('2')="$2400 Persistent";
table_mpc_for_paper.Who_Gets_Transfer('2')="Unemployed w/ Regular Benefits";
table_mpc_for_paper.Discount_Factor_Calibration('2')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('2')="Pandemic";
table_mpc_for_paper.MPC_horizon('2')="Month";
table_mpc_for_paper.MPC_model('2')=mpc_supplements.surprise('one_month');
table_mpc_for_paper.MPC_data('2')="-"

table_mpc_for_paper.Transfer('3')="$2400 Persistent";
table_mpc_for_paper.Who_Gets_Transfer('3')="Unemployed w/ Regular Benefits";
table_mpc_for_paper.Discount_Factor_Calibration('3')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('3')="Normal Times";
table_mpc_for_paper.MPC_horizon('3')="Month";
table_mpc_for_paper.MPC_model('3')= mpc_supplements_nodiscountfactorshock_targetwaiting.surprise('one_month');
table_mpc_for_paper.MPC_data('3')="-";

table_mpc_for_paper.Transfer('4')="$2400 One Time";
table_mpc_for_paper.Who_Gets_Transfer('4')="Unemployed w/ Regular Benefits";
table_mpc_for_paper.Discount_Factor_Calibration('4')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('4')="Normal Times";
table_mpc_for_paper.MPC_horizon('4')="Month";
table_mpc_for_paper.MPC_model('4')= mpc.surprise('2400 monthly-unemployed');
table_mpc_for_paper.MPC_data('4')="-";

table_mpc_for_paper.Transfer('5')="$500 One Time";
table_mpc_for_paper.Who_Gets_Transfer('5')="Unemployed w/ Regular Benefits";
table_mpc_for_paper.Discount_Factor_Calibration('5')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('5')="Normal Times";
table_mpc_for_paper.MPC_horizon('5')="Month";
table_mpc_for_paper.MPC_model('5')= mpc.surprise('500 monthly-unemployed');
table_mpc_for_paper.MPC_data('5')="-"

table_mpc_for_paper.Transfer('6')="$500 One Time";
table_mpc_for_paper.Who_Gets_Transfer('6')="Everyone";
table_mpc_for_paper.Discount_Factor_Calibration('6')="Target Waiting Design MPC";
table_mpc_for_paper.Economic_Environment('6')="Normal Times";
table_mpc_for_paper.MPC_horizon('6')="Month";
table_mpc_for_paper.MPC_model('6')= mpc.surprise('500 monthly');
table_mpc_for_paper.MPC_data('6')="-"

table_mpc_for_paper.Transfer('7')="$500 One Time";
table_mpc_for_paper.Who_Gets_Transfer('7')="Everyone";
table_mpc_for_paper.Discount_Factor_Calibration('7')="Target $500 MPC";
table_mpc_for_paper.Economic_Environment('7')="Normal Times";
table_mpc_for_paper.MPC_horizon('7')="Month";
table_mpc_for_paper.MPC_model('7')= mpc_pandemic_match500mpc.surprise('500 monthly');
table_mpc_for_paper.MPC_data('7')="-"

table_mpc_for_paper.Transfer('8')="$500 One Time";
table_mpc_for_paper.Who_Gets_Transfer('8')="Everyone";
table_mpc_for_paper.Discount_Factor_Calibration('8')="Target $500 MPC";
table_mpc_for_paper.Economic_Environment('8')="Normal Times";
table_mpc_for_paper.MPC_horizon('8')="Quarter";
table_mpc_for_paper.MPC_model('8')= mpc_pandemic_match500mpc.surprise('500 quarterly');
table_mpc_for_paper.MPC_data('8')="0.25"

writetable(table_mpc_for_paper,fullfile(release_path_paper,'table_mpc_for_paper.csv'),'WriteRowNames',true);


table_mpc_for_paper_alt = table();
table_mpc_for_paper_alt.one_month_mpc('$2400 for 4 months: Unemployed Households') = mpc_supplements.surprise('one_month');
table_mpc_for_paper_alt.one_month_mpc('$2400 for 4 months: Unemployed Households, prepandemic beta') = mpc_supplements_prepandemic.surprise('one_month');
table_mpc_for_paper_alt.one_month_mpc('$2400 one time: Unemployed Households, prepandemic beta') = mpc_prepandemic.surprise('2400 monthly-unemployed');
table_mpc_for_paper_alt.one_month_mpc('$500 one time: Unemployed Households, prepandemic beta') = mpc_prepandemic.surprise('500 monthly-unemployed');
table_mpc_for_paper_alt.one_month_mpc('$500 one time: All Households, prepandemic beta') = mpc_prepandemic.surprise('500 monthly')

mean_c_sim_e_dollars_level = mean_c_sim_e/mean(mean_c_sim_e(3)) * mean(total_spend_e(3));
figure
plot(1:14, mean_c_sim_e_dollars_level(1:14), 1:14, total_spend_e(1:14), '--', 'LineWidth', 2)

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


% Plot and table for search_asset_effects 
figure
hold on
plot(1:11, mean_search_sim_pandemic_surprise(1:11), '-s', 'Color', qual_green, 'MarkerFaceColor', qual_green, 'LineWidth', 2)
plot(1:11, mean_search_sim_pandemic_surprise_noFPUC(1:11), '-+', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
legend('With supplement','Without supplement', 'Location','SouthEast')
xticks([4 5 6 7 8 9 10 11])
xticklabels(label_months_apr20_nov20)
set(gca, 'Layer', 'top');
fig_paper_A27 = gcf;
saveas(fig_paper_A27, fullfile(release_path_paper, 'search_asset_effects.png'))
%saveas(fig_paper_A27, fullfile(release_path_slides, 'search_asset_effects.png'))


stat_for_text_liquidity_share=(mean_search_sim_pandemic_surprise_noFPUC(1:11)-mean_search_sim_pandemic_surprise(1:11))
table_stat_for_text_liquidity_share = table(stat_for_text_liquidity_share);
writetable(table_stat_for_text_liquidity_share, fullfile(release_path_paper,'table_stat_for_text_liquidity_share.csv'),'WriteRowNames',true);

a_increase_vs_y_bestfitmodel=mean_a_sim_pandemic_surprise(8)/mean(w);
save('a_increase_vs_y_bestfitmodel','a_increase_vs_y_bestfitmodel');

final_a = mean_a_sim_pandemic_surprise(11);

% Run elasticity, distortions, and aggregates
% Arguments newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, t_start, t_end, include_self_employed
newjob_exit_rate_FPUC = mean_search_sim_pandemic_surprise(4:18)';
newjob_exit_rate_onlyasseteffect = mean_search_sim_pandemic_surprise_onlyasseteffect(4:18)';
newjob_exit_rate_noasseteffect = mean_search_sim_pandemic_surprise_noasseteffect(4:18)';
newjob_exit_rate_no_FPUC = mean_search_sim_pandemic_surprise_noFPUC(4:18)';
newjob_exit_rate_FPUC(end:1000) = newjob_exit_rate_FPUC(end);
newjob_exit_rate_onlyasseteffect(end:1000) = newjob_exit_rate_onlyasseteffect(end);
newjob_exit_rate_noasseteffect(end:1000) = newjob_exit_rate_noasseteffect(end);
newjob_exit_rate_no_FPUC(end:1000) = newjob_exit_rate_no_FPUC(end);


newjob_exit_rate_data=monthly_search_data';
newjob_exit_rate_data(end:1000,:)=monthly_search_data(end);
newjob_exit_rate_inter_time_series_based_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_inter_time_series_based_no_FPUC(1:4)=newjob_exit_rate_inter_time_series_based_no_FPUC(1:4)+inter_time_series_expiration;
newjob_exit_rate_model_inter_time_series_based_no_FPUC=newjob_exit_rate_FPUC;
newjob_exit_rate_model_inter_time_series_based_no_FPUC(1:4)=newjob_exit_rate_model_inter_time_series_based_no_FPUC(1:4)+(newjob_exit_rate_FPUC(5)-newjob_exit_rate_FPUC(4));
newjob_exit_rate_cross_section_based_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_no_FPUC(1:4)=newjob_exit_rate_cross_section_based_no_FPUC(1:4)+cross_section_expiration;
newjob_exit_rate_cross_section_based_logit_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_logit_no_FPUC(1:4)=newjob_exit_rate_cross_section_based_logit_no_FPUC(1:4)+cross_section_expiration_logit;

newjob_exit_rate_FPUC_bywage = mean_search_sim_pandemic_surprise_bywage(:,4:18)';
newjob_exit_rate_no_FPUC_bywage = mean_search_sim_pandemic_surprise_noFPUC_bywage(:,4:18)';
newjob_exit_rate_FPUC_bywage(end:1000,:) = repmat(newjob_exit_rate_FPUC_bywage(end,:),1000-length(newjob_exit_rate_FPUC_bywage)+1,1);
newjob_exit_rate_no_FPUC_bywage(end:1000,:) = repmat(newjob_exit_rate_no_FPUC_bywage(end,:),1000-length(newjob_exit_rate_no_FPUC_bywage)+1,1);

recall_probs = recall_probs_pandemic_actual';
recall_probs(end:1000) = recall_probs(end);
mean_c_sim_pandemic_surprise_overall_FPUC = NaN;
mean_c_sim_pandemic_surprise_overall_noFPUC = NaN;
mean_c_sim_e_overall = NaN;
benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
perc_change_benefits_data = benefit_change_data.non_sym_per_change(1);
date_sim_start = datetime(2020, 4, 1);
t_start = 2;
% Surprise period minus one should be period 4 (April is 1, May is 2, June is 3, July is 4, August is 5)
t_end = 5;
include_self_employed = 0;
[elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC total_hazard] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);



%employment effects arising from assets
[elasticity_onlyasseteffect employment_distortion_onlyasseteffect total_diff_employment_onlyasseteffect share_unemployment_reduced_onlyasseteffect employment_FPUC_onlyasseteffect employment_noFPUC_onlyasseteffect monthly_spend_pce_onlyasseteffect monthly_spend_no_FPUC_onlyasseteffect] = elasticity_distortions_and_aggregates(newjob_exit_rate_onlyasseteffect, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

%employment effects from an interrupted time-series based approach in model
[elasticity_inter_time_series_based employment_distortion_inter_time_series_based total_diff_employment_inter_time_series_based share_unemployment_reduced_inter_time_series_based employment_FPUC_inter_time_series_based employment_noFPUC_inter_time_series_based monthly_spend_pce_inter_time_series_based monthly_spend_no_FPUC_inter_time_series_based] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_inter_time_series_based_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);



%Collect relevant elasticity and distortion results
elasticity_and_distortions_values = [elasticity, employment_distortion, total_diff_employment, share_unemployment_reduced];
elasticity_and_distortions_values_onlyasseteffect = [elasticity_onlyasseteffect, employment_distortion_onlyasseteffect, total_diff_employment_onlyasseteffect, share_unemployment_reduced_onlyasseteffect];
elasticity_and_distortions_values_inter_time_series_based = [elasticity_inter_time_series_based, employment_distortion_inter_time_series_based, total_diff_employment_inter_time_series_based, share_unemployment_reduced_inter_time_series_based];



display('Liquidity shares of distortions $600')
elasticity_onlyasseteffect/elasticity
employment_distortion_onlyasseteffect/employment_distortion
elasticity_inter_time_series_based/elasticity
employment_distortion_inter_time_series_based/employment_distortion
(mean_search_sim_pandemic_surprise(8)-mean_search_sim_pandemic_surprise(7))/(mean_search_sim_pandemic_surprise_noasseteffect(8)-mean_search_sim_pandemic_surprise_noasseteffect(7))

% Save the job search series and the spend series
newjob_exit_rate_data_2020=newjob_exit_rate_data;
newjob_exit_rate_FPUC_2020 = newjob_exit_rate_FPUC;
newjob_exit_rate_no_FPUC_2020 = newjob_exit_rate_no_FPUC;
newjob_exit_rate_onlyasseteffect_2020=newjob_exit_rate_onlyasseteffect;
newjob_exit_rate_model_inter_time_series_based_no_FPUC_2020=newjob_exit_rate_model_inter_time_series_based_no_FPUC;
newjob_exit_rate_inter_time_series_based_no_FPUC_2020=newjob_exit_rate_inter_time_series_based_no_FPUC;
newjob_exit_rate_cross_section_based_no_FPUC_2020=newjob_exit_rate_cross_section_based_no_FPUC;
newjob_exit_rate_cross_section_based_logit_no_FPUC_2020=newjob_exit_rate_cross_section_based_logit_no_FPUC;
newjob_exit_rate_FPUC_bywage_2020 = newjob_exit_rate_FPUC_bywage;
newjob_exit_rate_no_FPUC_bywage_2020 = newjob_exit_rate_no_FPUC_bywage;
recall_probs_2020 = recall_probs;
mean_c_sim_e_2020 = mean_c_sim_e;
mean_c_sim_pandemic_surprise_2020 = mean_c_sim_pandemic_surprise;
mean_c_sim_pandemic_surprise_noFPUC_2020 = mean_c_sim_pandemic_surprise_noFPUC;
save('inf_horizon_het_results_newjob_exit_rate.mat','newjob_exit_rate_model_inter_time_series_based_no_FPUC_2020','newjob_exit_rate_data_2020','newjob_exit_rate_onlyasseteffect_2020','newjob_exit_rate_cross_section_based_no_FPUC_2020','newjob_exit_rate_cross_section_based_logit_no_FPUC_2020','newjob_exit_rate_inter_time_series_based_no_FPUC_2020', 'newjob_exit_rate_FPUC_2020', 'newjob_exit_rate_no_FPUC_2020','newjob_exit_rate_FPUC_bywage_2020', 'newjob_exit_rate_no_FPUC_bywage_2020', 'recall_probs_2020', 'mean_c_sim_e_2020', 'mean_c_sim_pandemic_surprise_2020', 'mean_c_sim_pandemic_surprise_noFPUC_2020');
vars = {'newjob_exit_rate_FPUC_2020','newjob_exit_rate_model_inter_time_series_based_no_FPUC_2020','newjob_exit_rate_onlyasseteffect_2020','newjob_exit_rate_data_2020','newjob_exit_rate_cross_section_based_no_FPUC_2020','newjob_exit_rate_cross_section_based_logit_no_FPUC_2020','newjob_exit_rate_inter_time_series_based_no_FPUC_2020', 'newjob_exit_rate_no_FPUC_2020','newjob_exit_rate_FPUC_bywage_2020', 'newjob_exit_rate_no_FPUC_bywage_2020', 'recall_probs_2020', 'mean_c_sim_e_2020', 'mean_c_sim_pandemic_surprise_2020', 'mean_c_sim_pandemic_surprise_noFPUC_2020'};
clear(vars{:});



distortion_surprise = mean_search_sim_pandemic_surprise_noFPUC - mean_search_sim_pandemic_surprise;
distortion_expect = mean_search_sim_pandemic_expect_noFPUC - mean_search_sim_pandemic_expect;
distortion_surprise = distortion_surprise / distortion_surprise(7);
distortion_expect = distortion_expect / distortion_expect(7);
distortion_expect2 = distortion_expect / (mean_search_sim_pandemic_expect_noFPUC(8)-mean_search_sim_pandemic_expect_noFPUC(7));

%import data diff-in-diff:
data_DiD = readtable(jobfind_input_directory, 'Sheet', fig_a12b_expiry);
%compute 95% CIs
data_DiD.high_ci = data_DiD.estimate + norminv(0.95) .* data_DiD.std_error;
data_DiD.low_ci = data_DiD.estimate + norminv(0.05) .* data_DiD.std_error;
%re-order variables 
data_DiD = movevars(data_DiD, 'high_ci', 'After', 'estimate');
data_DiD = movevars(data_DiD, 'low_ci', 'After', 'high_ci');

data_DiD = table2array(data_DiD(:,2:end-1));

data_DiD = data_DiD - mean(data_DiD(9:end))
data_DiD = data_DiD / data_DiD(8);

weekly_index_preperiod = 5:.25:6.75;
weekly_index_full = [weekly_index_preperiod, [7:.2:7.8], [8:.25:8.75]];


figure
tiledlayout(1,2)
nexttile
hold on
p = patch([4 4 7 7], [0.001 .26 .26 0.001], [0.92 0.92 0.92], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
title('Job-Finding Rate','FontWeight','normal')
plot(1:11, mean_search_sim_pandemic_expect_match500MPC(1:11), '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(1:11, mean_search_sim_pandemic_surprise_match500MPC(1:11), '-v', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 3)
plot(1:11, mean_search_sim_pandemic_surprise(1:11), '-s', 'Color', qual_green, 'MarkerFaceColor', qual_green, 'LineWidth', 2)
plot(4:11, monthly_search_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
xlim([4 11])
ylim([0.045 0.115])
xticks([4 5 6 7 8 9 10 11])
xticklabels(label_months_apr20_nov20)
set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
set(gca,'fontsize', 12);
nexttile
p = patch([4 4 7 7], [-21 30 30 -21], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
title('Spending (U vs. E % Change from Jan 20)','FontWeight','normal')
plot(1:11, mean_c_sim_pandemic_expect_dollars_match500MPC(1:11)/total_spend_u(1)*100, '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(1:11, mean_c_sim_pandemic_surprise_dollars_match500MPC(1:11)/total_spend_u(1)*100, '-v', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(1:11, mean_c_sim_pandemic_surprise_dollars(1:11)/total_spend_u(1)*100, '-s', 'Color', qual_green, 'MarkerFaceColor', qual_green, 'LineWidth', 2)
plot(1:11, spend_data(1:11), '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)

yticks([-15 -10 -5 0 5 10 15])
yticklabels({'-15%','-10%','-5%', '0%','5%','15%', '20%'})
xlim([1 11])
xticks([1 2 3 4 5 6 7 8 9 10 11])
ylim([-10 15])
xticklabels(label_months_jan20_nov20)
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
lgd=legend('Pandemic Search Costs', 'Pandemic Search Costs + Myopic Expectations', 'Pandemic Search Costs + Myopic Expectations + High Impatience','Data', 'FontSize', 12);
title(lgd,'Changes From Standard Model:','FontSize',12,'FontWeight','normal')
lgd.Layout.Tile = 'South';
set(gcf, 'PaperPosition', [0 0 13 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [13 5]); %Keep the same paper size
fig_paper_9b = gcf;
saveas(fig_paper_9b, fullfile(release_path_paper, 'spend_and_search_expiration.png'))



figure
tiledlayout(1,2)
nexttile
hold on
p = patch([4 4 7 7], [0.001 .3 .3 0.001], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
title('Job-Finding Rate','FontWeight','normal')
plot(1:11, mean_search_sim_prepandemic_expect(1:11), '-+', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
plot(4:11, monthly_search_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
xlim([4 11])
ylim([0.00 0.3])
xticks([4 5 6 7 8 9 10 11])
xticklabels(label_months_apr20_nov20)
set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
set(gca, 'fontsize', 12);
set(gca, 'Layer', 'top');
nexttile
p = patch([4 4 7 7], [-30 30 30 -30], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
title('Spending (U vs. E % Change from Jan 20)','FontWeight','normal')
plot(1:11, mean_c_sim_prepandemic_expect_dollars(1:11)/total_spend_u(1)*100, '-+', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
plot(1:11, spend_data(1:11), '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
yticks([-15 -10 -5 0 5 10 15])
yticklabels({'-15%','-10%','-5%', '0%','5%','15%', '20%'})
ylim([-10 15])
xlim([1 11])
xticks([1 2 3 4 5 6 7 8 9 10 11])
xticklabels(label_months_jan20_nov20)
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
lgd=legend('Pre-pandemic Search Costs + Correct Expectations + Normal Impatience','Data', 'FontSize', 12);
lgd.Layout.Tile = 'South';
set(gcf, 'PaperPosition', [0 0 13 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [13 5]); %Keep the same paper size
fig_paper_9a = gcf;
saveas(fig_paper_9a, fullfile(release_path_paper, 'spend_and_search_prepandemic_expiration.png'))











expiry = readtable(jobfind_input_directory, 'Sheet', expiry_sample);
idx = datenum(expiry.week_start_date) >= datenum('2020-04-01');
expiry = expiry(idx, :);
%idx=string(expiry.type)=='All data';
% Convert expiry variables to a monthly level
% Apply week_to_month_exit.m to grouped, sorted monthly data
% Sort by type, cut, and week start date
expiry = sortrows(expiry, {'type', 'cut', 'week_start_date'});
expiry.month = dateshift(datetime(expiry.week_start_date), 'start', 'month');

% For the exit variables we want the week to month exit at a monthly level
expiry_month_exit = varfun(@week_to_month_exit, expiry, 'InputVariables', {'exit_ui_rate', 'exit_to_recall', 'exit_not_to_recall'}, 'GroupingVariables', {'month', 'cut', 'type'});
expiry_month_exit = renamevars(expiry_month_exit, ["week_to_month_exit_exit_ui_rate", "week_to_month_exit_exit_to_recall", "week_to_month_exit_exit_not_to_recall"], ["exit_ui_rate", "exit_to_recall", "exit_not_to_recall"]);
% For the per_change variable we want the average per_change at a monthly level
expiry_per_change = varfun(@mean, expiry, 'InputVariables', {'per_change'}, 'GroupingVariables', {'month', 'cut', 'type'});
expiry_per_change = renamevars(expiry_per_change, ["mean_per_change"], ["per_change"]);
% Expiry will be combined version of these
expiry = innerjoin(expiry_month_exit, expiry_per_change);

idx = (string(expiry.type) == 'By rep rate quintile');
expiry_by_wage_quintiles = expiry(idx, :);
expiry_by_wage_quintiles.cut = str2double(expiry_by_wage_quintiles.cut);
expiry_by_wage_quintiles = sortrows(expiry_by_wage_quintiles, 'cut');

for i = 1:5
    exit_not_to_recall_by_cut(:, i) = expiry_by_wage_quintiles.exit_not_to_recall(expiry_by_wage_quintiles.cut == i);
end

change_rep_rate_expiry_quintiles = grpstats(expiry_by_wage_quintiles, 'cut', 'mean', 'DataVars', {'per_change'});
change_rep_rate_expiry_quintiles = change_rep_rate_expiry_quintiles.mean_per_change;
change_rep_rate_expiry_quintiles = sort(change_rep_rate_expiry_quintiles, 'Descend');

index=0;
for i=1:5
    for t=1:10
        index=index+1;
        panel(index,1)=i;
        panel(index,2)=t;
        panel(index,3)=1-(1-newjob_exit_rate_FPUC_bywage(t,i))^(1/4.25);
        panel(index,4)=change_rep_rate_expiry_quintiles(i);
    end
end

binscatter = readtable(jobfind_input_directory, 'Sheet', 'binscatter_expiry_new_job_df');


data_change_weekly = binscatter.estimate;
data_change_monthly = 1 - (1 - data_change_weekly).^4;

data_per_change = binscatter.per_change;

surprise_period = 8;
% Change window of 8 weeks to 2 months
mean_search_sim_pandemic_surprise_bywage_weekly = 1 - (1 - mean_search_sim_pandemic_surprise_bywage).^(1/4.25);
diff_exit(1) = mean(mean_search_sim_pandemic_surprise_bywage_weekly(1, surprise_period:surprise_period + 2)) - mean(mean_search_sim_pandemic_surprise_bywage_weekly(1, surprise_period - 3:surprise_period - 1));
diff_exit(2) = mean(mean_search_sim_pandemic_surprise_bywage_weekly(2, surprise_period:surprise_period + 2)) - mean(mean_search_sim_pandemic_surprise_bywage_weekly(2, surprise_period - 3:surprise_period - 1));
diff_exit(3) = mean(mean_search_sim_pandemic_surprise_bywage_weekly(3, surprise_period:surprise_period + 2)) - mean(mean_search_sim_pandemic_surprise_bywage_weekly(3, surprise_period - 3:surprise_period - 1));
diff_exit(4) = mean(mean_search_sim_pandemic_surprise_bywage_weekly(4, surprise_period:surprise_period + 2)) - mean(mean_search_sim_pandemic_surprise_bywage_weekly(4, surprise_period - 3:surprise_period - 1));
diff_exit(5) = mean(mean_search_sim_pandemic_surprise_bywage_weekly(5, surprise_period:surprise_period + 2)) - mean(mean_search_sim_pandemic_surprise_bywage_weekly(5, surprise_period - 3:surprise_period - 1));

shift = (mean(data_change_weekly) - mean(diff_exit));
diff_exit = diff_exit + shift;

reg_input_col4 = mean_search_sim_pandemic_surprise_bywage_weekly';
reg_input_col3 = zeros(size(reg_input_col4));
reg_input_col3(8:end, :) = 1;
reg_input_col2 = zeros(size(reg_input_col4));
reg_input_col1 = zeros(size(reg_input_col4));

for i = 1:5
    reg_input_col2(:, i) = change_rep_rate_expiry_quintiles(i);
    reg_input_col1(:, i) = reg_input_col2(:, i) .* reg_input_col3(:, i);
end

reg_input_col1 = reg_input_col1(5:10, :);
reg_input_col2 = reg_input_col2(5:10, :);
reg_input_col3 = reg_input_col3(5:10, :);
reg_input_col4 = reg_input_col4(5:10, :);

reg_input_col1 = reshape(reg_input_col1, 5 * 6, 1);
reg_input_col2 = reshape(reg_input_col2, 5 * 6, 1);
reg_input_col3 = reshape(reg_input_col3, 5 * 6, 1);
reg_input_col4 = reshape(reg_input_col4, 5 * 6, 1);

reg_input = table();
reg_input.postXperchange_rep_rate = reg_input_col1;
reg_input.perchange_rep_rate = reg_input_col2;
reg_input.post = reg_input_col3;
reg_input.exit_rate = reg_input_col4;

reg_newjob_exit = fitlm(reg_input, 'linear');

%per_change_regular=change_rep_rate_expiry_quintiles./(1-.5*change_rep_rate_expiry_quintiles)
change_rep_rate_expiry_quintiles_extended = [1.4; change_rep_rate_expiry_quintiles; .45];
figure
scatter(data_per_change * 100, data_change_weekly, 60, 'o', 'filled', 'MarkerFaceColor', [0 0.2695 0.4648], 'MarkerEdgeColor', [0 0.2695 0.4648])
hold on
scatter(-change_rep_rate_expiry_quintiles * 100, diff_exit, 60, 's', 'filled', 'MarkerFaceColor', [0.7305 0.8477 0.4609], 'MarkerEdgeColor', [0.7305 0.8477 0.4609])

hold on
yhat = change_rep_rate_expiry_quintiles_extended .* reg_newjob_exit.Coefficients.Estimate(2) + reg_newjob_exit.Coefficients.Estimate(4) + shift;
plot(-change_rep_rate_expiry_quintiles_extended * 100, yhat, 'LineWidth', 1.25, 'Color', [0.7305 0.8477 0.4609])
yhat_data = change_rep_rate_expiry_quintiles_extended * .0166 - .004;
plot(-change_rep_rate_expiry_quintiles_extended * 100, yhat_data, 'LineWidth', 1.25, 'Color', [0 0.2695 0.4648])

xlabel('Change in benefits')
ylabel('Change in average exit rate to new job')
set(0, 'DefaultAxesTitleFontWeight', 'normal');
% t = title('Change in average exit rate to new job', 'Units', 'normalized', 'Position', [0.2, 1.03, 0], 'FontSize', 14); % Set Title with correct Position
grid on
xticks([-140 -120 -100 -80 -60])
xlim([-140 -50])
xticklabels({'-140%','-120%', '-100%', '-80%', '-60%'})
yticks([0.004 0.008 0.012 0.016]);
pbaspect([1.7 1 1])
set(gcf, 'PaperPosition', [0 0 8 4.5]);
set(gcf, 'PaperSize', [8 4.5]);
set(gca, 'FontSize', 14)
set(gca, 'Layer', 'top');
dim = [.16 .43 .3 .3];
str = strcat('Slope:', num2str(-reg_newjob_exit.Coefficients.Estimate(2)));
%t=annotation('textbox',dim,'String',str,'FitBoxToText','on');
%t.LineStyle='none';
%t.FontSize=12;
legend('Data', 'Best fit model')
fig_paper_A20 = gcf
saveas(fig_paper_A20, fullfile(release_path_paper, 'expiration_scatter_model.png'));
%saveas(fig_paper_A20, fullfile(release_path_slides, 'expiration_scatter_model.png'));



elasticity_and_distortions_values_prepandemic
elasticity_and_distortions_values

a_init_bestfit=mean_a_sim_pandemic_surprise_bywage(:,10);
a_init_pandemic_surprise_bestfit_endofjuly=mean_a_sim_pandemic_surprise_bywage(:,8);
a_init_pandemic_surprise_noFPUC_bestfit_endofjuly=mean_a_sim_pandemic_surprise_noFPUC_bywage(:,8);
a_init_e_bestfit_endofjuly=mean_a_sim_e_bywage(:,8);
save('a_init_bestfit','a_init_bestfit','a_init_pandemic_surprise_bestfit_endofjuly','a_init_pandemic_surprise_noFPUC_bestfit_endofjuly','a_init_e_bestfit_endofjuly');

stat_for_text_share_spent=2400*mpc_supplements.surprise('one_month')
stat_for_text_share_spent_full=10200*mpc_supplements.surprise('full')
stat_for_text_share_spent_fullplus3=10200*mpc_supplements.surprise('full+3')
save('stats_for_text_model_miscellaneous.mat', 'stat_for_text_share_spent', 'stat_for_text_share_spent_full', 'stat_for_text_share_spent_fullplus3', '-append')

