display('Simulating Full Model Effects of $600')
clearvars -except -regexp fig_paper_*
tic

FPUCsize=.35;
FPUC_length=[2 3 4];
FPUC_mult=1/12:1/12:1;
FPUC_mult=FPUC_mult*.1;
for beta_normal_loop=1:2
for FPUC_mult_index=1:length(FPUC_mult)    
    
    load('jobfind_input_directory.mat');
    load('jobfind_input_sheets.mat');
    load('spending_input_directory.mat');
    load spending_input_sheets.mat
    load('bls_employment_input_directory.mat')
    load inter_time_series_input_directory.mat
    load hh_wage_groups.mat
    load release_paths.mat

    load bestfit_prepandemic.mat
    load bestfit_target_waiting_MPC.mat
    
    table_effects_summary = readtable(inter_time_series_input_directory);
    inter_time_series_expiration=1-(1+table_effects_summary.ts_exit(1))^4;
    cross_section_expiration=1-(1+table_effects_summary.cs_exit(1))^4;
    inter_time_series_onset=1-(1+table_effects_summary.ts_exit(2))^4;
    cross_section_onset=1-(1+table_effects_summary.cs_exit(2))^4;
    
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
    %}
    
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
    
    stat_for_text_change_checking_u_vs_e_stimulus_onetenth=(checking_u(7)-checking_u(3))-(checking_e(7)-checking_e(3))
    save('stats_for_text_model_miscellaneous.mat', 'stat_for_text_change_checking_u_vs_e_stimulus_onetenth', '-append')
    
    
    
    
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
    
    
    
        
    
    shock_FPUC_multsize = .35*FPUC_mult(FPUC_mult_index);
    
    
    k_prepandemic = pre_pandemic_fit_match500MPC(1);
    gamma_prepandemic = pre_pandemic_fit_match500MPC(2);
    c_param_prepandemic = 0;
    
    % Assign parameter values
    load discountfactors.mat
    if beta_normal_loop==1
        beta_normal=beta_target500MPC;
    else
        beta_norma=beta_targetwaiting;
    end
    beta_high = beta_oneperiodshock;

    load model_parameters.mat
    initial_a = initial_a - aprimemin;
    
    n_ben_profiles_allowed = 8; %This captures the surprise vs. expected expiration scenarios w/ wait or no delay (and 2 extra benefit profiles for liquidity decomposition)
    
    FPUC_expiration = shock_FPUC_multsize;
    LWAsize = 1300 * FPUC_expiration / (4.5 * 600);
    
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
            benefit_profile_pandemic(1, 3) = 1.19*h; %note the +b here is to match the actual income decline for waiting group in the data which is a little more gradual, not because they actually receive benefits for one month
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
    
    
            %benefit_profile_pandemic(1, :) = benefit_profile_pandemic(1, :) + 350 * FPUC_expiration / (4.5 * 600);
    
%             recall_probs_pandemic(1:13, 1) = 0.00;
%             recall_probs_regular = recall_probs_pandemic;

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
                %while (iter <= 5000) & (diffC > tol) %& (diffC_percent > tol_percent) %| diffV > tol)
    
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
    
                    %while iter<199
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
                    ave_change_in_C_percent = 100 * mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
                  
                    ave_change_in_C_percent_transfer = 100 * mean([mean(mean(mean(abs(c_pol_e_with_transfer(:,:) - c_pol_e_with_transfer_guess(:,:)) ./ c_pol_e_with_transfer_guess(:, :)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
                  
                    ave_change_in_C_percent = (ave_change_in_C_percent + ave_change_in_C_percent_transfer) ./ 2;
       
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
                    c_pol_u_pandemic_betahigh = c_pol_u_pandemic;
                end
    
            end
    
            % Begin simulations using policy functions
    
            A = Aprime;

            %Note that we don't necessarily need all parts of this simulation step to
            %be internal to the parameter search, keeping only the absolute necessary
            %parts internal to that loop should speed things up some
    
            %note also i might be able to speed up by feeding only the adjacent points
            %into the interp step
    
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
            c_pol_u = c_pol_u_betanormal;
            c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
    
            v_e = v_e_betanormal;
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
            
            u_dur_sim = zeros(numhh, numsim);
            a_sim(:, 1) = tmp_a;
            u_dur_sim(:, 1) = tmp_u;
            e_sim(:, 1) = tmp_e;
    
            c_sim_with_500 = c_sim;
            a_sim_with_500 = a_sim;
            a_sim_with_500(:, 1) = a_sim_with_500(:, 1) + shock_FPUC_multsize;
            
            if use_initial_a == 1
                a_sim_pandemic_expect = tmp_a(tmp_u > 0) + initial_a;
                a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0) + initial_a;
                a_sim_e = tmp_a(tmp_u == 0) + initial_a;
                a_sim_pandemic_surprise_onlyasseteffect=a_sim_pandemic_noFPUC+shock_FPUC_multsize;
            else
                a_sim_pandemic_expect = tmp_a(tmp_u > 0);
                a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0);
                a_sim_e = tmp_a(tmp_u == 0);
                a_sim_pandemic_surprise_onlyasseteffect=a_sim_pandemic_noFPUC+shock_FPUC_multsize;
            end
    
            num_unemployed_hh = length(a_sim_pandemic_expect);
            num_employed_hh = length(a_sim_e);
            c_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
            c_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_expect), 30);
            c_sim_pandemic_surprise_onlyasseteffect = zeros(length(a_sim_pandemic_expect), 30);
            c_sim_e = zeros(length(a_sim_e), 30);
    
            search_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
            search_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_expect), 30);
            search_sim_pandemic_surprise_onlyasseteffect = zeros(length(a_sim_pandemic_expect), 30);
    
            %this is looping over all hh after the burnin period, to get the average
            %MPC
            for t = 1:numsim
    
                for i = 1:numhh
    
                    if e_sim(i, t) == 1
                        c_sim(i, t) = interp1(A, c_pol_e(:), a_sim(i, t), 'linear');
                        a_sim(i, t + 1) = max(y + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);
    
                        c_sim_with_500(i, t) = interp1(A, c_pol_e(:), a_sim_with_500(i, t), 'linear');
                        a_sim_with_500(i, t + 1) = max(y + (1 + r) * a_sim_with_500(i, t) - c_sim_with_500(i, t), 0);
    
                    else
                        c_sim(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim(i, t), 'linear');
                        a_sim(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim(i, t) - c_sim(i, t), 0);
    
                        c_sim_with_500(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim_with_500(i, t), 'linear');
                        a_sim_with_500(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim_with_500(i, t) - c_sim_with_500(i, t), 0);

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
    
            mean_c_sim = mean(c_sim, 1);
    
            mean_c_sim_with_500 = mean(c_sim_with_500);
                
            for tt = 1:numsim
                mean_c_sim_e_with_500(tt) = mean(c_sim_with_500(e_sim(:, 1) == 1, tt));
                mean_c_sim_e_without_500(tt) = mean(c_sim(e_sim(:, 1) == 1, tt));
                
            end
    
            mpc_500_e_by_t = (mean_c_sim_e_with_500 - mean_c_sim_e_without_500) / (shock_FPUC_multsize);
    
            mpc_500_by_t = (mean_c_sim_with_500 - mean_c_sim) / (shock_FPUC_multsize);
    
            for t = 1:numsim
                mpc_500_cum_dynamic(t) = sum(mpc_500_by_t(1:t));
            end
    
            %mpc_500_cum_dynamic(3)
            if surprise == 0
                mpc_expect_500_bywage(iy, :) = mpc_500_cum_dynamic(:);
            end
    
            
    
            if FPUC_mult_index>1
                mpc_lastdollars_by_t = (mean_c_sim_with_500 - mean_c_sim_smaller_bywage(iy, :)) / (.1*FPUCsize*1/12);
            else
                mpc_lastdollars_by_t=(mean_c_sim_with_500 - mean_c_sim) / (.1*FPUCsize*1/12);
            end
            for t = 1:numsim
                mpc_lastdollars_cum_dynamic(t) = sum(mpc_lastdollars_by_t(1:t));
            end
            if surprise == 0
                mpc_expect_lastdollars_bywage(iy, :) = mpc_lastdollars_cum_dynamic(:);
            end
    
            
            mean_c_sim_smaller_bywage(iy, :)=mean_c_sim_with_500;
    
    
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
                        a_sim_pandemic_surprise_onlyasseteffect(i,t)=a_sim_pandemic_surprise_onlyasseteffect(i,t);
                    end
                    %LWA
                    if length_u == 6
                        a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + LWAsize;
                        a_sim_pandemic_surprise_onlyasseteffect(i, t) = a_sim_pandemic_surprise_onlyasseteffect(i, t) + LWAsize;
                    end
    
                    %Jan EIP
                    if length_u == 10
                        a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                        a_sim_pandemic_surprise_onlyasseteffect(i, t) = a_sim_pandemic_surprise_onlyasseteffect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                    end
                        
                    c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect(i, t), 'linear');
                    a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);
    
                    %Note this will vary with params, but can just save it accordingly
                    %when taking means later
                    c_sim_pandemic_noFPUC(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 5), a_sim_pandemic_noFPUC(i, t), 'linear');
                    a_sim_pandemic_noFPUC(i, t + 1) = max(benefit_profile_pandemic(length_u, 5) + (1 + r) * a_sim_pandemic_noFPUC(i, t) - c_sim_pandemic_noFPUC(i, t), 0);
                    
                    c_sim_pandemic_surprise_onlyasseteffect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 5), a_sim_pandemic_surprise_onlyasseteffect(i, t), 'linear');
                    a_sim_pandemic_surprise_onlyasseteffect(i, t + 1) = max(benefit_profile_pandemic(length_u, 5) + (1 + r) * a_sim_pandemic_surprise_onlyasseteffect(i, t) - c_sim_pandemic_surprise_onlyasseteffect(i, t), 0);
                    
        
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

                    diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_expect(i, t + 1), 'linear');
                    search_sim_pandemic_expect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                    if imag(search_sim_pandemic_expect(i, t)) ~= 0
                        search_sim_pandemic_expect(i, t) = 0;
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
                mean_c_sim_e = mean(c_sim_e, 1);
            else
                mean_search_sim_pandemic_expect = mean(search_sim_pandemic_expect, 1);
                mean_search_sim_pandemic_expect_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
                mean_search_sim_pandemic_expect_onlyasseteffect = mean(search_sim_pandemic_surprise_onlyasseteffect, 1);
            end
    
        end
    
        mean_c_sim_e_bywage(iy, :) = mean_c_sim_e;
    
        %paste on initial Jan-March 3 months of employment   
        
        mean_y_sim_e_bywage(iy,:)=y*ones(16,1);
        mean_y_sim_e_bywage(iy,4)=mean_y_sim_e_bywage(iy,4);
    
        mean_search_sim_pandemic_expect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect(1:numsim - 3)];
        mean_search_sim_pandemic_expect_noFPUC_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_noFPUC(1:numsim - 3)];
        mean_search_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect_onlyasseteffect(1:numsim - 3)];
    end
    
    mpc_expect_500_6month = mean(mpc_expect_500_bywage(:, 6));
    mpc_expect_500_quarterly = mean(mpc_expect_500_bywage(:, 3));
    mpc_expect_500_monthly = mean(mpc_expect_500_bywage(:, 1));
    
    mpc_expect_lastdollars_6month = mean(mpc_expect_lastdollars_bywage(:, 6));
    mpc_expect_lastdollars_quarterly = mean(mpc_expect_lastdollars_bywage(:, 3));
    mpc_expect_lastdollars_monthly = mean(mpc_expect_lastdollars_bywage(:, 1));
    

    mean_y_sim_e = mean(mean_y_sim_e_bywage, 1);
    
    mean_c_sim_e = mean(mean_c_sim_e_bywage, 1);
    mean_search_sim_pandemic_expect = mean(mean_search_sim_pandemic_expect_bywage, 1);
    mean_search_sim_pandemic_expect_noFPUC = mean(mean_search_sim_pandemic_expect_noFPUC_bywage, 1);
    mean_search_sim_pandemic_expect_onlyasseteffect = mean(mean_search_sim_pandemic_expect_onlyasseteffect_bywage, 1);
    
    scale_factor=(total_spend_e(1)/income_e(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));
    

    mpc = table();
    mpc.expected('500 6 month') = mpc_expect_500_6month;
    mpc.expected('500 quarterly') = mpc_expect_500_quarterly;
    mpc.expected('500 monthly') = mpc_expect_500_monthly;
    mpc.expected('Last Dollars 6 month') = mpc_expect_lastdollars_6month;
    mpc.expected('Last Dollars quarterly') = mpc_expect_lastdollars_quarterly;
    mpc.expected('Last Dollars monthly') = mpc_expect_lastdollars_monthly;
    mpc.Variables=scale_factor*mpc.Variables
    
    mpc_by_size(FPUC_mult_index)=mpc.expected('500 monthly');
    mpc_marginal_by_size(FPUC_mult_index)=mpc.expected('Last Dollars monthly');
    
    mpc_by_size_quarterly(FPUC_mult_index)=mpc.expected('500 quarterly');
    mpc_marginal_by_size_quarterly(FPUC_mult_index)=mpc.expected('Last Dollars quarterly');
    
    mpc_by_size_6month(FPUC_mult_index)=mpc.expected('500 6 month');
    mpc_marginal_by_size_6month(FPUC_mult_index)=mpc.expected('Last Dollars 6 month');
    
      
    newjob_exit_rate_FPUC=mean_search_sim_pandemic_expect(4:numsim)';
    newjob_exit_rate_no_FPUC=mean_search_sim_pandemic_expect_noFPUC(4:numsim)';
    newjob_exit_rate_onlyasseteffect=mean_search_sim_pandemic_expect_onlyasseteffect(4:numsim)';
    
    newjob_exit_rate_FPUC(end:1000) = newjob_exit_rate_FPUC(end);
    newjob_exit_rate_no_FPUC(end:1000) = newjob_exit_rate_no_FPUC(end);
    newjob_exit_rate_onlyasseteffect(end:1000) = newjob_exit_rate_onlyasseteffect(end);
    
    recall_probs=.08*ones(1000,1);
    
    benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
    perc_change_benefits_data = benefit_change_data.non_sym_per_change(1) * FPUC_mult(FPUC_mult_index);
    %perc_change_benefits_data=FPUC_mult(FPUC_mult_index)*.51/.21;
    date_sim_start = datetime(2020, 4, 1);
    t_start = 2;
    % Surprise period minus one should be period 4 (April is 1, May is 2, June is 3, July is 4, August is 5)
    t_end = 10;
    include_self_employed = 0;
    mean_c_sim_pandemic_surprise_overall_FPUC = NaN;
    mean_c_sim_pandemic_surprise_overall_noFPUC = NaN;
    mean_c_sim_e_overall = NaN;
    [elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_onlyasseteffect, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
    elasticity_and_distortions_values_by_size(1:4,FPUC_mult_index)=[elasticity employment_distortion total_diff_employment share_unemployment_reduced]';
    elasticity_and_distortions_values_by_size(5,FPUC_mult_index)=newjob_exit_rate_no_FPUC(1);
    elasticity_and_distortions_values_by_size(6,FPUC_mult_index)=newjob_exit_rate_FPUC(1);
    elasticity_and_distortions_values_by_size(7,FPUC_mult_index)=elasticity*perc_change_benefits_data;
    elasticity_and_distortions_values_by_size(8,FPUC_mult_index)=average_duration(recall_probs+newjob_exit_rate_onlyasseteffect);
    
    if FPUC_mult_index==1
    baseline_duration=average_duration(recall_probs+newjob_exit_rate_no_FPUC);
    end


end

if beta_normal_loop==1
    mpc_by_size_beta_target500MPC_6month=mpc_by_size_6month;
    mpc_marginal_by_size_beta_target500MPC_6month=mpc_marginal_by_size_6month;

    mpc_by_size_beta_target500MPC_quarterly=mpc_by_size_quarterly;
    mpc_marginal_by_size_beta_target500MPC_quarterly=mpc_marginal_by_size_quarterly;

    mpc_by_size_beta_target500MPC=mpc_by_size;
    mpc_marginal_by_size_beta_target500MPC=mpc_marginal_by_size;
else
    duration_increase_total_beta_targetwaiting=squeeze(elasticity_and_distortions_values_by_size(8,1:end))-baseline_duration;
    duration_increase_marginal_beta_targetwaiting=squeeze(elasticity_and_distortions_values_by_size(8,2:end))-squeeze(elasticity_and_distortions_values_by_size(8,1:end-1));
    duration_increase_marginal_beta_targetwaiting=[duration_increase_total_beta_targetwaiting(1) duration_increase_marginal_beta_targetwaiting];
    
    mpc_by_size_beta_targetwaiting_6month=mpc_by_size_6month;
    mpc_marginal_by_size_beta_targetwaiting_6month=mpc_marginal_by_size_6month;

    mpc_by_size_beta_targetwaiting_quarterly=mpc_by_size_quarterly;
    mpc_marginal_by_size_beta_targetwaiting_quarterly=mpc_marginal_by_size_quarterly;
    
    mpc_by_size_beta_targetwaiting=mpc_by_size;
    mpc_marginal_by_size_beta_targetwaiting=mpc_marginal_by_size;
end

end

duration_increase_total_stimcheck_onetenth=duration_increase_total_beta_targetwaiting;
duration_increase_marginal_stimcheck_onetenth=duration_increase_marginal_beta_targetwaiting;

prop_adj=.91;  %computed in non one-tenth code to hit quarterly .25 MPC in model with heterogeneity

mpc_by_size_stimcheck_onetenth=.08*mpc_by_size_beta_targetwaiting+.92*mpc_by_size_beta_target500MPC*prop_adj;
mpc_marginal_by_size_stimcheck_onetenth=.08*mpc_marginal_by_size_beta_targetwaiting+.92*mpc_marginal_by_size_beta_target500MPC*prop_adj;

mpc_by_size_stimcheck_quarterly_onetenth=.08*mpc_by_size_beta_targetwaiting_quarterly+.92*mpc_by_size_beta_target500MPC_quarterly*prop_adj;
mpc_marginal_by_size_stimcheck_quarterly_onetenth=.08*mpc_marginal_by_size_beta_targetwaiting_quarterly+.92*mpc_marginal_by_size_beta_target500MPC_quarterly*prop_adj;

mpc_by_size_stimcheck_6month_onetenth=.08*mpc_by_size_beta_targetwaiting_6month+.92*mpc_by_size_beta_target500MPC_6month*prop_adj;
mpc_marginal_by_size_stimcheck_6month_onetenth=.08*mpc_marginal_by_size_beta_targetwaiting_6month+.92*mpc_marginal_by_size_beta_target500MPC_6month*prop_adj;

save('stimulus_check_size_results_onetenth','mpc_by_size_stimcheck_onetenth','mpc_marginal_by_size_stimcheck_onetenth','mpc_by_size_stimcheck_quarterly_onetenth','mpc_marginal_by_size_stimcheck_quarterly_onetenth','mpc_by_size_stimcheck_6month_onetenth','mpc_marginal_by_size_stimcheck_6month_onetenth','duration_increase_total_stimcheck_onetenth','duration_increase_marginal_stimcheck_onetenth')