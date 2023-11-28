display('Simulating Full Model Effects of $300')
clearvars -except -regexp fig_paper_*

load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load inter_time_series_input_directory.mat
load hh_wage_groups.mat
load release_paths.mat
load marginal_effects_hazard_calc_inputs.mat


load bestfit_prepandemic.mat
load bestfit_target_waiting_MPC.mat

table_effects_summary = readtable(inter_time_series_input_directory);
inter_time_series_expiration=1-(1+table_effects_summary.ts_exit(1))^4;
cross_section_expiration=1-(1+table_effects_summary.cs_exit(1))^4;
inter_time_series_onset=1-(1+table_effects_summary.ts_exit(2))^4;
cross_section_onset=1-(1+table_effects_summary.cs_exit(2))^4;
cross_section_expiration_logit=1-(1+table_effects_summary.cs_exit(1)*marginal_effects_hazard_calc_inputs.expiry_600(2)/marginal_effects_hazard_calc_inputs.expiry_600(1))^4;
cross_section_onset_logit=1-(1+table_effects_summary.cs_exit(2)*marginal_effects_hazard_calc_inputs.onset_300(2)/marginal_effects_hazard_calc_inputs.onset_300(1))^4;

% Plot settings
% Plot colors
load matlab_qual_colors.mat
global qual_blue qual_purple qual_green qual_orange matlab_red_orange qual_yellow
load graph_axis_labels_timeseries.mat


EIP2 = 1200;
EIP2_e=600;
EIP3 = 4000;

%EIP2=1100;
%EIP2_e=1100;


data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
total_spend_e = data_update_e.value;
total_spend_u = data_update_u.value;
total_spend_e_yoy = total_spend_e(13:end) ./ total_spend_e(1:end - 12) * total_spend_e(13);
total_spend_u_yoy = total_spend_u(13:end) ./ total_spend_u(1:end - 12) * total_spend_u(13);
total_spend_u_jan20=total_spend_u(13);
total_spend_u_feb21=total_spend_u(26);
total_spend_e_jan20=total_spend_e(13);
total_spend_e = total_spend_e(13 + 10:end);
total_spend_u = total_spend_u(13 + 10:end);
perc_spend_e = data_update_e.percent_change;
perc_spend_u = data_update_u.percent_change;
perc_spend_u_vs_e = perc_spend_u - perc_spend_e;
perc_spend_u_vs_e = perc_spend_u_vs_e(13 + 10:end);
spend_dollars_u_vs_e = perc_spend_u_vs_e * total_spend_u_jan20;

idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
income_e = data_update_e.value;
income_u = data_update_u.value;
income_e_yoy = income_e(13:end) ./ income_e(1:end - 12) * income_e(13);
income_u_yoy = income_u(13:end) ./ income_u(1:end - 12) * income_u(13);
income_u_jan20=income_u(13);
income_e_jan20=income_e(13);
income_e = income_e(13 + 10:end);
income_u = income_u(13 + 10:end);
perc_income_e = data_update_e.percent_change;
perc_income_u = data_update_u.percent_change;
perc_income_u_vs_e = perc_income_u - perc_income_e;
perc_income_u_vs_e = perc_income_u_vs_e(13 + 10:end);
income_dollars_u_vs_e = perc_income_u_vs_e * income_u_jan20;

income_u_vs_e=income_u./income_e;
total_spend_u_vs_e=total_spend_u./total_spend_e;

idx_emp = (string(data_update.category) == 'Checking account balance') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
idx_u = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
data_update_u = data_update(idx_u, :);
checking_e = data_update_e.value;
checking_u = data_update_u.value;
checking_e = checking_e(13:end);
checking_u = checking_u(13:end);

ratio_u = checking_u(10) / checking_u(1);
ratio_e = checking_e(10) / checking_e(1);

k_prepandemic = pre_pandemic_fit_match500MPC(1);
gamma_prepandemic = pre_pandemic_fit_match500MPC(2);
c_param_prepandemic = 0;


mpc_onset_data=((total_spend_u(3)-total_spend_u(2))-(total_spend_e(3)-total_spend_e(2)))/((income_u(3)-income_u(2))-(income_e(3)-income_e(2)));

% Assign parameter values
load discountfactors.mat
beta_normal = beta_targetwaiting;
beta_high = beta_oneperiodshock;

load model_parameters.mat
initial_a = initial_a - aprimemin;

n_ben_profiles_allowed = 3; %This captures the surprise vs. expected expiration and no FPUC

% Set on/off switches
infinite_dur = 0;
use_initial_a = 1;



% Start solving the model with EGM
for iy = 1:5

    y = w(iy);
    h = 0.7 * y;
    b = repshare * y;

    for surprise = 0:1

        rng('default')

        k = sse_surprise_fit_het_full(1);
        gamma = sse_surprise_fit_het_full(2);
        c_param = sse_surprise_fit_het_full(3);


        % Aprime grid
        aprimemax = 2000;
        Aprime = exp(linspace(0.00, log(aprimemax), n_aprime)) - 1;
        Aprime = Aprime';

        %regular benefits profile
        benefit_profile(1:6, 1) = h + b;

        if infinite_dur == 1
            benefit_profile(7:13, 1) = h + b;
        else
            benefit_profile(7:13, 1) = h;
        end

        %expect $300 for 10 months (expect)
        benefit_profile_pandemic(1:2, 1) = b + h;
        benefit_profile_pandemic(3:12, 1) = b + h + FPUC_onset;

        if infinite_dur == 1
            benefit_profile_pandemic(13:13, 1) = b + h;
        else
            benefit_profile_pandemic(13:13, 1) = h;
        end

        %expect $300 for 3 months (surprise)
        benefit_profile_pandemic(1:2, 2) = b + h;
        benefit_profile_pandemic(3:5, 2) = b + h + FPUC_onset;
        benefit_profile_pandemic(6:10, 2) = b + h;

        if infinite_dur == 1
            benefit_profile_pandemic(11:13, 2) = b + h + FPUC_onset;
        else
            benefit_profile_pandemic(11:13, 2) = h;
        end

        %No FPUC
        benefit_profile_pandemic(1:10, 3) = h + b;

        if infinite_dur == 1
            benefit_profile_pandemic(11:13, 3) = h + b;
        else
            benefit_profile_pandemic(11:13, 3) = h;
        end

        recall_probs_pandemic(1:13) = .08;
        recall_probs_regular = recall_probs_pandemic;

        %initialization of variables for speed
        c_pol_e = zeros(n_aprime, 1);
        c_pol_u = zeros(n_aprime, n_b, 1);
        c_pol_u_pandemic = zeros(n_aprime, n_b, n_ben_profiles_allowed);
        v_e = c_pol_e;
        v_u = c_pol_u;
        v_u_pandemic = c_pol_u_pandemic;

        rhs_e = zeros(n_aprime, 1);
        rhs_u = zeros(n_aprime, n_b);
        rhs_u_pandemic = zeros(n_aprime, n_b, n_ben_profiles_allowed);

        for beta_loop = 1:1

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

            tic
            %c_pol is c(a,y)
            %c_tilde is c(a',y)
            %while (iter <= 5000) & (diffC > tol) %& (diffC_percent > tol_percent) %| diffV > tol)

            if beta_loop == 2
                maxiter = 1; %this effectively governs how many periods households will think the high discount factor will last, setting maxiter=1 essentially runs one backward induction step from the beta_normal solutions
                %note that the code must be structured so that it solves the
                %beta_normal part first
                c_pol_e_guess = c_pol_e_betanormal;
                c_pol_u_guess = c_pol_u_betanormal;
                c_pol_u_pandemic_guess = c_pol_u_pandemic_betanormal;

                v_e_guess = v_e_betanormal;
                v_u_guess = v_u_betanormal;
                v_u_pandemic_guess = v_u_pandemic_betanormal;
            else
                maxiter = 1000;
           end

            while ((ave_change_in_C_percent > tol_c_percent) || (ave_change_in_S > tol_s)) && iter < maxiter

                %c_pol is c(a,y)
                %c_tilde is c(a',y)
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

                end

                diffC = max([max(max(abs(c_pol_e(:) - c_pol_e_guess(:)))), max(max(max(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), max(max(max(max(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :))))))]);

                diffC_percent = 100 * max([max(max(abs((c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess(:)))), max(max(max(abs((c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :))))), max(max(max(max(abs((c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess(:, :, :))))))]);

                % Absolute difference in value to measure convergence
                diffV = max([max(abs(v_e(:) - v_e_guess(:))), max(max(abs(v_u(:, :) - v_u_guess(:, :)))), max(max(max(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

                ave_change_in_C = mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :))))))]);

                ave_change_in_C_percent = 100 * mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);

                ave_change_in_V = mean([mean(abs(v_e(:) - v_e_guess(:))), mean(mean(abs(v_u(:, :) - v_u_guess(:, :)))), mean(mean(mean(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

                ave_change_in_S = mean([mean(mean(mean(abs(optimal_search(:, :) - optimal_search_guess(:, :))))), mean(mean(mean(mean(abs(optimal_search_pandemic(:, :, :) - optimal_search_pandemic_guess(:, :, :))))))]);

                % if iter == 100
                %[iter diffC ave_change_in_C ave_change_in_C_percent ave_change_in_S diffV ave_change_in_V]

                %[iter ave_change_in_C ave_change_in_C_percent ave_change_in_S]
                %     stop
                % end

                % Update guesses, fully for now.
                c_pol_e_guess = c_pol_e;
                c_pol_u_guess = c_pol_u;
                c_pol_u_pandemic_guess = c_pol_u_pandemic;
                v_e_guess = v_e;
                v_u_guess = v_u;
                v_u_pandemic_guess = v_u_pandemic;
                optimal_search_guess = optimal_search;
                optimal_search_pandemic_guess = optimal_search_pandemic;

                % Count the iteration
                iter = iter + 1;

            end

            if beta_loop == 1
                c_pol_e_betanormal = c_pol_e;
                c_pol_u_betanormal = c_pol_u;
                c_pol_u_pandemic_betanormal = c_pol_u_pandemic;

                v_e_betanormal = v_e;
                v_u_betanormal = v_u;
                v_u_pandemic_betanormal = v_u_pandemic;
            elseif beta_loop == 2
                c_pol_e_betahigh = c_pol_e;
                c_pol_u_betahigh = c_pol_u;
                c_pol_u_pandemic_betahigh = c_pol_u_pandemic;

                v_e_betahigh = v_e;
                v_u_betahigh = v_u;
                v_u_pandemic_betahigh = v_u_pandemic;
            end

        end

        A = Aprime;

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
        numsim = 15;
        burnin = 5;
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
        a_sim_with_500(:, 1) = a_sim_with_500(:, 1) + 500 * FPUC_onset / (4.5 * 600);

        if use_initial_a == 1
            load a_init_bestfit;
            initial_a_vec=a_init_bestfit;
            initial_a=initial_a_vec(iy);

            a_sim_pandemic_surprise = checking_u(10)/income_e(1);
            a_sim_pandemic_expect = checking_u(10)/income_e(1);
            a_sim_pandemic_expect_jan_start = checking_u(10)/income_e(1);
            a_sim_pandemic_surprise_wait = checking_u(10)/income_e(1);
            a_sim_pandemic_expect_wait = checking_u(10)/income_e(1);
            a_sim_regular = checking_u(10)/income_e(1);
            a_sim_pandemic_noFPUC = checking_u(10)/income_e(1);
            a_sim_e = checking_u(10)/income_e(1);
            a_sim_pandemic_expect_onlyasseteffect=a_sim_pandemic_noFPUC;

            a_sim_pandemic_surprise = initial_a;
            a_sim_pandemic_expect = initial_a;
            a_sim_pandemic_expect_jan_start = initial_a;
            a_sim_pandemic_surprise_wait = initial_a;
            a_sim_pandemic_expect_wait = initial_a;
            a_sim_regular = initial_a;
            a_sim_pandemic_noFPUC = initial_a;
            a_sim_e = initial_a;
            a_sim_pandemic_expect_onlyasseteffect=initial_a;
        else
            a_sim_pandemic_surprise = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect = tmp_a(tmp_u > 0);            
            a_sim_pandemic_expect_jan_start = tmp_a(tmp_u > 0);
            a_sim_pandemic_surprise_wait = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0);
            a_sim_regular = tmp_a(tmp_u > 0);
            a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0);
            a_sim_e = tmp_a(tmp_u == 0);
            a_sim_pandemic_expect_onlyasseteffect=a_sim_pandemic_noFPUC;
        end

        num_unemployed_hh = length(a_sim_pandemic_surprise);
        num_employed_hh = length(a_sim_e);
        c_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect_jan_start = zeros(length(a_sim_pandemic_expect_jan_start), 30);
        c_sim_regular = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect_onlyasseteffect=zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_e = zeros(length(a_sim_e), 30);

        search_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
        search_sim_pandemic_expect_jan_start = zeros(length(a_sim_pandemic_expect_jan_start), 30);
        search_sim_regular = zeros(length(a_sim_regular), 30);
        search_sim_pandemic_noFPUC = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_expect_onlyasseteffect=zeros(length(a_sim_pandemic_surprise), 30);

        %this is looping over just unemployed households (continuously unemployed)
        %to get u time-series patterns
        length_u = 0;

        for t = 1:15
            length_u = min(length_u + 1, n_b);

            for i = 1:num_unemployed_hh
                %Jan EIP
                if t == 3
                    a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + EIP2 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_surprise(i, t) = a_sim_pandemic_surprise(i, t) + EIP2 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_expect_jan_start(i, t) = a_sim_pandemic_expect_jan_start(i, t) + EIP2 * FPUC_onset / (4.5 * 600);

                    a_sim_pandemic_noFPUC(i, t) = a_sim_pandemic_noFPUC(i, t) + EIP2 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_expect_onlyasseteffect(i, t) = a_sim_pandemic_expect_onlyasseteffect(i, t) + EIP2 * FPUC_onset / (4.5 * 600);
                end

                if t == 5
                    a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_surprise(i, t) = a_sim_pandemic_surprise(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_expect_jan_start(i, t) = a_sim_pandemic_expect_jan_start(i, t) + EIP3 * FPUC_onset / (4.5 * 600);

                    a_sim_pandemic_noFPUC(i, t) = a_sim_pandemic_noFPUC(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_expect_onlyasseteffect(i, t) = a_sim_pandemic_expect_onlyasseteffect(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                end
                

                if t == 1
                    c_pol_u = c_pol_u_betanormal;
                    c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
                end
                
                if t==3
                   a_sim_pandemic_expect_onlyasseteffect(i, t) = a_sim_pandemic_expect_onlyasseteffect(i, t)+10*FPUC_onset; 
                end
                if t==11
                   a_sim_pandemic_expect_onlyasseteffect(i, t) = a_sim_pandemic_expect_onlyasseteffect(i, t)-2*FPUC_onset; 
                end

                c_sim_regular(i, t) = interp1(A, c_pol_u(:, length_u), a_sim_regular(i, t), 'linear');
                a_sim_regular(i, t + 1) = benefit_profile(length_u) + (1 + r) * a_sim_regular(i, t) - c_sim_regular(i, t);

                % Expect Jan onset in Nov:
                c_sim_pandemic_expect_jan_start(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect_jan_start(i, t), 'linear');
                a_sim_pandemic_expect_jan_start(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect_jan_start(i, t) - c_sim_pandemic_expect_jan_start(i, t), 0);
                
                if t <= 2 || t>=11
                    c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_expect(i, t), 'linear');
                    a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);
                else
                    c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect(i, t), 'linear');
                    a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);
                end
                
               
                c_sim_pandemic_expect_onlyasseteffect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_expect_onlyasseteffect(i, t), 'linear');
                a_sim_pandemic_expect_onlyasseteffect(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_expect_onlyasseteffect(i, t) - c_sim_pandemic_expect_onlyasseteffect(i, t), 0);
                
                


                if t <= 2 || t>=11 %pre-onset
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                elseif t >= 3 && t <= 4 %jan-feb
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                else %march extension
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                end

                c_sim_pandemic_noFPUC(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_noFPUC(i, t), 'linear');
                a_sim_pandemic_noFPUC(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_noFPUC(i, t) - c_sim_pandemic_noFPUC(i, t), 0);

                diff_v = interp1(A, v_e(:), a_sim_regular(i, t + 1), 'linear') - interp1(A, v_u(:, min(length_u + 1, n_b)), a_sim_regular(i, t + 1), 'linear');
                search_sim_regular(i, t) = min(1 - recall_probs_regular(ib), max(0, (beta * (diff_v) / k_prepandemic).^(1 / gamma_prepandemic)));
                if imag(search_sim_regular(i, t)) ~= 0
                    search_sim_regular(i, t) = 0;
                end
                
                
                diff_v = interp1(A, v_e(:), a_sim_pandemic_noFPUC(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_noFPUC(i, t + 1), 'linear');
                search_sim_pandemic_noFPUC(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_noFPUC(i, t)) ~= 0
                    search_sim_pandemic_noFPUC(i, t) = 0;
                end
                
                diff_v = interp1(A, v_e(:), a_sim_pandemic_expect_onlyasseteffect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_expect_onlyasseteffect(i, t + 1), 'linear');
                search_sim_pandemic_expect_onlyasseteffect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_expect_onlyasseteffect(i, t)) ~= 0
                    search_sim_pandemic_expect_onlyasseteffect(i, t) = 0;
                end
                
                

     
                % expect Jan onset in Nov:
                diff_v = interp1(A, v_e(:), a_sim_pandemic_expect_jan_start(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_expect_jan_start(i, t + 1), 'linear');
                search_sim_pandemic_expect_jan_start(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_expect_jan_start(i, t)) ~= 0
                    search_sim_pandemic_expect_jan_start(i, t) = 0;
                end


                if t <= 2 || t>=11
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_expect(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_expect(i, t + 1), 'linear');
                end

                search_sim_pandemic_expect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_expect(i, t)) ~= 0
                    search_sim_pandemic_expect(i, t) = 0;
                end

                if t <= 2 || t>=11
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_surprise(i, t + 1), 'linear');
                elseif t >= 3 && t <= 4
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 2), a_sim_pandemic_surprise(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_surprise(i, t + 1), 'linear');
                end

                search_sim_pandemic_surprise(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_surprise(i, t)) ~= 0
                    search_sim_pandemic_surprise(i, t) = 0;
                end

                %note for surprise case won't want to use i_b+1 in actuality will
                %want to use the expected one in the last period before surprise

            end

        end

        for t = 1:numsim

            for i = 1:num_employed_hh

                if t == 3
                    a_sim_e(i, t) = a_sim_e(i, t) + EIP2_e * FPUC_onset / (4.5 * 600);
                end

                if t == 5
                    a_sim_e(i, t) = a_sim_e(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                end

                %adjust initial assets isomorphic to allowing for borrowing
                %if t==1
                %    a_sim_e(i,t)=a_sim_e(i,t)+5*1320*FPUC_onset/(4.5*600);
                %end

                c_sim_e(i, t) = interp1(A, c_pol_e(:), a_sim_e(i, t), 'linear');
                a_sim_e(i, t + 1) = y + (1 + r) * a_sim_e(i, t) - c_sim_e(i, t);
            end

        end

        if surprise == 1
            mean_a_sim_pandemic_surprise = mean(a_sim_pandemic_surprise, 1);
            mean_c_sim_pandemic_surprise = mean(c_sim_pandemic_surprise, 1);

            mean_a_sim_pandemic_surprise_noFPUC = mean(a_sim_pandemic_noFPUC, 1);
            mean_c_sim_pandemic_surprise_noFPUC = mean(c_sim_pandemic_noFPUC, 1);

            mean_a_sim_regular = mean(a_sim_regular, 1);
            mean_c_sim_regular = mean(c_sim_regular, 1);

            mean_a_sim_e_surprise = mean(a_sim_e, 1);
            mean_c_sim_e_surprise = mean(c_sim_e, 1);

            mean_search_sim_regular = mean(search_sim_regular, 1);
            mean_search_sim_pandemic_surprise_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
            mean_search_sim_pandemic_surprise = mean(search_sim_pandemic_surprise, 1);
        else
            mean_a_sim_pandemic_expect = mean(a_sim_pandemic_expect, 1);
            mean_c_sim_pandemic_expect = mean(c_sim_pandemic_expect, 1);

            mean_a_sim_pandemic_expect_jan_start = mean(a_sim_pandemic_expect_jan_start, 1);
            mean_c_sim_pandemic_expect_jan_start = mean(c_sim_pandemic_expect_jan_start, 1);

            mean_a_sim_pandemic_expect_noFPUC = mean(a_sim_pandemic_noFPUC, 1);
            mean_c_sim_pandemic_expect_noFPUC = mean(c_sim_pandemic_noFPUC, 1);

            mean_search_sim_pandemic_expect = mean(search_sim_pandemic_expect, 1);
            mean_search_sim_pandemic_expect_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
            mean_search_sim_pandemic_expect_jan_start = mean(search_sim_pandemic_expect_jan_start, 1);
            mean_search_sim_pandemic_expect_jan_start_noFPUC = mean(search_sim_pandemic_noFPUC, 1);
            mean_search_sim_pandemic_expect_onlyasseteffect = mean(search_sim_pandemic_expect_onlyasseteffect, 1);
            
            mean_a_sim_pandemic_expect_onlyasseteffect = mean(a_sim_pandemic_expect_onlyasseteffect, 1);
            mean_c_sim_pandemic_expect_onlyasseteffect = mean(c_sim_pandemic_expect_onlyasseteffect, 1);

            mean_a_sim_e = mean(a_sim_e, 1);
            mean_c_sim_e = mean(c_sim_e, 1);
        end

    end

    % mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
    % mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);

    % mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
    % mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);

    % mean_a_sim_regular=mean(a_sim_regular,1);
    % mean_c_sim_regular=mean(c_sim_regular,1);

    mean_c_sim_e_bywage(iy, :) = mean_c_sim_e;
    mean_a_sim_e_bywage(iy, :) = mean_a_sim_e;

    mean_search_sim_regular_bywage(iy, :) = mean_search_sim_regular;
    mean_search_sim_pandemic_expect_bywage(iy, :) = mean_search_sim_pandemic_expect;
    mean_search_sim_pandemic_surprise_bywage(iy, :) = mean_search_sim_pandemic_surprise;
    mean_search_sim_pandemic_expect_noFPUC_bywage(iy, :) = mean_search_sim_pandemic_expect_noFPUC;
    mean_search_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = mean_search_sim_pandemic_expect_onlyasseteffect;
    mean_search_sim_pandemic_surprise_noFPUC_bywage(iy, :) = mean_search_sim_pandemic_surprise_noFPUC;
    mean_search_sim_pandemic_expect_jan_start_bywage(iy, :) = mean_search_sim_pandemic_expect_jan_start;

    mean_a_sim_regular_bywage(iy, :) = mean_a_sim_regular;
    mean_a_sim_pandemic_expect_bywage(iy, :) = mean_a_sim_pandemic_expect;
    mean_a_sim_pandemic_surprise_bywage(iy, :) = mean_a_sim_pandemic_surprise;
    mean_a_sim_pandemic_expect_noFPUC_bywage(iy, :) = mean_a_sim_pandemic_expect_noFPUC;
    mean_a_sim_pandemic_surprise_noFPUC_bywage(iy, :) = mean_a_sim_pandemic_surprise_noFPUC;
    mean_a_sim_pandemic_expect_jan_start_bywage(iy, :) = mean_a_sim_pandemic_expect_jan_start;
    mean_a_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = mean_a_sim_pandemic_expect_onlyasseteffect;

    mean_c_sim_regular_bywage(iy, :) = mean_c_sim_regular;
    mean_c_sim_pandemic_expect_bywage(iy, :) = mean_c_sim_pandemic_expect;
    mean_c_sim_pandemic_surprise_bywage(iy, :) = mean_c_sim_pandemic_surprise;
    mean_c_sim_pandemic_expect_noFPUC_bywage(iy, :) = mean_c_sim_pandemic_expect_noFPUC;
    mean_c_sim_pandemic_surprise_noFPUC_bywage(iy, :) = mean_c_sim_pandemic_surprise_noFPUC;
    mean_c_sim_pandemic_expect_jan_start_bywage(iy, :) = mean_c_sim_pandemic_expect_jan_start;
    mean_c_sim_pandemic_expect_onlyasseteffect_bywage(iy, :) = mean_c_sim_pandemic_expect_onlyasseteffect;

    mean_y_sim_e_bywage(iy,:)=w(iy)*ones(1,13);
    mean_y_sim_e_bywage(iy,3)=mean_y_sim_e_bywage(iy,3)+EIP2_e * FPUC_onset / (4.5 * 600);
    mean_y_sim_e_bywage(iy,5)=mean_y_sim_e_bywage(iy,5)+EIP3 * FPUC_onset / (4.5 * 600);

    mean_y_sim_pandemic_u_bywage(iy, :) = benefit_profile_pandemic(:, 1)';
    mean_y_sim_pandemic_u_bywage(iy, 3) = mean_y_sim_pandemic_u_bywage(iy, 3) + EIP2 * FPUC_onset / (4.5 * 600);
    mean_y_sim_pandemic_u_bywage(iy, 5) = mean_y_sim_pandemic_u_bywage(iy, 5) + EIP3 * FPUC_onset / (4.5 * 600);

    mean_y_sim_pandemic_noFPUC_bywage(iy, :) = benefit_profile_pandemic(:, 3)';
    mean_y_sim_pandemic_noFPUC_bywage(iy, 3) = mean_y_sim_pandemic_noFPUC_bywage(iy, 3) + EIP2 * FPUC_onset / (4.5 * 600);
    mean_y_sim_pandemic_noFPUC_bywage(iy, 5) = mean_y_sim_pandemic_noFPUC_bywage(iy, 5) + EIP3 * FPUC_onset / (4.5 * 600);

    mean_y_sim_pandemic_u_bywage(iy,11:12)=mean_y_sim_pandemic_noFPUC_bywage(iy,11:12);

    mean_y_sim_regular_bywage(iy, :) = [benefit_profile(:, 1)'];
end

mean_search_sim_pandemic_surprise = mean(mean_search_sim_pandemic_surprise_bywage, 1);
mean_search_sim_pandemic_surprise_noFPUC = mean(mean_search_sim_pandemic_surprise_noFPUC_bywage, 1);
mean_search_sim_pandemic_expect = mean(mean_search_sim_pandemic_expect_bywage, 1);
mean_search_sim_pandemic_expect_noFPUC = mean(mean_search_sim_pandemic_expect_noFPUC_bywage, 1);
mean_search_sim_pandemic_expect_jan_start = mean(mean_search_sim_pandemic_expect_jan_start_bywage, 1);
mean_search_sim_pandemic_expect_onlyasseteffect = mean(mean_search_sim_pandemic_expect_onlyasseteffect_bywage, 1);

mean_y_sim_pandemic_u = mean(mean_y_sim_pandemic_u_bywage, 1);
mean_y_sim_pandemic_noFPUC = mean(mean_y_sim_pandemic_noFPUC_bywage, 1);
mean_y_sim_e = mean(mean_y_sim_e_bywage,1);

mean_y_sim_u_vs_e=mean_y_sim_pandemic_u./mean_y_sim_e;

mean_c_sim_e = mean(mean_c_sim_e_bywage, 1);
mean_c_sim_pandemic_surprise = mean(mean_c_sim_pandemic_surprise_bywage, 1);
mean_c_sim_pandemic_surprise_noFPUC = mean(mean_c_sim_pandemic_surprise_noFPUC_bywage, 1);
mean_c_sim_pandemic_expect = mean(mean_c_sim_pandemic_expect_bywage, 1);
mean_c_sim_pandemic_expect_noFPUC = mean(mean_c_sim_pandemic_expect_noFPUC_bywage, 1);
mean_c_sim_pandemic_regular = mean(mean_c_sim_regular_bywage, 1);
mean_c_sim_pandemic_expect_jan_start = mean(mean_c_sim_pandemic_expect_jan_start_bywage, 1);
mean_c_sim_pandemic_expect_onlyasseteffect = mean(mean_c_sim_pandemic_expect_onlyasseteffect_bywage, 1);

exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-11-01') & datenum(exit_rates_data.week_start_date) < datenum('2021-04-01');
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
recall_probs_pandemic_actual = exit_rates_data_week_to_month.exit_to_recall';
recall_probs_alt=.08*ones(1,length(recall_probs_pandemic_actual));

%Convert model simulations to dollar deviations in U vs. E space
mean_c_sim_pandemic_surprise_dollars = mean_c_sim_pandemic_surprise(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_c_sim_pandemic_surprise_noFPUC_dollars = mean_c_sim_pandemic_surprise_noFPUC(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_c_sim_pandemic_expect_dollars = mean_c_sim_pandemic_expect(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_c_sim_pandemic_expect_jan_start_dollars = mean_c_sim_pandemic_expect_jan_start(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_c_sim_pandemic_expect_noFPUC_dollars = mean_c_sim_pandemic_expect_noFPUC(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_c_sim_e_dollars = mean_c_sim_e(1:18) ./ mean_c_sim_e(1:18) * total_spend_e_jan20 - total_spend_e_jan20;
mean_c_sim_pandemic_expect_onlyasseteffect_dollars = mean_c_sim_pandemic_expect_onlyasseteffect(1:18) ./ mean_c_sim_e(1:18) * total_spend_u_jan20 - total_spend_u_jan20;
mean_y_sim_pandemic_u_dollars = mean_y_sim_pandemic_u./mean_y_sim_e * total_spend_u_jan20 - total_spend_u_jan20;
mean_y_sim_pandemic_u_noFPUC_dollars = mean_y_sim_pandemic_noFPUC./mean_y_sim_e * total_spend_u_jan20 - total_spend_u_jan20;
mean_y_sim_e_dollars = 0;


perc_income_sim_e = mean_y_sim_e/mean_y_sim_e(1)-1;
perc_income_sim_u = mean_y_sim_pandemic_u/mean_y_sim_pandemic_u(1)-1;
perc_income_sim_u_vs_e = perc_income_sim_u - perc_income_sim_e;
income_dollars_sim_u_vs_e = perc_income_sim_u_vs_e * income_e(4);

mean_y_sim_pandemic_u_dollars = mean_y_sim_pandemic_u_dollars + (mean(income_dollars_u_vs_e(1:2)) - mean_y_sim_pandemic_u_dollars(1));
mean_y_sim_pandemic_u_noFPUC_dollars = mean_y_sim_pandemic_u_noFPUC_dollars + (mean(income_dollars_u_vs_e(1:2)) - mean_y_sim_pandemic_u_noFPUC_dollars(1));

scale_factor=(total_spend_e_jan20(1)/income_e_jan20(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));

load prepandemic_results_target500MPC
load prepandemic_results_onset_target500MPC
load prepandemic_andpandemic_results_onset_target500MPC.mat

% close all

spend_data = ((total_spend_u(1:4) ./ total_spend_e(1:4) - 1) - (total_spend_u_feb21 ./ total_spend_e(4) - 1)) * 100;
s1 = (mean_c_sim_pandemic_expect_jan_start_dollars_match500MPC(1:4) / total_spend_u_feb21 - mean_c_sim_pandemic_expect_jan_start_dollars_match500MPC(4) / total_spend_u_feb21 ) * 100;
s2 = (mean_c_sim_pandemic_expect_dollars_match500MPC(1:4) / total_spend_u_feb21 - mean_c_sim_pandemic_expect_dollars_match500MPC(4) / total_spend_u_feb21 ) * 100;
s3 = (mean_c_sim_pandemic_expect_dollars(1:4) / total_spend_u_feb21 - mean_c_sim_pandemic_expect_dollars(4) / total_spend_u_feb21 ) * 100;

s4 = (mean_c_sim_prepandemic_expect_dollars(1:4) / total_spend_u_feb21 - mean_c_sim_prepandemic_expect_dollars(4) / total_spend_u_feb21) * 100;















newjob_exit_rate_FPUC = mean_search_sim_pandemic_expect(1:numsim)';
newjob_exit_rate_no_FPUC = mean_search_sim_pandemic_expect_noFPUC(1:numsim)';
newjob_exit_rate_onlyasseteffect = mean_search_sim_pandemic_expect_onlyasseteffect(1:numsim)';
newjob_exit_rate_FPUC(end:1000) = newjob_exit_rate_FPUC(end);
newjob_exit_rate_no_FPUC(end:1000) = newjob_exit_rate_no_FPUC(end);
newjob_exit_rate_onlyasseteffect(end:1000) = newjob_exit_rate_onlyasseteffect(end);


newjob_exit_rate_data=monthly_search_data(1:5)';
newjob_exit_rate_data(end:1000,:)=monthly_search_data(5);

newjob_exit_rate_inter_time_series_based_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_inter_time_series_based_no_FPUC(3:10)=newjob_exit_rate_inter_time_series_based_no_FPUC(3:10)+inter_time_series_onset;
newjob_exit_rate_cross_section_based_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_no_FPUC(3:10)=newjob_exit_rate_cross_section_based_no_FPUC(3:10)+cross_section_onset;
newjob_exit_rate_cross_section_based_logit_no_FPUC=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_logit_no_FPUC(3:10)=newjob_exit_rate_cross_section_based_logit_no_FPUC(3:10)+cross_section_onset_logit;
newjob_exit_rate_model_inter_time_series_based_no_FPUC=newjob_exit_rate_FPUC;
newjob_exit_rate_model_inter_time_series_based_no_FPUC(3:10)=newjob_exit_rate_model_inter_time_series_based_no_FPUC(3:10)+(newjob_exit_rate_model_inter_time_series_based_no_FPUC(2)-newjob_exit_rate_model_inter_time_series_based_no_FPUC(3));


newjob_exit_rate_FPUC_bywage = mean_search_sim_pandemic_expect_bywage(:,1:numsim)';
newjob_exit_rate_no_FPUC_bywage = mean_search_sim_pandemic_expect_noFPUC_bywage(:,1:numsim)';
newjob_exit_rate_FPUC_bywage(end:1000,:) = repmat(newjob_exit_rate_FPUC_bywage(end,:),1000-length(newjob_exit_rate_FPUC_bywage)+1,1);
newjob_exit_rate_no_FPUC_bywage(end:1000,:) = repmat(newjob_exit_rate_no_FPUC_bywage(end,:),1000-length(newjob_exit_rate_no_FPUC_bywage)+1,1);

recall_probs = recall_probs_pandemic_actual(1:5)';
recall_probs(end:1000) = recall_probs(end);
mean_c_sim_pandemic_surprise_overall_FPUC = NaN;
mean_c_sim_pandemic_surprise_overall_noFPUC = NaN;
mean_c_sim_e_overall = NaN;
benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
perc_change_benefits_data = benefit_change_data.non_sym_per_change(2);
include_self_employed = 0;
date_sim_start = datetime(2020, 11, 1);
% Start and end times for onset
t_start = 4;
t_end = 11;
[elasticity_onset employment_distortion_onset total_diff_employment_onset share_unemployment_reduced_onset employment_FPUC_onset employment_noFPUC_onset monthly_spend_pce_onset monthly_spend_no_FPUC_onset total_hazard_elasticity_onset newjob_hazard_elasticity_onset newjob_duration_elasticity_onset total_percent_change_onset,elasticity_prepanlevel_newjob_onset,elasticity_prepanlevel_both_onset,elasticity_prepanlevel_recall_onset,table_elasticity_comparisons_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
% Save table_elasticity_comparisons_onset as csv
% writetable(table_elasticity_comparisons_onset, fullfile(release_path, 'table_elasticity_comparisons_onset.csv'));

%employment effects arising from assets
[elasticity_onlyasseteffect_onset employment_distortion_onlyasseteffect_onset total_diff_employment_onlyasseteffect_onset share_unemployment_reduced_onlyasseteffect_onset employment_FPUC_onlyasseteffect_onset employment_noFPUC_onlyasseteffect_onset monthly_spend_pce_onlyasseteffect_onset monthly_spend_no_FPUC_onlyasseteffect_onset total_hazard_elasticity_onlyasseteffect_onset newjob_hazard_elasticity_onlyasseteffect_onset newjob_duration_elasticity_onlyasseteffect_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_onlyasseteffect, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

%employment effects data based approach
[elasticity_inter_time_series_based_onset employment_distortion_inter_time_series_based_onset total_diff_employment_inter_time_series_based_onset share_unemployment_reduced_inter_time_series_based_onset employment_FPUC_inter_time_series_based_onset employment_noFPUC_inter_time_series_based_onset monthly_spend_pce_inter_time_series_based_onset monthly_spend_no_FPUC_inter_time_series_based_onset total_hazard_elasticity_inter_time_series_based_onset newjob_hazard_elasticity_inter_time_series_based_onset newjob_duration_elasticity_inter_time_series_based_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_inter_time_series_based_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
[elasticity_cross_section_based_onset employment_distortion_cross_section_based_onset total_diff_employment_cross_section_based_onset share_unemployment_reduced_cross_section_based_onset employment_FPUC_cross_section_based_onset employment_noFPUC_cross_section_based_onset monthly_spend_pce_cross_section_based_onset monthly_spend_no_FPUC_cross_section_based_onset total_hazard_elasticity_cross_section_based_onset newjob_hazard_elasticity_cross_section_based_onset newjob_duration_elasticity_cross_section_based_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_cross_section_based_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

[elasticity_model_inter_time_series_based_onset employment_distortion_model_inter_time_series_based_onset total_diff_employment_model_inter_time_series_based_onset share_unemployment_reduced_model_inter_time_series_based_onset employment_FPUC_model_inter_time_series_based_onset employment_noFPUC_model_inter_time_series_based_onset monthly_spend_pce_model_inter_time_series_based_onset monthly_spend_no_FPUC_model_inter_time_series_based_onset total_hazard_elasticity_model_inter_time_series_based_onset newjob_hazard_elasticity_model_inter_time_series_based_onset newjob_duration_elasticity_model_inter_time_series_based_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_model_inter_time_series_based_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

%hazard/duration elasticity specification robustness:
newjob_exit_rate_cross_section_based_no_FPUC_alt1=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_no_FPUC_alt1(3:10)=newjob_exit_rate_cross_section_based_no_FPUC_alt1(3:10)+cross_section_onset-.0023;
[elasticity_alt_1 tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_onset_alt_1] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_cross_section_based_no_FPUC_alt1, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
total_hazard_elasticity_cross_section_based_onset/total_hazard_elasticity_cross_section_based_onset_alt_1
newjob_exit_rate_cross_section_based_no_FPUC_alt2=newjob_exit_rate_data;
newjob_exit_rate_cross_section_based_no_FPUC_alt2(3:10)=newjob_exit_rate_cross_section_based_no_FPUC_alt2(3:10)+cross_section_onset+.0095;
[elasticity_alt_2 tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_onset_alt_2] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_cross_section_based_no_FPUC_alt2, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
total_hazard_elasticity_cross_section_based_onset/total_hazard_elasticity_cross_section_based_onset_alt_2


[elasticity_cross_section_based_logit_onset tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_logit_onset] = elasticity_distortions_and_aggregates(newjob_exit_rate_data, newjob_exit_rate_cross_section_based_logit_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);





display('Liquidity shares of distortions $300')
liquidity_share_onset=elasticity_onlyasseteffect_onset/elasticity_onset
employment_distortion_onlyasseteffect_onset/employment_distortion_onset
inter_model_share_onset=elasticity_model_inter_time_series_based_onset/elasticity_onset
employment_distortion_inter_time_series_based_onset/employment_distortion_onset

perc_change_benefits_data=NaN;
[elasticity_highwage_onset employment_distortion_highwage_onset total_diff_employment_highwage_onset share_unemployment_reduced_highwage_onset employment_FPUC_highwage_onset employment_noFPUC_highwage_onset monthly_spend_pce_highwage monthly_spend_no_FPUC_highwage] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_bywage(:,5), newjob_exit_rate_no_FPUC_bywage(:,5), recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
[elasticity_lowwage_onset employment_distortion_lowwage_onset total_diff_employment_lowwage_onset share_unemployment_reduced_lowwage_onset employment_FPUC_lowwage_onset employment_noFPUC_lowwage_onset monthly_spend_pce_lowwage monthly_spend_no_FPUC_lowwage] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_bywage(:,1), newjob_exit_rate_no_FPUC_bywage(:,1), recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

el_and_distortions_values_lowwage_onset = [elasticity_lowwage_onset, employment_distortion_lowwage_onset, total_diff_employment_lowwage_onset, share_unemployment_reduced_lowwage_onset];
el_and_distortions_values_highwage_onset = [elasticity_highwage_onset, employment_distortion_highwage_onset, total_diff_employment_highwage_onset, share_unemployment_reduced_highwage_onset];

%Collect relevant elasticity and distortion results
el_and_distortions_values_onset = [elasticity_onset, employment_distortion_onset, total_diff_employment_onset, share_unemployment_reduced_onset];
el_and_distortions_values_onlyasseteffect_onset = [elasticity_onlyasseteffect_onset, employment_distortion_onlyasseteffect_onset, total_diff_employment_onlyasseteffect_onset, share_unemployment_reduced_onlyasseteffect_onset];
el_and_distortions_values_inter_time_series_based_onset = [elasticity_inter_time_series_based_onset, employment_distortion_inter_time_series_based_onset, total_diff_employment_inter_time_series_based_onset, share_unemployment_reduced_inter_time_series_based_onset];



% Reset things to make spending figure
newjob_exit_rate_data_2021= newjob_exit_rate_data;
newjob_exit_rate_FPUC_2021 = newjob_exit_rate_FPUC;
newjob_exit_rate_no_FPUC_2021 = newjob_exit_rate_no_FPUC;
newjob_exit_rate_onlyasseteffect_2021 = newjob_exit_rate_onlyasseteffect;
newjob_exit_rate_model_inter_time_series_based_no_FPUC_2021=newjob_exit_rate_model_inter_time_series_based_no_FPUC;
newjob_exit_rate_inter_time_series_based_no_FPUC_2021=newjob_exit_rate_inter_time_series_based_no_FPUC;
newjob_exit_rate_cross_section_based_no_FPUC_2021=newjob_exit_rate_cross_section_based_no_FPUC;
newjob_exit_rate_cross_section_based_logit_no_FPUC_2021=newjob_exit_rate_cross_section_based_logit_no_FPUC;
newjob_exit_rate_FPUC_bywage_2021 = newjob_exit_rate_FPUC_bywage;
newjob_exit_rate_no_FPUC_bywage_2021 = newjob_exit_rate_no_FPUC_bywage;
recall_probs_2021 = recall_probs;
load inf_horizon_het_results_newjob_exit_rate
newjob_exit_rate_data_overall = [newjob_exit_rate_data_2020(1:7); newjob_exit_rate_data_2021(1:end - 7)];
newjob_exit_rate_overall_FPUC = [newjob_exit_rate_FPUC_2020(1:7); newjob_exit_rate_FPUC_2021(1:end - 7)];
newjob_exit_rate_overall_no_FPUC = [newjob_exit_rate_no_FPUC_2020(1:7); newjob_exit_rate_no_FPUC_2021(1:end - 7)];
newjob_exit_rate_onlyasseteffect_overall = [newjob_exit_rate_onlyasseteffect_2020(1:7); newjob_exit_rate_onlyasseteffect_2021(1:end - 7)];
newjob_exit_rate_model_inter_time_series_based_no_FPUC_overall = [newjob_exit_rate_model_inter_time_series_based_no_FPUC_2020(1:7); newjob_exit_rate_model_inter_time_series_based_no_FPUC_2021(1:end-7)];
newjob_exit_rate_inter_time_series_based_no_FPUC_overall = [newjob_exit_rate_inter_time_series_based_no_FPUC_2020(1:7); newjob_exit_rate_inter_time_series_based_no_FPUC_2021(1:end-7)];
newjob_exit_rate_cross_section_based_no_FPUC_overall = [newjob_exit_rate_cross_section_based_no_FPUC_2020(1:7); newjob_exit_rate_cross_section_based_no_FPUC_2021(1:end-7)];
newjob_exit_rate_cross_section_based_logit_no_FPUC_overall = [newjob_exit_rate_cross_section_based_logit_no_FPUC_2020(1:7); newjob_exit_rate_cross_section_based_logit_no_FPUC_2021(1:end-7)];
newjob_exit_rate_overall_FPUC_bywage = [newjob_exit_rate_FPUC_bywage_2020(1:7,:); newjob_exit_rate_FPUC_bywage_2021(1:end - 7,:)];
newjob_exit_rate_overall_no_FPUC_bywage = [newjob_exit_rate_no_FPUC_bywage_2020(1:7,:); newjob_exit_rate_no_FPUC_bywage_2021(1:end - 7,:)];
recall_probs_overall = [recall_probs_2020(1:7); recall_probs_2021(1:end - 7)];

global mean_c_sim_pandemic_surprise_overall_FPUC mean_c_sim_pandemic_surprise_overall_noFPUC mean_c_sim_e_overall
mean_c_sim_e_overall = [mean_c_sim_e_2020(4:11) mean_c_sim_e(3:end - 6)];
mean_c_sim_pandemic_surprise_overall_FPUC = [mean_c_sim_pandemic_surprise_2020(4:12) mean_c_sim_pandemic_expect(3:end - 6)];
mean_c_sim_pandemic_surprise_overall_noFPUC = [mean_c_sim_pandemic_surprise_noFPUC_2020(4:12) mean_c_sim_pandemic_expect_noFPUC(3:end - 6)];
newjob_exit_rate_data_overall(22:end) = newjob_exit_rate_data_overall(22);
newjob_exit_rate_overall_FPUC(22:end) = newjob_exit_rate_overall_FPUC(22);
newjob_exit_rate_overall_no_FPUC(22:end) = newjob_exit_rate_overall_no_FPUC(22);
newjob_exit_rate_onlyasseteffect_overall(22:end) = newjob_exit_rate_onlyasseteffect_overall(22);
newjob_exit_rate_inter_time_series_based_no_FPUC_overall(22:end)=newjob_exit_rate_inter_time_series_based_no_FPUC_overall(22);
newjob_exit_rate_model_inter_time_series_based_no_FPUC_overall(22:end)=newjob_exit_rate_model_inter_time_series_based_no_FPUC_overall(22);
newjob_exit_rate_cross_section_based_no_FPUC_overall(22:end)=newjob_exit_rate_cross_section_based_no_FPUC_overall(22);
newjob_exit_rate_cross_section_based_logit_no_FPUC_overall(22:end)=newjob_exit_rate_cross_section_based_logit_no_FPUC_overall(22);
newjob_exit_rate_overall_FPUC_bywage(22:end,:) = repmat(newjob_exit_rate_overall_FPUC_bywage(22,:),1000-21,1);
newjob_exit_rate_overall_no_FPUC_bywage(22:end,:) = repmat(newjob_exit_rate_overall_no_FPUC_bywage(22,:),1000-21,1);



%2020 expiration
benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
perc_change_benefits_data = benefit_change_data.non_sym_per_change(1);
date_sim_start = datetime(2020, 4, 1);
include_self_employed = 0;

t_distortion_start = 2;
t_distortion_end = 5;
% Covering April 2020 to July 2020
[elasticity_expiration employment_distortion_expiration total_diff_employment_expiration share_unemployment_reduced_expiration employment_FPUC_expiration employment_noFPUC_expiration monthly_spend_pce_expiration monthly_spend_no_FPUC_expiration total_hazard_elasticity_expiration newjob_hazard_elasticity_expiration newjob_duration_elasticity_expiration total_percent_change_expiration,elasticity_prepanlevel_newjob_expiration,elasticity_prepanlevel_both_expiration,elasticity_prepanlevel_recall_expiration,table_elasticity_comparisons_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_2020, newjob_exit_rate_no_FPUC_2020, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
% Save table_elasticity_comparisons_expiration as csv
% writetable(table_elasticity_comparisons_expiration, fullfile(release_path, 'table_elasticity_comparisons_expiration.csv'));
[elasticity_onlyasseteffect_expiration employment_distortion_onlyasseteffect_expiration total_diff_employment_onlyasseteffect_expiration share_unemployment_reduced_onlyasseteffect_expiration employment_FPUC_onlyasseteffect_expiration employment_noFPUC_onlyasseteffect_expiration monthly_spend_pce_onlyasseteffect_expiration monthly_spend_no_FPUC_onlyasseteffect_expiration total_hazard_elasticity_onlyasseteffect_expiration newjob_hazard_elasticity_onlyasseteffect_expiration newjob_duration_elasticity_onlyasseteffect_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_onlyasseteffect_2020, newjob_exit_rate_no_FPUC_2020, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
%employment effects from an interrupted time-series based approach in model
[elasticity_inter_time_series_based_expiration employment_distortion_inter_time_series_based_expiration total_diff_employment_inter_time_series_based_expiration share_unemployment_reduced_inter_time_series_based_expiration employment_FPUC_inter_time_series_based_expiration employment_noFPUC_inter_time_series_based_expiration monthly_spend_pce_inter_time_series_based_expiration monthly_spend_no_FPUC_inter_time_series_based_expiration total_hazard_elasticity_inter_time_series_based_expiration newjob_hazard_elasticity_inter_time_series_based_expiration newjob_duration_elasticity_inter_time_series_based_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_data_2020, newjob_exit_rate_inter_time_series_based_no_FPUC_2020, recall_probs_2020, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
[elasticity_cross_section_based_expiration employment_distortion_cross_section_based_expiration total_diff_employment_cross_section_based_expiration share_unemployment_reduced_cross_section_based_expiration employment_FPUC_cross_section_based_expiration employment_noFPUC_cross_section_based_expiration monthly_spend_pce_cross_section_based_expiration monthly_spend_no_FPUC_cross_section_based_expiration total_hazard_elasticity_cross_section_based_expiration newjob_hazard_elasticity_cross_section_based_expiration newjob_duration_elasticity_cross_section_based_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_data_2020, newjob_exit_rate_cross_section_based_no_FPUC_2020, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);

[elasticity_model_inter_time_series_based_expiration employment_distortion_model_inter_time_series_based_expiration total_diff_employment_model_inter_time_series_based_expiration share_unemployment_reduced_model_inter_time_series_based_expiration employment_FPUC_model_inter_time_series_based_expiration employment_noFPUC_model_inter_time_series_based_expiration monthly_spend_pce_model_inter_time_series_based_expiration monthly_spend_no_FPUC_model_inter_time_series_based_expiration total_hazard_elasticity_model_inter_time_series_based_expiration newjob_hazard_elasticity_model_inter_time_series_based_expiration newjob_duration_elasticity_model_inter_time_series_based_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_2020, newjob_exit_rate_model_inter_time_series_based_no_FPUC_2020, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);




%hazard/duration elasticity specification robustness:
newjob_exit_rate_cross_section_based_no_FPUC_2020_alt1=newjob_exit_rate_cross_section_based_no_FPUC_2020;
newjob_exit_rate_cross_section_based_no_FPUC_2020_alt1(1:4)=newjob_exit_rate_cross_section_based_no_FPUC_2020_alt1(1:4)+.00585;
[elasticity_expiration_alt_1 tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_expiration_alt_1] = elasticity_distortions_and_aggregates(newjob_exit_rate_data_2020, newjob_exit_rate_cross_section_based_no_FPUC_2020_alt1, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
total_hazard_elasticity_cross_section_based_expiration/total_hazard_elasticity_cross_section_based_expiration_alt_1
newjob_exit_rate_cross_section_based_no_FPUC_2020_alt2=newjob_exit_rate_cross_section_based_no_FPUC_2020;
newjob_exit_rate_cross_section_based_no_FPUC_2020_alt2(1:4)=newjob_exit_rate_cross_section_based_no_FPUC_2020_alt2(1:4)-.0088;
[elasticity_expiration_alt_2 tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_expiration_alt_2] = elasticity_distortions_and_aggregates(newjob_exit_rate_data_2020, newjob_exit_rate_cross_section_based_no_FPUC_2020_alt2, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
total_hazard_elasticity_cross_section_based_expiration/total_hazard_elasticity_cross_section_based_expiration_alt_2




[elasticity_cross_section_based_logit_expiration tmp tmp tmp tmp tmp tmp tmp total_hazard_elasticity_cross_section_based_logit_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_data_2020, newjob_exit_rate_cross_section_based_logit_no_FPUC_2020, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);


table_alt_job_find_specs=table();
table_alt_job_find_specs.Hazard_600('Baseline: Absolute pp')=total_hazard_elasticity_cross_section_based_expiration;
table_alt_job_find_specs.Hazard_600('Relative % change')=total_hazard_elasticity_cross_section_based_expiration_alt_2;
table_alt_job_find_specs.Hazard_600('Logit')=total_hazard_elasticity_cross_section_based_logit_expiration;
table_alt_job_find_specs.Hazard_300('Baseline: Absolute pp')=total_hazard_elasticity_cross_section_based_onset;
table_alt_job_find_specs.Hazard_300('Relative % change')=total_hazard_elasticity_cross_section_based_onset_alt_2;
table_alt_job_find_specs.Hazard_300('Logit')=total_hazard_elasticity_cross_section_based_logit_onset;
table_alt_job_find_specs.Duration_elasticity_600('Baseline: Absolute pp')=elasticity_cross_section_based_expiration;
table_alt_job_find_specs.Duration_elasticity_600('Relative % change')=elasticity_expiration_alt_2;
table_alt_job_find_specs.Duration_elasticity_600('Logit')=elasticity_cross_section_based_logit_expiration;
table_alt_job_find_specs.Duration_elasticity_300('Baseline: Absolute pp')=elasticity_cross_section_based_onset;
table_alt_job_find_specs.Duration_elasticity_300('Relative % change')=elasticity_alt_2;
table_alt_job_find_specs.Duration_elasticity_300('Logit')=elasticity_cross_section_based_logit_onset;
writetable(table_alt_job_find_specs,fullfile(release_path_paper,'table_alt_job_find_specs.csv'),'WriteRowNames',true);

change_checking_just_u=(checking_u(7)-checking_u(3));
newjob_exit_rate_no_FPUC_noliquidity=newjob_exit_rate_no_FPUC_2020+(2103/4868*.0021)*4.5;
[elasticity_noliquidity employment_distortion_noliquidity total_diff_employment_noliquidity share_unemployment_reduced_noliquidity employment_FPUC_noliquidity employment_noFPUC_noliquidity monthly_spend_pce_noliquidity monthly_spend_no_FPUC_noliquidity total_hazard_elasticity_noliquidity newjob_hazard_elasticity_noliquidity newjob_duration_elasticity_noliquidity total_percent_change_noliquidity,elasticity_prepanlevel_newjob_noliquidity,elasticity_prepanlevel_both_noliquidity,elasticity_prepanlevel_recall_noliquidity,table_elasticity_comparisons_noliquidity] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_2020, newjob_exit_rate_no_FPUC_noliquidity, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
total_hazard_elasticity_expiration;
total_hazard_elasticity_noliquidity;
stat_for_text_total_hazard_elasticity_no_liquidity=total_hazard_elasticity_noliquidity;
save('stats_for_text_model_miscellaneous.mat', 'stat_for_text_total_hazard_elasticity_no_liquidity', '-append')


%Collect relevant elasticity and distortion results
el_and_distortions_values_expiration = [elasticity_expiration, employment_distortion_expiration, total_diff_employment_expiration, share_unemployment_reduced_expiration];
el_and_distortions_values_onlyasseteffect_expiration = [elasticity_onlyasseteffect_expiration, employment_distortion_onlyasseteffect_expiration, total_diff_employment_onlyasseteffect_expiration, share_unemployment_reduced_onlyasseteffect_expiration];
el_and_distortions_values_inter_time_series_based_expiration = [elasticity_inter_time_series_based_expiration, employment_distortion_inter_time_series_based_expiration, total_diff_employment_inter_time_series_based_expiration, share_unemployment_reduced_inter_time_series_based_expiration];

perc_change_benefits_data=NaN;
[elasticity_lowwage_expiration employment_distortion_lowwage_expiration total_diff_employment_lowwage_expiration share_unemployment_reduced_lowwage_expiration employment_FPUC_lowwage_expiration employment_noFPUC_lowwage_expiration monthly_spend_pce_lowwage_expiration monthly_spend_no_FPUC_lowwage_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_bywage_2020(:,1), newjob_exit_rate_no_FPUC_bywage_2020(:,1), recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
[elasticity_highwage_expiration employment_distortion_highwage_expiration total_diff_employment_highwage_expiration share_unemployment_reduced_highwage_expiration employment_FPUC_highwage_expiration employment_noFPUC_highwage_expiration monthly_spend_pce_highwage_expiration monthly_spend_no_FPUC_highwage_expiration] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_bywage_2020(:,5), newjob_exit_rate_no_FPUC_bywage_2020(:,5), recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);

%Collect relevant elasticity and distortion results
el_and_distortions_values_lowwage_expiration = [elasticity_lowwage_expiration, employment_distortion_lowwage_expiration, total_diff_employment_lowwage_expiration, share_unemployment_reduced_lowwage_expiration]
el_and_distortions_values_highwage_expiration = [elasticity_highwage_expiration, employment_distortion_highwage_expiration, total_diff_employment_highwage_expiration, share_unemployment_reduced_highwage_expiration]

display('Liquidity shares of distortions expiration')
liquidity_share_expiration=elasticity_onlyasseteffect_expiration/elasticity_expiration
employment_distortion_onlyasseteffect_expiration/employment_distortion_expiration
inter_model_share_expiration=elasticity_model_inter_time_series_based_expiration/elasticity_expiration
employment_distortion_inter_time_series_based_expiration/employment_distortion_expiration



employment_distortion_prepandemic_expiration=elasticity_and_distortions_values_prepandemic(2);
employment_distortion_prepandemic_onset=elasticity_and_distortions_values_prepandemic_onset(2);

table_employment=table();
table_employment.Best_Fit_Model('Flow % Decline in Employment: 600 alone')=employment_distortion_expiration;
table_employment.Prepandemic_Model('Flow % Decline in Employment: 600 alone')=elasticity_and_distortions_values_prepandemic(2);
table_employment.Best_Fit_Model('Flow % Decline in Employment: 300 alone')=employment_distortion_onset;
table_employment.Prepandemic_Model('Flow % Decline in Employment: 300 alone')=elasticity_and_distortions_values_prepandemic_onset(2);
table_employment.Best_Fit_Model('Flow % Decline in Employment: Overall Effects')=employment_distortion_expiration*4/(4+8)+employment_distortion_onset*8/(4+8);
table_employment.Prepandemic_Model('Flow % Decline in Employment: Overall Effects')=elasticity_and_distortions_values_prepandemic(2)*4/(4+8)+elasticity_and_distortions_values_prepandemic_onset(2)*8/(4+8)
%
table_employment.Best_Fit_Model('Annual % Decline in Employment: 600 alone')=employment_distortion_expiration*4/12;
table_employment.Prepandemic_Model('Annual % Decline in Employment: 600 alone')=elasticity_and_distortions_values_prepandemic(2)*4/12;
table_employment.Best_Fit_Model('Annual % Decline in Employment: 300 alone')=employment_distortion_onset*8/12;
table_employment.Prepandemic_Model('Annual % Decline in Employment: 300 alone')=elasticity_and_distortions_values_prepandemic_onset(2)*8/12;
table_employment.Best_Fit_Model('Annual % Decline in Employment: Overall Effects')=employment_distortion_expiration*4/(4+8)+employment_distortion_onset*8/(4+8);
table_employment.Prepandemic_Model('Annual % Decline in Employment: Overall Effects')=elasticity_and_distortions_values_prepandemic(2)*4/(4+8)+elasticity_and_distortions_values_prepandemic_onset(2)*8/(4+8)
%
table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 600 alone')=total_diff_employment_expiration;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 600 alone')=elasticity_and_distortions_values_prepandemic(3);
table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 300 alone')=total_diff_employment_onset;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 300 alone')=elasticity_and_distortions_values_prepandemic_onset(3);
table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): Overall Effects')=total_diff_employment_expiration+total_diff_employment_onset;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): Overall Effects')=elasticity_and_distortions_values_prepandemic(3)+elasticity_and_distortions_values_prepandemic_onset(3)




table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 600 alone')=total_diff_employment_expiration;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 600 alone')=elasticity_and_distortions_values_prepandemic(3);
table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 300 alone')=total_diff_employment_onset;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): 300 alone')=elasticity_and_distortions_values_prepandemic_onset(3);
table_employment.Best_Fit_Model('Cumulative Reduction in Jobs at Supplement End (Millions): Overall Effects')=total_diff_employment_expiration+total_diff_employment_onset;
table_employment.Prepandemic_Model('Cumulative Reduction in Jobs at Supplement End (Millions): Overall Effects')=elasticity_and_distortions_values_prepandemic(3)+elasticity_and_distortions_values_prepandemic_onset(3)







%full run:
t_distortion_start = 2;
t_distortion_end = 18;
[elasticity_full employment_distortion_full total_diff_employment_full share_unemployment_reduced_full employment_FPUC_full employment_noFPUC_full monthly_spend_pce_full monthly_spend_no_FPUC_full total_hazard_elasticity_full newjob_hazard_elasticity_full newjob_duration_elasticity_full total_percent_change_full] = elasticity_distortions_and_aggregates(newjob_exit_rate_overall_FPUC, newjob_exit_rate_overall_no_FPUC, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
[elasticity_onlyasseteffect_full employment_distortion_onlyasseteffect_full total_diff_employment_onlyasseteffect_full share_unemployment_reduced_onlyasseteffect_full employment_FPUC_onlyasseteffect_full employment_noFPUC_onlyasseteffect_full monthly_spend_pce_onlyasseteffect_full monthly_spend_no_FPUC_onlyasseteffect_full] = elasticity_distortions_and_aggregates(newjob_exit_rate_onlyasseteffect_overall, newjob_exit_rate_overall_no_FPUC, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
%employment effects from an interrupted time-series based approach in model
[elasticity_inter_time_series_based_full employment_distortion_inter_time_series_based_full total_diff_employment_inter_time_series_based_full share_unemployment_reduced_inter_time_series_based_full employment_FPUC_inter_time_series_based_full employment_noFPUC_inter_time_series_based_full monthly_spend_pce_inter_time_series_based_full monthly_spend_no_FPUC_inter_time_series_based_full] = elasticity_distortions_and_aggregates(newjob_exit_rate_overall_FPUC, newjob_exit_rate_inter_time_series_based_no_FPUC_overall, recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

%Collect relevant elasticity and distortion results
el_and_distortions_values_full = [elasticity_full, employment_distortion_full, total_diff_employment_full, share_unemployment_reduced_full]
el_and_distortions_values_onlyasseteffect_full = [elasticity_onlyasseteffect_full, employment_distortion_onlyasseteffect_full, total_diff_employment_onlyasseteffect_full, share_unemployment_reduced_onlyasseteffect_full];
el_and_distortions_values_inter_time_series_based_full = [elasticity_inter_time_series_based_full, employment_distortion_inter_time_series_based_full, total_diff_employment_inter_time_series_based_full, share_unemployment_reduced_inter_time_series_based_full];

display('Liquidity shares of distortions overall')
elasticity_onlyasseteffect_full/elasticity_full
employment_distortion_onlyasseteffect_full/employment_distortion_full
elasticity_inter_time_series_based_full/elasticity_full
employment_distortion_inter_time_series_based_full/employment_distortion_full


perc_change_benefits_data=NaN;
%Distributional results, overall time series:
[elasticity_lowwage_overall employment_distortion_lowwage_overall total_diff_employment_lowwage_overall share_unemployment_reduced_lowwage_overall employment_FPUC_lowwage_overall employment_noFPUC_lowwage_overall monthly_spend_pce_lowwage_overall monthly_spend_no_FPUC_lowwage_overall] = elasticity_distortions_and_aggregates(newjob_exit_rate_overall_FPUC_bywage(:,1), newjob_exit_rate_overall_no_FPUC_bywage(:,1), recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);
[elasticity_highwage_overall employment_distortion_highwage_overall total_diff_employment_highwage_overall share_unemployment_reduced_highwage_overall employment_FPUC_highwage_overall employment_noFPUC_highwage_overall monthly_spend_pce_highwage_overall monthly_spend_no_FPUC_highwage_overall] = elasticity_distortions_and_aggregates(newjob_exit_rate_overall_FPUC_bywage(:,5), newjob_exit_rate_overall_no_FPUC_bywage(:,5), recall_probs_overall, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);

%Collect relevant elasticity and distortion results
el_and_distortions_values_lowwage_full = [elasticity_lowwage_overall, employment_distortion_lowwage_overall, total_diff_employment_lowwage_overall, share_unemployment_reduced_lowwage_overall]
el_and_distortions_values_highwage_full = [elasticity_highwage_overall, employment_distortion_highwage_overall, total_diff_employment_highwage_overall, share_unemployment_reduced_highwage_overall]



t=32:43;
t=t';
reg_emp = mvregress([ones(12,1) t], employment_FPUC_full(32:43));
t=44:61;
employment_no_pandemic(1:43)=employment_FPUC_full(1:43);
employment_no_pandemic(44:61)=t*reg_emp(2)+reg_emp(1);
employment_no_pandemic=employment_no_pandemic';

gap_employment_percent=employment_FPUC_full(45:61)./employment_no_pandemic(45:61)-1;
share_gap_600=-employment_distortion_expiration/mean(gap_employment_percent(1:4))
share_gap_300=-employment_distortion_onset/mean(gap_employment_percent(10:17))

share_gap_600_prepandemic=employment_distortion_prepandemic_expiration/mean(gap_employment_percent(1:4))
share_gap_300_prepandemic=employment_distortion_prepandemic_onset/mean(gap_employment_percent(10:17))




% Making figures

% ALSO make a version with percentage difference from Feb 2020
% Feb 2020 will be period 43
% Using same Employment in pd 43 as the base in FPUC and not calculations because indices are confusing...
bls_emp_long_employment_perc_diff = ((employment_FPUC_full - employment_FPUC_full(43)) ./ employment_FPUC_full(43)) * 100;
bls_emp_long_employment_noFPUC_perc_diff = ((employment_noFPUC_full - employment_FPUC_full(43)) ./ employment_FPUC_full(43)) * 100;

y_lim_lower_common2 = -20;
y_lim_upper_common2 = 5;

figure
p = patch([45 45 48 48], [y_lim_lower_common2 y_lim_upper_common2 y_lim_upper_common2 y_lim_lower_common2], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
p = patch([54 54 61 61], [y_lim_lower_common2 y_lim_upper_common2 y_lim_upper_common2 y_lim_lower_common2], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
plot(35:61, bls_emp_long_employment_perc_diff(35:61), '--', 'Color', qual_blue, 'LineWidth', 2)
plot(45:61, bls_emp_long_employment_noFPUC_perc_diff(45:61), 'Color', qual_yellow, 'LineWidth', 2)
xticks([36 42 48 54 60])
xticklabels({'July 19', 'Jan 20', 'July 20', 'Jan 21', 'July 21'})
xlim([35 61])
ylim([y_lim_lower_common2 y_lim_upper_common2])
legend('Actual', 'Without UI supplements', 'Location', 'SouthEast', 'FontSize', 14)
ylabel('Percent change in employment (relative to Feb 2020)')
set(gca, 'TickDir', 'out')
set(gca, 'Layer', 'top');
set(gca, 'fontsize', 11);
fig_paper_10a = gcf;
saveas(fig_paper_10a, fullfile(release_path_paper, 'employment_stock_full_perc_diff_feb_2020.png'))
%saveas(fig_paper_10a, fullfile(release_path_slides, 'employment_stock_full_perc_diff_feb_2020.png'))

include_self_employed = 1;
[var_1 var_2 var_3 var_4 var_5 var_6 monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);

mean_spend_increase_FPUC = (mean(monthly_spend_pce(45:61) ./ monthly_spend_no_FPUC(45:61)) - 1) * 100
mean_spend_increase_throughJuly_FPUC = (mean(monthly_spend_pce(45:48) ./ monthly_spend_no_FPUC(45:48)) - 1) * 100
mean_spend_increase_JantoAug_FPUC=(mean(monthly_spend_pce(54:61) ./ monthly_spend_no_FPUC(54:61)) - 1) * 100


t=32:43;
t=t';
reg_spend = mvregress([ones(12,1) t], monthly_spend_pce(32:43));
monthly_spend_pce_no_pandemic(1:43)=monthly_spend_pce(1:43);
t=44:61;
monthly_spend_pce_no_pandemic(44:61)=t*reg_spend(2)+reg_spend(1);
monthly_spend_pce_no_pandemic=monthly_spend_pce_no_pandemic';

gap_spend_percent=monthly_spend_no_FPUC(44:61)./monthly_spend_pce_no_pandemic(44:61)-1;
share_gap_spend_600=-mean_spend_increase_throughJuly_FPUC/mean(gap_spend_percent(1:4))
share_gap_spend_300=-mean_spend_increase_JantoAug_FPUC/mean(gap_spend_percent(10:17))


% ALSO make a version with percentage difference from Feb 2020
% Feb 2020 will be period 43
monthly_spend_pce_perc_diff = ((monthly_spend_pce - monthly_spend_pce(43)) ./ monthly_spend_pce(43)) * 100;
monthly_spend_no_FPUC_perc_diff = ((monthly_spend_no_FPUC - monthly_spend_no_FPUC(43)) ./ monthly_spend_no_FPUC(43)) * 100;

y_lim_lower_common = -30;
y_lim_upper_common = 10;

figure
p = patch([45 45 48 48], [y_lim_lower_common y_lim_upper_common y_lim_upper_common y_lim_lower_common], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
p = patch([54 54 61 61], [y_lim_lower_common y_lim_upper_common y_lim_upper_common y_lim_lower_common], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
plot(35:61, monthly_spend_pce_perc_diff(35:61), '--', 'Color', qual_blue, 'LineWidth', 2)
plot(45:61, monthly_spend_no_FPUC_perc_diff(45:61), 'Color', qual_yellow, 'LineWidth', 2)
xticks([36 42 48 54 60])
xticklabels({'July 19', 'Jan 20', 'July 20', 'Jan 21', 'July 21'})
xlim([35 61])
ylim([y_lim_lower_common y_lim_upper_common])
legend('Actual', 'Without UI supplements', 'Location', 'SouthEast', 'FontSize', 14)
ylabel('Percent change in agg. spending (relative to Feb 2020)')
set(gca, 'fontsize', 11);
set(gca, 'Layer', 'top');
set(gca, 'TickDir', 'out')
fig_paper_10b = gcf;
saveas(fig_paper_10b, fullfile(release_path_paper, 'aggspend_full_perc_diff_feb_2020.png'))
%saveas(fig_paper_10b, fullfile(release_path_slides, 'aggspend_full_perc_diff_feb_2020.png'))

save('spend_and_search_overall.mat', 'mean_c_sim_pandemic_surprise_overall_FPUC', 'mean_c_sim_pandemic_surprise_overall_noFPUC', 'newjob_exit_rate_overall_FPUC', 'newjob_exit_rate_overall_no_FPUC', 'recall_probs');




table_elasticities=table();
table_elasticities.supplement600('Best Fit Model')=elasticity_expiration;
table_elasticities.supplement300('Best Fit Model')=elasticity_onset;
table_elasticities.supplement600('Time-Series Based Statistical Model')=elasticity_inter_time_series_based_expiration;
table_elasticities.supplement300('Time-Series Based Statistical Model')=elasticity_inter_time_series_based_onset;
table_elasticities.supplement600('Cross-Section Based Statistical Model')=elasticity_cross_section_based_expiration;
table_elasticities.supplement300('Cross-Section Based Statistical Model')=elasticity_cross_section_based_onset
% writetable(table_elasticities,fullfile(release_path,'table_elasticities.csv'),'WriteRowNames',true);

table_agg_effects=table();
table_agg_effects.Supp600_Apr_2020_Through_July_2020('Percent Change In Aggregate Employment During Supplement Period')=-employment_distortion_expiration;
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021('Percent Change In Aggregate Employment During Supplement Period')=-employment_distortion_onset;
table_agg_effects.Supp600_Apr_2020_Through_July_2020('Percent of Pandemic Gap in Employment Explained During Supplement Period')=share_gap_600;
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021('Percent of Pandemic Gap in Employment Explained During Supplement Period')=share_gap_300;
table_agg_effects.Supp600_Apr_2020_Through_July_2020('Percent Change In Aggregate Spending During Supplement Period')=mean_spend_increase_throughJuly_FPUC;
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021('Percent Change In Aggregate Spending During Supplement Period')=mean_spend_increase_JantoAug_FPUC;
table_agg_effects.Supp600_Apr_2020_Through_July_2020('Percent of Pandemic Gap in Aggregate Spending Explained During Supplement Period')=share_gap_spend_600;
table_agg_effects.Supp300_Jan_2021_Through_Aug_2021('Percent of Pandemic Gap in Aggregate Spending Explained During Supplement Period')=share_gap_spend_300
writetable(table_agg_effects,fullfile(release_path_paper,'table_agg_effects.csv'),'WriteRowNames',true);


table_distortion_stats_for_text=table();
table_distortion_stats_for_text.statistic('Decline in Employment Apr-July 2020 (millions)')=total_diff_employment_expiration;
table_distortion_stats_for_text.statistic('Decline in Employment Jan-Aug 2021 (millions)')=total_diff_employment_onset;
table_distortion_stats_for_text.statistic('Percent Change Employment 600 annualized')=-employment_distortion_expiration*4/12;
table_distortion_stats_for_text.statistic('Percent Change Employment 300 annualized')=-employment_distortion_onset*8/12;
table_distortion_stats_for_text.statistic('Percent Change Employment 600 (prepandemic calibration)')=employment_distortion_prepandemic_expiration;
table_distortion_stats_for_text.statistic('Percent Change Employment 300 (prepandemic calibration)')=employment_distortion_prepandemic_onset
writetable(table_distortion_stats_for_text,fullfile(release_path_paper,'table_distortion_stats_for_text.csv'),'WriteRowNames',true);


table_elasticity_comparisons=table();
table_elasticity_comparisons.Supp600('Duration_Elasticity')=elasticity_expiration;
table_elasticity_comparisons.Supp300('Duration_Elasticity')=elasticity_onset;
table_elasticity_comparisons.Supp600('Duration_Elasticity_PrePandemicNewJobRate')=elasticity_prepanlevel_newjob_expiration;
table_elasticity_comparisons.Supp300('Duration_Elasticity_PrePandemicNewJobRate')=elasticity_prepanlevel_newjob_onset;
table_elasticity_comparisons.Supp600('Duration_Elasticity_PrePandemicRecallRate')=elasticity_prepanlevel_recall_expiration;
table_elasticity_comparisons.Supp300('Duration_Elasticity_PrePandemicRecallRate')=elasticity_prepanlevel_recall_onset;
table_elasticity_comparisons.Supp600('Duration_Elasticity_Both')=elasticity_prepanlevel_both_expiration;
table_elasticity_comparisons.Supp300('Duration_Elasticity_Both')=elasticity_prepanlevel_both_onset
% writetable(table_elasticity_comparisons,fullfile(release_path,'table_elasticity_comparisons.csv'),'WriteRowNames',true);



table_elasticity_dynamics_model=table()
table_elasticity_dynamics_model.Liquidity_Share('600')=liquidity_share_expiration;
table_elasticity_dynamics_model.Liquidity_Share('300')=liquidity_share_onset;
table_elasticity_dynamics_model.Inter_TimeSeries_Share('600')=inter_model_share_expiration;
table_elasticity_dynamics_model.Inter_TimeSeries_Share('300')=inter_model_share_onset
% writetable(table_elasticity_dynamics_model,fullfile(release_path,'table_elasticity_dynamics_model.csv'),'WriteRowNames',true);




% Prepandemic spend material

% Reload the consumption series
load mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021
load mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020
load prepandemic_newjob_2021
load prepandemic_newjob_2020

newjob_exit_rate_overall_prepandemic_FPUC = [newjob_exit_rate_prepandemic_FPUC_2020(1:7); newjob_exit_rate_prepandemic_FPUC_2021(1:end - 7)];
newjob_exit_rate_overall_prepandemic_no_FPUC = [newjob_exit_rate_prepandemic_no_FPUC_2020(1:7); newjob_exit_rate_prepandemic_no_FPUC_2021(1:end - 7)];

mean_c_sim_prepandemic_expect_overall_FPUC = [mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020(4:12, 1); mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021(3:end - 6, 1)]';
mean_c_sim_prepandemic_expect_overall_noFPUC = [mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020(4:12, 2); mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021(3:end - 6, 2)]';

% Other than consumption series keep other parameters the same
include_self_employed = 1;
[var_1 var_2 var_3 var_4 var_5 var_6 monthly_spend_pce_prepandemic monthly_spend_no_FPUC_prepandemic] = elasticity_distortions_and_aggregates(newjob_exit_rate_overall_prepandemic_FPUC, newjob_exit_rate_overall_prepandemic_no_FPUC, recall_probs, mean_c_sim_prepandemic_expect_overall_FPUC, mean_c_sim_prepandemic_expect_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_distortion_start, t_distortion_end, include_self_employed);

% Calculations of increases
mean_spend_increase_FPUC_prepandemic_calibration = (mean(monthly_spend_pce_prepandemic(45:61) ./ monthly_spend_no_FPUC_prepandemic(45:61)) - 1) * 100
mean_spend_increase_throughJuly_FPUC_prepandemic_calibration = (mean(monthly_spend_pce_prepandemic(45:48) ./ monthly_spend_no_FPUC_prepandemic(45:48)) - 1) * 100

diff_spend = mean_c_sim_pandemic_expect - mean_c_sim_pandemic_expect_noFPUC;

distortion_surprise = mean_search_sim_pandemic_surprise_noFPUC - mean_search_sim_pandemic_surprise;
distortion_expect = mean_search_sim_pandemic_expect_noFPUC - mean_search_sim_pandemic_expect;
distortion_expect_jan_start = mean_search_sim_pandemic_expect_jan_start_noFPUC - mean_search_sim_pandemic_expect_jan_start;
distortion_surprise = distortion_surprise / distortion_surprise(3);
distortion_expect = distortion_expect / distortion_expect(3);
distortion_expect_jan_start = distortion_expect_jan_start/ distortion_expect_jan_start(3);

%import data diff-in-diff:
data_DiD_raw = readtable(jobfind_input_directory, 'Sheet', fig_a12b_onset);
%compute 95% CIs
data_DiD_raw.high_ci = data_DiD_raw.estimate + norminv(0.95) .* data_DiD_raw.std_error;
data_DiD_raw.low_ci = data_DiD_raw.estimate + norminv(0.05) .* data_DiD_raw.std_error;
%re-order variables 
data_DiD_raw = movevars(data_DiD_raw, 'high_ci', 'After', 'estimate');
data_DiD_raw = movevars(data_DiD_raw, 'low_ci', 'After', 'high_ci');

data_DiD_raw = table2array(data_DiD_raw(:,2:end));
data_DiD_2021 = data_DiD_raw(end-8:end, 1:end-1);
data_DiD_2021_normalized = data_DiD_2021 / data_DiD_2021(3,1);
%data_DiD_2021_normalized_corrected = data_DiD_2021 / data_DiD_2021(1,1);
weekly_index_2021 = 3.25:.25:4.75;


data_DiD_full = data_DiD_raw(:, 1:end-1);
data_DiD_full_normalized = data_DiD_full / data_DiD_full(10,1);
%divide by data_DiD_full(12,1) for consistency with DiD_2021 plot
weekly_index_full = [[1:.2:2],[2.25:.25:2.75],weekly_index_2021, [5 5.25]];


% Save the job search series
writetable(table(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC), 'inf_horizon_het_results_onset_newjob_exit_rate.xlsx', 'Sheet', 'New Job Exit Rate', 'WriteVariableNames', true);







figure
tiledlayout(1,2)
nexttile
p = patch([3 3 4 4], [0.001 .5 .5 0.001], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
title('Job-Finding Rate','FontWeight','normal','FontSize',18)
plot(1:4, monthly_search_data(1:4), '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
plot(1:4, mean_search_sim_prepandemic_expect(1:4), '-+', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
plot(1:4, mean_search_sim_pandemic_expect_jan_start_match500MPC(1:4), '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
%slightly shifting line just to make it visible:
plot(1:4, mean_search_sim_pandemic_expect_match500MPC(1:4)+.0004, '-v', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(1:4, mean_search_sim_pandemic_expect(1:4), '-s', 'Color', qual_green, 'MarkerFaceColor', qual_green, 'LineWidth', 2)
%legend('Data', 'Pre-pandemic model', 'Perfect foresight + $500 MPC', 'Surprise start + $500 MPC', 'Surprise start + waiting MPC', 'Location', 'NorthEast', 'FontSize', 14)
% title('Monthly Search')
ylim([0 0.5])
xticks([1 2 3 4])
xticklabels(label_months_nov20_feb21)
set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
set(gca, 'fontsize', 12);
set(gca, 'Layer', 'top');
nexttile
p = patch([3 3 4 4], [-20 10 10 -20], [0.9 0.9 0.9], 'EdgeColor', 'none');
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
hold on
%plot(1:11,-10*ones(1:11,1),'w')
title('Spending (U vs. E % Change from Jan 20)','FontWeight','normal','FontSize',18)
plot(1:4, s4, '-+', 'Color', qual_purple, 'MarkerFaceColor', qual_purple, 'LineWidth', 2)
plot(1:4, s1, '-d', 'Color', matlab_red_orange, 'MarkerFaceColor', matlab_red_orange, 'LineWidth', 2)
plot(1:4, s2, '-v', 'Color', qual_orange, 'MarkerFaceColor', qual_orange, 'LineWidth', 2)
plot(1:4, s3, '-s', 'Color', qual_green, 'MarkerFaceColor', qual_green, 'LineWidth', 2)
plot(1:4, spend_data, '--o', 'Color', qual_blue, 'MarkerFaceColor', qual_blue, 'LineWidth', 2)
%plot(1:11, mean_c_sim_pandemic_surprise_noFPUC_dollars(1:11)/total_spend_u(1)*100, '-o', 'Color', [0.6 0.6 0.6], 'MarkerFaceColor', [0.6 0.6 0.6], 'LineWidth', 2)
%ylabel('\Delta $')
ylim([-10 10])
yticks([-10 -5 0 5 10])
yticklabels({'-10%', '-5%', '0%', '5%', '10%'})
xticks([1 2 3 4])
xticklabels(label_months_nov20_feb21)
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
lgd=legend('Standard Model w/ Pre-Pandemic Search Costs','Pandemic Search Costs + Perfect Foresight', 'Pandemic Search Costs + Myopic Expectations', 'Pandemic Search Costs + Myopic Expectations + High Impatience','Data', 'FontSize', 12);
%title(lgd,'Changes From Pre-Pandemic Model:','FontSize',12,'FontWeight','normal')
lgd.Layout.Tile = 'South';
set(gcf, 'PaperPosition', [0 0 13 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [13 5]); %Keep the same paper size
fig_paper_A19 = gcf;
saveas(fig_paper_A19, fullfile(release_path_paper, 'spend_and_search_onset.png'))
%saveas(fig_paper_A19, fullfile(release_path_slides, 'spend_and_search_onset.png'))



expiry = readtable(jobfind_input_directory, 'Sheet', onset_sample);
idx = datenum(expiry.week_start_date) >= datenum('2020-11-01') & datenum(expiry.week_start_date) < datenum('2021-03-21');
extra_sample_end_extension = 0;

expiry = expiry(idx, :);

% Convert expiry variables to a monthly level
% Apply week_to_month_exit.m to grouped, sorted monthly data
% Sort by type, cut, and week start date
expiry = sortrows(expiry, {'type', 'cut', 'week_start_date'});
expiry.month = dateshift(datetime(expiry.week_start_date), 'start', 'month');
% Need to convert NaN values of cut to something else
expiry.cut(isnan(expiry.cut)) = 0;

% For the exit variables we want the average exit probability at a monthly level
expiry_week_to_month_exit = varfun(@week_to_month_exit, expiry, 'InputVariables', {'exit_ui_rate', 'exit_to_recall', 'exit_not_to_recall'}, 'GroupingVariables', {'month', 'cut', 'type'});
expiry_week_to_month_exit = renamevars(expiry_week_to_month_exit, ["week_to_month_exit_exit_ui_rate", "week_to_month_exit_exit_to_recall", "week_to_month_exit_exit_not_to_recall"], ["exit_ui_rate", "exit_to_recall", "exit_not_to_recall"]);
% For the per_change variable we want the average per_change at a monthly level
expiry_per_change = varfun(@mean, expiry, 'InputVariables', {'per_change'}, 'GroupingVariables', {'month', 'cut', 'type'});
expiry_per_change = renamevars(expiry_per_change, ["mean_per_change"], ["per_change"]);
% Expiry will be combined version of these
expiry = innerjoin(expiry_week_to_month_exit, expiry_per_change);

idx = (string(expiry.type) == 'By rep rate quintile');
expiry_by_wage_quintiles = expiry(idx, :);
%expiry_by_wage_quintiles.cut=str2double(expiry_by_wage_quintiles.cut);
expiry_by_wage_quintiles = sortrows(expiry_by_wage_quintiles, 'cut');

change_rep_rate_expiry_quintiles = grpstats(expiry_by_wage_quintiles, 'cut', 'mean', 'DataVars', {'per_change'});
change_rep_rate_expiry_quintiles = change_rep_rate_expiry_quintiles.mean_per_change;
change_rep_rate_expiry_quintiles = sort(change_rep_rate_expiry_quintiles, 'Descend');

data_change_weekly = [0.000341671
-0.000403977
-0.000906744
-0.001147928
-0.003128249
-0.003579654
-0.004735044
-0.010840462
-0.007795702
-0.01212503

                ];

data_per_change = [0.402544486
0.457710961
0.46842071
0.491563099
0.524006054
0.584708008
0.660957597
0.741012231
0.829629212
1.074702666
            ];

surprise_period = 3;
% Change window of 8 weeks to 2 months
mean_search_sim_pandemic_expect_bywage_weekly = 1 - (1 - mean_search_sim_pandemic_expect_bywage).^(1/4.25);
diff_exit(1) = mean(mean_search_sim_pandemic_expect_bywage_weekly(1, surprise_period:surprise_period + 1)) - mean(mean_search_sim_pandemic_expect_bywage_weekly(1, surprise_period - 2:surprise_period - 1));
diff_exit(2) = mean(mean_search_sim_pandemic_expect_bywage_weekly(2, surprise_period:surprise_period + 1)) - mean(mean_search_sim_pandemic_expect_bywage_weekly(2, surprise_period - 2:surprise_period - 1));
diff_exit(3) = mean(mean_search_sim_pandemic_expect_bywage_weekly(3, surprise_period:surprise_period + 1)) - mean(mean_search_sim_pandemic_expect_bywage_weekly(3, surprise_period - 2:surprise_period - 1));
diff_exit(4) = mean(mean_search_sim_pandemic_expect_bywage_weekly(4, surprise_period:surprise_period + 1)) - mean(mean_search_sim_pandemic_expect_bywage_weekly(4, surprise_period - 2:surprise_period - 1));
diff_exit(5) = mean(mean_search_sim_pandemic_expect_bywage_weekly(5, surprise_period:surprise_period + 1)) - mean(mean_search_sim_pandemic_expect_bywage_weekly(5, surprise_period - 2:surprise_period - 1));

shift = (mean(data_change_weekly) - mean(diff_exit));
diff_exit = diff_exit + shift;

reg_input_col4 = mean_search_sim_pandemic_expect_bywage_weekly';
reg_input_col3 = zeros(size(reg_input_col4));
reg_input_col3(3:end, :) = 1;
reg_input_col2 = zeros(size(reg_input_col4));
reg_input_col1 = zeros(size(reg_input_col4));

for i = 1:5
    reg_input_col2(:, i) = change_rep_rate_expiry_quintiles(i);
    reg_input_col1(:, i) = reg_input_col2(:, i) .* reg_input_col3(:, i);
end

reg_input_col1 = reg_input_col1(1:4, :);
reg_input_col2 = reg_input_col2(1:4, :);
reg_input_col3 = reg_input_col3(1:4, :);
reg_input_col4 = reg_input_col4(1:4, :);

reg_input_col1 = reshape(reg_input_col1, 5 * 4, 1);
reg_input_col2 = reshape(reg_input_col2, 5 * 4, 1);
reg_input_col3 = reshape(reg_input_col3, 5 * 4, 1);
reg_input_col4 = reshape(reg_input_col4, 5 * 4, 1);

reg_input = table();
reg_input.postXperchange_rep_rate = reg_input_col1;
reg_input.perchange_rep_rate = reg_input_col2;
reg_input.post = reg_input_col3;
reg_input.exit_rate = reg_input_col4;

reg_newjob_exit = fitlm(reg_input, 'linear');

mpc_supplements_onset = table();
mpc_supplements_onset.expect('one_month') = (mean_c_sim_pandemic_expect(3) - mean_c_sim_pandemic_expect_noFPUC(3)) / (mean_y_sim_pandemic_u(3) - mean_y_sim_pandemic_noFPUC(3));
mpc_supplements_onset.expect('3_month') = sum(mean_c_sim_pandemic_expect(3:5) - mean_c_sim_pandemic_expect_noFPUC(3:5)) / sum(mean_y_sim_pandemic_u(3:5) - mean_y_sim_pandemic_noFPUC(3:5));
mpc_supplements_onset.expect('6_month') = sum(mean_c_sim_pandemic_expect(3:8) - mean_c_sim_pandemic_expect_noFPUC(3:8)) / sum(mean_y_sim_pandemic_u(3:8) - mean_y_sim_pandemic_noFPUC(3:8))
mpc_supplements_onset.expect('full') = sum(mean_c_sim_pandemic_expect(3:10) - mean_c_sim_pandemic_expect_noFPUC(3:10)) / sum(mean_y_sim_pandemic_u(3:10) - mean_y_sim_pandemic_noFPUC(3:10))
mpc_supplements_onset.expect('full+3') = sum(mean_c_sim_pandemic_expect(3:13) - mean_c_sim_pandemic_expect_noFPUC(3:13)) / sum(mean_y_sim_pandemic_u(3:13) - mean_y_sim_pandemic_noFPUC(3:13))
mpc_supplements_onset.Variables=scale_factor*mpc_supplements_onset.Variables
mpc_timeseries_onset=scale_factor*((mean_c_sim_pandemic_expect(3) - mean_c_sim_pandemic_expect(2))-(mean_c_sim_e(3) - mean_c_sim_e(2))) / ((mean_y_sim_pandemic_u(3) - mean_y_sim_pandemic_u(2))-(mean_y_sim_e(3)-mean_y_sim_e(2)));


load mpc_supplements_expiration;
load prepandemic_andpandemic_results_target500MPC;


table_mpc_supplements_for_paper=table();
table_mpc_supplements_for_paper.Supp600('best fit model: one month MPC')=mpc_supplements_expiration.surprise('one_month');
table_mpc_supplements_for_paper.Supp300('best fit model: one month MPC')=mpc_supplements_onset.expect('one_month');

table_mpc_supplements_for_paper.Supp600('prepandemic model: one month MPC')=mpc_supplements_pandemic_match500mpc.expect('one_month');
table_mpc_supplements_for_paper.Supp300('prepandemic model: one month MPC')=mpc_supplements_onset_target500MPC.expect('one_month')

table_supplement_effects=table_elasticities;
table_supplement_effects.supplement600('Best Fit Model: MPC from first month of supplements')=mpc_supplements_expiration.surprise('one_month');
table_supplement_effects.supplement300('Best Fit Model: MPC from first month of supplements')=mpc_supplements_onset.expect('one_month')
table_supplement_effects.supplement600('Best Fit Model: MPC from first quarter of supplements')=mpc_supplements_expiration.surprise('3_month');
table_supplement_effects.supplement300('Best Fit Model: MPC from first quarter of supplements')=mpc_supplements_onset.expect('3_month')
table_supplement_effects.supplement600('Best Fit Model: MPC over full supplement period')=mpc_supplements_expiration.surprise('full');
table_supplement_effects.supplement300('Best Fit Model: MPC over full supplement period')=mpc_supplements_onset.expect('full')
table_supplement_effects.supplement600('Best Fit Model: MPC over full supplement period+3')=mpc_supplements_expiration.surprise('full+3');
table_supplement_effects.supplement300('Best Fit Model: MPC over full supplement period+3')=mpc_supplements_onset.expect('full+3')
writetable(table_supplement_effects, fullfile(release_path_paper,'table_supplement_effects.csv'), 'WriteRowNames', true);
