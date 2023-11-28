clearvars -except -regexp fig_paper_*

load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load hh_wage_groups.mat
load release_paths.mat

data_update = readtable(spending_input_directory, 'Sheet', model_data);

idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'median')& data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'median') & data_update.periodid>=201901;
idx_w = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'median') & data_update.periodid>=202001;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
data_update_w=data_update(idx_w,:);
income_e=data_update_e.value;
income_u=data_update_u.value;
income_w=data_update_w.value;
income_e_yoy=income_e(13:end)./income_e(1:end-12)*income_e(13);
income_u_yoy=income_u(13:end)./income_u(1:end-12)*income_u(13);
income_w_yoy=income_w(13:end)./income_w(1:end-12)*income_w(13);
income_e=income_e(13:end);
income_u=income_u(13:end);
income_w=income_w;

idx_emp = (string(data_update.category) == 'Checking account balance') & (string(data_update.group) == 'Employed')& (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_w = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid>=202001;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
data_update_w=data_update(idx_w,:);
checking_e=data_update_e.value;
checking_u=data_update_u.value;
checking_w=data_update_w.value;
checking_e_yoy=checking_e(13:end)./checking_e(1:end-12)*checking_e(13);
checking_u_yoy=checking_u(13:end)./checking_u(1:end-12)*checking_u(13);
checking_w_yoy=checking_w(13:end)./checking_w(1:end-12)*checking_w(13);
checking_e=checking_e(13:end);
checking_u=checking_u(13:end);
checking_w=checking_w;


plot(1:8,checking_e(1:8),1:8,checking_u(1:8))

diff_check_e=checking_e(8)-checking_e(1);
diff_check_u=checking_u(8)-checking_u(1);
diff_check_u_vs_e=diff_check_u-diff_check_e;
bal_over_income=diff_check_u_vs_e/income_u(1)

load a_increase_vs_y_bestfitmodel;
hazardcoef_ccw=-0.094;
severance_ccw=2;
logchange_hazard_pandemic_implied=hazardcoef_ccw*a_increase_vs_y_bestfitmodel/severance_ccw

exit_rates_data = readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date = datetime(exit_rates_data.week_start_date);
idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-11-20');
exit_rates_data = exit_rates_data(idx, :);
exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');

% For the exit variables we want the average exit probability at a monthly level
exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
monthly_search_data = exit_rates_data_week_to_month.exit_not_to_recall';
search_aug_to_nov=mean(monthly_search_data(8:11));
search_pre=mean(monthly_search_data(1:3));
search_no_liquidity=exp(-logchange_hazard_pandemic_implied)*search_aug_to_nov;
share_explained_liquidity_ccw=(search_no_liquidity-search_aug_to_nov)/(search_pre-search_aug_to_nov);


for severance_amount_index=1:3


tic

load bestfit_prepandemic.mat

sse_expect_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_expect_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_expect_fit_het_match500MPC(3)=0;

sse_surprise_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_surprise_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_surprise_fit_het_match500MPC(3)=0;

data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed')& (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_w = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid>=202001;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
data_update_w=data_update(idx_w,:);
total_spend_e=data_update_e.value;
total_spend_u=data_update_u.value;
total_spend_w=data_update_w.value;
total_spend_e_yoy=total_spend_e(13:end)./total_spend_e(1:end-12)*total_spend_e(13);
total_spend_u_yoy=total_spend_u(13:end)./total_spend_u(1:end-12)*total_spend_u(13);
total_spend_w_yoy=total_spend_w(13:end)./total_spend_w(1:end-12)*total_spend_w(13);
total_spend_e=total_spend_e(13:end);
total_spend_u=total_spend_u(13:end);
perc_spend_e=data_update_e.percent_change;
perc_spend_u=data_update_u.percent_change;
perc_spend_w=data_update_w.percent_change;
perc_spend_u_vs_e=perc_spend_u-perc_spend_e;
perc_spend_u_vs_e=perc_spend_u_vs_e(13:end);
perc_spend_w_vs_e=perc_spend_w-perc_spend_e(13:end);
spend_dollars_u_vs_e=perc_spend_u_vs_e*total_spend_u(1);
spend_dollars_w_vs_e=perc_spend_w_vs_e*total_spend_w(1);


idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean')& data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_w = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid>=202001;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
data_update_w=data_update(idx_w,:);
income_e=data_update_e.value;
income_u=data_update_u.value;
income_w=data_update_w.value;
income_e_yoy=income_e(13:end)./income_e(1:end-12)*income_e(13);
income_u_yoy=income_u(13:end)./income_u(1:end-12)*income_u(13);
income_w_yoy=income_w(13:end)./income_w(1:end-12)*income_w(13);
income_e=income_e(13:end);
income_u=income_u(13:end);
perc_income_e=data_update_e.percent_change;
perc_income_u=data_update_u.percent_change;
perc_income_w=data_update_w.percent_change;
perc_income_u_vs_e=perc_income_u-perc_income_e;
perc_income_u_vs_e=perc_income_u_vs_e(13:end);
perc_income_w_vs_e=perc_income_w-perc_income_e(13:end);
income_dollars_u_vs_e=perc_income_u_vs_e*income_u(1);
income_dollars_w_vs_e=perc_income_w_vs_e*income_w(1);


idx_emp = (string(data_update.category) == 'Total inflows') & (string(data_update.group) == 'Employed')& (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Total inflows') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_w = (string(data_update.category) == 'Total inflows') & startsWith(string(data_update.group), 'Waiting') & (string(data_update.measure) == 'mean') & data_update.periodid>=202001;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
data_update_w=data_update(idx_w,:);
total_inflows_e=data_update_e.value;
total_inflows_u=data_update_u.value;
total_inflows_w=data_update_w.value;
total_inflows_e_yoy=total_inflows_e(13:end)./total_inflows_e(1:end-12)*total_inflows_e(13);
total_inflows_u_yoy=total_inflows_u(13:end)./total_inflows_u(1:end-12)*total_inflows_u(13);
total_inflows_w_yoy=total_inflows_w(13:end)./total_inflows_w(1:end-12)*total_inflows_w(13);
total_inflows_e=total_inflows_e(13:end);
total_inflows_u=total_inflows_u(13:end);
perc_inflows_e=data_update_e.percent_change;
perc_inflows_u=data_update_u.percent_change;
perc_inflows_w=data_update_w.percent_change;
perc_inflows_u_vs_e=perc_inflows_u-perc_inflows_e;
perc_inflows_u_vs_e=perc_inflows_u_vs_e(13:end);
perc_inflows_w_vs_e=perc_inflows_w-perc_inflows_e(13:end);
inflows_dollars_u_vs_e=perc_inflows_u_vs_e*total_inflows_u(1);
inflows_dollars_w_vs_e=perc_inflows_w_vs_e*total_inflows_w(1);


shock_500=500/income_e(1);



rng('default')


% We impose c, the search cost intercept is zero
k = pre_pandemic_fit_match500MPC(1);
gamma = pre_pandemic_fit_match500MPC(2);
c_param=0;

% Assign parameter values
load discountfactors.mat
beta_normal=beta_targetwaiting;
beta_high = beta_oneperiodshock;

load model_parameters.mat
initial_a = initial_a - aprimemin;

n_ben_profiles_allowed = 5; %This captures the surprise vs. expected expiration scenarios w/ wait or no delay

% Set on/off switches
infinite_dur = 0;
use_initial_a = 0;

include_recalls=1;
infinite_dur=false;

perc_change_benefits=.00001;

    % Loop over wage groups
    for iy = 1:5
        
        y = w(iy); %income when employed
        h = .7 * y; %home production value (will last as long as unemployed)
        b = repshare * y; %regular benefit level (lasts 12 months)

        % Aprime grid
        aprimemax = 2000;
        Aprime = exp(linspace(0.00, log(aprimemax), n_aprime)) - 1;
        Aprime = Aprime';

        benefit_profile(1:12, 1) = h + b;
        if infinite_dur == true
            benefit_profile(13:13, 1) = h + b;
        else
            benefit_profile(13:13, 1) = h;
        end

        % Augmented benefits profile with an additional small benefit for
        % calculating duration elasticity
        benefit_profile_augmented(1:12, 1) = h + b + perc_change_benefits * b;

        if infinite_dur == true
            benefit_profile_augmented(13:13, 1) = h + b + perc_change_benefits * b;
        else
            benefit_profile_augmented(13:13, 1) = h;
        end
        
        
        if severance_amount_index==1
            severance_amount=0;
        elseif severance_amount_index==2
            severance_amount=2;
        else
            severance_amount=perc_change_benefits*b*12;
            %10.7 weeks in chetty 08 divided by 4.3 weeks per month
        end
        

        % Recall setting
        if include_recalls == true
            recall_probs(1:13) = 0.08;
        else
            recall_probs(1:13) = 0;
        end

        %initialization of variables for speed
        c_pol_e = zeros(n_aprime, 1);
        c_pol_u = zeros(n_aprime, n_b, 1);
        c_pol_u_augmented = zeros(n_aprime, n_b);
        v_e = c_pol_e;
        v_u = c_pol_u;
        v_u_augmented = c_pol_u_augmented;

        rhs_e = zeros(n_aprime, 1);
        rhs_u = zeros(n_aprime, n_b);
        rhs_u_augmented = zeros(n_aprime, n_b);

        for beta_loop = 1:2

            %this solves model for stationary contraction mapping when
            %beta=beta_normal as well as for a one period temporarily
            %higher discount rate to capture pandemic declines in C
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

            for ib = 1:n_b
                c_pol_u_guess(:, ib) = benefit_profile(ib) + Aprime(:) * (1 + r) + r * aprimemin;
                v_u_guess(:, ib) = ((c_pol_u_guess(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k * 0^(1 + gamma)) / (1 + gamma)+c_param;
                c_pol_u_augmented_guess(:, ib) = benefit_profile_augmented(ib) + Aprime(:) * (1 + r) + r * aprimemin;
                v_u_augmented_guess(:, ib) = ((c_pol_u_augmented_guess(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k * 0^(1 + gamma)) / (1 + gamma)+c_param;
            end

            optimal_search_guess = zeros(n_aprime, n_b);
            optimal_search_augmented_guess = zeros(n_aprime, n_b);

            tic

            if beta_loop == 2
                maxiter = 1; %this effectively governs how many periods households will think the high discount factor will last, setting maxiter=1 essentially runs one backward induction step from the beta_normal solutions
                %note that the code must be structured so that it solves the
                %beta_normal part first
                c_pol_e_guess = c_pol_e_betanormal;
                c_pol_u_guess = c_pol_u_betanormal;
                c_pol_u_augmented_guess = c_pol_u_augmented_betanormal;

                v_e_guess = v_e_betanormal;
                v_u_guess = v_u_betanormal;
                v_u_augmented_guess = v_u_augmented_betanormal;
            else
                maxiter = 1000;
            end

            while ((ave_change_in_C_percent > tol_c_percent) || (ave_change_in_S > tol_s)) && iter < maxiter   %Value function iteration loop
                  
                %Note: Using (semi)-standard EGM notation,
                %c_pol is c(a,y)
                %c_tilde is c(a',y)

                %%%%%%%%%%%%%%%%%%%%%%
                % Employed
                %%%%%%%%%%%%%%%%%%%%%%
                rhs_e(:) = beta * (1 + r) * ((1 - sep_rate) * c_pol_e_guess(:).^(-mu) + sep_rate * c_pol_u_guess(:, 1).^(-mu)); %RHS of Euler equation when unconstrained
                c_tilde_e(:) = (rhs_e(:)).^(-1 / mu); %c choice unconstrained
                a_star_e(:) = (Aprime(:) + c_tilde_e(:) - y(1)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                a_star1_e = (c_tilde_e(1) - y(1)) / (1 + r); %get a from a' using budget constraint
                for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both

                    %With EGM we have C on a grid of a' but want to get on
                    %a grid of a. Need to interpolate:
                    if Aprime(ia) > a_star1_e %When unconstrained this is optimal C
                        c_pol_e(ia:end) = interp1(a_star_e(:), c_tilde_e(:), Aprime(ia:end), 'linear', 'extrap');
                        break
                    else %When constrained, optimal C given by budget constraint
                        c_pol_e(ia) = (1 + r) * Aprime(ia) + y + r * aprimemin;
                    end

                end

                a_prime_holder = (1 + r) * Aprime + y(1) - c_pol_e(:);
                v_e(:) = ((c_pol_e(:)).^(1 - mu) - 1) / (1 - mu) + beta * ((1 - sep_rate) .* interp1(Aprime, v_e_guess(:), a_prime_holder, 'linear', 'extrap') + sep_rate * interp1(Aprime, v_u_guess(:, 1), a_prime_holder, 'linear', 'extrap'));
                %We need to compute v_e rather than just its derivative
                %like in a normal EGM algorithm, because the level of V
                %will be relevant for optimal search

                %%%%%%%%%%%%%%%%%%%%%%%
                % Unemployed
                %%%%%%%%%%%%%%%%%%%%%%%
                for ib = 1:n_b
                    %Computing optimal search given current value function
                    %guesses for continuation values
                    tmp = min(1 - recall_probs(ib), max(0, (beta * (v_e_guess(:) - v_u_guess(:, min(ib + 1, n_b))) / k).^(1 / gamma)));
                    tmp(imag(tmp) ~= 0) = 0;
                    optimal_search(:, ib) = tmp;

                    %RHS of optimality condition computes expected marignal
                    %utility where probability of being employed or
                    %unemployed depends on current search and recall rates
                    rhs_u(:, ib) = beta * (1 + r) * ((recall_probs(ib) + optimal_search(:, ib)) .* c_pol_e_guess(:).^(-mu) + (1 - optimal_search(:, ib) - recall_probs(ib)) .* c_pol_u_guess(:, min(ib + 1, n_b)).^(-mu));
                    c_tilde_u(:, ib) = (rhs_u(:, ib)).^(-1 / mu); %unconstrained
                    a_star_u(:, ib) = (Aprime(:) + c_tilde_u(:, ib) - benefit_profile(ib)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                    a_star1_u(ib) = (c_tilde_u(1, ib) - benefit_profile(ib)) / (1 + r);
                end

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

                %Value function when unemployed will include the search
                %costs
                for ib = 1:n_b
                    a_prime_holder_u(:, ib) = (1 + r) * Aprime + benefit_profile(ib) - c_pol_u(:, ib);
                    v_u(:, ib) = ((c_pol_u(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k * optimal_search(:, ib).^(1 + gamma)) / (1 + gamma)+c_param + beta * ((optimal_search(:, ib) + recall_probs(ib)) .* interp1(Aprime, v_e_guess(:), a_prime_holder_u(:, ib), 'linear', 'extrap') + (1 - optimal_search(:, ib) - recall_probs(ib)) .* interp1(Aprime, v_u_guess(:, min(ib + 1, n_b)), a_prime_holder_u(:, ib), 'linear', 'extrap'));
                end

                %%%%%%%%%%%%%%%%%%%%%%%
                % Augmented benefits unemployed with slightly higher b
                % level
                %%%%%%%%%%%%%%%%%%%%%%%
                for ib = 1:n_b
                    tmp = min(1 - recall_probs(ib), max(0, (beta * (v_e_guess(:) - v_u_augmented(:, min(ib + 1, n_b))) / k).^(1 / gamma)));
                    tmp(imag(tmp) ~= 0) = 0;
                    optimal_search_augmented(:, ib) = tmp;

                    rhs_u_augmented(:, ib) = beta * (1 + r) * ((recall_probs(ib) + optimal_search_augmented(:, ib)) .* c_pol_e_guess(:).^(-mu) + (1 - optimal_search_augmented(:, ib) - recall_probs(ib)) .* c_pol_u_augmented_guess(:, min(ib + 1, n_b)).^(-mu));
                    c_tilde_u_augmented(:, ib) = (rhs_u_augmented(:, ib)).^(-1 / mu); %unconstrained
                    a_star_u_augmented(:, ib) = (Aprime(:) + c_tilde_u_augmented(:, ib) - benefit_profile_augmented(ib)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                    a_star1_u_augmented(ib) = (c_tilde_u_augmented(1, ib) - benefit_profile_augmented(ib)) / (1 + r);
                end


                for ib = 1:n_b
                    for ia = 1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
                        if Aprime(ia) > a_star1_u_augmented(ib)
                            c_pol_u_augmented(ia:end, ib) = interp1(a_star_u_augmented(:, ib), c_tilde_u_augmented(:, ib), Aprime(ia:end), 'linear', 'extrap');
                            break
                            %constrained_u(ia,ib,t)=0;
                        else
                            c_pol_u_augmented(ia, ib) = (1 + r) * Aprime(ia) + benefit_profile_augmented(ib) + r * aprimemin;
                            %constrained_u(ia,ib,t)=1;
                        end
                    end
                end

                for ib = 1:n_b
                    a_prime_holder_u_augmented(:, ib) = (1 + r) * Aprime + benefit_profile_augmented(ib) - c_pol_u_augmented(:, ib);
                    v_u_augmented(:, ib) = ((c_pol_u_augmented(:, ib)).^(1 - mu) - 1) / (1 - mu) - (k * optimal_search_augmented(:, ib).^(1 + gamma)) / (1 + gamma)+c_param + beta * ((recall_probs(ib) + optimal_search_augmented(:, ib)) .* interp1(Aprime, v_e_guess(:), a_prime_holder_u_augmented(:, ib), 'linear', 'extrap') + (1 - optimal_search_augmented(:, ib) - recall_probs(ib)) .* interp1(Aprime, v_u_augmented_guess(:, min(ib + 1, n_b)), a_prime_holder_u_augmented(:, ib), 'linear', 'extrap'));
                end

                % Analyzing convergence
                diffC = max([max(max(abs(c_pol_e(:) - c_pol_e_guess(:)))), max(max(max(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), max(max(max(max(abs(c_pol_u_augmented(:, :, :) - c_pol_u_augmented_guess(:, :, :))))))]);

                diffC_percent = 100 * max([max(max(abs((c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess(:)))), max(max(max(abs((c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :))))), max(max(max(max(abs((c_pol_u_augmented(:, :, :) - c_pol_u_augmented_guess(:, :, :)) ./ c_pol_u_augmented_guess(:, :, :))))))]);

                % Absolute difference in value to measure convergence
                diffV = max([max(abs(v_e(:) - v_e_guess(:))), max(max(abs(v_u(:, :) - v_u_guess(:, :)))), max(max(max(abs(v_u_augmented(:, :, :) - v_u_augmented_guess(:, :, :)))))]);

                ave_change_in_C = mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), mean(mean(mean(mean(abs(c_pol_u_augmented(:, :, :) - c_pol_u_augmented_guess(:, :, :))))))]);

                ave_change_in_C_percent = 100 * mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_augmented(:, :, :) - c_pol_u_augmented_guess(:, :, :)) ./ c_pol_u_augmented_guess))))]);

                ave_change_in_V = mean([mean(abs(v_e(:) - v_e_guess(:))), mean(mean(abs(v_u(:, :) - v_u_guess(:, :)))), mean(mean(mean(abs(v_u_augmented(:, :, :) - v_u_augmented_guess(:, :, :)))))]);

                ave_change_in_S = mean([mean(mean(mean(abs(optimal_search(:, :) - optimal_search_guess(:, :))))), mean(mean(mean(mean(abs(optimal_search_augmented(:, :, :) - optimal_search_augmented_guess(:, :, :))))))]);

                % Update guesses, fully for now.
                c_pol_e_guess = c_pol_e;
                c_pol_u_guess = c_pol_u;
                c_pol_u_augmented_guess = c_pol_u_augmented;
                v_e_guess = v_e;
                v_u_guess = v_u;
                v_u_augmented_guess = v_u_augmented;
                optimal_search_guess = optimal_search;
                optimal_search_augmented_guess = optimal_search_augmented;

                % Count the iteration
                iter = iter + 1;

            end

            %Just collecting results indexed by beta value
            if beta_loop == 1
                c_pol_e_betanormal = c_pol_e;
                c_pol_u_betanormal = c_pol_u;
                c_pol_u_augmented_betanormal = c_pol_u_augmented;

                v_e_betanormal = v_e;
                v_u_betanormal = v_u;
                v_u_augmented_betanormal = v_u_augmented;
            elseif beta_loop == 2
                c_pol_e_betahigh = c_pol_e;
                c_pol_u_betahigh = c_pol_u;
                c_pol_u_augmented_betahigh = c_pol_u_augmented;

                v_e_betahigh = v_e;
                v_u_betahigh = v_u;
                v_u_augmented_betahigh = v_u_augmented;
            end

        end

        %Use the same grid for A as for Aprime
        A = Aprime;

        % Number of time periods
        numt_sim = 36;
        % Unemployed assets and consumption initialization for speed
        a_u_sim = zeros(numt_sim, 1);
        c_u_sim = a_u_sim;
        c_u_augmented_sim = a_u_sim;
        % Consumption of employed same as unemployed
        c_e_sim = c_u_sim;
        % Initial assets unemployed and employed
        a_u_sim(1) = initial_a; %Note this is initial_a but before the "burnin" here so it doesn't really play any role
        a_u_augmented_sim = a_u_sim;
        a_e_sim = a_u_sim;



        % 500 households
        numhh = 500;
        numsim = 15;
        % Droppping some initial periods to reduce sensitivity to initial conditions
        burnin = 15;
        % Each household has dropped periods and one more
        a_sim = zeros(numhh, burnin + 1);
        c_sim = zeros(numhh, burnin + 1);
        e_sim = zeros(numhh, burnin + 1);
        u_dur_sim = zeros(numhh, burnin + 1);
        % First sim with initial assets all employed
        a_sim(:, 1) = initial_a;
        e_sim(:, 1) = 1;

        % Starter consumption
        c_pol_e = c_pol_e_betanormal;
        c_pol_u = c_pol_u_betanormal;
        c_pol_u_augmented = c_pol_u_augmented_betanormal;

        % Starter value
        v_e = v_e_betanormal;
        v_u = v_u_betanormal;
        v_u_augmented = v_u_augmented_betanormal;

        % Loop over the time periods up to the point that is dropped
        for t = 1:burnin

            % Loop over the households
            for i = 1:numhh
                % Get consumption and assets based on employment status
                if e_sim(i, t) == 1
                    c_sim(i, t) = interp1(A, c_pol_e(:), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = y + (1 + r) * a_sim(i, t) - c_sim(i, t);
                else
                    c_sim(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = max(benefit_profile(u_dur_sim(i, t)) + (1 + r) * a_sim(i, t) - c_sim(i, t),0);
                end

                % Draw employment status
                randy = rand(1, 1);

                % Determine whether employed next period if currently employed using sep rate
                if e_sim(i, t) == 1

                    if randy < sep_rate
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = 1;
                    else
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    end

                % If currently unemployed use exogenous job find rate to determine next period
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


        % Get rid of some initial periods with burnin
        tmp_a = a_sim(:, burnin + 1);
        tmp_u = u_dur_sim(:, burnin + 1);
        tmp_e = e_sim(:, burnin + 1);

        % Keeping track of assets, consumption, employment, unemployment duration
        a_sim = zeros(numhh, numsim);
        c_sim = zeros(numhh, numsim);
        e_sim = zeros(numhh, numsim);
        u_dur_sim = zeros(numhh, numsim);
        % New starter assets and unemployment and and employment
        a_sim(:, 1) = tmp_a;
        u_dur_sim(:, 1) = tmp_u;
        e_sim(:, 1) = tmp_e;

        
        
        % Start with some initial assets or not
        if use_initial_a == 1
            a_sim_augmented = tmp_a(tmp_u > 0) + initial_a;
            a_sim_augmented_wait = tmp_a(tmp_u > 0) + initial_a;
            a_sim_regular = tmp_a(tmp_u > 0) + initial_a;
            a_sim_augmented_noFPUC = tmp_a(tmp_u > 0) + initial_a;
            a_sim_e = tmp_a(tmp_u == 0) + initial_a;
        else
            a_sim_augmented = tmp_a(tmp_u > 0)+y*severance_amount;
            a_sim_augmented_wait = tmp_a(tmp_u > 0)+y*severance_amount;
            a_sim_regular = tmp_a(tmp_u > 0)+y*severance_amount;
            a_sim_augmented_noFPUC = tmp_a(tmp_u > 0)+y*severance_amount;
            a_sim_e = tmp_a(tmp_u == 0)+y*severance_amount;
        end

        % Count up number of households in each group unemp and emp
        num_unemployed_hh = length(a_sim_augmented);
        num_employed_hh = length(a_sim_e);
        c_sim_augmented = zeros(length(a_sim_augmented), 30);
        c_sim_regular = zeros(length(a_sim_augmented), 30);
        c_sim_e = zeros(length(a_sim_e), 30);

        % Search matrix- has width of 30
        search_sim_augmented = zeros(length(a_sim_augmented), 30);
        search_sim_regular = zeros(length(a_sim_regular), 30);

        % This is looping over just unemployed households (continuously unemployed) to get u time-series patterns
        length_u = 0;

        % Loop over times or also length unemployment I think
        for t = 1:15
            length_u = min(length_u + 1, n_b);

            c_pol_u = c_pol_u_betanormal;
            c_pol_u_augmented = c_pol_u_augmented_betanormal;
            v_e = v_e_betanormal;
            v_u = v_u_betanormal;
            v_u_augmented = v_u_augmented_betanormal;
            beta=beta_normal;

            % Loop over households
            for i = 1:num_unemployed_hh


                % Use consumption and assets functions to get regular unemployed values
                c_sim_regular(i, t) = interp1(A, c_pol_u(:, length_u), a_sim_regular(i, t), 'linear');
                a_sim_regular(i, t + 1) = max(benefit_profile(length_u) + (1 + r) * a_sim_regular(i, t) - c_sim_regular(i, t),0);

                % Also do for augmented values
                c_sim_augmented(i, t) = interp1(A, c_pol_u_augmented(:, length_u), a_sim_augmented(i, t), 'linear');
                a_sim_augmented(i, t + 1) = max(benefit_profile_augmented(length_u, 1) + (1 + r) * a_sim_augmented(i, t) - c_sim_augmented(i, t),0);

                % Difference in value functions, regular case
                diff_v = interp1(A, v_e(:), a_sim_regular(i, t + 1), 'linear') - interp1(A, v_u(:, min(length_u + 1, n_b)), a_sim_regular(i, t + 1), 'linear');
                % Bound search at one
                search_sim_regular(i, t) = min(1 - recall_probs(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                % Imaginary search set to zero
                if imag(search_sim_regular(i, t)) ~= 0
                    search_sim_regular(i, t) = 0;
                end

                % Difference in value functions, augmented
                diff_v = interp1(A, v_e(:), a_sim_augmented(i, t + 1), 'linear') - interp1(A, v_u_augmented(:, min(length_u + 1, n_b), 1), a_sim_augmented(i, t + 1), 'linear');
                % Bound search at one
                search_sim_augmented(i, t) = min(1 - recall_probs(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                % Imaginary search set to zero
                if imag(search_sim_augmented(i, t)) ~= 0
                    search_sim_augmented(i, t) = 0;
                end

                %note for surprise case won't want to use i_b+1 in actuality will
                %want to use the expected one in the last period before surprise

            end

        end

        % Mean search for the wage group
        mean_search_sim_regular_bywage(iy, :) = mean(search_sim_regular, 1);
        mean_search_sim_augmented_bywage(iy, :) = mean(search_sim_augmented, 1);

        
        mean_a_sim_regular_bywage(iy, :) = mean(a_sim_regular, 1);
        mean_c_sim_regular_bywage(iy, :) = mean(c_sim_regular, 1);
    end

    % Mean search across wage groups
    mean_search_sim_regular = mean(mean_search_sim_regular_bywage, 1);
    mean_search_sim_augmented = mean(mean_search_sim_augmented_bywage, 1);

    % Target elasticity of 0.5 and the monthly pre-period job finding rate

    % Produce elasticity
    % Need to create the total exit rate
    total_exit_rate_regular = mean_search_sim_regular(1:length(recall_probs))' + recall_probs';
    total_exit_rate_augmented = mean_search_sim_augmented(1:length(recall_probs))' + recall_probs';

    % Calculation should be over the entire period
    elasticity(severance_amount_index) = (average_duration(total_exit_rate_augmented) / average_duration(total_exit_rate_regular) - 1) / perc_change_benefits

    elasticity_newjob(severance_amount_index) = (average_duration(mean_search_sim_augmented(1:length(recall_probs))') / average_duration(mean_search_sim_regular(1:length(recall_probs))') - 1) / perc_change_benefits
    
    mean_search_sim_by_severance(severance_amount_index,:)=mean_search_sim_regular;
    mean_total_exit_rate_by_severance(severance_amount_index,:)=total_exit_rate_regular;


    newjob_find_holder_month1(severance_amount_index)=mean_search_sim_regular(1);
    newjob_find_augmented_holder_month1(severance_amount_index)=mean_search_sim_augmented(1);
    
 
end


ratio_newjob=mean(mean_search_sim_by_severance(2,1:5)./mean_search_sim_by_severance(1,1:5))

ratio_total=mean(mean_total_exit_rate_by_severance(2,1:5)./mean_total_exit_rate_by_severance(1,1:5))




log(ratio_newjob)

%note: can compare this to chetty 08 if want to
share_newrate=((mean_search_sim_by_severance(3,1)-newjob_find_holder_month1(1))/newjob_find_holder_month1(1))/((newjob_find_augmented_holder_month1(3)-newjob_find_holder_month1(1))/newjob_find_holder_month1(1))


table_stats_for_text_liquidity=table();
table_stats_for_text_liquidity.stat('Two Months Severance CCW06 effect on log hazard lower bound')=-.076;
table_stats_for_text_liquidity.stat('Two Months Severance CCW06 upper bound')=-.109;
table_stats_for_text_liquidity.stat('Two Months Severance pre pandemic model')=log(ratio_newjob)
table_stats_for_text_liquidity.stat('Back of Envelope pandemic share using CCW06 together with asset increase in model')=share_explained_liquidity_ccw
writetable(table_stats_for_text_liquidity,fullfile(release_path_paper,'table_stats_for_text_liquidity.csv'),'WriteRowNames',true);
