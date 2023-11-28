
% Variables from Slurm scheduler/array job
try
    disp(jobname);
    disp(mainjobid);
    disp(taskid);
    z.jobname = jobname;
    z.mainjobid = mainjobid;
    z.taskid = taskid;
catch %second part of this condition are the values filled in if run locally
    z.jobname = ['local_', strrep(char(datetime('today')), '-', '_')];
    z.taskid = 1;
end

% Add main model folder to path
oldpath = path;
addpath(oldpath, extractBefore(string(pwd), '/robustness/'));

% Load saved paths
load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load release_paths.mat
path_array_job = 'release/array_job'; 

% Plot settings
load matlab_qual_colors.mat
global qual_blue qual_purple qual_green qual_orange matlab_red_orange qual_yellow
load graph_axis_labels_timeseries.mat

% Matrix of beta and delta
% --> assign values using the task_id of the array job
array_beta = [.994 .9925 .99 .9875 .985 .9825 .98 .975 .97 .96 .94 .91 .88];
array_delta = array_beta;
[beta_layer, delta_layer] = ndgrid(array_beta, array_delta');
array_discounting = [beta_layer(:) delta_layer(:)];
array_discounting = sortrows(array_discounting(array_discounting(:, 1) >= array_discounting(:, 2), :), [1 2]);
num_tasks = size(array_discounting, 1);

beta_normal = array_discounting(z.taskid, 1);
delta = array_discounting(z.taskid, 2);

beta_normal_april = 1.0205 * beta_normal;
delta_april = 1.0205 * delta;

% Other parameter values
load hh_wage_groups.mat
load model_parameters.mat
initial_a = initial_a - aprimemin;

load bestfit_prepandemic.mat
k_prepandemic = pre_pandemic_fit_match500MPC(1);
gamma_prepandemic = pre_pandemic_fit_match500MPC(2);
c_param_prepandemic = 0;

n_ben_profiles_allowed = 8; %This captures the surprise vs. expected expiration scenarios w/ wait or no delay (and 2 extra benefit profiles for liquidity decomposition)

% Set on/off switches
infinite_dur = 0;
use_initial_a = 0;

% Loop over search cost parameters
array_k = [1 3 5 7 10 15 20 50 90 130];
array_gamma = [.1 .3 .5 .7 1 1.5 2 3 4 5];
array_c_param = [-15 -12 -9 -6 -4 -2 -1.5 -1 -.5 -.2];
N_k = size(array_k, 2);
N_gamma = size(array_gamma, 2);
N_c_param = size(array_c_param, 2);

% Start the loop
iter_count = 0;
for i_k = 1:N_k
for i_gamma = 1:N_gamma
for i_c_param = 1:N_c_param

tic
iter_count = iter_count + 1;
disp(['Task ID ', num2str(z.taskid), '; triple-loop iteration ', num2str(iter_count), '; k=', num2str(array_k(i_k)), ', gamma=', num2str(array_gamma(i_gamma)), ', c_param=', num2str(array_c_param(i_c_param))]);

k = array_k(i_k);
gamma = array_gamma(i_gamma);
c_param = array_c_param(i_c_param);

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

    rng('default')

    % Aprime grid
    aprimemax = 2000;    
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
    %benefit_profile_pandemic(1)=b+h+1.5*FPUC_expiration;
    %benefit_profile_pandemic(3)=b+h+1.05*FPUC_expiration;
    %benefit_profile_pandemic(4)=b+h+1.25*FPUC_expiration;
    %benefit_profile_pandemic(5:12,2)=b+h+1*FPUC_expiration;
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

    for beta_loop = 1:4

        if beta_loop == 1
            beta = beta_normal;
        elseif beta_loop == 2
            beta = delta;
        elseif beta_loop == 3 %!!!
            beta = beta_normal_april;
        elseif beta_loop == 4
            beta = delta_april;
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
        %while (iter <= 5000) & (diffC > tol) %& (diffC_percent > tol_percent) %| diffV > tol)

        if beta_loop == 1
            maxiter = 1000;

            %save the initial guess to use in iteration 3 (model for Apr)
            hold_guess_1 = c_pol_e_guess;
            hold_guess_2 = c_pol_e_with_transfer_guess;
            hold_guess_3 = c_pol_u_guess;
            hold_guess_4 = c_pol_u_pandemic_guess;

            hold_guess_5 = v_e_guess;
            hold_guess_6 = v_e_with_transfer_guess;
            hold_guess_7 = v_u_guess;
            hold_guess_8 = v_u_pandemic_guess;

        elseif beta_loop == 2
            maxiter = 1; 
            % This effectively governs how many periods households will
            % think the low discount factor (delta) will last. This is
            % the ***hyperbolic discounting***. Setting maxiter=1 essentially
            % runs one backward induction step from the beta_normal
            % solution. Note that the code must be structured so that
            % it solves the beta_normal part first.

            c_pol_e_guess = c_pol_e_betanormal;
            c_pol_e_with_transfer_guess = c_pol_e_with_transfer_betanormal;
            c_pol_u_guess = c_pol_u_betanormal;
            c_pol_u_pandemic_guess = c_pol_u_pandemic_betanormal;

            v_e_guess = v_e_betanormal;
            v_e_with_transfer_guess = v_e_with_transfer_betanormal;
            v_u_guess = v_u_betanormal;
            v_u_pandemic_guess = v_u_pandemic_betanormal;

        elseif beta_loop == 3
            maxiter = 1000; 
            % This is the discounting that agents in April expect for
            % future periods (but which never happens since after April
            % we switch to the policy functions solved above).

            % Fill in the initial guess (saved in first iteration of the loop):
            c_pol_e_guess = hold_guess_1;
            c_pol_e_with_transfer_guess = hold_guess_2;
            c_pol_u_guess = hold_guess_3;
            c_pol_u_pandemic_guess = hold_guess_4;

            v_e_guess = hold_guess_5;
            v_e_with_transfer_guess = hold_guess_6;
            v_u_guess = hold_guess_7;
            v_u_pandemic_guess = hold_guess_8;

        elseif beta_loop == 4 
            maxiter = 1; 
            % One backward induction for the hyperbolic discounting in
            % April.

            c_pol_e_guess = c_pol_e_betanormal_april;
            c_pol_e_with_transfer_guess = c_pol_e_with_transfer_betanormal_april;
            c_pol_u_guess = c_pol_u_betanormal_april;
            c_pol_u_pandemic_guess = c_pol_u_pandemic_betanormal_april;

            v_e_guess = v_e_betanormal_april;
            v_e_with_transfer_guess = v_e_with_transfer_betanormal_april;
            v_u_guess = v_u_betanormal_april;
            v_u_pandemic_guess = v_u_pandemic_betanormal_april;      

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
            diffV = max([max(abs(v_e(:) - v_e_guess(:))), max(max(abs(v_u(:, :) - v_u_guess(:, :)))), max(max(max(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

            ave_change_in_C = mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :))))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :))))))]);

            ave_change_in_C_percent = 100 * mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
          
            ave_change_in_C_percent_transfer = 100 * mean([mean(mean(mean(abs(c_pol_e_with_transfer(:,:) - c_pol_e_with_transfer_guess(:,:)) ./ c_pol_e_with_transfer_guess(:, :)))), mean(mean(mean(abs(c_pol_u(:, :) - c_pol_u_guess(:, :)) ./ c_pol_u_guess(:, :)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:, :, :) - c_pol_u_pandemic_guess(:, :, :)) ./ c_pol_u_pandemic_guess))))]);
          
            ave_change_in_C_percent = (ave_change_in_C_percent + ave_change_in_C_percent_transfer) ./ 2;

            ave_change_in_V = mean([mean(abs(v_e(:) - v_e_guess(:))), mean(mean(abs(v_u(:, :) - v_u_guess(:, :)))), mean(mean(mean(abs(v_u_pandemic(:, :, :) - v_u_pandemic_guess(:, :, :)))))]);

            ave_change_in_S = mean([mean(mean(mean(abs(optimal_search(:, :) - optimal_search_guess(:, :))))), mean(mean(mean(mean(abs(optimal_search_pandemic(:, :, :) - optimal_search_pandemic_guess(:, :, :))))))]);

            if mod(iter, 20) == 0
                %[iter diffC ave_change_in_C ave_change_in_C_percent ave_change_in_S diffV ave_change_in_V]

                %[iter ave_change_in_C ave_change_in_C_percent ave_change_in_S]
                %     stop
            end

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
            c_pol_e_delta = c_pol_e;
            c_pol_e_with_transfer_delta = c_pol_e_with_transfer;
            c_pol_u_delta = c_pol_u;
            c_pol_u_pandemic_delta = c_pol_u_pandemic;

            v_e_delta = v_e;
            v_e_with_transfer_delta = v_e_with_transfer;
            v_u_delta = v_u;
            v_u_pandemic_delta = v_u_pandemic;   
        elseif beta_loop == 3
            c_pol_e_betanormal_april = c_pol_e;
            c_pol_e_with_transfer_betanormal_april = c_pol_e_with_transfer;
            c_pol_u_betanormal_april = c_pol_u;
            c_pol_u_pandemic_betanormal_april = c_pol_u_pandemic;

            v_e_betanormal_april = v_e;
            v_e_with_transfer_betanormal_april = v_e_with_transfer;
            v_u_betanormal_april = v_u;
            v_u_pandemic_betanormal_april = v_u_pandemic;
        elseif beta_loop == 4 
            c_pol_e_delta_april = c_pol_e;
            c_pol_e_with_transfer_delta_april = c_pol_e_with_transfer;
            c_pol_u_delta_april = c_pol_u;
            c_pol_u_pandemic_delta_april = c_pol_u_pandemic;

            v_e_delta_april = v_e;
            v_e_with_transfer_delta_april = v_e_with_transfer;
            v_u_delta_april = v_u;
            v_u_pandemic_delta_april = v_u_pandemic; 
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

    c_pol_e = c_pol_e_delta;
    c_pol_e_with_transfer = c_pol_e_with_transfer_delta;
    c_pol_u = c_pol_u_delta;
    c_pol_u_pandemic = c_pol_u_pandemic_delta;

    v_e = v_e_delta;
    v_e_with_transfer = v_e_with_transfer_delta;
    v_u = v_u_delta;
    v_u_pandemic = v_u_pandemic_delta;

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

    if use_initial_a == 1
        a_sim_pandemic_expect = tmp_a(tmp_u > 0) + initial_a;
        a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0) + initial_a;
        a_sim_e = tmp_a(tmp_u == 0) + initial_a;
    else
        a_sim_pandemic_expect = tmp_a(tmp_u > 0);
        a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0);
        a_sim_e = tmp_a(tmp_u == 0);
    end

    num_unemployed_hh = length(a_sim_pandemic_expect);
    num_employed_hh = length(a_sim_e);
    c_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
    c_sim_pandemic_expect_wait = zeros(length(a_sim_pandemic_expect), 30);
    c_sim_e = zeros(length(a_sim_e), 30);

    search_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
    search_sim_pandemic_expect_wait = zeros(length(a_sim_pandemic_expect_wait), 30);


    %this is looping over all hh after the burnin period, to get the average
    %MPC
    for t = 1:numsim

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

    mean_c_sim = mean(c_sim, 1);

    %this is looping over just unemployed households (continuously unemployed)
    %to get u time-series patterns
    length_u = 0;

    for t = 1:numsim
        length_u = min(length_u + 1, n_b);
        c_pol_u = c_pol_u_delta;
        if t == 1
            c_pol_u_pandemic = c_pol_u_pandemic_delta_april;
            %beta=beta_high;
        else
            c_pol_u_pandemic = c_pol_u_pandemic_delta;
        end
        v_e=v_e_delta;
        v_u=v_u_delta;
        v_u_pandemic=v_u_pandemic_delta;
        beta=delta;
            
        for i = 1:num_unemployed_hh

            %allow for initial assets, isomorphic to borrowing
            if length_u==1
                a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t);
                a_sim_pandemic_expect_wait(i,t)=a_sim_pandemic_expect_wait(i,t);
            end
            %LWA
            if length_u == 6
                a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + LWAsize;
                a_sim_pandemic_expect_wait(i, t) = a_sim_pandemic_expect_wait(i, t) + LWAsize;
            end

            %Jan EIP
            if length_u == 10
                a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
                a_sim_pandemic_expect_wait(i, t) = a_sim_pandemic_expect_wait(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
            end
            

            c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect(i, t), 'linear');
            a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);

            c_sim_pandemic_expect_wait(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_expect_wait(i, t), 'linear');
            a_sim_pandemic_expect_wait(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_expect_wait(i, t) - c_sim_pandemic_expect_wait(i, t), 0);


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

        end

    end

    %this is looping over just employed households (continuously employed)
    %to get e time-series patterns
    for t = 1:numsim
        for i = 1:num_employed_hh

            if t == 4
                c_pol_e = c_pol_e_delta_april;
            else
                c_pol_e = c_pol_e_delta;
            end

            if t == 12
                a_sim_e(i, t) = a_sim_e(i, t) + 1500 * FPUC_expiration / (4.5 * 600);
            end

            %adjust initial assets isomorphic to allowing for borrowing
            if t==1
                a_sim_e(i,t)=a_sim_e(i,t);
            end

            c_sim_e(i, t) = interp1(A, c_pol_e(:), a_sim_e(i, t), 'linear', 'extrap');
            a_sim_e(i, t + 1) = y + (1 + r) * a_sim_e(i, t) - c_sim_e(i, t);
        end

    end

    mean_c_sim_e = mean(c_sim_e, 1);
    mean_c_sim_pandemic_expect = mean(c_sim_pandemic_expect, 1);
    mean_c_sim_pandemic_expect_wait = mean(c_sim_pandemic_expect_wait, 1);
    mean_search_sim_pandemic_expect = mean(search_sim_pandemic_expect, 1);

    mean_c_sim_e_bywage(iy, :) = mean_c_sim_e;

    %paste on initial Jan-March 3 months of employment
    mean_c_sim_pandemic_expect_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect(1:numsim - 3)];
    mean_c_sim_pandemic_expect_wait_bywage(iy, :) = [mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_wait(1:numsim - 3)];

    mean_y_sim_pandemic_u_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 1)'];
    mean_y_sim_pandemic_u_bywage(iy,4)=mean_y_sim_pandemic_u_bywage(iy,4);
    mean_y_sim_pandemic_u_bywage(iy, 9) = mean_y_sim_pandemic_u_bywage(iy, 9) + LWAsize;
    mean_y_sim_pandemic_u_bywage(iy, 13) = mean_y_sim_pandemic_u_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    mean_y_sim_pandemic_wait_bywage(iy, :) = [y y y benefit_profile_pandemic(:, 3)'];
    mean_y_sim_pandemic_wait_bywage(iy,4)=mean_y_sim_pandemic_wait_bywage(iy,4);
    mean_y_sim_pandemic_wait_bywage(iy, 9) = mean_y_sim_pandemic_wait_bywage(iy, 9) + LWAsize;
    mean_y_sim_pandemic_wait_bywage(iy, 13) = mean_y_sim_pandemic_wait_bywage(iy, 13) + 1500 * FPUC_expiration / (4.5 * 600);
    mean_y_sim_e_bywage(iy,:)=y*ones(16,1);
    mean_y_sim_e_bywage(iy,4)=mean_y_sim_e_bywage(iy,4);

    mean_search_sim_pandemic_expect_bywage(iy, :) = [NaN NaN NaN mean_search_sim_pandemic_expect(1:numsim - 3)];
end

% Average over wage groups
mean_y_sim_pandemic_u = mean(mean_y_sim_pandemic_u_bywage, 1);
mean_y_sim_pandemic_wait = mean(mean_y_sim_pandemic_wait_bywage, 1);
mean_y_sim_e = mean(mean_y_sim_e_bywage, 1);
mean_c_sim_e = mean(mean_c_sim_e_bywage, 1);
mean_c_sim_pandemic_expect = mean(mean_c_sim_pandemic_expect_bywage, 1);
mean_c_sim_pandemic_expect_wait = mean(mean_c_sim_pandemic_expect_wait_bywage, 1);
mean_search_sim_pandemic_expect = mean(mean_search_sim_pandemic_expect_bywage, 1);

% Load data for MPC calculation and plotting
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

idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid >= 201901;
data_update_e = data_update(idx_emp, :);
income_e = data_update_e.value;
income_e = income_e(13:end);

% Waiting MPC
scale_factor=(total_spend_e(1)/income_e(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));
mpc_expect_waiting = ((mean_c_sim_pandemic_expect(5) - mean_c_sim_pandemic_expect(3)) - (mean_c_sim_pandemic_expect_wait(5) - mean_c_sim_pandemic_expect_wait(3))) / ((mean_y_sim_pandemic_u(5) - mean_y_sim_pandemic_u(3)) - (mean_y_sim_pandemic_wait(5) - mean_y_sim_pandemic_wait(3)));
mpc_expect_waiting = scale_factor * mpc_expect_waiting;

% Save to disk (individually for each type)
runtime = toc;
save([path_array_job, '/type_', z.jobname, '_task', num2str(z.taskid), ...
    '_k', num2str(i_k), '_g', num2str(i_gamma), '_c', num2str(i_c_param), ...
    '.mat'], ...
    'beta_normal', 'delta', 'k', 'gamma', 'c_param', ...
    'mean_c_sim_e', 'mean_c_sim_pandemic_expect', ...
    'mean_search_sim_pandemic_expect', 'mpc_expect_waiting', 'runtime')


toc

% % Plot for calibrating April discount factor shock
% % --> target consumption drop of employed group
% mean_c_sim_e_dollars_level = mean_c_sim_e/mean(mean_c_sim_e(3)) * mean(total_spend_e(3));
% figure
% plot(1:14, mean_c_sim_e_dollars_level(1:14), 1:14, total_spend_e(1:14), '--', 'LineWidth', 2)

end
end
end
