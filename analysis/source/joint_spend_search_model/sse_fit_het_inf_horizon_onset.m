function fit = sse_fit_het_inf_horizon_onset(pars, surprise)

    rng('default')

    global monthly_search_data infinite_dur dt initial_a mu r sep_rate repshare FPUC_onset n_aprime n_b n_ben_profiles_allowed aprimemin aprimemax w exog_find_rate beta_normal beta_high use_initial_a

    load bestfit_prepandemic.mat

    k_prepandemic = pre_pandemic_fit_match500MPC(1);
    gamma_prepandemic = pre_pandemic_fit_match500MPC(2);
    c_param_prepandemic = 0;

    EIP2 = 1200;
    EIP2_e=600;
    EIP3 = 4000;

    k = pars(1);
    gamma = pars(2);
    c_param = pars(3);

    for iy = 1:5

        y = w(iy);
        h = .7 * y;
        b = repshare * y;

        % Aprime grid
        Aprime = exp(linspace(0.00, log(aprimemax), n_aprime)) - 1;
        Aprime = Aprime';

        %regular benefits profile
        benefit_profile(1:6, 1) = h + b;

        if infinite_dur == 1
            benefit_profile(7:13, 1) = h + b;
        else
            benefit_profile(7:13, 1) = h;
        end

    %{
    %expect $300 for 8 months
    benefit_profile_pandemic(1:8,1)=b+h+FPUC_onset;
    benefit_profile_pandemic(9:12,1)=b+h;
    if infinite_dur==1
        benefit_profile_pandemic(13,1)=b+h;
    else
        benefit_profile_pandemic(13,1)=h;
    end
    %expect $300 for 3 months
    benefit_profile_pandemic(1:3,2)=b+h+FPUC_onset;
    benefit_profile_pandemic(4:12,2)=b+h;
    if infinite_dur==1
        benefit_profile_pandemic(13,2)=b+h;
    else
        benefit_profile_pandemic(13,2)=h;
    end
    %}

        %expect $300 for 8 months
        benefit_profile_pandemic(1:2, 1) = b + h;
        benefit_profile_pandemic(3:12, 1) = b + h + FPUC_onset;

        if infinite_dur == 1
            benefit_profile_pandemic(13:13, 1) = b + h;
        else
            benefit_profile_pandemic(13:13, 1) = h;
        end

        %expect $300 for 3 months
        benefit_profile_pandemic(1:2, 2) = b + h;
        benefit_profile_pandemic(3:5, 2) = b + h + FPUC_onset;
        benefit_profile_pandemic(6:10, 2) = b + h;

        if infinite_dur == 1
            benefit_profile_pandemic(11:13, 2) = b + h + FPUC_onset;
        else
            benefit_profile_pandemic(11:13, 2) = h;
        end

        %No FPUC_onset
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

                    tmp = min(1 - recall_probs_regular(ib), max(0, (beta * (v_e_guess(:) - v_u_guess(:, min(ib + 1, n_b))) / k_prepandemic).^(1 / gamma_prepandemic)));
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

                        tmp = min(1 - recall_probs_pandemic(ib), max(0, (beta * (v_e_guess(:) - v_u_pandemic_guess(:, min(ib + 1, n_b), i_ben_profile)) / k).^(1 / gamma)));
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

        %Note that we don't necessarily need all parts of this simulation step to
        %be internal to the parameter search, keeping only the absolute necessary
        %parts internal to that loop should speed things up some

        %note also i might be able to speed up by feeding only the adjacent points
        %into the interp step

        numhh = 500;
        numsim = 15;
        burnin = 10;
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
            a_sim_pandemic_surprise = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_expect = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_surprise_wait = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0) + initial_a;
            a_sim_regular = tmp_a(tmp_u > 0) + initial_a;
            a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0) + initial_a;
            a_sim_e = tmp_a(tmp_u == 0) + initial_a;

            load a_init_sse
            initial_a_vec=a_init_sse;
            initial_a_tmp=initial_a_vec(iy);

            a_sim_pandemic_surprise = initial_a_tmp;
            a_sim_pandemic_expect = initial_a_tmp;
            a_sim_pandemic_surprise_wait = initial_a_tmp;
            a_sim_pandemic_expect_wait = initial_a_tmp;
            a_sim_regular = initial_a_tmp;
            a_sim_pandemic_noFPUC = initial_a_tmp;
            a_sim_e = 0.5867*w(iy);
        else
            a_sim_pandemic_surprise = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect = tmp_a(tmp_u > 0);
            a_sim_pandemic_surprise_wait = tmp_a(tmp_u > 0);
            a_sim_pandemic_expect_wait = tmp_a(tmp_u > 0);
            a_sim_regular = tmp_a(tmp_u > 0);
            a_sim_pandemic_noFPUC = tmp_a(tmp_u > 0);
            a_sim_e = tmp_a(tmp_u == 0);
        end

        num_unemployed_hh = length(a_sim_pandemic_surprise);
        num_employed_hh = length(a_sim_e);
        c_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_pandemic_expect = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_regular = zeros(length(a_sim_pandemic_surprise), 30);
        c_sim_e = zeros(length(a_sim_e), 30);

        search_sim_pandemic_surprise = zeros(length(a_sim_pandemic_surprise), 30);
        search_sim_pandemic_expect = zeros(length(a_sim_pandemic_expect), 30);
        search_sim_regular = zeros(length(a_sim_regular), 30);

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
                end

                if t == 5
                    a_sim_pandemic_expect(i, t) = a_sim_pandemic_expect(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                    a_sim_pandemic_surprise(i, t) = a_sim_pandemic_surprise(i, t) + EIP3 * FPUC_onset / (4.5 * 600);
                end

                if t == 1
                    c_pol_u = c_pol_u_betanormal;
                    c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
                end

                c_sim_regular(i, t) = interp1(A, c_pol_u(:, length_u), a_sim_regular(i, t), 'linear');
                a_sim_regular(i, t + 1) = max(benefit_profile(length_u) + (1 + r) * a_sim_regular(i, t) - c_sim_regular(i, t), 0);

                if t <= 2
                    c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_expect(i, t), 'linear');
                    a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);
                else
                    c_sim_pandemic_expect(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_expect(i, t), 'linear');
                    a_sim_pandemic_expect(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_expect(i, t) - c_sim_pandemic_expect(i, t), 0);
                end

                if t <= 2 %pre-onset
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                elseif t >= 3 && t <= 4 %jan-feb
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 2), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 2) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                else %march extension
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                end

                diff_v = interp1(A, v_e(:), a_sim_regular(i, t + 1), 'linear') - interp1(A, v_u(:, min(length_u + 1, n_b)), a_sim_regular(i, t + 1), 'linear');
                search_sim_regular(i, t) = min(1 - recall_probs_regular(ib), max(0, (beta * (diff_v) / k_prepandemic).^(1 / gamma_prepandemic)));

                if t <= 2 %pre-onset
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_expect(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_expect(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_expect(i, t + 1), 'linear');
                end

                search_sim_pandemic_expect(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));

                if imag(search_sim_pandemic_expect(i, t)) ~= 0
                    search_sim_pandemic_expect(i, t) = 0;
                end

                if t <= 2
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_surprise(i, t + 1), 'linear');
                elseif t >= 3 && length_u <= 4
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

        % mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
        % mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);

        % mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
        % mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);

        % mean_a_sim_regular=mean(a_sim_regular,1);
        % mean_c_sim_regular=mean(c_sim_regular,1);

        mean_search_sim_regular_bywage(iy, :) = mean(search_sim_regular, 1);
        mean_search_sim_pandemic_expect_bywage(iy, :) = mean(search_sim_pandemic_expect, 1);
        mean_search_sim_pandemic_surprise_bywage(iy, :) = mean(search_sim_pandemic_surprise, 1);
    end

    mean_search_sim_pandemic_surprise = mean(mean_search_sim_pandemic_surprise_bywage, 1);
    mean_search_sim_pandemic_expect = mean(mean_search_sim_pandemic_expect_bywage, 1);

    %weekly_search=[0.0074083963;0.0042134342;0.0033943381;0.0076518236;0.0058785966;0.0072370907;0.011853259;0.0097451219;0.0067729522;0.0083376039;0.0088560889;0.0092052780;0.0091834739;0.010498224;0.011711085;0.011793635;0.013426980;0.021680148;0.021625612;0.019983901;0.018380599;0.020592216;0.020819575;0.018611545;0.020868009;0.022053968;0.021287521;0.022725264;0.021265257;0.021397851;0.019634541;0.021541588;0.020100573;0.020108875;0.015584604;0.022505494;0.020821054;0.024468435;0.022326604;0.033451892;0.021188661;0.020094011;0.018982559;0.016847268;0.017690072;0.016456312;0.013985096;0.016348185;0.014748414;0.018534498;0.042390950;0.049047738;0.036877677];
    %weekly_search(40)=weekly_search(42);
    %weekly_search(41)=weekly_search(42);
    %for i=1:length(weekly_search)-4
    %    monthly_search_data(i)=1-(1-weekly_search(i))*(1-weekly_search(i+1))*(1-weekly_search(i+2))*(1-weekly_search(i+3));
    %end
    %monthly_search_data=monthly_search_data([1 5 9 13 18 22 27 31 35 40 44 48]);

    %from monthly search branch:
    %monthly_search_data=[
    %0.0331803567122787	0.0855436187739731
    %0.0269581852915193	0.0915464765056637
    %0.0325662545281693	0.0785480411247848
    %0.0369176195328030	0.0643146519295299
    %0.0484119134584009	0.0706906085654556
    %];
    %monthly_search_data=monthly_search_data(:,2)';

    if surprise == 1
        fit = sum((((mean_search_sim_pandemic_surprise(1:4) - monthly_search_data(1:4)) ./ (.5 * monthly_search_data(1:4) + .5 * mean_search_sim_pandemic_surprise(1:4)))).^2);
        %fit=fit+((mean_search_sim_pandemic_surprise(7)-monthly_search_data(7))/(.5*monthly_search_data(7)+.5*mean_search_sim_pandemic_surprise(7)))^2;
    else
        fit = sum((((mean_search_sim_pandemic_expect(1:4) - monthly_search_data(1:4)) ./ (.5 * monthly_search_data(1:4) + .5 * mean_search_sim_pandemic_expect(1:4)))).^2);
        %fit=fit+((mean_search_sim_pandemic_expect(7)-monthly_search_data(7))/(.5*monthly_search_data(7)+.5*mean_search_sim_pandemic_expect(7)))^2;
    end

end