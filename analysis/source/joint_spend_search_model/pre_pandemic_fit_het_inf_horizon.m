%This function takes a set of parameters, and solves for search with and
%without a small increase in benefits to compute a duration elasticity and
%an average job finding rate, and these are compared to the data values
function fit = pre_pandemic_fit_het_inf_horizon(pars, preperiod_target, infinite_dur, include_recalls)

    rng('default')

    global monthly_search_data dt initial_a mu r sep_rate repshare FPUC n_aprime n_b aprimemin aprimemax w exog_find_rate beta_normal beta_high use_initial_a

    % We impose c, the search cost intercept is zero
    k = pars(1);
    gamma = pars(2);
    c_param=0;

    % Loop over wage groups
    for iy = 1:5

        y = w(iy); %income when employed
        h = .7 * y; %home production value (will last as long as unemployed)
        b = repshare * y; %regular benefit level (lasts 12 months)

        % Aprime grid
        Aprime = exp(linspace(0.00, log(aprimemax), n_aprime)) - 1;
        Aprime = Aprime';

        %Always get h when unemployed. The first 12 months also get b. Then
        %lose benefits as the absorbing state when infinite_dur==false (all
        %recent versions of the code set infinite_dur=false)
        benefit_profile(1:12, 1) = h + b;
        if infinite_dur == true
            benefit_profile(13:13, 1) = h + b;
        else
            benefit_profile(13:13, 1) = h;
        end

        % Augmented benefits profile with an additional small benefit for
        % calculating duration elasticity
        benefit_profile_augmented(1:12, 1) = h + b + 0.01 * b;
        if infinite_dur == true
            benefit_profile_augmented(13:13, 1) = h + b + 0.01 * b;
        else
            benefit_profile_augmented(13:13, 1) = h;

        end

        % Recall setting
        if include_recalls == true
            %when recalls are turned on (which is the case in all recent
            %versions of the code), households assume a constant recall
            %rate in their expectations
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
                %this is actually looping over a not over aprime, just 
                %using the same grid for both. This loop will start from
                %the smallest value of aprime and then iterate executing
                %the else command until the first instance the constraint
                %doesn't bind, in which case all large values of aprime we 
                %know are also unconstrained and so the vector is filled
                %and loop is broken with the break command
                for ia = 1:n_aprime  
                    if Aprime(ia) > a_star1_e %When unconstrained this is optimal C.
                        %With EGM we have C on a grid of a' but want to get on
                        %a grid of a. Need to interpolate:
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
                %loop over each benefit profile
                for ib = 1:n_b
                    %Computing optimal search given current value function
                    %guesses for continuation values
                    %Two adjustments are necessary: 1) Search here (the
                    %endogenous part) is bounded above by 1-recall_probs so
                    %that total search will be bounded by 1)
                    %2) set optimal search to 0 in situations where v_u_guess
                    %exceeds v_e_guess in which case tmp may have imaginary
                    %roots. 
                    tmp = min(1-recall_probs(ib), max(0, (beta * (v_e_guess(:) - v_u_guess(:, min(ib + 1, n_b))) / k).^(1 / gamma)));
                    tmp(imag(tmp) ~= 0) = 0;
                    optimal_search(:, ib) = tmp;

                    %RHS of optimality condition computes expected marginal
                    %utility where probability of being employed or
                    %unemployed depends on current search and recall rates
                    rhs_u(:, ib) = beta * (1 + r) * ((recall_probs(ib) + optimal_search(:, ib)) .* c_pol_e_guess(:).^(-mu) + (1 - optimal_search(:, ib) - recall_probs(ib)) .* c_pol_u_guess(:, min(ib + 1, n_b)).^(-mu));
                    c_tilde_u(:, ib) = (rhs_u(:, ib)).^(-1 / mu); %unconstrained
                    a_star_u(:, ib) = (Aprime(:) + c_tilde_u(:, ib) - benefit_profile(ib)) / (1 + r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                    a_star1_u(ib) = (c_tilde_u(1, ib) - benefit_profile(ib)) / (1 + r); %a implied by the budget constraint when it binds
                end
                
                %loop over each benefit profile
                for ib = 1:n_b
                    %this is actually looping over a not over aprime, just 
                    %using the same grid for both. This loop will start from
                    %the smallest value of aprime and then iterate executing
                    %the else command until the first instance the constraint
                    %doesn't bind, in which case all large values of aprime we 
                    %know are also unconstrained and so the vector is filled
                    %and loop is broken with the break command
                    for ia = 1:n_aprime
                        if Aprime(ia) > a_star1_u(ib) %When unconstrained this is optimal C.
                            c_pol_u(ia:end, ib) = interp1(a_star_u(:, ib), c_tilde_u(:, ib), Aprime(ia:end), 'linear', 'extrap');
                            break
                        else  %When constrained, optimal C given by budget constraint
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
                % Augmented benefits unemployed with slightly higher b level
                %%%%%%%%%%%%%%%%%%%%%%%
                %See comments for the solution for the unemployed above,
                %all the logic is identical, we are just resolving for a
                %slightly different benefit profile now augmented by .01

                for ib = 1:n_b
                    tmp = min(1-recall_probs(ib), max(0, (beta * (v_e_guess(:) - v_u_augmented(:, min(ib + 1, n_b))) / k).^(1 / gamma)));
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
                        else
                            c_pol_u_augmented(ia, ib) = (1 + r) * Aprime(ia) + benefit_profile_augmented(ib) + r * aprimemin;
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
                %These will be the policy functions with the temporary
                %discount factor shock:
                c_pol_e_betahigh = c_pol_e;
                c_pol_u_betahigh = c_pol_u;
                c_pol_u_augmented_betahigh = c_pol_u_augmented;

                v_e_betahigh = v_e;
                v_u_betahigh = v_u;
                v_u_augmented_betahigh = v_u_augmented;
            end


        %this ends the solution part of the code for wage group iy
        %(this end loop is for the two discount factors)
        end 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Now simulating the model for the particular wage group given by iy
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


        %Simulate for 500 households and save 15 months worth of outcomes
        numhh = 500;
        numsim = 15;
        % Droppping some initial periods to reduce sensitivity to initial conditions
        burnin = 15;
        %initialization of simulated variables
        a_sim = zeros(numhh, burnin + 1);
        c_sim = zeros(numhh, burnin + 1);
        e_sim = zeros(numhh, burnin + 1);
        u_dur_sim = zeros(numhh, burnin + 1);
        % First sim with initial assets all employed
        a_sim(:, 1) = initial_a;
        e_sim(:, 1) = 1;

        %Policy functions to use (use the ones without discount factor
        %shock except in april 2020... here just doing burnin sims)
        c_pol_e = c_pol_e_betanormal;
        c_pol_u = c_pol_u_betanormal;
        c_pol_u_augmented = c_pol_u_augmented_betanormal;

        %Value functions to use for computing optimal search
        v_e = v_e_betanormal;
        v_u = v_u_betanormal;
        v_u_augmented = v_u_augmented_betanormal;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Loop over the initial burnin part of the simulation. This is just
        % to initialize variables for the next block of sims, so we don't
        % care much about this part of the sim
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for t = 1:burnin

            % Loop over the households
            for i = 1:numhh
                % Get consumption and assets based on employment status
                if e_sim(i, t) == 1
                    c_sim(i, t) = interp1(A, c_pol_e(:), a_sim(i, t), 'linear');
                    a_sim(i, t + 1) = y + (1 + r) * a_sim(i, t) - c_sim(i, t);
                else
                    c_sim(i, t) = interp1(A, c_pol_u(:, u_dur_sim(i, t)), a_sim(i, t), 'linear');
                    %The max here just deals with tiny rounding error that
                    %sometimes makes a_sim trivially negative in the
                    %interpolation step and then breaks things on next
                    %iteration
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

                %If currently unemployed use exogenous job find rate to 
                %determine next period For simplicity for the burnin sim
                %(which is just initializing initial conditions) just use
                %the exogenous
                else
                    if randy < exog_find_rate
                        e_sim(i, t + 1) = 1;
                        u_dur_sim(i, t + 1) = 0;
                    else
                        e_sim(i, t + 1) = 0;
                        u_dur_sim(i, t + 1) = min(u_dur_sim(i, t) + 1, n_b);
                    end
                end

            end %ends loop over hh
        end %ends loop over time period
        %Save the value of assets and u and e states immediately after the
        %burnin period
        tmp_a = a_sim(:, burnin + 1);
        tmp_u = u_dur_sim(:, burnin + 1);
        tmp_e = e_sim(:, burnin + 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %This is the end of the burnin block
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %This starts simulations we actually care about
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Initializing assets, consumption, employment, unemployment
        %duration matrices
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
            a_sim_augmented = tmp_a(tmp_u > 0);
            a_sim_augmented_wait = tmp_a(tmp_u > 0);
            a_sim_regular = tmp_a(tmp_u > 0);
            a_sim_augmented_noFPUC = tmp_a(tmp_u > 0);
            a_sim_e = tmp_a(tmp_u == 0);
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

        %Starting sims with a newly unemployed household
        length_u = 0;
        % Loop over time unemployed (we will focus on continuously
        % unemployed households who do not exit unemployment)
        for t = 1:15
            length_u = min(length_u + 1, n_b);

            % Loop over households
            for i = 1:num_unemployed_hh
                %For prepandemic model calibration we don't have a
                %discount factor shock so just work with betanormal
                %policy functions
                c_pol_u = c_pol_u_betanormal;
                c_pol_u_augmented = c_pol_u_augmented_betanormal;
                v_e = v_e_betanormal;
                v_u = v_u_betanormal;
                v_u_augmented = v_u_augmented_betanormal;
                beta=beta_normal;

                %Use consumption and assets functions to get regular unemployed values
                c_sim_regular(i, t) = interp1(A, c_pol_u(:, length_u), a_sim_regular(i, t), 'linear');
                a_sim_regular(i, t + 1) = max(benefit_profile(length_u) + (1 + r) * a_sim_regular(i, t) - c_sim_regular(i, t),0);

                %Also do for augmented values
                c_sim_augmented(i, t) = interp1(A, c_pol_u_augmented(:, length_u), a_sim_augmented(i, t), 'linear');
                a_sim_augmented(i, t + 1) = max(benefit_profile_augmented(length_u, 1) + (1 + r) * a_sim_augmented(i, t) - c_sim_augmented(i, t),0);

                %Difference in value functions, regular case
                diff_v = interp1(A, v_e(:), a_sim_regular(i, t + 1), 'linear') - interp1(A, v_u(:, min(length_u + 1, n_b)), a_sim_regular(i, t + 1), 'linear');
                %Bound search at one
                search_sim_regular(i, t) = min(1 - recall_probs(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                %Imaginary search set to zero (just as discussed above,
                %this corresponds to cases where V_U might exceed V_E, just
                %allowing for this possibility since this whole code is
                %embedded in a numerical search over search parameters and
                %we don't want it to break if those search parameters lead
                %to this case
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

            end %ends loop over hh
        end %ends loop over length of unemployment

        %Mean search, assets and spending for the wage group
        %We care mostly about search in this case since that is what we are
        %simulating inside our numerical optimization loop to minimize
        %distance vs. data (i.e. targeting a duration elasticity and prepan
        %job find rate)
        mean_search_sim_regular_bywage(iy, :) = mean(search_sim_regular, 1);
        mean_search_sim_augmented_bywage(iy, :) = mean(search_sim_augmented, 1);

        
        mean_a_sim_regular_bywage(iy, :) = mean(a_sim_regular, 1);
        mean_c_sim_regular_bywage(iy, :) = mean(c_sim_regular, 1);
    end %ends the loop over wage groups
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %This is the end of all the separate results by wage group iy
    %The rest of the code is combining results across groups and comparing
    %to the data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    % Mean search across wage groups
    mean_search_sim_regular = mean(mean_search_sim_regular_bywage, 1);
    mean_search_sim_augmented = mean(mean_search_sim_augmented_bywage, 1);

    % Target elasticity of 0.5 and the monthly pre-period job finding rate

    % Produce elasticity
    % Need to create the total exit rate since we are targeting a total
    % duration elasticity
    total_exit_rate_regular = mean_search_sim_regular(1:length(recall_probs))' + recall_probs';
    total_exit_rate_augmented = mean_search_sim_augmented(1:length(recall_probs))' + recall_probs';
    %The percentage change in benefits will be 0.01 (augmentation amount)
    perc_change_benefits = 0.01;
    %Calculate the elasticity (targeting a value of 0.5)
    elasticity = (average_duration(total_exit_rate_augmented) / average_duration(total_exit_rate_regular) - 1) / perc_change_benefits;

    %Search is not constant over time (since benefits last 12 months) and
    %so we calculate the average job finding rate weighting search at a
    %given duration of unemployment by the share of unemployed households
    %with that duration of unemployment
    weights(1)=1;
    for dur=2:numsim
        weights(dur)=prod(1-mean_search_sim_regular(1:dur-1));
    end
    
    %Compile total pre-period job-finding
    %preperiod_jobfind = mean(mean_search_sim_regular(1:numsim));
    preperiod_jobfind =sum(weights.*mean_search_sim_regular(1:numsim))/sum(weights);

    % Fit is based of mean square deviations from elasticity and job-finding
    fit = ((elasticity - 0.5) / 0.5)^2 + ((preperiod_jobfind - preperiod_target) / preperiod_target)^2;

end
