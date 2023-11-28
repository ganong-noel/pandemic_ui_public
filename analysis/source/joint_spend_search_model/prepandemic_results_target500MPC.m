display('Simulating Prepandemic Model Effects of $600')
clearvars -except -regexp fig_paper_*
tic

load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load hh_wage_groups.mat

load bestfit_prepandemic.mat



sse_expect_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_expect_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_expect_fit_het_match500MPC(3)=0;

sse_surprise_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_surprise_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_surprise_fit_het_match500MPC(3)=0;


data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
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


shock_500=500/income_e(1);

% Assign parameter values
load discountfactors.mat
beta_normal = beta_target500MPC;
beta_high = beta_oneperiodshock;

load model_parameters.mat
initial_a = initial_a - aprimemin;

n_ben_profiles_allowed = 5; %This captures the surprise vs. expected expiration scenarios w/ wait or no delay

% Set on/off switches
infinite_dur = 0;
use_initial_a = 0;

% Start solving the model with EGM
for iy=1:5

    y = w(iy);
    h = 0.7 * y;
    b = repshare * y;

    for surprise=0:1

    rng('default')

    % Set search cost parameters
    if surprise==1
        k=sse_surprise_fit_het_match500MPC(1);
        gamma=sse_surprise_fit_het_match500MPC(2);
        c_param=sse_surprise_fit_het_match500MPC(3);
    else
        k=sse_expect_fit_het_match500MPC(1);
        gamma=sse_expect_fit_het_match500MPC(2);
        c_param=sse_expect_fit_het_match500MPC(3);
    end
    
   

    % Aprime grid
    aprimemax = 2000;
    Aprime = linspace(0,aprimemax,n_aprime);
    Aprime=Aprime';

    Aprime=exp(linspace(0,log(aprimemax),n_aprime))-1;
    Aprime=Aprime';



    %regular benefits profile
    benefit_profile(1:6,1)=h+b;
    if infinite_dur==1
        benefit_profile(7:13,1)=h+b;
    else
        benefit_profile(7:13,1)=h;
    end

    %expect $600 for 4 months
    benefit_profile_pandemic(1:4,1)=b+h+FPUC_expiration;
    benefit_profile_pandemic(5:12,1)=b+h;
    if infinite_dur==1
        benefit_profile_pandemic(13,1)=b+h;
    else
        benefit_profile_pandemic(13,1)=h;
    end
    %expect $600 for 12 months
    benefit_profile_pandemic(1:12,2)=b+h+FPUC_expiration;
    %benefit_profile_pandemic(1)=b+h+1.5*FPUC_expiration;
    %benefit_profile_pandemic(3)=b+h+1.05*FPUC_expiration;
    %benefit_profile_pandemic(4)=b+h+1.25*FPUC_expiration;
    %benefit_profile_pandemic(5:12,2)=b+h+1*FPUC_expiration;
    if infinite_dur==1
        benefit_profile_pandemic(13,2)=b+h+FPUC_expiration;
    else
        benefit_profile_pandemic(13,2)=h;
    end



    %Matching actual income profile for waiting group
    %expect $600 for 4 months, but w/ 2 month wait
    benefit_profile_pandemic(1,3)=1.19*h;
    benefit_profile_pandemic(2,3)=h;
    benefit_profile_pandemic(3,3)=h+2.35*(b+FPUC_expiration);
    benefit_profile_pandemic(4,3)=h+b+FPUC_expiration;
    benefit_profile_pandemic(5:12,3)=b+h;
    if infinite_dur==1
        benefit_profile_pandemic(13,3)=b+h;
    else
        benefit_profile_pandemic(13,3)=h;
    end
    %expect $600 for 12 months, but w/ 2 month wait
    benefit_profile_pandemic(1,4)=1.19*h;
    benefit_profile_pandemic(2,4)=h;
    benefit_profile_pandemic(3,4)=h+2.35*(b+FPUC_expiration);
    benefit_profile_pandemic(4:12,4)=b+h+FPUC_expiration;
    if infinite_dur==1
        benefit_profile_pandemic(13,4)=b+h+FPUC_expiration;
    else
        benefit_profile_pandemic(13,4)=h;
    end

    %No FPUC
    benefit_profile_pandemic(1:12,5)=h+b;
    if infinite_dur==1
        benefit_profile_pandemic(13,5)=h+b;
    else
        benefit_profile_pandemic(13,5)=h;
    end

    %benefit_profile_pandemic(1,:)=benefit_profile_pandemic(1,:)+350*FPUC_expiration/(4.5*600);



    recall_probs_pandemic(1:13,1)=0.00;
    recall_probs_regular=recall_probs_pandemic;


    recall_probs_pandemic_actual(1)=.0078;
    recall_probs_pandemic_actual(2)=.113;
    recall_probs_pandemic_actual(3)=.18;
    recall_probs_pandemic_actual(4)=.117;
    recall_probs_pandemic_actual(5)=.112;
    recall_probs_pandemic_actual(6:13)=.107;
    
    recall_probs_pandemic(1:13)=.08;
    recall_probs_regular=recall_probs_pandemic;


    %initialization of variables for speed
    c_pol_e=zeros(n_aprime,1);
    c_pol_u=zeros(n_aprime,n_b,1);
    c_pol_u_pandemic=zeros(n_aprime,n_b,n_ben_profiles_allowed);
    v_e=c_pol_e;
    v_u=c_pol_u;
    v_u_pandemic=c_pol_u_pandemic;

    rhs_e=zeros(n_aprime,1);
    rhs_u=zeros(n_aprime,n_b);
    rhs_u_pandemic=zeros(n_aprime,n_b,n_ben_profiles_allowed);

    for beta_loop=1:2

        if beta_loop==1
            beta = beta_normal;
        elseif beta_loop==2
            beta = beta_high;
        end



    %Iteration counter
    iter=0;
    % Set tolerance for convergence
    tol=1e-4;
    tol_percent=0.0001;
    % Initialize difference in consumption from guess and new
    diffC=tol + 1;
    diffC_percent=tol_percent+1;    

    tol_s=1e-3;
    tol_c_percent=.05;

    ave_change_in_C_percent=100;
    ave_change_in_S=100;

    % Initial guesses
    c_pol_e_guess(:)=y(1)+Aprime(:)*(1+r)+r*aprimemin;
    v_e_guess(:)=((c_pol_e_guess(:)).^(1-mu)-1)/(1-mu);
    optimal_search_guess=zeros(n_aprime,n_b);
    optimal_search_pandemic_guess=zeros(n_aprime,n_b,n_ben_profiles_allowed);
    for ib=1:n_b
        c_pol_u_guess(:,ib)=benefit_profile(ib)+Aprime(:)*(1+r)+r*aprimemin;
        v_u_guess(:,ib)=((c_pol_u_guess(:,ib)).^(1-mu)-1)/(1-mu)- (k * 0^ (1 + gamma)) / (1 + gamma) + c_param;
    end
    for i_ben_profile=1:n_ben_profiles_allowed
        for ib=1:n_b
            c_pol_u_pandemic_guess(:,ib,i_ben_profile)=benefit_profile_pandemic(ib,i_ben_profile)+Aprime(:)*(1+r)+r*aprimemin;
            v_u_pandemic_guess(:,ib,i_ben_profile)=((c_pol_u_pandemic_guess(:,ib,i_ben_profile)).^(1-mu)-1)/(1-mu)- (k * 0^ (1 + gamma)) / (1 + gamma) + c_param;
        end
    end

    
    %c_pol is c(a,y)
    %c_tilde is c(a',y)
    %while (iter <= 5000) & (diffC > tol) %& (diffC_percent > tol_percent) %| diffV > tol)

    if beta_loop==2
        maxiter=1; %this effectively governs how many periods households will think the high discount factor will last, setting maxiter=1 essentially runs one backward induction step from the beta_normal solutions
                   %note that the code must be structured so that it solves the
                   %beta_normal part first
        c_pol_e_guess=c_pol_e_betanormal;
        c_pol_u_guess=c_pol_u_betanormal;
        c_pol_u_pandemic_guess=c_pol_u_pandemic_betanormal;

        v_e_guess=v_e_betanormal;
        v_u_guess=v_u_betanormal;
        v_u_pandemic_guess=v_u_pandemic_betanormal;


    else
        maxiter=1000;
    end

    while ((ave_change_in_C_percent>tol_c_percent) || (ave_change_in_S>tol_s) ) && iter<maxiter

    %while iter<199
        %employed

        rhs_e(:)=beta*(1+r)*((1-sep_rate)*c_pol_e_guess(:).^(-mu)+sep_rate*c_pol_u_guess(:,1).^(-mu));
        c_tilde_e(:)=(rhs_e(:)).^(-1/mu); %unconstrained
        a_star_e(:)=(Aprime(:)+c_tilde_e(:)-y(1))/(1+r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
        a_star1_e=(c_tilde_e(1)-y(1))/(1+r);
        %time1(t)=toc;
        %tic;
        for ia=1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
            if Aprime(ia)>a_star1_e
                c_pol_e(ia:end)=interp1(a_star_e(:),c_tilde_e(:),Aprime(ia:end),'linear','extrap');
                break
            else
                c_pol_e(ia)=(1+r)*Aprime(ia)+y+r*aprimemin;
            end
        end

        a_prime_holder=(1+r)*Aprime+y(1)-c_pol_e(:);
        v_e(:)=((c_pol_e(:)).^(1-mu)-1)/(1-mu)+beta*((1-sep_rate).*interp1(Aprime,v_e_guess(:),a_prime_holder,'linear','extrap')+sep_rate*interp1(Aprime,v_u_guess(:,1),a_prime_holder,'linear','extrap'));
        %time2(t)=toc;


       %unemployed 

       %tic;
       for ib=1:n_b

            tmp=min(1-recall_probs_regular(ib), max(0,(beta*(v_e_guess(:)-v_u_guess(:,min(ib+1,n_b))) / k).^(1 / gamma)));
            tmp(imag(tmp)~=0)=0;
            optimal_search(:,ib)=tmp;
            rhs_u(:,ib)=beta*(1+r)*((recall_probs_regular(ib)+optimal_search(:,ib)).*c_pol_e_guess(:).^(-mu)+(1-optimal_search(:,ib)-recall_probs_regular(ib)).*c_pol_u_guess(:,min(ib+1,n_b)).^(-mu));
            c_tilde_u(:,ib)=(rhs_u(:,ib)).^(-1/mu); %unconstrained
            a_star_u(:,ib)=(Aprime(:)+c_tilde_u(:,ib)-benefit_profile(ib))/(1+r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
            a_star1_u(ib)=(c_tilde_u(1,ib)-benefit_profile(ib))/(1+r);
        end
        %time3(t)=toc;
        %tic;
        for ib=1:n_b
            for ia=1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
                if Aprime(ia)>a_star1_u(ib)
                    c_pol_u(ia:end,ib)=interp1(a_star_u(:,ib),c_tilde_u(:,ib),Aprime(ia:end),'linear','extrap');
                    break
                else
                    c_pol_u(ia,ib)=(1+r)*Aprime(ia)+benefit_profile(ib)+r*aprimemin;
                end
            end
        end
        for ib=1:n_b
            a_prime_holder_u(:,ib)=(1+r)*Aprime+benefit_profile(ib)-c_pol_u(:,ib);
            v_u(:,ib)=((c_pol_u(:,ib)).^(1-mu)-1)/(1-mu)- (k * optimal_search(:,ib).^ (1 + gamma)) / (1 + gamma) + c_param+beta*((optimal_search(:,ib)+recall_probs_regular(ib)).*interp1(Aprime,v_e_guess(:),a_prime_holder_u(:,ib),'linear','extrap')+(1-optimal_search(:,ib)-recall_probs_regular(ib)).*interp1(Aprime,v_u_guess(:,min(ib+1,n_b)),a_prime_holder_u(:,ib),'linear','extrap'));
        end



       %pandemic unemployed 
       for i_ben_profile=1:n_ben_profiles_allowed

            %tic;
           for ib=1:n_b

                tmp=min(1-recall_probs_pandemic(ib), max(0,(beta*(v_e_guess(:)-v_u_pandemic_guess(:,min(ib+1,n_b),i_ben_profile)) / k).^(1 / gamma)));
                tmp(imag(tmp)~=0)=0;
                optimal_search_pandemic(:,ib,i_ben_profile)=tmp;
                rhs_u_pandemic(:,ib,i_ben_profile)=beta*(1+r)*((recall_probs_pandemic(ib)+optimal_search_pandemic(:,ib,i_ben_profile)).*c_pol_e_guess(:).^(-mu)+(1-optimal_search_pandemic(:,ib,i_ben_profile)-recall_probs_pandemic(ib)).*c_pol_u_pandemic_guess(:,min(ib+1,n_b),i_ben_profile).^(-mu));
                c_tilde_u_pandemic(:,ib,i_ben_profile)=(rhs_u_pandemic(:,ib,i_ben_profile)).^(-1/mu); %unconstrained
                a_star_u_pandemic(:,ib,i_ben_profile)=(Aprime(:)+c_tilde_u_pandemic(:,ib,i_ben_profile)-benefit_profile_pandemic(ib,i_ben_profile))/(1+r); %a implied by a' and optimal c (so mapping from c_tilde to a_star gives us c_pol)
                a_star1_u_pandemic(ib,i_ben_profile)=(c_tilde_u_pandemic(1,ib,i_ben_profile)-benefit_profile_pandemic(ib,i_ben_profile))/(1+r);
            end
            %time3(t)=toc;
            %tic;
            for ib=1:n_b
                for ia=1:n_aprime %this is actually looping over a not over aprime, just using the same grid for both
                    if Aprime(ia)>a_star1_u_pandemic(ib,i_ben_profile)
                        c_pol_u_pandemic(ia:end,ib,i_ben_profile)=interp1(a_star_u_pandemic(:,ib,i_ben_profile),c_tilde_u_pandemic(:,ib,i_ben_profile),Aprime(ia:end),'linear','extrap');
                        break
                        %constrained_u(ia,ib,t)=0;
                    else
                        c_pol_u_pandemic(ia,ib,i_ben_profile)=(1+r)*Aprime(ia)+benefit_profile_pandemic(ib,i_ben_profile)+r*aprimemin;
                        %constrained_u(ia,ib,t)=1;
                    end
                end
            end
            for ib=1:n_b
                a_prime_holder_u_pandemic(:,ib,i_ben_profile)=(1+r)*Aprime+benefit_profile_pandemic(ib,i_ben_profile)-c_pol_u_pandemic(:,ib,i_ben_profile);
                v_u_pandemic(:,ib,i_ben_profile)=((c_pol_u_pandemic(:,ib,i_ben_profile)).^(1-mu)-1)/(1-mu)- (k * optimal_search_pandemic(:,ib,i_ben_profile).^ (1 + gamma)) / (1 + gamma) + c_param +beta*((recall_probs_pandemic(ib)+optimal_search_pandemic(:,ib,i_ben_profile)).*interp1(Aprime,v_e_guess(:),a_prime_holder_u_pandemic(:,ib,i_ben_profile),'linear','extrap')+(1-optimal_search_pandemic(:,ib,i_ben_profile)-recall_probs_pandemic(ib)).*interp1(Aprime,v_u_pandemic_guess(:,min(ib+1,n_b),i_ben_profile),a_prime_holder_u_pandemic(:,ib,i_ben_profile),'linear','extrap'));
            end

       end

       diffC = max([max(max(abs(c_pol_e(:) - c_pol_e_guess(:)))), max(max(max(abs(c_pol_u(:,:) - c_pol_u_guess(:,:))))), max(max(max(max(abs(c_pol_u_pandemic(:,:,:) - c_pol_u_pandemic_guess(:,:,:))))))]);

        diffC_percent = 100*max([max(max(abs((c_pol_e(:) - c_pol_e_guess(:)) ./ c_pol_e_guess(:)))), max(max(max(abs((c_pol_u(:,:) - c_pol_u_guess(:,:)) ./ c_pol_u_guess(:,:))))), max(max(max(max(abs((c_pol_u_pandemic(:,:,:) - c_pol_u_pandemic_guess(:,:,:)) ./ c_pol_u_pandemic_guess(:,:,:))))))]);

        % Absolute difference in value to measure convergence
        diffV = max([max(abs(v_e(:) - v_e_guess(:))), max(max(abs(v_u(:,:) - v_u_guess(:,:)))), max(max(max(abs(v_u_pandemic(:,:,:) - v_u_pandemic_guess(:,:,:)))))]);

        ave_change_in_C = mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:)))), mean(mean(mean(abs(c_pol_u(:,:) - c_pol_u_guess(:,:))))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:,:,:) - c_pol_u_pandemic_guess(:,:,:))))))]);

        ave_change_in_C_percent = 100*mean([mean(mean(abs(c_pol_e(:) - c_pol_e_guess(:))./c_pol_e_guess)), mean(mean(mean(abs(c_pol_u(:,:) - c_pol_u_guess(:,:))./c_pol_u_guess(:,:)))), mean(mean(mean(mean(abs(c_pol_u_pandemic(:,:,:) - c_pol_u_pandemic_guess(:,:,:))./c_pol_u_pandemic_guess))))]);

        ave_change_in_V = mean([mean(abs(v_e(:) - v_e_guess(:))), mean(mean(abs(v_u(:,:) - v_u_guess(:,:)))), mean(mean(mean(abs(v_u_pandemic(:,:,:) - v_u_pandemic_guess(:,:,:)))))]);

        ave_change_in_S = mean([mean(mean(mean(abs(optimal_search(:,:) - optimal_search_guess(:,:))))), mean(mean(mean(mean(abs(optimal_search_pandemic(:,:,:) - optimal_search_pandemic_guess(:,:,:))))))]);

         if mod(iter,20) == 0
             %[iter diffC ave_change_in_C ave_change_in_C_percent ave_change_in_S diffV ave_change_in_V]

             %[iter ave_change_in_C ave_change_in_C_percent ave_change_in_S]
        %     stop
         end


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

        if beta_loop==1
            c_pol_e_betanormal=c_pol_e;
            c_pol_u_betanormal=c_pol_u;
            c_pol_u_pandemic_betanormal=c_pol_u_pandemic;

            v_e_betanormal=v_e;
            v_u_betanormal=v_u;
            v_u_pandemic_betanormal=v_u_pandemic;
        elseif beta_loop==2
            c_pol_e_betahigh=c_pol_e;
            c_pol_u_betahigh=c_pol_u;
            c_pol_u_pandemic_betahigh=c_pol_u_pandemic;

            v_e_betahigh=v_e;
            v_u_betahigh=v_u;
            v_u_pandemic_betahigh=v_u_pandemic;
        end


    end







    

    A=Aprime;
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

    numt_sim=36;
    a_u_sim=zeros(numt_sim,1);
    c_u_sim=a_u_sim;
    c_u_pandemic_expect_sim=a_u_sim;
    c_u_pandemic_surprise_sim=a_u_sim;
    c_e_sim=c_u_sim;
    a_u_sim(1)=initial_a;
    a_u_pandemic_expect_sim=a_u_sim;
    a_u_pandemic_surprise_sim=a_u_sim;
    a_e_sim=a_u_sim;


    c_e_with500_sim=c_e_sim;
    a_e_with500_sim=a_e_sim;

    c_u_with500_sim1=c_u_sim;
    a_u_with500_sim1=a_u_sim;
    c_u_with500_sim2=c_u_sim;
    a_u_with500_sim2=a_u_sim;
    c_u_with500_sim3=c_u_sim;
    a_u_with500_sim3=a_u_sim;



    %Note that we don't necessarily need all parts of this simulation step to
    %be internal to the parameter search, keeping only the absolute necessary
    %parts internal to that loop should speed things up some

    %note also i might be able to speed up by feeding only the adjacent points
    %into the interp step

    numhh=1000;
    numsim=18;
    burnin=15;
    a_sim=zeros(numhh,burnin+1);
    c_sim=zeros(numhh,burnin+1);
    e_sim=zeros(numhh,burnin+1);
    u_dur_sim=zeros(numhh,burnin+1);
    a_sim(:,1)=initial_a;
    e_sim(:,1)=1;

    c_pol_e=c_pol_e_betanormal;
    c_pol_u=c_pol_u_betanormal;
    c_pol_u_pandemic=c_pol_u_pandemic_betanormal;

    v_e=v_e_betanormal;
    v_u=v_u_betanormal;
    v_u_pandemic=v_u_pandemic_betanormal;

    for t=1:burnin
        for i=1:numhh
            if e_sim(i,t)==1
                c_sim(i,t)=interp1(A,c_pol_e(:),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(y+(1+r)*a_sim(i,t)-c_sim(i,t),0);
            else
                c_sim(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim(i,t)-c_sim(i,t),0);
            end
            randy=rand(1,1);
            if e_sim(i,t)==1
                if randy<sep_rate
                    e_sim(i,t+1)=0;
                    u_dur_sim(i,t+1)=1;
                else
                    e_sim(i,t+1)=1;
                    u_dur_sim(i,t+1)=0;
                end
            else
                if randy<exog_find_rate
                    e_sim(i,t+1)=1;
                    u_dur_sim(i,t+1)=0;
                else
                    e_sim(i,t+1)=0;
                    u_dur_sim(i,t+1)=min(u_dur_sim(i,t)+1,n_b);
                end
            end
        end
    end


    tmp_a=a_sim(:,burnin+1);
    tmp_u=u_dur_sim(:,burnin+1);
    tmp_e=e_sim(:,burnin+1);

    a_sim=zeros(numhh,numsim);
    c_sim=zeros(numhh,numsim);
    e_sim=zeros(numhh,numsim);
    u_dur_sim=zeros(numhh,numsim);
    a_sim(:,1)=tmp_a;
    u_dur_sim(:,1)=tmp_u;
    e_sim(:,1)=tmp_e;

    c_sim_with_500=c_sim;
    a_sim_with_500=a_sim;
    a_sim_with_500(:,1)=a_sim_with_500(:,1)+shock_500;
    
    c_sim_with_2400=c_sim;
    a_sim_with_2400=a_sim;
    a_sim_with_2400(:,1)=a_sim_with_500(:,1)+FPUC_expiration;

    if use_initial_a==1
        a_sim_pandemic_surprise=tmp_a(tmp_u>0)+initial_a;
        a_sim_pandemic_surprise_extramonth=tmp_a(tmp_u>0)+initial_a;
        a_sim_pandemic_expect=tmp_a(tmp_u>0)+initial_a;
        a_sim_pandemic_surprise_wait=tmp_a(tmp_u>0)+initial_a;
        a_sim_pandemic_expect_wait=tmp_a(tmp_u>0)+initial_a;
        a_sim_regular=tmp_a(tmp_u>0)+initial_a;
        a_sim_pandemic_noFPUC=tmp_a(tmp_u>0)+initial_a;
        a_sim_e=tmp_a(tmp_u==0)+initial_a;
    else
        a_sim_pandemic_surprise=tmp_a(tmp_u>0);
        a_sim_pandemic_surprise_extramonth=tmp_a(tmp_u>0);
        a_sim_pandemic_expect=tmp_a(tmp_u>0);
        a_sim_pandemic_surprise_wait=tmp_a(tmp_u>0);
        a_sim_pandemic_expect_wait=tmp_a(tmp_u>0);
        a_sim_regular=tmp_a(tmp_u>0);
        a_sim_pandemic_noFPUC=tmp_a(tmp_u>0);
        a_sim_e=tmp_a(tmp_u==0);
    end

    num_unemployed_hh=length(a_sim_pandemic_surprise);
    num_employed_hh=length(a_sim_e);
    c_sim_pandemic_surprise=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_surprise_extramonth=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_expect=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_surprise_wait=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_expect_wait=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_regular=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_noFPUC=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_e=zeros(length(a_sim_e),30);


    search_sim_pandemic_surprise=zeros(length(a_sim_pandemic_surprise),30);
    search_sim_pandemic_expect=zeros(length(a_sim_pandemic_expect),30);
    search_sim_pandemic_surprise_wait=zeros(length(a_sim_pandemic_surprise_wait),30);
    search_sim_pandemic_expect_wait=zeros(length(a_sim_pandemic_expect_wait),30);
    search_sim_regular=zeros(length(a_sim_regular),30);
    search_sim_pandemic_noFPUC=zeros(length(a_sim_pandemic_surprise),30);


    %this is looping over all hh after the burnin period, to get the average
    %MPC
    for t=1:numsim
        for i=1:numhh
            if e_sim(i,t)==1
                c_sim(i,t)=interp1(A,c_pol_e(:),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(y+(1+r)*a_sim(i,t)-c_sim(i,t),0);

                c_sim_with_500(i,t)=interp1(A,c_pol_e(:),a_sim_with_500(i,t),'linear');
                a_sim_with_500(i,t+1)=max(y+(1+r)*a_sim_with_500(i,t)-c_sim_with_500(i,t),0);
                
                c_sim_with_2400(i,t)=interp1(A,c_pol_e(:),a_sim_with_2400(i,t),'linear');
                a_sim_with_2400(i,t+1)=max(y+(1+r)*a_sim_with_2400(i,t)-c_sim_with_2400(i,t),0);
            else
                c_sim(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim(i,t)-c_sim(i,t),0);

                c_sim_with_500(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim_with_500(i,t),'linear');
                a_sim_with_500(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim_with_500(i,t)-c_sim_with_500(i,t),0);
                
                c_sim_with_2400(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim_with_2400(i,t),'linear');
                a_sim_with_2400(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim_with_2400(i,t)-c_sim_with_2400(i,t),0);
            end
            randy=rand(1,1);
            if e_sim(i,t)==1
                if randy<sep_rate
                    e_sim(i,t+1)=0;
                    u_dur_sim(i,t+1)=1;
                else
                    e_sim(i,t+1)=1;
                    u_dur_sim(i,t+1)=0;
                end
            else
                if randy<exog_find_rate
                    e_sim(i,t+1)=1;
                    u_dur_sim(i,t+1)=0;
                else
                    e_sim(i,t+1)=0;
                    u_dur_sim(i,t+1)=min(u_dur_sim(i,t)+1,n_b);
                end
            end
        end
    end

    mean_a_sim=mean(a_sim,1);
    mean_c_sim=mean(c_sim,1);

    mean_c_sim_with_500=mean(c_sim_with_500);
    
    for tt=1:18
        mean_c_sim_e_with_500(tt)=mean(c_sim_with_500(e_sim(:,1)==1,tt));
        mean_c_sim_e_without_500(tt)=mean(c_sim(e_sim(:,1)==1,tt));
    end

    mpc_500_e_by_t=(mean_c_sim_e_with_500-mean_c_sim_e_without_500)/(shock_500);
    for t=1:numsim
        mpc_500_e_cum_dynamic(t)=sum(mpc_500_e_by_t(1:t));
    end
    if surprise==1
        mpc_surprise_500_e_bywage(iy,:)=mpc_500_e_cum_dynamic(:);
    else
        mpc_expect_500_e_bywage(iy,:)=mpc_500_e_cum_dynamic(:);
    end
    
    for tt=1:18
        mean_c_sim_u_with_500(tt)=mean(c_sim_with_500(e_sim(:,1)==0,tt));
        mean_c_sim_u_without_500(tt)=mean(c_sim(e_sim(:,1)==0,tt));
    end

    mpc_500_u_by_t=(mean_c_sim_u_with_500-mean_c_sim_u_without_500)/(shock_500);
    for t=1:numsim
        mpc_500_u_cum_dynamic(t)=sum(mpc_500_u_by_t(1:t));
    end
    if surprise==1
        mpc_surprise_500_u_bywage(iy,:)=mpc_500_u_cum_dynamic(:);
    else
        mpc_expect_500_u_bywage(iy,:)=mpc_500_u_cum_dynamic(:);
    end

    mpc_500_by_t=(mean_c_sim_with_500-mean_c_sim)/(shock_500);
    for t=1:numsim
        mpc_500_cum_dynamic(t)=sum(mpc_500_by_t(1:t));
    end
    %mpc_500_cum_dynamic(3)
    if surprise==1
        mpc_surprise_500_bywage(iy,:)=mpc_500_cum_dynamic(:);
    else
        mpc_expect_500_bywage(iy,:)=mpc_500_cum_dynamic(:);
    end
    
    
    
    mean_a_sim_with_2400=mean(a_sim_with_2400);
    mean_c_sim_with_2400=mean(c_sim_with_2400);
    for tt=1:18
        mean_c_sim_e_with_2400(tt)=mean(c_sim_with_2400(e_sim(:,1)==1,tt));
        mean_c_sim_e_without_2400(tt)=mean(c_sim(e_sim(:,1)==1,tt));
    end

    mpc_2400_e_by_t=(mean_c_sim_e_with_2400-mean_c_sim_e_without_2400)/(FPUC_expiration);
    for t=1:numsim
        mpc_2400_e_cum_dynamic(t)=sum(mpc_2400_e_by_t(1:t));
    end
    if surprise==1
        mpc_surprise_2400_e_bywage(iy,:)=mpc_2400_e_cum_dynamic(:);
    else
        mpc_expect_2400_e_bywage(iy,:)=mpc_2400_e_cum_dynamic(:);
    end
    
    for tt=1:18
        mean_c_sim_u_with_2400(tt)=mean(c_sim_with_2400(e_sim(:,1)==0,tt));
        mean_c_sim_u_without_2400(tt)=mean(c_sim(e_sim(:,1)==0,tt));
    end

    mpc_2400_u_by_t=(mean_c_sim_u_with_2400-mean_c_sim_u_without_2400)/(FPUC_expiration);
    for t=1:numsim
        mpc_2400_u_cum_dynamic(t)=sum(mpc_2400_u_by_t(1:t));
    end
    if surprise==1
        mpc_surprise_2400_u_bywage(iy,:)=mpc_2400_u_cum_dynamic(:);
    else
        mpc_expect_2400_u_bywage(iy,:)=mpc_2400_u_cum_dynamic(:);
    end
    
    
    mpc_2400_by_t=(mean_c_sim_with_2400-mean_c_sim)/(FPUC_expiration);
    for t=1:numsim
        mpc_2400_cum_dynamic(t)=sum(mpc_2400_by_t(1:t));
    end
    %mpc_2400_cum_dynamic(3)
    if surprise==1
        mpc_surprise_2400_bywage(iy,:)=mpc_2400_cum_dynamic(:);
    else
        mpc_expect_2400_bywage(iy,:)=mpc_2400_cum_dynamic(:);
    end
    

    %this is looping over just unemployed households (continuously unemployed)
    %to get u time-series patterns
    length_u=0;
    for t=1:numsim
        length_u=min(length_u+1,n_b);
        if t == 1
            c_pol_u = c_pol_u_betahigh;
            c_pol_u_pandemic = c_pol_u_pandemic_betahigh;
            v_e=v_e_betahigh;
            v_u=v_u_betahigh;
            v_u_pandemic=v_u_pandemic_betahigh;
            beta=beta_high;
        else
            c_pol_u = c_pol_u_betanormal;
            c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
            v_e=v_e_betanormal;
            v_u=v_u_betanormal;
            v_u_pandemic=v_u_pandemic_betanormal;
            beta=beta_normal;
        end
        
        for i=1:num_unemployed_hh

            %allow for initial assets, isomorphic to borrowing
            %if length_u==1
            %    a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t)+5*1320*FPUC_expiration/(4.5*600);
            %    a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t)+5*1320*FPUC_expiration/(4.5*600);
            %    a_sim_pandemic_expect_wait(i,t)=a_sim_pandemic_expect_wait(i,t)+5*1320*FPUC_expiration/(4.5*600);
            %    a_sim_pandemic_surprise_wait(i,t)=a_sim_pandemic_surprise_wait(i,t)+5*1320*FPUC_expiration/(4.5*600);
            %    a_sim_regular(i,t)=a_sim_regular(i,1)+5*1320*FPUC_expiration/(4.5*600);
            %end
            %LWA
            if length_u==6
                a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t)+1320*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t)+1320*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise_extramonth(i,t)=a_sim_pandemic_surprise_extramonth(i,t)+1320*FPUC_expiration/(4.5*600);
                a_sim_pandemic_expect_wait(i,t)=a_sim_pandemic_expect_wait(i,t)+1320*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise_wait(i,t)=a_sim_pandemic_surprise_wait(i,t)+1320*FPUC_expiration/(4.5*600);
            end
            %Jan EIP
            if length_u==10
                a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t)+1500*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t)+1500*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise_extramonth(i,t)=a_sim_pandemic_surprise_extramonth(i,t)+1500*FPUC_expiration/(4.5*600);
                a_sim_pandemic_expect_wait(i,t)=a_sim_pandemic_expect_wait(i,t)+1500*FPUC_expiration/(4.5*600);
                a_sim_pandemic_surprise_wait(i,t)=a_sim_pandemic_surprise_wait(i,t)+1500*FPUC_expiration/(4.5*600);
            end

            


            c_sim_regular(i,t)=interp1(A,c_pol_u(:,length_u),a_sim_regular(i,t),'linear');
            a_sim_regular(i,t+1)=max(benefit_profile(length_u)+(1+r)*a_sim_regular(i,t)-c_sim_regular(i,t),0);

            c_sim_pandemic_expect(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_expect(i,t),'linear');
            a_sim_pandemic_expect(i,t+1)=max(benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_expect(i,t)-c_sim_pandemic_expect(i,t),0);

            c_sim_pandemic_expect_wait(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,3),a_sim_pandemic_expect_wait(i,t),'linear');
            a_sim_pandemic_expect_wait(i,t+1)=max(benefit_profile_pandemic(length_u,3)+(1+r)*a_sim_pandemic_expect_wait(i,t)-c_sim_pandemic_expect_wait(i,t),0);

            %Note this will vary with params, but can just save it accordingly
            %when taking means later
            c_sim_pandemic_noFPUC(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,5),a_sim_pandemic_noFPUC(i,t),'linear');
            a_sim_pandemic_noFPUC(i,t+1)=max(benefit_profile_pandemic(length_u,5)+(1+r)*a_sim_pandemic_noFPUC(i,t)-c_sim_pandemic_noFPUC(i,t),0);

            if length_u<=4
                c_sim_pandemic_surprise(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,2),a_sim_pandemic_surprise(i,t),'linear');
                a_sim_pandemic_surprise(i,t+1)=max(benefit_profile_pandemic(length_u,2)+(1+r)*a_sim_pandemic_surprise(i,t)-c_sim_pandemic_surprise(i,t),0);

                c_sim_pandemic_surprise_wait(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,4),a_sim_pandemic_surprise_wait(i,t),'linear');
                a_sim_pandemic_surprise_wait(i,t+1)=max(benefit_profile_pandemic(length_u,4)+(1+r)*a_sim_pandemic_surprise_wait(i,t)-c_sim_pandemic_surprise_wait(i,t),0);
            else
                c_sim_pandemic_surprise(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_surprise(i,t),'linear');
                a_sim_pandemic_surprise(i,t+1)=max(benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_surprise(i,t)-c_sim_pandemic_surprise(i,t),0);

                c_sim_pandemic_surprise_wait(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,3),a_sim_pandemic_surprise_wait(i,t),'linear');
                a_sim_pandemic_surprise_wait(i,t+1)=max(benefit_profile_pandemic(length_u,3)+(1+r)*a_sim_pandemic_surprise_wait(i,t)-c_sim_pandemic_surprise_wait(i,t),0);
            end
            
            if length_u<=5
                c_sim_pandemic_surprise_extramonth(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,2),a_sim_pandemic_surprise_extramonth(i,t),'linear');
                a_sim_pandemic_surprise_extramonth(i,t+1)=max(benefit_profile_pandemic(length_u,2)+(1+r)*a_sim_pandemic_surprise_extramonth(i,t)-c_sim_pandemic_surprise_extramonth(i,t),0);
            else
                c_sim_pandemic_surprise_extramonth(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_surprise_extramonth(i,t),'linear');
                a_sim_pandemic_surprise_extramonth(i,t+1)=max(benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_surprise_extramonth(i,t)-c_sim_pandemic_surprise_extramonth(i,t),0);
            end

            diff_v=interp1(A,v_e(:),a_sim_regular(i,t+1),'linear')-interp1(A,v_u(:,min(length_u+1,n_b)),a_sim_regular(i,t+1),'linear');
            search_sim_regular(i,t)=min(1-recall_probs_regular(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));

            diff_v=interp1(A,v_e(:),a_sim_pandemic_noFPUC(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),5),a_sim_pandemic_noFPUC(i,t+1),'linear');
            search_sim_pandemic_noFPUC(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_noFPUC(i,t))~=0 
                search_sim_pandemic_noFPUC(i,t)=0;
            end

            diff_v=interp1(A,v_e(:),a_sim_pandemic_expect(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),1),a_sim_pandemic_expect(i,t+1),'linear');
            search_sim_pandemic_expect(i,t)=min(1-recall_probs_pandemic(ib), max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_expect(i,t))~=0 
                search_sim_pandemic_expect(i,t)=0;
            end

            diff_v=interp1(A,v_e(:),a_sim_pandemic_expect_wait(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),3),a_sim_pandemic_expect_wait(i,t+1),'linear');
            search_sim_pandemic_expect_wait(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_expect_wait(i,t))~=0 
                search_sim_pandemic_expect_wait(i,t)=0;
            end

            if length_u<=4
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),2),a_sim_pandemic_surprise(i,t+1),'linear');
            else
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),1),a_sim_pandemic_surprise(i,t+1),'linear');
            end
            search_sim_pandemic_surprise(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_surprise(i,t))~=0 
                search_sim_pandemic_surprise(i,t)=0;
            end    

            if length_u<=4
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise_wait(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),4),a_sim_pandemic_surprise_wait(i,t+1),'linear');
            else
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise_wait(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),3),a_sim_pandemic_surprise_wait(i,t+1),'linear');
            end
            search_sim_pandemic_surprise_wait(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_surprise_wait(i,t))~=0 
                search_sim_pandemic_surprise_wait(i,t)=0;
            end

            %note for surprise case won't want to use i_b+1 in actuality will
            %want to use the expected one in the last period before surprise

        end
    end

    %this is looping over just employed households (continuously employed)
    %to get e time-series patterns
    for t=1:numsim
        for i=1:num_employed_hh
             if t==4
                 c_pol_e=c_pol_e_betahigh;
             else
                 c_pol_e=c_pol_e_betanormal;
             end

             if t==12
                a_sim_e(i,t)=a_sim_e(i,t)+1500*FPUC_expiration/(4.5*600);
             end

             %adjust initial assets isomorphic to allowing for borrowing
             %if t==1
             %    a_sim_e(i,t)=a_sim_e(i,t)+5*1320*FPUC_expiration/(4.5*600);
             %end

             c_sim_e(i,t)=interp1(A,c_pol_e(:),a_sim_e(i,t),'linear');
             a_sim_e(i,t+1)=y+(1+r)*a_sim_e(i,t)-c_sim_e(i,t);   
        end
    end


    if surprise==1
        mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
        mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);
        
        mean_a_sim_pandemic_surprise_extramonth=mean(a_sim_pandemic_surprise_extramonth,1);
        mean_c_sim_pandemic_surprise_extramonth=mean(c_sim_pandemic_surprise_extramonth,1);

        mean_a_sim_pandemic_surprise_noFPUC=mean(a_sim_pandemic_noFPUC,1);
        mean_c_sim_pandemic_surprise_noFPUC=mean(c_sim_pandemic_noFPUC,1);

        mean_a_sim_pandemic_surprise_wait=mean(a_sim_pandemic_surprise_wait,1);
        mean_c_sim_pandemic_surprise_wait=mean(c_sim_pandemic_surprise_wait,1);


        mean_a_sim_regular=mean(a_sim_regular,1);
        mean_c_sim_regular=mean(c_sim_regular,1);

        mean_a_sim_e=mean(a_sim_e,1);
        mean_c_sim_e=mean(c_sim_e,1);

        mean_search_sim_regular=mean(search_sim_regular,1);
        mean_search_sim_pandemic_surprise_noFPUC=mean(search_sim_pandemic_noFPUC,1);
        mean_search_sim_pandemic_surprise=mean(search_sim_pandemic_surprise,1);
        mean_search_sim_pandemic_surprise_wait=mean(search_sim_pandemic_surprise_wait,1);
    else
        mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
        mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);

        mean_a_sim_pandemic_expect_noFPUC=mean(a_sim_pandemic_noFPUC,1);
        mean_c_sim_pandemic_expect_noFPUC=mean(c_sim_pandemic_noFPUC,1);

        mean_a_sim_pandemic_expect_wait=mean(a_sim_pandemic_expect_wait,1);
        mean_c_sim_pandemic_expect_wait=mean(c_sim_pandemic_expect_wait,1);
        mean_search_sim_pandemic_expect=mean(search_sim_pandemic_expect,1);
        mean_search_sim_pandemic_expect_noFPUC=mean(search_sim_pandemic_noFPUC,1);
        mean_search_sim_pandemic_expect_wait=mean(search_sim_pandemic_expect_wait,1);
    end


    end
    
    mean_c_sim_e_bywage(iy,:)=mean_c_sim_e;
    mean_a_sim_e_bywage(iy,:)=mean_a_sim_e;

    %paste on initial Jan-March 3 months of employment 
    mean_a_sim_pandemic_surprise_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise(1:numsim-3)];
    mean_c_sim_pandemic_surprise_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise(1:numsim-3)];
    mean_a_sim_pandemic_surprise_extramonth_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_extramonth(1:numsim-3)];
    mean_c_sim_pandemic_surprise_extramonth_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_extramonth(1:numsim-3)];
    mean_a_sim_pandemic_surprise_wait_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_wait(1:numsim-3)];
    mean_c_sim_pandemic_surprise_wait_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_wait(1:numsim-3)];
    mean_a_sim_pandemic_surprise_noFPUC_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise_noFPUC(1:numsim-3)];
    mean_c_sim_pandemic_surprise_noFPUC_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise_noFPUC(1:numsim-3)];
    mean_a_sim_pandemic_expect_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_expect(1:numsim-3)];
    mean_c_sim_pandemic_expect_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_expect(1:numsim-3)];
    mean_a_sim_pandemic_expect_wait_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_wait(1:numsim-3)];
    mean_c_sim_pandemic_expect_wait_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_wait(1:numsim-3)];
    mean_a_sim_pandemic_expect_noFPUC_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_noFPUC(1:numsim-3)];
    mean_c_sim_pandemic_expect_noFPUC_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_noFPUC(1:numsim-3)];
    mean_a_sim_regular_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_regular(1:numsim-3)];
    mean_c_sim_regular_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_regular(1:numsim-3)];


    mean_y_sim_pandemic_u_bywage(iy,:)=[y y y benefit_profile_pandemic(:,1)'];
    mean_y_sim_pandemic_u_bywage(iy,4)=mean_y_sim_pandemic_u_bywage(iy,4);
    mean_y_sim_pandemic_u_bywage(iy,9)=mean_y_sim_pandemic_u_bywage(iy,9)+1320*FPUC_expiration/(4.5*600);
    mean_y_sim_pandemic_u_bywage(iy,13)=mean_y_sim_pandemic_u_bywage(iy,13)+1500*FPUC_expiration/(4.5*600);
    mean_y_sim_pandemic_wait_bywage(iy,:)=[y y y benefit_profile_pandemic(:,3)'];
    mean_y_sim_pandemic_wait_bywage(iy,4)=mean_y_sim_pandemic_wait_bywage(iy,4);
    mean_y_sim_pandemic_wait_bywage(iy,9)=mean_y_sim_pandemic_wait_bywage(iy,9)+1320*FPUC_expiration/(4.5*600);
    mean_y_sim_pandemic_wait_bywage(iy,13)=mean_y_sim_pandemic_wait_bywage(iy,13)+1500*FPUC_expiration/(4.5*600);
    mean_y_sim_pandemic_noFPUC_bywage(iy,:)=[y y y benefit_profile_pandemic(:,5)'];
    mean_y_sim_pandemic_noFPUC_bywage(iy,4)=mean_y_sim_pandemic_noFPUC_bywage(iy,4);
    mean_y_sim_pandemic_noFPUC_bywage(iy,13)=mean_y_sim_pandemic_noFPUC_bywage(iy,13)+1500*FPUC_expiration/(4.5*600);

    mean_y_sim_e_bywage(iy,:)=y*ones(16,1);
    mean_y_sim_e_bywage(iy,4)=mean_y_sim_e_bywage(iy,4);



    mean_y_sim_regular_bywage(iy,:)=[y y y benefit_profile(:,1)'];

    mean_search_sim_regular_bywage(iy,:)=[NaN NaN NaN mean_search_sim_regular(1:numsim-3)];
    mean_search_sim_pandemic_expect_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_expect(1:numsim-3)];
    mean_search_sim_pandemic_expect_wait_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_expect_wait(1:numsim-3)];
    mean_search_sim_pandemic_expect_noFPUC_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_expect_noFPUC(1:numsim-3)];
    mean_search_sim_pandemic_surprise_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_surprise(1:numsim-3)];
    mean_search_sim_pandemic_surprise_wait_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_surprise_wait(1:numsim-3)];
    mean_search_sim_pandemic_surprise_noFPUC_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_surprise_noFPUC(1:numsim-3)];

    mean_c_sim_pandemic_surprise_vs_e_bywage(iy,:)=mean_c_sim_pandemic_surprise./mean_c_sim_e;
    mean_c_sim_pandemic_surprise_extramonth_vs_e_bywage(iy,:)=mean_c_sim_pandemic_surprise_extramonth./mean_c_sim_e;
    mean_c_sim_pandemic_surprise_wait_vs_e_bywage(iy,:)=mean_c_sim_pandemic_surprise_wait./mean_c_sim_e;
    mean_c_sim_pandemic_expect_vs_e_bywage(iy,:)=mean_c_sim_pandemic_expect./mean_c_sim_e;
    mean_c_sim_pandemic_expect_wait_vs_e_bywage(iy,:)=mean_c_sim_pandemic_expect_wait./mean_c_sim_e;

    
    
    
    
end

mpc_surprise_500_monthly=mean(mpc_surprise_500_bywage(:,1));
mpc_surprise_500_quarterly=mean(mpc_surprise_500_bywage(:,3));
mpc_expect_500_quarterly=mean(mpc_expect_500_bywage(:,3));
mpc_expect_500_monthly=mean(mpc_expect_500_bywage(:,1));

mpc_surprise_500_e_quarterly=mean(mpc_surprise_500_e_bywage(:,3));
mpc_surprise_500_e_monthly=mean(mpc_surprise_500_e_bywage(:,1));
mpc_expect_500_e_quarterly=mean(mpc_expect_500_e_bywage(:,3));
mpc_expect_500_e_monthly=mean(mpc_expect_500_e_bywage(:,1));

mpc_surprise_500_u_quarterly=mean(mpc_surprise_500_u_bywage(:,3));
mpc_surprise_500_u_monthly=mean(mpc_surprise_500_u_bywage(:,1));
mpc_expect_500_u_quarterly=mean(mpc_expect_500_u_bywage(:,3));
mpc_expect_500_u_monthly=mean(mpc_expect_500_u_bywage(:,1));

mpc_surprise_2400_quarterly=mean(mpc_surprise_2400_bywage(:,3));
mpc_surprise_2400_monthly=mean(mpc_surprise_2400_bywage(:,1));
mpc_expect_2400_quarterly=mean(mpc_expect_2400_bywage(:,3));
mpc_expect_2400_monthly=mean(mpc_expect_2400_bywage(:,1));

mpc_surprise_2400_e_quarterly=mean(mpc_surprise_2400_e_bywage(:,3));
mpc_surprise_2400_e_monthly=mean(mpc_surprise_2400_e_bywage(:,1));
mpc_expect_2400_e_quarterly=mean(mpc_expect_2400_e_bywage(:,3));
mpc_expect_2400_e_monthly=mean(mpc_expect_2400_e_bywage(:,1));

mpc_surprise_2400_u_quarterly=mean(mpc_surprise_2400_u_bywage(:,3));
mpc_surprise_2400_u_monthly=mean(mpc_surprise_2400_u_bywage(:,1));
mpc_expect_2400_u_quarterly=mean(mpc_expect_2400_u_bywage(:,3));
mpc_expect_2400_u_monthly=mean(mpc_expect_2400_u_bywage(:,1));

mean_a_sim_e=mean(mean_a_sim_e_bywage,1);
mean_a_sim_pandemic_surprise=mean(mean_a_sim_pandemic_surprise_bywage,1);
mean_a_sim_pandemic_surprise_extramonth=mean(mean_a_sim_pandemic_surprise_extramonth_bywage,1);
mean_a_sim_pandemic_surprise_wait=mean(mean_a_sim_pandemic_surprise_wait_bywage,1);
mean_a_sim_pandemic_surprise_noFPUC=mean(mean_a_sim_pandemic_surprise_noFPUC_bywage,1);
mean_a_sim_pandemic_expect=mean(mean_a_sim_pandemic_expect_bywage,1);
mean_a_sim_pandemic_expect_wait=mean(mean_a_sim_pandemic_expect_wait_bywage,1);
mean_a_sim_pandemic_expect_noFPUC=mean(mean_a_sim_pandemic_expect_noFPUC_bywage,1);
mean_a_sim_pandemic_regular=mean(mean_a_sim_regular_bywage,1);

mean_y_sim_pandemic_u=mean(mean_y_sim_pandemic_u_bywage,1);
mean_y_sim_pandemic_wait=mean(mean_y_sim_pandemic_wait_bywage,1);
mean_y_sim_pandemic_noFPUC=mean(mean_y_sim_pandemic_noFPUC_bywage,1);
mean_y_sim_e = mean(mean_y_sim_e_bywage, 1);

mean_c_sim_e=mean(mean_c_sim_e_bywage,1);
mean_c_sim_pandemic_surprise=mean(mean_c_sim_pandemic_surprise_bywage,1);
mean_c_sim_pandemic_surprise_extramonth=mean(mean_c_sim_pandemic_surprise_extramonth_bywage,1);
mean_c_sim_pandemic_surprise_wait=mean(mean_c_sim_pandemic_surprise_wait_bywage,1);
mean_c_sim_pandemic_surprise_noFPUC=mean(mean_c_sim_pandemic_surprise_noFPUC_bywage,1);
mean_c_sim_pandemic_expect=mean(mean_c_sim_pandemic_expect_bywage,1);
mean_c_sim_pandemic_expect_wait=mean(mean_c_sim_pandemic_expect_wait_bywage,1);
mean_c_sim_pandemic_expect_noFPUC=mean(mean_c_sim_pandemic_expect_noFPUC_bywage,1);
mean_c_sim_regular=mean(mean_c_sim_regular_bywage,1);

mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020=[mean_c_sim_pandemic_expect' mean_c_sim_pandemic_expect_noFPUC'];
save('mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020.mat','mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2020')


mean_search_sim_pandemic_surprise=mean(mean_search_sim_pandemic_surprise_bywage,1);
mean_search_sim_pandemic_surprise_wait=mean(mean_search_sim_pandemic_surprise_wait_bywage,1);
mean_search_sim_pandemic_surprise_noFPUC=mean(mean_search_sim_pandemic_surprise_noFPUC_bywage,1);
mean_search_sim_pandemic_expect=mean(mean_search_sim_pandemic_expect_bywage,1);
mean_search_sim_pandemic_expect_wait=mean(mean_search_sim_pandemic_expect_wait_bywage,1);
mean_search_sim_pandemic_expect_noFPUC=mean(mean_search_sim_pandemic_expect_noFPUC_bywage,1);
mean_search_sim_pandemic_regular=mean(mean_search_sim_regular_bywage,1);

%Convert model simulations to dollar deviations in U vs. E space
mean_c_sim_pandemic_surprise_dollars=mean_c_sim_pandemic_surprise./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_surprise_extramonth_dollars=mean_c_sim_pandemic_surprise_extramonth./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_surprise_wait_dollars=mean_c_sim_pandemic_surprise_wait./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_surprise_noFPUC_dollars=mean_c_sim_pandemic_surprise_noFPUC./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_regular_dollars=mean_c_sim_regular/mean_c_sim_e(1)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_expect_dollars=mean_c_sim_pandemic_expect./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_expect_wait_dollars=mean_c_sim_pandemic_expect_wait./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_pandemic_expect_noFPUC_dollars=mean_c_sim_pandemic_expect_noFPUC./mean_c_sim_e(1:18)*total_spend_u(1)-total_spend_u(1);
mean_c_sim_e_dollars=mean_c_sim_e(1:18)./mean_c_sim_e(1:18)*total_spend_e(1)-total_spend_e(1);

mean_y_sim_pandemic_u_dollars = mean_y_sim_pandemic_u./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_pandemic_u_extramonth_dollars = mean_y_sim_pandemic_u_dollars;
mean_y_sim_pandemic_u_extramonth_dollars(8) = mean_y_sim_pandemic_u_extramonth_dollars(8) + FPUC_expiration * income_u(1);
mean_y_sim_pandemic_wait_dollars = mean_y_sim_pandemic_wait./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_pandemic_noFPUC_dollars = mean_y_sim_pandemic_noFPUC./mean_y_sim_e * income_u(1) - income_u(1);
mean_y_sim_e_dollars = 0;

scale_factor=(total_spend_e(1)/income_e(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));



mpc=table();
mpc.surprise('waiting')=((mean_c_sim_pandemic_surprise_dollars(5)-mean_c_sim_pandemic_surprise_dollars(3))-(mean_c_sim_pandemic_surprise_wait_dollars(5)-mean_c_sim_pandemic_surprise_wait_dollars(3)))/((mean_y_sim_pandemic_u_dollars(5)-mean_y_sim_pandemic_u_dollars(3))-(mean_y_sim_pandemic_wait_dollars(5)-mean_y_sim_pandemic_wait_dollars(3)));
mpc.expected('waiting')=((mean_c_sim_pandemic_expect_dollars(5)-mean_c_sim_pandemic_expect_dollars(3))-(mean_c_sim_pandemic_expect_wait_dollars(5)-mean_c_sim_pandemic_expect_wait_dollars(3)))/((mean_y_sim_pandemic_u_dollars(5)-mean_y_sim_pandemic_u_dollars(3))-(mean_y_sim_pandemic_wait_dollars(5)-mean_y_sim_pandemic_wait_dollars(3)));
mpc.surprise('600 expiration')=((mean_c_sim_pandemic_surprise_dollars(8)-mean_c_sim_pandemic_surprise_dollars(7))-(mean_c_sim_e(8)-mean_c_sim_e(7)))/(mean_y_sim_pandemic_u_dollars(8)-mean_y_sim_pandemic_u_dollars(7));
mpc.expected('600 expiration')=((mean_c_sim_pandemic_expect_dollars(8)-mean_c_sim_pandemic_expect_dollars(7))-(mean_c_sim_e(8)-mean_c_sim_e(7)))/(mean_y_sim_pandemic_u_dollars(8)-mean_y_sim_pandemic_u_dollars(7));
mpc.expected('500 quarterly')=mpc_expect_500_quarterly;
mpc.surprise('500 quarterly')=mpc_surprise_500_quarterly;
mpc.expected('500 monthly')=mpc_expect_500_monthly;
mpc.surprise('500 monthly')=mpc_surprise_500_monthly;
mpc.expected('500 quarterly-employed')=mpc_expect_500_e_quarterly;
mpc.surprise('500 quarterly-employed')=mpc_surprise_500_e_quarterly;
mpc.expected('500 monthly-employed')=mpc_expect_500_e_monthly;
mpc.surprise('500 monthly-employed')=mpc_surprise_500_e_monthly;
mpc.expected('500 quarterly-unemployed')=mpc_expect_500_u_quarterly;
mpc.surprise('500 quarterly-unemployed')=mpc_surprise_500_u_quarterly;
mpc.expected('500 monthly-unemployed')=mpc_expect_500_u_monthly;
mpc.surprise('500 monthly-unemployed')=mpc_surprise_500_u_monthly
mpc.expected('2400 quarterly')=mpc_expect_2400_quarterly;
mpc.surprise('2400 quarterly')=mpc_surprise_2400_quarterly;
mpc.expected('2400 monthly')=mpc_expect_2400_monthly;
mpc.surprise('2400 monthly')=mpc_surprise_2400_monthly;
mpc.expected('2400 quarterly-employed')=mpc_expect_2400_e_quarterly;
mpc.surprise('2400 quarterly-employed')=mpc_surprise_2400_e_quarterly;
mpc.expected('2400 monthly-employed')=mpc_expect_2400_e_monthly;
mpc.surprise('2400 monthly-employed')=mpc_surprise_2400_e_monthly;
mpc.expected('2400 quarterly-unemployed')=mpc_expect_2400_u_quarterly;
mpc.surprise('2400 quarterly-unemployed')=mpc_surprise_2400_u_quarterly;
mpc.expected('2400 monthly-unemployed')=mpc_expect_2400_u_monthly;
mpc.surprise('2400 monthly-unemployed')=mpc_surprise_2400_u_monthly;
mpc.Variables=scale_factor*mpc.Variables

mpc_supplements_prepandemic=table();
mpc_supplements_prepandemic.surprise('one_month')=(mean_c_sim_pandemic_surprise_dollars(4)-mean_c_sim_pandemic_surprise_noFPUC_dollars(4))/(mean_y_sim_pandemic_u_dollars(4)-mean_y_sim_pandemic_noFPUC_dollars(4));
mpc_supplements_prepandemic.surprise('3_month')=sum(mean_c_sim_pandemic_surprise_dollars(4:6)-mean_c_sim_pandemic_surprise_noFPUC_dollars(4:6))/sum(mean_y_sim_pandemic_u_dollars(4:6)-mean_y_sim_pandemic_noFPUC_dollars(4:6));
mpc_supplements_prepandemic.surprise('6_month')=sum(mean_c_sim_pandemic_surprise_dollars(4:9)-mean_c_sim_pandemic_surprise_noFPUC_dollars(4:9))/sum(mean_y_sim_pandemic_u_dollars(4:9)-mean_y_sim_pandemic_noFPUC_dollars(4:9));
mpc_supplements_prepandemic.surprise('full') = sum(mean_c_sim_pandemic_surprise_dollars(4:7) - mean_c_sim_pandemic_surprise_noFPUC_dollars(4:7)) / sum(mean_y_sim_pandemic_u_dollars(4:7) - mean_y_sim_pandemic_noFPUC_dollars(4:7));
mpc_supplements_prepandemic.surprise('full+3') = sum(mean_c_sim_pandemic_surprise_dollars(4:10) - mean_c_sim_pandemic_surprise_noFPUC_dollars(4:10)) / sum(mean_y_sim_pandemic_u_dollars(4:10) - mean_y_sim_pandemic_noFPUC_dollars(4:10));


%mpc_supplements.surprise('expire')=(mean_c_sim_pandemic_surprise_extramonth_dollars(8)-mean_c_sim_pandemic_surprise_dollars(8))/(mean_y_sim_pandemic_u_extramonth_dollars(8)-mean_y_sim_pandemic_u_dollars(8));
mpc_supplements_prepandemic.expect('one_month')=(mean_c_sim_pandemic_expect_dollars(4)-mean_c_sim_pandemic_expect_noFPUC_dollars(4))/(mean_y_sim_pandemic_u_dollars(4)-mean_y_sim_pandemic_noFPUC_dollars(4));
mpc_supplements_prepandemic.expect('3_month')=sum(mean_c_sim_pandemic_expect_dollars(4:6)-mean_c_sim_pandemic_expect_noFPUC_dollars(4:6))/sum(mean_y_sim_pandemic_u_dollars(4:6)-mean_y_sim_pandemic_noFPUC_dollars(4:6));
mpc_supplements_prepandemic.expect('6_month')=sum(mean_c_sim_pandemic_expect_dollars(4:9)-mean_c_sim_pandemic_expect_noFPUC_dollars(4:9))/sum(mean_y_sim_pandemic_u_dollars(4:9)-mean_y_sim_pandemic_noFPUC_dollars(4:9));
mpc_supplements_prepandemic.expect('full') = sum(mean_c_sim_pandemic_expect_dollars(4:7) - mean_c_sim_pandemic_expect_noFPUC_dollars(4:7)) / sum(mean_y_sim_pandemic_u_dollars(4:7) - mean_y_sim_pandemic_noFPUC_dollars(4:7));
mpc_supplements_prepandemic.expect('full+3') = sum(mean_c_sim_pandemic_expect_dollars(4:10) - mean_c_sim_pandemic_expect_noFPUC_dollars(4:10)) / sum(mean_y_sim_pandemic_u_dollars(4:10) - mean_y_sim_pandemic_noFPUC_dollars(4:10))
mpc_supplements_prepandemic.Variables=scale_factor*mpc_supplements_prepandemic.Variables;



exit_rates_data=readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date=datetime(exit_rates_data.week_start_date);
idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-01-01') & datenum(exit_rates_data.week_start_date) < datenum('2020-11-20');
exit_rates_data = exit_rates_data(idx, :);

exit_rates_data.month = dateshift(exit_rates_data.week_start_date, 'start', 'month');
% For the exit variables we want the average exit probability at a monthly level
exit_rates_data_week_to_month = varfun(@week_to_month_exit, exit_rates_data, 'InputVariables', {'ExitRateToRecall', 'ExitRateNotToRecall'}, 'GroupingVariables', {'month'});
exit_rates_data_week_to_month = renamevars(exit_rates_data_week_to_month, ["week_to_month_exit_ExitRateToRecall", "week_to_month_exit_ExitRateNotToRecall"], ["exit_to_recall", "exit_not_to_recall"]);
monthly_search_data=exit_rates_data_week_to_month.exit_not_to_recall';
monthly_search_data=monthly_search_data(4:end);


newjob_exit_rate_FPUC=mean_search_sim_pandemic_expect(4:18)';
newjob_exit_rate_no_FPUC=mean_search_sim_pandemic_expect_noFPUC(4:18)';
newjob_exit_rate_FPUC(end:1000)=newjob_exit_rate_FPUC(end);
newjob_exit_rate_no_FPUC(end:1000)=newjob_exit_rate_no_FPUC(end);
recall_probs=recall_probs_pandemic_actual(1:12)';
recall_probs(end:1000)=recall_probs(end);
mean_c_sim_pandemic_surprise_overall_FPUC = NaN;
mean_c_sim_pandemic_surprise_overall_noFPUC = NaN;
mean_c_sim_e_overall = NaN;
benefit_change_data = readtable(jobfind_input_directory, 'Sheet', per_change_overall);
perc_change_benefits_data = benefit_change_data.non_sym_per_change(1);
date_sim_start = datetime(2020,4,1);
t_start = 2;
% Surprise period minus one should be period 4 (April is 1, May is 2, June is 3, July is 4, August is 5)
t_end = 5;
include_self_employed=0;
[elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

% Wrapper for elasticity and distortion results
elasticity_and_distortions_values_prepandemic = [elasticity, employment_distortion, total_diff_employment, share_unemployment_reduced];

mean_c_sim_prepandemic_expect_dollars=mean_c_sim_pandemic_expect_dollars;
mean_c_sim_prepandemic_expect_noFPUC_dollars=mean_c_sim_pandemic_expect_noFPUC_dollars;
mean_c_sim_prepandemic_surprise_dollars=mean_c_sim_pandemic_surprise_dollars;
mean_c_sim_prepandemic_surprise_noFPUC_dollars=mean_c_sim_pandemic_surprise_noFPUC_dollars;
mpc_prepandemic=mpc;
mean_search_sim_prepandemic_expect=mean_search_sim_pandemic_expect;
mean_search_sim_prepandemic_surprise=mean_search_sim_pandemic_surprise;
mean_search_sim_prepandemic_expect_noFPUC=mean_search_sim_pandemic_expect_noFPUC;


newjob_exit_rate_prepandemic_FPUC_2020 = newjob_exit_rate_FPUC;
newjob_exit_rate_prepandemic_no_FPUC_2020 = newjob_exit_rate_no_FPUC;
save('prepandemic_newjob_2020','newjob_exit_rate_prepandemic_FPUC_2020','newjob_exit_rate_prepandemic_no_FPUC_2020')

save('prepandemic_results_target500MPC','mean_c_sim_prepandemic_expect_dollars','mean_c_sim_prepandemic_expect_noFPUC_dollars','mpc_prepandemic','mpc_supplements_prepandemic','mean_search_sim_prepandemic_expect','mean_search_sim_prepandemic_expect_noFPUC','mean_c_sim_prepandemic_surprise_dollars','mean_c_sim_prepandemic_surprise_noFPUC_dollars','mean_search_sim_prepandemic_surprise','elasticity_and_distortions_values_prepandemic')

a_init_prepandemic_target500MPC=mean_a_sim_pandemic_expect_bywage(:,10);
save('a_init_prepandemic_target500MPC','a_init_prepandemic_target500MPC');

toc
