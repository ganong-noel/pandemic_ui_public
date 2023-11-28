display('Simulating Prepandemic Model Effects of $300')
clearvars -except -regexp fig_paper_*

load jobfind_input_directory.mat
load jobfind_input_sheets.mat
load spending_input_directory.mat
load spending_input_sheets.mat
load hh_wage_groups.mat
load release_paths.mat

load bestfit_prepandemic.mat

load graph_axis_labels_timeseries.mat

sse_expect_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_expect_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_expect_fit_het_match500MPC(3)=0;

sse_surprise_fit_het_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_surprise_fit_het_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_surprise_fit_het_match500MPC(3)=0;


EIP2 = 1200;
EIP2_e=600;
EIP3 = 4000;

sse_expect_extension_fit_het_onset_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_expect_extension_fit_het_onset_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_expect_extension_fit_het_onset_match500MPC(3)=0;

sse_surprise_extension_fit_het_onset_match500MPC(1)=pre_pandemic_fit_match500MPC(1);
sse_surprise_extension_fit_het_onset_match500MPC(2)=pre_pandemic_fit_match500MPC(2);
sse_surprise_extension_fit_het_onset_match500MPC(3)=0;

data_update = readtable(spending_input_directory, 'Sheet', model_data);
idx_emp = (string(data_update.category) == 'Spending (total)') & (string(data_update.group) == 'Employed')& (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Spending (total)') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
total_spend_e=data_update_e.value;
total_spend_u=data_update_u.value;
total_spend_e_jan20=total_spend_e(13);
total_spend_u_jan20=total_spend_u(13);
total_spend_e_yoy=total_spend_e(13:end)./total_spend_e(1:end-12)*total_spend_e(13);
total_spend_u_yoy=total_spend_u(13:end)./total_spend_u(1:end-12)*total_spend_u(13);
total_spend_e=total_spend_e(13+10:end);
total_spend_u=total_spend_u(13+10:end);
perc_spend_e=data_update_e.percent_change;
perc_spend_u=data_update_u.percent_change;
perc_spend_u_vs_e=perc_spend_u-perc_spend_e;
perc_spend_u_vs_e=perc_spend_u_vs_e(13+10:end);
spend_dollars_u_vs_e=perc_spend_u_vs_e*total_spend_u(1);


idx_emp = (string(data_update.category) == 'Income') & (string(data_update.group) == 'Employed') & (string(data_update.measure) == 'mean')& data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Income') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
income_e=data_update_e.value;
income_u=data_update_u.value;
income_e_yoy=income_e(13:end)./income_e(1:end-12)*income_e(13);
income_u_yoy=income_u(13:end)./income_u(1:end-12)*income_u(13);
income_u_jan20=income_u(13);
income_e_jan20=income_e(13);
income_e=income_e(13+10:end);
income_u=income_u(13+10:end);
perc_income_e=data_update_e.percent_change;
perc_income_u=data_update_u.percent_change;
perc_income_u_vs_e=perc_income_u-perc_income_e;
perc_income_u_vs_e=perc_income_u_vs_e(13+10:end);
income_dollars_u_vs_e=perc_income_u_vs_e*income_u(1);


idx_emp = (string(data_update.category) == 'Checking account balance') & (string(data_update.group) == 'Employed')& (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
idx_u = (string(data_update.category) == 'Checking account balance') & startsWith(string(data_update.group), 'Unemployed') & (string(data_update.measure) == 'mean') & data_update.periodid>=201901;
data_update_e=data_update(idx_emp,:);
data_update_u=data_update(idx_u,:);
checking_e=data_update_e.value;
checking_u=data_update_u.value;
checking_e=checking_e(13:end);
checking_u=checking_u(13:end);

ratio_u=checking_u(10)/checking_u(1);
ratio_e=checking_e(10)/checking_e(1);

% Assign parameter values
load discountfactors.mat
beta_normal = beta_target500MPC;
beta_high = beta_oneperiodshock;

load model_parameters.mat
initial_a = initial_a - aprimemin;

n_ben_profiles_allowed = 3; %This captures the surprise vs. expected expiration and no FPUC

% Set on/off switches
infinite_dur = 0;
use_initial_a = 1;

% Start solving the model with EGM
for iy=1:5
    
    y = w(iy);
    h = 0.7 * y;
    b = repshare * y;

    for surprise=0:1

    rng('default')
    
    if surprise==1
        k=sse_surprise_extension_fit_het_onset_match500MPC(1);
        gamma=sse_surprise_extension_fit_het_onset_match500MPC(2);
        c_param=sse_surprise_extension_fit_het_onset_match500MPC(3);
    else
        k=sse_expect_extension_fit_het_onset_match500MPC(1);
        gamma=sse_expect_extension_fit_het_onset_match500MPC(2);
        c_param=sse_expect_extension_fit_het_onset_match500MPC(3);
    end

    % Aprime grid
    aprimemax = 2000;
    Aprime=exp(linspace(0.00,log(aprimemax),n_aprime))-1;
    Aprime=Aprime';

    %regular benefits profile
    benefit_profile(1:6,1)=h+b;
    if infinite_dur==1
        benefit_profile(7:13,1)=h+b;
    else
        benefit_profile(7:13,1)=h;
    end

    
    %expect $300 for 10 months
    benefit_profile_pandemic(1:2,1)=b+h;
    benefit_profile_pandemic(3:12,1)=b+h+FPUC_onset;
    if infinite_dur==1
        benefit_profile_pandemic(13:13,1)=b+h;
    else
        benefit_profile_pandemic(13:13,1)=h;
    end
    %expect $300 for 3 months
    benefit_profile_pandemic(1:2,2)=b+h;
    benefit_profile_pandemic(3:5,2)=b+h+FPUC_onset;
    benefit_profile_pandemic(6:10,2)=b+h;
    if infinite_dur==1
        benefit_profile_pandemic(11:13,2)=b+h+FPUC_onset;
    else
        benefit_profile_pandemic(11:13,2)=h;
    end
    

   %No FPUC
   benefit_profile_pandemic(1:10,3)=h+b;
    if infinite_dur==1
        benefit_profile_pandemic(11:13,3)=h+b;
    else
        benefit_profile_pandemic(11:13,3)=h;
    end


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

    for beta_loop=1:1

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

    tic
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


    %c_pol is c(a,y)
    %c_tilde is c(a',y)
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

    numhh=500;
    numsim=15;
    burnin=10;
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
                a_sim(i,t+1)=max(0,y+(1+r)*a_sim(i,t)-c_sim(i,t));
            else
                c_sim(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(0,benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim(i,t)-c_sim(i,t));
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
    a_sim_with_500(:,1)=a_sim_with_500(:,1)+500*FPUC_onset/(4.5*600);

    if use_initial_a==1
        load a_init_prepandemic_target500MPC;
        initial_a_vec=a_init_prepandemic_target500MPC;
        initial_a=initial_a_vec(iy);

        a_sim_pandemic_surprise = checking_u(10)/income_e(1);
        a_sim_pandemic_expect = checking_u(10)/income_e(1);
        a_sim_pandemic_expect_jan_start = checking_u(10)/income_e(1);
        a_sim_pandemic_surprise_wait = checking_u(10)/income_e(1);
        a_sim_pandemic_expect_wait = checking_u(10)/income_e(1);
        a_sim_regular = checking_u(10)/income_e(1);
        a_sim_pandemic_noFPUC = checking_u(10)/income_e(1);
        a_sim_e = checking_u(10)/income_e(1);

        a_sim_pandemic_surprise = initial_a;
        a_sim_pandemic_expect = initial_a;
        a_sim_pandemic_expect_jan_start = initial_a;
        a_sim_pandemic_surprise_wait = initial_a;
        a_sim_pandemic_expect_wait = initial_a;
        a_sim_regular = initial_a;
        a_sim_pandemic_noFPUC = initial_a;
        a_sim_e = checking_e(10)/income_e(1)*w(iy);
    else
        a_sim_pandemic_surprise=tmp_a(tmp_u>0);
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
    c_sim_pandemic_expect=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_regular=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_pandemic_noFPUC=zeros(length(a_sim_pandemic_surprise),30);
    c_sim_e=zeros(length(a_sim_e),30);


    search_sim_pandemic_surprise=zeros(length(a_sim_pandemic_surprise),30);
    search_sim_pandemic_expect=zeros(length(a_sim_pandemic_expect),30);
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
            else
                c_sim(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim(i,t),'linear');
                a_sim(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim(i,t)-c_sim(i,t),0);

                c_sim_with_500(i,t)=interp1(A,c_pol_u(:,u_dur_sim(i,t)),a_sim_with_500(i,t),'linear');
                a_sim_with_500(i,t+1)=max(benefit_profile(u_dur_sim(i,t))+(1+r)*a_sim_with_500(i,t)-c_sim_with_500(i,t),0);
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
    
    mean_c_sim_with_500=mean(c_sim_with_500);

    

    

    %this is looping over just unemployed households (continuously unemployed)
    %to get u time-series patterns
    length_u=0;
    for t=1:15
        length_u=min(length_u+1,n_b);
        for i=1:num_unemployed_hh
            %Jan EIP
            if t==3
                a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t)+EIP2*FPUC_onset/(4.5*600);
                a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t)+EIP2*FPUC_onset/(4.5*600);
            end
            if t==5
                a_sim_pandemic_expect(i,t)=a_sim_pandemic_expect(i,t)+EIP3*FPUC_onset/(4.5*600);
                a_sim_pandemic_surprise(i,t)=a_sim_pandemic_surprise(i,t)+EIP3*FPUC_onset/(4.5*600);
            end

            if t==1
                c_pol_u=c_pol_u_betanormal;
                c_pol_u_pandemic=c_pol_u_pandemic_betanormal;
            end

            c_sim_regular(i,t)=interp1(A,c_pol_u(:,length_u),a_sim_regular(i,t),'linear');
            a_sim_regular(i,t+1)=max(0,benefit_profile(length_u)+(1+r)*a_sim_regular(i,t)-c_sim_regular(i,t));

            if t<=2 || t>=11
                c_sim_pandemic_expect(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,3),a_sim_pandemic_expect(i,t),'linear');
                a_sim_pandemic_expect(i,t+1)=max(0,benefit_profile_pandemic(length_u,3)+(1+r)*a_sim_pandemic_expect(i,t)-c_sim_pandemic_expect(i,t));
            else
                c_sim_pandemic_expect(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_expect(i,t),'linear');
                a_sim_pandemic_expect(i,t+1)=max(0,benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_expect(i,t)-c_sim_pandemic_expect(i,t));
            end
            

            if t<=2 || t>=11 %pre-onset
                c_sim_pandemic_surprise(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,3),a_sim_pandemic_surprise(i,t),'linear');
                a_sim_pandemic_surprise(i,t+1)=max(0,benefit_profile_pandemic(length_u,3)+(1+r)*a_sim_pandemic_surprise(i,t)-c_sim_pandemic_surprise(i,t)); 
            elseif t>=3 && t<=4  %jan-feb
                c_sim_pandemic_surprise(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,2),a_sim_pandemic_surprise(i,t),'linear');
                a_sim_pandemic_surprise(i,t+1)=max(0,benefit_profile_pandemic(length_u,2)+(1+r)*a_sim_pandemic_surprise(i,t)-c_sim_pandemic_surprise(i,t)); 
            else %march extension
                c_sim_pandemic_surprise(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_surprise(i,t),'linear');
                a_sim_pandemic_surprise(i,t+1)=max(0,benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_surprise(i,t)-c_sim_pandemic_surprise(i,t));
            end
            
            c_sim_pandemic_noFPUC(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,3),a_sim_pandemic_noFPUC(i,t),'linear');
            a_sim_pandemic_noFPUC(i,t+1)=max(benefit_profile_pandemic(length_u,3)+(1+r)*a_sim_pandemic_noFPUC(i,t)-c_sim_pandemic_noFPUC(i,t));


            diff_v=interp1(A,v_e(:),a_sim_regular(i,t+1),'linear')-interp1(A,v_u(:,min(length_u+1,n_b)),a_sim_regular(i,t+1),'linear');
            search_sim_regular(i,t)=min(1-recall_probs_regular(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            
            diff_v=interp1(A,v_e(:),a_sim_pandemic_noFPUC(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),3),a_sim_pandemic_noFPUC(i,t+1),'linear');
            search_sim_pandemic_noFPUC(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_noFPUC(i,t))~=0 
                search_sim_pandemic_noFPUC(i,t)=0;
            end
            

            if t<=2 || t>=11
                diff_v=interp1(A,v_e(:),a_sim_pandemic_expect(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),3),a_sim_pandemic_expect(i,t+1),'linear');
            else
                diff_v=interp1(A,v_e(:),a_sim_pandemic_expect(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),1),a_sim_pandemic_expect(i,t+1),'linear');
            end
            search_sim_pandemic_expect(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_expect(i,t))~=0 
                search_sim_pandemic_expect(i,t)=0;
            end


            if t<=2 || t>=11
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),3),a_sim_pandemic_surprise(i,t+1),'linear');
            elseif t>=3 && t<=4 
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),2),a_sim_pandemic_surprise(i,t+1),'linear');
            else
                diff_v=interp1(A,v_e(:),a_sim_pandemic_surprise(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),1),a_sim_pandemic_surprise(i,t+1),'linear');
            end
            search_sim_pandemic_surprise(i,t)=min(1-recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
            if imag(search_sim_pandemic_surprise(i,t))~=0 
                search_sim_pandemic_surprise(i,t)=0;
            end




            %note for surprise case won't want to use i_b+1 in actuality will
            %want to use the expected one in the last period before surprise

        end
    end
    
    for t=1:numsim
        for i=1:num_employed_hh

             if t==3
                a_sim_e(i,t)=a_sim_e(i,t)+EIP2*FPUC_onset/(4.5*600);
             end
             if t==5
                a_sim_e(i,t)=a_sim_e(i,t)+EIP3*FPUC_onset/(4.5*600);
             end

             %adjust initial assets isomorphic to allowing for borrowing
             %if t==1
             %    a_sim_e(i,t)=a_sim_e(i,t)+5*1320*FPUC_onset/(4.5*600);
             %end

             c_sim_e(i,t)=interp1(A,c_pol_e(:),a_sim_e(i,t),'linear');
             a_sim_e(i,t+1)=y+(1+r)*a_sim_e(i,t)-c_sim_e(i,t);   
        end
    end

    if surprise==1
        mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
        mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);

        mean_a_sim_pandemic_surprise_noFPUC=mean(a_sim_pandemic_noFPUC,1);
        mean_c_sim_pandemic_surprise_noFPUC=mean(c_sim_pandemic_noFPUC,1);


        mean_a_sim_regular=mean(a_sim_regular,1);
        mean_c_sim_regular=mean(c_sim_regular,1);

        mean_a_sim_e_surprise=mean(a_sim_e,1);
        mean_c_sim_e_surprise=mean(c_sim_e,1);

        mean_search_sim_regular=mean(search_sim_regular,1);
        mean_search_sim_pandemic_surprise_noFPUC=mean(search_sim_pandemic_noFPUC,1);
        mean_search_sim_pandemic_surprise=mean(search_sim_pandemic_surprise,1);
    else
        mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
        mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);

        mean_a_sim_pandemic_expect_noFPUC=mean(a_sim_pandemic_noFPUC,1);
        mean_c_sim_pandemic_expect_noFPUC=mean(c_sim_pandemic_noFPUC,1);

        mean_search_sim_pandemic_expect=mean(search_sim_pandemic_expect,1);
        mean_search_sim_pandemic_expect_noFPUC=mean(search_sim_pandemic_noFPUC,1);
        
        mean_a_sim_e=mean(a_sim_e,1);
        mean_c_sim_e=mean(c_sim_e,1);
    end
   
    end
   % mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
   % mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);

   % mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
   % mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);

   % mean_a_sim_regular=mean(a_sim_regular,1);
   % mean_c_sim_regular=mean(c_sim_regular,1);

    mean_c_sim_e_bywage(iy,:)=mean_c_sim_e;
    mean_a_sim_e_bywage(iy,:)=mean_a_sim_e;

    mean_search_sim_regular_bywage(iy,:)=mean_search_sim_regular;
    mean_search_sim_pandemic_expect_bywage(iy,:)=mean_search_sim_pandemic_expect;
    mean_search_sim_pandemic_surprise_bywage(iy,:)=mean_search_sim_pandemic_surprise;
    mean_search_sim_pandemic_expect_noFPUC_bywage(iy,:)=mean_search_sim_pandemic_expect_noFPUC;
    mean_search_sim_pandemic_surprise_noFPUC_bywage(iy,:)=mean_search_sim_pandemic_surprise_noFPUC;
    
    mean_a_sim_regular_bywage(iy,:)=mean_a_sim_regular;
    mean_a_sim_pandemic_expect_bywage(iy,:)=mean_a_sim_pandemic_expect;
    mean_a_sim_pandemic_surprise_bywage(iy,:)=mean_a_sim_pandemic_surprise;
    mean_a_sim_pandemic_expect_noFPUC_bywage(iy,:)=mean_a_sim_pandemic_expect_noFPUC;
    mean_a_sim_pandemic_surprise_noFPUC_bywage(iy,:)=mean_a_sim_pandemic_surprise_noFPUC;
   
    
    mean_c_sim_regular_bywage(iy,:)=mean_c_sim_regular;
    mean_c_sim_pandemic_expect_bywage(iy,:)=mean_c_sim_pandemic_expect;
    mean_c_sim_pandemic_surprise_bywage(iy,:)=mean_c_sim_pandemic_surprise;
    mean_c_sim_pandemic_expect_noFPUC_bywage(iy,:)=mean_c_sim_pandemic_expect_noFPUC;
    mean_c_sim_pandemic_surprise_noFPUC_bywage(iy,:)=mean_c_sim_pandemic_surprise_noFPUC;

    mean_y_sim_e_bywage(iy,:)=w(iy)*ones(1,13);
    mean_y_sim_e_bywage(iy,3)=mean_y_sim_e_bywage(iy,3)+EIP2_e * FPUC_onset / (4.5 * 600);
    mean_y_sim_e_bywage(iy,5)=mean_y_sim_e_bywage(iy,5)+EIP3 * FPUC_onset / (4.5 * 600);
    
    
    mean_y_sim_pandemic_u_bywage(iy,:)=benefit_profile_pandemic(:,1)';
    mean_y_sim_pandemic_u_bywage(iy,3)=mean_y_sim_pandemic_u_bywage(iy,3)+EIP2*FPUC_onset/(4.5*600);
    mean_y_sim_pandemic_u_bywage(iy,5)=mean_y_sim_pandemic_u_bywage(iy,5)+EIP3*FPUC_onset/(4.5*600);

    mean_y_sim_pandemic_noFPUC_bywage(iy,:)=benefit_profile_pandemic(:,3)';
    mean_y_sim_pandemic_noFPUC_bywage(iy,3)=mean_y_sim_pandemic_noFPUC_bywage(iy,3)+EIP2*FPUC_onset/(4.5*600);
    mean_y_sim_pandemic_noFPUC_bywage(iy,5)=mean_y_sim_pandemic_noFPUC_bywage(iy,5)+EIP3*FPUC_onset/(4.5*600);


    mean_y_sim_regular_bywage(iy,:)=[benefit_profile(:,1)'];
end

mean_search_sim_pandemic_surprise=mean(mean_search_sim_pandemic_surprise_bywage,1);
mean_search_sim_pandemic_expect=mean(mean_search_sim_pandemic_expect_bywage,1);
mean_search_sim_pandemic_surprise_noFPUC=mean(mean_search_sim_pandemic_surprise_noFPUC_bywage,1);
mean_search_sim_pandemic_expect_noFPUC=mean(mean_search_sim_pandemic_expect_noFPUC_bywage,1);

mean_y_sim_pandemic_u=mean(mean_y_sim_pandemic_u_bywage,1);
mean_y_sim_pandemic_noFPUC=mean(mean_y_sim_pandemic_noFPUC_bywage,1);
mean_y_sim_e = mean(mean_y_sim_e_bywage,1);

mean_c_sim_e=mean(mean_c_sim_e_bywage,1);
mean_c_sim_pandemic_surprise=mean(mean_c_sim_pandemic_surprise_bywage,1);
mean_c_sim_pandemic_surprise_noFPUC=mean(mean_c_sim_pandemic_surprise_noFPUC_bywage,1);
mean_c_sim_pandemic_expect=mean(mean_c_sim_pandemic_expect_bywage,1);
mean_c_sim_pandemic_expect_noFPUC=mean(mean_c_sim_pandemic_expect_noFPUC_bywage,1);
mean_c_sim_pandemic_regular=mean(mean_c_sim_regular_bywage,1);

mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021=[mean_c_sim_pandemic_expect' mean_c_sim_pandemic_expect_noFPUC'];
save('mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021.mat','mean_c_with_and_withoutFPUC_prepandemic_target500MPC_2021')

exit_rates_data=readtable(jobfind_input_directory, 'Sheet', fig1_df);
exit_rates_data.week_start_date=datetime(exit_rates_data.week_start_date);
idx = datenum(exit_rates_data.week_start_date) >= datenum('2020-11-01') & datenum(exit_rates_data.week_start_date) < datenum('2021-05-21');
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
monthly_search_data=exit_rates_data_week_to_month.exit_not_to_recall';
recall_probs_pandemic_actual=exit_rates_data_week_to_month.exit_to_recall';

%Convert model simulations to dollar deviations in U vs. E space
mean_c_sim_pandemic_surprise_dollars=mean_c_sim_pandemic_surprise(1:18)./mean_c_sim_e(1:18)*total_spend_u_jan20-total_spend_u_jan20;
mean_c_sim_pandemic_expect_dollars=mean_c_sim_pandemic_expect(1:18)./mean_c_sim_e(1:18)*total_spend_u_jan20-total_spend_u_jan20;
mean_c_sim_pandemic_surprise_noFPUC_dollars=mean_c_sim_pandemic_surprise_noFPUC(1:18)./mean_c_sim_e(1:18)*total_spend_u_jan20-total_spend_u_jan20;
mean_c_sim_pandemic_expect_noFPUC_dollars=mean_c_sim_pandemic_expect_noFPUC(1:18)./mean_c_sim_e(1:18)*total_spend_u_jan20-total_spend_u_jan20;
mean_c_sim_e_dollars=mean_c_sim_e(1:18)./mean_c_sim_e(1:18)*total_spend_e_jan20-total_spend_e_jan20;
mean_y_sim_pandemic_u_dollars = mean_y_sim_pandemic_u./mean_y_sim_e * total_spend_u_jan20 - total_spend_u_jan20;
mean_y_sim_e_dollars=0;
mean_y_sim_pandemic_u_dollars=mean_y_sim_pandemic_u_dollars+(mean(income_dollars_u_vs_e(1:2))-mean_y_sim_pandemic_u_dollars(1));

scale_factor=(total_spend_e_jan20(1)/income_e_jan20(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));

% close all
figure
p=patch([3 3 4 4],[0.075 .11 .11 0.075],[0.9 0.9 0.9],'EdgeColor','none');
set(get(get(p(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on
plot(1:4,monthly_search_data(1:4),'--',1:4,mean_search_sim_pandemic_expect(1:4),1:4,mean_search_sim_pandemic_surprise(1:4),'LineWidth',2)
legend('New Job Finding: Data','New Job Finding: Expected Through Sept Model','New Job Finding: Surprise March Extension Model','Location','NorthWest','Location','SouthWest')
xticks([1 2 3 4])
xticklabels(label_months_nov20_feb21)
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(gca,'fontsize', 12); 
set(gca, 'Layer', 'top');
set(gcf, 'PaperPosition', [0 0 10.4 4.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [7.8 4.5]); %Keep the same paper size
fig=gcf;
%saveas(fig, fullfile(release_path, 'search_onset_target500MPC.png'))

figure
p=patch([3 3 4 4],[-500 100 100 -500],[0.9 0.9 0.9],'EdgeColor','none');
set(get(get(p(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on
plot(1:4,spend_dollars_u_vs_e(1:4),'--',1:4,mean_c_sim_pandemic_expect_dollars(1:4),1:4,mean_c_sim_pandemic_surprise_dollars(1:4),'LineWidth',2)
legend('C Data','C: Expected Through Sept Model','C: Surprise March Extension Model','Location','NorthWest')
xticks([1 2 3 4])
xticklabels(label_months_nov20_feb21)
yticks([-500 -250 0 250])
yticklabels({'-$500','-$250','$0', '$5000'})
set(gca,'fontsize', 12); 
set(gca, 'Layer', 'top');
set(gcf, 'PaperPosition', [0 0 10.4 4.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [7.8 4.5]); %Keep the same paper size
fig=gcf;
%saveas(fig, fullfile(release_path, 'spend_onset_target500MPC.png'))


newjob_exit_rate_FPUC=mean_search_sim_pandemic_expect(1:numsim)';
newjob_exit_rate_no_FPUC=mean_search_sim_pandemic_expect_noFPUC(1:numsim)';
newjob_exit_rate_FPUC(end:1000)=newjob_exit_rate_FPUC(end);
newjob_exit_rate_no_FPUC(end:1000)=newjob_exit_rate_no_FPUC(end);
recall_probs=recall_probs_pandemic_actual';
recall_probs(end:1000)=recall_probs(end);
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
[elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);

% Wrapper for elasticity and distortion results
elasticity_and_distortions_values_prepandemic_onset = [elasticity, employment_distortion, total_diff_employment, share_unemployment_reduced];

mean_c_sim_prepandemic_expect_dollars=mean_c_sim_pandemic_expect_dollars;
mean_c_sim_prepandemic_expect_noFPUC_dollars=mean_c_sim_pandemic_expect_noFPUC_dollars;
mean_c_sim_prepandemic_surprise_dollars=mean_c_sim_pandemic_surprise_dollars;
mean_c_sim_prepandemic_surprise_noFPUC_dollars=mean_c_sim_pandemic_surprise_noFPUC_dollars;
mean_search_sim_prepandemic_expect=mean_search_sim_pandemic_expect;
mean_search_sim_prepandemic_surprise=mean_search_sim_pandemic_surprise;

newjob_exit_rate_prepandemic_FPUC_2021 = newjob_exit_rate_FPUC;
newjob_exit_rate_prepandemic_no_FPUC_2021 = newjob_exit_rate_no_FPUC;
save('prepandemic_newjob_2021','newjob_exit_rate_prepandemic_FPUC_2021','newjob_exit_rate_prepandemic_no_FPUC_2021')

mpc_supplements_prepandemic_onset_target500MPC = table();
mpc_supplements_prepandemic_onset_target500MPC.expect('one_month') = (mean_c_sim_pandemic_expect(3) - mean_c_sim_pandemic_expect_noFPUC(3)) / (mean_y_sim_pandemic_u(3) - mean_y_sim_pandemic_noFPUC(3));
mpc_supplements_prepandemic_onset_target500MPC.expect('3_month') = sum(mean_c_sim_pandemic_expect(3:5) - mean_c_sim_pandemic_expect_noFPUC(3:5)) / sum(mean_y_sim_pandemic_u(3:5) - mean_y_sim_pandemic_noFPUC(3:5));
mpc_supplements_prepandemic_onset_target500MPC.expect('6_month') = sum(mean_c_sim_pandemic_expect(3:8) - mean_c_sim_pandemic_expect_noFPUC(3:8)) / sum(mean_y_sim_pandemic_u(3:8) - mean_y_sim_pandemic_noFPUC(3:8));
mpc_supplements_prepandemic_onset_target500MPC.expect('full') = sum(mean_c_sim_pandemic_expect(3:10) - mean_c_sim_pandemic_expect_noFPUC(3:10)) / sum(mean_y_sim_pandemic_u(3:10) - mean_y_sim_pandemic_noFPUC(3:10));
mpc_supplements_prepandemic_onset_target500MPC.expect('full+3') = sum(mean_c_sim_pandemic_expect(3:13) - mean_c_sim_pandemic_expect_noFPUC(3:13)) / sum(mean_y_sim_pandemic_u(3:13) - mean_y_sim_pandemic_noFPUC(3:13));
mpc_supplements_prepandemic_onset_target500MPC.Variables=scale_factor*mpc_supplements_prepandemic_onset_target500MPC.Variables

save('prepandemic_results_onset_target500MPC','mpc_supplements_prepandemic_onset_target500MPC','mean_c_sim_prepandemic_expect_dollars','mean_c_sim_prepandemic_expect_noFPUC_dollars','mean_search_sim_prepandemic_expect','mean_c_sim_prepandemic_surprise_dollars','mean_c_sim_prepandemic_surprise_noFPUC_dollars','mean_search_sim_prepandemic_surprise','elasticity_and_distortions_values_prepandemic_onset')

