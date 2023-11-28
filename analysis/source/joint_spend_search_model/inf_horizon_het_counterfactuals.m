display('Simulating Full Model Effects of Different Supplement Sizes')
clearvars -except -regexp fig_paper_*
tic

load jobfind_input_directory.mat 
load jobfind_input_sheets.mat
load spending_input_directory.mat 
load spending_input_sheets.mat
load hh_wage_groups.mat
load release_paths.mat

load bestfit_prepandemic.mat
load bestfit_target_waiting_MPC.mat

load matlab_qual_colors.mat
global qual_blue qual_purple qual_green qual_orange matlab_red_orange qual_yellow



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



shock_500=500/income_e(1);


k_prepandemic=pre_pandemic_fit_match500MPC(1);
gamma_prepandemic=pre_pandemic_fit_match500MPC(2);
c_param_prepandemic=0;

FPUCsize=.35;

for pre_pandemic=0:0
    

FPUC_length=[2 3 4];
FPUC_mult=0/6:1/12:1;

for FPUC_length_index=1:length(FPUC_length)
for FPUC_mult_index=1:length(FPUC_mult)
    
    % Assign parameter values
    load discountfactors.mat
    beta_normal = beta_targetwaiting;
    if FPUC_length_index==1
        beta_high = beta_normal; %FPUC length 1 will be the non pandemic counterfactual exercises
    else
        beta_high = beta_oneperiodshock;
    end

    load model_parameters.mat
    initial_a = initial_a - aprimemin;

    n_ben_profiles_allowed=3;

    hshare = .7;

    % Set on/off switches
    infinite_dur=0;
    use_initial_a=0;

    % Start solving the model with EGM
    for iy=1:5
        
        y = w(iy);
        y = 1;
        h = hshare * y;
        b = repshare * y;

        FPUC = FPUCsize * FPUC_mult(FPUC_mult_index) * y;
        LWAsize = 0;

        if iy==3
            median_rep_rate(FPUC_mult_index)=(h+b+FPUC)/y;
        end

        for surprise=0:0

        rng('default')

        k=sse_surprise_fit_het_full(1);
        gamma=sse_surprise_fit_het_full(2);
        c_param=sse_surprise_fit_het_full(3);

        if pre_pandemic==1
            k=pre_pandemic_fit_match500MPC(1);
            gamma=pre_pandemic_fit_match500MPC(2);
            c_param=0;
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

        %expect $600 for FPUC_length months
        benefit_profile_pandemic(1:FPUC_length(FPUC_length_index),1)=b+h+FPUC;
        if FPUC_length(FPUC_length_index)<12
            benefit_profile_pandemic(FPUC_length(FPUC_length_index)+1:12,1)=b+h;
        end
        if infinite_dur==1
            benefit_profile_pandemic(13,1)=b+h;
        else
            benefit_profile_pandemic(13,1)=h;
        end
        
        %No FPUC
        benefit_profile_pandemic(1:12,2)=h+b;
        if infinite_dur==1
            benefit_profile_pandemic(13,2)=h+b;
        else
            benefit_profile_pandemic(13,2)=h;
        end
        
        
        benefit_profile_pandemic(1:12,3)=b+h+FPUC;
        if infinite_dur==1
            benefit_profile_pandemic(13,3)=b+h;
        else
            benefit_profile_pandemic(13,3)=h;
        end
        

        recall_probs_pandemic(1:13,1)=0.00;
        recall_probs_regular=recall_probs_pandemic;


        %recall_probs_pandemic_actual(1)=.0078;
        %recall_probs_pandemic_actual(2)=.113;
        %recall_probs_pandemic_actual(3)=.18;
        %recall_probs_pandemic_actual(4)=.117;
        %recall_probs_pandemic_actual(5)=.112;
        %recall_probs_pandemic_actual(6:13)=.107;

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
            v_u_guess(:,ib)=((c_pol_u_guess(:,ib)).^(1-mu)-1)/(1-mu)- (k_prepandemic * 0^ (1 + gamma_prepandemic)) / (1 + gamma_prepandemic) + c_param_prepandemic;
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

                tmp=min(1-recall_probs_regular(ib),max(0,(beta*(v_e_guess(:)-v_u_guess(:,min(ib+1,n_b))) / k_prepandemic).^(1 / gamma_prepandemic)));
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
                v_u(:,ib)=((c_pol_u(:,ib)).^(1-mu)-1)/(1-mu)- (k_prepandemic * optimal_search(:,ib).^ (1 + gamma_prepandemic)) / (1 + gamma_prepandemic) + c_param_prepandemic+beta*((optimal_search(:,ib)+recall_probs_regular(ib)).*interp1(Aprime,v_e_guess(:),a_prime_holder_u(:,ib),'linear','extrap')+(1-optimal_search(:,ib)-recall_probs_regular(ib)).*interp1(Aprime,v_u_guess(:,min(ib+1,n_b)),a_prime_holder_u(:,ib),'linear','extrap'));
            end



           %pandemic unemployed 
           for i_ben_profile=1:n_ben_profiles_allowed

                %tic;
               for ib=1:n_b

                    tmp=min(1-recall_probs_pandemic(ib),max(0,(beta*(v_e_guess(:)-v_u_pandemic_guess(:,min(ib+1,n_b),i_ben_profile)) / k).^(1 / gamma)));
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
        a_sim_with_2400(:,1)=a_sim_with_500(:,1)+FPUC;

        if use_initial_a==1
            a_sim_pandemic_expect=tmp_a(tmp_u>0)+initial_a;
            a_sim_pandemic_surprise=tmp_a(tmp_u>0)+initial_a;
            a_sim_pandemic_noFPUC=tmp_a(tmp_u>0)+initial_a;
            a_sim_e=tmp_a(tmp_u==0)+initial_a;
        else
            a_sim_pandemic_expect=tmp_a(tmp_u>0);
            a_sim_pandemic_surprise=tmp_a(tmp_u>0);
            a_sim_pandemic_noFPUC=tmp_a(tmp_u>0);
            a_sim_e=tmp_a(tmp_u==0);
        end

        num_unemployed_hh=length(a_sim_pandemic_expect);
        num_employed_hh=length(a_sim_e);
        c_sim_pandemic_expect=zeros(length(a_sim_pandemic_expect),30);
        c_sim_pandemic_surprise=zeros(length(a_sim_pandemic_expect),30);
        c_sim_pandemic_noFPUC=zeros(length(a_sim_pandemic_expect),30);
        c_sim_e=zeros(length(a_sim_e),30);


        
        search_sim_pandemic_expect=zeros(length(a_sim_pandemic_expect),30);
        search_sim_pandemic_noFPUC=zeros(length(a_sim_pandemic_expect),30);
        search_sim_pandemic_surprise=zeros(length(a_sim_pandemic_expect),30);

 
        %this is looping over just unemployed households (continuously unemployed)
        %to get u time-series patterns
        length_u=0;
        for t=1:numsim
            length_u=min(length_u+1,n_b);
            c_pol_u = c_pol_u_betanormal;
            if t == 1
                c_pol_u_pandemic = c_pol_u_pandemic_betahigh;
                %v_e=v_e_betahigh;
                %v_u=v_u_betahigh;
                %v_u_pandemic=v_u_pandemic_betahigh;
                %beta=beta_high;
            else
                c_pol_u_pandemic = c_pol_u_pandemic_betanormal;
            end
            v_e=v_e_betanormal;
            v_u=v_u_betanormal;
            v_u_pandemic=v_u_pandemic_betanormal;
            beta=beta_normal;
            
            for i=1:num_unemployed_hh
                
                if length_u <= FPUC_length(FPUC_length_index)
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 3), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 3) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                else
                    c_sim_pandemic_surprise(i, t) = interp1(A, c_pol_u_pandemic(:, length_u, 1), a_sim_pandemic_surprise(i, t), 'linear');
                    a_sim_pandemic_surprise(i, t + 1) = max(benefit_profile_pandemic(length_u, 1) + (1 + r) * a_sim_pandemic_surprise(i, t) - c_sim_pandemic_surprise(i, t), 0);
                end

                c_sim_pandemic_expect(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,1),a_sim_pandemic_expect(i,t),'linear');
                a_sim_pandemic_expect(i,t+1)=max(benefit_profile_pandemic(length_u,1)+(1+r)*a_sim_pandemic_expect(i,t)-c_sim_pandemic_expect(i,t),0);

                %Note this will vary with params, but can just save it accordingly
                %when taking means later
                c_sim_pandemic_noFPUC(i,t)=interp1(A,c_pol_u_pandemic(:,length_u,2),a_sim_pandemic_noFPUC(i,t),'linear');
                a_sim_pandemic_noFPUC(i,t+1)=max(benefit_profile_pandemic(length_u,2)+(1+r)*a_sim_pandemic_noFPUC(i,t)-c_sim_pandemic_noFPUC(i,t),0);


                diff_v=interp1(A,v_e(:),a_sim_pandemic_noFPUC(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),2),a_sim_pandemic_noFPUC(i,t+1),'linear');
                search_sim_pandemic_noFPUC(i,t)=min(1 - recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_noFPUC(i,t))~=0 
                    search_sim_pandemic_noFPUC(i,t)=0;
                end

                diff_v=interp1(A,v_e(:),a_sim_pandemic_expect(i,t+1),'linear')-interp1(A,v_u_pandemic(:,min(length_u+1,n_b),1),a_sim_pandemic_expect(i,t+1),'linear');
                search_sim_pandemic_expect(i,t)=min(1 - recall_probs_pandemic(ib),max(0,(beta*(diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_expect(i,t))~=0 
                    search_sim_pandemic_expect(i,t)=0;
                end
                
                if length_u <= FPUC_length(FPUC_length_index)
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 3), a_sim_pandemic_surprise(i, t + 1), 'linear');
                else
                    diff_v = interp1(A, v_e(:), a_sim_pandemic_surprise(i, t + 1), 'linear') - interp1(A, v_u_pandemic(:, min(length_u + 1, n_b), 1), a_sim_pandemic_surprise(i, t + 1), 'linear');
                end
                search_sim_pandemic_surprise(i, t) = min(1 - recall_probs_pandemic(ib), max(0, (beta * (diff_v) / k).^(1 / gamma)));
                if imag(search_sim_pandemic_surprise(i, t)) ~= 0
                    search_sim_pandemic_surprise(i, t) = 0;
                end



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
                    a_sim_e(i,t)=a_sim_e(i,t)+1500*FPUC/(4.5*600);
                 end

                 %adjust initial assets isomorphic to allowing for borrowing
                 %if t==1
                 %    a_sim_e(i,t)=a_sim_e(i,t)+5*1320*FPUC/(4.5*600);
                 %end

                 c_sim_e(i,t)=interp1(A,c_pol_e(:),a_sim_e(i,t),'linear');
                 a_sim_e(i,t+1)=y+(1+r)*a_sim_e(i,t)-c_sim_e(i,t);   
            end
        end



            mean_a_sim_e=mean(a_sim_e,1);
            mean_c_sim_e=mean(c_sim_e,1);

            mean_a_sim_pandemic_expect=mean(a_sim_pandemic_expect,1);
            mean_c_sim_pandemic_expect=mean(c_sim_pandemic_expect,1);
            
            mean_a_sim_pandemic_surprise=mean(a_sim_pandemic_surprise,1);
            mean_c_sim_pandemic_surprise=mean(c_sim_pandemic_surprise,1);

            mean_a_sim_pandemic_expect_noFPUC=mean(a_sim_pandemic_noFPUC,1);
            mean_c_sim_pandemic_expect_noFPUC=mean(c_sim_pandemic_noFPUC,1);

            mean_search_sim_pandemic_expect=mean(search_sim_pandemic_expect,1);
            mean_search_sim_pandemic_expect_noFPUC=mean(search_sim_pandemic_noFPUC,1);
            mean_search_sim_pandemic_surprise=mean(search_sim_pandemic_surprise,1);

        end

        mean_c_sim_e_bywage(iy,:)=mean_c_sim_e;
        mean_a_sim_e_bywage(iy,:)=mean_a_sim_e;

        %paste on initial Jan-March 3 months of employment 
        mean_a_sim_pandemic_expect_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_expect(1:numsim-3)];
        mean_c_sim_pandemic_expect_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_expect(1:numsim-3)];
        mean_a_sim_pandemic_expect_noFPUC_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_expect_noFPUC(1:numsim-3)];
        mean_c_sim_pandemic_expect_noFPUC_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_expect_noFPUC(1:numsim-3)];
        mean_a_sim_pandemic_surprise_bywage(iy,:)=[mean_a_sim_e(1:3) mean_a_sim_pandemic_surprise(1:numsim-3)];
        mean_c_sim_pandemic_surprise_bywage(iy,:)=[mean_c_sim_e(1:3) mean_c_sim_pandemic_surprise(1:numsim-3)];
        

        mean_y_sim_pandemic_u_bywage(iy,:)=[y y y benefit_profile_pandemic(:,1)'];
        mean_y_sim_pandemic_u_bywage(iy,9)=mean_y_sim_pandemic_u_bywage(iy,9)+LWAsize;
        mean_y_sim_pandemic_u_bywage(iy,13)=mean_y_sim_pandemic_u_bywage(iy,13)+1500*FPUC/(4.5*600);
        mean_y_sim_pandemic_noFPUC_bywage(iy,:)=[y y y benefit_profile_pandemic(:,2)'];
        mean_y_sim_pandemic_noFPUC_bywage(iy,13)=mean_y_sim_pandemic_noFPUC_bywage(iy,13)+1500*FPUC/(4.5*600);

        mean_y_sim_e_bywage(iy,:)=y*ones(16,1);


       
        mean_search_sim_pandemic_expect_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_expect(1:numsim-3)];
        mean_search_sim_pandemic_expect_noFPUC_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_expect_noFPUC(1:numsim-3)];
        mean_search_sim_pandemic_surprise_bywage(iy,:)=[NaN NaN NaN mean_search_sim_pandemic_surprise(1:numsim-3)];
        
        mean_c_sim_pandemic_expect_vs_e_bywage(iy,:)=mean_c_sim_pandemic_expect./mean_c_sim_e;
        mean_c_sim_pandemic_surprise_vs_e_bywage(iy,:)=mean_c_sim_pandemic_surprise./mean_c_sim_e;
        
    end

    mean_a_sim_e=mean(mean_a_sim_e_bywage,1);
    mean_a_sim_pandemic_expect=mean(mean_a_sim_pandemic_expect_bywage,1);
    mean_a_sim_pandemic_expect_noFPUC=mean(mean_a_sim_pandemic_expect_noFPUC_bywage,1);
    mean_a_sim_pandemic_surprise=mean(mean_a_sim_pandemic_surprise_bywage,1);

    mean_y_sim_pandemic_u=mean(mean_y_sim_pandemic_u_bywage,1);
    mean_y_sim_pandemic_noFPUC=mean(mean_y_sim_pandemic_noFPUC_bywage,1);
    mean_y_sim_e = mean(mean_y_sim_e_bywage, 1);

    mean_c_sim_e=mean(mean_c_sim_e_bywage,1);
    mean_c_sim_pandemic_expect=mean(mean_c_sim_pandemic_expect_bywage,1);
    mean_c_sim_pandemic_expect_noFPUC=mean(mean_c_sim_pandemic_expect_noFPUC_bywage,1);
    mean_c_sim_pandemic_surprise=mean(mean_c_sim_pandemic_surprise_bywage,1);

    mean_search_sim_pandemic_expect=mean(mean_search_sim_pandemic_expect_bywage,1);
    mean_search_sim_pandemic_expect_noFPUC=mean(mean_search_sim_pandemic_expect_noFPUC_bywage,1);
    mean_search_sim_pandemic_surprise=mean(mean_search_sim_pandemic_surprise_bywage,1);

    %Convert model simulations to dollar deviations in U vs. E space
    mean_c_sim_pandemic_expect_dollars=mean_c_sim_pandemic_expect./mean_c_sim_e(1:18)*income_u(1)-income_u(1);
    mean_c_sim_pandemic_expect_noFPUC_dollars=mean_c_sim_pandemic_expect_noFPUC./mean_c_sim_e(1:18)*income_u(1)-income_u(1);
    mean_c_sim_e_dollars=mean_c_sim_e(1:18)./mean_c_sim_e(1:18)*income_e(1)-income_e(1);
    mean_c_sim_pandemic_surprise_dollars=mean_c_sim_pandemic_surprise./mean_c_sim_e(1:18)*income_u(1)-income_u(1);

    mean_y_sim_pandemic_u_dollars=mean_y_sim_pandemic_u*income_u(1)-income_u(1);
    mean_y_sim_pandemic_noFPUC_dollars=mean_y_sim_pandemic_noFPUC*income_u(1)-income_u(1);
    mean_y_sim_e_dollars=0;

    scale_factor=(total_spend_e(1)/income_e(1))/(mean_c_sim_e(1)/mean_y_sim_e(1));


    mpc_supplements=table();
    mpc_supplements.expect('one_month')=(mean_c_sim_pandemic_expect(4)-mean_c_sim_pandemic_expect_noFPUC(4))/(mean_y_sim_pandemic_u(4)-mean_y_sim_pandemic_noFPUC(4));
    mpc_supplements.expect('3_month')=sum(mean_c_sim_pandemic_expect(4:6)-mean_c_sim_pandemic_expect_noFPUC(4:6))/sum(mean_y_sim_pandemic_u(4:6)-mean_y_sim_pandemic_noFPUC(4:6));
    mpc_supplements.expect('6_month')=sum(mean_c_sim_pandemic_expect(4:9)-mean_c_sim_pandemic_expect_noFPUC(4:9))/sum(mean_y_sim_pandemic_u(4:9)-mean_y_sim_pandemic_noFPUC(4:9));
    mpc_supplements.surprise('one_month')=(mean_c_sim_pandemic_surprise(4)-mean_c_sim_pandemic_expect_noFPUC(4))/(mean_y_sim_pandemic_u(4)-mean_y_sim_pandemic_noFPUC(4));
    mpc_supplements.surprise('3_month')=sum(mean_c_sim_pandemic_surprise(4:6)-mean_c_sim_pandemic_expect_noFPUC(4:6))/sum(mean_y_sim_pandemic_u(4:6)-mean_y_sim_pandemic_noFPUC(4:6));
    mpc_supplements.surprise('6_month')=sum(mean_c_sim_pandemic_surprise(4:9)-mean_c_sim_pandemic_expect_noFPUC(4:9))/sum(mean_y_sim_pandemic_u(4:9)-mean_y_sim_pandemic_noFPUC(4:9));
      
    
    if FPUC_mult_index>1
        mpc_supplements.expect_lasthundred('one_month')=(mean_c_sim_pandemic_expect(4)-mean_c_sim_pandemic_expect_smaller(4))/(mean_y_sim_pandemic_u(4)-mean_y_sim_pandemic_u_smaller(4));
        mpc_supplements.expect_lasthundred('3_month')=sum(mean_c_sim_pandemic_expect(4:6)-mean_c_sim_pandemic_expect_smaller(4:6))/sum(mean_y_sim_pandemic_u(4:6)-mean_y_sim_pandemic_u_smaller(4:6));
        mpc_supplements.expect_lasthundred('6_month')=sum(mean_c_sim_pandemic_expect(4:9)-mean_c_sim_pandemic_expect_smaller(4:9))/sum(mean_y_sim_pandemic_u(4:9)-mean_y_sim_pandemic_u_smaller(4:9));
        mpc_supplements.surprise_lasthundred('one_month')=(mean_c_sim_pandemic_surprise(4)-mean_c_sim_pandemic_surprise_smaller(4))/(mean_y_sim_pandemic_u(4)-mean_y_sim_pandemic_u_smaller(4));
        mpc_supplements.surprise_lasthundred('3_month')=sum(mean_c_sim_pandemic_surprise(4:6)-mean_c_sim_pandemic_surprise_smaller(4:6))/sum(mean_y_sim_pandemic_u(4:6)-mean_y_sim_pandemic_u_smaller(4:6));
        mpc_supplements.surprise_lasthundred('6_month')=sum(mean_c_sim_pandemic_surprise(4:9)-mean_c_sim_pandemic_surprise_smaller(4:9))/sum(mean_y_sim_pandemic_u(4:9)-mean_y_sim_pandemic_u_smaller(4:9));
        mpc_supplements.Variables=scale_factor*mpc_supplements.Variables;
        
        
        
        mean_c_sim_pandemic_expect_smaller=mean_c_sim_pandemic_expect;
        mean_c_sim_pandemic_surprise_smaller=mean_c_sim_pandemic_surprise;
        mean_y_sim_pandemic_u_smaller=mean_y_sim_pandemic_u;
    else
        mpc_supplements.expect_lasthundred('one_month')=mpc_supplements.expect('one_month');
        mpc_supplements.expect_lasthundred('3_month')=mpc_supplements.expect('3_month');
        mpc_supplements.expect_lasthundred('6_month')=mpc_supplements.expect('6_month')
        mpc_supplements.surprise_lasthundred('one_month')=mpc_supplements.surprise('one_month');
        mpc_supplements.surprise_lasthundred('3_month')=mpc_supplements.surprise('3_month');
        mpc_supplements.surprise_lasthundred('6_month')=mpc_supplements.surprise('6_month')
        mpc_supplements.Variables=scale_factor*mpc_supplements.Variables;

        mean_c_sim_pandemic_expect_smaller=mean_c_sim_pandemic_expect;
        mean_c_sim_pandemic_surprise_smaller=mean_c_sim_pandemic_surprise;
        mean_y_sim_pandemic_u_smaller=mean_y_sim_pandemic_u;
    end
    
    mpc_supplements_by_size(:,:,FPUC_mult_index,FPUC_length_index)=table2array(mpc_supplements);


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
    recall_probs_pandemic_actual=exit_rates_data_week_to_month.exit_to_recall';
   

    newjob_exit_rate_FPUC=mean_search_sim_pandemic_expect(4:numsim)';
    newjob_exit_rate_no_FPUC=mean_search_sim_pandemic_expect_noFPUC(4:numsim)';
    newjob_exit_rate_FPUC_surprise=mean_search_sim_pandemic_surprise(4:numsim)';

    mean_c_sim_pandemic_expect_overall_FPUC(1:7)=mean_c_sim_pandemic_expect(4:10);
    mean_c_sim_pandemic_expect_overall_noFPUC(1:7)=mean_c_sim_pandemic_expect_noFPUC(4:10);
    mean_c_sim_pandemic_SURPRISE_overall_FPUC(1:7)=mean_c_sim_pandemic_surprise(4:10);

    newjob_exit_rate_overall_FPUC=mean_search_sim_pandemic_expect(4:end)';
    newjob_exit_rate_overall_no_FPUC=newjob_exit_rate_no_FPUC;

    if pre_pandemic==1
        %use pre pan search costs but adjust to depressed no supp job find
        %rate in fall
        %newjob_exit_rate_FPUC=newjob_exit_rate_FPUC*(mean(monthly_search_data(5:8))/mean(newjob_exit_rate_overall_no_FPUC(5:8)));
        %newjob_exit_rate_no_FPUC=newjob_exit_rate_FPUC*(mean(monthly_search_data(5:8))/mean(newjob_exit_rate_overall_no_FPUC(5:8)));
    end

    newjob_exit_rate_overall_FPUC_surprise=mean_search_sim_pandemic_surprise(4:end)';
    newjob_exit_rate_overall_FPUC(end:1000)=newjob_exit_rate_overall_FPUC(end);
    newjob_exit_rate_overall_no_FPUC(end:1000)=newjob_exit_rate_overall_no_FPUC(end);
    newjob_exit_rate_overall_FPUC_surprise(end:1000)=newjob_exit_rate_overall_FPUC_surprise(end);
    newjob_exit_rate_FPUC(end:1000) = newjob_exit_rate_FPUC(end);
    newjob_exit_rate_no_FPUC(end:1000) = newjob_exit_rate_no_FPUC(end);
    newjob_exit_rate_FPUC_surprise(end:1000) = newjob_exit_rate_FPUC_surprise(end);
    
    recall_probs = recall_probs_pandemic_actual';
    recall_probs(end:1000) = recall_probs(end);
    recall_probs=.08*ones(1000,1);

    %weekly_or_monthly='monthly';
    %onset_or_expiry='expiry';
    %include_self_employed=0;
    %global nofigs
    %nofigs=0;
    %elasticity_and_distortions_values_by_size(:,FPUC_mult_index)=(elasticity_and_distortions(newjob_exit_rate_overall_FPUC, newjob_exit_rate_overall_no_FPUC, recall_probs, weekly_or_monthly, onset_or_expiry,include_self_employed))


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
    [elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
    elasticity_and_distortions_values_by_size(1:4,FPUC_mult_index,FPUC_length_index)=[elasticity employment_distortion total_diff_employment share_unemployment_reduced]';
    elasticity_and_distortions_values_by_size(5,FPUC_mult_index,FPUC_length_index)=newjob_exit_rate_no_FPUC(1);
    elasticity_and_distortions_values_by_size(6,FPUC_mult_index,FPUC_length_index)=newjob_exit_rate_FPUC(1);
    elasticity_and_distortions_values_by_size(7,FPUC_mult_index,FPUC_length_index)=elasticity*perc_change_benefits_data;
    elasticity_and_distortions_values_by_size(8,FPUC_mult_index,FPUC_length_index)=average_duration(recall_probs+newjob_exit_rate_FPUC)
    
    [elasticity employment_distortion total_diff_employment share_unemployment_reduced employment_FPUC employment_noFPUC monthly_spend_pce monthly_spend_no_FPUC] = elasticity_distortions_and_aggregates(newjob_exit_rate_FPUC_surprise, newjob_exit_rate_no_FPUC, recall_probs, mean_c_sim_pandemic_surprise_overall_FPUC, mean_c_sim_pandemic_surprise_overall_noFPUC, mean_c_sim_e_overall, perc_change_benefits_data, date_sim_start, t_start, t_end, include_self_employed);
    elasticity_and_distortions_values_by_size_surprise(1:4,FPUC_mult_index,FPUC_length_index)=[elasticity employment_distortion total_diff_employment share_unemployment_reduced]';
    elasticity_and_distortions_values_by_size_surprise(5,FPUC_mult_index,FPUC_length_index)=newjob_exit_rate_no_FPUC(1);
    elasticity_and_distortions_values_by_size_surprise(6,FPUC_mult_index,FPUC_length_index)=newjob_exit_rate_FPUC_surprise(1);
    elasticity_and_distortions_values_by_size_surprise(7,FPUC_mult_index,FPUC_length_index)=elasticity*perc_change_benefits_data;
    elasticity_and_distortions_values_by_size_surprise(8,FPUC_mult_index,FPUC_length_index)=average_duration(recall_probs+newjob_exit_rate_FPUC_surprise)
    toc
    
    mean_c_month1(FPUC_mult_index,FPUC_length_index)=mean_c_sim_pandemic_expect(5);
    
end
end


if pre_pandemic==0


duration_increase_total=squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1,:))';
duration_increase_marginal=squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:));
duration_growth_total=(squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1,:))')./squeeze(elasticity_and_distortions_values_by_size(8,1,:))';
duration_growth_marginal=(squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:)))./squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:));

duration_increase_total_surprise=squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))';
duration_increase_marginal_surprise=squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:));
duration_growth_total_surprise=(squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))')./squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))';
duration_growth_marginal_surprise=(squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:)))./squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:));

 

%figure
%tiledlayout(1,2)
%nexttile
%hold on
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(1,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(1,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%title('One month MPCs')
%ylabel('MPC')
%xlabel('UI supplement size (per week)')
%xticks([0 300 600])
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(600*FPUC_mult(2:end),4.2*duration_increase_total_surprise(:,3),'LineWidth',2,'Color',qual_blue)
%plot(600*FPUC_mult(2:end),4.2*duration_increase_marginal_surprise(:,3),'--','LineWidth',2,'Color',qual_blue)
%title('Change in unemp. duration')
%ylabel('Change (in weeks)')
%xlabel('UI supplement size (per week)')
%ylim([0 4])
%xticks([0 300 600])
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%set(gca, 'Layer', 'top');
%lg  = legend('Total effect of supplement','Effect from last $50 of supplement', 'FontSize', 14);
%lg.Layout.Tile = 'South';
%grid on
%fig=gcf;
%set(gcf, 'PaperPosition', [0 0 10 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
%set(gcf, 'PaperSize', [10 5]); %Keep the same paper size
%fig_paper_12 = gcf
%saveas(fig_paper_12, fullfile(release_path_paper, 'U_and_C_total_and_marginal_4month_surprise.png'))
%saveas(fig_paper_12, fullfile(release_path_slides, 'U_and_C_total_and_marginal_4month_surprise.png'))



%figure
%tiledlayout(1,3)
%nexttile
%hold on
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(1,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(1,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('One month MPCs')
%ylabel('MPC')
%xlabel('UI supplement size (per week)')
%xticks([0 300 600])
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(2,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(2,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('Three month MPCs')
%ylabel('MPC')
%xlabel('UI supplement size (per week)')
%xticks([0 300 600])
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(3,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(600*FPUC_mult,squeeze(mpc_supplements_by_size(3,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('Six month MPCs')
%ylabel('MPC')
%xlabel('UI supplement size (per week)')
%xticks([0 300 600])
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%set(gca, 'Layer', 'top');
%lg  = legend('Total effect of supplement','Effect from last $50 of supplement', 'FontSize', 14);
%lg.Layout.Tile = 'South';
%grid on
%set(gcf, 'PaperPosition', [0 0 10 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
%set(gcf, 'PaperSize', [10 5]); %Keep the same paper size
%fig_paper_13 = gcf;
%saveas(fig_paper_13, fullfile(release_path_paper, 'U_and_C_total_and_marginal_diff_mpc_horizons_surprise.png'))

load stimulus_check_size_results
load stimulus_check_size_results_onetenth


figure
hold on
s=squeeze(mpc_supplements_by_size(2,3,:,1));
s(2:end)=smooth(s(2:end));
plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(2,1,:,1)),'LineWidth',2,'Color',qual_blue)
plot(2400*FPUC_mult,s,'--','LineWidth',2,'Color',qual_blue)
plot(2400*FPUC_mult(2:end),mpc_by_size_stimcheck_quarterly,'LineWidth',2,'Color',qual_yellow)
%plot(2400*FPUC_mult(2:end),mpc_marginal_by_size_stimcheck_quarterly,'--','LineWidth',2,'Color',qual_yellow)
%title('MPC in 1st Quarter')
ylabel('Quarterly MPC')
xlabel('Stimulus check or severance size')
legend('Total effect of severance','Effect of last $50 of severance','Total effect of stimulus check', 'FontSize', 13)
legend('BoxOff')
ylim([0 0.9])
xticks([0, 500, 1000, 1500, 2000, 2500])
xticklabels({'$0','$500','$1000','$1500','$2000','$2500'})
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
grid on
fig_paper_13 = gcf;
x = [0.2 0.7];
y = [0.363 0.363];
a = annotation('doublearrow', x, y);
a.Color = "#228B22"
a.Head1Style = "vback3"
a.Head2Style = "vback3"
a.Head1Length = 6
a.Head2Length = 6
a.Head1Width = 6
a.Head2Width = 6
a.LineWidth = 0.7
dim = [0.30 0.40 0.075 0.075];
str = {'MPC from $2000 of','severance ≈ MPC', 'from $1 of stimulus'};
t=annotation('textbox',dim,'String',str,'FitBoxToText','on');
t.Color = "#228B22"
t.FontSize = 8;
t.EdgeColor = 'none'
saveas(fig_paper_13, fullfile(release_path_paper, 'UI_vs_stimulus_check_w_arrows.png'))
%saveas(fig_paper_13, fullfile(release_path_slides, 'UI_vs_stimulus_check_w_arrows.png'))

%figure
%hold on
%s2=2400*FPUC_mult(2:end)'.*squeeze(mpc_supplements_by_size(2,3,2:end,1))./(2400*FPUC_mult(2:end).*mpc_marginal_by_size_stimcheck_quarterly_onetenth)';
%s2(1:end)=smooth(s2(1:end));
%plot(2400*FPUC_mult(2:end),smooth(2400*FPUC_mult(2:end)'.*squeeze(mpc_supplements_by_size(2,1,2:end,1))./(2400*FPUC_mult(2:end).*mpc_by_size_stimcheck_quarterly_onetenth)'),'LineWidth',2,'Color',qual_purple)
%plot(2400*FPUC_mult(2:end),s2,'--','LineWidth',2,'Color',qual_purple)
%title( {'Quarterly Agg Spending Effect','1 Month UI vs. Equal Cost Stimulus Check'})
%title( {'Aggregate Effect:' 'UI Supplement vs. Equal Cost Stimulus Check'})
%ylabel('UI effect relative to equal cost stimulus check')
%xlabel('UI supplement size (per month)')
%ylim([0.8 2.9])
%legend('Total relative spending effect','Relative spending effect from last $50', 'FontSize', 13)
%legend('BoxOff')
%set(gca,'fontsize', 12);
%set(gca, 'Layer', 'top');
%grid on
%fig_paper_15 = gcf;
%saveas(fig_paper_15, fullfile(release_path_paper, 'UI_vs_equalcost_stimulus_check.png'))



total_cost=FPUC_length'*FPUC_mult*2400;
total_spend=FPUC_length'*FPUC_mult*2400.*squeeze(mpc_supplements_by_size(3,1,:,:))';
duration_increase_total=duration_increase_total';

pol_length=zeros(3,13);
pol_length(1,:)=2;
pol_length(2,:)=3;
pol_length(3,:)=4;

pol_size=zeros(3,13);
pol_size(1,:)=FPUC_mult*2400;
pol_size(2,:)=FPUC_mult*2400;
pol_size(3,:)=FPUC_mult*2400;

total_cost_vec=reshape(total_cost,3*13,1);
total_spend_vec=reshape(total_spend,3*13,1);
pol_length_vec=reshape(pol_length,3*13,1);
duration_increase_total_tmp=[NaN;NaN;NaN;];
duration_increase_total_tmp=[duration_increase_total_tmp duration_increase_total]
duration_increase_total_vec=reshape(duration_increase_total_tmp,3*13,1);

data=table();
data.total_cost=total_cost_vec;
data.total_spend=total_spend_vec;
data.pol_length_vec=pol_length_vec;
data.duration_increase=duration_increase_total_vec;

figure
hold on
index=1;
for i=1:length(total_cost_vec)
    idx=data.total_cost==total_cost_vec(i);
    data_tmp=data(idx,:);
    data_tmp=sortrows(data_tmp,3);
    data_cell=table2array(data_tmp);
    [a b]=size(data_cell);
    if a>=2
        plot(data_cell(:,3),data_cell(:,2))
        ylabel('Total Spending Over 6 Months')
        xlabel('Policy Duration')
        costholder(index)=total_cost_vec(i);
        index=index+1;
    end
end
legend('Total Cost=2400','Total Cost=4800')

figure
hold on
index=1;
for i=1:length(total_cost_vec)
    idx=data.total_cost==total_cost_vec(i);
    data_tmp=data(idx,:);
    data_tmp=sortrows(data_tmp,3);
    data_cell=table2array(data_tmp);
    [a b]=size(data_cell);
    if a>=2
        plot(data_cell(:,3),data_cell(:,4))
        ylabel('Unemployment Duration Increase')
        xlabel('Policy Duration')
        costholder(index)=total_cost_vec(i);
        index=index+1;
    end
end
legend('Total Cost=2400','Total Cost=4800')


duration_increase_marginal_pandemic=duration_increase_marginal;
mpc_marginal_pandemic=squeeze(mpc_supplements_by_size(1,3,:,:));
    
duration_increase_marginal_pandemic_surprise=duration_increase_marginal_surprise;
mpc_marginal_pandemic_surprise=squeeze(mpc_supplements_by_size(1,4,:,:));
        
 
    
else
  

duration_increase_total=squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1,:))';
duration_increase_marginal=squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:));
duration_growth_total=(squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1,:))')./squeeze(elasticity_and_distortions_values_by_size(8,1,:))';
duration_growth_marginal=(squeeze(elasticity_and_distortions_values_by_size(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:)))./squeeze(elasticity_and_distortions_values_by_size(8,1:end-1,:));

duration_increase_total_surprise=squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))';
duration_increase_marginal_surprise=squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:));
duration_growth_total_surprise=(squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))')./squeeze(elasticity_and_distortions_values_by_size_surprise(8,1,:))';
duration_growth_marginal_surprise=(squeeze(elasticity_and_distortions_values_by_size_surprise(8,2:end,:))-squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:)))./squeeze(elasticity_and_distortions_values_by_size_surprise(8,1:end-1,:));

 

%figure
%tiledlayout(1,2)
%nexttile
%hold on
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(1,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(1,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%title('One month MPCs')
%ylabel('MPC')
%xlabel('UI Supplement Size (per month)')
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(2400*FPUC_mult(2:end),duration_increase_total_surprise(:,3),'LineWidth',2,'Color',qual_blue)
%plot(2400*FPUC_mult(2:end),duration_increase_marginal_surprise(:,3),'--','LineWidth',2,'Color',qual_blue)
%title('Change in U Duration')
%ylabel('Change (in Months)')
%xlabel('UI Supplement Size (per month)')
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%set(gca, 'Layer', 'top');
%lg  = legend('Total Effect of Supplement','Effect from last $ of Supplement');
%lg.Layout.Tile = 'South';
%grid on
%fig=gcf;
%set(gcf, 'PaperPosition', [0 0 10 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
%set(gcf, 'PaperSize', [10 5]); %Keep the same paper size
%saveas(fig, fullfile(release_path_paper, 'U_and_C_total_and_marginal_4month_surprise.png'))



%figure
%tiledlayout(1,3)
%nexttile
%hold on
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(1,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(1,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('One month MPCs')
%ylabel('MPC')
%xlabel('UI Supplement Size (per month)')
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(2,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(2,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('Three month MPCs')
%ylabel('MPC')
%xlabel('UI Supplement Size (per month)')
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%grid on
%nexttile
%hold on
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(3,2,:,3)),'LineWidth',2,'Color',qual_blue)
%plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(3,4,:,3)),'--','LineWidth',2,'Color',qual_blue)
%ylim([0.1,0.75])
%title('Six month MPCs')
%ylabel('MPC')
%xlabel('UI Supplement Size (per month)')
%legend('From Full Supplement','From last $100 of supplement')
%set(gca,'fontsize', 12);
%set(gca, 'Layer', 'top');
%lg  = legend('Total Effect of Supplement','Effect from last $100 of Supplement');
%lg.Layout.Tile = 'South';
%grid on
%set(gcf, 'PaperPosition', [0 0 10 5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
%set(gcf, 'PaperSize', [10 5]); %Keep the same paper size
%fig=gcf;
%saveas(fig, fullfile(release_path_paper, 'U_and_C_total_and_marginal_diff_mpc_horizons_surprise.png'))



load stimulus_check_size_results
load stimulus_check_size_results_onetenth


figure
hold on
s=squeeze(mpc_supplements_by_size(2,3,:,1));
s(6:end)=smooth(s(6:end));
plot(2400*FPUC_mult,squeeze(mpc_supplements_by_size(2,1,:,1)),'LineWidth',2,'Color',qual_blue)
plot(2400*FPUC_mult,s,'--','LineWidth',2,'Color',qual_blue)
plot(2400*FPUC_mult(2:end),mpc_by_size_stimcheck_quarterly,'LineWidth',2,'Color',qual_yellow)
plot(2400*FPUC_mult(2:end),mpc_marginal_by_size_stimcheck_quarterly,'--','LineWidth',2,'Color',qual_yellow)
title('MPC in 1st Quarter')
ylabel('MPC')
xlabel('Stimulus or UI Supplement Size (per month)')
legend('Total Effect of UI Supplement','Effect of Last $ of UI Supplement','Total Effect of Stimulus Check','Effect of Last $ of Stimulus Check')
legend('BoxOff')
set(gca,'fontsize', 12);
set(gca, 'Layer', 'top');
grid on
fig=gcf;
%saveas(fig, fullfile(release_path_paper, 'UI_vs_stimulus_check.png'))

%figure
%hold on
%s2=2400*FPUC_mult(2:end)'.*squeeze(mpc_supplements_by_size(2,3,2:end,1))./(2400*FPUC_mult(2:end).*mpc_marginal_by_size_stimcheck_quarterly_onetenth)';
%s2(6:end)=smooth(s2(6:end));
%plot(2400*FPUC_mult(2:end),2400*FPUC_mult(2:end)'.*squeeze(mpc_supplements_by_size(2,1,2:end,1))./(2400*FPUC_mult(2:end).*mpc_by_size_stimcheck_quarterly_onetenth)','LineWidth',2,'Color',qual_purple)
%plot(2400*FPUC_mult(2:end),s2,'--','LineWidth',2,'Color',qual_purple)
%title( {'Quarterly Agg Spending Effect','1 Month UI vs. Equal Cost Stimulus Check'})
%title( {'Aggregate Effect:' 'UI Supplement vs. Equal Cost Stimulus Check'})
%ylabel('UI Effect relative to Stimulus Check')
%xlabel('UI Supplement Size (per month)')
%ylim([1 2])
%legend('Total Relative Spending Effect','Relative Spending Effect from last $')
%legend('BoxOff')
%set(gca,'fontsize', 12);
%grid on
%fig=gcf;
%saveas(fig, fullfile(release_path_paper, 'UI_vs_equalcost_stimulus_check.png'))







total_cost=FPUC_length'*FPUC_mult*2400;
total_spend=FPUC_length'*FPUC_mult*2400.*squeeze(mpc_supplements_by_size(3,1,:,:))';
duration_increase_total=duration_increase_total';

pol_length=zeros(3,13);
pol_length(1,:)=2;
pol_length(2,:)=3;
pol_length(3,:)=4;

pol_size=zeros(3,13);
pol_size(1,:)=FPUC_mult*2400;
pol_size(2,:)=FPUC_mult*2400;
pol_size(3,:)=FPUC_mult*2400;

total_cost_vec=reshape(total_cost,3*13,1);
total_spend_vec=reshape(total_spend,3*13,1);
pol_length_vec=reshape(pol_length,3*13,1);
duration_increase_total_tmp=[NaN;NaN;NaN;];
duration_increase_total_tmp=[duration_increase_total_tmp duration_increase_total]
duration_increase_total_vec=reshape(duration_increase_total_tmp,3*13,1);

data=table();
data.total_cost=total_cost_vec;
data.total_spend=total_spend_vec;
data.pol_length_vec=pol_length_vec;
data.duration_increase=duration_increase_total_vec;

figure
hold on
index=1;
for i=1:length(total_cost_vec)
    idx=data.total_cost==total_cost_vec(i);
    data_tmp=data(idx,:);
    data_tmp=sortrows(data_tmp,3);
    data_cell=table2array(data_tmp);
    [a b]=size(data_cell);
    if a>=2
        plot(data_cell(:,3),data_cell(:,2))
        ylabel('Total Spending Over 6 Months')
        xlabel('Policy Duration')
        costholder(index)=total_cost_vec(i);
        index=index+1;
    end
end
legend('Total Cost=2400','Total Cost=4800')

figure
hold on
index=1;
for i=1:length(total_cost_vec)
    idx=data.total_cost==total_cost_vec(i);
    data_tmp=data(idx,:);
    data_tmp=sortrows(data_tmp,3);
    data_cell=table2array(data_tmp);
    [a b]=size(data_cell);
    if a>=2
        plot(data_cell(:,3),data_cell(:,4))
        ylabel('Unemployment Duration Increase')
        xlabel('Policy Duration')
        costholder(index)=total_cost_vec(i);
        index=index+1;
    end
end
legend('Total Cost=2400','Total Cost=4800')


duration_increase_marginal_pandemic=duration_increase_marginal;
mpc_marginal_pandemic=squeeze(mpc_supplements_by_size(1,3,:,:));
    
duration_increase_marginal_pandemic_surprise=duration_increase_marginal_surprise;
mpc_marginal_pandemic_surprise=squeeze(mpc_supplements_by_size(1,4,:,:));
        

end



end





