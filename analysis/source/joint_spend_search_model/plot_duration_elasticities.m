% Plot comparing duration elasticities from our data/model to literature
% There are four versions ('iter' runs from 1 to 4):
%   (1) all estimates, normal size
%   (2) only literature estimates, normal size
%   (3) all estimates, smaller size
%   (4) only literature estimates, smaller size

% Previous versions of this plot have been compiled in R:
% https://github.com/ganong-noel/uieip/blob/bd62d7625fc818f03ccec61f8065612ebf06cdc5/analysis/source/animate_key_social_plots.R#L16-L82
% https://github.com/ganong-noel/uieip/commit/2573b7069bc23fac99ae235d653da82e758bc31f

load literature_elasticities_input_directory.mat
load elasticity_uieip_input_directory.mat
load release_paths.mat

% Load data (literature and UIEIP estimates separately) -------------------
data_literature = readtable(literature_elasticities_input_directory);
data_uieip = readtable(elasticity_uieip_input_directory);

Xliterature = data_literature.elasticity;
Xuieip = [data_uieip.supplement600(2:3); data_uieip.supplement300(2:3)];

% Loop: create figure including the UIEIP estimates & version with
% literature estimates only:
for iter = 1:4
    if any(iter == [1, 3]) %first/third iteration: UIEIP included
        % Sort (including an indicator for values from UIEIP project) -------------
        joint = sortrows([Xliterature zeros(length(Xliterature), 1); Xuieip ones(length(Xuieip), 1)], 1);
        X = joint(:,1); %sorted elasticity estimates
        flag_uieip = (joint(:,2) == 1); %flag for UIEIP estimates
    elseif any(iter == [2, 4]) %second/fourth iteration: only literature
        X = sortrows(Xliterature);
        flag_uieip = zeros(length(Xliterature)); %vector of zeros (to not break rest of the code)
    end
    n = length(X);
    
    % Set Y dimension & stack if overlapping ----------------------------------
    xdist = .035; %stack if estimates are within `xdist` of each other
    ydist = .053; %such that stacked values are directly next to each other
    X(X>1) = 1.1; %collect large estimates in one category
    Y = zeros(n,1) + ydist / 2;
    overlap = zeros(n,1);
    for i = 2:n-1 %stacking
        if (X(i-1) + xdist / 2 > X(i) - xdist / 2)
            Y(i) = Y(i-1) + ydist;
            X(i-1) = (X(i-1) + X(i))/2; %stack at the mean of the two values
    %         (simply comment this out to stack at lower of the two values)
    %         this is helpful to avoid overlap of stacks
            X(i) = X(i-1);
            if (X(i) + xdist / 2 > X(i+1) - xdist / 2) %for highest value
                Y(i+1) = Y(i) + ydist;
                X(i+1) = X(i-1);
            end
        end
    end
    
    % Sort again such that stacked estimates are grouped ----------------------
    joint2 = sortrows([X flag_uieip], [1 2], {'ascend', 'descend'});
    X = joint2(:,1);
    flag_uieip = joint2(:,2)==1;
    
    % Plot --------------------------------------------------------------------
    figure();
    grid on;
    hold on;
    scatter(X(~flag_uieip), Y(~flag_uieip), 110, 'filled', 'o', 'MarkerFaceColor', '#bfbfbf'); %literature estimates
    scatter(X(flag_uieip), Y(flag_uieip), 110 , 'filled', 'o', 'MarkerFaceColor', '#004577'); %UIEIP estimates
    hold off;
    
    if any(iter == [1, 2])
        ylim([-.05 1.15]);
        yticks([0:1/6:1.15])
    elseif any(iter == [3, 4])
        ylim([-.05 .515]);
        yticks([0:1/6:.515])
    end

    yticklabels([]); %no labels
    ylabel("Number of studies")

    xticklabels({'0', '.25', '.5', '.75', '1', '>1'})
    xlabel(["Duration elasticity"; "(percent change in duration unemployed / percent change in benefit level)"])
    xlim([-.05 1.15]);
    xticks([0:.25:1 1.1])
        
    set(gcf, 'color', 'w'); %white background
    set(gca, 'TickLength', [0 0]) %no ticks
    
    if iter == 1
        legend('Pre-pandemic studies', 'Effect of supplements in pandemic', 'Location', 'west')
    elseif iter == 2
        legend('Pre-pandemic studies', 'Location', 'west')
    elseif iter == 3
        legend('Pre-pandemic studies', 'Effect of supplements in pandemic', 'Location', 'east')
    elseif iter == 4
        legend('Pre-pandemic studies', 'Location', 'east')
    end
    legend('boxoff')
    
    if any(iter == [1, 2])
        text(.06, 5/6, "\leftarrow Smaller disincentive")
        text(.57, 5/6, "Larger disincentive \rightarrow")
    elseif any(iter == [3, 4])
        text(.06, .45, "\leftarrow Smaller disincentive")
        text(.57, .45, "Larger disincentive \rightarrow")
    end
    
    fig = gcf;

    if iter == 1
        fig_paper_8 = fig;
        saveas(fig_paper_8, fullfile(release_path_paper, 'elasticity_comparison.png')) %save for paper without resize
    end

    if any(iter == [1, 2])
        set(fig, 'Position', fig.Position .* [1 1.1 1.1 1]) %resize
    elseif any(iter == [3, 4])
        set(fig, 'Position', fig.Position .* [.5 1.1 1.1 .5]) %resize
    end

    if iter == 1
        saveas(fig, fullfile(release_path_slides, 'elasticity_comparison.png'))
   elseif iter == 2
        saveas(fig, fullfile(release_path_slides, 'elasticity_comparison_onlyliterature.png'))
    %elseif iter == 3
        %saveas(fig, fullfile(release_path_slides, 'elasticity_comparison_wide.png'))
    %elseif iter == 4
        %saveas(fig, fullfile(release_path_slides, 'elasticity_comparison_onlyliterature_wide.png'))
    end
end