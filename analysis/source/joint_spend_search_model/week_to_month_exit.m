
function week_to_month_exit = week_to_month_exit(weekly_search)
    % Just take the average of weekly rates and then do a geometric mean of that using a common number of weeks. I set the common number of weeks to 4.34 since that's the average number of weeks in a month 
    mean_weekly = mean(weekly_search);
    week_to_month_exit = 1 - (1 - mean_weekly)^4.34;
    %week_to_month_exit=1-prod(1-weekly_search);
end
  