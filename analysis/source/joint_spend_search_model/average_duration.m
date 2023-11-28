%This function takes in a time-series of the unemployment exit rate and computes average
%duration of unemployment

function average_duration = average_duration(exits)
    %Assume that the exit rate is constant for a long number of periods
    %after the end of whatever series is passed to the function (only
    %relevant when the series that is passed to the function is short)
    exits=[exits; exits(end)*ones(10000,1)];
    %calculate the survival function
    cp = (cumprod(1-exits));
    %Expected duration is the length of unemployment times the survival
    %function (noting that duration 1 has a probability of 100% of being
    %reached, so everything shifted by 1)
    average_duration = sum(exits .* [1; cp(1:end-1)] .* [1:length(exits)]');
end
