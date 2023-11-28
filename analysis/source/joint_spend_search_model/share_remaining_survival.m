
function share_remaining_survival = share_remaining_survival(search)

    cp = (cumprod(1-search));
    share_remaining_survival = [1; cp(1:end-1)];

end
