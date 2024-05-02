clear all


table = readtable("Heathrow.xlsx");
data = table2array(table);
alpha = 0.05;

features = data(:, 2:end);
combinations = nchoosek(1:9, 2);
L = 1000;
r = nan(L+1, 1);
[p, I] = deal(nan(length(combinations), 1));

for i= 1:length(combinations)
    vector1 = features(:, combinations(i, 1));
    vector2 = features(:, combinations(i, 2));

    vector1_nan_idxs = find(isnan(vector1));
    vector2_nan_idxs = find(isnan(vector2));

    vector1(unique([vector1_nan_idxs; vector2_nan_idxs])) = [];
    vector2(unique([vector1_nan_idxs; vector2_nan_idxs])) = [];

    X = [vector1 vector2];
    n = length(X);
    r_temp = corrcoef(X);
    r(1) = r_temp(1, 2);
    for j=1:L
        Xr = [X(:, 1) X(randperm(n), 2)];
        r_temp = corrcoef(Xr);
        r(j+1) = r_temp(1, 2);
    end
    I(i) = Mutual_Information(vector1, vector2);

    [~, idx] = sort(r);
    rand_rank = find(idx == 1); 
    if rand_rank > 0.5*(L+1)
       p(i) = 2*(1-rand_rank/(L+1));
    else
       p(i) = 2*rand_rank/(L+1);
    end  
    figure(i)
    scatter(vector1, vector2)
    title(sprintf('Combination[%d, %d] with p=%f', combinations(i, 1), combinations(i, 2), p(i)))
end

% If p < alpha and I > 0, then non-linear correlation
% If p > alpha and I > 0, then linear correlation
for i=1:length(combinations)
    if p(i) > alpha & I(i) > 0
        fprintf("\n Combination of features [%d, %d] has non-linear correlation\n", combinations(i, 1), combinations(i, 2));
    elseif p(i) < alpha & I(i) > 0
        fprintf("\n Combination of features [%d, %d] has linear correlation\n", combinations(i, 1), combinations(i, 2));  
    else
        fprintf("\n Combination of features [%d, %d] has no correlation whatsoever\n", combinations(i, 1), combinations(i, 2));  
    end
end



function I = Mutual_Information(X, Y)
    
    % Modify the X vector
    m = median(X);
    for i=1:length(X)
        if X(i) < m
            X(i) = 0;
        else
            X(i) = 1;
        end
    end

    % Modify the X vector
    m = median(Y);
    for i=1:length(Y)
        if Y(i) < m
            Y(i) = 0;
        else
            Y(i) = 1;
        end
    end
    
    data = [X;Y];

    % PMF X
    t = tabulate(X);
    pmf_x  = [t(:, 1), t(:, 3) ./ 100]; 

    % PMF Y
    t = tabulate(Y);
    pmf_y  = [t(:, 1), t(:, 3) ./ 100]; 

    % PMF X,Y
    t = tabulate(data);
    pmf_xy  = [t(:, 1), t(:, 3) ./ 100]; 
    
    % Entropies
    [H_x, H_y, H_xy] = deal(0);

    % Entropy X
    for i=1:size(pmf_x, 1)
        H_x = H_x -pmf_x(i, 2)*log10(pmf_x(i, 2));
    end

    % Entropy X
    for i=1:size(pmf_y, 1)
        H_y = H_y -pmf_y(i, 2)*log10(pmf_y(i, 2));
    end

    % Entropy X
    for i=1:size(pmf_xy, 1)
        H_xy = H_xy -pmf_xy(i, 2)*log10(pmf_xy(i, 2));
    end
    
    
    % Mutual Information
    I = H_x + H_y - H_xy;

end
