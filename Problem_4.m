clear all


table = readtable("Heathrow.xlsx");
data = table2array(table);
alpha = 0.05;

features = data(:, 2:end);

combinations = nchoosek(1:9, 2);
B = 1000;
L = 1000;
r = nan(L+1, 1);
[p, ci_param, ci_boot] = deal(nan(length(combinations), 2));
vector_lengths = nan(length(combinations), 1);

for i=1:length(combinations)
    vector1 = features(:, combinations(i, 1));
    vector2 = features(:, combinations(i, 2));

    vector1_nan_idxs = find(isnan(vector1));
    vector2_nan_idxs = find(isnan(vector2));

    vector1(unique([vector1_nan_idxs; vector2_nan_idxs])) = [];
    vector2(unique([vector1_nan_idxs; vector2_nan_idxs])) = [];

    X = [vector1 vector2];
    n = length(X);
    vector_lengths(i) = n;
    r_temp = corrcoef(X);
    r(1) = r_temp(1, 2);

    for j=1:L
        boot_idxs = unidrnd(n, n , 1);
        Xr = [X(:, 1) X(boot_idxs, 2)];
        r_temp = corrcoef(Xr);
        r(j+1) = r_temp(1, 2);
    end
    
    % Parametric for rho ci
    % Fisher transformation
    z = 0.5 * log((1 + r(1))./(1 - r(1)));

    zcrit = norminv(1-alpha/2);
    zsd = sqrt(1/(n-3));
    
    zl = z - zcrit * zsd;
    zu = z + zcrit * zsd;
    
    rl = (exp(2*zl) - 1) ./ (exp(2*zl) + 1);
    ru = (exp(2*zu) - 1) ./ (exp(2*zu) + 1);

    ci_param(i, :) = [rl, ru];
    
    % Bootstrap for rho ci
    B = 1000;
    ci_boot(i, :) = bootci(B, {@corr, X(:, 1), X(:, 2)}, 'Alpha', alpha);
    

    % Parametric null hypothesis test
    tsample = r(1) * sqrt((n-2)/(1-r(1)^2));
    p(i, 1) = min(2 * tcdf(tsample, n-2), 2*(1 - tcdf(tsample, n-2)));

    % Randomization null hypothesis test
    [~, idx] = sort(r);
    rand_rank = find(idx == 1); 
    if rand_rank > 0.5*(L+1)
        p(i,2) = 2*(1-rand_rank/(L+1));
    else
        p(i,2) = 2*rand_rank/(L+1);
    end  

end

[linear_dependence_pairs_param_ci, linear_dependence_pairs_boot_ci, ...
    linear_dependence_pairs_param_test, linear_dependence_pairs_rand_test] ...
    = deal([]);


for i=1:length(combinations)
    % Parametric ci
    if 0 < ci_param(i, 1) | 0 > ci_param(i, 2)
        linear_dependence_pairs_param_ci = [linear_dependence_pairs_param_ci; combinations(i, :)];
    end

    % Bootstrap ci
    if 0 < ci_boot(i, 1) | 0 > ci_boot(i, 2) %#ok<*OR2>
        linear_dependence_pairs_boot_ci = [linear_dependence_pairs_boot_ci; combinations(i, :)]; %#ok<*AGROW>
    end

    % Parametric test
    if p(i, 1) < alpha
        linear_dependence_pairs_param_test = [linear_dependence_pairs_param_test; combinations(i, :)];
    end

    % Randomization test
    if p(i, 2) < alpha
        linear_dependence_pairs_rand_test = [linear_dependence_pairs_rand_test; combinations(i, :)];
    end
end


common_rows = intersect(intersect(intersect(linear_dependence_pairs_param_ci, ... 
    linear_dependence_pairs_boot_ci, 'rows'), linear_dependence_pairs_param_test ...
    , 'rows'), linear_dependence_pairs_rand_test, 'rows');

fprintf("\n From the 4 different ways of finding whether two features have \n linear " + ...
    "correlation, the pairs upon which all ways of testing \n agreed where the following: \n")
disp(common_rows)

[most_important_combinations_param, most_important_combinations_rand] = deal(nan(3, 2));
[least_important_combinations_param, least_important_combinations_rand] = deal(nan(3, 2));

p_temp = p;
for i=1:3
    % Parametric
    idx =  find(p_temp(:, 1) == min(p_temp(:, 1)), 1);
    most_important_combinations_param(i, :) = combinations(idx, :);
    p_temp(idx, 1) = nan; 

    % Randomization
    idx =  find(p_temp(:, 2) == min(p_temp(:, 2)), 1);
    most_important_combinations_rand(i, :) = combinations(idx, :);
    p_temp(idx, 2) = nan; 
    
end

fprintf("\n Pairs with the most important correlation from the parametric testing where: \n")
disp(most_important_combinations_param)
fprintf("\n Pairs with the most important correlation from the randomization testing where: \n")
disp(most_important_combinations_rand)

for i=1:3
    % Parametric
    idx =  find(p_temp(:, 1) == max(p_temp(:, 1)), 1);
    least_important_combinations_param(i, :) = combinations(idx, :);
    p_temp(idx, 1) = nan; 

    % Randomization
    idx =  find(p_temp(:, 2) == max(p_temp(:, 2)), 1);
    least_important_combinations_rand(i, :) = combinations(idx, :);
    p_temp(idx, 2) = nan; 
    
end

fprintf("\n Pairs with the least important correlation from the parametric testing where: \n")
disp(least_important_combinations_param)
fprintf("\n Pairs with the least important correlation from the randomization testing where: \n")
disp(least_important_combinations_rand)
