clear all

warning('off', 'all');

dataNamesPeriphrastic = {'Year', 'Mean annual temperature', ...
        'Mean annual maximum temperature', 'Mean annual minimum temperature', ...
        'Total annual rainfall or snowfall', 'Mean annual wind velocity', ...
        'Number of days with rain', 'Number of days with snow', ...
        'Number of days with wind', 'Number of days with fog', ...
        'Number of days with tornado', 'Number of days with hail'};

table = readtable("Heathrow.xlsx");
data = table2array(table);
alpha = 0.05;

target_variable_FG = data(:, end - 2);
features = data(:, 2:end);
features(:, end-2:end-1) = [];

dataNamesPeriphrastic(1) = [];
dataNamesPeriphrastic(end-2:end-1) = [];

L = 1000;

% adjR2 = @(n, k, Y, Y_hat) 1-((n-1)/(n-k-1))*(sum((Y-Y_hat).^2))/(sum((Y-mean(Y)).^2));
adjR2 = @(R2, k, n) (n*R2 - R2 - k) / (n - k - 1 ); 

ADJR2 = nan(L+1, size(features, 2));
[p, p_empirical] = deal(nan(size(features, 2), 1));
dotSize = 25;
p_test = nan(L, size(features, 2));

for i= 1:size(features, 2)
    figure(i);

    X = features(:, i);
    Y = target_variable_FG;

    X_nan_idxs = find(isnan(X));
    Y_nan_idxs = find(isnan(Y));

    X(unique([X_nan_idxs; Y_nan_idxs])) = [];
    Y(unique([X_nan_idxs; Y_nan_idxs])) = [];
    n = length(X);

    % Polynomial 3rd degree
    X_3 = [ones(n, 1) X X.^2 X.^3];

    [b_3, ~, ~, ~, stats] = regress(Y, X_3);

    Y_pred = X_3 * b_3;

    k = length(b_3) - 1;

    ADJR2(1, i) = adjR2(stats(1), k, n);

    for j=1:L
        X_3_rand = X_3(randperm(n), :);
    
        [b_3_rand, ~, ~, ~, stats_rand] = regress(Y, X_3_rand);

        Y_pred_rand = X_3_rand * b_3_rand;
    
        k = length(b_3) - 1;
    
        ADJR2(j+1, i) = adjR2(stats_rand(1), k, n);
        p_test(j, i) = stats_rand(3);
    end

    p_empirical(i) = sum(p_test(:, i) <= stats(3)) / L;

    [~, idx] = sort(ADJR2(:, i));
    rand_rank = find(idx == 1); 
    if rand_rank > 0.5*(L+1)
       p(i) = 2*(1-rand_rank/(L+1));
    else
       p(i) = 2*rand_rank/(L+1);
    end  

    subplot(1, 2, 1);
    scatter(X, Y, dotSize, 'filled');
    title({'Polynomial 3rd Degree'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    hold on;
    X_grid = min(X):0.01:max(X);
    Y_grid = [ones(length(X_grid), 1) X_grid' X_grid.^2' X_grid.^3'] * b_3;
    plot(X_grid, Y_grid, 'Color', 'r')

    error = Y - Y_pred;
    errorNorm = error / std(error);

    subplot(1, 2, 2);
    scatter(Y, errorNorm, dotSize, 'filled');
    title({'Polynomial 3rd Degree - Diagnostic Plot'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    zcrit = norminv(1-alpha/2);

    yline(zcrit)
    yline(-zcrit)

    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.2*(ax(4)-ax(3)),['adjR^2=',...
    num2str(ADJR2(1, i),3)])
    
    fprintf("\n p-value for the feature: %s : %f\n", dataNamesPeriphrastic{i}, p(i));
    fprintf("\n p empirical value for the feature: %s : %f\n", dataNamesPeriphrastic{i}, p_empirical(i));
    fprintf("-------------------------------------\n");
end


