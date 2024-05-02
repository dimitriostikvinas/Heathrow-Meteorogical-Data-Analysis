clear all

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
features(:, end-2) = [];

dataNamesPeriphrastic(1) = [];
dataNamesPeriphrastic(end-2) = [];

% adjR2 = @(n, k, Y, Y_hat) 1-((n-1)/(n-k-1))*(sum((Y-Y_hat).^2))/(sum((Y-mean(Y)).^2));
adjR2 = @(R2, k, n) (n*R2 - R2 - k) / (n - k - 1 ); 

ADJR2 = nan(size(features, 2), 4);
dotSize = 25;

for i= 1:size(features, 2)
    figure(i);

    X = features(:, i);
    Y = target_variable_FG;

    X_nan_idxs = find(isnan(X));
    Y_nan_idxs = find(isnan(Y));

    X(unique([X_nan_idxs; Y_nan_idxs])) = [];
    Y(unique([X_nan_idxs; Y_nan_idxs])) = [];
    n = length(X);


    % Polynomial 1st degree
    X_1 = [ones(n, 1) X];

    [b_1, ~, ~, ~, stats] = regress(Y, X_1);

    Y_pred = X_1 * b_1;

    k = length(b_1) - 1;

    ADJR2(i, 1) = adjR2(stats(1), k, n);

    subplot(4, 2, 1);
    scatter(X, Y, dotSize, 'filled');
    title({'Polynomial 1st Degree'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    hold on;
    X_grid = min(X):0.01:max(X);
    Y_grid = [ones(length(X_grid), 1) X_grid'] * b_1;
    plot(X_grid, Y_grid, 'Color', 'r')
    
    error = Y - Y_pred;
    errorNorm = error / std(error);

    subplot(4, 2, 2);
    scatter(Y, errorNorm, dotSize, 'filled');
    title({'Polynomial 1st Degree - Diagnostic Plot'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    zcrit = norminv(1-alpha/2);

    yline(zcrit)
    yline(-zcrit)

    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.2*(ax(4)-ax(3)),['adjR^2=',...
    num2str(ADJR2(i, 1),3)])

    % Polynomial 2nd degree
    X_2 = [ones(n, 1) X X.^2];

    [b_2, ~, ~, ~, stats] = regress(Y, X_2);

    Y_pred = X_2 * b_2;

    k = length(b_2) - 1;

    ADJR2(i, 2) = adjR2(stats(1), k, n);

    subplot(4, 2, 3);
    scatter(X, Y, dotSize, 'filled');
    title({'Polynomial 2nd Degree'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    hold on;
    X_grid = min(X):0.01:max(X);
    Y_grid = [ones(length(X_grid), 1) X_grid' X_grid.^2'] * b_2;
    plot(X_grid, Y_grid, 'Color', 'r')
    
    error = Y - Y_pred;
    errorNorm = error / std(error);

    subplot(4, 2, 4);
    scatter(Y, errorNorm, dotSize, 'filled');
    title({'Polynomial 1st Degree - Diagnostic Plot'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    zcrit = norminv(1-alpha/2);

    yline(zcrit)
    yline(-zcrit)

    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.2*(ax(4)-ax(3)),['adjR^2=',...
    num2str(ADJR2(i, 2),3)])

    % Polynomial 3rd degree
    X_3 = [ones(n, 1) X X.^2 X.^3];

    [b_3, ~, ~, ~, stats] = regress(Y, X_3);

    Y_pred = X_3 * b_3;

    k = length(b_3) - 1;

    ADJR2(i, 3) = adjR2(stats(1), k, n);

    subplot(4, 2, 5);
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

    subplot(4, 2, 6);
    scatter(Y, errorNorm, dotSize, 'filled');
    title({'Polynomial 3rd Degree - Diagnostic Plot'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    zcrit = norminv(1-alpha/2);

    yline(zcrit)
    yline(-zcrit)

    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.2*(ax(4)-ax(3)),['adjR^2=',...
    num2str(ADJR2(i, 3),3)])

    % Exponential Model : Y = a*e^(b*X) => ln(Y) = ln(a) + b*X => Y_acute =
    % b0 + b1*X, Y_acute = ln(Y), b0 = ln(a), b1 = b

    Y_acute = log(Y);
    X_4 = [ones(n, 1) X];

    [b_4, ~, ~, ~, stats] = regress(Y_acute, X_4);

    Y_pred = X_4 * b_4;
    
    k = length(b_4) - 1;

    ADJR2(i, 4) = adjR2(stats(1), k, n);

    subplot(4, 2, 7);
    scatter(X, Y, dotSize, 'filled');
    title({'Exponential'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    hold on;
    X_grid = min(X):0.01:max(X);
    Y_grid = [ones(length(X_grid), 1) X_grid'] * b_4;
    plot(X_grid, exp(Y_grid), 'Color', 'r')

    error = Y - exp(Y_pred);
    errorNorm = error / std(error);

    subplot(4, 2, 8);
    scatter(Y, errorNorm, dotSize, 'filled');
    title({'Exponential - Diagnostic Plot'});
    xlabel(dataNamesPeriphrastic(i));
    ylabel('FG');
    zcrit = norminv(1-alpha/2);

    yline(zcrit)
    yline(-zcrit)
    
    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.2*(ax(4)-ax(3)),['adjR^2=',...
    num2str(ADJR2(i, 4),3)])
    
    [max_adjR2, idx] = max(ADJR2(i, :));

    switch idx
        case 1
            string = "Y = " + b_1(1) + " + " + b_1(2) + "*X";
        case 2
            string = "Y = " + b_2(1) + " + " + b_2(2) + "*X + " + b_2(3) + "*X^2";
        case 3
            string = "Y = " + b_3(1) + " + " + b_3(2) + "*X + " + b_3(3) + "*X^2 + " + b_3(4) + "*X^3";
        case 4
            string = "Y = " + exp(b_4(1)) + "*e^(" + b_4(2) + "*X)";
    end
    fprintf("Best Model using the feature : %s :\n", dataNamesPeriphrastic{i});
    fprintf(string);
    fprintf("\nwith adjR2 = %f\n", max_adjR2);
    fprintf("---------------------------------\n");

end
