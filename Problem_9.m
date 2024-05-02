clear all

warning('off', 'all');

dataNamesPeriphrastic = {'Year', 'Mean annual temperature', ...
        'Mean annual maximum temperature', 'Mean annual minimum temperature', ...
        'Total annual rainfall or snowfall', 'Mean annual wind velocity', ...
        'Number of days with rain', 'Number of days with snow', ...
        'Number of days with wind', 'Number of days with fog', ...
        'Number of days with tornado', 'Number of days with hail'};

target_variable_names = {'Number of days with hail', 'Number of days with fog'};
target_variable_shorts = {'GR', 'FG'};

table = readtable("Heathrow.xlsx");
data = table2array(table);
alpha = 0.05;

adjR2 = @(R2, k, n) (n*R2 - R2 - k) / (n - k - 1 ); 

% The analysis will be on the data after 1973
idx_1973 = find(data(:, 1) == 1973);
data = data(idx_1973:end, :);

target_variable_FG = data(:, end - 2);
features_FG = data(:, 2:end);
features_FG(:, end-2) = [];

target_variable_GR = data(:, end);
features_GR = data(:, 2:end-1);

features = {features_GR, features_FG};
targets = {target_variable_GR, target_variable_FG};
figure_counter = 1;
ADJR2 = nan(2, 3);

for i=1:2
    j = 1;
    X = features{i};
    Y = targets{i};

    % To keep only the rows without any nan values
    no_nan_idxs = ~any(isnan(X), 2);
    X = X(no_nan_idxs, :);
    Y = Y(no_nan_idxs);
    n = length(Y);
    
    %% Linear regression using full feature vector
    [b, ~, ~, ~, stats] = regress(Y, [ones(n, 1) X]);

    Y_pred = [ones(n, 1) X] * b;

    e = Y - Y_pred;
    s_e = sqrt(1/(n-2) * sum((Y - Y_pred).^2));

    e_norm = e / s_e;
    
    ADJR2(i, j) = adjR2(stats(1), length(b) - 1, n);
    

    figure(figure_counter)
    figure_counter = figure_counter + 1;
    scatter(Y, e_norm, 25, "blue");
    title(sprintf("Diagnostic plot for %s\n Linear: Using %d features", target_variable_names{i}, length(b)))
    xlabel(target_variable_shorts{i})
    ylabel('e^*')
    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.1*(ax(4)-ax(3)),['adjR^2=',...
        num2str(ADJR2(i, j),6)])
    j = j + 1;
    zcrit = norminv(1-alpha/2);
    yline(zcrit)
    yline(-zcrit)

    

    %% Linear Stepwise Regression
    [b_step, ~, ~,  step_model_vector, stats] = stepwisefit(X, Y, "display","off");

    X_selected = X(:, step_model_vector);
    b_step_selected = b_step(step_model_vector);
    b0 = stats.intercept;
    
    Y_step = b0 + X_selected * b_step_selected;

    e = Y - Y_step;
    s_e = sqrt(1/(n-2) * sum((Y - Y_step).^2));

    e_norm_step = e / s_e;
    k = length(b_step_selected);

    ADJR2(i, j) = 1-((n-1)/(n-k-1))*(sum((Y-Y_step).^2))/(sum((Y-mean(Y)).^2));
    

    figure(figure_counter)
    figure_counter = figure_counter + 1;
    scatter(Y, e_norm_step, 25, "blue");
    title(sprintf("Diagnostic plot for %s\n Stepwise: Using %d features", target_variable_names{i}, length(b_step_selected)))
    xlabel(target_variable_shorts{i})
    ylabel('e^*')
    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.1*(ax(4)-ax(3)),['adjR^2=',...
        num2str(ADJR2(i, j),6)])
    j = j + 1;
    zcrit = norminv(1-alpha/2);
    yline(zcrit)
    yline(-zcrit)

    %% PCA Dimensionality Reduction
    X_ = X - mean(X);

    covX_ = cov(X_);
    [eig_vec,eig_val] = eig(covX_);
    
    eig_val_diag = diag(eig_val); % Extract the diagonal elements
    % Order in descending order
    [eig_val_diag, idx] = sort(eig_val_diag, 'descend');
    eig_vec = eig_vec(:,idx);

    pervarV = 100 * cumsum(eig_val_diag) / sum(eig_val_diag);
    d = find(pervarV > 99, 1); % New dimensionality for > 99% explained variance
    X_pca = X_ * eig_vec(:, 1:d);
    
    % Perform Linear Regression now

    [b, ~, ~, ~, stats] = regress(Y, [ones(n, 1) X_pca]);

    Y_pca = [ones(n, 1) X_pca] * b;

    e = Y - Y_pca;
    s_e = sqrt(1/(n-2) * sum((Y - Y_pca).^2));

    e_norm = e / s_e;
    
    ADJR2(i, j) = adjR2(stats(1), length(b) - 1, n);
    

    figure(figure_counter)
    figure_counter = figure_counter + 1;
    scatter(Y, e_norm, 25, "blue");
    title(sprintf("Diagnostic plot for %s\n PCA: Using %d features", target_variable_names{i}, length(b)))
    xlabel(target_variable_shorts{i})
    ylabel('e^*')
    ax = axis;
    text(ax(1)+0.3*(ax(2)-ax(1)),ax(3)+0.1*(ax(4)-ax(3)),['adjR^2=',...
        num2str(ADJR2(i, j),6)])
    j = j + 1;
    zcrit = norminv(1-alpha/2);
    yline(zcrit)
    yline(-zcrit)

    Y_ = [Y, Y_pred, Y_step, Y_pca];

end