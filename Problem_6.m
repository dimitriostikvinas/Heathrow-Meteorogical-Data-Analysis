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

features = data(:, 2:end);
features(:, end-1) = [];

R2 = nan(size(features, 2), 1);

for i= 1:size(features, 2)
    X = features;
    Y = features(:, i);
    X(:, i) = [];
    
    % Linear regression model
    [B, ~, ~, ~, STATS] = regress(Y, [ones(length(X), 1) X]);
    
    R2(i) = STATS(1);

end

[~, idx_max_1] = max(R2);
temp_R2 = R2;
temp_R2(idx_max_1) = [];
[~, idx_max_2] = max(temp_R2);

name_1 = dataNamesPeriphrastic(idx_max_1 + 1);
name_2 = dataNamesPeriphrastic(idx_max_2 + 1);

fprintf("The two target variables that can be best explained using a\n" + ...
    "linear regression model based on the remaining features are:\n%s and %s\n", name_1{1}, name_2{1});