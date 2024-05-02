clear all


table = readtable("Heathrow.xlsx");
data = table2array(table);

idx_1973 = find(data(:, 1) == 1973);
idx_1958 = find(data(:, 1) == 1958);
data_for_ci = data(idx_1973:end, 2:end);
data_for_test = data(1:idx_1958, 2:end);

alpha = 0.05;
[ci_param, ci_boot] = deal(nan(2, 9));

for i=1:9
    feature = data_for_ci(:, i);
    feature = rmmissing(feature);

    % Parametric ci for mean
    [~, ~, ci_param(:, i)] = ttest(feature, mean(feature), 'Alpha', alpha );

    % Bootstrap ci for mean
    B = 1000;
    ci_boot(:, i) = bootci(B, {@mean, feature}, 'type', 'per', 'Alpha', alpha);
end

mean_remained_same = zeros(1, 9);
for j=1:9
        mu = mean(rmmissing(data_for_test(:, j)));

        if mu < ci_boot(1, j) | mu > ci_boot(2, j) | isnan(mu)
            mean_remained_same(j) = 0;
        else
            mean_remained_same(j) = 1;
        end
end

fprintf("The mean of each feature for the two periods \n seems to remain the same for the" + ...
    " features 3 and 7")