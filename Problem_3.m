clear all


table = readtable("Heathrow.xlsx");
data = table2array(table);

years = data(:, 1);
features = data(:, 2:end);
p = nan(9, 2);

for i = 1:9

    vector = [years features(:, i)];
    cont_break_point_idx = nan;
    for j=1:length(years)
        if years(j) + 1 ~= years(j+1)
            cont_break_point_idx = j + 1;
            break;
        end
    end

    if isnan(cont_break_point_idx)
        disp("Error, no breaking point found");
        break;
    end
    
    X1 = vector(1:cont_break_point_idx - 1, 2);
    X2 = vector(cont_break_point_idx:end, 2);
    
    n = length(X1);
    m = length(X2);
    diffmean = mean(rmmissing(X1))- mean(rmmissing(X2));

    % Parametric hypothesis test for difference of means

    pooled_var = ((n-1)*std(rmmissing(X1))^2 + (m-1)*std(rmmissing(X2))^2) / (n + m -2);
    pooled_std = sqrt(pooled_var);
    tsample = diffmean / (pooled_std * sqrt(1/n + 1/m));
    p(i, 1) = 2*(1-tcdf(abs(tsample), n+m-2));

    % Bootstrap hypothesis test for difference of means
    
    B = 1000;
    boot_diff_means = NaN(B,1);
    pooled_sample = [X1; X2];
    for j=1:B
        boot_idxs = unidrnd(n+m,n+m,1);
        boot_sample = pooled_sample(boot_idxs);
        X_boot = boot_sample(1:n);
        Y_boot = boot_sample(n+1:end);
        boot_diff_means(j) = mean(rmmissing(X_boot)) - mean(rmmissing(Y_boot));
    end
    
    boot_combined = [diffmean;boot_diff_means];
    [~,idx] = sort(boot_combined);
    boot_rank = find(idx == 1);
    

    if boot_rank > 0.5*(B+1)
        p(i, 2) = 2*(1-boot_rank/(B+1));
    else
        p(i, 2) = 2*boot_rank/(B+1);
    end
end

disp(p)
fprintf("The null hypothesis for difference of means between the two periods \n is not rejected for " + ...
    "the features 3 and 7");