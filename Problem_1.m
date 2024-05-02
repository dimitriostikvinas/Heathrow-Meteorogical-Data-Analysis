clear all


table = readtable("Heathrow.xlsx");
data = table2array(table);

for i=2:12
    feature = data(:, i);
    feature = rmmissing(feature);
    
    disc_val = unique(feature);
    disc_val_count = length(disc_val);

    if disc_val_count > 10
        [~, p1] = chi2gof(feature); % Normal Distribution
        [~, p2] = chi2gof(feature ,'cdf',{@unifcdf, min(feature), max(feature)}); % Continuous Uniform Distribution
        
        figure(i)
        histogram(feature);
        xlabel('Values')
        ylabel('Frequencies')
         title({'Histogram of values in sample';[ 'p-value for normal dist.: ', ...
            num2str(p1)];...
            ['p-value for uniform dist.: ', num2str(p2)]});
    else
        propability_of_success = mean(feature) / 365;
        number_of_trials = 365;
        [~, p1] = chi2gof(feature, 'cdf', ...
            {@binocdf,number_of_trials, propability_of_success}); % Binomial Distribution
        [~, p2] = chi2gof(feature,'cdf',{@unidcdf, max(feature)}); % Discrete Uniform Distribution
        figure(i);
        X = categorical(feature, disc_val, cellstr(num2str(unique(disc_val, 'sorted'))));
        histogram(X, 'BarWidth', 0.5);
        title({'Bar graph of values in sample'; ['p-value for binomial dist.: ', ...
            num2str(p1)];...
            ['p-value for discrete uniform dist.: ', num2str(p2)]});
        xlabel('Values');
        ylabel('Frequency')
    end
end