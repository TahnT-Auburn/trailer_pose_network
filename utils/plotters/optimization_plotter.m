clc
clear
close all

%% Load Data

opt_study = readtable('googlenet_optuna_studies_wide.csv');
opt_study = table2struct(opt_study);

% sort
[~,index] = sortrows([opt_study.params_lr].', 'descend');
opt_study = opt_study(index);

% populate heatmap
heat_map = NaN(6,2);
L = length(vertcat(opt_study.params_lr));
lr = vertcat(opt_study.params_lr);
loss_scale = vertcat(opt_study.params_LOSS_SCALE);
value = vertcat(opt_study.value);
iter = 0;
for i = 1:L/2
    for j = i+iter:i+iter+1
        if loss_scale(j) == 100
            heat_map(i,1) = value(j);
        elseif loss_scale(j) == 1000
            heat_map(i,2) = value(j);
        end
    end
    iter = iter + 1;
end
    
% plot heatmap
xvalues = {'1e2','1e3'};
yvalues = {'1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-7'};
h = heatmap(xvalues, yvalues, heat_map);
h.GridVisible = 'off';
h.Colormap = abyss;
h.XLabel = 'Loss Scale';
h.YLabel = 'Learning Rate';
colorbar.Label.String = 'Val RMSE';

% scatter plot
figure()
scatter(loss_scale, log10(lr), 200, value, "filled")
colorbar;
coloroder abyss
xlim([95,1005])
