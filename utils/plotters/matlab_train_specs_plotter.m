%% Matlab Plotter for COMP 6650 Project
%
%  Tahn Thawainin

clc
clear
close all

% Load training data
RESNET18_CSV = "C:\Users\pzt0029\Documents\Classes\COMP_6650_Deep_Learning\Project\csv_outputs\resnet18_train.csv";
RESNET34_CSV = "C:\Users\pzt0029\Documents\Classes\COMP_6650_Deep_Learning\Project\csv_outputs\resnet34_train.csv";
DENSENET121_CSV = "C:\Users\pzt0029\Documents\Classes\COMP_6650_Deep_Learning\Project\csv_outputs\densenet121_train.csv";
MOBILENETV2_CSV = "C:\Users\pzt0029\Documents\Classes\COMP_6650_Deep_Learning\Project\csv_outputs\mobilenetv2_train.csv";
INCEPTIONV3_CSV = "C:\Users\pzt0029\Documents\Classes\COMP_6650_Deep_Learning\Project\csv_outputs\inceptionv3_train.csv";

RESNET18_T = readtable(RESNET18_CSV);
RESNET34_T = readtable(RESNET34_CSV);
DENSENET121_T = readtable(DENSENET121_CSV);
MOBILENETV2_T = readtable(MOBILENETV2_CSV);
INCEPTIONV3_T = readtable(INCEPTIONV3_CSV);

RESNET18_S = table2struct(RESNET18_T);
RESNET34_S = table2struct(RESNET34_T);
DENSENET121_S = table2struct(DENSENET121_T);
MOBILENETV2_S = table2struct(MOBILENETV2_T);
INCEPTIONV3_S = table2struct(INCEPTIONV3_T);


% Loss plot
figure()
plot(vertcat(RESNET18_S.Var1)+1, vertcat(RESNET18_S.loss_history), '-o',DisplayName='ResNet18', LineWidth=1.5)
hold on
plot(vertcat(RESNET34_S.Var1)+1, vertcat(RESNET34_S.loss_history), '-o', DisplayName='ResNet34',LineWidth=1.5)
plot(vertcat(DENSENET121_S.Var1)+1, vertcat(DENSENET121_S.loss_history), '-o', DisplayName='DENSENET121',LineWidth=1.5)
plot(vertcat(MOBILENETV2_S.Var1)+1, vertcat(MOBILENETV2_S.loss_history), '-o', DisplayName='MOBILENETV2',LineWidth=1.5)
plot(vertcat(INCEPTIONV3_S.Var1)+1, vertcat(INCEPTIONV3_S.loss_history), '-o', DisplayName='INCEPTIONV3',LineWidth=1.5)
hold off
legend()
% yscale("log")
grid()
set(gcf,'color','w')
title('Loss')
xlabel('Epoch')

% Train and Val RMSE
figure()
plot(vertcat(RESNET18_S.Var1)+1, vertcat(RESNET18_S.rmse_train_history), '-o',DisplayName='ResNet18', LineWidth=1.5)
hold on
plot(vertcat(RESNET34_S.Var1)+1, vertcat(RESNET34_S.rmse_train_history), '-o', DisplayName='ResNet34',LineWidth=1.5)
plot(vertcat(DENSENET121_S.Var1)+1, vertcat(DENSENET121_S.rmse_train_history), '-o', DisplayName='DENSENET121',LineWidth=1.5)
plot(vertcat(MOBILENETV2_S.Var1)+1, vertcat(MOBILENETV2_S.rmse_train_history), '-o', DisplayName='MOBILENETV2',LineWidth=1.5)
plot(vertcat(INCEPTIONV3_S.Var1)+1, vertcat(INCEPTIONV3_S.rmse_train_history), '-o', DisplayName='INCEPTIONV3',LineWidth=1.5)
hold off
yscale("log")
legend()
grid()
set(gcf,'color','w')
title('Train RMSE')
ylabel('Deg')
xlabel('Epoch')

figure()
plot(vertcat(RESNET18_S.Var1)+1, vertcat(RESNET18_S.rmse_val_history), '-o',DisplayName='ResNet18', LineWidth=1.5)
hold on
plot(vertcat(RESNET34_S.Var1)+1, vertcat(RESNET34_S.rmse_val_history), '-o', DisplayName='ResNet34',LineWidth=1.5)
plot(vertcat(DENSENET121_S.Var1)+1, vertcat(DENSENET121_S.rmse_val_history), '-o', DisplayName='DENSENET121',LineWidth=1.5)
plot(vertcat(MOBILENETV2_S.Var1)+1, vertcat(MOBILENETV2_S.rmse_val_history), '-o', DisplayName='MOBILENETV2',LineWidth=1.5)
plot(vertcat(INCEPTIONV3_S.Var1)+1, vertcat(INCEPTIONV3_S.rmse_val_history), '-o', DisplayName='INCEPTIONV3',LineWidth=1.5)
hold off
yscale("log")
legend()
grid()
set(gcf,'color','w')
title('Val RMSE')
ylabel('Deg')
xlabel('Epoch')