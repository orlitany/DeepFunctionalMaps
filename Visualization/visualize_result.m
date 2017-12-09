clear all; close all; clc
addpath(genpath('./Utils/'));
%% data params
part_fname = './Data/tr_reg_080.mat';
part_off_fname = './Data/tr_reg_080.off';
model_fname = './Data/tr_reg_081.mat';
model_off_fname = './Data/tr_reg_081.off';


%%
part = load(part_fname);
part.shape = loadoff(part_off_fname);
model = load(model_fname);
model.shape = loadoff(model_off_fname);

%% read matches
load('./Results/test_list.mat');
C_est = squeeze(C_est);
softCorr = squeeze(softCorr);

%% plot result
figure, imagesc(C_est); title('Estimated C');
[~, matches] = max(softCorr,[],1);

colors = create_colormap(model.shape,model.shape);
figure(2);subplot(1,2,1);colormap(colors);
plot_scalar_map(model.shape,[1: size(model.shape.VERT,1)]');freeze_colors;title('Target');

figure(2);subplot(1,2,2);colormap(colors(matches,:));
plot_scalar_map(part.shape,[1: size(part.shape.VERT,1)]');freeze_colors;title('Source');
        





