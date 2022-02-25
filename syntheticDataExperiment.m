%% Synthetic Data Experiment

%% Load Visualization of Synthetic Dataset

[X,M,N,D,HSI,GT,Y,n, K] = loadHSI('Synthetic HSI');

%% Synthetic data visualization

h = figure;
imagesc(GT)
xticks([])
yticks([])
axis equal tight
title(['Synthetic HSI Ground Truth'], 'interpreter','latex', 'FontSize', 17) 
 
saveas(h, 'SyntheticGT', 'epsc')

h = figure;
[~,scores] = pca(X);
imagesc(reshape(scores(:,1), M,N))


a = colorbar;
%     a.Label.String = 'OA (%)';
xticks([])
yticks([])
axis equal tight
title(['First Principal Component Scores'], 'interpreter','latex', 'FontSize', 17) 
set(gca,'FontName', 'Times', 'FontSize', 14)
saveas(h, 'SyntheticPC', 'epsc')

close all 

%% LUND

% Extract optimal hyperparameters
Hyperparameters = loadHyperparameters(HSI, 'Synthetic HSI', 'LUND');
NN = max(Hyperparameters.DiffusionNN,Hyperparameters.DensityNN);

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

% Graph decomposition
G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

% KDE Computation
density = KDE_large(Dist_NN, Hyperparameters);

% Run LUND
Clusterings = MLUND_large(X, Hyperparameters, G, density);

% Calculate accuracy
[ OAstats(1),~  , tIdx] = calcAccuracy(Y, Clusterings,1);

C = Clusterings.Labels(:,tIdx);
C(Y == 0 ) = 0;
C = alignClusterings(Y, C);

h = figure;
imagesc(reshape(C, M,N))
xticks([])
yticks([])
axis equal tight
title(['Optimal LUND Clustering of Synthetic HSI'], 'interpreter','latex', 'FontSize', 17)  
saveas(h, 'SyntheticLUND', 'epsc')
close all 

%% D-VIS

% Extract optimal hyperparameters
Hyperparameters = loadHyperparameters(HSI, 'Synthetic HSI', 'D-VIS');
NN = max(Hyperparameters.DiffusionNN,Hyperparameters.DensityNN);

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

% Spectral Unmixing Step
Hyperparameters.EndmemberParams.K = hysime(X'); % implement hysime to get best estimate for number of endmembers 
pixelPurity = compute_purity(X,Hyperparameters);

% Graph decomposition
G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

% KDE Computation
density = KDE_large(Dist_NN, Hyperparameters);

% Run D-VIS
Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));

[ OAstats(2),~, tIdx] = calcAccuracy(Y, Clusterings, 1);

C = Clusterings.Labels(:,tIdx);
C(Y == 0 ) = 0;
C = alignClusterings(Y, C);

h = figure;
imagesc(reshape(C, M,N))
xticks([])
yticks([])
axis equal tight
title(['Optimal D-VIS Clustering of Synthetic HSI'], 'interpreter','latex', 'FontSize', 17) 
 
saveas(h, 'SyntheticDVIS', 'epsc')
close all 


OAstats = array2table(OAstats, 'VariableNames', {'LUND OA', 'D-VIS OA'});
disp(OAstats)


