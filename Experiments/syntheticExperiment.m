%% 

%% Load Synthetic Dataset

% Load dataset
% [X,M,N,D,HSI,GT,Y,n, K] = loadToy(1000);
load('/Users/sampolk/Documents/GitHub/DVISREpo/Data/syntheticData.mat')

% Nearest neighbor searches
[Idx_NN, Dist_NN] = knnsearch(X,X,'K',n);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = []; 

%% Run LUND

Hyperparameters = loadHyperparameters(X, 'Synthetic Data', 'LUND');

G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
density = KDE_large(Dist_NN, Hyperparameters);
[Clusterings, ~] = MLUND_large(X, Hyperparameters, G, density);
[ OALUND, kappaLUND, tIdx] = calcAccuracy(Y, Clusterings, 0);
CLUND = Clusterings.Labels(:,tIdx);

%% Run D-VIS

Hyperparameters = loadHyperparameters(X, 'Synthetic Data', 'D-VIS');
G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
density = KDE_large(Dist_NN, Hyperparameters);

% Spectral Unmixing Step
endmembers = zeros([size(hyperAvmax([X,X,X]', K, 0)), Hyperparameters.EndmemberParams.NumReplicates]);
volumes = zeros(Hyperparameters.EndmemberParams.NumReplicates,1);
for l = 1:Hyperparameters.EndmemberParams.NumReplicates
    [endmembers(:,:,l), volumes(l)] = hyperAvmax([X,X,X]', K, 0);
end
[~,l] = max(volumes);
endmembers = endmembers(:,:,l);

abundances = reshape(hyperNnls(X', endmembers(1:2,:))', M, N, K);
abundances = reshape(abundances,M*N,K);
pixelPurity = max(abundances,[],2);
pixelPurity(isnan(pixelPurity)) = 0; 

% Run D-VIS
[Clusterings, ~] = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));

[ OADVIC, kappaDVIC, tIdx] = calcAccuracy(Y, Clusterings, 0);
CDVIC = Clusterings.Labels(:,tIdx);


%% Visualize Results

subplot(1,3,1)
scatter(X(:,1), X(:,2), 36, Y, 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Dataset $X$, Colored By Ground Truth Label', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')

subplot(1,3,2)
scatter(X(:,1), X(:,2), 36, alignClusterings(Y,CLUND), 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Optimal D-VIC Clustering', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')

subplot(1,3,3)
scatter(X(:,1), X(:,2), 36, alignClusterings(Y,CDVIC), 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Optimal D-VIC Clustering', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
