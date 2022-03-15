%% Toy Data Experiment

%% Load Synthetic Dataset

% Load dataset
[X,M,N,D,HSI,GT,Y,n, K] = loadToy(1000);

% Nearest neighbor searches
[Idx_NN, Dist_NN] = knnsearch(X,X,'K',n);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = []; 

 
%% Parameters
load('syntheticData.mat')

% Load default hyperparameters
Hyperparameters = loadHyperparameters(X, 'Synthetic Data', 'D-VIS'); 
Hyperparameters.SpatialParams.ImageSize = [n,1];

% Parameter grid to be used
NNs = 140:10:400;
prctiles = 0.5:20;
numReplicates = 8;

%% Grid searches
% This file runs the grid search across the relevant values to recover the
% best-case D-VIS and LUND performances.

% Preallocate memory
maxOA = 0;
OAsLUND     = NaN*zeros(length(NNs), length(prctiles));
kappasLUND  = NaN*zeros(length(NNs), length(prctiles));
CsLUND      = zeros(n,length(NNs), length(prctiles));

OAsDVIC     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
kappasDVIC  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
CsDVIC      = zeros(n,length(NNs), length(prctiles), numReplicates);

delete(gcp('nocreate'))
poolObj = parpool;

OADiff = 0; 
j = 0;
% Run Grid Searches
for i = 1:length(NNs)

    Hyperparameters.DiffusionNN = NNs(i);
    Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
    Hyperparameters.EndmemberParams.K = K;
    Hyperparameters.K_Known = length(unique(Y));

    % Graph decomposition
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

    if G.EigenVals(2)<1

        for j = 1:length(prctiles)

            % Compute KDe
            Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
            density = KDE_large(Dist_NN, Hyperparameters);

            % Run M-LUND
            [Clusterings, ~] = MLUND_large(X, Hyperparameters, G, density);
            [ OAsLUND(i,j), kappasLUND(i,j), tIdx] = calcAccuracy(Y, Clusterings, 0);
            CsLUND(:,i,j) = Clusterings.Labels(:,tIdx);

            if poolObj.NumWorkers<6
                delete(gcp('nocreate'));
                poolObj = parpool;
            end

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


            parfor k = 1:numReplicates



                % Run D-VIS
                [Clusterings, ~] = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
                [ OAsDVIC(i,j,k), kappasDVIC(i,j,k), tIdx] = calcAccuracy(Y, Clusterings, 0);
                CsDVIC(:,i,j,k) = Clusterings.Labels(:,tIdx);

                disp([i/length(NNs), j/length(prctiles), k/numReplicates, OADiff])

            end
             

            OADiff = [max(median(OAsDVIC,3),[],'all') , max(OAsLUND,[],'all')];
            disp([i/length(NNs), j/length(prctiles), OADiff])

        end


    end

    save(strcat('SyntheticResults'),  'OAsLUND', 'kappasLUND', 'CsLUND',  'OAsDVIC', 'kappasDVIC', 'CsDVIC', 'NNs', 'prctiles', 'numReplicates', 'maxOA')
end

%% Visualization of best results

% Load results of grid search
load('SyntheticResults')

% Optimal LUND clustering
[~,k] = max(OAsLUND,[],'all');
[i,j] = ind2sub(size(OAsLUND),k);

h = figure;
scatter(X(:,1), X(:,2), 36,alignClusterings(Y, CsLUND(:,i,j)), 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Optimal LUND Clustering', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
saveas(h, 'SyntheticOptimalLUND', 'epsc')
close all 
%% Optimal D-VIC clustering
[~,k] = max(median(OAsDVIC,3),[],'all');
[i,j] = ind2sub(size(OAsDVIC),k);

h = figure;
scatter(X(:,1), X(:,2), 36, alignClusterings(Y,CsDVIC(:,i,j,end)), 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Optimal D-VIC Clustering', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
saveas(h, 'SyntheticOptimalDVIC', 'epsc')
close all 

%% Ground Truth

h = figure;
scatter(X(:,1), X(:,2), 36, Y, 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Synthetic Dataset, Colored By Ground Truth Label', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
saveas(h, 'SyntheticGT', 'epsc')
close all 

%% Pixel Purity Visualization

h = figure;
% Spectral Unmixing Step
Hyperparameters.EndmemberParams.K = K;
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

scatter(X(:,1), X(:,2), 36, pixelPurity, 'filled')
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Synthetic Dataset, Colored by Purity', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
colorbar
saveas(h, 'SyntheticPurity', 'epsc')
close all 
%% Density Visualization
h = figure;
Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
density = KDE_large(Dist_NN, Hyperparameters);

scatter(X(:,1), X(:,2), 36, log10(density))
set(gca,'FontSize', 16, 'FontName', 'Times')
title('Dataset $X$, colored by $\log_{10}(p(x))$', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
xlabel('$x_2$', 'interpreter', 'latex')
colorbar
saveas(h, 'SyntheticDensity', 'epsc')
close all 