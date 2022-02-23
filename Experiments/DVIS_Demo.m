%% Choose the dataset

profile off;
profile on;
prompt = 'Which dataset? \n 1) Salinas A \n 2) Jasper Ridge \n 3) Pavia Subset \n 4) Indian Pines \n';
datasetNames = {'Salinas A', 'Jasper Ridge',  'Pavia Subset', 'Indian Pines'};
dataSelectedName = datasetNames{input(prompt)};

OATable = zeros(9,1);

% Load selected dataset
[X,M,N,D,HSI,GT,Y,n, K] = loadHSI(dataSelectedName);

% Load all optimal hyperparameter sets
algNames = {'K-Means','K-Means+PCA', 'GMM+PCA', 'H2NMF', 'SC', 'SymNMF', 'KNN-SSC', 'LUND', 'D-VIS'};
OAs = zeros(1,9);
kappas = zeros(1,9); 
runtimes = zeros(1,9);
hyperparameters = cell(1,9);
Cs = zeros(n,9);
for i = 1:9
    hyperparameters{i} = loadHyperparameters(HSI, dataSelectedName, algNames{i});
end
disp('Dataset loaded.')

% Determine number of replicates for stochastic algorithms
profile off;
profile on;
prompt = 'Enter the number of desired runs for each stochastic algorithm: \n';
numReplicates = input(prompt);
if ~(round(numReplicates)-numReplicates == 0)
    error('The number of replicates must be an integer.')
end

% Determine number of replicates for stochastic algorithms
profile off;
profile on;
prompt = 'Do you want to visualize results?\n 1) Yes \n 2) No \n';

visualizeOn = (input(prompt) == 1);

clc
disp('Ready to Analyze HSI data.')

%% K-Means

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic
    Cstemp(:,i) = kmeans(X,K);
    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(1) = mean(OAtemp);
kappas(1) = mean(kappatemp);
runtimes(1) = mean(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(1)));
Cs(:,1) = Cstemp(:,i);

%% K-Means+PCA

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic
    [~, SCORE, ~, ~, pctExplained] = pca(X);
    nPCs = find(cumsum(pctExplained)>99,1,'first');
    XPCA = SCORE(:,1:nPCs);
    Cstemp(:,i) = kmeans(XPCA,K);
    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(2) = mean(OAtemp);
kappas(2) = mean(kappatemp);
runtimes(2) = mean(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(2)));
Cs(:,2) = Cstemp(:,i);

%% GMM+PCA

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic
    Cstemp(:,i) = evaluateGMMPCA(X,K);
    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(3) = mean(OAtemp);
kappas(3) = mean(kappatemp);
runtimes(3) = mean(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(3)));
Cs(:,3) = Cstemp(:,i);

%% H2NMF

tic
Cs(:,4) = hierclust2nmf(X'-min(X,[],'all'),K);
runtimes(4) = toc;
[ OAs(4), kappas(4)] = calcAccuracy(Y, Cs(:,4), ~strcmp('Jasper Ridge', dataSelectedName));

%% SC

NN = hyperparameters{5}.DiffusionNN;

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic
    
    % Nearest neighbor search
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
    Idx_NN(:,1)  = []; 

    % Graph decomposition
    G = extract_graph_large(X, hyperparameters{5}, Idx_NN, Dist_NN);

    % Spectral Clustering
    Cstemp(:,i) = SpectralClustering(G,K);

    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(5) = mean(OAtemp);
kappas(5) = mean(kappatemp);
runtimes(5) = mean(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(5)));
Cs(:,5) = Cstemp(:,i);

%% SymNMF

NN = hyperparameters{6}.DiffusionNN;
options.kk = NN;

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic
    
    % Nearest neighbor search
    [Idx_NN, ~] = knnsearch(X,X,'K', NN+1);
    Idx_NN(:,1)  = []; 

    % SymNMF Clustering
    Cstemp(:,i) = symnmf_cluster(X, K, options, Idx_NN);

    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(6) = mean(OAtemp);
kappas(6) = mean(kappatemp);
runtimes(6) = mean(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(6)));
Cs(:,6) = Cstemp(:,i);

%% KNN-SSC

NN = hyperparameters{7}.DiffusionNN;
alpha = 10;

tic

% Nearest neighbor search
[Idx_NN, ~] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 

% Extract KNN-SSC weight matrix
W = knn_SSC( X', alpha, NN, Idx_NN);

% Run spectral clustering with the KNN-SSC weight matrix
[~,Cs(:,7)] = spectral_clustering(abs(W)+abs(W'),K, Y-(~strcmp('Jasper Ridge', dataSelectedName))); 

runtimes(7) = toc;
[ OAs(7), kappas(7)] = calcAccuracy(Y, Cs(:,7), ~strcmp('Jasper Ridge', dataSelectedName));

%% LUND

NN = max(hyperparameters{8}.DiffusionNN,hyperparameters{8}.DensityNN);

tic

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

% Graph decomposition
G = extract_graph_large(X, hyperparameters{8}, Idx_NN, Dist_NN);

% KDE Computation
density = KDE_large(Dist_NN, hyperparameters{8});

runtimes(8) = toc;

% Run spectral clustering with the KNN-SSC weight matrix
[Clusterings, runtimesLUND] = MLUND_large(X, hyperparameters{8}, G, density);

[ OAs(8), kappas(8), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', dataSelectedName));

runtimes(8) = runtimes(8) + runtimesLUND(tIdx);
Cs(:,8) = Clusterings.Labels(:,tIdx);

%% D-VIS

NN = max(hyperparameters{9}.DiffusionNN,hyperparameters{9}.DensityNN);

tic

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

% Spectral Unmixing
pixelPurity = compute_purity(X,hyperparameters{9});

% Graph decomposition
G = extract_graph_large(X, hyperparameters{9}, Idx_NN, Dist_NN);

% KDE Computation
density = KDE_large(Dist_NN, hyperparameters{9});

runtimes(9) = toc;

% Run spectral clustering with the KNN-SSC weight matrix
[Clusterings, runtimesLUND] = MLUND_large(X, hyperparameters{9}, G, density);

[ OAs(9), kappas(9), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', dataSelectedName));

runtimes(9) = runtimes(9) + runtimesLUND(tIdx);
Cs(:,9) = Clusterings.Labels(:,tIdx);


%% Visualizations

if visualizeOn

    figure
    subplot(2,5,1)
    imagesc(GT)
    title([dataSelectedName, ' Ground Truth Labels'])
    axis equal tight
    xticks([])
    yticks([])
    
    for i = 1:9
        
        if ~strcmp('Jasper Ridge', dataSelectedName)
            C = zeros(size(Y));
            C(Y>1) = alignClusterings(Y(Y>1)-1,Cs(Y>1,i));
        else
            C = alignClusterings(Y,Cs(:,i));
        end 
    
        subplot(2,5,1+i)
        if ~strcmp('Pavia Subset', dataSelectedName)
            imagesc(reshape(C, M,N))
        else
            imagesc(reshape(C, M,N)')
        end
        title([algNames{i}, ' Clustering of ', dataSelectedName])
        axis equal tight
        xticks([])
        yticks([])
    end
end