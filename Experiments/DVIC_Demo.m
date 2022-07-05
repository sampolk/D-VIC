%{

This script replicates Figures 6-9 and Tables I-II in the following article

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

D-VIC is shown to substantially outperform related hypserspectral image 
clustering algorithms on 3 real datasets. 

To run this script, real hyperspectral image data (Salinas A, Indian Pines, 
& Jasper Ridge) must be downloaded from the following links:

    - http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    - https://rslab.ut.ac.ir/data

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}
%% Choose the dataset
clear
clc

profile off;
profile on;
prompt = 'Which dataset? \n 1) Salinas A \n 2) Jasper Ridge \n 3) Indian Pines \n';
datasetNames = {'Salinas A', 'Jasper Ridge', 'Indian Pines'};
dataSelectedName = datasetNames{input(prompt)};

% Load selected dataset
[X,M,N,D,HSI,GT,Y,n, K] = loadHSI(dataSelectedName);

% Load all optimal hyperparameter sets
algNames = {'K-Means','K-Means+PCA', 'GMM+PCA', 'SC', 'SymNMF', 'KNN-SSC', 'FSSC', 'LUND', 'D-VIC'};
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
elseif isempty(numReplicates)
    numReplicates = 1;
end

% Determine number of replicates for stochastic algorithms
profile off;
profile on;
prompt = 'Do you want to visualize results?\n 1) Yes \n 2) No \n';

visualizeOn = (input(prompt) == 1);

clc
profile off;
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

OAs(1) = median(OAtemp);
kappas(1) = median(kappatemp);
runtimes(1) = median(runtimetemp);
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

OAs(2) = median(OAtemp);
kappas(2) = median(kappatemp);
runtimes(2) = median(runtimetemp);
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

OAs(3) = median(OAtemp);
kappas(3) = median(kappatemp);
runtimes(3) = median(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(3)));
Cs(:,3) = Cstemp(:,i);

%% SC

NN = hyperparameters{4}.DiffusionNN;

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
    G = extractGraph(X, hyperparameters{4}, Idx_NN, Dist_NN);

    % Spectral Clustering
    Cstemp(:,i) = SpectralClustering(G,K);

    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(4) = median(OAtemp);
kappas(4) = median(kappatemp);
runtimes(4) = median(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(4)));
Cs(:,4) = Cstemp(:,i);

%% SymNMF

NN = hyperparameters{5}.DiffusionNN;
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

OAs(5) = median(OAtemp);
kappas(5) = median(kappatemp);
runtimes(5) = median(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(5)));
Cs(:,5) = Cstemp(:,i);

%% KNN-SSC

NN = hyperparameters{6}.DiffusionNN;
alpha = 10;

tic

% Nearest neighbor search
[Idx_NN, ~] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 

% Extract KNN-SSC weight matrix
W = sparse(knn_SSC( X', alpha, NN, Idx_NN));
W = (W+W')./2;   %  (W+W')./2 forces symmetry

% First n_eigs eigenpairs of transition matrix D^{-1}W:
[V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs'); 
    
if flag
    % Didn't converge. Try again with larger subspace dimension.
    [V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs',  'SubspaceDimension', max(4*min(K,10),40));  
end         

EigenVecs_Normalized = real(V(:,1:min(K,10))./vecnorm(V(:,1:min(K,10)),2,2));
Cs(:,6) = kmeans(EigenVecs_Normalized, K);

runtimes(6) = toc;
[ OAs(6), kappas(6)] = calcAccuracy(Y, Cs(:,6), ~strcmp('Jasper Ridge', dataSelectedName));

%% FSSC

NN = hyperparameters{7}.DiffusionNN;
alpha_u = hyperparameters{7}.alpha_u;

OAtemp = zeros(numReplicates,1);
kappatemp = zeros(numReplicates,1);
runtimetemp = zeros(numReplicates,1);
Cstemp = zeros(n,numReplicates);

for i = 1:numReplicates
    tic

    % SymNMF Clustering
    [~,~,Cstemp(:,i),~,~] = FSSC(X,11,NN,K,10,alpha_u);

    runtimetemp(i) = toc;
    [ OAtemp(i), kappatemp(i)] = calcAccuracy(Y, Cstemp(:,i), ~strcmp('Jasper Ridge', dataSelectedName));
end

OAs(7) = median(OAtemp);
kappas(7) = median(kappatemp);
runtimes(7) = median(runtimetemp);
[~,i] = min(abs(OAtemp - OAs(7)));
Cs(:,7) = Cstemp(:,i);

%% LUND

NN = max(hyperparameters{8}.DiffusionNN,hyperparameters{8}.DensityNN);

tic

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

% Graph decomposition
G = extractGraph(X, hyperparameters{8}, Idx_NN, Dist_NN);

% KDE Computation
density = KDE(Dist_NN, hyperparameters{8});

runtimes(8) = toc;

% Run spectral clustering with the KNN-SSC weight matrix
[Clusterings, runtimesLUND] = MLUND(X, hyperparameters{8}, G, density);

[ OAs(8), kappas(8), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', dataSelectedName));

runtimes(8) = runtimes(8) + runtimesLUND(tIdx);
Cs(:,8) = Clusterings.Labels(:,tIdx);

%% D-VIC

Hyperparameters = hyperparameters{9};
NN = max(Hyperparameters.DiffusionNN,Hyperparameters.DensityNN);

OAtemp = NaN*zeros(numReplicates,1);
kappatemp = NaN*zeros(numReplicates,1);
runtimetemp = NaN*zeros(numReplicates,1);
Cstemp = NaN*zeros(n,numReplicates);

for k = 1:numReplicates 

    tic
    
    % Nearest neighbor search
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K', NN+1);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];

    % Graph decomposition
    G = extractGraph(X, Hyperparameters, Idx_NN, Dist_NN);
    
    % KDE Computation
    density = KDE(Dist_NN, Hyperparameters);

    % Spectral Unmixing Step
    Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
    pixelPurity = compute_purity(X,Hyperparameters);

    runtimetemp(k) =  toc;

    if G.EigenVals(2)<1 % Only use graphs with good spectral decompositions

        [Clusterings, DVISruntimes] = MLUND(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
        [ OAtemp(k), kappatemp(k), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', dataSelectedName));
        Cstemp(:,k) = Clusterings.Labels(:,tIdx);
        runtimetemp(k) = runtimetemp(k) + DVISruntimes(tIdx);
    else
        Cstemp(:,k) = NaN;
        OAtemp(k) = NaN;
        kappatemp(k) = NaN;
        runtimetemp(k) = NaN;
    end
end
OAs(9) = nanmedian(OAtemp);
kappas(9) = nanmedian(kappatemp);
runtimes(9) = nanmedian(runtimetemp);
[~,i] = min(abs(OAtemp-OAs(9))); % clustering producing the closest OA to the mean performance
Cs(:,9) = Cstemp(:,i);
 
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
        
        subplot(2,5,i+1)
        if ~strcmp('Jasper Ridge', dataSelectedName)
            C = zeros(size(Y));
            C(Y>1) = alignClusterings(Y(Y>1)-1,Cs(Y>1,i));
        else
            C = alignClusterings(Y,Cs(:,i));
        end 
        imagesc(reshape(C, M,N)) 
        title([algNames{i}, ' Clustering of ', dataSelectedName])
        axis equal tight
        xticks([])
        yticks([])
    end
end