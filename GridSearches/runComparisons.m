%% Prellocate Memory and load results

load('results')

datasets = {'IndianPinesCorrected', 'SalinasACorrected', 'paviaU'};

OATable = zeros(3,10);
kappaTable = zeros(3,10);
runtimeTable = zeros(3,10);

%% K-Means

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));

    % Run over 10 trials
    OATemp = zeros(10,1);
    KappaTemp = zeros(10,1);
    Cs = zeros(length(X),10);
    runtimeTemp = zeros(10,1);
    parfor j = 1:10
        tic
        C = kmeans(X, K);        
        runtimeTemp(j) = toc;
        [~, ~, OATemp(j), ~, KappaTemp(j)]= measure_performance(C, Y);
        Cs(:,j) = C; 
    end
    
    % Average performance across 10 runs
    OATable(i,1) = mean(OATemp);
    kappaTable(i,1) = mean(KappaTemp);
    runtimeTable(i,1) = mean(runtimeTemp);

end

%% K-Means+PCA

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));

    % Run over 10 trials
    OATemp = zeros(10,1);
    KappaTemp = zeros(10,1);
    Cs = zeros(length(X),10);
    parfor j = 1:10
        tic
        [~, SCORE, ~, ~, pctExplained] = pca(X);
        nPCs = find(cumsum(pctExplained)>99,1,'first');
        XPCA = SCORE(:,1:nPCs);
        C = kmeans(XPCA, K);        
        runtimeTemp(j) = toc;
        [~, ~, OATemp(j), ~, KappaTemp(j)]= measure_performance(C, Y);
        Cs(:,j) = C; 
    end
    
    % Average performance across 10 runs
    OATable(i,2) = mean(OATemp);
    kappaTable(i,2) = mean(KappaTemp);
    runtimeTable(i,2) = mean(runtimeTemp);

end

%% GMM+PCA

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));

    % Run over 10 trials
    OATemp = zeros(10,1);
    KappaTemp = zeros(10,1);
    Cs = zeros(length(X),10);
    parfor j = 1:10
        tic
        C = evaluateGMMPCA(X,K);        
        runtimeTemp(j) = toc;
        [~, ~, OATemp(j), ~, KappaTemp(j)]= measure_performance(C, Y);
        Cs(:,j) = C; 
    end
    
    % Average performance across 10 runs
    OATable(i,3) = mean(OATemp);
    kappaTable(i,3) = mean(KappaTemp);
    runtimeTable(i,3) = mean(runtimeTemp);

end

%% DBSCAN

for i = 1:3

    if i == 3
        OATable(i,4) = NaN;
        kappaTable(i,4) = NaN;
        runtimeTable(i,4) = NaN;
    else
    
        load(datasets{i})
    
        minPts = hyperparameters{1,4}.MinPts;
        epsilon = hyperparameters{1,4}.Epsilon;

        tic
        C = dbscan(X,epsilon, minPts);
        runtimeTable(i,4) = toc;
        
        [~, ~, OATable(i,4), ~, kappaTable(i,4)]= measure_performance(C, Y);
    end
end

%% SC

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));

    % Run over 10 trials
    OATemp = zeros(10,1);
    KappaTemp = zeros(10,1);
    runtimeTemp = zeros(10,1);
    Cs = zeros(length(X),10);
    parfor j = 1:10
        tic
        C = evaluateSC(X, hyperparameters{i,6}.NN, K);
        runtimeTemp(j) = toc;
        [~,~, OATemp(j), ~, KappaTemp(j)]= measure_performance(C, Y);
        Cs(:,i,j) = C;
    end

    % Average performance across 10 runs
    OATable(i,5) = mean(OATemp);
    kappaTable(i,5) = mean(KappaTemp);
    runtimeTable(i,5) = mean(runtimeTemp);

end

%% H2NMF

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));

    tic
    C = hierclust2nmf(X'-min(X,[],'all'),K);
    runtimeTable(i,6) = toc;

    [~, ~, OATable(i,6) , ~, kappaTable(i,6)]= measure_performance(C, Y);
end

%% SymNMF

for i = 1:3
    
    load(datasets{i})
    K = length(unique(Y));
    options.kk = hyperparameters{i,7}.NN;

    % Run over 10 trials
    OATemp = zeros(10,1);
    KappaTemp = zeros(10,1);
    runtimeTemp = zeros(10,1);
    Cs = zeros(length(X),10);
    parfor j = 1:10

        tic
        
        % Perform KNN search
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', hyperparameters{i,7}.NN+1);
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);

        % Implement SymNMF
        C = symnmf_cluster(X, K, options, Idx_NN);

        runtimeTemp(j) = toc;

        [~, ~, OATemp(j), ~, KappaTemp(j)]= measure_performance(C, Y);
 
    end

    % Average performance across 10 runs
    OATable(i,7) = mean(OATemp);
    kappaTable(i,7) = mean(KappaTemp);
    runtimeTable(i,7) = mean(runtimeTemp);

end

%% KNN-SSC

for i = 1:3


    load(datasets{i})
    K = length(unique(Y));

    alpha = hyperparameters{1,8}.Alpha;
    NN = hyperparameters{1,8}.NN;

    tic

    [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', NN+1);
    Idx_NN(:,1) = [];
    Dist_NN(:,1) = [];
    
    W = knn_SSC( X', alpha, NN, Idx_NN);
    [~,C] = spectral_clustering(abs(W)+abs(W'),K,Y);
    runtimeTable(i,8) = toc;
    
    [~, ~, OATable(i,8), ~, kappaTable(i,8)]= measure_performance(C, Y);
     
end

%% LUND
% TODO: there is an issue with the hyperparameter choices

for i = 1:3

    load(datasets{i})
    Hyperparameters = hyperparameters{i,9};

    tic

    [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', Hyperparameters.DiffusionNN+1);
    Idx_NN(:,1) = [];
    Dist_NN(:,1) = [];

    [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    density = KDE_large(Dist_NN, Hyperparameters);

    Clusterings = MLUND_large(X, Hyperparameters, G, density); 
    
    runtimeTable(i,9) = toc;

    [~,~, OATable(i,9), ~, kappaTable(i,9), ~]= measure_performance(Clusterings, Y);

end

%% D-VIS

for i = 1:3

    load(datasets{i})
    Hyperparameters = hyperparameters{i,10};

    OATemp = zeros(10,1);
    kappaTemp = zeros(10,1);
    runtimeTemp = zeros(10,1);
    
    for j = 1:10

        tic
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', NN+1);
        Idx_NN(:,1) = [];
        Dist_NN(:,1) = [];
    
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
        density = KDE_large(Dist_NN, Hyperparameters);
    
        Clusterings = MLUND_large(X, Hyperparameters, G, density);
        runtimeTemp(j) = toc;
        [~,~, OATemp(j), ~, kappaTemp(j), ~]= measure_performance(Clusterings, Y);

    end
    
    OATable(i,10) = mean(OATemp);
    kappaTable(i,10) = mean(kappaTemp);
    runtimeTable(i,10) = mean(runtimeTemp);
    
end