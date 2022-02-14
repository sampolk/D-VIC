%% Get and preprocess datasets
% Preprocesses by normalizing and storing nearest neighbor searches
extractData

%% Optimal Parameters

numReplicates = 10; % Number of times we run stochastic algorithms

% DBSCANParams = [[210, 0.006775988125625]; [30, 0.002105692703406]; [3, 0.025222783601406]];
% SCParams = [30, 80, 900];
% SymNMFParams = [400, 400, 600];
% KNNSSCParams = [10, 160, 900];
% LUNDParams = [[40, 0.004909144261198]; [40, 0.007173061688410]; [40, 0.005614907669686]];
% DVISParams = [[10, 0.022345402316921]; [90, 0.030433597983468]; [40, 0.011550449082158]];


%% 

kappaTable = zeros(3,10);
OATable = zeros(3,10);
runtimeTable = zeros(3,10);

datasets = {'SalinasACorrected','IndianPinesCorrected', 'PaviaSubset1'};

for dataIdx = 1:3

    load(datasets{dataIdx})

    % ========================== Evaluate K-Means =========================
    % We run numReplicates trials
    OATemp = zeros(numReplicates,1);
    kappaTemp = zeros(numReplicates,1);
    runtimeTemp = zeros(numReplicates,1);
    parfor i = 1:numReplicates
        tic
        C = kmeans(X, K);        
        runtimeTemp(i) = toc;
        [~, ~, OATemp(i), ~, kappaTemp(i)]= measure_performance(C, Y);
    end
    
    % Average performance across numReplicates runs
    OATable(dataIdx,1)=mean(OATemp);
    kappaTable(dataIdx,1) = mean(kappaTemp);
    runtimeTable(dataIdx,1) = mean(runtimeTemp);

    % ======================== Evaluate K-Means+PCA =======================
    % We run numReplicates trials
    OATemp = zeros(numReplicates,1);
    kappaTemp = zeros(numReplicates,1);
    runtimeTemp = zeros(numReplicates,1);
    parfor i = 1:numReplicates
        tic
        [~, SCORE, ~, ~, pctExplained] = pca(X);
        nPCs = find(cumsum(pctExplained)>99,1,'first');
        XPCA = SCORE(:,1:nPCs);
        C = kmeans(XPCA, K, 'MaxIter', 200);
        runtimeTemp(i) =  toc;
        [~, ~, OATemp(i), ~, kappaTemp(i)]= measure_performance(C, Y);
    end
 
    % Average performance across numReplicates runs
    OATable(dataIdx,2) = mean(OATemp);
    kappaTable(dataIdx,2) = mean(kappaTemp);
    runtimeTable(dataIdx,2) = mean(runtimeTemp);

    % ========================== Evaluate GMM+PCA =========================
    % Run over numReplicates trials
    OATemp = zeros(numReplicates,1);
    kappaTemp = zeros(numReplicates,1);
    runtimeTemp = zeros(numReplicates,1);
    for i = 1:numReplicates
        clear C

        % We run the same code (at most) 10 times and keep the clustering
        % that is the first to converge. 
        options = statset('MaxIter',200);
        iter = 1;
        keepRunning = true;
        while iter<10 && keepRunning 
            try
                tic
                [COEFF, SCORE, ~, ~, pctExplained] = pca(X);
                nPCs = find(cumsum(pctExplained)>99,1,'first');
                XPCA = SCORE(:,1:nPCs);
    
                C = cluster(fitgmdist(XPCA,K, 'Options', options), XPCA);
                runtimeTemp(i) = toc;
                keepRunning = false;
            catch
            end
            iter = iter+1;
        end

        % If the EM algorithm doesn't converge, that indicates that the
        % dimension-reduced dataset is still too high-dimensional for EM.
        % So, we reduce dimensionality by 1 PC until we converge and 
        % increase number of EM iterations to 1000.

        subtractNum = 1;
        while ~exist('C') && subtractNum<nPCs-1
            % Too many PCs
            options = statset('MaxIter',1000); 
            try
                tic
                [COEFF, SCORE, ~, ~, pctExplained] = pca(X);
                nPCs = find(cumsum(pctExplained)>99,1,'first');
                XPCA = SCORE(:,1:nPCs);
                C = cluster(fitgmdist(XPCA(:,1:end-subtractNum),K, 'Options', options), XPCA(:,1:end-subtractNum));
                runtimeTemp(i) = toc;
            catch
                subtractNum = subtractNum + 1;
            end
        end
        % If, after all this, we still don't converge, we assign a
        % random clustering.
        if  ~exist('C')
            C = randi(K, M*N,1);
        end

        [~, ~, OATemp(i), ~, kappaTemp(i)]= measure_performance(C, Y);
    end
    
    % Average performance across numReplicates runs
    OATable(dataIdx,3) = mean(OATemp);
    kappaTable(dataIdx,3) = mean(kappaTemp);
    runtimeTable(dataIdx,3) = mean(runtimeTemp);

    % ========================== Evaluate DBSCAN ==========================

    if isfile(strcat('DBSCANHP', datasets{dataIdx}))

        % Load Optimal Hyperparameters 
        load(strcat('DBSCANHP', datasets{dataIdx}))
    
        tic
        C = dbscan(X,epsilon, minPts);
        runtimeTable(dataIdx,4) = toc;
        [~,~, OATable(dataIdx,4), ~, kappaTable(dataIdx,4)]= measure_performance(C, Y);

    end

    % ========================== Evaluate H2NMF ===========================

    tic
    C = hierclust2nmf(X'-min(X,[],'all'),K);
    runtimeTable(dataIdx,5) = toc;
    [~, ~, OATable(dataIdx,5) , ~, kappaTable(dataIdx,5)]= measure_performance(C, Y);


    % ============================ Evaluate SC ============================

    % Load Optimal Hyperparameters
    load(strcat('SCHP', datasets{dataIdx}))

    % Preallocate memory
    OATemp     = NaN*zeros(numReplicates,1);
    kappaTemp  = NaN*zeros(numReplicates,1);
    runtimeTemp     = NaN*zeros(numReplicates,1);
    for i=1:numReplicates

        tic
    
        % Perform nearest neighbor searches
        [Idx_NN, Dist_NN] = knnsearch(X,X,'K',NN+1);
        Idx_NN(:,1) = [];
        Dist_NN(:,1) = [];

        % Extract  graph structure
        Hyperparameters.DiffusionNN = NN;
        Hyperparameters.NEigs = 10;
        Hyperparameters.SpatialParams = [];
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

        % Run SC using that graph
        C = SpectralClustering(G,K);

        runtimeTemp(i) = toc;

        [~,~, OATemp(i), ~, kappaTemp(i)]= measure_performance(C, Y);
    end
    % Average performance across numReplicates runs
    OATable(dataIdx,6) = mean(OATemp);
    kappaTable(dataIdx,6) = mean(kappaTemp);
    runtimeTable(dataIdx,6) = mean(runtimeTemp);

    % ========================== Evaluate SymNMF ==========================

    % Load optimal hyperparameters
    load(strcat('SymNMFHP', datasets{dataIdx}))

    % Preallocate memory
    OATemp     = NaN*zeros(numReplicates,1);
    kappaTemp  = NaN*zeros(numReplicates,1);
    runtimeTemp     = NaN*zeros(numReplicates,1);
    for i=1:numReplicates

        tic
    
        % Perform nearest neighbor searches
        [Idx_NN, Dist_NN] = knnsearch(X,X,'K',NN+1);
        Idx_NN(:,1) = [];
        Dist_NN(:,1) = [];

        % Implement SymNMF clustering       
        options.kk = NN;
        C = symnmf_cluster(X, K, options, Idx_NN);

        runtimeTemp(i) = toc;

        [~,~, OATemp(i), ~, kappaTemp(i)]= measure_performance(C, Y);
    end
    % Average performance across numReplicates runs
    OATable(dataIdx,7) = mean(OATemp);
    kappaTable(dataIdx,7) = mean(kappaTemp);
    runtimeTable(dataIdx,7) = mean(runtimeTemp);

%     % ========================= Evaluate KNN-SSC ==========================
% 
%     % Load optimal hyperparameters
%     load(strcat('KNNSSCHP', datasets{dataIdx}))
% 
%     tic
%     % Perform nearest neighbor searches
%     [Idx_NN, Dist_NN] = knnsearch(X,X,'K',NN+1);
%     Idx_NN(:,1) = [];
%     Dist_NN(:,1) = [];
% 
%     % Extract KNN-SSC Weight matrix 
%     W = knn_SSC( X', 10, NN, Idx_NN);
% 
%     % Run Spectral clustering using that weight matrix
%     [~,C] = spectral_clustering(abs(W)+abs(W'),K,Y);
%     
%     runtimeTable(dataIdx,8) = toc;
%     [~, ~, OATable(dataIdx,8) , ~, kappaTable(dataIdx,8)]= measure_performance(C, Y);

    % ========================== Evaluate LUND ============================

    % Load optimal hyperparameters
    load(strcat('LUNDHP', datasets{dataIdx}))
    
    tic 
    % Perform nearest neighbor searches
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',NN+1);
    Idx_NN(:,1) = [];
    Dist_NN(:,1) = [];
    
    % Extract graph
    [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

    % Extract density
    density = KDE_large(Dist_NN, Hyperparameters);
 
    % Run LUND across all relevant time steps
    Clusterings = MLUND_large(X, Hyperparameters, G, density);
    
    runtimeTable(dataIdx,9) = toc;

    % Optimize over t for LUND OA
    [~,~, OATable(dataIdx,9), ~, kappaTable(dataIdx,9), ~]= measure_performance(Clusterings, Y);

    % ========================= Evaluate D-VIS ============================

    % Load optimal hyperparameters
    load(strcat('DVISHP', datasets{dataIdx}))

    % Preallocate memory
    OATemp     = NaN*zeros(numReplicates,1);
    kappaTable  = NaN*zeros(numReplicates,1);
    runtimeTemp     = NaN*zeros(numReplicates,1);
    for i = 1:10
      
        tic 
        % Perform Spectral Unmixing
        Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
        Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
        Hyperparameters.EndmemberParams.NumReplicates = 100;
        [pixelPurity, ~, ~] = compute_purity(X,Hyperparameters);
    
        % Perform nearest neighbor searches
        [Idx_NN, Dist_NN] = knnsearch(X,X,'K',Hyperparameters.DiffusionNN+1);
        Idx_NN(:,1) = [];
        Dist_NN(:,1) = [];
        
        % Extract graph
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    
        % Extract density
        density = KDE_large(Dist_NN, Hyperparameters);
        
        % Run D-VIS across all relevant time steps
         Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
        
        runtimeTemp(i) = toc;
    
        % Optimize over t for DVIS OA
        [~,~, OATemp(i), ~, kappaTemp(i), ~]= measure_performance(Clusterings, Y);
    end
    runtimeTable(dataIdx,10) = mean(runtimeTemp);
    OATable(dataIdx,10) = mean(OATemp);
    kappaTable(dataIdx,10) = mean(kappaTemp);

end