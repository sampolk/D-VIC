%% Run Experiments

%% Preallocate memory

OATable = zeros(9,4);
kappaTable = zeros(9,4);
runtimeTable = zeros(9,4);

%% Run Experiments

% Load optimal hyperparameters
load('results', 'hyperparameters')

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

ct = 0;
for dataIdx =  [1,2, 5, 11]

    ct = ct+1;

    if dataIdx == 13
        % Set number of nearest neighbors to use in graph and KDE construction.
        NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];
    end

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    if dataIdx <7
        load(datasets{dataIdx})
        HSI = reshape(X, M,N,size(X,2));
        GT = reshape(Y,M,N);
     end

    if dataIdx == 6
        
        % Perfor knnsearch for new datasets
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);

    end
    if dataIdx == 7 || dataIdx == 8
        load('Pavia_gt.mat')
        load('Pavia')
        if dataIdx == 7
            HSI = pavia(101:400,241:300,:);
            GT = pavia_gt(101:400,241:300);
        elseif dataIdx == 8
            HSI = HSI(498:end,1:100,:);
            GT = GT(498:end,1:100);        
        end
    elseif dataIdx == 9 || dataIdx == 10 || dataIdx == 11

        if dataIdx == 9
            load('Botswana.mat')
            load('Botswana_gt.mat')
            HSI = Botswana(285:507, 204:253,:);
            GT = Botswana_gt(285:507, 204:253);
        elseif dataIdx == 10
            load('Pavia_gt')
            load('Pavia.mat')
            HSI = pavia(101:250,201:350,:);
            GT = pavia_gt(101:250,201:350);
        elseif dataIdx == 11
            load('Pavia_gt')
            load('Pavia.mat')
            HSI = pavia(201:400, 430:530,:);
            GT = pavia_gt(201:400, 430:530);
        end
        X = reshape(HSI, size(HSI, 1)*size(HSI, 2), size(HSI,3));
        X=X./repmat(sqrt(sum(X.*X,1)),size(X,1),1); % Normalize HSI
        HSI = reshape(X, size(HSI, 1),size(HSI, 2), size(HSI,3));
    end


    if dataIdx == 12 || dataIdx == 13
        load(datasets{dataIdx})
%         X = X./vecnorm(X,2,2);
        HSI = reshape(X,M,N,D);
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);

        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end


    [M,N,D] = size(HSI);

    if dataIdx >= 7

        X = reshape(HSI, M*N,D);
        Y = reshape(GT,M*N,1);
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end


    % If Salinas A, we add gaussian noise and redo nearest neighbor searches. 
    if dataIdx == 5
        X = X + randn(size(X)).*10^(-7);
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end 

    newGT = zeros(size(GT));
    uniqueClass = unique(GT);
    K = length(uniqueClass);
    for k = 1:K
    newGT(GT==uniqueClass(k)) = k;
    end
    if ~(dataIdx==2)
        K = K-1;
    end
    Y = reshape(newGT,M*N,1);
    GT = newGT;

    Idx_NN = Idx_NN(:,1:901);
    Dist_NN = Dist_NN(:,1:901);
    clear pavia pavia_gt uniqueClass k 


    % ============================== DVIS ==============================

    Hyperparameters = hyperparameters{ct,9};
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    runtimeTemp = zeros(numReplicates,1);
    OATemp = zeros(numReplicates,1);
    kappaTemp = zeros(numReplicates,1);

    for i = 1:numReplicates

        tic

        % Spectral unmixing step
        Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
        Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
        Hyperparameters.EndmemberParams.NumReplicates = 100;
        [pixelPurity, ~, ~] = compute_purity(X,Hyperparameters);

        % Perform nearest neighbor search
        [Idx_NN, Dist_NN] = knnsearch(X,X,'K', Hyperparameters.DiffusionNN); 

        % Calculate KDE and graph
        density = KDE_large(Dist_NN, Hyperparameters);
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    
        % Perform clustering 
        Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
        runtimeTemp(i) = toc/length(Clusterings.K); % Average runtime across the many runs

        if dataIdx == 2
            % Jasper Ridge has no unlabeled pixels. So no exclusion for
            % performance analysis
            
            thisOAsTemp = zeros(length(Clusterings.K),1);
            thiskappasTemp = zeros(length(Clusterings.K),1);
            for t = 1:length(Clusterings.K)

                C = alignClusterings(Y,Clusterings.Labels(:,t));
                confMat = confusionmat(Y,C);
    
                thisOAsTemp(t) = sum(diag(confMat)/length(C)); 
                p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
                thiskappasTemp(t)=(thisOAsTemp(t)-p)/(1-p);
            end
    
            [OATemp(i), tIdx] = max(thisOAsTemp);
            kappaTemp(i) = thiskappasTemp(tIdx);
    
        else
            % We exclude unlabeled pixels for remaining datasets 

            thisOAsTemp = zeros(length(Clusterings.K),1);
            thiskappasTemp = zeros(length(Clusterings.K),1);
            for t = 1:length(Clusterings.K)
                C = alignClusterings(Y(Y>1),Clusterings.Labels(Y>1,t));
                confMat = confusionmat(Y(Y>1)-1,C); % confusion matrix for labeled pixels
    
                thisOAsTemp(t) = sum(diag(confMat)/length(C)); 
    
                p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
                thiskappasTemp(t)=(thisOAsTemp(t)-p)/(1-p);
            end
    
            [OATemp(i), tIdx] = max(thisOAsTemp);
            kappaTemp(i) = thiskappasTemp(tIdx);
        end
    end

    runtimeTable(9,ct) = mean(runtimeTemp);
    OATable(9,ct) = mean(OATemp);
    kappaTable(9,ct) = mean(kappaTemp); 

end
 