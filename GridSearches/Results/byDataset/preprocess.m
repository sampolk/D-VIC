%% 

datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaUSubset1', 'PaviaUSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  7:11

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    if dataIdx <7
        load(datasets{dataIdx})
    end
    if dataIdx == 2
        X = knn_store(reshape(X,M,N,size(X,2)), 900);
    end

    % If Salinas A, we add gaussian noise and redo nearest neighbor searches. 
    if dataIdx == 5
        X = X + randn(size(X)).*10^(-7);
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end 
    if dataIdx == 6

        % Perfor knnsearch for new datasets
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);

    end
    if dataIdx == 7
        load('PaviaU')
        load('PaviaU_gt.mat')
        HSI = double(paviaU(101:400,241:300,:));
        GT = double(paviaU_gt(101:400,241:300));
        HSI = HSI+ randn(size(X)).*10^(-7);
    elseif dataIdx == 8
        load('PaviaU')
        load('PaviaU_gt.mat')
        HSI = double(paviaU(498:end,1:100,:));
        GT = double(paviaU_gt(498:end,1:100));        
        HSI = HSI+ randn(size(X)).*10^(-7);
    elseif dataIdx == 9
        load('Botswana.mat')
        load('Botswana_gt.mat')
        HSI = Botswana(285:507, 204:253,:);
        GT = Botswana_gt(285:507, 204:253);
        HSI = HSI+ randn(size(X)).*10^(-7);
    elseif dataIdx == 10
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(101:250,201:350,:);
        GT = pavia_gt(101:250,201:350);
        HSI = HSI+ randn(size(X)).*10^(-7);
    elseif dataIdx == 11
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(201:400, 430:530,:);
        GT = pavia_gt(201:400, 430:530);
        HSI = HSI+ randn(size(X)).*10^(-7);
    end

    [M,N] = size(GT);
    D = size(HSI,3);
    X = reshape(HSI,M*N,D);
    
    if dataIdx >= 6  
        [X, M,N, Idx_NN, Dist_NN] = knn_store(HSI, 900); % 
    end



    newGT = zeros(size(GT));
    uniqueClass = unique(GT);
    K = length(uniqueClass);
    for k = 1:K
    newGT(GT==uniqueClass(k)) = k;
    end
    if dataIdx == 2
        newGT = newGT+1;
    end
    Y = reshape(newGT,M*N,1);
    GT = newGT;
    

    clear Botswana Botswana_gt  pavia pavia_gt uniqueClass k 
 
    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.Tolerance = 1e-8;
    if ~(dataIdx == 2)
        K = length(unique(Y))-1;
    else
        K = length(unique(Y));
    end
    Hyperparameters.K_Known = K; % We subtract 1 since we discard gt labels

    save(datasets{dataIdx}, 'X', 'HSI', 'Y', 'GT', 'M', 'N', 'D', 'Idx_NN', 'Dist_NN', 'K')


end