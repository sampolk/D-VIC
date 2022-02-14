%% Grid searches
datasets = {'IndianPinesCorrected', 'SalinasACorrected', 'PaviaSubset1', 'syntheticHSI5149Stretched'};

for dataIdx =  1:3

    % ===================== Load and Preprocess Data ======================
    
    clear HSI X Y GT paviaU paviaU_gt M N Idx_NN Dist_NN D
    % Load data
    if dataIdx <3
        load(datasets{dataIdx})
    end


    % If Salinas A, we add gaussian noise and redo nearest neighbor searches. 
    if dataIdx == 2
        X = X + randn(size(X)).*10^(-7);
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end 
    if dataIdx == 3
        load('PaviaU')
        load('PaviaU_gt.mat')
        HSI = double(paviaU(101:400,241:300,:));
        GT = double(paviaU_gt(101:400,241:300));
    end

    [M,N] = size(GT);
    D = size(HSI,3);
    X = reshape(HSI,M*N,D);
    
    if dataIdx == 3 
        [X, M,N, Idx_NN, Dist_NN] = knn_store(HSI, 900);  
    end



    newGT = zeros(size(GT));
    uniqueClass = unique(GT);
    K = length(uniqueClass);
    for k = 1:K
    newGT(GT==uniqueClass(k)) = k;
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
    Hyperparameters.K_Known = K; 

    save(strcat(datasets{dataIdx}, 'Preprocessed'), 'HSI' ,'X', 'Y', 'GT',  'M', 'N', 'Idx_NN', 'Dist_NN', 'D')

end
