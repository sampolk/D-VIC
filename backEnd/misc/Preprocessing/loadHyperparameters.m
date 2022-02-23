function Hyperparameters = loadHyperparameters(HSI, HSIName, AlgName)

[M,N,D] = size(HSI);
X = reshape(HSI,M*N,D);

if strcmp(AlgName, 'D-VIS')

    [~,Dist_NN] = knnsearch(X,X,'K', 1000);
    if strcmp(HSIName, 'Synthetic HSI')
        NN  = 130;
        pct = 2.36842105263158;
        K = 5;
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 30;
        pct = 43.4210526315789;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 20;
        pct = 98.1578947368421;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 50;
        pct = 100;
        K = 16;
    elseif strcmp(HSIName, 'Pavia Subset')
        NN = 30;
        pct = 43.4210526315789;
        K = 5;
    end

    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.Tolerance = 1e-8;
    Hyperparameters.K_Known = K; 
    Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
    Hyperparameters.EndmemberParams.NumReplicates = 100;
    Hyperparameters.EndmemberParams.K = hysime(X');
    Hyperparameters.DiffusionNN = NN;
    Hyperparameters.DensityNN = NN; % must be ≤ 1000
    Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), pct, 'all');

elseif strcmp(AlgName, 'LUND')

    [~,Dist_NN] = knnsearch(X,X,'K', 1000);
    if strcmp(HSIName, 'Synthetic HSI')
        NN  = 50;
        pct = 5;
        K = 5;
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 40;
        pct = 15;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 20;
        pct = 25;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 40;
        pct = 65;
        K = 16;
    elseif strcmp(HSIName, 'Pavia Subset')
        NN = 10;
        pct = 5;
        K = 5;
    end

    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.Tolerance = 1e-8;
    Hyperparameters.K_Known = K; 
    Hyperparameters.DiffusionNN = NN;
    Hyperparameters.DensityNN = NN; % must be ≤ 1000
    Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), pct, 'all');

elseif strcmp(AlgName, 'SymNMF')

    if strcmp(HSIName, 'Synthetic HSI')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 500;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 600;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 600;
    elseif strcmp(HSIName, 'Pavia Subset')
        NN = 500;
    end

    Hyperparameters.DiffusionNN = NN;

elseif strcmp(AlgName, 'SC')

    if strcmp(HSIName, 'Synthetic HSI')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 20;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 20;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 100;
        K = 16;
    elseif strcmp(HSIName, 'Pavia Subset')
        NN = 20;
        K = 5;
    end

    Hyperparameters.DiffusionNN = NN;
    Hyperparameters.NEigs = min(K, 10); 
    Hyperparameters.SpatialParams.ImageSize = [M,N];    

elseif strcmp(AlgName, 'KNN-SSC')

    if strcmp(HSIName, 'Synthetic HSI')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 10;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 500;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 700;
    elseif strcmp(HSIName, 'Pavia Subset')
        NN = 900;
    end

    Hyperparameters.DiffusionNN = NN;

else 
    Hyperparameters = [];
end
