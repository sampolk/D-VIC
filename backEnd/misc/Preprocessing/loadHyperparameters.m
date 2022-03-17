function Hyperparameters = loadHyperparameters(HSI, HSIName, AlgName)

if strcmp(HSIName, 'Synthetic Data') % Not an HSI 
    [n,D] = size(HSI);
    M=n; N=1;
    X = HSI;
else
    [M,N,D] = size(HSI);
    X = reshape(HSI,M*N,D);
end

if strcmp(AlgName, 'D-VIC')

    [~,Dist_NN] = knnsearch(X,X,'K', 1000);
    if strcmp(HSIName, 'Synthetic Data')
        NN  = 320;
        pct = 10.5;
        K = 3;
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 30;
        pct = 96.3157894736842;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 20;
        pct = 92.758620689655170;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 40;
        pct = 76.0526315789474;
        K = 16;
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
    if strcmp(HSIName, 'Synthetic Data')
        NN  = 140;
        pct = 0.5;
        K = 3;
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 40;
        pct = 5;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 40;
        pct = 75;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 40;
        pct = 65;
        K = 16;
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

    if strcmp(HSIName, 'Synthetic Data')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 400;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 130;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 600;
    end

    Hyperparameters.DiffusionNN = NN;

elseif strcmp(AlgName, 'SC')

    if strcmp(HSIName, 'Synthetic Data')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 50;
        K = 6;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 10;
        K = 4;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 100;
        K = 16;
    end

    Hyperparameters.DiffusionNN = NN;
    Hyperparameters.NEigs = min(K+1, 10); 
    Hyperparameters.SpatialParams.ImageSize = [M,N];    

elseif strcmp(AlgName, 'KNN-SSC')

    if strcmp(HSIName, 'Synthetic Data')
        error('This algorithm is not supported for Synthetic Dataset')
    elseif strcmp(HSIName, 'Salinas A')
        NN  = 10;
    elseif strcmp(HSIName, 'Jasper Ridge')
        NN  = 50;
    elseif strcmp(HSIName, 'Indian Pines')
        NN = 50;
    end

    Hyperparameters.DiffusionNN = NN;

else 
    Hyperparameters = [];
end
