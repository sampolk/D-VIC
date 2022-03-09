%%  ===================== Load and Preprocess Data ======================
[X,M,N,D,HSI,GT,Y,n, K] = loadHSI('Jasper Ridge');
[Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];  
Hyperparameters = loadHyperparameters(HSI, 'Jasper Ridge', 'D-VIS'); % Load default hyperparameters

%% Analyze Previous Grid Search
% Load coarse grid search
load(strcat('DVISResults', 'JasperRidge'))

nNN = length(NNs);
npct = length(prctiles);
OAs(isnan(OAs)) = 0;
OAsAvg = reshape(mean(OAs,3), nNN*npct, 1);
OAsAvg(isnan(OAsAvg)) = 0;
[OAsAvg, sorting] = sort(OAsAvg, 'descend');
nCandidates = sum(OAsAvg>0.869);

%% 

nReplicates = 100;

delete(gcp('nocreate'))
poolObj = parpool;
Cs = zeros(n, nCandidates,nReplicates);
OAs = zeros(nCandidates,nReplicates);
HPs = zeros(nCandidates, 2);
bestSoFar = 0;

for k = 1:nCandidates

    bestSoFar = max(mean(OAs,2));

    % Extract optimal hyperparameters for this grid node
    [i,j] = ind2sub([nNN, npct], sorting(k));
    NN = NNs(i); 
    pct = prctiles(j);
    HPs(k,:) = [NN, pct];

    Hyperparameters.DiffusionNN = NN;
    Hyperparameters.DensityNN = NN; % must be ≤ 1000
    Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), pct, 'all');

    if poolObj.NumWorkers<6
        delete(gcp('nocreate'));
        poolObj = parpool;
    end

    ms = zeros(nReplicates,1);
    parfor k1 = 1:nReplicates
        ms(k1) = hysime(X'); % compute hysime to get best estimate for number of endmembers 
    end
    Hyperparameters.EndmemberParams.K = mode(ms);  % Most frequent estimate of HySime across nReplicates runs.

    parfor k1 = 1:nReplicates

        % Graph decomposition
        G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
            
        % KDE Computation
        density = KDE_large(Dist_NN, Hyperparameters);

        % Spectral Unmixing Step
        pixelPurity = compute_purity(X,Hyperparameters);


        [Clusterings, ~] = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
        [ OAs(k,k1), kappas(k,k1), tIdx] = calcAccuracy(Y, Clusterings, 0);
        Cs(:,k,k1) = Clusterings.Labels(:,tIdx);
        
        disp([k/nCandidates, k1/nReplicates, bestSoFar])
    end
end
                            
save('JasperRidgeRefinedSearch', 'OAs', 'Cs', 'HPs')
    



 