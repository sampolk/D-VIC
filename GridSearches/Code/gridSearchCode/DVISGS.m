%% DVIS
% Extracts performances for DVIS

%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = 10:10:100;

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 

for k = 1:13
    prcts{k} = 5:10:95;
end
prcts{1} =  45:(100-45)/19:100;
prcts{5} = 88:(100-88)/19:100;
% prcts{5} = 5:(45-5)/19:45;
prcts{13} = 73:(100-73)/19:100;

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  8
    prctiles = prcts{dataIdx};

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
        HSI = HSI(101:400,241:300,:);
        GT = GT(101:400,241:300);
    elseif dataIdx == 8
        load('PaviaU')
        HSI = HSI(498:end,1:100,:);
        GT = GT(498:end,1:100);        
    elseif dataIdx == 9
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

    [M,N] = size(GT);
    D = size(X,2);
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
    if dataIdx >= 12 && ~(dataIdx == 2)
        K = length(unique(Y))-1;
    else
        K = length(unique(Y));
    end
    Hyperparameters.K_Known = K; % We subtract 1 since we discard gt labels

    % ============================== DVIS ==============================

    % Preallocate memory
    UAcc    = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    AAcc    = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    OAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    kappas  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    Cs      = zeros(M*N,length(NNs), length(prctiles), numReplicates);

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)
            for k = 1:numReplicates

                Hyperparameters.DiffusionNN = NNs(i);
                Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
                Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
                if dataIdx >=12
                    Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
                else
                    Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
                end
                Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
                Hyperparameters.EndmemberParams.NumReplicates = 100;
                

                tic
                [pixelPurity, U, A] = compute_purity(X,Hyperparameters);
                toc
                if dataIdx == 12
                    UAcc(i,j,k) = norm(U-U_GT')./norm(U_GT');
                    AAcc(i,j,k) = norm(A-A_GT)./norm(A_GT');
                end

                density = KDE_large(Dist_NN, Hyperparameters);
                [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

                if G.EigenVals(2)<1
                    Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));

                    [~,~, OAs(i,j,k), ~, kappas(i,j,k), tIdx]= measure_performance(Clusterings, Y);
                    C =  Clusterings.Labels(:,tIdx);
                    Cs(:,i,j,k) = C;
                end
    
                disp(['DVIS: '])
                disp([i/length(NNs), j/length(prctiles), k/numReplicates, currentPerf])

            end
            currentPerf = max(nanmean(OAs,3),[],'all');
        end 
        [maxOA, k] = max(nanmean(OAs,3),[],'all');
        [l,j] = ind2sub(size(mean(OAs,3)), k);
        stdOA = nanstd(squeeze(OAs(l,j,:)));
        save(strcat('DVISResults', datasets{dataIdx}, '1ManyAVMAX'),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA', "UAcc", "AAcc")

    end

end

% 
% %% 
% OAs = zeros(length(NNs), length(prctiles), numReplicates);
% kappas  = zeros(length(NNs), length(prctiles), numReplicates);
% CAgg = zeros(length(X), length(NNs), length(prctiles), numReplicates);
% 
% % Run Grid Searches
% for i = 1:length(NNs)
%     for j = 1:length(prctiles) 
%         for k = 1:numReplicates
%             C = Cs(:,i,j,k);                    
%             [~,~, OAs(i,j,k), ~, kappas(i,j,k), tIdx]= measure_performance(C, Y);
%         end
% 
%         OAs(i,j,:) = OAtemp;
%         kappas(i,j,:) = kappatemp;
% 
%         if mean(squeeze( OAs(i,j,:)))== max(mean(OAs, 3), [],'all')
% 
%             OA = mean(OAtemp);
%             kappa = mean(kappatemp);
%             [~,k] = min(abs(OA - OAtemp));
%             C = Cs(:,i,j,k);
%             NN = NNs(i);
%             sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all')
%             save('optimalJasperRidgeClustering', 'C', 'OA', 'AA', 'kappa','NN')
%         end
% 
% 
%     end
% end
%  