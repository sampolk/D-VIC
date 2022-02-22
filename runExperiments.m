%% Run Experiments

%% Preallocate memory

OATable = zeros(9,4);

kappaTable = zeros(9,4);

%% Run Experiments

% Load optimal hyperparameters
load('results', 'hyperparameters')
 
%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  [1,2, 5, 11, 13]

    prctiles = prcts{dataIdx};
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
    clear Botswana Botswana_gt  pavia pavia_gt uniqueClass k 
 
    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.Tolerance = 1e-8;
    Hyperparameters.K_Known = K; % We subtract 1 since we discard gt labels

    % ============================== DVIS ==============================

    % Preallocate memory
    UAcc    = NaN*zeros(length(NNs), length(prctiles));
    AAcc    = NaN*zeros(length(NNs), length(prctiles));
    OAs     = NaN*zeros(length(NNs), length(prctiles));
    kappas  = NaN*zeros(length(NNs), length(prctiles));
    Cs      = zeros(M*N,length(NNs), length(prctiles));

    tic 
    pixelPurity = zeros(M*N,1);
    for k = 1:numReplicates

       
        if dataIdx >=12
            Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
        else
            Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
        end
        Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
        Hyperparameters.EndmemberParams.NumReplicates = 100;
        
        [purityTemp, U, A] = compute_purity(X,Hyperparameters);
        pixelPurity = pixelPurity + purityTemp/numReplicates;
        disp(['Purity:', 100*k/numReplicates])
    end
% end
% %%
% for dataIdx = 1
    
%     prctiles = 5:10:95;

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)

            Hyperparameters.DiffusionNN = NNs(i);
            Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
            Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
%             if dataIdx == 12
%                 UAcc(i,j) = norm(U-U_GT')./norm(U_GT');
%                 AAcc(i,j) = norm(A-A_GT)./norm(A_GT');
%             end

            density = KDE_large(Dist_NN, Hyperparameters);
            [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

            if G.EigenVals(2)<1
                Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));

                if dataIdx == 2
                    
                    OAsTemp = zeros(length(Clusterings.K),1);
                    kappasTemp = zeros(length(Clusterings.K),1);
                    for t = 1:length(Clusterings.K)
                        C = alignClusterings(Y,Clusterings.Labels(:,t));
                        confMat = confusionmat(Y,C);

                        OAsTemp(t) = sum(diag(confMat)/length(C)); 

                        p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
                        kappasTemp(t)=(OAsTemp(t)-p)/(1-p);
                    end

                    [OAs(i,j), tIdx] = max(OAsTemp);
                    kappas(i,j) = kappasTemp(tIdx);

                elseif dataIdx == 13

                    OAsTemp = zeros(length(Clusterings.K),1);
                    kappasTemp = zeros(length(Clusterings.K),1);
                    for t = 1:length(Clusterings.K)
                        C = alignClusterings(Y(Y>1)-1, Clusterings.Labels(Y>1,t));
                        confMat = confusionmat(Y(Y>1)-1,C);
                        OAsTemp(t) = sum(diag(confMat)/length(C));
                        p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
                        kappasTemp(t)=(OAsTemp(t)-p)/(1-p);
                    end

                    [OAs(i,j), tIdx] = max(OAsTemp);
                    kappas(i,j) = kappasTemp(tIdx);
                else
                    [~,~, OAs(i,j), ~, kappas(i,j), tIdx]= measure_performance(Clusterings, Y);
                end
                C =  Clusterings.Labels(:,tIdx);
                Cs(:,i,j) = C;
                
                currentPerf = OAs(i,j);
                [maxOA, k] = max(OAs,[],'all');
    
                if currentPerf >= maxOA
                
                    [l,j] = ind2sub(size(mean(OAs,3)), k);
                    stdOA = nanstd(squeeze(OAs(l,j,:)));
                    save(strcat('DVISResults', datasets{dataIdx}, 'ManyAVMAXRefined1'),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA', "UAcc", "AAcc")
                end
            end

            disp(['DVIS: '])
            disp([i/length(NNs), j/length(prctiles), maxOA])
        end

    end
    save(strcat('DVISResults', datasets{dataIdx}, 'ManyAVMAX'),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA', "UAcc", "AAcc")

end
 