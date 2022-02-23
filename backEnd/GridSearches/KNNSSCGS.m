%% knnssc
% Extracts performances for knnssc


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
alpha = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  11

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    if dataIdx <7
        load(datasets{dataIdx})
        HSI = reshape(X,M,N,size(X,2));
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
    if ~(dataIdx == 2)
        K = K-1;  % We subtract 1 since we discard gt labels
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
    Hyperparameters.K_Known = K;
    
    % ============================== knnssc ==============================
 
    % Preallocate memory
    OAs     = NaN*zeros(length(NNs),1);
    kappas  = NaN*zeros(length(NNs), 1);
    Cs      = zeros(M*N,length(NNs));

    % Run Grid Searches
    parfor i = 1:length(NNs)
 
        W = sparse(knn_SSC( X', alpha, NNs(i), Idx_NN));
        W = (W+W')./2;   %  (W+W)./2 forces symmetry

        % First n_eigs eigenpairs of transition matrix D^{-1}W:
        [V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs'); 
            
        if flag
            % Didn't converge. Try again with larger subspace dimension.
            [V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs',  'SubspaceDimension', max(4*min(K,10),40));  
        end         

        EigenVecs_Normalized = V(:,1:min(K,10))./sqrt(sum(V(:,1:min(K,10)).^2,2));
        C = kmeans(EigenVecs_Normalized, K);

        if dataIdx == 2
            C = alignClusterings(Y,C);
            confMat = confusionmat(Y,C);

            OAs(i) = sum(diag(confMat)/length(C)); 

            p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
            kappas(i)=(OAs(i)-p)/(1-p);
        else
            [~,~, OAs(i), ~, kappas(i)]= measure_performance(C, Y);
        end
        thisK = length(unique(C));
        Cs(:,i) = C;

        disp(['KNNSSC: '])
        disp([i/length(NNs),  thisK])
    end
        save(strcat('KNNSSCResults', datasets{dataIdx}), 'OAs', 'kappas','Cs', 'NNs')

end

% 
% %% Visualize and save table
% clear
% close all 
% datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected'};
% 
% OATable = zeros(5,1);
% KappaTable = zeros(5,1);
% for dataIdx = [1,2, 5]
%     
%      % Load data
%     load(datasets{dataIdx})
% 
%     % Load results
%     load(strcat('KNNSSCResults', datasets{dataIdx}))
% 
%     % Find optimal hyperparameters
%     [OATable(dataIdx), k] = max(OAs);
%     KappaTable(dataIdx) = kappas(k);
%     NN = NNs(k);
%     C = Cs(:,k);
% 
%     % Save optimal results
%     save(strcat('KNNSSCClustering', datasets{dataIdx}), 'C', 'NN')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('KNN-SSC Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'KNNSSC');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('KNNSSCPerformances', 'KappaTable', 'OATable')
% 
% close all