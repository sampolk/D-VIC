%% SC
% Extracts performances for SC


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  7

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
    elseif dataIdx == 8
        load('PaviaU')
        load('PaviaU_gt.mat')
        HSI = double(paviaU(498:end,1:100,:));
        GT = double(paviaU_gt(498:end,1:100));        
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
    if dataIdx >= 12 && ~(dataIdx == 2)
        K = length(unique(Y))-1;
    else
        K = length(unique(Y));
    end
    Hyperparameters.K_Known = K; % We subtract 1 since we discard gt labels


    % ============================== SC ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs),numReplicates);
    kappas  = NaN*zeros(length(NNs), numReplicates);
    Cs      = zeros(M*N,length(NNs), numReplicates);

    % Run Grid Searches
    for i = 1:length(NNs)

        Hyperparameters.DiffusionNN = NNs(i);
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

        if G.EigenVals(2)<1

            for j = 1:numReplicates
                C = SpectralClustering(G,K);
                [~,~, OAs(i,j), ~, kappas(i,j)]= measure_performance(C, Y);
                Cs(:,i,j) = C;
        
                disp('SC')
                disp([i/length(NNs), j/numReplicates, dataIdx/5])
            end
        end
    end

    save(strcat('SCResults', datasets{dataIdx}), 'OAs', 'kappas','Cs', 'NNs', 'numReplicates')
end
% 
% 
% %% Visualize and save table
% clear
% close all 
% datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected'};
% 
% OATable = zeros(5,1);
% KappaTable = zeros(5,1);
% for dataIdx = 1:5
%     
%      % Load data
%     load(datasets{dataIdx})
% 
%     % Load results
%     load(strcat('SCResults', datasets{dataIdx}))
% 
%     % Find optimal hyperparameters
%     [OATable(dataIdx), k] = max(mean(OAs,2));
%     KappaTable(dataIdx) = kappas(k);
%     NN = NNs(k);
%     [~,i] = min(abs(OAs(k,:) - mean(OAs(k,:))));
%     C = Cs(:,k,i);
% 
%     % Save optimal results
%     save(strcat('SCClustering', datasets{dataIdx}), 'C', 'NN')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('SC Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'SC');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('SCPerformances', 'KappaTable', 'OATable')
% 
% close all
