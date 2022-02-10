%% lund
% Extracts performances for lund


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 5:10:95; 

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'KSCSubset', 'PaviaSubset1', 'PaviaSubset2', 'Botswana', 'PaviaCenterSubset1',  'PaviaCenterSubset2', 'syntheticHSI5050', 'syntheticHSI5149Stretched'};

for dataIdx =  8

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
    


    % ============================== lund ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs), length(prctiles));
    maxOA   = 0 ;
    kappas  = NaN*zeros(length(NNs), length(prctiles));
    Cs      = NaN*zeros(M*N,length(NNs), length(prctiles));

    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)

            Hyperparameters.DiffusionNN = NNs(i);
            Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
            Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');

            [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
            density = KDE_large(Dist_NN, Hyperparameters);

            if G.EigenVals(2)<1 
                Clusterings = MLUND_large(X, Hyperparameters, G, density);
    
                [~,~, OAs(i,j), ~, kappas(i,j), tIdx]= measure_performance(Clusterings, Y);
                C =  Clusterings.Labels(:,tIdx);
                Cs(:,i,j) = C;
                if OAs(i,j)>maxOA
                    maxOA = OAs(i,j);
                end
            end
    
            disp('LUND:')
            disp(datasets{dataIdx})
            disp([i/length(NNs), j/length(prctiles), maxOA])
        end
    end

    save(strcat('LUNDResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'maxOA')
end

clc 
'done'

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
%     load(strcat('LUNDResults', datasets{dataIdx}))
% 
%     % Find optimal hyperparameters
%     [OATable(dataIdx), k] = max(OAs(:));
%     [i,j] = ind2sub(size(OAs), k);
%     KappaTable(dataIdx) = kappas(i,j);
%     NN = NNs(i);
%     sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
%     C = Cs(:,i,j);
% 
%     % Save optimal results
%     save(strcat('LUNDClustering', datasets{dataIdx}), 'C', 'NN','sigma0')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('LUND Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'LUND');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('LUNDPerformances', 'KappaTable', 'OATable')
% 
% close all
