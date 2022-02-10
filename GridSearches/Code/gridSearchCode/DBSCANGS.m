%% DBSCAN
% Extracts performances for DBSCAN


%% Grid Search Parameters


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 1:1:99; 

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
    


    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.K_Known = length(unique(Y))-1 + (dataIdx==2); % We subtract 1 since we discard gt labels but not for Jasper Ridge
    Hyperparameters.Tolerance = 1e-8;
    K = length(unique(Y))-1+ (dataIdx==2);

    minPtVals =  floor([3,10:10:3*size(X,2)]);



    % ============================== DBSCAN ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(minPtVals),length(prctiles));
    kappas  = NaN*zeros(length(minPtVals),length(prctiles));
    Cs      = zeros(M*N,length(minPtVals),length(prctiles));

    % Run Grid Search 
    for i = 1:length(minPtVals)
        for j = 1:length(prctiles)
            C = dbscan(X,prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all'),minPtVals(i));

            if length(unique(C)) == length(unique(Y)) 

                [~,~, OAs(i,j), ~, kappas(i,j)]= measure_performance(C, Y);
                Cs(:,i,j) = C;
            end

            disp(['DBSCAN:'])
            disp([i/length(minPtVals), j/length(prctiles), dataIdx/5])
        end
    end

    save(strcat('DBSCANResults', datasets{dataIdx}), 'OAs', 'kappas','Cs', 'prctiles', 'minPtVals')
end
% % 
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
%     load(strcat('DBSCANResults', datasets{dataIdx}))
% 
%     % Find optimal hyperparameters
%     [OATable(dataIdx), k] = max(OAs(:));
%     [i,j] = ind2sub(size(OAs), k);
%     KappaTable(dataIdx) = kappas(i,j);
%     minPts = minPtVals(i);
%     epsilon = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
%     C = Cs(:,i,j);
% 
%     % Save optimal results
%     save(strcat('DBSCANClustering', datasets{dataIdx}), 'C', 'minPts','epsilon')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('DBSCAN Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'DBSCAN');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('DBSCANPerformances', 'KappaTable', 'OATable')
% 
% close all
