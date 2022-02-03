%% DBSCAN
% Extracts performances for DBSCAN


%% Grid Search Parameters


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 1:1:99; 

datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'syntheticHSI5149Stretched'};


%% Grid searches

for dataIdx = 6 

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    load(datasets{dataIdx})

    % If Salinas A, we add gaussian noise and redo nearest neighbor searches. 
    if dataIdx == 5
        X = X + randn(size(X)).*10^(-7);
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end

    if dataIdx == 6
        load(datasets{dataIdx})
        X = X./vecnorm(X,2,2);
        HSI = reshape(X,M,N,D);
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);

        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end

    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.K_Known = length(unique(Y))-1; % We subtract 1 since we discard gt labels
    Hyperparameters.Tolerance = 1e-8;
    K = length(unique(Y))-1;

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
