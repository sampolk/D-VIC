%% SymNMF
% Extracts performances for SymNMF


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'syntheticHSI5149Stretched'};

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
    % ============================== SymNMF ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs),1);
    kappas  = NaN*zeros(length(NNs), 1);
    Cs      = zeros(M*N,length(NNs),numReplicates);

    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:numReplicates
            options.kk = NNs(i);
            C = symnmf_cluster(X, K, options, Idx_NN);
            [~, ~, OAs(i,j), ~, kappas(i,j)]= measure_performance(C, Y);
            Cs(:,i,j) = C;
            disp('SymNMF:')
            disp([ i/length(NNs), j/numReplicates, dataIdx/5])

        end
    end

    save(strcat('SymNMFResults', datasets{dataIdx}), 'OAs', 'kappas','Cs', 'NNs', 'numReplicates')
end

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
%     load(strcat('SymNMFResults', datasets{dataIdx}))
% 
%     % Find optimal hyperparameters
%     [OATable(dataIdx), k] = max(mean(OAs,2));
%     KappaTable(dataIdx) = kappas(k);
%     NN = NNs(k);
%     [~,i] = min(abs(OAs(k,:) - mean(OAs(k,:))));
%     C = Cs(:,k,i);
% 
%     % Save optimal results
%     save(strcat('SymNMFClustering', datasets{dataIdx}), 'C', 'NN')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('SymNMF Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'SymNMF');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('SymNMFPerformances', 'KappaTable', 'OATable')
% 
% close all