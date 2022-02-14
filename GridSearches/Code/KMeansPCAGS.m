%% KMeans
% Extracts performances for K-Means+PCA

numReplicates = 10;

%% Run K-Means
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

    % ============================ K-Means+PCA ============================

    % Run over 10 trials
    OATemp = zeros(numReplicates,1);
    KappaTemp = zeros(numReplicates,1);
    Cs = zeros(M*N,numReplicates);
    parfor i = 1:numReplicates
        [~, SCORE, ~, ~, pctExplained] = pca(X);
        nPCs = find(cumsum(pctExplained)>99,1,'first');
        XPCA = SCORE(:,1:nPCs);
        C = kmeans(XPCA, K, 'MaxIter', 200);
        [~, ~, OATemp(i), ~, KappaTemp(i)]= measure_performance(C, Y);
        Cs(:,i) = C;

        disp(['K-MeansPCA: '])
        disp([i/numReplicates, dataIdx/5])
    end
    
    % Average performance across 10 runs
    OA = mean(OATemp);
    Kappa = mean(KappaTemp);

    % Save "centroid" clustering
    [~,i] = min(abs(OA-OATemp));
    C = Cs(:,i);
    save(strcat('KMeansPCAResults', datasets{dataIdx}), 'C', 'OA', 'Kappa')

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
%     load(strcat('KMeansPCAResults', datasets{dataIdx}))
% 
%     % Performances
%     OATable(dataIdx) = OA;
%     KappaTable(dataIdx) = Kappa;
% 
%     % Save optimal results
%     save(strcat('KMeansPCAClustering', datasets{dataIdx}), 'C')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('$K$-Means+PCA Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'KMeansPCA');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('KMeansPCAPerformances', 'KappaTable', 'OATable')
% 
% close all
