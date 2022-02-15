%% GMM
% Extracts performances for GMM+PCA

numReplicates = 10;

%% Run GMM
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected', 'syntheticHSI5149Stretched'};


%% Grid searches

for dataIdx = 6 
    clear C

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

    % ============================ GMM+PCA ============================

    % Run over 10 trials
    OATemp = zeros(numReplicates,1);
    KappaTemp = zeros(numReplicates,1);
    Cs = zeros(M*N,numReplicates);
    for i = 1:numReplicates
        % We run the same code (at most) 10 times and keep the clustering
        % that is the first to converge. 
        maxNumTries = 10;
        options = statset('MaxIter',200);
        iter = 1;
        keepRunning = true;
        while iter<maxNumTries && keepRunning 
            try
                tic
                [COEFF, SCORE, ~, ~, pctExplained] = pca(X);
                nPCs = find(cumsum(pctExplained)>99,1,'first');
                XPCA = SCORE(:,1:nPCs);
    
                C = cluster(fitgmdist(XPCA,K, 'Options', options), XPCA);
                keepRunning = false;
            catch
            end
            iter = iter+1;
        end
        subtractNum = 1;
        while ~exist('C') && subtractNum<nPCs-1
            % Too many PCs
            options = statset('MaxIter',1000); 
            try
                C = cluster(fitgmdist(XPCA(:,1:end-subtractNum),K, 'Options', options), XPCA(:,1:end-subtractNum));
            catch
                subtractNum = subtractNum + 1;
            end
        end
        if  ~exist('C')
            C = randi(K, M*N,1);
        end

        [~, ~, OATemp(i), ~, KappaTemp(i)]= measure_performance(C, Y);
        Cs(:,i) = C;

        disp(['GMMPCA: '])
        disp([ i/numReplicates, dataIdx/5])
    end
    
    % Average performance across 10 runs
    OA = mean(OATemp);
    Kappa = mean(KappaTemp);

    % Save "centroid" clustering
    [~,i] = min(abs(OA-OATemp));
    C = Cs(:,i);
    save(strcat('GMMResults', datasets{dataIdx}), 'C', 'OA', 'Kappa')

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
%     load(strcat('GMMResults', datasets{dataIdx}))
% 
%     % Performances
%     OATable(dataIdx) = OA;
%     KappaTable(dataIdx) = Kappa;
% 
%     % Save optimal results
%     save(strcat('GMMClustering', datasets{dataIdx}), 'C')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('GMM+PCA Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'GMM');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('GMMPerformances', 'KappaTable', 'OATable')
% 
% close all