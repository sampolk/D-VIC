%% H2NMF
% Extracts performances for H2NMF

%% Grid searches
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

    % ============================== H2NMF ==============================
     
    C = hierclust2nmf(X'-min(X,[],'all'),K);
    [~, ~, OA , ~, Kappa]= measure_performance(C, Y);

    disp(['H2NMF: '])
    disp([dataIdx/5])


    save(strcat('H2NMFResults', datasets{dataIdx}), 'C', 'OA', 'Kappa')
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
%     load(strcat('H2NMFResults', datasets{dataIdx}))
% 
%     % Performances
%     OATable(dataIdx) = OA;
%     KappaTable(dataIdx) = Kappa;
% 
%     % Save optimal results
%     save(strcat('H2NMFClustering', datasets{dataIdx}), 'C')
% 
%     % Visualize clustering
%     h = figure;
%     eda(C, 0, Y)
%     title('H2NMF Clustering', 'interpreter', 'latex', 'FontSize', 16)
% 
%     % Save Figure
%     fileName = strcat(datasets{dataIdx}, 'H2NMF');
%     save(fileName, 'C')
%     savefig(h, fileName)
%     saveas(h, fileName, 'epsc')   
% 
% end
% 
% save('H2NMFPerformances', 'KappaTable', 'OATable')
% 
% close all
