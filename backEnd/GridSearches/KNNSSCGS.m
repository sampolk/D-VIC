%% knnssc
% Extracts performances for knnssc


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
alpha = 10;

%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

for dataIdx =  1:4

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    Hyperparameters = loadHyperparameters(HSI, datasetNames{dataIdx}, 'KNN-SSC'); % Load default hyperparameters
    
    % ============================== knnssc ==============================
 
    % Preallocate memory
    OAs     = NaN*zeros(length(NNs),1);
    kappas  = NaN*zeros(length(NNs), 1);
    Cs      = zeros(M*N,length(NNs));

    % Run Grid Searches
    for i = 1:length(NNs)
 
        W = sparse(knn_SSC( X', alpha, NNs(i), Idx_NN));
        W = (W+W')./2;   %  (W+W)./2 forces symmetry

        % First n_eigs eigenpairs of transition matrix D^{-1}W:
        [V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs'); 
            
        if flag
            % Didn't converge. Try again with larger subspace dimension.
            [V,~, flag] = eigs(spdiags(1./sum(W)',0,n,n)*W, min(K,10), 'largestabs',  'SubspaceDimension', max(4*min(K,10),40));  
        end         

        EigenVecs_Normalized = real(V(:,1:min(K,10))./vecnorm(V(:,1:min(K,10)),2,2));
        C = kmeans(EigenVecs_Normalized, K);

        [ OAs(i), kappas(i)] = calcAccuracy(Y, C, ~strcmp('Jasper Ridge', datasets{dataIdx}));

        thisK = length(unique(C));
        Cs(:,i) = C;

        disp(['KNNSSC: '])
        disp([i/length(NNs),  thisK, max(OAs)])
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