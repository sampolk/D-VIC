%% SC
% Extracts performances for SC


%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

numReplicates = 10;

%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

for dataIdx =  1:4

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    Hyperparameters = loadHyperparameters(HSI, datasetNames{dataIdx}, 'SC'); % Load default hyperparameters

    % ============================== SC ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs),numReplicates);
    kappas  = NaN*zeros(length(NNs), numReplicates);
    Cs      = zeros(M*N,length(NNs), numReplicates);

    % Run Grid Searches
    for i = 1:length(NNs)

        Hyperparameters.DiffusionNN = NNs(i);
        Hyperparameters.SpatialParams.ImageSize = [M,N];
        [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

        if G.EigenVals(2)<1

            for j = 1:numReplicates
                C = SpectralClustering(G,K);

                [ OAs(i,j), kappas(i,j), tIdx] = calcAccuracy(Y, C, ~strcmp('Jasper Ridge', datasets{dataIdx}));
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