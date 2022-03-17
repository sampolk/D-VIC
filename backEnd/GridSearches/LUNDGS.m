%{
This script runs a grid search over relevant hyperparameter values for the
Learning by Unsupervised Nonlinear Diffusion (LUND) on real hyperspectral 
images. This script was used in the following article:

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

To run this script, real hyperspectral image data (Salinas A, Indian Pines, 
& Jasper Ridge) must be downloaded from the following links:

    - http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    - https://rslab.ut.ac.ir/data

(c) Copyright Sam L. Polk, Tufts University, 2022.
%}
%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 5:10:95;

numReplicates = 10;

%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','IndianPinesCorrected'};
datasetNames = {'Salinas A',      'Jasper Ridge','Indian Pines'};

for dataIdx =  1:3


    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    Hyperparameters = loadHyperparameters(HSI, datasetNames{dataIdx}, 'LUND'); % Load default hyperparameters

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

            [G,W] = extractGraph(X, Hyperparameters, Idx_NN, Dist_NN);
            density = KDE(Dist_NN, Hyperparameters);

            if G.EigenVals(2)<1 
                Clusterings = MLUND(X, Hyperparameters, G, density);

                [ OAs(i,j), kappas(i,j), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', datasets{dataIdx}));

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
        save(strcat('LUNDResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'maxOA')
    end

end
 

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