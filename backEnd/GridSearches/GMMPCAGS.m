%{

This script runs the Gaussian Mixture Model Clustering (GMM) on real HSI 
data. This script was used in the following article:

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

(c) Copyright Sam L. Polk, Tufts University, 2022.

To run this script, real hyperspectral image data (Salinas A, Indian Pines, 
& Jasper Ridge) must be downloaded from the following links:

    - http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    - https://rslab.ut.ac.ir/data

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}
%% Run GMM
datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};
numReplicates = 100;

for dataIdx =  1:4

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    Hyperparameters = loadHyperparameters(HSI, datasetNames{dataIdx}, 'SymNMF'); % Load default hyperparameters

    % ============================ GMM+PCA ============================

    % Run over 10 trials
    OATemp = zeros(numReplicates,1);
    KappaTemp = zeros(numReplicates,1);
    Cs = zeros(M*N,numReplicates);
    parfor i = 1:numReplicates

        C = evaluateGMMPCA(X,K);
        [ OATemp(i), KappaTemp(i)] = calcAccuracy(Y, C, ~strcmp('Jasper Ridge', datasets{dataIdx}));
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
