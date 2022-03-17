%{

This script runs a grid search over relevant hyperparameter values for the
Diffusion and Volume maximization-based Clustering (D-VIC) on real 
hyperspectral images. This script was used in the following article:

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
NNs = 10:10:100;


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prcts{4} =  65:(100-65)/19:100;
prcts{2} =  65:(100-65)/29:100;
prcts{1} =  88:(100-88)/19:100;
prcts{5} = 5:10:95;

numReplicates = 100;
 
%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','IndianPinesCorrected'};
datasetNames = {'Salinas A',      'Jasper Ridge','Indian Pines'};

for dataIdx =  [5]

    prctiles = prcts{dataIdx};
    if dataIdx == 5
        % Set number of nearest neighbors to use in graph and KDE construction.
        NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];
    end
    

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    Hyperparameters = loadHyperparameters(HSI, datasetNames{dataIdx}, 'D-VIS'); % Load default hyperparameters

    % ============================== DVIS ==============================
 
    % Preallocate memory
    maxOA = 0;
    OAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    kappas  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    Cs      = zeros(M*N,length(NNs), length(prctiles), numReplicates);

    delete(gcp('nocreate'))
    poolObj = parpool;

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)
    
            Hyperparameters.DiffusionNN = NNs(i);
            Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
            Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');

            if poolObj.NumWorkers<6
                delete(gcp('nocreate'));
                poolObj = parpool;
            end
    
            if dataIdx ==5
                Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
            else
                Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
            end 

            parfor k = 1:numReplicates 

                % Graph decomposition
                G = extractGraph(X, Hyperparameters, Idx_NN, Dist_NN);
                
                % KDE Computation
                density = KDE(Dist_NN, Hyperparameters);
        
                % Spectral Unmixing Step
                pixelPurity = compute_purity(X,Hyperparameters);
    
                if G.EigenVals(2)<1

                    [Clusterings, ~] = MLUND(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
                    [ OAs(i,j, k), kappas(i,j, k), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('JasperRidge', datasets{dataIdx}));
                    Cs(:,i,j, k) = Clusterings.Labels(:,tIdx);
                     
               end
    
                disp(['DVIS: '])
                disp([i/length(NNs), j/length(prctiles), k/numReplicates, maxOA])
            end
            maxOA = max(mean(OAs,3), [],'all');
            
            save(strcat('DVISResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA')
        end
    end

end