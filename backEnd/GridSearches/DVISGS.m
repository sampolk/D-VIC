%% DVIS
% Extracts performances for DVIS

%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = 10:10:100;


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prcts{4} =  65:(100-65)/19:100;
prcts{2} =  65:(100-65)/29:100;
prcts{1} =  88:(100-88)/39:100;
prcts{3} = 15:(60 -15)/19:60;
prcts{5} = 85:(100-85)/19:100;

numReplicates = 50;
 
%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

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
    for i = 9:13
        for j = 1:length(prctiles)-1
    
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
                G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
                
                % KDE Computation
                density = KDE_large(Dist_NN, Hyperparameters);
        
                % Spectral Unmixing Step
                pixelPurity = compute_purity(X,Hyperparameters);
    
                if G.EigenVals(2)<1

                    [Clusterings, ~] = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
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