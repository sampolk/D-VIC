%% DVIS
% Extracts performances for DVIS

%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = 10:10:100;


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prcts{4} =  65:(100-65)/19:100;
prcts{2} =  65:(100-65)/19:100;
prcts{1} =  88:(100-88)/19:100;
prcts{3} = 15:(60 -15)/19:60;
prcts{5} = 0: (45 - 0)/19:45;

numReplicates = 10;
 
%% Grid searches
datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

for dataIdx =  5

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
    OAs     = NaN*zeros(length(NNs), length(prctiles));
    kappas  = NaN*zeros(length(NNs), length(prctiles));
    Cs      = zeros(M*N,length(NNs), length(prctiles));

    delete(gcp('nocreate'))
    poolObj = parpool;
    pixelPurity = zeros(M*N,1);
    for k = 1:numReplicates 

        if dataIdx ==5
            Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
        else
            Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
        end
        Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
        Hyperparameters.EndmemberParams.NumReplicates = 100;
        
        [purityTemp, ~, ~] = compute_purity(X,Hyperparameters);
        pixelPurity = pixelPurity + purityTemp/numReplicates;
    end

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)

            if poolObj.NumWorkers<6
                delete(gcp('nocreate'));
                poolObj = parpool;
            end

            Hyperparameters.DiffusionNN = NNs(i);
            Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
            Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');

            density = KDE_large(Dist_NN, Hyperparameters);
            [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

            if G.EigenVals(2)<1
                Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
            
                [ OAs(i,j), kappas(i,j), tIdx] = calcAccuracy(Y, Clusterings, ~strcmp('Jasper Ridge', datasets{dataIdx}));
                Cs(:,i,j) = Clusterings.Labels(:,tIdx);
                
                currentPerf = OAs(i,j);
                [maxOA, k] = max(OAs,[],'all');
    
                if currentPerf >= maxOA
                
                    [l,j] = ind2sub(size(mean(OAs,3)), k);
                    stdOA = nanstd(squeeze(OAs(l,j,:)));
                end
                save(strcat('DVISResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA')
           end

            disp(['DVIS: '])
            disp([i/length(NNs), j/length(prctiles), maxOA])
        end

    end
    save(strcat('DVISResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA')

end