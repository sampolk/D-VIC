%% DPC-DLP
% Extracts performances for DPC-DLP

%% Grid Search Parameters
clear;
clc;
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];


% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 5:10:95;

% Set parameters of DPC-DLP
mu=1;
alfa=1;
lambda=0;
t2=round(10.^(0:0.2:1.5));


%%

datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

for dataIdx =  [1]
    
    if dataIdx == 5
        % Set number of nearest neighbors to use in graph and KDE construction.
        NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];
    end
    

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});
    dis=pdist2(X,X);

    % ============================== DVIS ==============================
 
    % Preallocate memory
    maxOA = 0;
    OAs     = NaN*zeros(length(NNs), length(prctiles), length(t2));
    kappas  = NaN*zeros(length(NNs), length(prctiles), length(t2));
    Cs      = zeros(M*N,length(NNs), length(prctiles), length(t2));

    delete(gcp('nocreate'))
    poolObj = parpool;

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)

            if poolObj.NumWorkers<6
                delete(gcp('nocreate'));
                poolObj = parpool;
            end
    
            if dataIdx ==5
                Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
            else
                Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
            end 
            
            sigma=calculateSigma(X,prctiles(j)/100);
            W=calculateP(X,dis,mu,sigma);
            
            % Find Density Peaks
            [peaks,~,~,neighbs,~]=densityPeaks(X,dis,NNs(i),K);
            
            % Label the nearest neighbors of density peaks
            [Y0,Labeled]=labeling(peaks,K,n,neighbs);
            
            % Recompute the graph and then labeling
            [P,~]=calculatePP(W,neighbs(:,1:NNs(i)));
            Labels=DLP(X,Y0,Labeled,P,alfa,lambda,t2);

            % Parameter t2 represents the number of iterations, not the
            % number of replicates
            for k = 1:length(t2) 

                C = DLP(X,Y0,Labeled,P,alfa,lambda,t2(k));
                [ OAs(i,j, k), kappas(i,j, k), tIdx] = calcAccuracy(Y, C, ~strcmp('JasperRidge', datasets{dataIdx}));
                Cs(:,i,j,k) = C;
                
                maxOA = max(OAs, [],'all');
    
                disp(['DPC-DLP: '])
                disp([i/length(NNs), j/length(prctiles), k/length(t2), maxOA])
            end
            
            save(strcat('DPCDLPResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 't2', 'maxOA')
        end
    end

end