%% PGDPC
% Extracts performances for DPC-DLP

%% Grid Search Parameters
clear;
clc;

% Set parameters of PGDPC
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

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

    % ============================== DVIS ==============================
 
    % Preallocate memory
    maxOA = 0;
    OAs     = NaN*zeros(length(NNs),1);
    kappas  = NaN*zeros(length(NNs),1);
    Cs      = zeros(M*N,length(NNs),1);

    delete(gcp('nocreate'))
    poolObj = parpool;

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)

        if poolObj.NumWorkers<6
            delete(gcp('nocreate'));
            poolObj = parpool;
        end
        
        C = PGDPC(X,NNs(i),K);
        [ OAs(i), kappas(i), tIdx] = calcAccuracy(Y, C, ~strcmp('JasperRidge', datasets{dataIdx}));
        Cs(:,i) = C;

        disp(['PGDPC: '])
        disp([i/length(NNs), maxOA])
        
        maxOA = max(mean(OAs,3), [],'all');

        save(strcat('PGDPCResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'maxOA')
        
    end

end