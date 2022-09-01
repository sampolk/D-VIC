%% FSSC
% Extracts performances for FSSC

%% Grid Search Parameters
clear;
clc;
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set parameters of FSSC
rate=10;
alpha_u = [0,1e-5,1e-3,0.1,0.5,0.9,0.99,0.9999];
q=max(ceil(log2(NNs))+1);
numReplicates = 10;

%%

datasets = {'SalinasACorrected',  'JasperRidge','PaviaCenterSubset2','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',  'Pavia Subset',    'Indian Pines',           'Synthetic HSI'};

for dataIdx =  [1]

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{dataIdx});

    % ============================== DVIS ==============================
 
    % Preallocate memory
    maxOA = 0;
    OAs     = NaN*zeros(length(NNs), length(alpha_u), numReplicates);
    kappas  = NaN*zeros(length(NNs), length(alpha_u), numReplicates);
    Cs      = zeros(M*N,length(NNs), length(alpha_u), numReplicates);

    delete(gcp('nocreate'))
    poolObj = parpool;

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(alpha_u)

            if poolObj.NumWorkers<6
                delete(gcp('nocreate'));
                poolObj = parpool;
            end
            
            

            % Parameter t2 represents the number of iterations, not the
            % number of replicates
            for k = 1:numReplicates

                [~,~,C,~,~] = FSSC(X,q, NNs(i),K,rate,alpha_u(j));
                [ OAs(i,j, k), kappas(i,j, k), tIdx] = calcAccuracy(Y, C, ~strcmp('JasperRidge', datasets{dataIdx}));
                Cs(:,i,j,k) = C;
               
    
                disp(['FSSC: '])
                disp([i/length(NNs), j/length(alpha_u), k/numReplicates, maxOA])
            end
            maxOA = max(mean(OAs,3), [],'all');
            
            save(strcat('FSSCResults1', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'alpha_u', 'numReplicates', 'q', 'maxOA')
        end
    end

end