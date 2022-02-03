%% DVIS
% Extracts performances for DVIS

%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = [unique(round(10.^(1:0.1:2.7),-1)), 600, 700, 800, 900];

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prctiles = 5:10:95; 

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected'};

for dataIdx = [5]

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    load(datasets{dataIdx})

    % If Salinas A, we add gaussian noise and redo nearest neighbor searches. 
    if dataIdx == 5
        X = X + randn(size(X)).*10^(-7);
    
        [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    
        Dist_NN = Dist_NN(:,2:end);
        Idx_NN = Idx_NN(:,2:end);
    end
    Dist_NN = Dist_NN(:,2:end);
    Idx_NN = Idx_NN(:,2:end);

    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.K_Known = length(unique(Y)); % We subtract 1 since we discard gt labels
    Hyperparameters.Tolerance = 1e-8;
    K = length(unique(Y));

    % ============================== DVIS ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    kappas  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    Cs      = zeros(M*N,length(NNs), length(prctiles), numReplicates);
    Us = cell(length(NNs), length(prctiles), numReplicates);
    As = cell(length(NNs), length(prctiles), numReplicates);

    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)
            for k = 1:numReplicates


                Hyperparameters.DiffusionNN = NNs(i);
                Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
                Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
  
                X = reshape(X,size(HSI,3) ,M*N); % X is Dxn
                kf = hysime(X);
                
                % remove noise by projecting onto the first kf PCs.
                [UU, ~, ~] = svds(X,kf); % UU is Dxkf
                Lowmixed = UU'*X; %
                X = UU*Lowmixed; % X is still Dxn
                 
                % vca algorithm
                % A_vca = endmembers
                [A_vca, ~] = vca(X,'Endmembers', kf); % A_vca is Dxkf
                
                % FCLS
                warning off;
                AA = [1e-5*A_vca;ones(1,length(A_vca(1,:)))]; % AA is (D+1)xkf
                s_fcls = zeros(length(A_vca(1,:)),M*N); % s_fcls is kf x n
                for l=1:M*N
                    r = [1e-5*X(:,l); 1]; % r is (D+1)x1
                %   s_fcls(:,j) = nnls(AA,r);
                    s_fcls(:,l) = lsqnonneg(AA,r); % kfx1
                end
                % s_fcls = endmember abundances
                
                % use vca to initiate
                Ainit = A_vca;
                sinit = s_fcls;
                
                % % random initialization
                % idx = ceil(rand(1,c)*(M*N-1));
                % Ainit = mixed(:,idx);
                % sinit = zeros(c,M*N);
                
                % PCA
                %[PrinComp,meanData] = pca(mixed',0);')
                meanData = mean(X');
                [~,~,PrinComp] = svd(X'-meanData,'econ');

                % test mvcnmf
                tol = 1e-6;
                maxiter = 150;
                T = 0.015; 
                                
                % use vca to initiate
                Ainit = A_vca;
                sinit = s_fcls;
                
                % use conjugate gradient to find A can speed up the learning
                [U, A, volume, loss] = mvcnmf(X,Ainit,sinit,PrinComp,meanData,T,tol,maxiter,2,1);
                pixelPurity = max(A)';

                 
                density = KDE_large(Dist_NN, Hyperparameters);
                [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

                if G.EigenVals(2)<1
                    Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));

                    [~,~, OAs(i,j,k), ~, kappas(i,j,k), tIdx]= measure_performance(Clusterings, Y);
                    C =  Clusterings.Labels(:,tIdx);
                    Cs(:,i,j,k) = C;
                end
    
                disp(['DVIS: '])
                disp([i/length(NNs), j/length(prctiles), k/numReplicates, dataIdx/5])

            end
        end
    end

    save(strcat('DVISResults', datasets{dataIdx}),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'Us', 'As')
end