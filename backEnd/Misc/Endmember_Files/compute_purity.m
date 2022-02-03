function [purity, endmembers, abundances] = compute_purity(X,Hyperparameters)
%{
Purpose: Calculates the endmember decomposition of a Hyperspectral image X
         as well as pixel purity. The specific endmember extraction method
         can be any of the following:

            - ATGP
            - N-FINDR
            - Vertex Component Analysis (VCA) 
            - Group Robust NMF (GRNMF)
            - Minimum Volume Constrained NMF (MVC-NMF)
            - Pixel Purity Index (PPI)
            - Alternating Volume Maximization (AVMAX)
            - Simplex Growing Algorithm (SGA)

Inputs:     X:                  (M*N)xD data matrix. 
            Hyperparameters:    Structure with the following fields:
                - SpatialParams:    Stores the dimensions of the original 
                                    image.  (Required)
                                    
                - EndmemberParams:  Endmember extraction algorithm and 
                                    number of endmembers (Optional). 

SpatialParams.ImageSize must be a 1x2 vector containing the spatial 
dimensions of the HSI. I.e., if the original data cube has M rows, N
columns, and D spectral bands, SpatialParams.ImageSize = [M,N].

If EndmemberParams is not included as a field in Hyperparameters, we 
automatically use HySime to measure the number of endmembers and VCA to 
measure endmembers.  

If EndmemberParams is included as a field in Hyperparameters, it must be a
structure with the following fields: 

    - K:         The number of endmembers.
    - Algorithm: The algorithm used for endmember extraction. 

Options for Hyperparameters.EndmemberParams.Algorithm are: 

    - 'ATGP' 
    - 'N-FINDR' % Done 
    - 'VCA'  
    - 'GRNMF'
    - 'PLM' % No Good
    - 'MVC-NMF' % Done
    - 'PPI' % Done
    - 'AVMAX' % Done 
    - 'SGA' % Done

Outputs:    
            endmembers:     Calculated endmembers.
            purity:         Purity of pixels, as measured by endmember unmixing.
%}


if ~isfield(Hyperparameters, 'EndmemberParams')
    
    % compute hysime to get best estimate for number of endmembers
    Hyperparameters.EndmemberParams.K = hysime(X');
    Hyperparameters.EndmemberParams.Algorithm = 'VCA';
 end

% Size of HSI.
M = Hyperparameters.SpatialParams.ImageSize(1); 
N = Hyperparameters.SpatialParams.ImageSize(2);
D = size(X,2);

HSI = reshape(X, M,N,D);

% Number of endmembers
K = Hyperparameters.EndmemberParams.K; 

if strcmp(Hyperparameters.EndmemberParams.Algorithm, 'ATGP')
    
    [endmembers,~] = EIA_ATGP(X', K);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'N-FINDR')
    
    [endmembers,~] = EIA_NFINDR(X',K,200, Hyperparameters.SpatialParams.ImageSize);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K); 

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'VCA')

    [ endmembers, ~, ~ ] = hyperVca(double( X'), K );
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K); 
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'GRNMF')
    
    [endmembers, abundances] = GroupRobustNMF(X, K, 1e-4);
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'PLM')
    
    Unmixing = PerturbedLinearMixing(X,[M,N], K);
    
    abundances = Unmixing.Abundance;
    endmembers = Unmixing.Endmembers;
    
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'MVC-NMF')
    
    [endmembers, abundances, ~, ~] = mvcnmf_shell(X, K);
 
elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'PPI')
    
    endmembers = fippi(HSI, K);
    abundances = hyperNnls(X', endmembers)';
    K = size(endmembers,2); 

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'AVMAX')

    endmembers = hyperAvmax(X', K, 0);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);


elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'AVMAXWithVCA')

    endmembers = hyperAvmax(X', K, 1);
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'ManyAVMAX')

    endmembers = zeros([size(hyperAvmax(X', K, 1)), Hyperparameters.EndmemberParams.NumReplicates]);
    volumes = zeros(Hyperparameters.EndmemberParams.NumReplicates,1);
    parfor i = 1:Hyperparameters.EndmemberParams.NumReplicates
        [endmembers(:,:,i), volumes(i)] = hyperAvmax(X', K, 1);
    end
    [~,i] = max(volumes);
    endmembers = endmembers(:,:,i);

    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

elseif strcmp(Hyperparameters.EndmemberParams.Algorithm, 'SGA')
 
    [endmemberindex,~] = SGA(HSI,K);

    endmembers = zeros(D,K);
    for i = 1:K
        endmembers(:,i) = HSI(endmemberindex(i,2), endmemberindex(i,1), :);
    end
    abundances = reshape(hyperNnls(X', endmembers)', M, N, K);

end

abundances = reshape(abundances,M*N,K);
abundances = abundances./sum(abundances,2); % Row normalize to make a distribution
purity = max(abundances,[],2);
purity(isnan(purity)) = 0; 

    
end 
