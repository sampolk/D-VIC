function [C, K, Dt] = ADVIS(X, Hyperparameters, t, Y, G, density, pixelPurity) 
%{
 - This function produces image segmentations 

 

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:  
                                    - numLabels: the budget provided for
                                                 ground truth label queries
                                    - K_Known:   The number of classes for
                                                 image segmentation.
        t:                      Diffusion time parameter.
        Y:                      Ground truth labels to be used in active
                                learning.
        G:                      Graph structure computed using  
                                'extract_graph_large.m' 
        zeta:                   Kernel Density Estimator
Output: 
            - C:                n x 1 vector storing the ADVIS partition. 
            - K:                Scalar, number of classes in C.
            - Dt:               n x 1 matrix storing \mathcal{D}_t(x). 

© 2021 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%}  

% ============================ PARSE ARGUMENTS ============================
% Parse Arguments
if nargin == 5
    % Extract Graph
    G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
end
if nargin < 5
    % Extract KDE 
    density = KDE_large(Dist_NN, Hyperparameters);
end
if nargin < 6
    % Extract Pixel Purity

    % compute hysime to get best estimate for number of endmembers
    [kf,~]=hysime(X'); 
    
    % Store in hyperparameter structure
    Hyperparameters.EndmemberParams.K = kf;
    Hyperparameters.EndmemberParams.Algorithm = 'VCA';
    
    % Extract Pixel Purity using VCA
    pixelPurity = compute_purity(X,Hyperparameters);  
end

% Aggregate purity and density into single measure, zeta
zeta = harmmean([density./max(density), pixelPurity./max(pixelPurity)],2);

% ========================= Parse Hyperparameters =========================
if ~isfield(Hyperparameters, 'NEigs')
    Hyperparameters.NEigs = size(G.EigenVecs,2);
end
if ~isfield(Hyperparameters, 'DtNNs')
    Hyperparameters.DtNNs = 100;
end
if ~isfield(Hyperparameters, 'numLabels')
    Hyperparameters.K_Known = length(unique(Y));
 end
if ~isfield(Hyperparameters, 'numLabels')
    Hyperparameters.numLabels = K;
end

% ========= Perform Diffusion Distance Nearest Neighbor Searches ==========
n = length(X);
% Calculate diffusion map
DiffusionMap = zeros(n,Hyperparameters.NEigs);
for l = 1:size(DiffusionMap,2)
    DiffusionMap(:,l) = G.EigenVecs(:,l).*(G.EigenVals(l).^t);
end
 
% Compute Hyperparameters.NumDtNeighbors Dt-nearest neighbors.
[IdxNN, D] = knnsearch(DiffusionMap, DiffusionMap, 'K', Hyperparameters.NumDtNeighbors);

% =========================== Calculate d_t(x) ============================
% compute d_t(x), stored as dt
dt = zeros(n,1);
zeta = zeta./sum(zeta);
zeta_max = max(zeta);
for i=1:n
    if zeta(i) == zeta_max
        dt(i) = max(pdist2(DiffusionMap(i,:),DiffusionMap));
    else
        
        idces =  find(zeta(IdxNN(i,:))>zeta(i));
        
        if ~isempty(idces)
            % In this case, at least one of the Hyperparameters.NumDtNeighbors Dt-nearest neighbors
            % of X(i,:) is also higher density, so we have already calculated rho_t(X(i,:)).
            dt(i) = D(i,idces(1));
            
        else
            % In this case, none of the first Hyperparameters.NumDtNeighbors Dt-nearest neighbors of
            % X(i,:) are also higher density. So, we do the full search.
            dt(i) = min(pdist2(DiffusionMap(i,:),DiffusionMap(zeta>zeta(i),:)));
        end
    end
end

% =========================== Label Class Modes ===========================

% Extract Dt(x) and sort in descending order
Dt = dt.*zeta;
[~, m_sorting] = sort(Dt,'descend');

% Determine K and label queried points
K = Hyperparameters.K_Known;
numLabels = Hyperparameters.numLabels;
 
C = zeros(n,1);
% Applied queried labels
C(m_sorting(1:numLabels)) = Y(m_sorting(1:numLabels)); 

% Assign labels to class modes if we don't query enough ground truth
% classes
missingLabels = setdiff(1:K, unique(Y(m_sorting(1:numLabels))));
numMissingLabels = length(missingLabels);
if numMissingLabels >0
    C(m_sorting(numLabels+1:numLabels+numMissingLabels)) = missingLabels;
end

% ======================== Label Non-Modal Points =========================

if K == 1
    C = ones(n,1);
else
    
    % For indexing purposes
    idx = 1:n; 

    % Label non-modal points according to the label of their Dt-nearest
    % neighbor of higher density that is already labeled.
    [~,l_sorting] = sort(zeta,'descend');
     
    for j = 1:n
        i = l_sorting(j);
        if C(i)==0 % unlabeled point
            
            NNs = IdxNN(i,:);
            idces = find(and(C(NNs)>0, zeta(NNs)>zeta(i))); % Labeled, higher-density points in NNs
            
            if isempty(idces)
                % None of the Dt-nearest neighbors are also higher-density
                % & labeled. So, we do a full search.
                candidates = idx(and(C>0, zeta>zeta(i))); % All labeled points of higher density.
                if isempty(candidates)
                    disp([])
                end
                
                [~,temp_idx] = min(pdist2(DiffusionMap(i,:), DiffusionMap(candidates,:)));
                C(i) = C(candidates(temp_idx));    
            else
                % At least one of the Dt-nearest neighbors is higher
                % density & labeled. So, we pick the closest point.
                C(i) = C(NNs(idces(1)));                
            end
        end
    end
    
end 