function [C, K, Dt] = DVIS(X, Hyperparameters, t, Y, G, density, pixelPurity) 
%{
This function produces image segmentations for the Diffusion and Volume
maximization-based Image Clustering algorithm for HSI material 
discrimination.

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:   
                                    - K_Known:   The number of classes for
                                                 image segmentation.
        t:                      Diffusion time parameter.
        Y:                      Ground truth labels to be used in active
                                learning.
        G:                      Graph structure computed using  
                                'extract_graph_large.m' 
        zeta:                   Kernel Density Estimator
Output: 
            - C:                n x 1 vector storing the DVIS partition. 
            - K:                Scalar, number of classes in C.
            - Dt:               n x 1 matrix storing \mathcal{D}_t(x). 

© 2022 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%}  

% ============================ PARSE ARGUMENTS ============================
% Parse Arguments
if nargin == 5
    % Extract Graph
    G = extractGraph(X, Hyperparameters, Idx_NN, Dist_NN);
end
if nargin < 5
    % Extract KDE 
    density = KDE(Dist_NN, Hyperparameters);
end
if nargin < 6
    % Extract Pixel Purity

    % compute hysime to get best estimate for number of endmembers
    [kf,~]=hysime(X'); 
    
    % Store in hyperparameter structure
    Hyperparameters.EndmemberParams.K = kf;
    Hyperparameters.EndmemberParams.Algorithm = 'VCA';
    
    % Extract Pixel Purity using VCA
    pixelPurity = computePurity(X,Hyperparameters);  
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
diffusionMap = zeros(n,Hyperparameters.NEigs);
for l = 1:size(diffusionMap,2)
    diffusionMap(:,l) = G.EigenVecs(:,l).*(G.EigenVals(l).^t);
end
 
% Compute Hyperparameters.NumDtNeighbors Dt-nearest neighbors.
[IdxNN, D] = knnsearch(diffusionMap, diffusionMap, 'K', Hyperparameters.NumDtNeighbors);

% =========================== Calculate d_t(x) ============================
% compute d_t(x), stored as dt
dt = zeros(n,1);
zeta = zeta./sum(zeta);
zeta_max = max(zeta);
for i=1:n
    if zeta(i) == zeta_max
        dt(i) = max(pdist2(diffusionMap(i,:),diffusionMap));
    else
        
        idces =  find(zeta(IdxNN(i,:))>zeta(i));
        
        if ~isempty(idces)
            % In this case, at least one of the Hyperparameters.NumDtNeighbors Dt-nearest neighbors
            % of X(i,:) is also higher density, so we have already calculated rho_t(X(i,:)).
            dt(i) = D(i,idces(1));
            
        else
            % In this case, none of the first Hyperparameters.NumDtNeighbors Dt-nearest neighbors of
            % X(i,:) are also higher density. So, we do the full search.
            dt(i) = min(pdist2(diffusionMap(i,:),diffusionMap(zeta>zeta(i),:)));
        end
    end
end

% =========================== Label Class Modes ===========================

% Extract Dt(x) and sort in descending order
Dt = rt.*zeta;
[~, m_sorting] = sort(Dt,'descend');

% Determine K based on the ratio of sorted Dt(x_{m_k}). 
if isfield(Hyperparameters, 'K_Known')
    K = Hyperparameters.K_Known;
else
    [~, K] = max(Dt(m_sorting(2:ceil(n/2)-1))./Dt(m_sorting(3:ceil(n/2))));
    K=K+1;
end

% ========================== Label Non Modal Pts ==========================
if K == 1
    C = ones(n,1);
else
    
    idx = 1:n;
    C = zeros(n,1);
    % Label modes
    C(m_sorting(1:K)) = 1:K;

    % Label non-modal points according to the label of their Dt-nearest
    % neighbor of higher density that is already labeled.
    [~,lSorting] = sort(zeta,'descend');
     
    for j = 1:n
        i = lSorting(j);
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
                
                [~,temp_idx] = min(pdist2(diffusionMap(i,:), diffusionMap(candidates,:)));
                C(i) = C(candidates(temp_idx));    
            else
                % At least one of the Dt-nearest neighbors is higher
                % density & labeled. So, we pick the closest point.
                C(i) = C(NNs(idces(1)));                
            end
        end
    end
    
end 