function [C, K, Dt] = SRDL_large(X, Hyperparameters, G, p)
%{
 - This function produces a structure with multiscale clusterings produced
   with the SRDL algorithm, presented in the following paper. 

        - Murphy, James M., and Mauro Maggioni. "Spectral-spatial diffusion 
          geometry for hyperspectral image clustering." IEEE Geoscience and 
          Remote Sensing Letters (2019).
    
   and analyzed further in the following paper:

        - Polk, Sam L. and Murphy James M., 2021. Multiscale Spectral-
          Spatial Diffusion Geometry for Hyperspectral Image Clustering. 
          (In Review)

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:  
                                    - DiffusionTime:    Diffusion time parameter.
        G:                      Graph structure computed using  
                                'extract_graph_large.m' 
        p:                      Kernel Density Estimator (Optional).

Output: 
            - C:                n x 1 vector storing the LUND clustering 
                                of X at time t.
            - K:                Scalar, number of clusters in C.
                                clusters in the Labels(:,t) clustering.
            - Dt:               n x 1 matrix storing \mathcal{D}_t(x). 

© 2021 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%} 

if ~isfield(Hyperparameters, 'NEigs')
    Hyperparameters.NEigs = size(G.EigenVecs,2);
end

n = length(X);
C = zeros(n,1);

% Calculate diffusion map
DiffusionMap = zeros(size(G.EigenVecs,1), Hyperparameters.NEigs);
parfor l = 1:size(DiffusionMap,2)
    DiffusionMap(:,l) = G.EigenVecs(:,l).*(G.EigenVals(l).^Hyperparameters.DiffusionTime);
end
disp('Diffusion Map Calculated')

% compute rho_t(x), stored as rt
rt = zeros(n,1);
p_max = max(p);
% p_pct = prctile(p, Hyperparameters.PPct);
parfor i=1:n
    if p(i) == p_max
        rt(i) = max(pdist2(DiffusionMap(i,:),DiffusionMap));
    else
        rt(i) = min(pdist2(DiffusionMap(i,:),DiffusionMap(p>p(i),:)));
    end
        
    if mod(i,100) == 0
        disp(strcat('Rho calculation, ', num2str((1-i/n)*100, 3), '% complete.'))
    end
end

% Extract Dt(x) and sort in descending order
Dt = rt.*p;
[~, m_sorting] = sort(Dt,'descend');

% Determine K based on the ratio of sorted Dt(x_{m_k}). 
if isfield(Hyperparameters, 'K_Known')
    K = Hyperparameters.K_Known;
else
    [~, K] = max(Dt(m_sorting(1:n-1))./Dt(m_sorting(2:n)));
end
disp('Cluster Modes Calculated')

if K == 1
    C = ones(n,1);
else
    
    % Label modes
    C(m_sorting(1:K)) = 1:K;

    % Label non-modal points according to the label of their Dt-nearest
    % neighbor of higher density that is already labeled.
    [~,l_sorting] = sort(p,'descend');
    
    % Define spatial consensus variables
    M = Hyperparameters.SpatialParams.ImageSize(1);
    N = Hyperparameters.SpatialParams.ImageSize(2);  
    R = Hyperparameters.SpatialParams.SpatialRadius;  
    [I,J]=find(sum(X,3)~=0); 
    
    % Labeling Pass 1
    for j = 1:n
        i = l_sorting(j);
        if C(i)==0 % unlabeled point
            candidates = find(and(p>=p(i), C>0)); % Labeled points of higher density.
            Dtxi = pdist2(DiffusionMap(i,:),DiffusionMap(candidates,:));
            [~,temp_idx] = min(Dtxi);
            C(i) = C(candidates(temp_idx));
        end
    end
    
    % Labeling Pass 2
    for j=1:n
        
        i = l_sorting(j);
        if C(i)==0
            
            candidates = find(and(p>=p(i), C>0)); % Labeled points of higher density.            
            Dtxi = pdist2(DiffusionMap(i,:),DiffusionMap(candidates,:));
            [~,temp_idx] = min(Dtxi); % index of the Dt-nearest neighbor of higher density that is already labeled. 
            C(i) = C(candidates(temp_idx));
            
            % Special case when multiple points have equal density
            if C(i) == 0
                temp = find(Labels>0); % indices of labeled points
                NN = knnsearch(X(temp,:),X(i,:)); 
                C(i)=C(temp(NN));% Assign label of nearest Euclidean distance-nearest neighbor that is already labeled.
            end
            
            % Check spatial consensus
            try
                LabelsMatrix=sparse(I(l_sorting(1:j)),J(l_sorting(1:j)),C(l_sorting(1:j)),M,N);
                SpatialConsensusLabel=SpatialConsensus(LabelsMatrix,i,M,N,I,J,R);
                if ~(SpatialConsensusLabel==C(i)) && (SpatialConsensusLabel>0)
                    C(l_sorting(j))=0; % reset if spatial consensus label disagrees with DL label
                end
            catch
                keyboard
            end
        end
    end

end
end