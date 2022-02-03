function [C, K, Dt] = DVIS(X, Hyperparameters, t, G, density, pixelPurity)
%{
 - This function produces the clusterings used for the D-VIS algorithm, 
   presented in the following paper. 

        - 

   and analyzed further in the following papers:

        - 

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:
        t:                      Diffusion time step
        G:                      Graph structure computed using  
                                'extract_graph_large.m' 
        p:                      Kernel Density Estimator.
        pixelPurity:            pixel purity estimate

Output: 
            - C:                n x 1 vector storing the D-VIS clustering 
                                of X at time t.
            - K:                Scalar, number of clusters in C.
            - Dt:               n x 1 matrix storing \mathcal{D}_t(x). 

© 2021 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu
%} 

if ~isfield(Hyperparameters, 'NEigs')
    Hyperparameters.NEigs = size(G.EigenVecs,2);
end

% Parse Arguments
if nargin == 4
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

% Implement LUND with zeta
[C, K, Dt] = LearningbyUnsupervisedNonlinearDiffusion_large(X, Hyperparameters, t, G, zeta);
