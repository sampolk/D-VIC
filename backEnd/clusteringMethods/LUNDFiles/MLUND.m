function [Clusterings,runtimes] = MLUND(X, Hyperparameters, G, p)
%{
 - This function produces a structure with multiscale clusterings produced
   with the M-LUND algorithm, presented in the following paper. 

        - Murphy, J. M., & Polk, S. L. (2022). A multiscale environment for 
          learning by diffusion. Applied and Computational Harmonic 
          Analysis, 57, 58-100.

   and analyzed further in the following papers:

        - Polk, S. L., & Murphy, J. M. (2021, July). Multiscale Clustering 
          of Hyperspectral Images Through Spectral-Spatial Diffusion 
          Geometry. In 2021 IEEE International Geoscience and Remote 
          Sensing Symposium IGARSS (pp. 4688-4691). IEEE.
        - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
          Diffusion and Volume Maximization-Based Clustering of Highly 
          Mixed Hyperspectral Images. (In Review).

Inputs: X:                      Data matrix.
        Hyperparameters:        Optional structure with graph parameters
                                with required fields:  
                                    - Beta:    Exponential time scaling parameter.
                                    - Tau:     Diffusion Stationarity Threshold.
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

T = full(ceil(log( log(Hyperparameters.Tau*min(G.StationaryDist)/2)/log(G.EigenVals(2)))/log(Hyperparameters.Beta)));

if isreal(T)

    % ========================== Cluster analysis =========================

    % Extract Time Steps
    timesamples = [0, Hyperparameters.Beta.^(0:T)];

    % Initialize
    Ct = zeros(n,T+2);
    Kt = zeros(T+2,1);
    Dt = zeros(n,T+2);
    runtimes = zeros(T+2,1);
    for i = 1:T+2
        t = timesamples(i);
        tic
        [C, K, Dt_temp] = LUND(X, Hyperparameters, t, G, p);
        runtimes(i) = toc;
        Ct(:,i) = C;
        Kt(i) = K;
        Dt(:,i) = Dt_temp;
    end

    % ============================ VI analysis ============================

    J = find(and(Kt<n/2, Kt>1)); % Time samples during which a nontrivial clusering is extracted
    n_J = length(J);

    if n_J > 0 % There is a nontrivial clustering of X.
        [~,VI_tot,t] = totalVI_minimization(Ct, Kt);
        TotalVI.Vector = VI_tot;
        TotalVI.Minimizer_Idx = t;
    else
        TotalVI.Vector = zeros(T+2,1);
        TotalVI.Vector(:) = NaN; 
        TotalVI.Minimizer_Idx = NaN;
    end

    % Store results in structure "Clusterings"
    Clusterings.Graph = G;
    Clusterings.Hyperparameters = Hyperparameters;
    Clusterings.Labels = Ct;
    Clusterings.K = Kt;
    Clusterings.TotalVI = TotalVI;
    Clusterings.TimeSamples = timesamples;
    Clusterings.Density = p;
    Clusterings.Dt = Dt;

else
    % Store results in structure "Clusterings"
    Clusterings.Graph = G;
    Clusterings.Hyperparameters = Hyperparameters;
    Clusterings.Labels = ones(n,1);
    Clusterings.K = NaN;
    
    TotalVI.Vector = zeros(real(T)+2,1);
    TotalVI.Vector(:) = NaN; 
    TotalVI.Minimizer_Idx = NaN;
    
    Clusterings.TotalVI = TotalVI;
    Clusterings.TimeSamples = NaN;
    Clusterings.Density = p;
    Clusterings.Dt = NaN*ones(n,1);
    runtimes = NaN;
end