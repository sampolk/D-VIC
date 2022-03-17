function C = SpectralClustering(G,K)
%{
This function evaluates Spectral Clustering (SC) and was used in
experiments in the following paper:

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}
EigenVecs_Normalized = G.EigenVecs(:,1:min(K,10))./sqrt(sum(G.EigenVecs(1:min(K,10)).^2,2));

C = kmeans(EigenVecs_Normalized, K);
