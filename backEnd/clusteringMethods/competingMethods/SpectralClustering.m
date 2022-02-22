function C = SpectralClustering(G,K)

EigenVecs_Normalized = G.EigenVecs(:,1:min(K,10))./sqrt(sum(G.EigenVecs(1:min(K,10)).^2,2));

C = kmeans(EigenVecs_Normalized, K);
