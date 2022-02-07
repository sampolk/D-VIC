function C = evaluateSC(X,NN, K)

% Perform nearest neighbor searches
[Idx_NN, Dist_NN] = knnsearch(X, X, 'K', NN+1);
Idx_NN(:,1) = [];
Dist_NN(:,1) = [];

% Set parameters for graph construction
Hyperparameters.SpatialParams.ImageSize = [];
Hyperparameters.NEigs = 10;
Hyperparameters.DiffusionNN = NN;

% Construct graph
[G,~] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

% Implement K-Means on normalized eigenvectors of graph Laplacian
EigenVecs_Normalized = G.EigenVecs(:,1:min(K,10))./sqrt(sum(G.EigenVecs(1:min(K,10)).^2,2));
C = kmeans(EigenVecs_Normalized, K);
