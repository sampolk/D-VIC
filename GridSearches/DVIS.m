function Clusterings = DVIS(X, Idx_NN, Dist_NN, Hyperparameters, Graph, density, purity)

if nargin == 2
    Graph = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);
    density = KDE_large(Dist_NN, Hyperparameters);
    purity = compute_purity(X,Hyperparameters);
end

Clusterings = MLUND_large(X, Hyperparameters, Graph, harmmean([density./max(density), purity./max(purity)],2));
