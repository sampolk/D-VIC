function [endmembers, abundances] = GroupRobustNMF(X, K, error_tol)

Y=X';
[N,F] = size(X);

M_ini = abs(randn(F,K))+1;
A_ini = abs(randn(K,N))+1;
R_ini = abs(randn(F,N))+1;
 
lambda = 0.1; % penalisation weight hyperparameter
beta = 2; % beta divergence shape parameter
n_iter_max = 1000; % maximum number of iterations

[W, H, ~, ~] = group_robust_nmf(Y, beta, M_ini, A_ini, R_ini, lambda, error_tol, n_iter_max, 0);

endmembers = W';
abundances = H';