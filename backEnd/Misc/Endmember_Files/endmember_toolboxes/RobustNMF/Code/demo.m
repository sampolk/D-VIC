%% generate data
F = 10;
N = 100;
K = 3;
Y = abs(randn(F,N));

%% initialise
M_ini = abs(randn(F,K))+1;
A_ini = abs(randn(K,N))+1;
R_ini = abs(randn(F,N))+1;

%% run algorithm
lambda = 0.1; % penalisation weight hyperparameter
beta = 1; % beta divergence shape parameter
tol = 1e-5; % convergence tolerance parameter
n_iter_max = 1000; % maximum number of iterations

[W, H, E, obj] = group_robust_nmf(Y, beta, M_ini, A_ini, R_ini, lambda, tol, n_iter_max);

%% display objective function
figure;
semilogy(obj);

%% display data and approximate
figure;
subplot(211)
imagesc(Y);
title('data')

subplot(212)
imagesc(W*H + E);
title('approximate')

%% 

load('SalinasACorrected.mat')

