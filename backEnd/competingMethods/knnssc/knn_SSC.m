function C = knn_SSC( Y, alpha, num_neighbors, Idx_NN)
% this function implements KNN-SSC algorithm
% min ||C||_1 + \lambda/2  \sum||Y(:,i)-D_i C(:,i)||_F^2.
% input: Y = input data
%        alpha = regularization parameter for setting \lambda adaptively:
%        see paper Appendix B.
%        num_neighbors = number of neighbors for forming dictionary D_i
%        based on KNN
%        precalculated nearest neighbor structure, where Idx_NN(i,j) is the
%        index of the jth nearest neighbor of Y(:,i);
% output: coefficient matrix

N = size(Y,2);

% setting penalty parameters for the ADMM
mu1 = alpha * 1/computeLambda_mat(Y);
mu2 = alpha * 1;
thr = 2*10^-4; 
maxIter=200;
C = zeros(N);
% for each data point
for i = 1 : N
    %find k nearest neighbors
    Idx=Idx_NN(i,1:num_neighbors);
    %form dictionary
    D = Y(:,Idx);
    n = size(D,2);
    % initialization
    A = inv(mu1*(D'*D)+mu2*eye(n));
    C1 = zeros(n,1);
    Lambda2 = zeros(n,1);
    err1 = 10*thr; %err2 = 10*thr2;
    iter = 1;
    % ADMM iterations
    while ( err1 > thr && iter < maxIter )
        % updating Z
        Z = A * (mu1*(D'*Y(:,i))+mu2*(C1-Lambda2/mu2));
%         Z = Z - diag(diag(Z));
        % updating C
        C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(n,1))) .* sign(Z+Lambda2/mu2);
        %C2 = C2 - diag(diag(C2));
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 * (Z - C2);
        % computing errors
        err1 = errorCoef(Z,C2);
%         err2(i+1) = errorLinSys(Y,Z);
        %
        C1 = C2;
        iter = iter + 1;
    end
    % update the i-th column of the coefficient matrix
    C(Idx,i) = C1;
end
end

