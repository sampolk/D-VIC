%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns the regularization constant of the L1 norm
% Y: DxN data matrix
% lambda: regularization parameter for lambda*||C||_1 + 0.5 ||Y-YC||_F^2
% lambda is chosen to be: 
% lambda = min_{y\in X, y≠x} max_{x\in X} |y^Tx|
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function lambda = computeLambda_mat(Y,P)

if (nargin < 2)
    P = Y;
end

N = size(Y,2);
T = P' * Y; % nxn matrix of inner products
T(1:N,:) = T(1:N,:) - diag(diag(T(1:N,:))); % set zero-value at diagonal entries
T = abs(T); 
lambda = min(max(T,[],1)); 
