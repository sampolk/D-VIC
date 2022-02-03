function [U, A, volume, loss] = mvcnmf_shell(X, kf)
%{
Inputs:
X is a nxD data matrix (with rows equal to pixel spectra). 
kf is the number of endmembers. 

Outputs:
U is a Dxkf matrix with each column encoding an endmember signature. 
A is a kfxn matrix with each column encoding endmember abundances. 
volume is the volume contained by the endmembers.
loss is the value of the MVC-NMF loss function after convergence.

%}

if size(X,1)/5 >size(X,2)
    % Probably a nxD data matrix. So we transpose. 
    X = X';
end 
[D,n] = size(X);

% remove noise by projecting onto the first kf PCs.
[UU, ~, ~] = svds(X,kf); % UU is Dxkf
Lowmixed = UU'*X; %
X = UU*Lowmixed; % X is still Dxn
 
% vca algorithm
% A_vca = endmembers
[A_vca, ~] = vca(X,'Endmembers', kf); % A_vca is Dxkf

% FCLS
warning off;
AA = [1e-5*A_vca;ones(1,length(A_vca(1,:)))]; % AA is (D+1)xkf
s_fcls = zeros(length(A_vca(1,:)),n); % s_fcls is kf x n
for l=1:n
    r = [1e-5*X(:,l); 1]; % r is (D+1)x1
%   s_fcls(:,j) = nnls(AA,r);
    s_fcls(:,l) = lsqnonneg(AA,r); % kfx1
end
% s_fcls = endmember abundances

% % random initialization
% idx = ceil(rand(1,c)*(M*N-1));
% Ainit = mixed(:,idx);
% sinit = zeros(c,M*N);

% PCA
%[PrinComp,meanData] = pca(mixed',0);')
meanData = mean(X');
[~,~,PrinComp] = svd(X'-meanData,'econ');

% test mvcnmf
tol = 1e-6;
maxiter = 150;
T = 0.015; 
                
% use vca to initiate
Ainit = A_vca;
sinit = s_fcls;

% use conjugate gradient to find A can speed up the learning
[U, A, volume, loss] = mvcnmf(X,Ainit,sinit,PrinComp,meanData,T,tol,maxiter,2,1); 