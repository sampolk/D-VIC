function [missrate,grp] = spectral_clustering(W,n,gtruth)
%{ 
This function performs spectral clustering using modified code from the 
author's online repository: https://github.com/sjtrny/kssc. Modifications
have been indicated where they appear.
%}
W(isnan(W))=0;
if (n == 1)
    gtruth = ones(1,size(W,1));
end

MAXiter = 1000;
REPlic = 20;
N = size(W,1);
n = max(gtruth);

% cluster the data using the normalized symmetric Laplacian 
D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;

% =========================================================================
% Sam L. Polk (samuel.polk@tufts.edu) added the following code to improve
% convergence of eigendecomposition and memory allocation
if N >= 1e4
    [~,~,V] = svds(L,n,'largest', 'SubspaceDimension', 30);
else
    [~,~,V] = svd(L,'econ');
end
[~, msgid] = lastwarn;
if strcmp(msgid, 'MATLAB:svds:BadResidual')
    [V,~] = eigs(D*W,n,'largestabs');
end
clear L 
% =========================================================================

Yn = V(:,end:-1:end-n+1);

% =========================================================================
% Sam L. Polk (samuel.polk@tufts.edu) added the following code to improve
% memory allocation.
clear V
% =========================================================================

for i = 1:N
    Yn(i,:) = Yn(i,1:n) ./ norm(Yn(i,1:n)+eps);
end

if n > 1
    grp = kmeans(Yn(:,1:n),n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
else 
    grp = ones(1,N);
end

% compute the misclassification rate if n > 1
% missrate = missclassGroups(grp,gtruth) ./ length(gtruth); 
grp=bestMap(gtruth,grp);
missrate = sum(grp~=gtruth)/length(gtruth);