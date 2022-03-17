function C = evaluateGMMPCA(X,K)
%{
This function evaluates Gaussian Mixture Model (GMM) Clustering after
principal component analysis (PCA). 

The number of PCs is initialized as the z such that 99 % of the variation
in the data is encoding in the low-dimensional embedding. If GMM does not
converge on this embedding in any of 10 attempts, we increase the number of
iterations in its optimization and begin reducing the number of PCs
implemented on. 

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

(c) Copyright Sam L. Polk, Tufts University, 2022.
%}

% We run the same code (at most) 10 times and keep the clustering
% that is the first to converge. 
maxNumTries = 10;
options = statset('MaxIter',200);
iter = 1;
keepRunning = true;
while iter<maxNumTries && keepRunning 
    try
        tic
        [COEFF, SCORE, ~, ~, pctExplained] = pca(X);
        nPCs = find(cumsum(pctExplained)>99,1,'first');
        XPCA = SCORE(:,1:nPCs);

        C = cluster(fitgmdist(XPCA,K, 'Options', options), XPCA);
        keepRunning = false;
    catch
    end
    iter = iter+1;
end
subtractNum = 1;
while ~exist('C') && subtractNum<nPCs-1
    % Too many PCs
    options = statset('MaxIter',1000); 
    try
        C = cluster(fitgmdist(XPCA(:,1:end-subtractNum),K, 'Options', options), XPCA(:,1:end-subtractNum));
    catch
        subtractNum = subtractNum + 1;
    end
end
if  ~exist('C')
    C = randi(K, length(X),1);
end
