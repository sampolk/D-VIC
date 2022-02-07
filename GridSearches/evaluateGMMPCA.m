function C = evaluateGMMPCA(X,K)

maxNumTries = 10;
options = statset('MaxIter',200);
iter = 1;
keepRunning = true;
while iter<maxNumTries && keepRunning 
    try
        tic
        [~, SCORE, ~, ~, pctExplained] = pca(X);
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
    C = randi(K, length(X), 1);
end