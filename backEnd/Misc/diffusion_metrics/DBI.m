function out = DBI(X, C, exemplars)

K = length(unique(C));

if nargin == 2     
    % Calculate within-cluster means as cluster exemplars
    exemplars = zeros(K, size(X,2));
    
    for k = 1:K
        exemplars(k,:) = mean(X(C==k,:));
    end
end

% calculate between-modal distance
modalDistances = squareform(pwdist(exemplars));

averageSkew = zeros(K,1); 
for k = 1:K
    averageSkew(k) = mean(pdist2(exemplars(k,:), X(C==k,:)));
end
averageSkewMat = averageSkew + averageSkew';

out = max(averageSkewMat./modalDistances, [], 'all');

