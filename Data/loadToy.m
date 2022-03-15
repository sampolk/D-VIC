function [X,M,N,D,HSI,GT,Y,n, K] = loadToy(clusterSize)

if nargin == 0
    clusterSize = 500;
end

endmembers = [[-1,0]; [1,0]; [0,sqrt(3)]].*sqrt(2/3);
endmembers = endmembers-mean(endmembers);
% gaussianCenterAbundances = [[1,0,0]; [[0.51,0.49, 0]; [0,1,0]; [0,0.51,0.49]; [0,0,1]; [0.49, 0, 0.51]]]; 
% 
% gaussianCenters = gaussianCenterAbundances*endmembers;
% 
% n = clusterSize*length(gaussianCenters);
% 
% X = zeros(n, 2);
% Y = zeros(n, 1);
% for i = 1:n    
%     X(i,:) = [-Inf,Inf];
%     while ~inpolygon(X(i,1),X(i,2), endmembers(:,1), endmembers(:,2))
%         k = ceil(i/clusterSize); % current cluster index
%         if mod(k,2)
%             X(i,:) = 0.1*randn(1,2)+ gaussianCenters(k,:);
%         else
%             X(i,:) = 0.02*randn(1,2)+ gaussianCenters(k,:);
%         end
%         [~,Y(i)] = min(pdist2(endmembers, X(i,:))); % Assign index of closest of closest endmember 
%     end
% end
% 
% HSI = reshape(X,clusterSize,6,2);
% [M,N,D] = size(HSI);
% GT = reshape(Y,M,N);
% K = 3;

gaussianCenterAbundances = [eye(3); zeros(1,3)];
gaussianCenters = gaussianCenterAbundances*endmembers;

n = clusterSize*(length(gaussianCenters)+1);

X = zeros(n, 2);
Y = zeros(n, 1);
for i = 1:n    
    X(i,:) = [-Inf,Inf];
    while ~inpolygon(X(i,1),X(i,2), endmembers(:,1), endmembers(:,2))
        k = min(ceil(i/clusterSize), 4); % current cluster index
        if k<4
            X(i,:) = 0.175*randn(1,2)+ gaussianCenters(k,:);
        else
            X(i,:) = 0.0175*randn(1,2)+ gaussianCenters(k,:);
        end
        [~,Y(i)] = min(pdist2(endmembers, X(i,:))); % Assign index of closest of closest endmember 
    end
end

HSI = reshape(X,n,1,2);
[M,N,D] = size(HSI);
GT = reshape(Y,M,N);
K = 3;