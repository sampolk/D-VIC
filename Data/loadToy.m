function [X,M,N,D,HSI,GT,Y,n, K] = loadToy(clusterSize)
%{

This function creates the synthetic dataset used to replicate Figures 3-4 
in the following article: 

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

This synthetic, 2-dimensional dataset has two classes of points:
high-density, low-purity points that are not indicative of latent material
structure and low-density, high-purity points that are. 

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}

if nargin == 0
    clusterSize = 500;
end

% Form ground truth endmember set
endmembers = [[-1,0]; [1,0]; [0,sqrt(3)]].*sqrt(2/3);
endmembers = endmembers-mean(endmembers);

% Create Gaussian centers 
gaussianCenterAbundances = [eye(3); zeros(1,3)]; % We have one Gaussian center near the origin.
gaussianCenters = gaussianCenterAbundances*endmembers;

% Number of data points, n
n = clusterSize*(length(gaussianCenters)+1); % +1 because the Gaussian at the origin is size clusterSize*2. 

% Preallocate memory
X = zeros(n, 2);
Y = zeros(n, 1);

% Extract dataset
for i = 1:n    
    % Initialize with a point that is clearly not in the triangle
    X(i,:) = [-Inf,Inf];

    % The following while-loop samples data points from a Gaussian
    % distribution until that data point is inside the polygon with
    % vertices equal to the ground truth endmembers. 
    while ~inpolygon(X(i,1),X(i,2), endmembers(:,1), endmembers(:,2))
        k = min(ceil(i/clusterSize), 4); % current cluster index
        if k<4
            % Gaussians with mean=endmembers have a larger standard  
            % deviation, so high-purity points are also relatively
            % low-density.
            X(i,:) = 0.175*randn(1,2)+ gaussianCenters(k,:);
        else
            % The Gaussian with mean=[0,0] has a larger standard deviation,
            % deviation, so low-purity points are also relatively
            % high-density.
            X(i,:) = 0.0175*randn(1,2)+ gaussianCenters(k,:);
        end
        % We assign the label according to the index of closest of closest 
        % endmember
        [~,Y(i)] = min(pdist2(endmembers, X(i,:)));  
    end
end

% For data processing purpose, we create a 3-dimensional object HSI with
% dimensions nx1x2.
HSI = reshape(X,n,1,2);
[M,N,D] = size(HSI);
GT = reshape(Y,M,N);
K = 3; % three ground truth classes, corresponding to the ground truth materials. 