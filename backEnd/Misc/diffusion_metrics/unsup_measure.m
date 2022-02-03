function [sil, dbi, dunn] = unsup_measure(clusteringIdx, Clusterings,label)
G = Clusterings.Graph;
Hyperparameters = Clusterings.Hyperparameters;
t = Clusterings.TimeSamples(clusteringIdx);
Dt = Clusterings.Dt;
Y = Clusterings.Labels(:,clusteringIdx);
 
Y = Y(label~=1);
eigenVecs = G.EigenVecs(label~=1,:);
Dt = Dt(label~=1,clusteringIdx);

if ~isfield(Hyperparameters, 'NEigs')
    Hyperparameters.NEigs = size(G.EigenVecs,2);
end
if ~isfield(Hyperparameters, '')
    Hyperparameters.DtNNs = 100;
end

n = length(Y);

%% Calculate diffusion map
DiffusionMap = zeros(n,Hyperparameters.NEigs);
for l = 1:size(DiffusionMap,2)
    DiffusionMap(:,l) = eigenVecs(:,l).*(G.EigenVals(l).^t);
end

%% Silhouette using diffusion distances
evas = evalclusters(DiffusionMap, Y, 'silhouette');
sil = evas.CriterionValues;

% Davies-Bouldin Index using diffusion distances to centroids
% evad = evalclusters(DiffusionMap, Y, 'DaviesBouldin');
% dbi = evad.CriterionValues;

%% Davies-Bouldin Index using diffusion distances to modes
[~,I] = sort(Dt,'descend');
Imodes = I(1:Hyperparameters.K_Known);
[dbi,~] = db_index(DiffusionMap, Y, DiffusionMap(Imodes,:));

%% Dunn's Index
dist_in = zeros(1,n);
dist_btw = zeros(1,Hyperparameters.K_Known);
for clusteringIdx = 1:n
    dist_in(clusteringIdx) = max(pdist2(DiffusionMap(clusteringIdx,:),DiffusionMap(Y==Y(clusteringIdx),:)));
%     k = convhulln(DiffusionMap, {'QJ'});
%     dist_in(i) = max(pdist2(k,k));
end

for k = 1: Hyperparameters.K_Known
    [~, dist_temp] = knnsearch(DiffusionMap(Y==k,:),DiffusionMap(Y~=k,:), 'K', 2);
    if size(dist_temp,2)==2
        dist_temp = dist_temp(:,2);
        dist_btw(k) = min(dist_temp~=0);
    end
end
dt_in = max(dist_in);
dt_btw = min(dist_btw);
dunn = dt_in/dt_btw;
% dunn = 1;


end
