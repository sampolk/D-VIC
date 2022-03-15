
%% 

[X,Y] = twomoons(500);

%% 
h = figure;
scatter(X(:,1), X(:,2),36, Y, 'filled')
set(gca,'FontSize', 20, 'FontName', 'Times')
title('Dataset, Colored By Ground Truth Label', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
ylabel('$x_2$', 'interpreter', 'latex')

saveas(h, 'twoMoonsGT', 'epsc')

h = figure;
scatter(X(:,1), X(:,2),36, alignClusterings(Y,kmeans(X,2, 'Replicates', 100 )), 'filled')
set(gca,'FontSize', 20, 'FontName', 'Times')
title('Dataset, Colored By $K$-Means Cluster', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
ylabel('$x_2$', 'interpreter', 'latex')

saveas(h, 'twoMoonsKMeans', 'epsc')

h = figure;
options = statset('MaxIter',200);
scatter(X(:,1), X(:,2),36,  alignClusterings(Y,cluster(fitgmdist(X,2,'Options', options, 'Replicates', 100), X)), 'filled')
set(gca,'FontSize', 20, 'FontName', 'Times')
title('Dataset, Colored By GMM Cluster', 'interpreter', 'latex')
axis equal tight
box on 
xlabel('$x_1$', 'interpreter', 'latex')
ylabel('$x_2$', 'interpreter', 'latex')

saveas(h, 'twoMoonsGMM', 'epsc')

close all 