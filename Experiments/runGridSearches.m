%% RunGridSearches

% LUNDGS
% DVISGS
SCGS
SymNMFGS
KMeansGS
KMeansPCAGS
H2NMFGS
GMMPCAGS 
% KNNSSCGS

%% Aggregate into a table
clear


kappaTable = zeros(4,9);
OATable = zeros(4,9);
Clusterings = cell(4,9);
hyperparameters = cell(4,9);

datasets = {'SalinasACorrected','IndianPinesCorrected', 'JasperRidge', 'PaviaCenterSubset2'};

for i = 1:4

    if i == 4
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(201:400, 430:530,:);
        GT = pavia_gt(201:400, 430:530);
    else
        load(datasets{i})
    end

    algs = cell(1,9);
    ending = strcat('Results', datasets{i});

    % Algorithms without hyperparameter optimization 
    algsTemp= {'KMeans', 'KmeansPCA', 'GMM','H2NMF'};
    idces = [1,2,3, 4]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        OATable(i,idces(j)) = OA;
        kappaTable(i,idces(j)) = Kappa;
        Clusterings{i,idces(j)} = C;
        algs{idces(j)} = algsTemp{j};
%         catch
%         end
    end
        
%     % Algorithms with 1 hyperparameter to optimize
    algsTemp= {'KNNSSC'};
    idces = [7]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(OAs);
        kappaTable(i,idces(j)) = mean(kappas(k));
        algs{idces(j)} = algsTemp{j};
        Clusterings{i,idces(j)} = Cs(:,k);
        hyperparameters{i,idces(j)} = array2table([10, NNs(k)], 'VariableNames', {'Alpha','NN'});
%         catch
%         end
    end
    
    % Algorithms with 1 hyperparameter to optimize, but with averages
    algsTemp= {'SC', 'SymNMF'};
    idces = [5,6]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(mean(OAs,2));
        kappaTable(i,idces(j)) = mean(kappas(k,:));
        algs{idces(j)} = algsTemp{j};
        [~,l] = min(abs(OATable(i,idces(j))-OAs(k,:)));
        Clusterings{i,idces(j)} = Cs(:,k,l);

        hyperparameters{i,idces(j)} = array2table(NNs(l), 'VariableNames', {'NN'});


%         catch
%         end
    end
    
    % Algorithms with 2 hyperparameters to optimize
    algsTemp= {'LUND'};
    idces = [8]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(OAs, [],'all');
        [l,k] = ind2sub(size(OAs), k);
        kappaTable(i,idces(j)) = kappas(l,k);
        algs{idces(j)} = algsTemp{j};
        Clusterings{i,idces(j)} = Cs(:,l,k);


        Hyperparameters.SpatialParams.ImageSize = [M,N];
        Hyperparameters.NEigs = 10;
        Hyperparameters.NumDtNeighbors = 200;
        Hyperparameters.Beta = 2;
        Hyperparameters.Tau = 10^(-5);
        Hyperparameters.K_Known = length(unique(Y))-1; % We subtract 1 since we discard gt labels
        Hyperparameters.Tolerance = 1e-8;
        Hyperparameters.DiffusionNN = NNs(l);
        Hyperparameters.DensityNN = NNs(l); % must be ≤ 1000
        Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(k), 'all');

        hyperparameters{i,idces(j)} = Hyperparameters;
        

%         catch
%         end
    end
    
    % Algorithms with 2 hyperparameters to optimize, but with averages
    algsTemp= {'DVIS'};
    idces = [9]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending, 'ManyAVMAX'))
        [OATable(i,idces(j)),k] = max(mean(OAs,3), [],'all');
        [l,k] = ind2sub(size(mean(OAs,3)), k);
        kappaTable(i,idces(j)) = mean(kappas(l,k,:));
        algs{idces(j)} = algsTemp{j};
    
        [~,m] = min(abs(OATable(i,idces(j))-squeeze(OAs(l,k,:))));
        Clusterings{i,idces(j)} = Cs(:,l,k,m);

        Hyperparameters.SpatialParams.ImageSize = [M,N];
        Hyperparameters.NEigs = 10;
        Hyperparameters.NumDtNeighbors = 200;
        Hyperparameters.Beta = 2;
        Hyperparameters.Tau = 10^(-5);
        Hyperparameters.Tolerance = 1e-8;
        if i==3
            K = length(unique(Y))-1;
        else
            K = length(unique(Y));
        end
        Hyperparameters.K_Known = K; % We subtract 1 since we discard gt labels

        Hyperparameters.DiffusionNN = NNs(i);
        Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
        Hyperparameters.Sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
        if i==3
            Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
        else
            Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
        end
        Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
        Hyperparameters.EndmemberParams.NumReplicates = 100;

        hyperparameters{i,idces(j)} = Hyperparameters;
%         catch
%         end
    end
end

table = zeros(9,10);
table(:,1:2:7) = OATable';
table(:,2:2:8) = kappaTable';
table(:,9) = mean(OATable)';
table(:,10) = mean(kappaTable)';

table = array2table(round(table,3), 'RowNames', algs, 'VariableNames',{'SalinasAOA','SalinasAKappa', 'IndianPinesOA','IndianPinesKappa','JasperRidgeSubsetOA','JasperRidgeSubsetKappa','PaviaSubsetOA', 'PaviaSubsetKappa', 'OA_Avg', 'Kappa_Avg'} );


save('results', 'table', 'Clusterings', 'algs', 'datasets', 'hyperparameters')

%% Visualize Clusterings and Ground Truth

load('results.mat')

algsFormal = {'$K$-Means', '$K$-Means+PCA', 'GMM+PCA', 'H2NMF','SC','SymNMF','KNN-SSC','LUND','D-VIS'};
datasetsFormal = {'Salinas A', 'Indian Pines', 'Jasper Ridge', 'Pavia Subset'};

for i = 1:4

    if i == 4
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(201:400, 430:530,:);
        [M,N,D] = size(HSI);
        X = reshape(HSI,M*N,D);
        X=X./repmat(sqrt(sum(X.*X,1)),size(X,1),1); % Normalize HSI
        GT = pavia_gt(201:400, 430:530);
        Y = reshape(GT,M*N,1)+1;
    else
        load(datasets{i})
        GT = reshape(Y,M,N);
    end



    h = figure;
    if i == 4
        imagesc(GT')
    else
        imagesc(GT)
    end
    xticks([])
    yticks([])
    axis equal tight
    title([datasetsFormal{i}, ' Ground Truth'], 'interpreter','latex', 'FontSize', 17) 
     
    saveas(h, strcat(datasets{i}, 'GT'), 'epsc')

    h = figure;
    [~,scores] = pca(X);
    if i == 4
        imagesc(reshape(scores(:,1), M,N)')
    else
        imagesc(reshape(scores(:,1), M,N))
    end

    a = colorbar;
%     a.Label.String = 'OA (%)';
    xticks([])
    yticks([])
    axis equal tight
    title(['First Principal Component Scores'], 'interpreter','latex', 'FontSize', 17) 
    set(gca,'FontName', 'Times', 'FontSize', 14)
     
    saveas(h, strcat(datasets{i}, 'PC'), 'epsc')

    for j = setdiff(1:9,[])

        U = reshape(alignClusterings(Y,Clusterings{i,j}), M,N);
        if i < 3
            if ~(j==10)
                U(GT == 1) = 0;
            else
                U(U==0) = -1;
                U(GT == 1) = 0;
            end
        end
        if i == 4
            U = zeros(M*N,1);
            U(Y>1) = alignClusterings(Y(Y>1)-1,Clusterings{i,j}(Y>1));
            U = reshape(U,M,N);
            U = U';
        end

        h = figure;
        
        imagesc(U)
        
        v = colormap;
        if j == 10
            v(1,:) = [0,0,0];
            colormap(v);
        end
        xticks([])
        yticks([])
        axis equal tight
        title([algsFormal{j}, ' Clustering of ' datasetsFormal{i}], 'interpreter','latex', 'FontSize', 17) 
         
        saveas(h, strcat(datasets{i}, algs{j}), 'epsc')
    end
end
close all 

%% Hyperparameter Robustness

means = zeros(4,1);
CIs = zeros(4,2);

for i = 1:4

    if i == 4
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(201:400, 430:530,:);
        [M,N,D] = size(HSI);
        X = reshape(HSI,M*N,D);
        X=X./repmat(sqrt(sum(X.*X,1)),size(X,1),1); % Normalize HSI
        GT = pavia_gt(201:400, 430:530);
        Y = reshape(GT,M*N,1);
    else
        load(datasets{i})
        GT = reshape(Y,M,N);
    end

    ending = strcat('Results', datasets{i});
    load(strcat('DVIS', ending, 'ManyAVMAX'))

    mat = mean(OAs,3);

    DistTemp = Dist_NN(Dist_NN>0);
    sigmas = zeros(10,1);
    for j = 1:10
        sigmas(j) = prctile(DistTemp, min(prctiles(j), 99.5), 'all');
    end

    set(groot,'defaultAxesTickLabelInterpreter','latex');  % Enforces latex x-tick labels
    h = figure;

    imagesc(mat)
    xticks(2:2:20)
    exponents = floor(log10(sigmas));
    val = sigmas()./(10.^exponents)    ;
    labels = cell(10,1);
    for k = 1:10
        labels{k} = strcat('$',num2str(val(k), 2) ,'\times 10^{' , num2str(exponents(k)), '}$');
    end    
    xticklabels(labels)
    
    xlabel('$\sigma_0$', 'interpreter', 'latex')
    ylabel('$N$', 'interpreter', 'latex')
    yticks(2:2:10)
    yticklabels(NNs(2:2:10))
    axis tight equal

    a = colorbar;
    a.Label.String = 'OA';

    set(gca,'FontSize', 14, 'FontName', 'Times')
    title(['D-VIS Performance on ' datasetsFormal{i}], 'interpreter','latex', 'FontSize', 17) 
    saveas(h, strcat(datasets{i}, 'Robustness'), 'epsc')

    
    x = reshape(mat, numel(mat),1);
    means(i) = mean(x);
    SEM = std(x)/sqrt(length(x));               % Standard Error
    ts = tinv([0.025  0.975],length(x)-1);      % T-Score
    CIs(i,:) = means(i) + ts*SEM;                      % Confidence Intervals


end
close all 



%% Synthetic data visualization

load('syntheticHSI5149Stretched.mat')

h = figure;
imagesc(GT)
xticks([])
yticks([])
axis equal tight
title(['Synthetic HSI Ground Truth'], 'interpreter','latex', 'FontSize', 17) 
 
saveas(h, 'SyntheticGT', 'epsc')

h = figure;
[~,scores] = pca(X);
imagesc(reshape(scores(:,1), M,N))


a = colorbar;
%     a.Label.String = 'OA (%)';
xticks([])
yticks([])
axis equal tight
title(['First Principal Component Scores'], 'interpreter','latex', 'FontSize', 17) 
set(gca,'FontName', 'Times', 'FontSize', 14)
saveas(h, 'SyntheticPC', 'epsc')

close all 

%%

load('LUNDResultssyntheticHSI5149Stretched')

[OAstats(1),k] = max(OAs,[],'all');
[i,j] = ind2sub(size(OAs),k);
C = Cs(:,i,j);
C(Y == 0 ) = 0;
C = alignClusterings(Y, C);


h = figure;
imagesc(reshape(C, M,N))
xticks([])
yticks([])
axis equal tight
title(['Optimal LUND Clustering of Synthetic HSI'], 'interpreter','latex', 'FontSize', 17) 
 
saveas(h, 'SyntheticLUND', 'epsc')

clear C
load('DVISResultssyntheticHSI5149StretchedManyAVMAX')

[OAstats(2),k] = max(OAs,[],'all');
[i,j] = ind2sub(size(OAs),k);
C = Cs(:,i,j);
C(Y == 0 ) = 0;
C = alignClusterings(Y, C);

h = figure;
imagesc(reshape(C, M,N))
xticks([])
yticks([])
axis equal tight
title(['Optimal D-VIS Clustering of Synthetic HSI'], 'interpreter','latex', 'FontSize', 17) 
 
saveas(h, 'SyntheticDVIS', 'epsc')

close all 


disp(OAstats)

