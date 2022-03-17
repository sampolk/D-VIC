%{

This script runs the grid searches outlined in Appendix A of the following
article:

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

The optimal hyperparameters for each algorithm are then used to visualize 
clusterings in Figures 6-7 and provide performances and runtimes in Tables 
I and II respectively. D-VIC is shown to substantially outperform related 
hypserspectral image clustering algorithms on 3 real datasets after 
hyperparameter optimization. 

To run this script, real hyperspectral image data (Salinas A, Indian Pines, 
& Jasper Ridge) must be downloaded from the following links:

    - http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    - https://rslab.ut.ac.ir/data

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}
%% RunGridSearches

LUNDGS
DVISGS
SCGS
SymNMFGS
KMeansGS
KMeansPCAGS
GMMPCAGS 
KNNSSCGS

%% Aggregate into a table
clear


kappaTable = zeros(3,8);
OATable = zeros(3,8);
Clusterings = cell(3,8);
hyperparameters = cell(3,8);

datasets = {'SalinasACorrected','IndianPinesCorrected', 'JasperRidge'};

for i = 1

    if i == 4
        load('Pavia_gt')
        load('Pavia.mat')
        HSI = pavia(201:400, 430:530,:);
        GT = pavia_gt(201:400, 430:530);
    else
        load(datasets{i})
    end

    algs = cell(1,8);
    ending = strcat('Results', datasets{i});

    % Algorithms without hyperparameter optimization 
    algsTemp= {'KMeans', 'KmeansPCA', 'GMM'};
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
    idces = [6]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(OAs);
        kappaTable(i,idces(j)) = median(kappas(k));
        algs{idces(j)} = algsTemp{j};
        Clusterings{i,idces(j)} = Cs(:,k);
        hyperparameters{i,idces(j)} = array2table([10, NNs(k)], 'VariableNames', {'Alpha','NN'});
%         catch
%         end
    end
    
    % Algorithms with 1 hyperparameter to optimize, but with averages
    algsTemp= {'SC', 'SymNMF'};
    idces = [4,5]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(median(OAs,2));
        kappaTable(i,idces(j)) = median(kappas(k,:));
        algs{idces(j)} = algsTemp{j};
        [~,l] = min(abs(OATable(i,idces(j))-OAs(k,:)));
        Clusterings{i,idces(j)} = Cs(:,k,l);

        hyperparameters{i,idces(j)} = array2table(NNs(l), 'VariableNames', {'NN'});


%         catch
%         end
    end
    
    % Algorithms with 2 hyperparameters to optimize
    algsTemp= {'LUND'};
    idces = [7]; 
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
    idces = [8]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending, '50'))
        [OATable(i,idces(j)),k] = max(median(OAs,3), [],'all');
        [l,k] = ind2sub(size(median(OAs,3)), k);
        kappaTable(i,idces(j)) = median(kappas(l,k,:));
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

table = zeros(8,6);
table(:,1:2:5) = OATable';
table(:,2:2:6) = kappaTable'; 

table = array2table(round(table,3), 'RowNames', algs, 'VariableNames',{'SalinasAOA','SalinasAKappa', 'IndianPinesOA','IndianPinesKappa','JasperRidgeSubsetOA','JasperRidgeSubsetKappa'} );


save('results', 'table', 'Clusterings', 'algs', 'datasets', 'hyperparameters')

%% Visualize Clusterings and Ground Truth

% load('results.mat')

algsFormal = {'$K$-Means', '$K$-Means+PCA', 'GMM+PCA', 'H2NMF','SC','SymNMF','KNN-SSC','LUND','D-VIC'};
datasetsFormal = {'Salinas A', 'Indian Pines', 'Jasper Ridge'};

for i = 1

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

    for j = 3

        if i<2
            U = zeros(M*N,1);
            U(Y>1) = alignClusterings(Y(Y>1)-1,Clusterings{i,j}(Y>1));
            U = reshape(U,M,N)+1;
        else 
            U = reshape(alignClusterings(Y,Clusterings{i,j}), M,N);
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

datasets = {'SalinasACorrected',  'JasperRidge','IndianPinesCorrected',  'syntheticHSI5149Stretched'};
datasetNames = {'Salinas A',      'Jasper Ridge',   'Indian Pines',           'Synthetic HSI'};

for i =  2

    % ===================== Load and Preprocess Data ======================
    [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(datasetNames{i});
    [Idx_NN, Dist_NN] = knnsearch(X,X,'K',1000);
    Idx_NN(:,1)  = []; 
    Dist_NN(:,1) = [];  
    ending = strcat('Results', datasets{i});
    load(strcat('DVIS', ending, '50'))

    mat = median(OAs,3); 
    mat = mat(:,11:30);

    DistTemp = Dist_NN(Dist_NN>0);
    sigmas = zeros(10,1);
    for j = 1:10
        sigmas(j) = prctile(DistTemp, min(prctiles(j), 99.5), 'all');
    end

    set(groot,'defaultAxesTickLabelInterpreter','latex');  % Enforces latex x-tick labels
    h = figure;

    imagesc(mat')
    yticks(2:2:20)
    exponents = floor(log10(sigmas));
    val = sigmas()./(10.^exponents)    ;
    labels = cell(10,1);
    for k = 1:10
        labels{k} = strcat('$',num2str(val(k), 3) ,'\times 10^{' , num2str(exponents(k)), '}$');
    end    
    yticklabels(labels)
    
    ylabel('$\sigma_0$', 'interpreter', 'latex')
    xlabel('$N$', 'interpreter', 'latex')
    xticks(2:2:10)
    xticklabels(NNs(2:2:10))
    axis tight equal

    a = colorbar;
    a.Label.String = 'OA';

    set(gca,'FontSize', 18, 'FontName', 'Times')
    title([datasetNames{i}], 'interpreter','latex', 'FontSize', 20) 
    saveas(h, strcat(datasets{i}, 'Robustness'), 'epsc')
end
close all 

