%% RunGridSearches

LUNDGS
DVISGS
KMeansGS
KMeansPCAGS
SCGS
H2NMFGS
DBSCANGS  
SymNMFGS
GMMPCAGS 
KNNSSCGS

%% Aggregate into a table
clear


kappaTable = zeros(3,10);
OATable = zeros(3,10);
Clusterings = cell(3,10);
hyperparameters = cell(3,10);

datasets = {'SalinasACorrected','IndianPinesCorrected', 'syntheticHSI5149Stretched'};

for i = 1:3

    load(datasets{i})

    algs = cell(1,10);
    ending = strcat('Results', datasets{i});

    % Algorithms without hyperparameter optimization 
    algsTemp= {'KMeans', 'KmeansPCA', 'GMM','H2NMF'};
    idces = [1,2,3, 5]; 
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
        
    % Algorithms with 1 hyperparameter to optimize
    algsTemp= {'KNNSSC'};
    idces = [8]; 
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
    idces = [6,7]; 
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
    algsTemp= {'DBSCAN', 'LUND'};
    idces = [4, 9]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending))
        [OATable(i,idces(j)),k] = max(OAs, [],'all');
        [l,k] = ind2sub(size(OAs), k);
        kappaTable(i,idces(j)) = kappas(l,k);
        algs{idces(j)} = algsTemp{j};
        Clusterings{i,idces(j)} = Cs(:,l,k);


        if j == 2
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
        else
            hyperparameters{i,idces(j)} = array2table([minPtVals(l), prctile(Dist_NN(Dist_NN>0), prctiles(k), 'all')], 'VariableNames', {'MinPts', 'Epsilon'});
        end

%         catch
%         end
    end
    
    % Algorithms with 2 hyperparameters to optimize, but with averages
    algsTemp= {'DVIS'};
    idces = [10]; 
    for j = 1:length(algsTemp)
%         try
        load(strcat(algsTemp{j}, ending, 'ManyAVMAXFinalized'))
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

OATable = array2table(OATable, 'VariableNames',algs, 'RowNames',{'SalinasA', 'IndianPines','Synthetic'});
kappaTable = array2table(kappaTable, 'VariableNames',algs, 'RowNames',{'SalinasA', 'IndianPines','Synthetic'});

save('results', 'kappaTable', 'OATable', 'Clusterings', 'algs', 'datasets', 'hyperparameters')

%% Visualize Clusterings and Ground Truth

load('results.mat')

algsFormal = {'$K$-Means', '$K$-Means+PCA', 'GMM+PCA', 'DBSCAN','H2NMF','SC','SymNMF','KNN-SSC','LUND','D-VIS'};
datasetsFormal = {'Salinas A', 'Indian Pines', 'Synthetic Data'};

for i = 1:3

    load(datasets{i})
    h = figure;
    imagesc(GT)
    xticks([])
    yticks([])
    axis equal tight
    title([datasetsFormal{i}, ' Ground Truth'], 'interpreter','latex', 'FontSize', 17) 
     
    saveas(h, strcat(datasets{i}, 'GT'), 'epsc')

    h = figure;
    [~,scores] = pca(X);
    imagesc(reshape(scores(:,1), M,N))
    a = colorbar;
    a.Label.String = 'OA (%)';
    xticks([])
    yticks([])
    axis equal tight
    title(['First Principal Component Scores'], 'interpreter','latex', 'FontSize', 17) 
    set(gca,'FontName', 'Times', 'FontSize', 14)
     
    saveas(h, strcat(datasets{i}, 'PC'), 'epsc')

    for j = 1:10

        U = reshape(alignClusterings(Y,Clusterings{i,j}), M,N);
        if i <=2 
            if ~(j==4)
                U(GT == 1) = 0;
            else
                U(U==0) = -1;
                U(GT == 1) = 0;
            end
        else
            if ~(j==4)
                U(GT == 0) = 0;
            else
                U(U==0) = -1;
                U(GT == 1) = 0;
            end
        end 

        h = figure;
        imagesc(U)
        v = colormap;
        if j == 4
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

means = zeros(3,1);
CIs = zeros(3,2);

for i = 1:3
    load(datasets{i})

    ending = strcat('Results', datasets{i});
    load(strcat('DVIS', ending, 'ManyAVMAXFinalized'))
    prctiles = prctiles(end-19:end);
    prctiles(end) = 99.5;

    mat = mean(OAs,3);
    mat = mat(:,end-19:end);

    DistTemp = Dist_NN(Dist_NN>0);
    sigmas = zeros(10,1);
    for j = 1:10
        sigmas(j) = prctile(DistTemp, prctiles(2*j), 'all');
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
    yticks([1,5,8,12, 15, 19])
    yticklabels(NNs([1,5,8,12, 15, 19]))
    axis tight

    a = colorbar;
    a.Label.String = 'OA (%)';

    set(gca,'FontSize', 14, 'FontName', 'Times')
    title(['D-VIS Performance on ' datasetsFormal{i}], 'interpreter','latex', 'FontSize', 17) 
    saveas(h, strcat(datasets{i}, 'Robustness'), 'jpeg')

    
    x = reshape(mat, numel(mat),1);
    means(i) = mean(x);
    SEM = std(x)/sqrt(length(x));               % Standard Error
    ts = tinv([0.025  0.975],length(x)-1);      % T-Score
    CIs(i,:) = means(i) + ts*SEM;                      % Confidence Intervals


end
close all 









