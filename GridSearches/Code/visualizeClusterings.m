%% Preallocate memory

OATable = zeros(5,9);
KappaTable = zeros(5,9);

datasets = {'IndianPinesCorrected', 'JasperRidge', 'PaviaU', 'SalinasCorrected', 'SalinasACorrected'};


%% 1. K-Means

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('KMeansResults', datasets{dataIdx}))

    % Performances
    OATable(dataIdx,1) = OA;
    KappaTable(dataIdx,1) = Kappa;

    % Save optimal results
    save(strcat('KMeansClustering', datasets{dataIdx}), 'C')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('$K$-Means Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'KMeans');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all

%% 2. K-Means+PCA

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('KMeansPCAResults', datasets{dataIdx}))

    % Performances
    OATable(dataIdx,2) = OA;
    KappaTable(dataIdx,2) = Kappa;

    % Save optimal results
    save(strcat('KMeansPCAClustering', datasets{dataIdx}), 'C')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('$K$-Means+PCA Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'KMeansPCA');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all

%% 3. GMM+PCA

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('GMMResults', datasets{dataIdx}))

    % Performances
    OATable(dataIdx,3) = OA;
    KappaTable(dataIdx,3) = Kappa;

    % Save optimal results
    save(strcat('GMMClustering', datasets{dataIdx}), 'C')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('GMM+PCA Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'GMM');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all

%% 4. DBSCAN

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('DBSCANResults', datasets{dataIdx}))

    % Find optimal hyperparameters
    [OATable(dataIdx,4), k] = max(OAs(:));
    [i,j] = ind2sub(size(OAs), k);
    KappaTable(dataIdx,4) = kappas(i,j);
    minPts = minPtVals(i);
    epsilon = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
    C = Cs(:,i,j);

    % Save optimal results
    save(strcat('DBSCANClustering', datasets{dataIdx}), 'C', 'minPts','epsilon')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('DBSCAN Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'DBSCAN');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all



%% 5. SC

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('SCResults', datasets{dataIdx}))

    % Find optimal hyperparameters
    [OATable(dataIdx,5), k] = max(mean(OAs,2));
    KappaTable(dataIdx,5) = kappas(k);
    NN = NNs(k);
    [~,i] = min(abs(OAs(k,:) - mean(OAs(k,:))));
    C = Cs(:,k,i);

    % Save optimal results
    save(strcat('SCClustering', datasets{dataIdx}), 'C', 'NN')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('SC Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'SC');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all


%% 6. KNN-SSC
for dataIdx = [1,2, 5]
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('KNNSSCResults', datasets{dataIdx}))

    % Find optimal hyperparameters
    [OATable(dataIdx,6), k] = max(OAs);
    KappaTable(dataIdx,6) = kappas(k);
    NN = NNs(k);
    C = Cs(:,k);

    % Save optimal results
    save(strcat('KNNSSCClustering', datasets{dataIdx}), 'C', 'NN')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('KNN-SSC Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'KNNSSC');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all

%% 7. SymNMF

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('SymNMFResults', datasets{dataIdx}))

    % Find optimal hyperparameters
    [OATable(dataIdx,7), k] = max(mean(OAs,2));
    KappaTable(dataIdx,7) = kappas(k);
    NN = NNs(k);
    [~,i] = min(abs(OAs(k,:) - mean(OAs(k,:))));
    C = Cs(:,k,i);

    % Save optimal results
    save(strcat('SymNMFClustering', datasets{dataIdx}), 'C', 'NN')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('SymNMF Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'SymNMF');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end
close all

%% 8. H2NMF

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('H2NMFResults', datasets{dataIdx}))

    % Performances
    OATable(dataIdx,8) = OA;
    KappaTable(dataIdx,8) = Kappa;

    % Save optimal results
    save(strcat('H2NMFClustering', datasets{dataIdx}), 'C')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('H2NMF Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'H2NMF');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end

close all

%% 9. LUND

for dataIdx = 1:5
    
     % Load data
    load(datasets{dataIdx})

    % Load results
    load(strcat('LUNDResults', datasets{dataIdx}))

    % Find optimal hyperparameters
    [OATable(dataIdx,9), k] = max(OAs(:));
    [i,j] = ind2sub(size(OAs), k);
    KappaTable(dataIdx,9) = kappas(i,j);
    NN = NNs(i);
    sigma0 = prctile(Dist_NN(Dist_NN>0), prctiles(j), 'all');
    C = Cs(:,i,j);

    % Save optimal results
    save(strcat('LUNDClustering', datasets{dataIdx}), 'C', 'NN','sigma0')

    % Visualize clustering
    h = figure;
    eda(C, 0, Y)
    title('LUND Clustering', 'interpreter', 'latex', 'FontSize', 16)

    % Save Figure
    fileName = strcat(datasets{dataIdx}, 'LUND');
    save(fileName, 'C')
    savefig(h, fileName)
    saveas(h, fileName, 'epsc')   

end

close all

%% 

cols = {'KMeans', 'KMeansPCA', 'GMMPCA', 'DBSCAN', 'SC', 'KNN-SSC','SymNMF', 'H2NMF', 'LUND'};
rows = cellfun(@(x) erase(x,'Corrected'), datasets , 'UniformOutput', 0);

OATable = array2table(OATable, 'RowNames',rows, 'VariableNames',cols);
KappaTable = array2table(KappaTable, 'RowNames',rows, 'VariableNames',cols);

OATable.('KNN-SSC')(3:4) = NaN;
KappaTable.('KNN-SSC')(3:4) = NaN;

OATable(6,:) = array2table(nanmean(table2array(OATable)), 'VariableNames', cols, 'RowNames', {'Average'});
OATable.Properties.RowNames{6} = 'Average';

KappaTable(6,:) = array2table(nanmean(table2array(KappaTable)), 'VariableNames', cols, 'RowNames', {'Average'});
KappaTable.Properties.RowNames{6} = 'Average';


