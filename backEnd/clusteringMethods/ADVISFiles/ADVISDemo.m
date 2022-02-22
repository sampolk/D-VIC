%% Load Data

load('SalinasACorrected.mat')
load('ADVISSalinasAHyperparameters')

%% Extract Pixel Purity

pixelPurity = zeros(M*N,1);
for k = 1:100

    [kf,~]=hysime(X'); % compute hysime to get best estimate for number of endmembers
    Hyperparameters.EndmemberParams.K = kf;
    Hyperparameters.EndmemberParams.Algorithm = 'VCA';
    pixelPurity = pixelPurity + (compute_purity(X,Hyperparameters))/100; % Run 100 times to ensure we get stable purity

end

%% Restrict Analysis to labeled points

% find indices of the labeled points
idx0 = find(Y==1); 
labeledPts = find(Y>1); 
n = length(labeledPts);

% Restrict to labeled pixels
X = X(labeledPts, :);
Y = Y(labeledPts);
pixelPurity = pixelPurity(labeledPts);

% Perform nearest neighbor searches
[Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
Dist_NN = Dist_NN(:,2:end); 
Idx_NN = Idx_NN(:,2:end);
 
%% Extract Graph

G = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

%% Extract KDE and zeta

density = KDE_large(Dist_NN, Hyperparameters);

%% Implement D-VIS

% Run algorithm
C_DVIS = DVIS(X, Hyperparameters, Hyperparameters.DiffusionTime, G, density, pixelPurity);

% Evaluate partition
NMI_DVIS = nmi(C_DVIS, Y);

%% Implement ADVIS for B=1:20

budgets = [0:5:100];

C_ADVIS = zeros(length(Y),length(budgets));
NMI_ADVIS = zeros(1,length(budgets));
for bIdx = 1:length(budgets)
    % Store budget in Hyperparameter structure
    Hyperparameters.numLabels = budgets(bIdx); 
    
    % Run ADVIS with that budget
    C_ADVIS(:,bIdx) = ADVIS(X, Hyperparameters, Hyperparameters.DiffusionTime, Y, G, density, pixelPurity);

    % Evaluate partition
    NMI_ADVIS(bIdx) = nmi(C_ADVIS(:,bIdx), Y); 
end

% Replicate Fig. 2
 
close all 
  
plot(budgets, NMI_ADVIS, 'LineWidth', 2)
hold on 
plot(budgets, NMI_DVIS*ones(length(budgets),1), 'LineWidth', 2)
xlabel('Budget, $B$', 'interpreter', 'latex')
xlim([min(budgets),max(budgets)])

ylabel('$NMI(\hat{\mathcal{C}}, \mathcal{C}_{GT})$', 'interpreter', 'latex')
ylim([0,1])
title('Recovery of Ground Truth Labels by ADVIS', 'interpreter', 'latex')

legend({'ADVIS', 'D-VIS'}, 'location', 'southeast')
pbaspect([1,1,1])
set(gca,'FontName', 'Times', 'FontSize', 20)

%% Replicate Fig. 3

Y1 = ones(M*N,1);
Y1(labeledPts) = Y;

subplot(2,2,1)
C = zeros(M*N,1);
C(labeledPts)  = C_DVIS;
C = alignClusterings(Y1,C);
eda(C)
title('D-VIS Partition', 'interpreter', 'latex')

subplot(2,2,2)
C = zeros(M*N,1);
C(labeledPts)  = C_ADVIS(:,10);
C = alignClusterings(Y1,C);
eda(C)
title(strcat('ADVIS Partition, $B=10$'), 'interpreter', 'latex')

subplot(2,2,3)
C = zeros(M*N,1);
C(labeledPts)  = C_ADVIS(:,15);
C = alignClusterings(Y1,C);
eda(C)
title(strcat('ADVIS Partition, $B=15$'), 'interpreter', 'latex')

subplot(2,2,4)
C = zeros(M*N,1);
C(labeledPts)  = C_ADVIS(:,20);
C = alignClusterings(Y1,C);
eda(C)
title(strcat('ADVIS Partition, $B=20$'), 'interpreter', 'latex')
