%{

This script replicates Figure 9 in the following article: 

    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

D-VIC is shown to achieve OA values comparable to the optimal value across
a broad range of choices of t on each of 3 real hyperspectral datasets. 

To run this script, real hyperspectral image data (Salinas A, Indian Pines, 
& Jasper Ridge) must be downloaded from the following links:

    - http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    - https://rslab.ut.ac.ir/data

(c) Copyright Sam L. Polk, Tufts University, 2022.

%}
%% Choose the dataset
clear
clc

profile off;
profile on;
prompt = 'Which dataset? \n 1) Salinas A \n 2) Jasper Ridge \n 3) Indian Pines \n';
datasetNames = {'Salinas A', 'Jasper Ridge', 'Indian Pines'};
dataSelectedName = datasetNames{input(prompt)};

% Load selected dataset
[X,M,N,D,HSI,GT,Y,n, K] = loadHSI(dataSelectedName);

% Load all optimal hyperparameter sets
Hyperparameters = loadHyperparameters(HSI, dataSelectedName, 'D-VIC');

% Nearest neighbor search
[Idx_NN, Dist_NN] = knnsearch(X,X,'K', max(Hyperparameters.DiffusionNN,Hyperparameters.DensityNN)+1);
Idx_NN(:,1)  = []; 
Dist_NN(:,1) = [];

disp('Dataset loaded.')

% Determine number of replicates for stochastic algorithms
profile off;
profile on;
prompt = 'Enter the number of desired runs for D-VIC: \n';
numReplicates = input(prompt);
if ~(round(numReplicates)-numReplicates == 0)
    error('The number of replicates must be an integer.')
elseif isempty(numReplicates)
    numReplicates = 1;
end
 
% Determine number of replicates for stochastic algorithms
profile off;
profile on;
prompt = 'Do you want to visualize results?\n 1) Yes \n 2) No \n';

visualizeOn = (input(prompt) == 1);

clc
profile off;
disp('Ready to Analyze HSI data.')

%% Run D-VIC 100 times and keep all clustering OAs

ts = 10:10:200;
numClusterings = length(ts);
OAs = zeros(numReplicates,numClusterings); % Where we will store the OA values produced at each t-value

for k = 1:numReplicates     

    % Graph decomposition
    G = extractGraph(X, Hyperparameters, Idx_NN, Dist_NN);
    
    % KDE Computation
    density = KDE(Dist_NN, Hyperparameters);

    % Spectral Unmixing Step
    Hyperparameters.EndmemberParams.K = hysime(X'); % compute hysime to get best estimate for number of endmembers
    pixelPurity = compute_purity(X,Hyperparameters);

    parfor tIdx = 1:numClusterings
        [C, K, Dt_temp] = LUND(X, Hyperparameters, ts(tIdx), G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
        OAs(k,tIdx) = calcAccuracy(Y,C,~strcmp('Jasper Ridge', dataSelectedName));
    end
end


%% Visualizations

if visualizeOn

    h = figure;
    plot(ts, median(OAs,1), 'LineWidth', 2, 'Color','k', 'Marker','diamond')

    axis tight 
    xlabel('$t$', 'interpreter', 'latex')
    ylabel('OA', 'interpreter', 'latex')
    if ~strcmp('Indian Pines', dataSelectedName)
        yticks(round(min(median(OAs)),1):0.05:round(max(median(OAs)),1))
    else
        ylim([0.4,max(median(OAs))])
        yticks(0.4:0.01:0.44)
    end
    grid on 

    tickIdces = 1:2:numClusterings;
     
    title(['D-VIC OA on ', dataSelectedName, ', as $t$ Varies'], 'interpreter', 'latex')
    pbaspect([4,1,1])
    set(gca,'FontSize', 14, 'FontName', 'Times')

    saveas(h, strcat('tAnalysis', erase(dataSelectedName,' ')), 'epsc')
 
end
