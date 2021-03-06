function [X,M,N,D,HSI,GT,Y,n, K] = loadHSI(HSIName)

if strcmp(HSIName, 'Salinas A')

    HSI = load('SalinasA_corrected').salinasA_corrected;
    GT = load('SalinasA_gt').salinasA_gt;

elseif strcmp(HSIName, 'Indian Pines')

    HSI = load('Indian_pines_corrected.mat').indian_pines_corrected;
    GT = load('indian_pines_gt.mat').indian_pines_gt; 

elseif strcmp(HSIName, 'Jasper Ridge')

    HSI = reshape(load('jasperRidge2_R198.mat').Y', load('jasperRidge2_R198.mat').nRow, load('jasperRidge2_R198.mat').nCol, length(load('jasperRidge2_R198.mat').SlectBands));
    [~,Y] = max(load('end4.mat').A',[],2);
    GT = reshape(Y,  load('jasperRidge2_R198.mat').nRow, load('jasperRidge2_R198.mat').nCol);

elseif strcmp(HSIName, 'Synthetic HSI')

    genSyntheticData;

end

[M,N,D] = size(HSI);
n = M*N;
X = reshape(HSI,M*N,D);
X=X./repmat(sqrt(sum(X.*X,1)),size(X,1),1); 

if strcmp(HSIName, 'Salinas A')
    X = X + 10^(-6).*randn(size(X));
end

HSI = reshape(X, M,N,D);

% Correct GT labels
newGT = zeros(size(GT));
uniqueClass = unique(GT);
K = length(uniqueClass);
for k = 1:K
    newGT(GT==uniqueClass(k)) = k;
end
if ~strcmp(HSIName, 'Jasper Ridge') % Jasper Ridge has no unlabeled pixels
    K = K-1;
end
Y = reshape(newGT,M*N,1);
GT = newGT;
