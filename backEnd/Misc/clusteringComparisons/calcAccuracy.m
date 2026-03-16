function [ OA, kappa, tIdx, OATemp, kappaTemp, NMI, AMI, ARI, FMI, purity, alignedLabels, mapping] = calcAccuracy(Y, C, ignore1flag)
%{
Calculates a variety of clustering statistics.

Inputs: Y (GT Labels), C (Estimated Clustering), and ignore1flag (1 if Y=1
        class is ignored in performance calculations).

Outputs: OA (Overall Accuracy), kappa (Cohen's kappa), tIdx (optimal
         clustering index), OATemp (OA values for each tIdx), kappaTemp
         (kappa values for each tIdx), and additional clustering metrics
         NMI (normalized mutual information), AMI (adjusted mutual
         information), ARI (adjusted Rand index), FMI (Fowlkes-Mallows
         index), purity, the aligned labels (alignedLabels) and the mapping
         from predicted clusters to true labels (mapping).

C may be one of the following formats:

    - Structure with "Labels" field that is an nxM array with a clustering
    of X in each column.
    - an nx1 clustering of X.

Copyright: Sam L. Polk (2022). Modified 2024.
%}
if isstruct(C)

    numC = size(C.Labels,2);
    OATemp = zeros(numC,1);
    kappaTemp = zeros(numC,1);
    NMITemp = zeros(numC,1);
    AMITemp = zeros(numC,1);
    ARITemp = zeros(numC,1);
    FMITemp = zeros(numC,1);
    purityTemp = zeros(numC,1);
    alignedLabelsCell = cell(numC,1);
    mappingCell = cell(numC,1);
    for i = 1:numC
        [ OATemp(i), kappaTemp(i), NMITemp(i), AMITemp(i), ARITemp(i), FMITemp(i), purityTemp(i), alignedLabelsCell{i}, mappingCell{i}] = evaluatePerformances(Y, C.Labels(:,i), ignore1flag);
    end
    [OA, tIdx] = max(OATemp);
    kappa = kappaTemp(tIdx);
    NMI = NMITemp(tIdx);
    AMI = AMITemp(tIdx);
    ARI = ARITemp(tIdx);
    FMI = FMITemp(tIdx);
    purity = purityTemp(tIdx);
    alignedLabels = alignedLabelsCell{tIdx};
    mapping = mappingCell{tIdx};
else
    % Single clustering (no need to run over multiple clusterings)
    [ OA, kappa, NMI, AMI, ARI, FMI, purity, alignedLabels, mapping] = evaluatePerformances(Y, C, ignore1flag);
    tIdx =1;
    OATemp = OA;
    kappaTemp = kappa;
end

end

function [OA, kappa, NMI, AMI, ARI, FMI, purity, alignedLabels, mapping] = evaluatePerformances(Y,C, ignore1flag)
    if ignore1flag
        % If true, we restric performance evaluation to unlabeled points (those
        % marked as index 1).

        % Perform hungarian algorithm to align clustering labels
        CNew = C(Y>1);

        missingk = setdiff(1:max(CNew), unique(CNew)');
        if length(missingk) == 1
            CNew(CNew>=missingk) = CNew(CNew>=missingk) - 1;
        else
            Ctemp = zeros(size(CNew));
            uniqueClass = unique(CNew);
            actualK = length(uniqueClass);
            for k = 1:actualK
            Ctemp(CNew==uniqueClass(k)) = k;
            end
            CNew =Ctemp;
        end

        [CAligned, mapping] = alignClusterings(Y(Y>1)-1,CNew);
        C = CAligned;

        % Implement performance calculations
        confMat = confusionmat(Y(Y>1)-1,C);
        labelsTrue = Y(Y>1)-1;
        alignedLabels = zeros(size(Y));
        alignedLabels(Y>1) = C;
    else
        % We consider the entire dataset

        [C, mapping] = alignClusterings(Y,C);

        % Implement performance calculations
        confMat = confusionmat(Y,C);
        labelsTrue = Y;
        alignedLabels = C;
    end

    n = sum(confMat(:));
    OA = sum(diag(confMat))/n;

    p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
    kappa=(OA-p)/(1-p);

    labelsPred = C;
    [NMI, MI, Hu, Hv] = nmi(labelsTrue, labelsPred);
    EMI = expectedMutualInformation(confMat, n);
    denom = ((Hu + Hv)/2) - EMI;
    if denom == 0
        AMI = 0;
    else
        AMI = (MI - EMI)/denom;
    end

    a = sum(confMat,2);
    b = sum(confMat,1);
    tp = sum(sum(confMat.*(confMat-1)/2));
    sumA = sum(a.*(a-1)/2);
    sumB = sum(b.*(b-1)/2);
    fp = sumB - tp;
    fn = sumA - tp;
    denomARI = (0.5*(sumA + sumB) - (sumA*sumB)/(n*(n-1)/2));
    if denomARI == 0
        ARI = 0;
    else
        ARI = (tp - (sumA*sumB)/(n*(n-1)/2)) / denomARI;
    end

    denomFMI = sqrt((tp+fp)*(tp+fn));
    if denomFMI == 0
        FMI = 0;
    else
        FMI = tp / denomFMI;
    end

    purity = sum(max(confMat,[],1))/n;

end

function EMI = expectedMutualInformation(confMat, n)
    a = sum(confMat,2);
    b = sum(confMat,1);
    EMI = 0;
    for i = 1:length(a)
        for j = 1:length(b)
            maxnij = max(1, a(i) + b(j) - n);
            minnij = min(a(i), b(j));
            if maxnij > minnij
                continue;
            end
            for nij = maxnij:minnij
                term1 = (nij/n) * log((n*nij)/(a(i)*b(j)));
                logTerm2 = gammaln(a(i)+1) - gammaln(nij+1) - gammaln(a(i)-nij+1) + ...
                           gammaln(n-a(i)+1) - gammaln(b(j)-nij+1) - gammaln(n-a(i)-b(j)+nij+1) - ...
                           (gammaln(n+1) - gammaln(b(j)+1) - gammaln(n-b(j)+1));
                term2 = exp(logTerm2);
                EMI = EMI + term1*term2;
            end
        end
    end
end
