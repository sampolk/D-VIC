function [ OA, kappa, tIdx, OATemp, kappaTemp] = calcAccuracy(Y, C, ignore1flag)

if isstruct(C)

    numC = size(C.Labels,2);
    OATemp = zeros(numC,1);
    kappaTemp = zeros(numC,1);
    for i = 1:numC
        [ OATemp(i), kappaTemp(i)] = evaluatePerformances(Y, C.Labels(:,i), ignore1flag);
    end
    [OA, tIdx] = max(OATemp);
    kappa = kappaTemp(tIdx);

else
    % Single clustering (no need to run over multiple clusterings
    [ OA, kappa] = evaluatePerformances(Y, C, ignore1flag);
    tIdx =1;
end

end

function [OA, kappa] = evaluatePerformances(Y,C, ignore1flag)
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
    
        C = alignClusterings(Y(Y>1)-1,CNew);
        
        % Implement performance calculations
        confMat = confusionmat(Y(Y>1)-1,C);
    
    else
        % We consider the entire dataset
    
         C = alignClusterings(Y,C);
        
        % Implement performance calculations
        confMat = confusionmat(Y,C);
    
    end
    
    OA = sum(diag(confMat)/length(C)); 
    
    p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
    kappa=(OA-p)/(1-p);
  
    
end
