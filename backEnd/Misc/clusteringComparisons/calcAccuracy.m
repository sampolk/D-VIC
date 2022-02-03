function [ OA, kappa, AA, ProdAcc, UserAcc] = calcAccuracy(Y, C)

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
OA = sum(diag(confMat)/length(C)); 

p=nansum(confMat,2)'*nansum(confMat)'/(nansum(nansum(confMat)))^2;
kappa=(OA-p)/(1-p);

% #Calculate Producer Accuracies
ProdAcc = diag(confMat)./sum(confMat,2);
mask = isnan(ProdAcc);
ProdAcc(mask) = [];
UserAcc = diag(confMat)./(sum(confMat,1)');
UserAcc(mask) = [];

AA = mean(ProdAcc);

end



