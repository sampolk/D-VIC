%%Compute the accuracy of a set of indices, idxs, with respect to the group
%%truth, GT.  OA is overall accuracy, AA is average accuracy, M is
%%confusion matrix

function [OA, AA, kappa, M, idxs] = GetAccuracies(idxs,GT,classes)

% Align indices with ground truth.

idxs=AlignClustersHungarian(GT,idxs,classes);
    
    for k=1:classes
        class_idxs{k}=squeeze(find(GT==k));
        class_size(k)=squeeze(length(class_idxs{k}));
        
        assignments{k}=idxs(class_idxs{k});
        numCorrect(k)=length(find(assignments{k}==k));
        CA(k)=numCorrect(k)/class_size(k);
        
        for r=1:classes
            M(k,r)=length(find(assignments{k}==r));
        end
        
    end
    AA=mean(CA);
    OA=sum(numCorrect)/sum(class_size);
    p=sum(M,2)'*sum(M)'/(sum(sum(M)))^2;
    kappa=(OA-p)/(1-p);
end

