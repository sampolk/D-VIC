% Align clusters in Clustering2 to maximize overall accuracy with
% Clustering1.  Uses the Hungarian algorithm as implemented by Yi Cao.  

function [NewLabels,IdxOptimal]=AlignClustersHungarian(Clustering1,Clustering2,K)

% Input:
%
% -Clustering1: one assignment of labels of a dataset
% -Clustering2: another assignment of labels of the same dataset as
% clustering1
% -K: number of clusters; should be the same for Clustering1 and
% Clustering2
%
% Output:
%
% -NewLabels: new labels of Clustering2, designed to maximize correspondence with
% clustering 1
% -IdxOptimal: permuation maximizing the alignment
%

NewLabels=zeros(size(Clustering2));

%% Set the labels in the two clusterings to be the same, if necessary

LabelNames1=unique(Clustering1);
LabelNames2=unique(Clustering2);
LabelsUse=setdiff(LabelNames1,LabelNames2);
LabelsChange=setdiff(LabelNames2,LabelNames1);

for i=1:length(LabelsChange)
    Clustering2(Clustering2==LabelsChange(i))=LabelsUse(i);
end

%% If number of input classes is different from number of labels, do nothing.

if K~=length(unique(Clustering1))
    NewLabels=[];
else
    
    Overlap=zeros(K,K);
    
    for i=1:K
        for j=1:K
            disp([i,j,length((Clustering1==i)), length((Clustering2==j))])
            Overlap(i,j)=sum((Clustering1==i).*(Clustering2==j));
        end
    end
    
    Overlap=Overlap';
            
end

%% Run the Hungarian algorithm

Cost=repmat(max(Overlap')',[1,K])-Overlap;
IdxOptimal = munkres(Cost);

for k=1:K
    NewLabels(find(Clustering2==k))=IdxOptimal(k);
end

