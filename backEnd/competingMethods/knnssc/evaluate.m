function [ACC,NMI,ARI]=evaluate(GT,PR)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
ACC=100-sum(PR~=GT)/length(GT)*100;
[~,NMI]=AMI(GT,PR);
NMI = NMI * 100;
ARI=adjrand(PR,GT)*100;
end

