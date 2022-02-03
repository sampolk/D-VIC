function [ dd ] = Edistance( data, anchors )
%EDISTANCE Summary of this function goes here
%   Detailed explanation goes here
    aa=sum(data.*data, 2); 
    bb=sum(anchors.*anchors, 2); 
    ab=data*anchors'; 
    dd = repmat(aa,[1 size(bb,1)]) + repmat(bb',[size(aa,1) 1]) - 2*ab;
end