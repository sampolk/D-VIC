function [ Z ] = PKN( data, anchors, BPoints, APoints, k, sz, alpha, mflag)
    if(mflag==0)
        dd = Edistance(data, anchors);
    elseif(mflag==1)
        dd = Edistance(data, anchors) + Edistance(BPoints, APoints).*alpha;
    elseif(mflag==2)
        NData = reshape(data,sz);
        conv = [1,1,1;1,1,1;1,1,1]./9;
        for i =1:sz(3)
            MeanData(:,:,i) = filter2(conv, NData(:,:,i));
        end
        MeanData = reshape(MeanData, [sz(1)*sz(2),sz(3)]);
        dd = Edistance(data, anchors) + Edistance(MeanData, anchors).*alpha;
    end
    dd = real(dd);
    dd = max(dd,0);
    [dumb, idx] = sort(dd, 2);
    Z = zeros([size(data,1), size(anchors,1)]);
    for i = 1:size(data,1)
        if(k+1>=size(idx,2))
            k = size(idx,2)-1;
        end
        id = idx(i,1:k+1);
        di = dd(i, id);
        Z(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end;
end

