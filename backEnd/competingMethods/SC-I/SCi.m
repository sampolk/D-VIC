function IDX = SCi(mdata, nc, iternum, sz, mflag, sigma, beta)

lm = 1/3; 

LabMat = Edistance(mdata, mdata);
sigmaData = mean(mean(LabMat)) ./ sigma;
if(mflag==0)
    S = exp( -LabMat./sigmaData );
elseif(mflag==1)
    [X, Y] = meshgrid((1:sz(1)), (1:sz(2))');
    Points = zeros(size(X,1)*size(X,2),1);
    [Points(:,1), Points(:,2)] = deal(reshape(X',[],1), reshape(Y',[],1));
    DisMat = Edistance(Points, Points);
    sigmaDist = mean(mean(DisMat)) ./ sigma;
    S = exp(-((DisMat./sigmaDist).*beta+(LabMat./sigmaData)));
elseif(mflag==2)
    NData = reshape(mdata,sz);
    conv = [1,1,1;1,1,1;1,1,1]./9;
    MeanData=zeros(sz(1),sz(2),sz(3));
    for i =1:sz(3)
        MeanData(:,:,i) = filter2(conv, NData(:,:,i));
    end
    MeanData = reshape(MeanData, [sz(1)*sz(2),sz(3)]);
    MeanMat = Edistance(MeanData, MeanData);
    sigmaMean = mean(mean(MeanMat)) ./ sigma;
    S = exp(-((MeanMat./sigmaMean).*beta+(LabMat./sigmaData)));
else
    print('placeholder');
end
S = S - diag(diag(S));
Dn = sparse(sqrt(diag(1./sum(S))));
% Lpl = sparse( eye(size(S,1)) );
Lpr = Dn * S * Dn;
Sz = size(mdata,1);
rand('seed', 1234);
F = abs(rand(Sz, nc).*10);
for j = 1:iternum
    upPre = Lpr*F + 2*lm*F;
    downPre = F + 2*lm*F*(F'*F);
    F = F .* ( (upPre ./ downPre) );
end
IDX = kmeans(F, nc);
end