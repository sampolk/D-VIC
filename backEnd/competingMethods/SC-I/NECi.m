function IDX = NECi(mdata, nc, scale, iterNum, sz, mflag)
sigma = 10;
lm = 1/2; 
beta = 3/5;

if(size(sz,1)==0) % for synthetic data sets
    member = zeros(size(mdata,1),1);
    member(1:scale:end)=1;
    member = logical(member);
else % for HSI data sets
    [X, Y] = meshgrid((1:sz(1)), (1:sz(2))');
    Points = zeros(size(X,1)*size(X,2),1);
    [Points(:,1), Points(:,2)] = deal(reshape(X',[],1), reshape(Y',[],1));
    [X, Y] = meshgrid((1:scale:sz(1)), (1:scale:sz(2))');
    APoints = zeros(size(X,1)*size(X,2),1);
    [APoints(:,1), APoints(:,2)] = deal(reshape(X',[],1), reshape(Y',[],1));
    BPoints = Points;
    member = ismember(BPoints, APoints, 'rows');
    BPoints(member,:) = [];
end
AData = mdata(member,:);
BData = mdata(~member,:);
ALabMat = Edistance(AData, AData);
BLabMat = Edistance(AData, BData);
sigmaData = mean(mean([ALabMat, BLabMat] / sigma));
if(mflag==0)
    A = exp( -ALabMat./sigmaData );
    B = exp( -BLabMat./sigmaData );
elseif(mflag==1)
    ADisMat = Edistance(APoints, APoints);
    BDisMat = Edistance(APoints, BPoints);
    sigmaDist = mean(mean([ADisMat, BDisMat] / sigma));
    A = exp(-((ADisMat./sigmaDist).*beta+(ALabMat./sigmaData)));
    B = exp(-((BDisMat./sigmaDist).*beta+(BLabMat./sigmaData)));
elseif(mflag==2)
    NData = reshape(mdata,sz);
    conv = [1,1,1;1,1,1;1,1,1]./9;
    for i =1:sz(3)
        MeanData(:,:,i) = filter2(conv, NData(:,:,i));
    end
    MeanData = reshape(MeanData, [sz(1)*sz(2),sz(3)]);
    AMeanData = MeanData(member,:);
    BMeanData = MeanData(~member,:);
    AMeanMat = Edistance(AMeanData, AMeanData);
    BMeanMat = Edistance(AMeanData, BMeanData);
    sigmaMean = mean(mean([AMeanMat, BMeanMat] / sigma));
    A = exp(-((AMeanMat./sigmaMean).*beta+(ALabMat./sigmaData)));
    B = exp(-((BMeanMat./sigmaMean).*beta+(BLabMat./sigmaData)));
else
    print('???');
end
d1 = sum([A; B'], 1);
d2 = sum(B,1) + sum(B,2)'*(A\B);
dhat = sqrt(1./[d1 d2])';
A = A.*(dhat(1:size(AData,1)) * dhat(1:size(AData,1))');
B = B.*(dhat(1:size(AData,1)) * dhat(size(AData,1)+(1:size(BData,1)))');

C = [A, B];
rand('seed', 1234);
Fr = abs(rand(size(mdata,1), nc));

for j = 1:iterNum
    upPre = C'*(A\(C*Fr)) + 2*lm*Fr;
    downPre = Fr + 2*lm*Fr*(Fr'*Fr);
    upPre(upPre<0) = eps;
    downPre(downPre<0) = eps;
    Fr = Fr .* ((upPre ./ downPre).^2);
end

clear F;
F(member,:) = Fr(1:size(AData),:);
F(~member,:) = Fr(size(AData)+1:end,:);
IDX = F;
end