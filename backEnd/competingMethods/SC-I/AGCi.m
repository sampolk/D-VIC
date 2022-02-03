function IDX = AGCi(mdata, nc, scale, iternum, sz, mflag )
k = 10;
lm = 1/2; 
beta = 3/5;

if(size(sz,1)==0) % for synthetic data sets
    member = zeros(size(mdata,1),1);
    member(1:scale:end)=1;
    member = logical(member);
    APoints = [];
    BPoints = [];
else % for HSI data sets
    [X, Y] = meshgrid((1:sz(1)), (1:sz(2))');
    Points = zeros(size(X,1)*size(X,2),1);
    [Points(:,1), Points(:,2)] = deal(reshape(X',[],1), reshape(Y',[],1));
    [X, Y] = meshgrid((1:scale:sz(1)), (1:scale:sz(2))');
    APoints = zeros(size(X,1)*size(X,2),1);
    [APoints(:,1), APoints(:,2)] = deal(reshape(X',[],1), reshape(Y',[],1));
    BPoints = Points;
    member = ismember(BPoints, APoints, 'rows');
end

% AData = mdata; % no choose
AData = mdata(member,:); % anchor is direct choose
% [pdx, AData, sumd, kd] = kmeans(mdata, 500); % anchor is kmeans 
BData = mdata;
Z = PKN(BData,AData,BPoints,APoints,k,sz,beta,mflag); 

Tdiag = diag(sum(Z,1));
rand('seed', 1234);
Fr = abs(rand(size(mdata,1), nc));
for k = 1:iternum
    upPre = Z*(Tdiag\(Z'*Fr)) + 2*lm*Fr;
    downPre = Fr + 2*lm*Fr*(Fr'*Fr);
    Fr = Fr .* ((upPre ./ downPre).^2);
end
IDX = Fr;
end