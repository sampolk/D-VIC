%% Generate synthetic HSI
%{
The motivation behind D-VIS is that not all high-density pixels necessarily 
correspond to underlying material structure because hyperspectral images 
are generated at a coarse spatial resolution. A density maximizer could 
correspond correspond to a spatial region containing a group of commonly 
co-occuring materials (rather than a single material). So, D-VIS should do 
best when the highest-density points in latent clusters are indicative of a 
group of materials rather than a single material. It is in this setting 
when we would be most likely to outperform LUND.

To further motivate this, let’s consider a highly mixed hyperspectral 
dataset.  For any one cluster, there are probably multiple reasonable 
choices for modes that are exemplary of underlying cluster structure. In 
LUND, we select the highest-density pixels that are far in diffusion 
distance from other high-density pixels. However, if the HSI is 
sufficiently mixed, this will bias us towards selecting cluster exemplars 
that are not actually representative of underlying material structure. By 
weighting pixel purity and density equally, D-VIS will downweight 
high-density pixels that are not indicative of underlying material 
structure. So, it will choose better cluster modes. 

%}
%% Set Dimensions
M = 100;
N = 100;
n = M*N;
D = 100;
m = 5;
K = 5;

pHQPixels = 0.01;
pHDPixels = 0.09;

%% Create Ground Truth Image

GT = zeros(M,N);

GT(2:24, 2:24) = 1;
GT(27:49,2:24)= 2;
GT(52:74,2:24)= 3;
GT(77:99, 2:24) = 4;

GT(2:24, 27:49) = 2;
GT(27:49,27:49)= 3;
GT(52:74,27:49)= 4;
GT(77:99, 27:49) = 5;

GT(2:24, 52:74) = 3;
GT(27:49,52:74)= 4;
GT(52:74,52:74)= 5;
GT(77:99,52:74) = 1;

GT(2:24, 77:99) = 4;
GT(27:49,77:99)= 5;
GT(52:74,77:99)= 1;
GT(77:99, 77:99) = 2;
 
%% Create Ground Truth Endmembers
% random orthonormal vectors

A = rand(D);
A = A'*A;
[U,~] = eigs(A,m);
% U = U'.*([1, 2, 1, 2, 1])';

% U = rand(m,D);
% U = U./vecnorm(U,2,2);

%% Create Ground Truth Abundances

Y = reshape(GT,n,1);
A = zeros(n,m);

for k = 1:K
 
    nk = sum(Y == k);
    Xk = find(Y==k);

    % Generate high-quality pixel indices
    nHQPixels = floor(nk*pHQPixels);
    idx1 = randsample(nk, nHQPixels);
    HQPixels = Xk(idx1);

    % calculate abundances for high-quality pixels
    for j = 1:nHQPixels
        i = HQPixels(j);
        temp = zeros(m,1);
        temp(k) = 1;
        A(i, :) = temp./vecnorm(temp,1,1);
    end

    % Generate high-density, mixed pixel indices
    nHDPixels = floor(nk*pHDPixels);
    idx2 = randsample(nk, nHDPixels);
    while ~isempty(intersect(idx1,idx2))
        idx2 = randsample(nk, nHDPixels);
    end
    HDPixels = Xk(idx2);
    
    % Calculate abundances for high-density pixels
    for j = 1:nHDPixels
        i = HDPixels(j);
        temp = zeros(m,1);
        temp(k) = 0.51;
        temp(mod(k+1,m)+1) = 0.49;
        A(i, :) = temp./vecnorm(temp,1,1);
    end

    % Calculate non-modal points
    remainingPts = setdiff(1:nk, [idx1; idx2]);
    exemplars = [idx1; idx2];
    nExemplars = length(exemplars);
    nremainingPts = length(remainingPts);

    for j = 1:nremainingPts
        i = Xk(remainingPts(j));

        match = randsample(nExemplars, 1);
        A(i, :) = A(Xk(exemplars(match)),:) + 0.1*abs(randn(1,5));
        A(i, :) = A(i, :)./norm(A(i, :),1);

    end
end

% assign background points abundances
remainingPts = find(sum(A,2) == 0);
nremainingPts = length(remainingPts);
for j = 1:nremainingPts
    i =  remainingPts(j);
    A(i, :) =  rand(1,5);
    A(i, :) = A(i, :)./norm(A(i, :),1);

end

%% Construct dataset via linear mixing model

X = A*U' + 0.2*abs(randn(n,D));
HSI = reshape(X,M,N,D);

clear A A_GT exemplars HDPixels HQPixels i idx1 idx2 j k match nExemplars nHDPixels nHQPixels nremainingPts pHDPixels pHQPixels temp U Xk remainingPts U_GT nk
% save('syntheticHSI5149Stretched', 'U_GT', "A_GT", "X", "Y", "GT", "M", 'N', 'n', 'D')






