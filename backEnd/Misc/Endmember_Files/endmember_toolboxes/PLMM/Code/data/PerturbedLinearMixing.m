function Unmixing = PerturbedLinearMixing(X,ImageSize, K)

%--------------------------------------------------------------
% Unmixing parameters
%--------------------------------------------------------------

H = ImageSize(1);
W = ImageSize(2);
[N,L] = size(X);

Y = (reshape(permute(reshape(X,H,W,L),[2 1 3]),H*W,L))';

% Stopping criteria
epsilon = 1e-4; % BCD iteration
eps_abs = 1e-2; % primal residual
eps_rel = 1e-4; % dual residual
% ADMM parameters
nIterADMM = 100;       % maximum number of ADMM subiterations
nIterBCD = 100;
rhoA = 100; 
rhoM = 1e-1;
rhodM = 1e-1;
tau_incr = 1.1;
tau_decr = 1.1;
mu = 10.;
% Regularization 
typeM = 'mdist';    % regularization type ('NONE','MUTUAL DISTANCE','VOLUME','DISTANCE')
percent = 0.1;
gamma = 1e-1;       % variability regularization parameter
flag_proj = false;  % select dM update version (constrained or penalized version)
if N > 1e4          % enable parallelization (or not), depending on the number of pixels
    flag_parallel = true;
else
    flag_parallel = false;
end
flag_update = true; % enable augemented Lagrangian update

%--------------------------------------------------------------
% Initialization
%--------------------------------------------------------------
% Endmember initialization (VCA [1])
[M0, V, U, Y_bar, ~, ~] = find_endm(Y,K,'vca');
[L,R] = size(M0);
% Abundance initialization (SUNSAL [2])
A0 = sunsal(M0,Y,'POSITIVITY','yes','ADDONE','yes');
A0 = max(bsxfun(@minus,A0,max(bsxfun(@rdivide,cumsum(sort(A0,1,'descend'),1)-1,(1:size(A0,1))'),[],1)),0);

% Perturbation matrices initialization
dM0 = zeros(L,R,N);
switch lower(typeM)
    case 'none'
        aux = {typeM};
        aux1 = {typeM,0.};
    case 'dist'
       aux = {typeM,M0};
       aux1 = {typeM,0.,M0};
    case 'mdist'
        aux = {typeM};
        aux1 = {typeM,0.};
    case 'volume'
        aux = {typeM,Y_bar,V};
        aux1 = {typeM,0.,Y_bar,U,V};
    otherwise
        typeM = 'none';
        aux = {typeM};
        aux1 = {typeM,0.};
end
[alpha,beta,~] = penalty_term_plmm(Y,M0,A0,dM0,H,W,percent,'PENALTY',aux);
alpha = 10*alpha;
aux1{2} = beta;             

%--------------------------------------------------------------
% BCD/ADMM unmixing (based on the PLMM)
%--------------------------------------------------------------
[f,A,M,dM] = bcd_admm(Y,A0,M0,dM0,W,H,gamma,flag_proj,flag_parallel,flag_update,eps_abs,eps_rel,epsilon,'AL PARAMETERS',{rhoA,rhoM,rhodM},'PENALTY A',alpha,'PENALTY M',aux1,'AL INCREMENT',{tau_incr,tau_decr,mu},'MAX STEPS',{nIterADMM,nIterBCD});

%--------------------------------------------------------------
% Error computation
%--------------------------------------------------------------
[RE, ~, ~] = real_error(Y,A,M,dM,W,H);


Unmixing.Abundance = A';
Unmixing.Endmembers = M';
Unmixing.Perturbations = dM;
Unmixing.Convergence.ReconstructionMSE = RE;
Unmixing.Convergence.ObjectiveFunction = f;
Unmixing.SpatialParams = [H,W];

clc
disp('Perturbed Linear Mixing Complete')

