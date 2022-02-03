function [varargout]=hysime(varargin)
%
% HySime: Hyperspectral signal subspace estimation
%
% [kf,Ek]=hysime(y,w,Rw,verbose);
%
% Input:
%        y  hyperspectral data set (each column is a pixel)
%           with (L x N), where L is the number of bands
%           and N the number of pixels
%        w  (L x N) matrix with the noise in each pixel
%        Rw noise correlation matrix (L x L)
%        verbose [optional] (on)/off
% Output
%        kf signal subspace dimension
%        Ek matrix which columns are the eigenvectors that span 
%           the signal subspace
%
%  Copyright: Jos� Nascimento (zen@isel.pt)
%             & 
%             Jos� Bioucas-Dias (bioucas@lx.it.pt)
%
%  For any comments contact the authors

error(nargchk(1, 3, nargin))
if nargout > 2, error('too many output parameters'); end
y = varargin{1};
if ~isnumeric(y), error('the data set must an L x N matrix'); end
noise_type = 'additive'; % default value
verbose = 1; verb ='on'; % default value
for i=2:nargin 
   switch lower(varargin{i}) 
       case {'additive'}, noise_type = 'additive';
       case {'poisson'}, noise_type = 'poisson';
       case {'on'}, verbose = 1; verb = 'on';
       case {'off'}, verbose = 0; verb = 'off';
       otherwise, error('parameter [%d] is unknown',i);
   end
end
verbose =0;


[L,N] = size(y);
if L<2, error('Too few bands to estimate the noise.'); end

if verbose, fprintf(1,'Noise estimates:\n'); end

if strcmp(noise_type,'poisson')
    sqy = sqrt(y.*(y>0));          % prevent negative values
    [u,Ru] = estAdditiveNoise(sqy,verb); % noise estimates
    x = (sqy - u).^2;            % signal estimates 
    w = sqrt(x).*u*2;
    Rw = w*w'/N; 
else % additive noise
    [w,Rw] = estAdditiveNoise(y,verb); % noise estimates        
end



[Ln,Nn] = size(w);
[d1,d2] = size(Rw);

if Ln~=L || Nn~=N  % n is an empty matrix or with different size
   error('empty noise matrix or its size does not agree with size of y\n'),
end
if (d1~=d2 || d1~=L)
   fprintf('Bad noise correlation matrix\n'),
   Rw = w*w'/N; 
end    


x = y - w;

if verbose,fprintf(1,'Computing the correlation matrices\n');end
[L,N]=size(y);
Ry = y*y'/N;   % sample correlation matrix 
Rx = x*x'/N;   % signal correlation matrix estimates 
if verbose,fprintf(1,'Computing the eigen vectors of the signal correlation matrix\n');end
[E,D]=svd(Rx); % eigen values of Rx in decreasing order, equation (15)
dx = diag(D);

if verbose,fprintf(1,'Estimating the number of endmembers\n');end
Rw=Rw+sum(diag(Rx))/L/10^5*eye(L);

Py = diag(E'*Ry*E); %equation (23)
Rw = diag(E'*Rw*E); %equation (24)
cost_F = -Py + 2 * Rw; %equation (22)
kf = sum(cost_F<0);
[dummy,ind_asc] = sort( cost_F ,'ascend');
Ek = E(:,ind_asc(1:kf));
if verbose,fprintf(1,'The signal subspace dimension is: k = %d\n',kf);end

varargout(1) = {kf};
if nargout == 2, varargout(2) = {Ek};end
return


function [w,Rw]=estAdditiveNoise(r,verbose) 

verbose = 0;
small = 1e-6;
verbose = 0;
[L, N] = size(r);
% the noise estimation algorithm
w=zeros(L,N);
if verbose 
   fprintf(1,'computing the sample correlation matrix and its inverse\n');
end
RR=r*r';     % equation (11)
if rcond(RR+small*eye(L))<1e-8
    % near singular, so we double perturbation size.
    small = 5e-5;
end
RRi=inv(RR+small*eye(L)); % equation (11)
if verbose, fprintf(1,'computing band    ');end
for i=1:L
    if verbose, fprintf(1,'\b\b\b%3d',i);end
    % equation (14)
    XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
    RRa = RR(:,i); RRa(i)=0; % this remove the effects of XX(:,i)
    % equation (9)
    beta = XX * RRa; beta(i)=0; % this remove the effects of XX(i,:)
    % equation (10)
    w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
end
if verbose, fprintf(1,'\ncomputing noise correlation matrix\n');end
Rw=diag(diag(w*w'/N));
return