function [M, A, R, obj] = group_robust_nmf(Y, beta, M, A, R, lambda, tol, n_iter_max, verbose)

% Block-coordinate robust NMF algorithm for local solution of
%
% min D(Y|MA+R) + lambda ||R||_2,1 subject to ||a_n||_1 = 1
%
% where D is the beta-divergence. Special cases:
% - beta = 2 corresponds to squared-Euclidean distance
% - beta = 1 corresponds to generalised Kullback-Leibler divergence
% - beta = 0 corresponds to Itakura-Saito divergence
%
% For details, refer to
%
% C. Fevotte and N. Dobigeon. Nonlinear spectral unmixing with robust
% nonnegative matrix factorization. IEEE Trans. Image Processing, 2015.
%
% Notes:
% - if needed, updates for specific cases beta = 1 and 2 could be
%   simplified and made slightly more efficient,
% - the sum-to-one assumption on A can be easily lifted, see commented
%   update at the end.
%
% Submit bugs & comments to Cedric Fevotte (CNRS)

%%% Copyright Cedric Fevotte & Nicolas Dobigeon, 2015.
%%%
%%% This program is free software: you can redistribute it and/or modify
%%% it under the terms of the GNU General Public License as published by
%%% the Free Software Foundation, either version 3 of the License, or
%%% (at your option) any later version.
%%% 
%%% This program is distributed in the hope that it will be useful,
%%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%%% GNU General Public License for more details.
%%% 
%%% You should have received a copy of the GNU General Public License
%%% along with this program.  If not, see <http://www.gnu.org/licenses/>.

if nargin ==8
    verbose =0;
end

[F,K] = size(M);
A = A./repmat(sum(A,1),K,1); % normalise A
S = M*A; % low-rank part
Y_ap = S + R + eps; % data approximate

fit = zeros(1,n_iter_max);
obj = zeros(1,n_iter_max);

err_old = Inf;

% monitor convergence
iter = 1;
fit(iter) = betadiv(Y,Y_ap,beta); % compute fit
obj(iter) = fit(iter) + lambda*sum(sqrt(sum(R.^2,1))); % compute objective
err = Inf;
if verbose
    fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
end
while (err >= tol ) && (iter < n_iter_max)
    
    % update R, the outlier term
    R = R.*((Y.*Y_ap.^(beta-2))./(Y_ap.^(beta-1) + lambda*R./repmat(sqrt(sum(R.^2,1))+eps,F,1)));
    Y_ap = S + R + eps;
    
    % update A, the abundance/activation matrix
    Y_ap1 = Y_ap.^(beta-1);
    Y_ap2 = Y_ap.^(beta-2);
    Gn = M'*(Y.*Y_ap2) + repmat(sum(S.*Y_ap1,1),K,1);
    Gp = M'*Y_ap1 + repmat(sum(S.*Y.*Y_ap2,1),K,1);
    A = A.*(Gn./Gp);
    A = A./repmat(sum(A,1),K,1);
    S = M*A;
    Y_ap = S + R + eps;
    
    % update M, the endmembers/dictionary matrix
    M = M .* (((Y.*Y_ap.^(beta-2))*A')./((Y_ap.^(beta-1))*A'));
    S = M*A;
    Y_ap = S + R + eps;
    
    % monitor convergence
    iter = iter + 1;
    fit(iter) = betadiv(Y,Y_ap,beta);
    obj(iter) = fit(iter) + lambda*sum(sqrt(sum(R.^2,1)));
    err = abs((obj(iter-1)-obj(iter))/obj(iter));
    
    if err>err_old
        disp([])
    end
    
    if rem(iter,50)==0
        
        err_old = err;
        if verbose
            fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
        end
    end
    
end

if verbose
% final values
    fprintf('iter = %4i | obj = %+5.2E | err = %4.2E (target is %4.2E) \n',iter,obj(iter),err,tol)
end
% clean
obj = obj(1:iter); fit = fit(1:iter);

%% goodie: update of A without the sum-to-one constraint
% A = A .* (M'*((Y.*Y_ap.^(beta-2)))./(M'*(Y_ap.^(beta-1))));
% S = M*A;
% Y_ap = S + R + eps;

end

%% beta-divergence
function d = betadiv(A,B,beta)
switch beta
    case 2 % squared-Euclidean distance
        d = sum((A(:)-B(:)).^2)/2;
    case 1 % generalised Kullback-Leibler divergence
        ind_0 = find(A(:)<=eps);
        ind_1 = 1:length(A(:));
        ind_1(ind_0) = [];
        d = sum( A(ind_1).*log(A(ind_1)./B(ind_1)) - A(ind_1) + B(ind_1) ) + sum(B(ind_0));
    case 0 % Itakura-Saito divergence
        d = sum( A(:)./B(:) - log(A(:)./B(:)) ) - length(A(:));
    otherwise
        d = sum( A(:).^beta + (beta-1)*B(:).^beta - beta*A(:).*B(:).^(beta-1) )/(beta*(beta-1));
end
end

