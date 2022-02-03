function SNR = calcSNR(M, q )
%HYPERVCA Vertex Component Analysis algorithm
%   hyperVca performs the vertex component analysis algorithm to find pure
% pixels in an HSI scene
%
% Usage
%   [ U, indicies, snrEstimate ] = hyperVca( M, q )
% Inputs
%   M - HSI data as 2D matrix (p x N).
%   q - Number of endmembers to find.
% Outputs
%   U - Matrix of endmembers (p x q).
%   indicies - Indicies of pure pixels in U
%   snrEstimate - SNR estimate of data [dB]
%
% References
%   J. M. P. Nascimento and J. M. B. Dias, Vertex component analysis: A 
% fast algorithm to unmix hyperspectral data, IEEE Transactions on 
% Geoscience and Remote Sensing, vol. 43, no. 4, apr 2005.

%Initialization.
N = size(M, 2);
L = size(M, 1);

% Compute SNR estimate.  Units are dB.
% Equation 13.
% Prefer using mean(X, dim).  I believe this should be faster than doing
% mean(X.') since matlab doesnt have to worry about the matrix
% transposition.
rMean = mean(M, 2);
RZeroMean = M - repmat(rMean, 1, N);
% This is essentially doing PCA here since we have zero mean data.
%  RZeroMean*RZeroMean.'/N -> covariance matrix estimate.
[Ud, Sd, Vd] = svds(RZeroMean*RZeroMean.'/N, q);
Rd = Ud.'*(RZeroMean);
P_R = sum(M(:).^2)/N;
P_Rp = sum(Rd(:).^2)/N + rMean.'*rMean;
SNR = abs(10*log10( (P_Rp - (q/L)*P_R) / (P_R - P_Rp) ));
 