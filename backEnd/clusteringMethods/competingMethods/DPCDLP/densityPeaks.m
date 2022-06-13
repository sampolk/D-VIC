function [peaks,nPeak,k,neighbs,rho]=densityPeaks(X0,dis,k,tetha)
    [disFromNeighb,neighbs]=neighborhood(X0,k);
    rho=calculateRho(disFromNeighb(:,1:k),k);
    delta=calculateDelta(rho,dis);
    peaks=selectPeak(rho,delta,tetha);
    nPeak=size(peaks,1);
end