function p = KDE(Dist_NN, Hyperparameters)
%{
Inputs:     Dist_NN:            nxM matrix where Dist_NN(i,:) encodes the M 
                                nearest neighbors of X(i,:), sorted in 
                                ascending order.
            Hyperparameters:    Structure with 

Outputs:    p:                  Kernel density estimator. 

This function was used in the following papers:

    - Murphy, J. M., & Polk, S. L. (2022). A multiscale environment for 
      learning by diffusion. Applied and Computational Harmonic 
      Analysis, 57, 58-100.
    - Polk, S. L., & Murphy, J. M. (2021, July). Multiscale Clustering 
      of Hyperspectral Images Through Spectral-Spatial Diffusion 
      Geometry. In 2021 IEEE International Geoscience and Remote 
      Sensing Symposium IGARSS (pp. 4688-4691). IEEE.
    - Polk, S. L., Cui, K., Plemmons, R. J., and Murphy, J. M., (2022). 
      Diffusion and Volume Maximization-Based Clustering of Highly 
      Mixed Hyperspectral Images. (In Review).

© 2022 Sam L Polk, Tufts University. 
email: samuel.polk@tufts.edu

%}

% Extract hyperparameters
NN = Hyperparameters.DensityNN;
sigma0 = Hyperparameters.Sigma0;

% Calculate density
D_trunc = Dist_NN(:,1:NN);
p = sum(exp(-(D_trunc.^2)./(sigma0^2)),2);
p = p./sum(p);


% avoids p=0 by setting 0-value entries as the minimum value 
p(p == 0) = min(p(p>0)); 