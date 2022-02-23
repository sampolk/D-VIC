# DVISREpo

This toolbox allows the implementation of the following diffusion-based clustering algorithms on synthetic and real datasets included in the repository:

- Learning by Unsupervised Nonlinear Diffusion (LUND)
- Multiscale Learning by Unsupervised Nonlinear Diffusion (M-LUND)
- Spatially Regularized Diffusion Learning (SRDL)
- Multiscale Spatially Regularized Diffusion Learning (M-SRDL)
- Diffusion and VCA-Assisted Image Segmentation (D-VIS)
- Active Diffusion and VCA-Assisted Image Segmentation (ADVIS)

This package can be used to generate experiments in the following articles:

1. Polk, S. L., Cui, Kangning, Plemmons, R. J., & Murphy, J. M. "Clustering Highly Mixed Hyperspectral Images Using Diffusion and VCA-Assisted Image Segmentation." To Appear.
2. Polk, S. L., Cui, Kangning, Plemmons, R. J., & Murphy, J. M. "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." To Appear

The following scripts (in the Experiments folder) generate the relevant experiments:

- DVIS_demo.m evaluates the D-VIS clustering algorithm on four benchmark hyperspectral images. This script can be used to replicate experiments that appear in article 1.
- ADVIS_demo.m compares the ADVIS active learning algorithm against the D-VIS clustering algorithm . This script is used for experiments that appear in article 2.

Real hyperspectral image data (Salinas A, which was used in articles 1-4, as well as Indian Pines, Jasper Ridge, and Pavia Centre, which were used in article 1.) can be downloaded at the following links:

- http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
- https://rslab.ut.ac.ir/data
    
Users are free to modify the Multiscale Diffusion Clustering toolbox as they wish. If you find it useful or use it in any publications, please cite the following papers:

- Polk, S. L., Chan, A. H. A., Cui, Kangning, Plemmons, R. J., Coomes, D. A., & Murphy, J. M. "Unsupervised detection of ash dieback disease (\emph{Hymenoscyphus fraxineus}) using diffusion-based hyperspectral image clustering" To Appear
- Polk, S. L., Cui, Kangning, Plemmons, R. J., & Murphy, J. M. "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." To Appear
- Polk, S. L., Cui, Kangning, Plemmons, R. J., & Murphy, J. M. "Clustering Highly Mixed Hyperspectral Images Using Diffusion and VCA-Assisted Image Segmentation." To Appear.
- Murphy, J. M., & Polk, S. L. (2022). "A multiscale environment for learning by diffusion." Applied and Computational Harmonic Analysis, 57, 58-100.
- Polk, S. L. & Murphy, J. M. (2021) "Multiscale Clustering of Hyperspectral Images Through Spectral-Spatial Diffusion Geometry." Proceedings of the 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, 4688-4691.
- Murphy, J. M., & Maggioni, M. (2019). Spectral–spatial diffusion geometry for hyperspectral image clustering. IEEE Geoscience and Remote Sensing Letters, 17(7), 1243-1247.
- Maggioni, M., & Murphy, J. M. (2019). Learning by Unsupervised Nonlinear Diffusion. Journal of Machine Learning Research, 20(160), 1-56.

Please write with any questions: samuel.polk@tufts.edu

(c) Copyright Sam L. Polk, Tufts University, 2022.
