# D-VIC

This toolbox allows the implementation of the following diffusion-based unsupervised hyperspectral image clustering algorithms on synthetic and real hyperspectral images:

- Learning by Unsupervised Nonlinear Diffusion (LUND)
- Multiscale Learning by Unsupervised Nonlinear Diffusion (M-LUND)
- Diffusion and Volume maximization-based Image Clustering (D-VIC)
- Active Diffusion and VCA-Assisted Image Segmentation (ADVIS)  

This package can be used to generate experiments in the following articles:

1. Polk, S. L., Cui, K., Chan, A. H., Coomes, D. A., Plemmons, R. J., & Murphy, J. M. (2023). Unsupervised Diffusion and Volume Maximization-Based Clustering of Hyperspectral Images. _Remote Sensing_, 15(4), 1053.
2. Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. (2022). "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." In the _Proceedings of IEEE IGARSS 2022_.

The following scripts (in the Experiments folder) generate the relevant experiments:

- DVIC_demo.m evaluates the D-VIS clustering algorithm on four benchmark hyperspectral images. This script can be used to replicate experiments that appear in Section 4.1 of article 1.
- ADVIS_demo.m compares the ADVIS active learning algorithm against the D-VIS clustering algorithm. This script is used for experiments that appear in article 2.
- syntheticExperiment.m replicates the synthetic data experiments presented in Section 3.2 of article 1.  
- runGridSearches.m replicates hyperparameter optimization performed in article 1 and replicates Figure 8 in article 1. 
- tAnalysis.m analyzes the robustness of D-VIC to diffusion time and replicates Figure 9 in article 1. 

Real hyperspectral image data (Salinas A, which was used in both articles, as well as Indian Pines and Jasper Ridge, which were used in article 1.) can be downloaded at the following links:

- http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
- https://rslab.ut.ac.ir/data
    
Users are free to modify the D-VIC toolbox as they wish. If you find it useful or use it in any publications, please cite the following papers:

- Polk, S. L., Cui, K., Chan, A. H., Coomes, D. A., Plemmons, R. J., & Murphy, J. M. (2023). Unsupervised Diffusion and Volume Maximization-Based Clustering of Hyperspectral Images. _Remote Sensing_, 15(4), 1053.
- Polk, S. L., Chan, A. H. A., Cui, K., Plemmons, R. J., Coomes, D. A., & Murphy, J. M. (2022). "Unsupervised detection of ash dieback disease (_Hymenoscyphus fraxineus_) using diffusion-based hyperspectral image clustering" In the _Proceedings of IEEE IGARSS 2022_.
- Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. (2022). "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." In the _Proceedings of IEEE IGARSS 2022_.
- Murphy, J. M., & Polk, S. L. (2022). "A multiscale environment for learning by diffusion." _Applied and Computational Harmonic Analysis_, 57, 58-100.
- Maggioni, M., & Murphy, J. M. (2019). Learning by Unsupervised Nonlinear Diffusion. _Journal of Machine Learning Research_, 20(160), 1-56.

Please write with any questions: sam.polk@outlook.com

(c) Copyright Sam L. Polk, Tufts University, 2022.
