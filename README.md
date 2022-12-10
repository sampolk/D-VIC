# D-VIC

This toolbox allows the implementation of the following diffusion-based clustering algorithms on synthetic and real datasets included in the repository:

- Learning by Unsupervised Nonlinear Diffusion (LUND)
- Multiscale Learning by Unsupervised Nonlinear Diffusion (M-LUND)
- Diffusion and Volume maximization-based Image Clustering (D-VIC)
- Active Diffusion and VCA-Assisted Image Segmentation (ADVIS)  

This package can be used to generate experiments in the following articles:

1. Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. "Diffusion and Volume Maximization-Based Clustering of Highly Mixed Hyperspectral Images." arXiv. arXiv:2203.09992
2. Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." In the Proceedings of IEEE IGARSS 2022 (2022).

The following scripts (in the Experiments folder) generate the relevant experiments:

- DVIC_demo.m evaluates the D-VIS clustering algorithm on four benchmark hyperspectral images. This script can be used to replicate experiments that appear in article 1.
- ADVIS_demo.m compares the ADVIS active learning algorithm against the D-VIS clustering algorithm . This script is used for experiments that appear in article 2.
- syntheticExperiment.m replicates the synthetic data experiments presented in Section 4.B.3 of article 1.  
- runGridSearches.m replicates hyperparameter optimization performed in article 1 and replicates Figure 8. 
- tAnalysis.m analyzes the robustness of D-VIC to diffusion time and replicates Figure 9 in article 1. 

Real hyperspectral image data (Salinas A, which was used in both articles, as well as Indian Pines and Jasper Ridge, which were used in article 1.) can be downloaded at the following links:

- http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
- https://rslab.ut.ac.ir/data
    
Users are free to modify the D-VIC toolbox as they wish. If you find it useful or use it in any publications, please cite the following papers:

- Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. "Diffusion and Volume Maximization-Based Clustering of Highly Mixed Hyperspectral Images." arXiv. arXiv:2203.09992
- Polk, S. L., Chan, A. H. A., Cui, K., Plemmons, R. J., Coomes, D. A., & Murphy, J. M. (2022). "Unsupervised detection of ash dieback disease (_Hymenoscyphus fraxineus_) using diffusion-based hyperspectral image clustering" In the Proceedings of IEEE IGARSS 2022 (2022).
- Polk, S. L., Cui, K., Plemmons, R. J., & Murphy, J. M. (2022). "Active Diffusion and VCA-Assisted Image Segmentation of Hyperspectral Images." In the Proceedings of IEEE IGARSS 2022 (2022).
- Murphy, J. M., & Polk, S. L. (2022). "A multiscale environment for learning by diffusion." Applied and Computational Harmonic Analysis, 57, 58-100.
- Maggioni, M., & Murphy, J. M. (2019). Learning by Unsupervised Nonlinear Diffusion. Journal of Machine Learning Research, 20(160), 1-56.

Please write with any questions: samuel.polk@tufts.edu

(c) Copyright Sam L. Polk, Tufts University, 2022.
