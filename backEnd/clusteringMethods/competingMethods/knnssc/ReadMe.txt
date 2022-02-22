This code implements nonlinear subspace clustering approaches that are discussed in the survey paper:
"Beyond Linear Subspace Clustering: A Comparative Study of Nonlinear Manifold Clustering Algorithms" by M. Abdolali and N. Gillis.

The code contains the code of the following approaches taken from the authors websites:
	-SMCE (Sparse Manifold Clustering and Embedding),
	-KNN-SSC (k-nearest neighbors based sparse subspace clustering) which uses vlfeat library,
	-SMR (Smooth Representation).

We have also provided our implementation of:
	-LR$\ell_1$-SSC (Laplacian Regularized $\ell_1$-SSC),
	-KSSC (Kernel based Sparse Subspace Clustering),
	-LKG (Low-rank Kernel Learning for Graph matrix)
	-KNN-SSC (k-nearest neighbors based sparse subspace clustering) which is based on ADMM and does not rely on installing vlfeat library.


The main numerical experiments from the paper include comparing nonlinear SC approaches on
 -real datasets: COIL-20, MNIST and Extended Yale B.
 -nonlinear synthetic datasets: Halfkernels, Two spirals, Four Corners and Two Arcs.
 -linear synthetic datasets from both independent and disjoint subspaces.
