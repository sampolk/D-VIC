hierclust2nmf.m is the main function, performing the hierarchical clustering of hyperspectral images. (Note that is can also be applied to any other kind of data.) 

Run the file RunMe.m to have an example with the Urban dataset. 

See N. Gillis, D. Kuang and H. Park, "Hierarchical Clustering of Hyperspectral Images using Rank-Two Nonnegative Matrix Factorization", IEEE Trans. on Geoscience and Remote Sensing 53 (4), pp. 2066-2078, 2015

Modifications w.r.t. v.1: 
- Fix a bug when the input matrix had repeated columns. 
- New output: the index set of the columns of the input matrix correpsonding to the endmembers (=cluster centroids). 