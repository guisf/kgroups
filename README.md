# Kernel k-Groups

Clustering method based on energy statistics (non-parametric), developed on kernel spaces and based on Hartigan's method.
The method is often more robust and with better performance than other clustering methods, such as (kernel) k-means, spectral clustering, and gaussian mixture models (GMM).

The implementation of the algorithm can be found in the file "eclust.py" (the implementation follows the paper and is not optimized for speed).

An example of clustering a stochastic block model, in comparison with a well-known method in the literature (Bethe Hessian) is illustrated below:

![](https://github.com/guisf/kgroups/blob/main/figs/sbm_phase_15.png)

Comparison of kernel k-groups (red line) with k-means (blue), gaussian mixture model (green), spectral clustering (cyan), and kernel k-means (magenta) when clustering  a mixture of gaussians:

![](https://github.com/guisf/kgroups/blob/main/figs/gauss_n.png)


* See [G. França et. al., "Kernel k-Groups via Hartigan's Method," IEEE Transactions on Pattern Analysis and Machine Intelligence 43 12 (2020)](https://doi.org/10.1109/TPAMI.2020.2998120) for details.

