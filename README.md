# Kernel k-Groups

This is a clustering method based on energy statistics, developed on kernel spaces and based on Hartigan's method.
The method is often more robust and with better performance than other clustering methods, such as (kernel) k-means, spectral clustering, and Gaussian mixture models.

The implementation of the algorithm can be found in the file "eclust.py" (the implementation follows the paper and is not optimized for speed).

An example of clustering with a stochastic block model, in comparison with a well-known method in the literature is illustrated below.

![]()


* See [G. França et. al., "Kernel k-Groups via Hartigan's Method," IEEE Transactions on Pattern Analysis and Machine Intelligence 43 12 2020](https://doi.org/10.1109/TPAMI.2020.2998120) for details.
