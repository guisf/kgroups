"""Experiment for more than 2 clusters."""

# author: Guilherme S. Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
from sklearn import metrics
from prettytable import PrettyTable

import data
import eclust
import wrapper
import metric


# artificial data with many clusters
total_points = 1500
k = 150
scale = 3
X = []
z = []
ns = np.random.multinomial(total_points, [1./k]*k)
for i, n in zip(range(k), ns):
    mean = np.random.uniform(0,scale*k,2)
    sigma = np.diag(np.random.uniform(0,10,2))
    points = np.random.multivariate_normal(mean, sigma, size=n)
    X.append(points)
    z.append([i]*n)
X = np.concatenate(X)
z = np.concatenate(z)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
z = z[idx]

data.plot2(X, z, '%iclusters.pdf'%k)

zh = wrapper.kmeans(k, X, run_times=10)
print('kmeans', metric.accuracy(z, zh), metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh = wrapper.gmm(k, X, run_times=10)
print('gmm', metric.accuracy(z, zh), metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

G = eclust.kernel_matrix(X, lambda x, y: np.power(np.linalg.norm(x-y), 1))
#sigma = 5
#G = eclust.kernel_matrix(G, lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*sigma)))
#G = eclust.kernel_matrix(G, lambda x, y: 2-2*np.exp(-np.power(np.linalg.norm(x-y),2)/(2*sigma**2)))

zh, _ = wrapper.kernel_kmeans(k, X, G, run_times=10, ini='k-means')
print('kernel kmeans', metric.accuracy(z, zh), metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh, _ = wrapper.kernel_kgroups(k, X, G, run_times=10, ini='k-means')
print('kernel groups', metric.accuracy(z, zh), metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh, _ = wrapper.spectral_clustering(k, X, G, run_times=10)
print('spectral clustering', metric.accuracy(z, zh), metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))
