
import numpy as np
from sklearn import metrics

import eclust
import wrapper
import data
import metric


"""
total_points = 2000
k = 60
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
data.plot2(X, z, '60clusters.pdf')
"""

total_points = 2000
k = 20
scale = 4
X = []
z = []
ns = np.random.multinomial(total_points, [1./k]*k)
for i, n in zip(range(k), ns):
    mean = np.random.uniform(0,scale*k,2)
    sigma = np.diag(np.random.uniform(0,12,2))
    points = np.random.multivariate_normal(mean, sigma, size=n)
    X.append(points)
    z.append([i]*n)
X = np.concatenate(X)
z = np.concatenate(z)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
z = z[idx]
data.plot2(X, z, '20clusters.pdf')

rt = 10

zh = wrapper.kmeans(k, X, run_times=10)
print('kmeans', metric.accuracy(z, zh), 
    metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh = wrapper.gmm(k, X, run_times=10)
print('gmm', metric.accuracy(z, zh), 
    metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

G = eclust.kernel_matrix(X, lambda x, y: np.power(np.linalg.norm(x-y), 1))

zh, _ = wrapper.kernel_kmeans(k, X, G, run_times=10, ini='k-means')
print('kernel kmeans', metric.accuracy(z, zh), 
    metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh, _ = wrapper.kernel_kgroups(k, X, G, run_times=10, ini='k-means')
print('kernel groups', metric.accuracy(z, zh), 
    metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

zh, _ = wrapper.spectral_clustering(k, X, G, run_times=10)
print('spectral clustering', metric.accuracy(z, zh), 
    metrics.normalized_mutual_info_score(z, zh,average_method='arithmetic'))

# k = 60
#kmeans 0.875 0.9605716257150889
#gmm 0.892 0.964607640453817
#kernel kmeans 0.8775 0.9648565088781947
#kernel groups 0.9115 0.9674940344630694
#spectral clustering 0.825 0.9521443817784696


