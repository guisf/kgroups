"""Clustering using the waveform dataset:

https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import pandas as pd
import sklearn

from prettytable import PrettyTable

import wrapper
import eclust
import metric

import sys

# some semmimetric functions
rho = lambda x, y: np.power(np.linalg.norm(x-y), 1.5)
#rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2))
#rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*(4)**2))

# get dermatology data, last column are labels
df = pd.read_csv('data/yeast.data', sep=',', header=None)
classes = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4, 'ME1': 5,
           'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}

data = df.values[:,1:-1]
z = df.values[:,-1]  # 3 classes, starting with 0
z = np.array([classes[name] for name in z])
data = np.array(data, dtype=float)

#idx = np.random.choice(range(len(data)), 2000)
#data = data[idx]
#z = z[idx]

data = (data - data.mean(axis=0))/data.std(axis=0)

#sigma2 = sum([np.linalg.norm(x-y)**2 
#                for x in data for y in data])/(len(data)**2)
#sigma = np.sqrt(sigma2)

G = eclust.kernel_matrix(data, rho)
#G = energy.eclust.kernel_matrix(data, rho_gauss)
#G = energy.eclust.kernel_matrix(data, rho_exp)

k = 10
nt = 10
ini = 'random'

km = wrapper.kernel_kmeans(k, data, G, run_times=nt, ini=ini)[0]
kg = wrapper.kernel_kgroups(k,data,G,run_times=nt, ini=ini)[0]

sc = wrapper.spectral_clustering(k, data, G, run_times=nt)[0]
Z0 = eclust.ztoZ(sc)
kg2 = wrapper.kernel_kgroups(k,data,G,run_times=nt, Z0=Z0)[0]

t = PrettyTable(['metric', 'k-groups/k-means', 'k-groups/spectral'])
met_names = ['accuracy', 'aRAND', 'NMI']
for name, met in zip(met_names, [metric.accuracy,
            sklearn.metrics.adjusted_rand_score,
            sklearn.metrics.normalized_mutual_info_score]):
    r1 = met(z, kg)/met(z, km)
    r2 = met(z, kg2)/met(z, sc)
    t.add_row([name, r1, r2])

print(t)

print(metric.accuracy(z, kg), sklearn.metrics.adjusted_rand_score(z, kg), sklearn.metrics.normalized_mutual_info_score(z, kg))

sys.exit()




k = 10
nt = 5
r = []
r.append(wrapper.kmeans(k, data, run_times=nt))
r.append(wrapper.gmm(k, data, run_times=nt))
#r.append(wrapper.spectral_clustering(k, data, G, run_times=nt)[0])
r.append(wrapper.kernel_kmeans(k, data, G, run_times=nt, ini='k-means++')[0])
r.append(wrapper.kernel_kgroups(k,data,G,run_times=nt, ini='k-means++')[0])

t = PrettyTable(['Algorithm', 'Accuracy', 'A-Rand', 'NMI'])
algos = ['kmeans', 'GMM', #'spectral clustering',  
         'kernel k-means', 'kernel k-groups']

for algo, zh in zip(algos, r):
    t.add_row([algo, 
        metric.accuracy(z, zh),
        sklearn.metrics.adjusted_rand_score(z, zh),
        sklearn.metrics.normalized_mutual_info_score(z, zh)
    ])

print(t)

