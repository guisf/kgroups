"""Clustering several datasets."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import pandas as pd
import sklearn

from prettytable import PrettyTable

import wrapper
import eclust
import metric
import get_data

import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cluster_results3(G, k, nt=15, ne=15, dataname='data', X=None):
    
    ini = 'k-means++'
    #ini = 'k-means'
    #ini = 'spectral'
    #ini = 'random'

    kk = []
    km = []
    sc = []
    for _ in range(ne):
    
        zh = wrapper.kernel_kgroups(k, data, G, run_times=nt, ini=ini)[0]
        kk.append([metric.accuracy(z, zh), 
                   sklearn.metrics.normalized_mutual_info_score(z, zh)])
        
        zh = wrapper.kernel_kmeans(k, data, G, run_times=nt, ini=ini)[0]
        km.append([metric.accuracy(z, zh), 
                   sklearn.metrics.normalized_mutual_info_score(z, zh)])
        
        zh = wrapper.spectral_clustering(k, data, G, run_times=nt)[0]
        sc.append([metric.accuracy(z, zh), 
                   sklearn.metrics.normalized_mutual_info_score(z, zh)])
    kk = np.array(kk)
    km = np.array(km)
    sc = np.array(sc)

    #out = [dataname, '%i'%G.shape[0], '%i'%k, '%i'%len(X[0]),
    #       '%.3f'%kk[:,0].mean(), '%.3E'%kk[:,0].std(),
    #       '%.3f'%kk[:,1].mean(), '%.3E'%kk[:,1].std(),
    #       '%.3f'%km[:,0].mean(), '%.3E'%km[:,0].std(),
    #       '%.3f'%km[:,1].mean(), '%.3E'%km[:,1].std(),
    #       '%.3f'%sc[:,0].mean(), '%.3E'%sc[:,0].std(),
    #       '%.3f'%sc[:,1].mean(), '%.3E'%sc[:,1].std()
    #]
    out = [dataname, '%i'%G.shape[0], '%i'%k, '%i'%len(X[0]),
           '$ %.3f \pm %.3E $'%(sc[:,1].mean(), sc[:,1].std()),
           '$ %.3f \pm %.3E $'%(km[:,1].mean(), km[:,1].std()),
           '$ %.3f \pm %.3E $'%(kk[:,1].mean(), kk[:,1].std()),
    ]
    print(' & '.join(out))


################################################################################

rho1 = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*2))
rho2 = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*1))
rho3 = lambda x, y: np.linalg.norm(x-y)
rho4 = lambda x, y: np.power(np.linalg.norm(x-y), 0.3)
rho5 = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
rho6 = lambda x, y: np.power(np.linalg.norm(x-y), 1.5)
rho7 = lambda x, y: np.power(np.linalg.norm(x-y), 1.8)
rho8 = lambda x, y: 2-2*np.exp(-(np.linalg.norm(x-y)**2)/(2*(2**2)))
rho9 = lambda x, y: 2-2*np.exp(-(np.linalg.norm(x-y)**2)/(2*(1**2)))

#rhos = [rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9]
rhos = [rho1]

gdata = [
    get_data.wine,
    get_data.synthetic_control, 
    get_data.iris, 
    get_data.contraceptive,
    get_data.fertility, 
    get_data.libras_movement, 
    get_data.covertype, 
    get_data.forest, 
    get_data.glass, 
    get_data.ionosphere, 
    get_data.spect_heart,

    get_data.epileptic, 
    get_data.yeast, 
    get_data.anuran,
    get_data.gene_expression,

    get_data.pendigits,
    get_data.statlog,
    get_data.vehicle,
    get_data.segmentation,
]
ks = [
    3,
    6,
    3,
    3,
    2,
    15,
    7,
    4,
    6,
    2,
    2,

    5,
    10,
    10,
    5,

    10,
    6,
    4,
    7,
]

for k, func_get_data in zip(ks, gdata):
    
    name = func_get_data.__name__
    data, z = func_get_data()

    for rho in rhos:
        G = eclust.kernel_matrix(data, rho)
        cluster_results3(G, k, dataname=name, X=data)

