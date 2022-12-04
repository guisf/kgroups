"""Clustering several datasets."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import pandas as pd
import sklearn
import pickle

import wrapper
import eclust
import metric
import get_data

import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cluster_results(k, X, G, num_experiments=15, num_runs=15, 
                    data_name='data', ini='k-means++'):
    """"Cluster the data set and return a pandas dataframe.
    
    Input
    -----
    k : number of clusters
    X : data set
    G : distance matrix
    num_experiments : number of experiments to run
    num_run : number of times each method is run, return best objective
    data_name : to append the name of data set
    ini : initialization, {'k-means++', 'k-means', 'random'}
    
    """
    ne = num_experiments
    nt = num_runs
    m1 = metric.accuracy
    m2 = sklearn.metrics.normalized_mutual_info_score
    
    df = []
    for _ in range(ne):
    
        zh = wrapper.kernel_kgroups(k, X, G, run_times=nt, ini=ini)[0]
        df.append([m1(z, zh), m2(z, zh), 'k-groups', data_name])
        zh = wrapper.kernel_kmeans(k, X, G, run_times=nt, ini=ini)[0]
        df.append([m1(z, zh), m2(z, zh), 'k-means', data_name])
        zh = wrapper.spectral_clustering(k, X, G, run_times=nt)[0]
        df.append([m1(z, zh), m2(z, zh), 'spectral', data_name])
    
    df = pd.DataFrame(data=df, columns=['accuracy','nmi','algorithm','data'])
    return df 

rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*2))

gdata = [
    get_data.wine,
    get_data.synthetic_control, 
    get_data.fertility, 
    get_data.glass, 
    get_data.ionosphere, 
    get_data.epileptic, 
    get_data.anuran,
    get_data.gene_expression,
    get_data.pendigits,
    get_data.forest, 
    get_data.seeds,
    get_data.spect_heart,
    get_data.iris, 
    get_data.contraceptive,
    get_data.covertype, 
    
    get_data.libras_movement, 
    get_data.segmentation,
    get_data.vehicle,
    
#    get_data.yeast, 
#    get_data.statlog,
]

results = []
for func_get_data in gdata:
    data_name = func_get_data.__name__
    data, z = func_get_data()
    k = len(np.unique(z))
    
    print('%s (%i, %i, %i) ...'%(data_name,data.shape[0],k,data.shape[1]))

    G = eclust.kernel_matrix(data, rho)
    #G = eclust.normalized_rbf(data, k=10)
    df = cluster_results(k, data, G, num_experiments=100, num_runs=1, 
                         data_name=data_name, ini='k-means++')
    results.append(df)

results = pd.concat(results)

f = open('data/many_exp_kmeansplus.pickle', 'wb')
#f = open('data/many_exp_random.pickle', 'wb')
#f = open('data/many_exp_kmeans.pickle', 'wb')
pickle.dump(results, f)

