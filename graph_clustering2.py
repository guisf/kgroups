"""Some graph clustering examples on real data."""

# Guilherme Franca <guifranca@gmail.com>
# Johsn Hopkins University

from __future__ import division

import numpy as np
from prettytable import PrettyTable
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi
import pandas as pd

import graph_clustering as gc
from metric import accuracy
import wrapper
import init
import eclust
import get_data


def bethe_hessian(A, z, r=None, s=1):
    """Cluster based on Bethe Hessian approach."""
    k = len(np.unique(z))
    degrees = A.sum(axis=1)
    d = degrees.mean()
    D = np.diag(degrees)
    if not r:
        r = np.sqrt(d)
        #r = np.sqrt((degrees**2).mean()/degrees.mean() - 1)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + s*D
    zh = init.spectral2(k, H, run_times=20, largest=False)
    return zh

def kgroups_bethe(A, z, r=None, useW=True, s=1):
    """Cluster based on Kernel k-groups approach but use 
    Bethe Hessian matrix, also as initialization.
    
    """
    k = len(np.unique(z))
    degrees = A.sum(axis=1)
    d = degrees.mean()
    D = np.diag(degrees)+0.0001*np.eye(A.shape[0])
    if not r:
        r = np.sqrt(d)
        #r = np.sqrt((degrees**2).mean()/degrees.mean() - 1)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + s*D

    z0 = init.spectral2(k, H, run_times=20, largest=False)
    Z0 = eclust.ztoZ(z0)

    if useW:
        W = D
    else:
        W = np.eye(A.shape[0])

    # run kgroups with weights and Bethe Hessian as initialization
    zh = wrapper.kernel_kgroups(k, -H, -H, W=W, run_times=20, Z0=Z0)[0]
    return zh

def spectralA(A, z, r=None, useW=True):
    """Cluster based on Kernel k-groups approach. Use 
    the adjacency matrix.
    
    """
    k = len(np.unique(z))
    degrees = A.sum(axis=1)
    d = degrees.mean()
    D = np.diag(degrees)+0.0001*np.eye(A.shape[0])
    if useW:
        W = D
    else:
        W = np.eye(A.shape[0])
    zh = wrapper.spectral_clustering(k, A, A, W=W, run_times=20)[0]
    return zh

def kmeans_bethe(A, z, r=None, useW=True, ini='bethe', s=1):
    """Cluster based on Kernel k-groups approach but use 
    Bethe Hessian matrix, also as initialization.
    
    """
    k = len(np.unique(z))
    degrees = A.sum(axis=1)
    d = degrees.mean()
    D = np.diag(degrees)+0.0001*np.eye(A.shape[0])
    if not r:
        r = np.sqrt(d)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + s*D

    if ini == 'bethe':
        # this is the Bethe Hessian solution
        z0 = init.spectral2(k, H, run_times=20, largest=False)
        Z0 = eclust.ztoZ(z0)
    else:
        Z0 = None

    if useW:
        W = D
    else:

        W = np.eye(A.shape[0])
    # run kgroups with weights and Bethe Hessian as initialization
    zh = wrapper.kernel_kmeans(k, -H, -H, W=W, run_times=20, Z0=Z0, ini=ini)[0]
    return zh

def overlap(z, zh, k):
    return (k/(k-1))*(accuracy(z, zh)-1/k)


###############################################################################

if __name__ == "__main__":
    import sys

    get_data_func = [
        #get_data.eucore,
        get_data.karate,
        get_data.football,
        get_data.dolphin,
        get_data.polbooks,
        #get_data.polblogs
        #A, z = get_data.adjnoun
    ]

    df = []
    for gdata in get_data_func:
        
        data_name = gdata.__name__
        A, z = gdata()
        k = len(np.unique(z))
        for _ in range(1):

            zh = bethe_hessian(A, z)
            df.append([data_name, 'bethe', overlap(z, zh, k), nmi(z, zh)])
        
            zh = kgroups_bethe(A, z, useW=False)
            df.append([data_name,'kgroups_bethe',overlap(z,zh,k),nmi(z,zh)])
        
            zh = spectralA(A, z, useW=False)
            df.append([data_name, 'spectralA', overlap(z, zh, k), nmi(z, zh)])

    df = pd.DataFrame(df, columns=['dataset', 'method', 'overlap', 'nmi'])
    
    t = PrettyTable()
    t.field_names = ['dataset', 'method', 'overlap', 'nmi']
    datasets = df['dataset'].unique()
    methods = df['method'].unique()
    for data in datasets:
        for meth in methods:
            over = df.loc[(df['dataset']==data) & \
                          (df['method']==meth)]['overlap'].values
            nmi = df.loc[(df['dataset']==data) & \
                         (df['method']==meth)]['nmi'].values
            t.add_row([data, meth, 
                   '%.3f \pm %.3f'%(over.mean(), over.std()),
                   '%.3f \pm %.3f'%(nmi.mean(), nmi.std())
            ])
    print(t)


