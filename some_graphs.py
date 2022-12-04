"""Graph Clustering of some datasets."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
import pandas as pd
from community import modularity
from community import best_partition
from networkx.algorithms import community
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle
from sklearn.metrics import normalized_mutual_info_score as nmi
    
import sys

import get_data
import wrapper
import init
import eclust
from metric import accuracy
import plot_network

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###############################################################################
### clustering methods ###

def spectralA(k, A):
    """Cluster the adjacency matrix."""
    zh = init.spectral2(k, A, run_times=10, largest=True)
    return zh

def spectralNL(k, A):
    """Cluster the normalized Laplacian."""
    D = np.diag(A.sum(axis=1)) + 0.0001*np.eye(A.shape[0])
    L = D - A
    D2 = np.sqrt(np.linalg.inv(D))
    L = D2.dot(L.dot(D2))
    zh = init.spectral2(k, L, run_times=10, largest=False)
    return zh

def bethe_hessian(k, H):
    """Cluster based on Bethe Hessian approach."""
    zh = init.spectral2(k, H, run_times=10, largest=False)
    return zh

def kgroups_bethe_hessian(k, A, H, useW=False):
    """Cluster based on Kernel k-groups approach but use 
    Bethe Hessian matrix, also as initialization.
    
    """
    if useW:
        D = np.diag(A.sum(axis=1)) + 0.001*np.eye(A.shape[0])
    else:
        D = np.eye(A.shape[0])
    z0 = init.spectral2(k, H, run_times=10, largest=False)
    Z0 = eclust.ztoZ(z0)
    zh = wrapper.kernel_kgroups(k, -H, -H, W=D, run_times=10, Z0=Z0)[0]
    return zh


###############################################################################
# utility functions

def labels_to_dict(z):
    """"Pick labels [0,0,1,1,...] and return 
    {0:0, 1:0, 2:1, 3:1, ... }
    This is the format of partition used for 'community' package.
    
    """
    return {i: int(z[i]) for i in range(len(z))}

def labels_to_set(z):
    """"Now we generate sets containing the nodes, like
    [[0,1,4], [2,3], ...]
    
    """
    return [list(np.where(z==c)[0]) for c in np.unique(z)]

def dict_to_labels(part):
    """Return {0:0, 1:0, ...} to [0, 0, ...]"""
    z = np.zeros(len(part), dtype=int)
    for node, label in part.items():
        z[node] = label
    return z

def dict_to_set(part):
    """"Convert dictionary to sets."""
    z = dict_to_labels(part)
    return labels_to_set(z)

def labels_to_matrix(z):
    return eclust.ztoZ(z)

def bethe_matrix(A, weighted=False, r=None):
    """Construct Bethe Hessian matrix from adjacency."""
    if not r:
        c = A.sum(axis=1).mean()
        r = np.sqrt(c)
    if not weighted:
        D = np.diag(A.sum(axis=1))
        H = (r**2-1)*np.eye(A.shape[0]) - r*A + D
    else:
        H = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                H[i,j] = - r*A[i,j]/(r**2-A[i,j]**2)
                if i == j:
                    s = sum([A[i,k]**2/(r**2-A[i,k]**2) 
                             for k in range(0, len(A[0]))])
                    H[i,j] += (1 + s)
    return H

def organize_adjacency(A, z):
    """Organize the adjacency matrix according to class labels."""
    phi = {}
    i = 0
    for c in np.unique(z):
        nodes = np.where(z==c)[0]
        for node in nodes:
            phi[i] = node
            i += 1
    B = np.zeros(A.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            B[i,j] = A[phi[i], phi[j]]
    return B

def eval_metrics(G, zh, true_labels=[]):
    """Compute some metrics."""
    k = len(np.unique(zh))
    sets = labels_to_set(zh)
    part = labels_to_dict(zh)
    metrics = [
        community.quality.performance(G, sets), 
        community.quality.coverage(G, sets), 
        modularity(part,G),
    ]
    if true_labels != []:
        z = true_labels
        metrics.append(k/(k-1)*(accuracy(z, zh)-1/k))
        metrics.append(nmi(z, zh))
    return metrics

def cluster_comparison(k, z, A, H, G, useW=False):
    results = []
    
    #part = best_partition(G)
    #zh = dict_to_labels(part)
    #results.append(['louvain', *some_metrics(G, zh, z)])
    
    zh = spectralA(k, A)
    results.append(['spectralA', *eval_metrics(G, zh, z)])
    
    zh = bethe_hessian(k, H)
    results.append(['bethe', *eval_metrics(G, zh, z)])
    
    zh = kgroups_bethe_hessian(k, A, H, useW=useW)
    results.append(['kgroups', *eval_metrics(G, zh, z)])
    
    df = pd.DataFrame(results, columns=['method', 'performance', 'coverage',
                        'modularity', 'overlap', 'nmi'])
    return df
    
def plot_eigenvalues(H, A): 
    """Plot eigenvalues of Bethe Hessian and Adjacency."""
    
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    
    lamb = eigvalsh(H)
    k_est = np.where(lamb>0)[0][0]
    n = k_est + 5
    ax.plot(lamb[:n], 'o', color='b', fillstyle='full')
    ax.plot(range(n),[0]*n,'--',color='k',lw=1,label=r'$k=%i$'%k_est)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\lambda_i(\mathcal{H}_r)$')
    ax.legend()
    
    ax = fig.add_subplot(122)
    ax.plot(np.flip(eigvalsh(A))[0:n], 'o', color='b', fillstyle='full')
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\lambda_i(\mathcal{A})$')
    
    fig.savefig('eigenvalues.pdf')

def plot_graph(G, zh, output):
    """Create a graph according to the labels."""
    partition = labels_to_dict(zh)
    pos = plot_network.community_layout(G, partition)

    #fig = plt.figure(figsize=(8, 8))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    col_map = [colors[int(a)] for a in zh]

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col_map,
                        node_size=40, width=0.05, linewidths=1, alpha=1)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, width=0.1, edge_color='k',
                                    alpha=0.4)
    nodes.set_edgecolor('k')
    ax.axis('off')
    fig.savefig(output, bbox_inches='tight')


###############################################################################
### clustering some datasets

def cluster_real_data(datasets, useW=None):
    """Cluster several 'datasets', which is a list with the functions
    to be called to return the data.
    
    """
    if useW == None:
        useW = [False]*len(datasets)
    for uW, datasetfunc in zip(useW, datasets):
        A, z = datasetfunc()
        k = len(np.unique(z))
        d = A.sum(axis=1).mean()
        print('%s, n=%i, k=%i, d=%.2f'%(datasetfunc.__name__, A.shape[0], k ,d))
        H = bethe_matrix(A)
        G = nx.from_numpy_matrix(A)
        results = cluster_comparison(k, z, A, H, G, useW=uW)
        print(results)
   
def cluster_GRQC():
    """Clustering the arXiv GR-QC collaboration network."""

    # arXiv GR-QC collaboration
    A = get_data.GrQc()
    H = bethe_matrix(A)
    G = nx.from_numpy_matrix(A)
    
    #print(A.shape[0], A.sum(), A.sum(axis=1).mean())
    #plot_eigenvalues(H, A)

    #part = best_partition(G)
    #zh = dict_to_labels(part)
    #print('louvain---')
    #print(len(np.unique(zh)))
    #r = eval_metrics(G, zh)
    #print(r)
    #pickle.dump(zh, open('data/labels_louvain_GrQc.pickle', 'wb'))

    #k = 165
    k = 392
    print('spectral---')
    zh = spectralA(k, A)
    r = eval_metrics(G, zh)
    print(r)
    #pickle.dump(zh, open('data/labels_spectral_GrQc.pickle', 'wb'))
    pickle.dump(zh, open('data/labels_spectral_GrQc_392.pickle', 'wb'))

    print('bethe---')
    zh = bethe_hessian(k, H)
    r = eval_metrics(G, zh)
    print(r)
    #pickle.dump(zh, open('data/labels_bethe_GrQc.pickle', 'wb'))
    pickle.dump(zh, open('data/labels_bethe_GrQc_392.pickle', 'wb'))
    
    print('kgroups----')
    zh = kgroups_bethe_hessian(k, A, H, useW=False)
    r = eval_metrics(G, zh)
    print(r)
    #pickle.dump(zh, open('data/labels_kgroups_GrQc.pickle', 'wb'))
    pickle.dump(zh, open('data/labels_kgroups_GrQc_392.pickle', 'wb'))
    
    
##############################################################################
if __name__ == '__main__':

    ###############
    # clustering several small examples
    #cluster_real_data([
    #    get_data.karate,
    #    get_data.dolphin,
    #    get_data.football,
    #    get_data.polbooks,
    #], useW=[False,False,False,False])

    ##############
    # GR-QC network, it's a little big, takes some time
    #cluster_GRQC()
    # plotting graph
    #A = get_data.GrQc()
    #G = nx.from_numpy_matrix(A)
    #zh = pickle.load(open('data/labels_kgroups_GrQc.pickle', 'rb'))
    #zh = pickle.load(open('data/labels_kgroups_GrQc_392.pickle', 'rb'))
    #zh = pickle.load(open('data/labels_louvain_GrQc.pickle', 'rb'))
    #Zh = eclust.ztoZ(zh)
    #print(list(np.sort(np.diag(Zh.T.dot(Zh)))))
    #plot_graph(G, zh, 'grqc_network_392.pdf')

    ###############
    # Drosophila
    A = get_data.drosophila_left()
    #A = get_data.drosophila_right()
    A = A + A.T
    A[np.where(A > 0)] = 1 # drop weights

    H = bethe_matrix(A)
    G = nx.from_numpy_matrix(A)
    #plot_eigenvalues(H, A)
    k = 2

    # using Louvain heuristic which gives highest modularity
    part = best_partition(G)
    z = dict_to_labels(part)
    zf = z
    print('louvain', len(np.unique(z)), eval_metrics(G, z))

    z = spectralA(k, A)
    print('spectralA', eval_metrics(G, z))
    
    z = bethe_hessian(k, H)
    print('Bethe', eval_metrics(G, z))
    
    z = kgroups_bethe_hessian(k, A, H, useW=False)
    #zf = z
    print('kgroups', eval_metrics(G, z))
    
    Z = eclust.ztoZ(z)
    print(A.shape[0], A.sum(axis=1).sum(), A.sum(axis=1).mean(), 
            list(np.sort(np.diag(Z.T.dot(Z)))))
    #plot_graph(G, zf, 'drosophila_left.pdf')
    #plot_graph(G, z, 'drosophila_right.pdf')
   

    #A, z = get_data.polbooks()
    #H = bethe_matrix(A)
    #k = 3
    #zh = kgroups_bethe_hessian(k, A, H, useW=False)
    #print(accuracy(z, zh))
    #B = organize_adjacency(A, zh)

    B = organize_adjacency(A, zf)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.imshow(A, cmap='hot')
    ax = fig.add_subplot(122)
    ax.imshow(B, cmap='hot')

    plt.show()
