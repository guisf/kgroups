"""Graph Clustering."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import wrapper
import init
import eclust
from metric import accuracy
from sklearn.metrics import normalized_mutual_info_score as nmi
import get_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# clustering methods

def spectralA(A, z):
    """Cluster the adjacency matrix."""
    k = len(np.unique(z))
    zh = init.spectral2(k, A, run_times=10, largest=True)
    return nmi(z, zh), accuracy(z, zh)

def spectralL(A, z):
    """Cluster the Laplacian matrix."""
    k = len(np.unique(z))
    D = np.diag(A.sum(axis=1))
    L = D - A
    zh = init.spectral2(k, L, run_times=10, largest=False)
    return nmi(z, zh), accuracy(z, zh)

def spectralNL(A, z):
    """Cluster the normalized Laplacian."""
    k = len(np.unique(z))
    D = np.diag(A.sum(axis=1)) + 0.0001*np.eye(A.shape[0])
    L = D - A
    D2 = np.sqrt(np.linalg.inv(D))
    L = D2.dot(L.dot(D2))
    zh = init.spectral2(k, L, run_times=10, largest=False)
    return nmi(z, zh), accuracy(z, zh)

def bethe_hessian(A, z, c=None):
    """Cluster based on Bethe Hessian approach."""
    k = len(np.unique(z))
    if not c:
        c = A.sum(axis=1).mean()
    D = np.diag(A.sum(axis=1))
    r = np.sqrt(c)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + D
    zh = init.spectral2(k, H, run_times=10, largest=False)
    return nmi(z, zh), accuracy(z, zh)

def graph_kgroups(A, z):
    """Cluster based on Kernel k-groups approach. We simply cluster
    the adjacency matrix with weights equal to the degree.
    Use initialization from spectral clustering.
    
    """
    k = len(np.unique(z))
    D = np.diag(A.sum(axis=1)) + 0.001*np.eye(A.shape[0])
    z0 = init.spectralNg(k, A, D, run_times=10, largest=True)
    Z0 = eclust.ztoZ(z0)
    zh = wrapper.kernel_kgroups(k, A, A, W=D, run_times=10, Z0=Z0)[0]
    return nmi(z, zh), accuracy(z, zh)

def kgroups_bethe(A, z, c=None):
    """Cluster based on Kernel k-groups approach but use 
    Bethe Hessian matrix, also as initialization.
    
    """
    k = len(np.unique(z))
    if not c:
        c = A.sum(axis=1).mean()
    D = np.diag(A.sum(axis=1)) + 0.001*np.eye(A.shape[0])
    r = np.sqrt(c)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + D
    z0 = init.spectral2(k, H, run_times=10, largest=False)
    Z0 = eclust.ztoZ(z0)
    #zh = wrapper.kernel_kgroups(k, -H, -H, W=D, run_times=10, Z0=Z0)[0]
    zh = wrapper.kernel_kgroups(k, -H, -H, run_times=10, Z0=Z0)[0]
    return nmi(z, zh), accuracy(z, zh)

def kgroups_bethe_hessian(A, z, c=None):
    """Cluster based on Kernel k-groups approach but use 
    Bethe Hessian matrix, also as initialization.
    
    """
    k = len(np.unique(z))
    if not c:
        c = A.sum(axis=1).mean()
    D = np.diag(A.sum(axis=1)) + 0.001*np.eye(A.shape[0])
    r = np.sqrt(c)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + D
    z0 = init.spectral2(k, H, run_times=10, largest=False)
    Z0 = eclust.ztoZ(z0)
    zh = wrapper.kernel_kgroups(k, -H, -H, W=D, run_times=10, Z0=Z0)[0]
    return zh

def bethe_hess(A, z, c=None):
    """Cluster based on Bethe Hessian approach."""
    k = len(np.unique(z))
    if not c:
        c = A.sum(axis=1).mean()
    D = np.diag(A.sum(axis=1))
    r = np.sqrt(c)
    H = (r**2-1)*np.eye(A.shape[0]) - r*A + D
    zh = init.spectral2(k, H, run_times=10, largest=False)
    return zh



###############################################################################
# clustering stochastic block model

def clusterSBM(n, k, a, b, runs=10): 
    """Generate a stochastic block model with 'n' vertices and probability
    matrix with 'a' and 'b'. Here 'k' is the number of communities.
    Return average and standard deviation over number of 'runs'.
    
    """
    # probability matrix
    Q = np.zeros((k,k))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                Q[i,i] = a
            else:
                Q[i, j] = Q[j, i] = b

    df = []
    for e in range(runs):
        
        A, z = get_data.SBM(n, k, Q)
        d = (a + (k-1)*b)/k # average degree
        lamb = (a-b)/(k*np.sqrt(d)) # signal to noise ratio
        
        # cluster with these methods
        bethe = bethe_hessian(A, z, d)
        #specA = spectralA(A, z)
        #specL = spectralL(A, z)
        #specNL = spectralNL(A, z)
        #kgroups = graph_kgroups(A, z)
        kgroups = kgroups_bethe(A, z, d)

        df.append(['bethe', d, lamb, bethe[0], bethe[1]])
        #df.append(['spectralA', d, lamb, specA[0], specA[1]])
        #df.append(['spectralL', d, lamb, specL[0], specL[1]])
        #df.append(['spectralNL', d, lamb, specNL[0], specNL[1]])
        df.append(['kgroups', d, lamb, kgroups[0], kgroups[1]])
    
    df = pd.DataFrame(df,columns=['method','degree','lambda','nmi','accuracy'])

    return df

def phase_data(n=500, k=2, d=10, lamb_range=np.linspace(0,1,10), num_runs=10):
    """Generate data to make phase diagram."""
    dfs = []
    for lamb in lamb_range:
        a = lamb*(k-1)*np.sqrt(d) + d
        b = d - lamb*np.sqrt(d)
        df = clusterSBM(n, k, a, b, runs=10)
        dfs.append(df)
    return pd.concat(dfs)


###############################################################################
# plotting results


def get_means_stds(df, method, k):
    """Get means and standard deviations of overlap for a given
    method.
    
    """
    m = method
    lambdas = df['lambda'].unique() # signal to noise ratio
    values = [df.loc[(df['method']==m)&(df['lambda']==l)]['accuracy'].values 
              for l in lambdas]
    # converte accuracy to overlap
    values = [(k/(k-1))*(v-1/k) for v in values]
    means = np.array([v.mean() for v in values])
    stds = np.array([v.std() for v in values])
    return lambdas, means, stds

def phase_diagram1(df, k, fname='sbm_phase.pdf', df_ref=None, k_ref=None):
    """Plot phase diagram, one line for each method."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    if df_ref is not None and k_ref is not None:
        xs, ys, err = get_means_stds(df_ref, 'kgroups', k_ref)
        ax.plot(xs, ys, label=r'kernel k-groups $k=%i$'%k_ref,
                color='black', linestyle='--', linewidth=1,
                marker='o', fillstyle='none')

    xs, ys, err = get_means_stds(df, 'bethe', k)
    ax.errorbar(xs, ys, err, label=r'Bethe Hessian', color='b',
                marker='s', elinewidth=1, 
                linewidth=2, 
                #markersize=5, 
                fillstyle='right')
    
    xs, ys, err = get_means_stds(df, 'kgroups', k)
    ax.errorbar(xs, ys, err, label=r'kernel k-groups', color='r',
                marker='s', elinewidth=1, 
                linewidth=2, 
                #markersize=5,
                fillstyle='left')

    #ax.plot([1,1], [0, 1], '--', dashes=(7, 8), color='black', linewidth=1)
    ax.fill_between([0.5,1], [1,1], color='gray', alpha=0.15)

    ax.annotate(r'$\bm{k=%i}$'%k, (1.8,0.5))
    
    ax.legend(loc=4)
    ax.set_xlabel('signal to noise ratio')
    ax.set_ylabel('overlap')
    fig.savefig(fname, bbox_inches='tight')


###############################################################################
# tests

def testSBM(n=128, k=4, d=16, lamb=2, runs=10):
    a = lamb*(k-1)*np.sqrt(d) + d
    b = d - lamb*np.sqrt(d)

    df = clusterSBM(n, k, a, b, runs=runs)
    methods = df['method'].unique()
    for meth in methods:
        acc = df.loc[df['method']==meth]['accuracy'].values
        overlap = np.array([k/(k-1)*(v-1/k) for v in acc])
        print(meth, overlap.mean(), overlap.std())


##############################################################################


if __name__ == '__main__':

    """
    # generate data
    #lr = np.linspace(0.5, 2.5, 30)
    lr = np.linspace(0.5, 2.5, 20)
    
    #df = phase_data(n=1000, k=2, d=3, lamb_range=lr, num_runs=20)
    #df = phase_data(n=1000, k=2, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k2_d3.pickle', 'wb')
    #f = open('data/sbm_phase_k2_d3_v2.pickle', 'wb')
    
    #df = phase_data(n=1000, k=5, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k5_d3.pickle', 'wb')
    #f = open('data/sbm_phase_k5_d3_v2.pickle', 'wb')
    
    df = phase_data(n=1000, k=15, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k15_d3.pickle', 'wb')
    f = open('data/sbm_phase_k15_d3_v2.pickle', 'wb')
    pickle.dump(df, f)
    
    # plotting
    #df_ref = pickle.load(open('data/sbm_phase_k2_d3.pickle', 'rb'))
    df_ref = pickle.load(open('data/sbm_phase_k2_d3_v2.pickle', 'rb'))
    k_ref = 2
    #phase_diagram1(df_ref, k_ref, 'sbm_phase_2.pdf')
    
    #df = pickle.load(open('data/sbm_phase_k5_d3.pickle', 'rb'))
    #df = pickle.load(open('data/sbm_phase_k5_d3_v2.pickle', 'rb'))
    #phase_diagram1(df, 5, 'sbm_phase_5.pdf', df_ref, k_ref)
    
    #df = pickle.load(open('data/sbm_phase_k15_d3.pickle', 'rb'))
    df = pickle.load(open('data/sbm_phase_k15_d3_v2.pickle', 'rb'))
    phase_diagram1(df, 15, 'sbm_phase_15.pdf', df_ref, k_ref)
    """
    
    #df = phase_data(n=1000, k=2, d=3, lamb_range=lr, num_runs=20)
    #df = phase_data(n=1000, k=2, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k2_d3.pickle', 'wb')
    #f = open('data/sbm_phase_k2_d3_v2.pickle', 'wb')
    
    #testSBM(n=128, k=4, d=16, lamb=1.1, runs=500)
    #testSBM(n=128, k=4, d=16, lamb=1.8, runs=500)
    #testSBM(n=128, k=4, d=16, lamb=3.5, runs=10)
    #testSBM(n=128, k=4, d=16, lamb=2.0, runs=10)
    #testSBM(n=500, k=50, d=3, lamb=1.8, runs=10)
    
    """
    # phase transition Girvin-Newman
    lr = np.linspace(0.5, 2.5, 20)
    df = phase_data(n=128, k=4, d=16, lamb_range=lr, num_runs=1000)
    f = open('data/girvin_newman.pickle', 'wb')
    pickle.dump(df, f)
    
    #df = pickle.load(open('data/girvin_newman.pickle', 'rb'))
    phase_diagram1(df, 4, 'phase_girvin_newman.pdf')
    """
    
    lr = np.linspace(0.5, 2.5, 20)
    #df = phase_data(n=1000, k=2, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_k2_d3_noweight.pickle', 'wb')
    #pickle.dump(df, f)
    #df = pickle.load(open('data/sbm_k2_d3_noweight.pickle', 'rb'))
    #phase_diagram1(df, 2, 'sbm_k2_d3_noweight.pdf')
    
    #df = phase_data(n=1000, k=5, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k5_d3_noweight.pickle', 'wb')
    #pickle.dump(df, f)
    #df = pickle.load(open('data/sbm_phase_k5_d3_noweight.pickle', 'rb'))
    #df_ref = pickle.load(open('data/sbm_k2_d3_noweight.pickle', 'rb'))
    #k_ref = 2
    #phase_diagram1(df, 5, 'sbm_k5_d3_noweight.pdf', df_ref, k_ref)
    
    #df = phase_data(n=1000, k=15, d=3, lamb_range=lr, num_runs=100)
    #f = open('data/sbm_phase_k15_d3_noweight.pickle', 'wb')
    #pickle.dump(df, f)
    #df = pickle.load(open('data/sbm_phase_k5_d3_noweight.pickle', 'rb'))
    #df_ref = pickle.load(open('data/sbm_k2_d3_noweight.pickle', 'rb'))
    #k_ref = 2
    #phase_diagram1(df, 15, 'sbm_k15_d3_noweight.pdf', df_ref, k_ref)
    
    df = phase_data(n=1000, k=25, d=3, lamb_range=lr, num_runs=100)
    f = open('data/sbm_phase_k25_d3_noweight.pickle', 'wb')
    pickle.dump(df, f)
    #df = pickle.load(open('data/sbm_phase_k5_d3_noweight.pickle', 'rb'))
    df_ref = pickle.load(open('data/sbm_k2_d3_noweight.pickle', 'rb'))
    k_ref = 2
    phase_diagram1(df, 25, 'sbm_k25_d3_noweight.pdf', df_ref, k_ref)
    



