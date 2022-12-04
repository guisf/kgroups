"""Generate data on a grid and cluster."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
import pandas as pd
import pickle

import wrapper
from metric import accuracy
import eclust


def grid2D(limit):
    """Make a grid"""
    lims = [0, limit]
    g = [[x, y] for x in range(*lims) 
                for y in range(*lims)]
    g = np.array(g)
    return g

def grid3D(limit):
    """Make a grid"""
    lims = [0, limit]
    g = [[x, y, z] for x in range(*lims) 
                   for y in range(*lims)
                   for z in range(*lims)]
    g = np.array(g)
    return g

def create_clusters(grid, sigma, points_per_cluster):
    """Generate cloud of Gaussian points on the grid."""
    data = []
    z = []
    n = points_per_cluster
    var = sigma*np.eye(grid[0].shape[0])
    for i, center in enumerate(grid):
        X = np.random.multivariate_normal(center, var, n)
        data.append(X)
        z.append([i]*n)
    data = np.concatenate(data)
    z = np.concatenate(z)
    idx = list(range(len(data)))
    np.random.shuffle(idx)
    data = data[idx]
    z = z[idx]
    return data, z

def one_experiment(k, sigma, num_points, 
                   rho = lambda a, b: np.power(np.linalg.norm(a-b), 1)):
    """Run one single experiment.
    
    "k" is the number of clusters
    "sigma" is the standard deviation for the gaussian (all gaussians have
        the same variance)
    "num_points" is the number of points for each cluster (all clusters
        have the same number of points).
    "rho" is the function to generate the gram Matrix
    
    """
    
    index = int(np.ceil(np.sqrt(k)))
    # generate grid and pick the first k centers
    g = grid2D(index)[:k]
    
    # generate gaussian on each center
    X, z = create_clusters(g, sigma, num_points)

    # create the gram matrix
    G = eclust.kernel_matrix(X, rho)

    r = []
    zh = wrapper.kmeans(k, X)
    r.append(['k-means', k, accuracy(z, zh)])

    zh = wrapper.gmm(k, X)
    r.append(['gmm', k, accuracy(z, zh)])

    zh = wrapper.spectral_clustering(k, X, G)[0]
    r.append(['spectral', k, accuracy(z, zh)])

    zh = wrapper.kernel_kmeans(k, X, G)[0]
    r.append(['kernel k-means', k, accuracy(z, zh)])

    zh = wrapper.kernel_kgroups(k, X, G)[0]
    r.append(['kernel k-groups', k, accuracy(z, zh)])

    r = pd.DataFrame(r, columns=['method', 'k', 'accuracy'])

    return r

def many_experiments(kmax, sigma, num_points, num_experiments, skip=4,
                     rho = lambda a, b: np.power(np.linalg.norm(a-b), 1)):
    """Repeat the above many times."""
    results = []
    for e in range(num_experiments):
        for k in range(2, kmax, skip):
            print('exp=%i, k=%i, n=%i'%(e, k, (k)*num_points))
            results.append(one_experiment(k, sigma, num_points))
    results = pd.concat(results)
    return results
    

###############################################################################
if __name__ == '__main__':
    import sys
    import time

    ############
    kmax = 50
    sigma = 0.1
    points_per_cluster = 10
    num_experiments = 10
    skip = 4
    metric = lambda a, b: np.power(np.linalg.norm(a-b), 1)
    ############

    #a = time.time()
    #r = many_experiments(kmax, sigma, points_per_cluster, num_experiments, skip)
    #pickle.dump(r, open('data/many_clusters.pickle', 'wb'))
    #pickle.dump(r, open('data/many_clusters2.pickle', 'wb'))
    #b = time.time()
    #print(b-a)
    
    #sys.exit()

    r1 = pickle.load(open('data/many_clusters.pickle', 'rb'))
    r2 = pickle.load(open('data/many_clusters2.pickle', 'rb'))
    r = pd.concat([r1,r2])

    methods = r['method'].unique()
    ks = r['k'].unique()

    names = ['k-means', 'GMM', 'spectral', 'ker. k-means', 'ker. k-groups']
    mks = ['p', 'D', 'h', 'o', 's']
    cols = ["blue", "green", "cyan", "magenta", "red"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, meth in enumerate(methods):
        ys = [r[(r['method']==meth)&(r['k']==k)]['accuracy'].values.mean()
              for k in ks]
        err = [r[(r['method']==meth)&(r['k']==k)]['accuracy'].values.std()
               for k in ks]
        ax.errorbar(ks, ys, yerr=err, marker=mks[i], elinewidth=1, 
                    label=names[i], color=cols[i])

    ax.set_xlabel(r'\# clusters')
    ax.set_ylabel(r'accuracy')
    #ax.legend(loc=0)

    fig.savefig('uniform_clusters.pdf', bbox_inches='tight')

