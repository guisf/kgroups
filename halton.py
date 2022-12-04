"""Generate clusters through space with centers determined
by Halton sequences.

"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import wrapper
from metric import accuracy
import eclust


def _haltonseq(i, b=2):
    f = 1
    r = 0
    while i > 0:

        f = f/b
        r = r + f*(i%b)
        i = np.floor(i/b)
    return r

def next_prime():
    def is_prime(num):
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True
    prime = 3
    while True:
        if is_prime(prime):
            yield prime
        prime += 2

def halton_sequence(size, dim):
    """size is the number of points, dim is the dimension."""
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([_haltonseq(i, base) for i in range(size)])
    return np.array(seq)

def halton_gaussians(num_points, sigma, dim):
    """Num points is a list with the number of elements for each
    center.
    
    """
    k = len(num_points)
    sigma = np.eye(dim)*sigma
    centers = np.sqrt(k)*np.transpose(halton_sequence(k, dim))
    centers = centers
    data = []
    z = []
    for i, (mu, n) in enumerate(zip(centers, num_points)):
        points = np.random.multivariate_normal(mu, sigma, n)
        data.append(points)
        z.append([i]*n)
    data = np.concatenate(data)
    z = np.concatenate(z)
    idx = list(range(data.shape[0]))
    np.random.shuffle(idx)
    return data[idx], z[idx]

def one_experiment(k, sigma, points_per_cluster):
    """Run one single experiment with many clustering methods."""

    rho = lambda a, b: np.power(np.linalg.norm(a-b), 0.5)

    X, z = halton_gaussians([points_per_cluster]*k, sigma, 2)
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

def many_experiments(kmax, sigma, points_per_cluster, num_experiments):
    """Repeat the above many times."""

    results = []
    for e in range(num_experiments):
        for k in range(2, kmax):
            print('exp=%i, k=%i, n=%i'%(e, k, k*points_per_cluster))
            results.append(one_experiment(k, sigma, points_per_cluster))
    results = pd.concat(results)
    return results



###############################################################################
if __name__ == '__main__':

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data, z = halton_gaussians([10]*20, 0.004, 2)
    ax.plot(data[:,0], data[:,1], 'o', color='b', fillstyle='full', 
             markersize=2, alpha=0.3)
    fig.savefig('halton.pdf')
    """
    r = many_experiments(10, 0.5, 10, 5)
    methods = r['method'].unique()
    ks = r['k'].unique()

    method_names = ['k-means', 'GMM', 'spectral', 'ker. k-means',
                    'ker. k-groups']
    markers = ['p', 'D', 'h', 'o', 's']
    colors = ["blue", "green", "cyan", "magenta", "red"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, meth in enumerate(methods):
        ys = [r[(r['method']==meth)&(r['k']==k)]['accuracy'].values.mean()
              for k in ks]
        err = [r[(r['method']==meth)&(r['k']==k)]['accuracy'].values.std()
               for k in ks]
        ax.errorbar(ks, ys, yerr=err, marker=markers[i], elinewidth=1,
                    label=method_names[i], color=colors[i])

    ax.set_xlabel(r'\# clusters')
    ax.set_ylabel(r'accuracy')
    ax.legend(loc=0)

    fig.savefig('halton.pdf', bbox_inches='tight')
