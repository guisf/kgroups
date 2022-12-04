# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot the varying k problem.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

#methods = ['k-means', 'gmm', 'spectral clustering', 'kernel k-means', 
#           'kernel k-groups']
methods = ['spectral clustering', 'kernel k-means', 'kernel k-groups']
#markers = iter(['^', 'v', 'p', 's', 'o'])
markers = iter(['p', 's', 'o'])
#colors = iter(["#54278f", "#5F8793", "#99B4C6", "#318dde", "#F41711"])
colors = iter(["#99B4C6", "#318dde", "#F41711"])

ks = np.unique(df['clusters'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for method in methods:
    r = []
    for k in ks:
        df2 = df[(df['method']==method) & (df['clusters']==k)]
        r.append([k, df2['score'].mean(), df2['score'].sem()])
    r = np.array(r)
    mk = next(markers)
    cl = next(colors)
    #ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
    #            label=method)
    ax.plot(r[:,0], r[:,1], marker=mk, color=cl, label=method)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'objective function')
ax.set_xlim([5,50])
ax.legend()
fig.savefig(output, bbox_inches='tight')

