# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Plot unbalanced Gaussians.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

methods = ['k-means', 'gmm', 'spectral clustering', 'kernel k-means', 
           'kernel k-groups']
method_names = ['k-means', 'GMM', 'spectral', 'ker. k-means', 
           'ker. k-groups']
#markers = iter(['^', 'v', 'p', 's', 'o'])
markers = iter(['p', 'D', 'h', 'o', 's'])
#colors = iter(["#54278f", "#5F8793", "#99B4C6", "#318dde", "#F41711"])
#colors = iter(["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c"])
colors = iter(["blue", "green", "cyan", "magenta", "red"])


points = np.unique(df['points'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for meth_name, method in zip(method_names, methods):
    r = []
    for p in points:
        df2 = df[(df['method']==method) & (df['points']==p)]
        r.append([p, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    mk = next(markers)
    cl = next(colors)
    #ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
    #            label=method)
    ax.plot(r[:,0], r[:,1], marker=mk, color=cl, label=meth_name)
ax.set_xlabel(r'\# unbalanced points')
ax.set_ylabel(r'accuracy')
#ax.set_xlim([0,240])

elems = [
    Line2D([0], [0], color='blue', lw=2, marker='p', markersize=8, 
            label='k-means'),
    Line2D([0], [0], color='green', lw=2, marker='D', markersize=8, 
            label='GMM'),
    Line2D([0], [0], color='cyan', lw=2, marker='h', markersize=8, 
            label='spectral'),
    Line2D([0], [0], color='magenta', lw=2, marker='o', markersize=8, 
            label='kernel k-means'),
    Line2D([0], [0], color='red', lw=2, marker='s', markersize=8, 
            label='kernel k-groups'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', 
            label=r'kernel k-groups $\rho_1$'),
    Line2D([0], [0], color='red', lw=2, linestyle='-.', 
            label=r'kernel k-groups $\rho_{1/2}$'),
]

ax.legend(handles=elems, loc=3)
fig.savefig(output, bbox_inches='tight')

