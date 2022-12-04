# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot 1D experiments.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

methods = ['k-means', 'GMM', 'kernel k-groups']
markers = iter(['p', 'o', 's'])
#colors = iter(["#54278f", "#5F8793", "#F41711"])
#colors = iter(["#4daf4a", "#377ed8", "#e41a1c"])
colors = iter(["blue", "green", "red"])

num_points = np.unique(df['num_points'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for method in methods:
    r = []
    for n in num_points:
        df2 = df[(df['method']==method.lower()) & (df['num_points']==n)]
        r.append([n, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    ax.plot(r[:,0], [0.88]*len(r[:,0]), '--', linewidth=0.5, color='k',
            dashes=(12, 12))
    mk = next(markers)
    cl = next(colors)
    #ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
    #            label=method)
    ax.plot(r[:,0], r[:,1], marker=mk, label=method, color=cl, markevery=2)
ax.set_xlabel(r'\# points')
ax.set_ylabel(r'accuracy')
#ax.set_xlim([10,800])
#ax.legend()
fig.savefig(output, bbox_inches='tight')

