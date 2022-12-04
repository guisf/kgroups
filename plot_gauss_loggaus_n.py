# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot the high dimensional"\
                "Gaussian or LogGaussian with varying number of points.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

methods = ['k-means', 'gmm', 
            r'spectral clustering $\widetilde{\rho}_1$', 
            r'kernel k-groups $\rho_{1}$',
            r'kernel k-groups $\rho_{1/2}$',
            r'kernel k-groups $\widetilde{\rho}_{1}$']
method_names = ['k-means', 'GMM', 
            r'spectral $\widetilde{\rho}_1$', 
            r'ker. k-groups $\rho_{1}$',
            r'ker. k-groups $\rho_{1/2}$',
            r'ker. k-groups $\widetilde{\rho}_{1}$']
#markers = iter(['^', 'v', 'p', 'D', 's', 'o'])
markers = iter(['p', 'D', 'h', None, None, 's'])
#colors = iter(["#54278f", "#5F8793", "#99B4C6", "#318dde", "#F41711"])
#colors = iter(["#54278f", "#5F8793", "#99B4C6", 
#               "#F41711", "#F41711", "#F41711"])
#colors = iter(["#4daf4a", "#377eb8", "#984ea3", 
#               #"#ffff33", 
#               "#f781bf",
#               "#a65628", 
#               "#e41a1c"
#               ])
#colors = iter(["#4daf4a", "#377eb8", "#984ea3", 
#               #"#ffff33", 
#               "#f781bf",
#               "#a65628", 
#               "#e41a1c"
#               ])
colors = iter(["blue", "green", "cyan", 
               "red",
               "red", 
               "red"
               ])

lines = iter(["-", "-", "-", "--", "-.", "-"])

points = np.unique(df['points'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for i, (meth_name, method) in enumerate(zip(method_names,methods)):
    r = []
    for p in points:
        df2 = df[(df['method']==method) & (df['points']==p)]
        r.append([p, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    #ax.plot(r[:,0], [0.9]*len(r[:,0]), '--', linewidth=1, color='k',
    #        alpha=0.5)
    mk = next(markers)
    cl = next(colors)
    ln = next(lines)
    #ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
    #            label=method)
    ax.errorbar(r[:,0], r[:,1], marker=mk, linestyle=ln, color=cl, 
                label=meth_name, markevery=2)
ax.set_xlabel(r'\# points')
ax.set_ylabel(r'accuracy')
#ax.set_xlim([10,400])
#ax.legend(loc=4)
fig.savefig(output, bbox_inches='tight')

